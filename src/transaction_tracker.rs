use crate::compat::HashMap;
#[cfg(not(feature = "std"))]
use crate::compat::Mutex;
use crate::error::StorageError;
use crate::tree_store::TransactionalMemory;
use crate::{Key, Result, Savepoint, TypeName, Value};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use core::cmp::Ordering;
use core::mem;
use core::mem::size_of;
#[cfg(feature = "logging")]
use log::debug;
#[cfg(feature = "std")]
use std::sync::{Condvar, Mutex};

#[derive(Copy, Clone, Hash, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub(crate) struct TransactionId(u64);

impl TransactionId {
    pub(crate) fn new(value: u64) -> TransactionId {
        Self(value)
    }

    pub(crate) fn raw_id(self) -> u64 {
        self.0
    }

    pub(crate) fn next(self) -> Result<TransactionId> {
        let value = self.0.checked_add(1).ok_or(StorageError::OutOfSpace)?;
        Ok(TransactionId(value))
    }

    pub(crate) fn increment(&mut self) -> Result<TransactionId> {
        let next = self.next()?;
        *self = next;
        Ok(next)
    }
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Debug)]
pub(crate) struct SavepointId(pub u64);

impl SavepointId {
    pub(crate) fn next(self) -> Result<SavepointId> {
        let value = self.0.checked_add(1).ok_or(StorageError::OutOfSpace)?;
        Ok(SavepointId(value))
    }
}

impl Value for SavepointId {
    type SelfType<'a> = SavepointId;
    type AsBytes<'a> = [u8; size_of::<u64>()];

    fn fixed_width() -> Option<usize> {
        Some(size_of::<u64>())
    }

    #[allow(clippy::big_endian_bytes)]
    fn from_bytes<'a>(data: &'a [u8]) -> Self::SelfType<'a>
    where
        Self: 'a,
    {
        if data.len() < size_of::<u64>() {
            return SavepointId(0);
        }
        SavepointId(u64::from_be_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]))
    }

    #[allow(clippy::big_endian_bytes)]
    fn as_bytes<'a, 'b: 'a>(value: &'a Self::SelfType<'b>) -> Self::AsBytes<'a>
    where
        Self: 'b,
    {
        value.0.to_be_bytes()
    }

    fn type_name() -> TypeName {
        TypeName::internal("redb::SavepointId")
    }
}

impl Key for SavepointId {
    fn compare(data1: &[u8], data2: &[u8]) -> Ordering {
        // Big-endian serialization means raw byte comparison is correct.
        data1[..size_of::<u64>()].cmp(&data2[..size_of::<u64>()])
    }
}

struct State {
    next_savepoint_id: SavepointId,
    // reference count of read transactions per transaction id
    live_read_transactions: BTreeMap<TransactionId, u64>,
    next_transaction_id: TransactionId,
    live_write_transaction: Option<TransactionId>,
    valid_savepoints: BTreeMap<SavepointId, TransactionId>,
    // Non-durable commits that are still in-memory, and waiting for a durable commit to get flushed
    // We need to make sure that the freed-table does not get processed for these, since they are not durable yet
    // Therefore, we hold a read transaction on their nearest durable ancestor
    //
    // Maps non-durable transaction id -> durable ancestor
    pending_non_durable_commits: HashMap<TransactionId, TransactionId>,
    // How many pending non-durable commits hold each durable ancestor. Always equal to the
    // multiset of `pending_non_durable_commits.values()`; kept incrementally so that
    // `oldest_live_read_transaction_excluding_nondurable()`, which every non-durable commit calls
    // several times, does not rebuild it from all pending commits (up to 10 000) each time.
    non_durable_ancestor_holds: BTreeMap<TransactionId, u64>,
    // Non-durable commits which have NOT been processed in the freed table
    unprocessed_freed_non_durable_commits: BTreeSet<TransactionId>,
}

pub(crate) struct TransactionTracker {
    state: Mutex<State>,
    #[cfg(feature = "std")]
    live_write_transaction_available: Condvar,
}

impl TransactionTracker {
    pub(crate) fn new(next_transaction_id: TransactionId) -> Self {
        Self {
            state: Mutex::new(State {
                next_savepoint_id: SavepointId(0),
                live_read_transactions: Default::default(),
                next_transaction_id,
                live_write_transaction: None,
                valid_savepoints: Default::default(),
                pending_non_durable_commits: Default::default(),
                non_durable_ancestor_holds: Default::default(),
                unprocessed_freed_non_durable_commits: Default::default(),
            }),
            #[cfg(feature = "std")]
            live_write_transaction_available: Condvar::new(),
        }
    }

    #[cfg(feature = "std")]
    pub(crate) fn start_write_transaction(&self) -> Result<TransactionId> {
        let mut state = self.state.lock()?;
        while state.live_write_transaction.is_some() {
            state = self.live_write_transaction_available.wait(state)?;
        }
        if state.live_write_transaction.is_some() {
            return Err(StorageError::Internal(
                "Write transaction still active after condvar wait".into(),
            ));
        }
        let transaction_id = state.next_transaction_id.increment()?;
        #[cfg(feature = "logging")]
        debug!("Beginning write transaction id={transaction_id:?}");
        state.live_write_transaction = Some(transaction_id);

        Ok(transaction_id)
    }

    /// NOTE (`no_std`): `spin::Mutex::lock()` does not detect poisoning, so if a
    /// thread panics while holding the lock the mutex silently continues with
    /// potentially inconsistent inner state. This is an accepted limitation of
    /// the `no_std` spin-lock implementation -- callers that require strict
    /// consistency guarantees should use the `std` feature.
    #[cfg(not(feature = "std"))]
    pub(crate) fn start_write_transaction(&self) -> Result<TransactionId> {
        const MAX_SPIN_RETRIES: u32 = 1000;
        let mut retries: u32 = 0;
        let mut backoff: u32 = 1;

        loop {
            let mut state = self.state.lock();
            if state.live_write_transaction.is_none() {
                let transaction_id = state.next_transaction_id.increment()?;
                #[cfg(feature = "logging")]
                debug!("Beginning write transaction id={transaction_id:?}");
                state.live_write_transaction = Some(transaction_id);
                return Ok(transaction_id);
            }
            drop(state);

            retries += 1;
            if retries >= MAX_SPIN_RETRIES {
                return Err(StorageError::LockTimeout(
                    "Timed out waiting for write transaction lock after 1000 spin iterations"
                        .into(),
                ));
            }

            for _ in 0..backoff {
                core::hint::spin_loop();
            }
            backoff = backoff.saturating_mul(2).min(64);
        }
    }

    #[cfg(feature = "std")]
    pub(crate) fn end_write_transaction(&self, id: TransactionId) -> Result {
        let mut state = self.state.lock()?;
        match state.live_write_transaction {
            Some(active_id) if active_id == id => {
                state.live_write_transaction = None;
                self.live_write_transaction_available.notify_one();
                Ok(())
            }
            Some(active_id) => Err(StorageError::Internal(format!(
                "end_write_transaction called with id {id:?}, but active transaction is {active_id:?}"
            ))),
            None => Err(StorageError::Internal(format!(
                "end_write_transaction called with id {id:?}, but no write transaction is active"
            ))),
        }
    }

    /// NOTE (`no_std`): see `start_write_transaction` for spin-lock poisoning caveat.
    #[cfg(not(feature = "std"))]
    pub(crate) fn end_write_transaction(&self, id: TransactionId) -> Result {
        let mut state = self.state.lock();
        match state.live_write_transaction {
            Some(active_id) if active_id == id => {
                state.live_write_transaction = None;
                Ok(())
            }
            Some(active_id) => Err(StorageError::Internal(format!(
                "end_write_transaction called with id {id:?}, but active transaction is {active_id:?}"
            ))),
            None => Err(StorageError::Internal(format!(
                "end_write_transaction called with id {id:?}, but no write transaction is active"
            ))),
        }
    }

    pub(crate) fn clear_pending_non_durable_commits(&self) -> Result {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        let ids = mem::take(&mut state.pending_non_durable_commits);
        state.non_durable_ancestor_holds.clear();
        // Everything those commits left behind is persisted now, so nothing in
        // the freed tables needs the non-durable pass any more: `free_if_unpersisted`
        // would refuse every one of those pages. Dropping the set matters for
        // cost, not only memory. A commit that freed no pages never appears in the
        // freed table, so it was never marked processed and stayed here forever;
        // the oldest such id was then the lower bound of the range scan in
        // `process_freed_pages_nondurable_helper`, making each non-durable commit
        // walk every freed-table row since the oldest leftover: commit cost grew
        // linearly with the number of earlier non-durable commits.
        state.unprocessed_freed_non_durable_commits.clear();
        for (_, durable_ancestor) in ids {
            if let Some(ref_count) = state.live_read_transactions.get_mut(&durable_ancestor) {
                *ref_count -= 1;
                if *ref_count == 0 {
                    state.live_read_transactions.remove(&durable_ancestor);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn is_unprocessed_non_durable_commit(&self, id: TransactionId) -> Result<bool> {
        #[cfg(feature = "std")]
        let state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let state = self.state.lock();
        Ok(state.unprocessed_freed_non_durable_commits.contains(&id))
    }

    pub(crate) fn mark_non_durable_commit_processed(&self, id: TransactionId) -> Result {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        state.unprocessed_freed_non_durable_commits.remove(&id);
        Ok(())
    }

    pub(crate) fn oldest_unprocessed_non_durable_commit(&self) -> Result<Option<TransactionId>> {
        #[cfg(feature = "std")]
        let state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let state = self.state.lock();
        Ok(state
            .unprocessed_freed_non_durable_commits
            .iter()
            .next()
            .copied())
    }

    /// Maximum number of non-durable commits that can be pending before a
    /// durable commit is required. Prevents unbounded memory growth.
    const MAX_PENDING_NON_DURABLE: usize = 10_000;

    pub(crate) fn register_non_durable_commit(
        &self,
        id: TransactionId,
        durable_ancestor: TransactionId,
    ) -> Result {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        if state.pending_non_durable_commits.len() >= Self::MAX_PENDING_NON_DURABLE {
            return Err(StorageError::Internal(alloc::format!(
                "too many pending non-durable commits (limit: {}); a durable commit is required",
                Self::MAX_PENDING_NON_DURABLE
            )));
        }
        state
            .live_read_transactions
            .entry(durable_ancestor)
            .and_modify(|x| *x += 1)
            .or_insert(1);
        state
            .pending_non_durable_commits
            .insert(id, durable_ancestor);
        *state
            .non_durable_ancestor_holds
            .entry(durable_ancestor)
            .or_insert(0) += 1;
        state.unprocessed_freed_non_durable_commits.insert(id);
        Ok(())
    }

    pub(crate) fn restore_savepoint_counter_state(&self, next_savepoint: SavepointId) -> Result {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        if !state.valid_savepoints.is_empty() {
            return Err(StorageError::Internal(
                "restore_savepoint_counter_state called with active savepoints".into(),
            ));
        }
        state.next_savepoint_id = next_savepoint;
        Ok(())
    }

    pub(crate) fn register_persistent_savepoint(&self, savepoint: &Savepoint) -> Result {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        state
            .live_read_transactions
            .entry(savepoint.get_transaction_id())
            .and_modify(|x| *x += 1)
            .or_insert(1);
        state
            .valid_savepoints
            .insert(savepoint.get_id(), savepoint.get_transaction_id());
        Ok(())
    }

    pub(crate) fn register_read_transaction(
        &self,
        mem: &TransactionalMemory,
    ) -> Result<TransactionId> {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        let id = mem.get_last_committed_transaction_id()?;
        state
            .live_read_transactions
            .entry(id)
            .and_modify(|x| *x += 1)
            .or_insert(1);

        Ok(id)
    }

    /// Register a read hold pegged to the last **durable** transaction id
    /// (the primary slot's id), regardless of whether `read_from_secondary`
    /// is currently set.
    ///
    /// This is the correct registration when the caller intends to read the
    /// **persisted** roots (`get_persisted_data_root` / `get_persisted_system_root`).
    /// Using `register_read_transaction` in that case creates an id/root
    /// mismatch: the registered id may be the secondary (non-durable) txn id
    /// `N`, but the persisted root being walked corresponds to the older
    /// primary id `N-K`. A concurrent durable commit's `process_freed_pages`
    /// then frees pages reachable from the primary root (entries in
    /// `DATA_FREED_TABLE` / `SYSTEM_FREED_TABLE` for txns in `N-K+1 ..= N`),
    /// causing the reader to follow pointers into reclaimed pages.
    ///
    /// Pegging the registration to the durable id ensures
    /// `oldest_live_read_transaction <= durable_id`, so `process_freed_pages`
    /// only frees entries with txn `<= durable_id` -- which by construction
    /// are not reachable from the durable root being walked.
    #[cfg(feature = "std")]
    pub(crate) fn register_durable_read_transaction(
        &self,
        mem: &TransactionalMemory,
    ) -> Result<TransactionId> {
        let mut state = self.state.lock()?;
        let id = mem.get_last_durable_transaction_id()?;
        state
            .live_read_transactions
            .entry(id)
            .and_modify(|x| *x += 1)
            .or_insert(1);

        Ok(id)
    }

    /// Atomically attempt to extend the read hold on an existing historical
    /// snapshot. Returns `true` if `transaction_id` was already present in
    /// `live_read_transactions` (i.e. the snapshot's `register_history_hold`
    /// is still alive) and the count was incremented; returns `false` if the
    /// snapshot has already been evicted from the retention window.
    ///
    /// This is the correct registration for time-travel reads
    /// (`Database::begin_read_at`, `Database::begin_read_at_time`): the
    /// caller looks up a historical snapshot in the system history table,
    /// then must extend the snapshot's hold before walking pages reachable
    /// from its `user_root`. Without this, retention pruning could
    /// `deallocate_history_hold(transaction_id)` between the lookup and the
    /// read, allowing `oldest_live_read_transaction` to advance past
    /// `transaction_id` and `process_freed_pages` to reclaim pages
    /// reachable from the historical user tree -- the same id/root mismatch
    /// class fixed for `verify_integrity` and the integrity scanner.
    ///
    /// The check-and-increment is atomic under the tracker lock, so it
    /// serialises correctly with retention pruning's
    /// `deallocate_history_hold` call.
    pub(crate) fn try_register_historical_read_transaction(
        &self,
        transaction_id: TransactionId,
    ) -> Result<bool> {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        if let Some(ref_count) = state.live_read_transactions.get_mut(&transaction_id) {
            *ref_count += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub(crate) fn deallocate_read_transaction(&self, id: TransactionId) -> Result {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        if let Some(ref_count) = state.live_read_transactions.get_mut(&id) {
            *ref_count -= 1;
            if *ref_count == 0 {
                state.live_read_transactions.remove(&id);
            }
        }
        Ok(())
    }

    pub(crate) fn any_savepoint_exists(&self) -> Result<bool> {
        #[cfg(feature = "std")]
        let state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let state = self.state.lock();
        Ok(!state.valid_savepoints.is_empty())
    }

    pub(crate) fn allocate_savepoint(&self, transaction_id: TransactionId) -> Result<SavepointId> {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        let id = state.next_savepoint_id.next()?;
        state.next_savepoint_id = id;
        state.valid_savepoints.insert(id, transaction_id);
        Ok(id)
    }

    // Deallocates the given savepoint and its matching reference count on the transaction
    pub(crate) fn deallocate_savepoint(
        &self,
        savepoint: SavepointId,
        transaction: TransactionId,
    ) -> Result {
        {
            #[cfg(feature = "std")]
            let mut state = self.state.lock()?;
            #[cfg(not(feature = "std"))]
            let mut state = self.state.lock();
            state.valid_savepoints.remove(&savepoint);
        }
        self.deallocate_read_transaction(transaction)
    }

    pub(crate) fn is_valid_savepoint(&self, id: SavepointId) -> Result<bool> {
        #[cfg(feature = "std")]
        let state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let state = self.state.lock();
        Ok(state.valid_savepoints.contains_key(&id))
    }

    pub(crate) fn invalidate_savepoints_after(&self, id: SavepointId) -> Result {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        state.valid_savepoints.retain(|x, _| *x <= id);
        Ok(())
    }

    pub(crate) fn oldest_savepoint(&self) -> Result<Option<(SavepointId, TransactionId)>> {
        #[cfg(feature = "std")]
        let state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let state = self.state.lock();
        Ok(state
            .valid_savepoints
            .first_key_value()
            .map(|x| (*x.0, *x.1)))
    }

    pub(crate) fn oldest_live_read_transaction(&self) -> Result<Option<TransactionId>> {
        #[cfg(feature = "std")]
        let state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let state = self.state.lock();
        Ok(state.live_read_transactions.keys().next().copied())
    }

    pub(crate) fn register_history_hold(&self, transaction_id: TransactionId) -> Result {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        state
            .live_read_transactions
            .entry(transaction_id)
            .and_modify(|x| *x += 1)
            .or_insert(1);
        Ok(())
    }

    pub(crate) fn deallocate_history_hold(&self, transaction_id: TransactionId) -> Result {
        #[cfg(feature = "std")]
        let mut state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let mut state = self.state.lock();
        if let Some(ref_count) = state.live_read_transactions.get_mut(&transaction_id) {
            *ref_count -= 1;
            if *ref_count == 0 {
                state.live_read_transactions.remove(&transaction_id);
            }
        }
        Ok(())
    }

    // Returns the oldest external read transaction -- one not solely held by
    // durable-ancestor registrations from pending non-durable commits. Used to
    // guard non-durable page freeing: if real readers exist, their snapshots
    // must not be invalidated by freeing pages they might traverse.
    pub(crate) fn oldest_live_read_transaction_excluding_nondurable(
        &self,
    ) -> Result<Option<TransactionId>> {
        #[cfg(feature = "std")]
        let state = self.state.lock()?;
        #[cfg(not(feature = "std"))]
        let state = self.state.lock();

        for (id, refcount) in &state.live_read_transactions {
            let nondurable_holds = state.non_durable_ancestor_holds.get(id).copied().unwrap_or(0);
            if *refcount > nondurable_holds {
                return Ok(Some(*id));
            }
        }
        Ok(None)
    }
}

#[cfg(all(test, feature = "std"))]
mod test {
    use super::{TransactionId, TransactionTracker};

    fn tid(n: u64) -> TransactionId {
        TransactionId::new(n)
    }

    /// Non-durable commits hold their durable ancestor with a synthetic read
    /// reference. `oldest_live_read_transaction_excluding_nondurable` must see
    /// through those, and report only references that real readers hold.
    #[test]
    fn excluding_nondurable_ignores_synthetic_holds_only() {
        let tracker = TransactionTracker::new(tid(10));
        // No readers, no pending commits.
        assert_eq!(
            tracker
                .oldest_live_read_transaction_excluding_nondurable()
                .unwrap(),
            None
        );

        // Three non-durable commits on top of durable transaction 5, one on 7.
        for id in [11, 12, 13] {
            tracker.register_non_durable_commit(tid(id), tid(5)).unwrap();
        }
        tracker.register_non_durable_commit(tid(14), tid(7)).unwrap();
        assert_eq!(
            tracker
                .oldest_live_read_transaction_excluding_nondurable()
                .unwrap(),
            None,
            "synthetic holds alone are not readers"
        );

        // A real reader on 7 shows up on top of the synthetic hold.
        *tracker
            .state
            .lock()
            .unwrap()
            .live_read_transactions
            .get_mut(&tid(7))
            .unwrap() += 1;
        assert_eq!(
            tracker
                .oldest_live_read_transaction_excluding_nondurable()
                .unwrap(),
            Some(tid(7))
        );

        // A real reader on 5 is older and wins.
        *tracker
            .state
            .lock()
            .unwrap()
            .live_read_transactions
            .get_mut(&tid(5))
            .unwrap() += 1;
        assert_eq!(
            tracker
                .oldest_live_read_transaction_excluding_nondurable()
                .unwrap(),
            Some(tid(5))
        );

        // The incremental bookkeeping always equals the multiset of ancestors.
        {
            let state = tracker.state.lock().unwrap();
            let mut expected = std::collections::BTreeMap::new();
            for ancestor in state.pending_non_durable_commits.values() {
                *expected.entry(*ancestor).or_insert(0u64) += 1;
            }
            assert_eq!(state.non_durable_ancestor_holds, expected);
        }
    }

    /// A durable commit persists everything the pending non-durable commits
    /// left behind: their synthetic holds, their ancestor bookkeeping and the
    /// "unprocessed" markers must all be gone, whether or not their freed pages
    /// were ever processed.
    #[test]
    fn clearing_pending_commits_resets_all_non_durable_state() {
        let tracker = TransactionTracker::new(tid(10));
        for id in 11..2_011 {
            tracker.register_non_durable_commit(tid(id), tid(5)).unwrap();
        }
        // Only some of them were processed through the freed table.
        for id in (11..2_011).step_by(7) {
            tracker.mark_non_durable_commit_processed(tid(id)).unwrap();
        }
        assert!(tracker.oldest_unprocessed_non_durable_commit().unwrap().is_some());

        tracker.clear_pending_non_durable_commits().unwrap();

        assert_eq!(tracker.oldest_unprocessed_non_durable_commit().unwrap(), None);
        assert!(!tracker.is_unprocessed_non_durable_commit(tid(12)).unwrap());
        let state = tracker.state.lock().unwrap();
        assert!(state.pending_non_durable_commits.is_empty());
        assert!(state.non_durable_ancestor_holds.is_empty());
        assert!(state.unprocessed_freed_non_durable_commits.is_empty());
        assert!(
            state.live_read_transactions.is_empty(),
            "the synthetic holds must be released"
        );
    }

    /// The pending limit still applies, and clearing makes room again.
    #[test]
    fn pending_limit_is_enforced_and_cleared_by_durable_commit() {
        let tracker = TransactionTracker::new(tid(10));
        for id in 0..TransactionTracker::MAX_PENDING_NON_DURABLE as u64 {
            tracker.register_non_durable_commit(tid(100 + id), tid(5)).unwrap();
        }
        assert!(tracker.register_non_durable_commit(tid(1), tid(5)).is_err());
        tracker.clear_pending_non_durable_commits().unwrap();
        tracker.register_non_durable_commit(tid(1), tid(5)).unwrap();
    }
}
