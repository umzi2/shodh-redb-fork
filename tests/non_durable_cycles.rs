//! Long runs of non-durable commits punctuated by durable ones: the tracker
//! state for pending non-durable commits must be reset at every durable commit
//! (it used to grow without bound), and neither data nor free space may leak.

use shodh_redb::{
    Database, Durability, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition,
};
use std::collections::BTreeMap;

const T: TableDefinition<u64, &[u8]> = TableDefinition::new("nd");

fn value(seed: u64) -> Vec<u8> {
    vec![(seed % 251) as u8; 40 + (seed % 90) as usize]
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[test]
fn many_non_durable_commits_then_durable_keep_data_and_space() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let mut db = Database::create(file.path()).unwrap();
    let mut model: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
    let mut rng = Rng(0xDEAD_BEEF_1234_5678);
    let mut sizes = Vec::new();

    for cycle in 0..24u64 {
        for i in 0..400u64 {
            let mut txn = db.begin_write().unwrap();
            txn.set_durability(Durability::None).unwrap();
            {
                let mut t = txn.open_table(T).unwrap();
                // A mix that frees pages: overwrite, insert, remove.
                for _ in 0..3 {
                    let k = rng.next() % 3_000;
                    match rng.next() % 4 {
                        0 => {
                            t.remove(&k).unwrap();
                            model.remove(&k);
                        }
                        _ => {
                            let v = value(cycle * 1_000 + i + k);
                            t.insert(&k, v.as_slice()).unwrap();
                            model.insert(k, v);
                        }
                    }
                }
            }
            txn.commit().unwrap();
        }
        // The durable commit that settles the pending ones.
        let txn = db.begin_write().unwrap();
        txn.commit().unwrap();
        sizes.push(file.as_file().metadata().unwrap().len());
    }

    // Free space is recycled: the file stops growing once it is warm.
    let warm = sizes[6];
    let last = *sizes.last().unwrap();
    assert!(
        last <= warm * 3 / 2,
        "file kept growing across cycles: {sizes:?}"
    );

    let rtxn = db.begin_read().unwrap();
    let t = rtxn.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), model.len() as u64);
    for (k, v) in &model {
        assert_eq!(t.get(k).unwrap().unwrap().value(), v.as_slice(), "key {k}");
    }
    drop(t);
    drop(rtxn);
    db.check_integrity().unwrap();
}

/// A reader opened before a run of non-durable commits keeps its snapshot, and
/// pages it may still traverse are not recycled underneath it.
#[test]
fn reader_snapshot_survives_non_durable_commits() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let db = Database::create(file.path()).unwrap();
    {
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(T).unwrap();
            for k in 0..2_000u64 {
                t.insert(&k, value(k).as_slice()).unwrap();
            }
        }
        txn.commit().unwrap();
    }

    let reader = db.begin_read().unwrap();

    for round in 0..300u64 {
        let mut txn = db.begin_write().unwrap();
        txn.set_durability(Durability::None).unwrap();
        {
            let mut t = txn.open_table(T).unwrap();
            for k in (round..2_000).step_by(37) {
                t.insert(&k, value(round + 5_000 + k).as_slice()).unwrap();
            }
        }
        txn.commit().unwrap();
        if round % 100 == 99 {
            let txn = db.begin_write().unwrap();
            txn.commit().unwrap();
        }
    }

    let t = reader.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 2_000);
    for k in 0..2_000u64 {
        assert_eq!(
            t.get(&k).unwrap().unwrap().value(),
            value(k).as_slice(),
            "reader's snapshot changed at key {k}"
        );
    }
    drop(t);
    drop(reader);

    // A fresh reader sees the latest state.
    let fresh = db.begin_read().unwrap();
    let t = fresh.open_table(T).unwrap();
    assert_eq!(t.len().unwrap(), 2_000);
    assert_eq!(
        t.get(&299).unwrap().unwrap().value(),
        value(299 + 5_000 + 299).as_slice()
    );
}

/// The pending limit is a hard error, and one durable commit lifts it.
#[test]
fn durable_commit_lifts_the_pending_non_durable_limit() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let db = Database::create(file.path()).unwrap();
    let mut failed_at = None;
    for i in 0..12_000u64 {
        let mut txn = db.begin_write().unwrap();
        txn.set_durability(Durability::None).unwrap();
        {
            let mut t = txn.open_table(T).unwrap();
            t.insert(&(i % 10), value(i).as_slice()).unwrap();
        }
        if txn.commit().is_err() {
            failed_at = Some(i);
            break;
        }
    }
    let failed_at = failed_at.expect("the pending limit must trigger");
    assert!((9_000..=10_100).contains(&failed_at), "failed at {failed_at}");

    let txn = db.begin_write().unwrap();
    txn.commit().unwrap();

    let mut txn = db.begin_write().unwrap();
    txn.set_durability(Durability::None).unwrap();
    {
        let mut t = txn.open_table(T).unwrap();
        t.insert(&1, value(1).as_slice()).unwrap();
    }
    txn.commit().unwrap();
}
