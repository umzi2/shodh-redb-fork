//! Coverage for the "insert past the last key" fast path in the B-tree
//! mutator: ascending loads must pack leaves and must never lose, reorder or
//! corrupt data, whatever the key/value shapes and transaction boundaries.

use shodh_redb::{
    Database, Durability, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition,
};

const U64_BYTES: TableDefinition<u64, &[u8]> = TableDefinition::new("u64_bytes");
const STR_BYTES: TableDefinition<&str, &[u8]> = TableDefinition::new("str_bytes");
const U64_STR: TableDefinition<u64, String> = TableDefinition::new("u64_str");

fn tmp() -> tempfile::NamedTempFile {
    tempfile::NamedTempFile::new().unwrap()
}

/// Deterministic xorshift so failures reproduce.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn value_for(i: u64, len: usize) -> Vec<u8> {
    (0..len).map(|j| (i as u8).wrapping_add(j as u8)).collect()
}

#[test]
fn variable_length_values_ascending_roundtrip() {
    let file = tmp();
    let mut db = Database::create(file.path()).unwrap();
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut lens = Vec::new();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_BYTES).unwrap();
        for i in 0u64..6_000 {
            // 0 .. 3000 bytes: some entries span most of a page, some are empty.
            let len = (rng.next() % 3000) as usize;
            lens.push(len);
            t.insert(i, value_for(i, len).as_slice()).unwrap();
        }
        assert_eq!(t.len().unwrap(), 6_000);
    }
    txn.commit().unwrap();

    let rtxn = db.begin_read().unwrap();
    let t = rtxn.open_table(U64_BYTES).unwrap();
    assert_eq!(t.len().unwrap(), 6_000);
    for (i, len) in lens.iter().enumerate() {
        let got = t.get(i as u64).unwrap().unwrap();
        assert_eq!(got.value(), value_for(i as u64, *len).as_slice(), "key {i}");
    }
    let keys: Vec<u64> = t.iter().unwrap().map(|r| r.unwrap().0.value()).collect();
    assert!(keys.windows(2).all(|w| w[0] < w[1]), "iteration out of order");
    drop(t);
    drop(rtxn);
    db.check_integrity().unwrap();
}

#[test]
fn string_keys_ascending_roundtrip() {
    let file = tmp();
    let mut db = Database::create(file.path()).unwrap();
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(STR_BYTES).unwrap();
        for i in 0u64..8_000 {
            // Zero padded => lexicographic order == numeric order.
            let key = format!("user:{i:012}:{}", "k".repeat((i % 40) as usize));
            t.insert(key.as_str(), value_for(i, 60).as_slice()).unwrap();
        }
    }
    txn.commit().unwrap();

    let rtxn = db.begin_read().unwrap();
    let t = rtxn.open_table(STR_BYTES).unwrap();
    assert_eq!(t.len().unwrap(), 8_000);
    let mut prev: Option<String> = None;
    let mut n = 0u64;
    for r in t.iter().unwrap() {
        let (k, v) = r.unwrap();
        let k = k.value().to_string();
        if let Some(p) = &prev {
            assert!(*p < k);
        }
        assert_eq!(v.value(), value_for(n, 60).as_slice());
        prev = Some(k);
        n += 1;
    }
    assert_eq!(n, 8_000);
    drop(t);
    drop(rtxn);
    db.check_integrity().unwrap();
}

/// Same key written again and again at the right edge: replaces, never grows.
#[test]
fn repeated_overwrite_of_last_key() {
    let file = tmp();
    let db = Database::create(file.path()).unwrap();
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_BYTES).unwrap();
        for i in 0u64..3_000 {
            t.insert(i, value_for(i, 100).as_slice()).unwrap();
        }
        for round in 0..500u64 {
            let old = t
                .insert(2_999, value_for(round, 100).as_slice())
                .unwrap()
                .map(|g| g.value().to_vec());
            if round == 0 {
                assert_eq!(old.unwrap(), value_for(2_999, 100));
            } else {
                assert_eq!(old.unwrap(), value_for(round - 1, 100));
            }
        }
        assert_eq!(t.len().unwrap(), 3_000);
        assert_eq!(t.get(2_999).unwrap().unwrap().value(), value_for(499, 100).as_slice());
    }
    txn.commit().unwrap();
}

/// Ascending loads split across many commits, with reopen in between.
#[test]
fn ascending_across_commits_and_reopen() {
    let file = tmp();
    let mut next = 0u64;
    for round in 0..12 {
        let db = Database::create(file.path()).unwrap();
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(U64_BYTES).unwrap();
            for _ in 0..1_500 {
                t.insert(next, value_for(next, 90).as_slice()).unwrap();
                next += 1;
            }
        }
        txn.commit().unwrap();
        if round % 3 == 0 {
            // non-durable commit followed by a durable one must keep both
            let mut txn = db.begin_write().unwrap();
            txn.set_durability(Durability::None).unwrap();
            {
                let mut t = txn.open_table(U64_BYTES).unwrap();
                t.insert(next, value_for(next, 90).as_slice()).unwrap();
                next += 1;
            }
            txn.commit().unwrap();
            let txn = db.begin_write().unwrap();
            txn.commit().unwrap();
        }
    }
    let mut db = Database::create(file.path()).unwrap();
    {
        let rtxn = db.begin_read().unwrap();
        let t = rtxn.open_table(U64_BYTES).unwrap();
        assert_eq!(t.len().unwrap(), next);
        for i in (0..next).step_by(97) {
            assert_eq!(t.get(i).unwrap().unwrap().value(), value_for(i, 90).as_slice());
        }
        let stats = t.stats().unwrap();
        // 90B value + 8B key ~ 38 per 4K page; packed => close to next/38 leaves.
        let max_leaves = (next / 30) as u64;
        assert!(stats.leaf_pages() < max_leaves, "{} leaves for {next} rows", stats.leaf_pages());
    }
    db.check_integrity().unwrap();
}

/// Ascending load, delete a random half, then insert past the end again and
/// into the holes: tree stays consistent.
#[test]
fn ascending_then_delete_then_mixed_insert() {
    let file = tmp();
    let mut db = Database::create(file.path()).unwrap();
    let mut rng = Rng(42);
    let mut model = std::collections::BTreeMap::new();

    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_STR).unwrap();
        for i in 0u64..10_000 {
            let v = format!("v{i}-{}", "z".repeat((i % 50) as usize));
            t.insert(i, v.clone()).unwrap();
            model.insert(i, v);
        }
        for _ in 0..5_000 {
            let k = rng.next() % 10_000;
            let a = t.remove(k).unwrap().map(|g| g.value());
            assert_eq!(a, model.remove(&k));
        }
        for i in 10_000u64..12_000 {
            let v = format!("tail{i}");
            t.insert(i, v.clone()).unwrap();
            model.insert(i, v);
        }
        for i in 0u64..2_000 {
            let k = rng.next() % 12_000;
            let v = format!("mix{i}");
            t.insert(k, v.clone()).unwrap();
            model.insert(k, v);
        }
    }
    txn.commit().unwrap();

    let rtxn = db.begin_read().unwrap();
    let t = rtxn.open_table(U64_STR).unwrap();
    assert_eq!(t.len().unwrap(), model.len() as u64);
    let got: Vec<(u64, String)> = t
        .iter()
        .unwrap()
        .map(|r| {
            let (k, v) = r.unwrap();
            (k.value(), v.value())
        })
        .collect();
    let want: Vec<(u64, String)> = model.into_iter().collect();
    assert_eq!(got, want);
    drop(t);
    drop(rtxn);
    db.check_integrity().unwrap();
}

/// Descending and "zig-zag" loads must not be affected by the append path.
#[test]
fn descending_and_zigzag_loads() {
    let file = tmp();
    let mut db = Database::create(file.path()).unwrap();
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_BYTES).unwrap();
        for i in (0u64..4_000).rev() {
            t.insert(i, value_for(i, 80).as_slice()).unwrap();
        }
        // zig-zag above the existing range
        let (mut lo, mut hi) = (4_000u64, 8_999u64);
        while lo <= hi {
            t.insert(lo, value_for(lo, 80).as_slice()).unwrap();
            if lo != hi {
                t.insert(hi, value_for(hi, 80).as_slice()).unwrap();
            }
            lo += 1;
            hi -= 1;
        }
    }
    txn.commit().unwrap();
    let rtxn = db.begin_read().unwrap();
    let t = rtxn.open_table(U64_BYTES).unwrap();
    assert_eq!(t.len().unwrap(), 9_000);
    let keys: Vec<u64> = t.iter().unwrap().map(|r| r.unwrap().0.value()).collect();
    assert_eq!(keys, (0..9_000).collect::<Vec<_>>());
    drop(t);
    drop(rtxn);
    db.check_integrity().unwrap();
}

/// A reader holding a snapshot must be unaffected by an ascending writer that
/// appends to (and so leaves behind) the leaf the reader is using.
#[test]
fn snapshot_isolated_from_ascending_writer() {
    let file = tmp();
    let db = Database::create(file.path()).unwrap();
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_BYTES).unwrap();
        for i in 0u64..2_000 {
            t.insert(i, value_for(i, 100).as_slice()).unwrap();
        }
    }
    txn.commit().unwrap();

    let reader = db.begin_read().unwrap();
    for chunk in 0..10u64 {
        let txn = db.begin_write().unwrap();
        {
            let mut t = txn.open_table(U64_BYTES).unwrap();
            for i in 2_000 + chunk * 500..2_000 + (chunk + 1) * 500 {
                t.insert(i, value_for(i, 100).as_slice()).unwrap();
            }
        }
        txn.commit().unwrap();
    }
    let t = reader.open_table(U64_BYTES).unwrap();
    assert_eq!(t.len().unwrap(), 2_000);
    assert!(t.get(2_000).unwrap().is_none());
    for i in 0..2_000u64 {
        assert_eq!(t.get(i).unwrap().unwrap().value(), value_for(i, 100).as_slice());
    }
}

#[cfg(feature = "compression_lz4")]
#[test]
fn ascending_with_lz4_compression() {
    use shodh_redb::CompressionConfig;
    let file = tmp();
    let mut db = Database::builder()
        .set_compression(CompressionConfig::Lz4)
        .create(file.path())
        .unwrap();
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(U64_BYTES).unwrap();
        for i in 0u64..5_000 {
            let v = vec![(i % 7) as u8; 300 + (i % 400) as usize];
            t.insert(i, v.as_slice()).unwrap();
        }
    }
    txn.commit().unwrap();
    let rtxn = db.begin_read().unwrap();
    let t = rtxn.open_table(U64_BYTES).unwrap();
    for i in 0u64..5_000 {
        let want = vec![(i % 7) as u8; 300 + (i % 400) as usize];
        assert_eq!(t.get(i).unwrap().unwrap().value(), want.as_slice());
    }
    let mut n = 0u64;
    for r in t.iter().unwrap() {
        let (k, v) = r.unwrap();
        assert_eq!(k.value(), n);
        assert_eq!(v.value().len(), 300 + (n % 400) as usize);
        n += 1;
    }
    assert_eq!(n, 5_000);
    drop(t);
    drop(rtxn);
    db.check_integrity().unwrap();
}
