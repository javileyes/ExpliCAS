use std::fs::File;
use std::io::{BufWriter, Write};

use crate::snapshot::SessionSnapshot;
use crate::{cache::SimplifyCacheKey, state_core::SessionState, SessionStore};
use cas_session_core::snapshot_header::SnapshotHeader;
use tempfile::tempdir;

#[test]
fn test_session_snapshot_save_load() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("test.session");

    // Create a context with some expressions
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);
    let expr = ctx.add(cas_ast::Expr::Add(x, one));

    // Create a session store with an entry
    let mut store = SessionStore::new();
    store.push(
        cas_session_core::types::EntryKind::Expr(expr),
        "x + 1".to_string(),
    );

    let key = SimplifyCacheKey {
        domain: crate::cache::CacheDomainMode::Generic,
        ruleset_rev: 1,
    };

    // Save
    let snapshot = SessionSnapshot::new(&ctx, &store, key.clone());
    snapshot.save_atomic(&path).expect("save");

    // Load
    let loaded = SessionSnapshot::load(&path).expect("load");
    assert!(loaded.is_compatible(&key));

    // Verify
    let (restored_ctx, restored_store) = loaded.into_parts();
    assert_eq!(ctx.nodes.len(), restored_ctx.nodes.len());
    assert_eq!(store.len(), restored_store.len());
}

#[test]
fn test_load_compatible_snapshot_short_circuits_before_payload_on_incompatible_header() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("test-incompatible.session");

    let file = File::create(&path).expect("create");
    let mut writer = BufWriter::new(file);
    let header = SnapshotHeader::new(
        SessionSnapshot::MAGIC,
        SessionSnapshot::VERSION,
        SimplifyCacheKey::from_domain_flag("strict"),
    );
    bincode::serialize_into(&mut writer, &header).expect("serialize header");
    writer
        .write_all(b"this-is-not-a-valid-context-payload")
        .expect("write trailing garbage");
    writer.flush().expect("flush");

    let loaded = SessionState::load_compatible_snapshot(
        &path,
        &SimplifyCacheKey::from_domain_flag("generic"),
    )
    .expect("load incompatible snapshot");

    assert!(loaded.is_none());
}

#[test]
fn test_save_snapshot_dirty_seed_structural_regression_guard() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("dirty-seed.session");

    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);
    let expr = ctx.add(cas_ast::Expr::Add(x, one));

    let mut state = SessionState::new();
    state.history_push(
        cas_session_core::types::EntryKind::Expr(expr),
        "x + 1".to_string(),
    );

    let key = SimplifyCacheKey::from_domain_flag("generic");
    state
        .save_snapshot(&ctx, &path, key.clone())
        .expect("save dirty-seed snapshot");

    let metadata = std::fs::metadata(&path).expect("snapshot metadata");
    assert!(
        metadata.len() > 0,
        "dirty-seed snapshot should write a non-empty payload"
    );
    assert!(
        metadata.len() <= 4 * 1024,
        "dirty-seed snapshot should stay compact; got {} bytes",
        metadata.len()
    );

    let (loaded_ctx, loaded_state) = SessionState::load_compatible_snapshot(&path, &key)
        .expect("load compatible snapshot")
        .expect("compatible snapshot");
    assert_eq!(
        loaded_state.history_len(),
        1,
        "dirty-seed snapshot should preserve exactly one stored entry"
    );
    assert!(
        loaded_ctx.nodes.len() <= 16,
        "dirty-seed snapshot should avoid pathological context growth; got {} nodes",
        loaded_ctx.nodes.len()
    );
}

#[test]
fn test_save_snapshot_overwrite_dirty_seed_structural_regression_guard() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("dirty-seed-overwrite.session");

    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);
    let expr = ctx.add(cas_ast::Expr::Add(x, one));

    let mut state = SessionState::new();
    state.history_push(
        cas_session_core::types::EntryKind::Expr(expr),
        "x + 1".to_string(),
    );

    let key = SimplifyCacheKey::from_domain_flag("generic");
    state
        .save_snapshot(&ctx, &path, key.clone())
        .expect("initial overwrite seed save");
    state
        .save_snapshot(&ctx, &path, key.clone())
        .expect("overwrite dirty-seed snapshot");

    let metadata = std::fs::metadata(&path).expect("snapshot metadata");
    assert!(
        metadata.len() > 0,
        "overwrite dirty-seed snapshot should write a non-empty payload"
    );
    assert!(
        metadata.len() <= 4 * 1024,
        "overwrite dirty-seed snapshot should stay compact; got {} bytes",
        metadata.len()
    );

    let (loaded_ctx, loaded_state) = SessionState::load_compatible_snapshot(&path, &key)
        .expect("load compatible snapshot")
        .expect("compatible snapshot");
    assert_eq!(
        loaded_state.history_len(),
        1,
        "overwrite dirty-seed snapshot should preserve exactly one stored entry"
    );
    assert!(
        loaded_ctx.nodes.len() <= 16,
        "overwrite dirty-seed snapshot should avoid pathological context growth; got {} nodes",
        loaded_ctx.nodes.len()
    );
}

#[test]
fn test_save_snapshot_overwrite_after_mutation_preserves_both_entries() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("dirty-seed-overwrite-mutated.session");

    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);
    let two = ctx.num(2);
    let expr1 = ctx.add(cas_ast::Expr::Add(x, one));
    let expr2 = ctx.add(cas_ast::Expr::Add(x, two));

    let mut state = SessionState::new();
    state.history_push(
        cas_session_core::types::EntryKind::Expr(expr1),
        "x + 1".to_string(),
    );

    let key = SimplifyCacheKey::from_domain_flag("generic");
    state
        .save_snapshot(&ctx, &path, key.clone())
        .expect("initial overwrite seed save");

    state.history_push(
        cas_session_core::types::EntryKind::Expr(expr2),
        "x + 2".to_string(),
    );
    state
        .save_snapshot(&ctx, &path, key.clone())
        .expect("overwrite after mutation");

    let (loaded_ctx, loaded_state) = SessionState::load_compatible_snapshot(&path, &key)
        .expect("load compatible snapshot")
        .expect("compatible snapshot");
    assert_eq!(
        loaded_state.history_len(),
        2,
        "overwrite-after-mutation snapshot should preserve both stored entries"
    );
    assert!(
        loaded_ctx.nodes.len() <= 16,
        "overwrite-after-mutation snapshot should avoid pathological context growth; got {} nodes",
        loaded_ctx.nodes.len()
    );
}
