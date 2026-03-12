use std::fs::File;
use std::io::{BufWriter, Write};

use crate::snapshot::SessionSnapshot;
use crate::{state_core::SessionState, SessionStore, SimplifyCacheKey};
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
    store.push(crate::EntryKind::Expr(expr), "x + 1".to_string());

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
