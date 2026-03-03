use crate::snapshot::SessionSnapshot;
use crate::{SessionStore, SimplifyCacheKey};
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
        domain: crate::CacheDomainMode::Generic,
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
