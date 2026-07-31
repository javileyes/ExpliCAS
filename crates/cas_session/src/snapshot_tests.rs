use std::fs::File;
use std::io::{BufWriter, Write};

use crate::snapshot::SessionSnapshot;
use crate::{cache::SimplifyCacheKey, env::Environment, state_core::SessionState, SessionStore};
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
    let env = Environment::new();
    let snapshot = SessionSnapshot::new(&ctx, &store, &env, key.clone());
    snapshot.save_atomic(&path).expect("save");

    // Load
    let loaded = SessionSnapshot::load(&path).expect("load");
    assert!(loaded.is_compatible(&key));

    // Verify
    let (restored_ctx, restored_store, restored_env) = loaded.into_parts_with_env();
    assert_eq!(ctx.nodes.len(), restored_ctx.nodes.len());
    assert_eq!(store.len(), restored_store.len());
    assert!(restored_env.is_empty());
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

#[test]
fn test_session_snapshot_save_load_preserves_function_bindings() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("test-functions.session");

    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);
    let body = ctx.add(cas_ast::Expr::Add(x, one));

    let store = SessionStore::new();
    let mut env = Environment::new();
    env.set_function("f".to_string(), vec!["x".to_string()], body);

    let key = SimplifyCacheKey::from_domain_flag("generic");
    let snapshot = SessionSnapshot::new(&ctx, &store, &env, key.clone());
    snapshot.save_atomic(&path).expect("save");

    let loaded = SessionSnapshot::load(&path).expect("load");
    let (_restored_ctx, _restored_store, restored_env) = loaded.into_parts_with_env();
    let binding = restored_env.get_function("f").expect("restored function");
    assert_eq!(binding.params, vec!["x".to_string()]);
}

/// The wasm stop/restore cycle, natively: encode the session to BYTES (no
/// filesystem), decode into a fresh engine+state, and verify that BOTH the
/// `#N` store and the `:=` environment keep resolving. This is the coverage
/// for `WasmSession::snapshot`/`restore`, which are thin wrappers over these
/// two calls.
#[test]
fn snapshot_bytes_roundtrip_preserves_refs_and_bindings() {
    use cas_api_models::{
        EvalAssumeScope, EvalBranchMode, EvalBudgetPreset, EvalComplexMode, EvalConstFoldMode,
        EvalContextMode, EvalDomainMode, EvalExpandPolicy, EvalInvTrigPolicy, EvalStepsMode,
        EvalValueDomain,
    };

    fn config(expr: &str) -> crate::eval::EvalCommandConfig<'_> {
        crate::eval::EvalCommandConfig {
            expr,
            auto_store: true,
            max_chars: 2000,
            time_budget_ms: None,
            steps_mode: EvalStepsMode::Off,
            budget_preset: EvalBudgetPreset::Standard,
            strict: false,
            domain: EvalDomainMode::Generic,
            context_mode: EvalContextMode::Auto,
            branch_mode: EvalBranchMode::Strict,
            expand_policy: EvalExpandPolicy::Auto,
            complex_mode: EvalComplexMode::Auto,
            const_fold: EvalConstFoldMode::Safe,
            value_domain: EvalValueDomain::Real,
            complex_branch: EvalBranchMode::Principal,
            inv_trig: EvalInvTrigPolicy::Strict,
            assume_scope: EvalAssumeScope::Real,
            numeric_display: cas_api_models::EvalNumericDisplay::Exact,
            approx_hint: false,
        }
    }
    fn eval(
        engine: &mut cas_solver::runtime::Engine,
        state: &mut SessionState,
        expr: &str,
    ) -> cas_api_models::EvalWireOutput {
        crate::eval::evaluate_eval_command_in_memory_with_state(
            engine,
            state,
            config(expr),
            cas_solver_core::eval_option_axes::Language::Es,
            |_steps, _events, _ctx, _mode| Vec::new(),
        )
        .expect("eval")
    }

    let mut engine = cas_solver::runtime::Engine::new();
    let mut state = SessionState::new();
    let first = eval(&mut engine, &mut state, "20 + 22");
    assert_eq!(first.result, "42");
    assert_eq!(first.stored_id, Some(1));
    eval(&mut engine, &mut state, "zz := 7");

    // The worker's pre-eval snapshot…
    let bytes = state
        .encode_snapshot_bytes(&engine.simplifier.context, "generic")
        .expect("encode");

    // …restored into a brand-new engine after the worker was terminated.
    let (context, restored_state) =
        SessionState::decode_compatible_snapshot_bytes(&bytes, "generic")
            .expect("decode")
            .expect("compatible");
    let mut engine2 = cas_solver::runtime::Engine::with_context(context);
    let mut state2 = restored_state;

    let resumed = eval(&mut engine2, &mut state2, "#1 + zz");
    assert_eq!(resumed.result, "49", "wire: {:?}", resumed.result);

    // A wrong domain seal declines instead of restoring garbage.
    assert!(
        SessionState::decode_compatible_snapshot_bytes(&bytes, "strict")
            .expect("decode ok")
            .is_none()
    );
}
