//! Session-related components extracted from `cas_engine`.

pub mod env;

pub use cas_engine::session::{
    resolve_session_refs, resolve_session_refs_with_diagnostics, resolve_session_refs_with_mode,
    CacheHitTrace, Entry, ResolvedExpr, SessionStore, SimplifiedCache, SimplifyCacheKey,
};
pub use cas_engine::session_snapshot::{SessionSnapshot, SnapshotError};
pub use cas_engine::session_state::SessionState;
pub use cas_session_core::types::{CacheConfig, EntryId, EntryKind, RefMode, ResolveError};
pub use env::{is_reserved, substitute, substitute_with_shadow, Environment};

/// Resolve session references (`#N`) and environment bindings from raw parts.
pub fn resolve_all(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
    env: &Environment,
) -> Result<cas_ast::ExprId, ResolveError> {
    let expr_with_refs = resolve_session_refs(ctx, expr, store)?;
    Ok(substitute(ctx, env, expr_with_refs))
}

/// Resolve session references (`#N`) and environment bindings from `SessionState`.
pub fn resolve_all_from_state(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    state: &SessionState,
) -> Result<cas_ast::ExprId, ResolveError> {
    resolve_all(ctx, expr, &state.store, &state.env)
}
