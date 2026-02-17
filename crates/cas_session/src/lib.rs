//! Session-related components extracted from `cas_engine`.

pub mod env;
mod snapshot;
mod state;

pub use cas_engine::session::{
    resolve_session_refs, resolve_session_refs_with_diagnostics, resolve_session_refs_with_mode,
    CacheHitTrace, Entry, ResolvedExpr, SessionStore, SimplifiedCache, SimplifyCacheKey,
};
pub use cas_session_core::types::{CacheConfig, EntryId, EntryKind, RefMode, ResolveError};
pub use env::{is_reserved, substitute, substitute_with_shadow, Environment};
pub use snapshot::{SessionSnapshot, SnapshotError};
pub use state::SessionState;

/// Resolve session references (`#N`) and environment bindings from raw parts.
pub fn resolve_all(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
    env: &Environment,
) -> Result<cas_ast::ExprId, ResolveError> {
    let mut lookup = |id: EntryId| store.get(id).map(|entry| entry.kind.clone());
    cas_session_core::resolve::resolve_all_with_lookup_and_env(ctx, expr, &mut lookup, env)
}

/// Resolve references and return inherited diagnostics + cache hit traces.
pub fn resolve_all_with_diagnostics(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
    env: &Environment,
    domain_mode: cas_engine::domain::DomainMode,
) -> Result<
    (
        cas_ast::ExprId,
        cas_engine::diagnostics::Diagnostics,
        Vec<CacheHitTrace>,
    ),
    ResolveError,
> {
    cas_engine::session::resolve_all_with_diagnostics(ctx, expr, store, env, domain_mode)
}

/// Resolve session references (`#N`) and environment bindings from `SessionState`.
pub fn resolve_all_from_state(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    state: &SessionState,
) -> Result<cas_ast::ExprId, ResolveError> {
    resolve_all(ctx, expr, &state.store, &state.env)
}

/// Resolve references and return inherited diagnostics + cache hit traces.
pub fn resolve_all_with_diagnostics_from_state(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    state: &SessionState,
) -> Result<
    (
        cas_ast::ExprId,
        cas_engine::diagnostics::Diagnostics,
        Vec<CacheHitTrace>,
    ),
    ResolveError,
> {
    resolve_all_with_diagnostics(
        ctx,
        expr,
        &state.store,
        &state.env,
        state.options.shared.semantics.domain_mode,
    )
}
