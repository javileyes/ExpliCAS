mod plumbing;

use cas_session_core::cache::ResolvedExpr;
use cas_solver_core::diagnostics_model::{Diagnostics, RequiredItem};

use crate::cache::SimplifyCacheKey;
use crate::env::Environment;
use crate::SessionStore;
use cas_session_core::types::{EntryId, RefMode, ResolveError};

use self::plumbing::{
    mode_resolve_config, push_session_propagated_requirement, with_mode_resolution_plumbing,
};

/// Resolve all `Expr::SessionRef` in an expression tree.
pub fn resolve_session_refs(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
) -> Result<cas_ast::ExprId, ResolveError> {
    let mut lookup = |id: EntryId| store.get(id).map(|entry| entry.kind.clone());
    cas_session_core::resolve::resolve_session_refs_with_lookup(ctx, expr, &mut lookup)
}

/// Resolve session refs and apply environment substitution.
pub fn resolve_session_refs_with_env(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
    env: &Environment,
) -> Result<cas_ast::ExprId, ResolveError> {
    let mut lookup = |id: EntryId| store.get(id).map(|entry| entry.kind.clone());
    cas_session_core::resolve::resolve_all_with_lookup_and_env(ctx, expr, &mut lookup, env)
}

/// Resolve session refs and accumulate inherited diagnostics.
pub fn resolve_session_refs_with_diagnostics(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
) -> Result<(cas_ast::ExprId, Diagnostics), ResolveError> {
    let mut lookup = |id: EntryId| store.get(id).map(|entry| entry.kind.clone());
    cas_session_core::resolve::resolve_session_refs_with_lookup_accumulator(
        ctx,
        expr,
        &mut lookup,
        Diagnostics::new(),
        |inherited, id| {
            if let Some(entry) = store.get(id) {
                inherited.inherit_requires_from(&entry.diagnostics);
            }
        },
    )
}

/// Resolve session refs with mode selection and cache checking.
pub fn resolve_session_refs_with_mode(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
    mode: RefMode,
    cache_key: &SimplifyCacheKey,
) -> Result<ResolvedExpr<RequiredItem>, ResolveError> {
    with_mode_resolution_plumbing(store, |mut lookup, mut same, mut mark| {
        cas_session_core::resolve::resolve_session_refs_with_mode_lookup(
            ctx,
            expr,
            mode,
            cache_key,
            &mut lookup,
            &mut same,
            &mut mark,
        )
    })
}

/// Resolve session refs with mode selection and apply environment substitution.
pub fn resolve_session_refs_with_mode_and_env(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
    mode: RefMode,
    cache_key: &SimplifyCacheKey,
    env: &Environment,
) -> Result<ResolvedExpr<RequiredItem>, ResolveError> {
    with_mode_resolution_plumbing(store, |mut lookup, mut same, mut mark| {
        cas_session_core::resolve::resolve_all_with_mode_lookup_and_env(
            ctx,
            expr,
            mode_resolve_config(mode, cache_key, env),
            &mut lookup,
            &mut same,
            &mut mark,
        )
    })
}

/// Resolve session refs with mode + env and return inherited diagnostics + cache hits.
pub fn resolve_session_refs_with_mode_and_diagnostics(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
    mode: RefMode,
    cache_key: &SimplifyCacheKey,
    env: &Environment,
) -> Result<(cas_ast::ExprId, Diagnostics, Vec<EntryId>), ResolveError> {
    with_mode_resolution_plumbing(store, |mut lookup, mut same, mut mark| {
        cas_session_core::resolve::resolve_mode_with_env_and_diagnostics(
            ctx,
            expr,
            mode_resolve_config(mode, cache_key, env),
            &mut lookup,
            &mut same,
            &mut mark,
            Diagnostics::new(),
            push_session_propagated_requirement,
        )
    })
}
