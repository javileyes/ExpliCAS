//! Session reference resolution.

use cas_ast::ExprId;
use cas_session_core::resolve::{
    resolve_all_with_lookup_and_env, resolve_session_refs_with_lookup,
    resolve_session_refs_with_lookup_on_visit,
};
pub use cas_session_core::types::ResolveError;

use super::store::*;

/// Resolve all references in an expression:
/// 1. Resolve session references (`#id`) -> ExprId
/// 2. Substitute environment variables (`x=5`) -> ExprId
pub fn resolve_all(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    store: &SessionStore,
    env: &cas_session_core::env::Environment,
) -> Result<ExprId, ResolveError> {
    let mut lookup = |id: EntryId| store.get(id).map(|entry| entry.kind.clone());
    resolve_all_with_lookup_and_env(ctx, expr, &mut lookup, env)
}

/// Resolve all references and return inherited diagnostics + cache hits.
///
/// When the expression contains session references (`#id`), diagnostics from
/// those entries are accumulated for SessionPropagated origin tracking.
pub fn resolve_all_with_diagnostics(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    store: &SessionStore,
    env: &cas_session_core::env::Environment,
    domain_mode: crate::domain::DomainMode,
) -> Result<(ExprId, crate::diagnostics::Diagnostics, Vec<CacheHitTrace>), ResolveError> {
    let cache_key = SimplifyCacheKey::from_context(domain_mode);

    let resolved =
        resolve_session_refs_with_mode(ctx, expr, store, RefMode::PreferSimplified, &cache_key)?;

    let mut inherited = crate::diagnostics::Diagnostics::new();
    for item in resolved.requires {
        inherited.push_required(
            item.cond,
            crate::diagnostics::RequireOrigin::SessionPropagated,
        );
    }

    let fully_resolved = cas_session_core::env::substitute(ctx, env, resolved.expr);
    Ok((fully_resolved, inherited, resolved.cache_hits))
}

/// Resolve all `Expr::SessionRef` in an expression tree.
///
/// - For expression entries: replaces `#id` with the stored ExprId
/// - For equation entries: replaces `#id` with `(lhs - rhs)` (residue form)
///
/// Uses memoization to avoid re-resolving the same reference.
/// Detects circular references and returns an error.
pub fn resolve_session_refs(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    store: &SessionStore,
) -> Result<ExprId, ResolveError> {
    let mut lookup = |id: EntryId| store.get(id).map(|entry| entry.kind.clone());
    resolve_session_refs_with_lookup(ctx, expr, &mut lookup)
}

/// Resolve session refs AND accumulate inherited diagnostics.
///
/// When an expression references `#id`, the diagnostics from that entry
/// are accumulated for SessionPropagated tracking.
pub fn resolve_session_refs_with_diagnostics(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    store: &SessionStore,
) -> Result<(ExprId, crate::diagnostics::Diagnostics), ResolveError> {
    let mut inherited = crate::diagnostics::Diagnostics::new();
    let mut lookup = |id: EntryId| store.get(id).map(|entry| entry.kind.clone());
    let mut on_visit = |id: EntryId| {
        if let Some(entry) = store.get(id) {
            inherited.inherit_requires_from(&entry.diagnostics);
        }
    };
    let resolved =
        resolve_session_refs_with_lookup_on_visit(ctx, expr, &mut lookup, &mut on_visit)?;
    Ok((resolved, inherited))
}

/// Resolve session refs with mode selection and cache checking (V2.15.36).
///
/// This is the preferred resolution method when you have a `SimplifyCacheKey`.
/// It checks the simplified cache before falling back to raw expressions.
///
/// # Arguments
/// * `ctx` - The expression context
/// * `expr` - Expression to resolve (may contain `#N` references)
/// * `store` - Session store with entries
/// * `mode` - PreferSimplified (use cache) or Raw (use parsed expr)
/// * `cache_key` - Current context key for cache validation
///
/// # Returns
/// * `ResolvedExpr` with resolved expression and accumulated requires
pub fn resolve_session_refs_with_mode(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    store: &SessionStore,
    mode: RefMode,
    cache_key: &SimplifyCacheKey,
) -> Result<ResolvedExpr, ResolveError> {
    let mut lookup = |id: EntryId| {
        let entry = store.get(id)?;
        Some(cas_session_core::resolve::ModeEntry {
            kind: entry.kind.clone(),
            requires: entry.diagnostics.requires.clone(),
            cache: entry.simplified.as_ref().map(|cache| {
                cas_session_core::resolve::ModeCacheEntry {
                    key: cache.key.clone(),
                    expr: cache.expr,
                    requires: cache.requires.clone(),
                }
            }),
        })
    };
    let mut same_requirement =
        |lhs: &crate::diagnostics::RequiredItem, rhs: &crate::diagnostics::RequiredItem| {
            lhs.cond == rhs.cond
        };
    let mut mark_session_propagated = |item: &mut crate::diagnostics::RequiredItem| {
        item.merge_origin(crate::diagnostics::RequireOrigin::SessionPropagated);
    };

    cas_session_core::resolve::resolve_session_refs_with_mode_lookup(
        ctx,
        expr,
        mode,
        cache_key,
        &mut lookup,
        &mut same_requirement,
        &mut mark_session_propagated,
    )
}
