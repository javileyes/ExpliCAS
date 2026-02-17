//! Stateless helpers for session and environment resolution.
//!
//! These functions resolve session references (`#N`) and environment bindings
//! without requiring a full `SessionState` object, which helps decouple the
//! engine pipeline from session storage concerns.

use crate::env::Environment;
use crate::session::{
    resolve_session_refs, resolve_session_refs_with_mode, CacheHitTrace, RefMode, ResolveError,
    SessionStore, SimplifyCacheKey,
};
use cas_ast::{Context, ExprId};

/// Resolve all references in an expression:
/// 1. Resolve session references (#id) -> ExprId
/// 2. Substitute environment variables (x=5) -> ExprId
pub fn resolve_all(
    ctx: &mut Context,
    expr: ExprId,
    store: &SessionStore,
    env: &Environment,
) -> Result<ExprId, ResolveError> {
    let expr_with_refs = resolve_session_refs(ctx, expr, store)?;
    Ok(crate::env::substitute(ctx, env, expr_with_refs))
}

/// Resolve all references and return inherited diagnostics + cache hits.
///
/// When the expression contains session references (`#id`), diagnostics from
/// those entries are accumulated for SessionPropagated origin tracking.
pub fn resolve_all_with_diagnostics(
    ctx: &mut Context,
    expr: ExprId,
    store: &SessionStore,
    env: &Environment,
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

    let fully_resolved = crate::env::substitute(ctx, env, resolved.expr);
    Ok((fully_resolved, inherited, resolved.cache_hits))
}
