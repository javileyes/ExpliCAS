//! Session-related components extracted from `cas_engine`.

pub mod env;
mod snapshot;
mod state;

pub type SimplifyCacheKey = cas_engine::eval::SimplifyCacheKey;
pub type SimplifiedCache = cas_engine::eval::SimplifiedCache;
pub type CacheHitTrace = cas_engine::eval::CacheHitTrace;
pub type ResolvedExpr =
    cas_session_core::cache::ResolvedExpr<cas_engine::diagnostics::RequiredItem>;
#[derive(Debug, Clone)]
pub struct SessionSimplifiedCache {
    inner: SimplifiedCache,
}

impl SessionSimplifiedCache {
    fn as_engine(&self) -> &SimplifiedCache {
        &self.inner
    }
}

impl From<SimplifiedCache> for SessionSimplifiedCache {
    fn from(inner: SimplifiedCache) -> Self {
        Self { inner }
    }
}

impl cas_session_core::store::SessionCacheValue for SessionSimplifiedCache {
    fn steps_len(&self) -> usize {
        self.inner.steps_len()
    }

    fn apply_light_cache(self, light_cache_threshold: Option<usize>) -> Self {
        Self {
            inner: self.inner.apply_light_cache(light_cache_threshold),
        }
    }
}

pub type Entry =
    cas_session_core::store::Entry<cas_engine::diagnostics::Diagnostics, SessionSimplifiedCache>;
pub type SessionStore = cas_session_core::store::SessionStore<
    cas_engine::diagnostics::Diagnostics,
    SessionSimplifiedCache,
>;
pub use cas_session_core::types::{CacheConfig, EntryId, EntryKind, RefMode, ResolveError};
pub use env::{is_reserved, substitute, substitute_with_shadow, Environment};
pub use snapshot::SnapshotError;
pub use state::SessionState;

fn map_cache_hit_traces(
    hits: Vec<cas_session_core::cache::CacheHitTrace<cas_engine::diagnostics::RequiredItem>>,
) -> Vec<CacheHitTrace> {
    hits.into_iter()
        .map(|h| CacheHitTrace {
            entry_id: h.entry_id,
        })
        .collect()
}

/// Resolve all `Expr::SessionRef` in an expression tree.
pub fn resolve_session_refs(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
) -> Result<cas_ast::ExprId, ResolveError> {
    let mut lookup = |id: EntryId| store.get(id).map(|entry| entry.kind.clone());
    cas_session_core::resolve::resolve_session_refs_with_lookup(ctx, expr, &mut lookup)
}

/// Resolve session refs and accumulate inherited diagnostics.
pub fn resolve_session_refs_with_diagnostics(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
) -> Result<(cas_ast::ExprId, cas_engine::diagnostics::Diagnostics), ResolveError> {
    let mut inherited = cas_engine::diagnostics::Diagnostics::new();
    let mut lookup = |id: EntryId| store.get(id).map(|entry| entry.kind.clone());
    let mut on_visit = |id: EntryId| {
        if let Some(entry) = store.get(id) {
            inherited.inherit_requires_from(&entry.diagnostics);
        }
    };
    let resolved = cas_session_core::resolve::resolve_session_refs_with_lookup_on_visit(
        ctx,
        expr,
        &mut lookup,
        &mut on_visit,
    )?;
    Ok((resolved, inherited))
}

/// Resolve session refs with mode selection and cache checking.
pub fn resolve_session_refs_with_mode(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
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
                let cache = cache.as_engine();
                cas_session_core::resolve::ModeCacheEntry {
                    key: cache.key.clone(),
                    expr: cache.expr,
                    requires: cache.requires.clone(),
                }
            }),
        })
    };
    let mut same_requirement =
        |lhs: &cas_engine::diagnostics::RequiredItem,
         rhs: &cas_engine::diagnostics::RequiredItem| { lhs.cond == rhs.cond };
    let mut mark_session_propagated = |item: &mut cas_engine::diagnostics::RequiredItem| {
        item.merge_origin(cas_engine::diagnostics::RequireOrigin::SessionPropagated);
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
    let cache_key = SimplifyCacheKey::from_context(domain_mode);
    let resolved =
        resolve_session_refs_with_mode(ctx, expr, store, RefMode::PreferSimplified, &cache_key)?;

    let mut inherited = cas_engine::diagnostics::Diagnostics::new();
    for item in resolved.requires {
        inherited.push_required(
            item.cond,
            cas_engine::diagnostics::RequireOrigin::SessionPropagated,
        );
    }

    let fully_resolved = substitute(ctx, env, resolved.expr);
    Ok((
        fully_resolved,
        inherited,
        map_cache_hit_traces(resolved.cache_hits),
    ))
}

/// Resolve session references (`#N`) and environment bindings from `SessionState`.
pub fn resolve_all_from_state(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    state: &SessionState,
) -> Result<cas_ast::ExprId, ResolveError> {
    state.resolve_state_refs(ctx, expr)
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
    state.resolve_state_refs_with_diagnostics(ctx, expr)
}
