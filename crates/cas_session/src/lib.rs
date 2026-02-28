//! Session-related components extracted from `cas_engine`.

use cas_engine::{Diagnostics, RequireOrigin, RequiredItem};

mod cache;
pub mod env;
mod snapshot;
mod state;

pub use cache::{CacheDomainMode, SimplifiedCache, SimplifyCacheKey};
pub type CacheHitEntryId = u64;

pub type ResolvedExpr = cas_session_core::cache::ResolvedExpr<RequiredItem>;

pub type Entry = cas_session_core::store::Entry<Diagnostics, SimplifiedCache>;
pub type SessionStore = cas_session_core::store::SessionStore<Diagnostics, SimplifiedCache>;
pub use cas_session_core::types::{CacheConfig, EntryId, EntryKind, RefMode, ResolveError};
pub use env::{is_reserved, substitute, substitute_with_shadow, Environment};
pub use snapshot::SnapshotError;
pub use state::SessionState;

fn mode_entry_from_store_entry(
    entry: &Entry,
) -> cas_session_core::resolve::ModeEntry<SimplifyCacheKey, RequiredItem> {
    cas_session_core::resolve::ModeEntry {
        kind: entry.kind.clone(),
        requires: entry.diagnostics.requires.clone(),
        cache: entry
            .simplified
            .as_ref()
            .map(|cache| cas_session_core::resolve::ModeCacheEntry {
                key: cache.key.clone(),
                expr: cache.expr,
                requires: cache.requires.clone(),
            }),
    }
}

fn same_requirement(lhs: &RequiredItem, rhs: &RequiredItem) -> bool {
    lhs.cond == rhs.cond
}

fn mark_session_propagated(item: &mut RequiredItem) {
    item.merge_origin(RequireOrigin::SessionPropagated);
}

fn mode_resolve_config<'a>(
    mode: RefMode,
    cache_key: &'a SimplifyCacheKey,
    env: &'a Environment,
) -> cas_session_core::resolve::ModeResolveConfig<'a, SimplifyCacheKey> {
    cas_session_core::resolve::ModeResolveConfig {
        mode,
        cache_key,
        env,
    }
}

fn push_session_propagated_requirement(diagnostics: &mut Diagnostics, item: RequiredItem) {
    diagnostics.push_required(item.cond, RequireOrigin::SessionPropagated);
}

fn with_mode_resolution_plumbing<T, F>(store: &SessionStore, run: F) -> T
where
    F: FnOnce(
        &mut dyn FnMut(
            EntryId,
        ) -> Option<
            cas_session_core::resolve::ModeEntry<SimplifyCacheKey, RequiredItem>,
        >,
        &mut dyn FnMut(&RequiredItem, &RequiredItem) -> bool,
        &mut dyn FnMut(&mut RequiredItem),
    ) -> T,
{
    let mut lookup = |id: EntryId| store.get(id).map(mode_entry_from_store_entry);
    let mut same = same_requirement;
    let mut mark = mark_session_propagated;
    run(&mut lookup, &mut same, &mut mark)
}

pub(crate) fn simplify_cache_steps_len(cache: &SimplifiedCache) -> usize {
    cache.steps.as_ref().map(|s| s.len()).unwrap_or(0)
}

pub(crate) fn apply_simplified_light_cache(
    mut cache: SimplifiedCache,
    light_cache_threshold: Option<usize>,
) -> SimplifiedCache {
    if let Some(threshold) = light_cache_threshold {
        if simplify_cache_steps_len(&cache) > threshold {
            cache.steps = None;
        }
    }
    cache
}

pub(crate) fn session_store_with_cache_config(cache_config: CacheConfig) -> SessionStore {
    SessionStore::with_cache_config_and_policy(
        cache_config,
        simplify_cache_steps_len,
        apply_simplified_light_cache,
    )
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
) -> Result<ResolvedExpr, ResolveError> {
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
) -> Result<ResolvedExpr, ResolveError> {
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
