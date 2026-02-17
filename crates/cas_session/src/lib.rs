//! Session-related components extracted from `cas_engine`.

pub mod env;
mod snapshot;
mod state;

pub type SimplifyCacheKey =
    cas_session_core::cache::SimplifyCacheKey<cas_engine::domain::DomainMode>;
pub type SimplifiedCache = cas_session_core::cache::SimplifiedCache<
    cas_engine::domain::DomainMode,
    cas_engine::diagnostics::RequiredItem,
    cas_engine::step::Step,
>;
pub type CacheHitTrace =
    cas_session_core::cache::CacheHitTrace<cas_engine::diagnostics::RequiredItem>;
pub type ResolvedExpr =
    cas_session_core::cache::ResolvedExpr<cas_engine::diagnostics::RequiredItem>;
pub type Entry =
    cas_session_core::store::Entry<cas_engine::diagnostics::Diagnostics, SimplifiedCache>;
pub type SessionStore =
    cas_session_core::store::SessionStore<cas_engine::diagnostics::Diagnostics, SimplifiedCache>;
pub use cas_session_core::types::{CacheConfig, EntryId, EntryKind, RefMode, ResolveError};
pub use env::{is_reserved, substitute, substitute_with_shadow, Environment};
pub use snapshot::{SessionSnapshot, SnapshotError};
pub use state::SessionState;
use std::collections::HashMap;

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
    use std::collections::HashSet;

    let mut memo: HashMap<cas_ast::ExprId, cas_ast::ExprId> = HashMap::new();
    let mut visiting: Vec<EntryId> = Vec::new();
    let mut requires: Vec<cas_engine::diagnostics::RequiredItem> = Vec::new();
    let mut used_cache = false;
    let mut ref_chain: smallvec::SmallVec<[EntryId; 4]> = smallvec::SmallVec::new();
    let mut seen_hits: HashSet<EntryId> = HashSet::new();
    let mut cache_hits: Vec<CacheHitTrace> = Vec::new();

    let resolved = resolve_with_mode_recursive(
        ctx,
        expr,
        store,
        mode,
        cache_key,
        &mut memo,
        &mut visiting,
        &mut requires,
        &mut used_cache,
        &mut ref_chain,
        &mut seen_hits,
        &mut cache_hits,
    )?;

    Ok(ResolvedExpr {
        expr: resolved,
        requires,
        used_cache,
        ref_chain,
        cache_hits,
    })
}

#[allow(clippy::too_many_arguments)]
fn resolve_with_mode_recursive(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    store: &SessionStore,
    mode: RefMode,
    cache_key: &SimplifyCacheKey,
    memo: &mut HashMap<cas_ast::ExprId, cas_ast::ExprId>,
    visiting: &mut Vec<EntryId>,
    requires: &mut Vec<cas_engine::diagnostics::RequiredItem>,
    used_cache: &mut bool,
    ref_chain: &mut smallvec::SmallVec<[EntryId; 4]>,
    seen_hits: &mut std::collections::HashSet<EntryId>,
    cache_hits: &mut Vec<CacheHitTrace>,
) -> Result<cas_ast::ExprId, ResolveError> {
    if let Some(&cached) = memo.get(&expr) {
        return Ok(cached);
    }

    let result =
        cas_session_core::resolve::rewrite_session_refs(ctx, expr, &mut |ctx, ref_expr_id, id| {
            resolve_entry_with_mode(
                ctx,
                ref_expr_id,
                id,
                store,
                mode,
                cache_key,
                memo,
                visiting,
                requires,
                used_cache,
                ref_chain,
                seen_hits,
                cache_hits,
            )
        })?;

    memo.insert(expr, result);
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn resolve_entry_with_mode(
    ctx: &mut cas_ast::Context,
    ref_expr_id: cas_ast::ExprId,
    id: EntryId,
    store: &SessionStore,
    mode: RefMode,
    cache_key: &SimplifyCacheKey,
    memo: &mut HashMap<cas_ast::ExprId, cas_ast::ExprId>,
    visiting: &mut Vec<EntryId>,
    requires: &mut Vec<cas_engine::diagnostics::RequiredItem>,
    used_cache: &mut bool,
    ref_chain: &mut smallvec::SmallVec<[EntryId; 4]>,
    seen_hits: &mut std::collections::HashSet<EntryId>,
    cache_hits: &mut Vec<CacheHitTrace>,
) -> Result<cas_ast::ExprId, ResolveError> {
    use cas_ast::Expr;

    if visiting.contains(&id) {
        return Err(ResolveError::CircularReference(id));
    }

    let entry = store.get(id).ok_or(ResolveError::NotFound(id))?;
    ref_chain.push(id);

    if mode == RefMode::PreferSimplified {
        if let Some(cache) = &entry.simplified {
            if cache.key.is_compatible(cache_key) {
                *used_cache = true;
                requires.extend(cache.requires.iter().cloned());

                if seen_hits.insert(id) {
                    cache_hits.push(CacheHitTrace {
                        entry_id: id,
                        before_ref_expr: ref_expr_id,
                        after_expr: cache.expr,
                        requires: cache.requires.clone(),
                    });
                }

                return Ok(cache.expr);
            }
        }
    }

    visiting.push(id);

    for item in &entry.diagnostics.requires {
        if !requires.iter().any(|r| r.cond == item.cond) {
            let mut new_item = item.clone();
            new_item.merge_origin(cas_engine::diagnostics::RequireOrigin::SessionPropagated);
            requires.push(new_item);
        }
    }

    let resolved = match &entry.kind {
        EntryKind::Expr(stored_expr) => resolve_with_mode_recursive(
            ctx,
            *stored_expr,
            store,
            mode,
            cache_key,
            memo,
            visiting,
            requires,
            used_cache,
            ref_chain,
            seen_hits,
            cache_hits,
        )?,
        EntryKind::Eq { lhs, rhs } => {
            let resolved_lhs = resolve_with_mode_recursive(
                ctx, *lhs, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            let resolved_rhs = resolve_with_mode_recursive(
                ctx, *rhs, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            ctx.add(Expr::Sub(resolved_lhs, resolved_rhs))
        }
    };

    visiting.pop();
    Ok(resolved)
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
    Ok((fully_resolved, inherited, resolved.cache_hits))
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
