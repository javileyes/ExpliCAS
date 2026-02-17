//! Session reference resolution.

use cas_ast::ExprId;
use cas_session_core::resolve::{
    parse_legacy_session_ref, resolve_all_with_lookup_and_env, resolve_session_refs_with_lookup,
    resolve_session_refs_with_lookup_on_visit,
};
pub use cas_session_core::types::ResolveError;
use std::collections::HashMap;

use super::store::*;

/// Resolve all references in an expression:
/// 1. Resolve session references (`#id`) -> ExprId
/// 2. Substitute environment variables (`x=5`) -> ExprId
pub fn resolve_all(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    store: &SessionStore,
    env: &crate::env::Environment,
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
    env: &crate::env::Environment,
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
    use std::collections::HashSet;

    let mut memo: HashMap<ExprId, ExprId> = HashMap::new();
    let mut visiting: Vec<EntryId> = Vec::new();
    let mut requires: Vec<crate::diagnostics::RequiredItem> = Vec::new();
    let mut used_cache = false;
    let mut ref_chain: smallvec::SmallVec<[EntryId; 4]> = smallvec::SmallVec::new();
    // V2.15.36: Track cache hits for synthetic timeline step
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

/// Internal recursive resolver with cache checking
#[allow(clippy::too_many_arguments)]
fn resolve_with_mode_recursive(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    store: &SessionStore,
    mode: RefMode,
    cache_key: &SimplifyCacheKey,
    memo: &mut HashMap<ExprId, ExprId>,
    visiting: &mut Vec<EntryId>,
    requires: &mut Vec<crate::diagnostics::RequiredItem>,
    used_cache: &mut bool,
    ref_chain: &mut smallvec::SmallVec<[EntryId; 4]>,
    seen_hits: &mut std::collections::HashSet<EntryId>,
    cache_hits: &mut Vec<CacheHitTrace>,
) -> Result<ExprId, ResolveError> {
    use cas_ast::Expr;

    // Check memo first
    if let Some(&cached) = memo.get(&expr) {
        return Ok(cached);
    }

    let node = ctx.get(expr).clone();

    let result = match node {
        Expr::SessionRef(id) => resolve_entry_with_mode(
            ctx, expr, id, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
            seen_hits, cache_hits,
        )?,

        // Handle Variable that might be a #N reference (legacy parsing)
        Expr::Variable(sym_id) => {
            if let Some(id) = parse_legacy_session_ref(ctx.sym_name(sym_id)) {
                resolve_entry_with_mode(
                    ctx, expr, id, store, mode, cache_key, memo, visiting, requires, used_cache,
                    ref_chain, seen_hits, cache_hits,
                )?
            } else {
                expr
            }
        }

        // Binary operators - recurse into children
        Expr::Add(l, r) => {
            let new_l = resolve_with_mode_recursive(
                ctx, l, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            let new_r = resolve_with_mode_recursive(
                ctx, r, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            if new_l == l && new_r == r {
                expr
            } else {
                ctx.add(Expr::Add(new_l, new_r))
            }
        }
        Expr::Sub(l, r) => {
            let new_l = resolve_with_mode_recursive(
                ctx, l, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            let new_r = resolve_with_mode_recursive(
                ctx, r, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            if new_l == l && new_r == r {
                expr
            } else {
                ctx.add(Expr::Sub(new_l, new_r))
            }
        }
        Expr::Mul(l, r) => {
            let new_l = resolve_with_mode_recursive(
                ctx, l, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            let new_r = resolve_with_mode_recursive(
                ctx, r, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            if new_l == l && new_r == r {
                expr
            } else {
                ctx.add(Expr::Mul(new_l, new_r))
            }
        }
        Expr::Div(l, r) => {
            let new_l = resolve_with_mode_recursive(
                ctx, l, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            let new_r = resolve_with_mode_recursive(
                ctx, r, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            if new_l == l && new_r == r {
                expr
            } else {
                ctx.add(Expr::Div(new_l, new_r))
            }
        }
        Expr::Pow(b, e) => {
            let new_b = resolve_with_mode_recursive(
                ctx, b, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            let new_e = resolve_with_mode_recursive(
                ctx, e, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            if new_b == b && new_e == e {
                expr
            } else {
                ctx.add(Expr::Pow(new_b, new_e))
            }
        }

        // Unary
        Expr::Neg(e) => {
            let new_e = resolve_with_mode_recursive(
                ctx, e, store, mode, cache_key, memo, visiting, requires, used_cache, ref_chain,
                seen_hits, cache_hits,
            )?;
            if new_e == e {
                expr
            } else {
                ctx.add(Expr::Neg(new_e))
            }
        }

        // Function
        Expr::Function(name, args) => {
            let mut changed = false;
            let mut new_args = Vec::with_capacity(args.len());
            for arg in &args {
                let new_arg = resolve_with_mode_recursive(
                    ctx, *arg, store, mode, cache_key, memo, visiting, requires, used_cache,
                    ref_chain, seen_hits, cache_hits,
                )?;
                if new_arg != *arg {
                    changed = true;
                }
                new_args.push(new_arg);
            }
            if changed {
                ctx.add(Expr::Function(name, new_args))
            } else {
                expr
            }
        }

        // Matrix
        Expr::Matrix { rows, cols, data } => {
            let mut changed = false;
            let mut new_data = Vec::with_capacity(data.len());
            for elem in &data {
                let new_elem = resolve_with_mode_recursive(
                    ctx, *elem, store, mode, cache_key, memo, visiting, requires, used_cache,
                    ref_chain, seen_hits, cache_hits,
                )?;
                if new_elem != *elem {
                    changed = true;
                }
                new_data.push(new_elem);
            }
            if changed {
                ctx.add(Expr::Matrix {
                    rows,
                    cols,
                    data: new_data,
                })
            } else {
                expr
            }
        }

        // Leaf nodes
        Expr::Number(_) | Expr::Constant(_) => expr,

        // Hold: recurse into inner
        Expr::Hold(inner) => {
            let new_inner = resolve_with_mode_recursive(
                ctx, inner, store, mode, cache_key, memo, visiting, requires, used_cache,
                ref_chain, seen_hits, cache_hits,
            )?;
            if new_inner == inner {
                expr
            } else {
                ctx.add(Expr::Hold(new_inner))
            }
        }
    };

    memo.insert(expr, result);
    Ok(result)
}

/// Resolve a single entry ID using cache if available
#[allow(clippy::too_many_arguments)]
fn resolve_entry_with_mode(
    ctx: &mut cas_ast::Context,
    ref_expr_id: ExprId, // The ExprId of the #N node in AST (for cache hit trace)
    id: EntryId,
    store: &SessionStore,
    mode: RefMode,
    cache_key: &SimplifyCacheKey,
    memo: &mut HashMap<ExprId, ExprId>,
    visiting: &mut Vec<EntryId>,
    requires: &mut Vec<crate::diagnostics::RequiredItem>,
    used_cache: &mut bool,
    ref_chain: &mut smallvec::SmallVec<[EntryId; 4]>,
    seen_hits: &mut std::collections::HashSet<EntryId>,
    cache_hits: &mut Vec<CacheHitTrace>,
) -> Result<ExprId, ResolveError> {
    use cas_ast::Expr;

    // Cycle detection
    if visiting.contains(&id) {
        return Err(ResolveError::CircularReference(id));
    }

    // Get entry
    let entry = store.get(id).ok_or(ResolveError::NotFound(id))?;

    // Track reference chain
    ref_chain.push(id);

    // 1) PreferSimplified: check cache first
    if mode == RefMode::PreferSimplified {
        if let Some(cache) = &entry.simplified {
            if cache.key.is_compatible(cache_key) {
                // Cache hit! Use cached expression and accumulate requires
                *used_cache = true;
                requires.extend(cache.requires.iter().cloned());

                // V2.15.36: Record cache hit for synthetic step (dedup by entry_id)
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

    // 2) Fallback: use raw parsed expression
    visiting.push(id);

    // Inherit requires from entry's diagnostics (for SessionPropagated tracking)
    for item in &entry.diagnostics.requires {
        if !requires.iter().any(|r| r.cond == item.cond) {
            let mut new_item = item.clone();
            new_item.merge_origin(crate::diagnostics::RequireOrigin::SessionPropagated);
            requires.push(new_item);
        }
    }

    let resolved = match &entry.kind {
        EntryKind::Expr(stored_expr) => {
            // Recursively resolve (it may contain #refs too)
            resolve_with_mode_recursive(
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
            )?
        }
        EntryKind::Eq { lhs, rhs } => {
            // Equation as expression: use residue form (lhs - rhs)
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
