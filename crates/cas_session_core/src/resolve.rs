use std::collections::{HashMap, HashSet};

use cas_ast::{Context, Expr, ExprId};

use crate::cache::{CacheHitTrace, ResolvedExpr};
use crate::types::{EntryId, EntryKind, ResolveError};

/// Return the first encountered `Expr::SessionRef` id in a tree.
pub fn first_session_ref(ctx: &Context, root: ExprId) -> Option<EntryId> {
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::SessionRef(ref_id) => return Some(*ref_id),
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) => {}
        }
    }
    None
}

/// Parse legacy session reference names like `#123`.
pub fn parse_legacy_session_ref(name: &str) -> Option<EntryId> {
    if !name.starts_with('#') || name.len() <= 1 {
        return None;
    }
    if !name[1..].chars().all(char::is_numeric) {
        return None;
    }
    name[1..].parse::<EntryId>().ok()
}

fn root_session_ref_id(ctx: &Context, expr: ExprId) -> Option<EntryId> {
    match ctx.get(expr) {
        Expr::SessionRef(id) => Some(*id),
        Expr::Variable(sym_id) => parse_legacy_session_ref(ctx.sym_name(*sym_id)),
        _ => None,
    }
}

/// Rewrite session references (`Expr::SessionRef` and legacy `Variable("#N")`)
/// in an expression tree using the provided resolver callback.
///
/// The callback receives:
/// - the mutable context,
/// - the `ExprId` of the reference node in the current AST,
/// - the parsed entry id.
pub fn rewrite_session_refs<E, F>(
    ctx: &mut Context,
    expr: ExprId,
    resolver: &mut F,
) -> Result<ExprId, E>
where
    F: FnMut(&mut Context, ExprId, EntryId) -> Result<ExprId, E>,
{
    match ctx.get(expr) {
        Expr::SessionRef(id) => resolver(ctx, expr, *id),
        Expr::Variable(sym_id) => match parse_legacy_session_ref(ctx.sym_name(*sym_id)) {
            Some(id) => resolver(ctx, expr, id),
            None => Ok(expr),
        },

        Expr::Add(l, r) => rewrite_binary(ctx, expr, *l, *r, Expr::Add, resolver),
        Expr::Sub(l, r) => rewrite_binary(ctx, expr, *l, *r, Expr::Sub, resolver),
        Expr::Mul(l, r) => rewrite_binary(ctx, expr, *l, *r, Expr::Mul, resolver),
        Expr::Div(l, r) => rewrite_binary(ctx, expr, *l, *r, Expr::Div, resolver),
        Expr::Pow(l, r) => rewrite_binary(ctx, expr, *l, *r, Expr::Pow, resolver),

        Expr::Neg(inner) => {
            let inner = *inner;
            let new_inner = rewrite_session_refs(ctx, inner, resolver)?;
            if new_inner == inner {
                Ok(expr)
            } else {
                Ok(ctx.add(Expr::Neg(new_inner)))
            }
        }

        Expr::Function(name, args) => {
            let name = *name;
            let args = args.clone();
            let mut changed = false;
            let mut new_args = Vec::with_capacity(args.len());
            for arg in args {
                let new_arg = rewrite_session_refs(ctx, arg, resolver)?;
                if new_arg != arg {
                    changed = true;
                }
                new_args.push(new_arg);
            }
            if changed {
                Ok(ctx.add(Expr::Function(name, new_args)))
            } else {
                Ok(expr)
            }
        }

        Expr::Matrix { rows, cols, data } => {
            let rows = *rows;
            let cols = *cols;
            let data = data.clone();
            let mut changed = false;
            let mut new_data = Vec::with_capacity(data.len());
            for elem in data {
                let new_elem = rewrite_session_refs(ctx, elem, resolver)?;
                if new_elem != elem {
                    changed = true;
                }
                new_data.push(new_elem);
            }
            if changed {
                Ok(ctx.add(Expr::Matrix {
                    rows,
                    cols,
                    data: new_data,
                }))
            } else {
                Ok(expr)
            }
        }

        Expr::Hold(inner) => {
            let inner = *inner;
            let new_inner = rewrite_session_refs(ctx, inner, resolver)?;
            if new_inner == inner {
                Ok(expr)
            } else {
                Ok(ctx.add(Expr::Hold(new_inner)))
            }
        }

        Expr::Number(_) | Expr::Constant(_) => Ok(expr),
    }
}

/// Resolve all session references in an expression using a store lookup callback.
///
/// The callback must return the `EntryKind` for each entry id.
/// Returns `ResolveError::NotFound` when an entry does not exist and
/// `ResolveError::CircularReference` when a cycle is detected.
pub fn resolve_session_refs_with_lookup<F>(
    ctx: &mut Context,
    expr: ExprId,
    lookup: &mut F,
) -> Result<ExprId, ResolveError>
where
    F: FnMut(EntryId) -> Option<EntryKind>,
{
    let mut on_visit = |_id: EntryId| {};
    resolve_session_refs_with_lookup_on_visit(ctx, expr, lookup, &mut on_visit)
}

/// Resolve refs with lookup and fold visit-side metadata into an accumulator.
///
/// This keeps "what to accumulate on each visited entry id" outside of the
/// resolver recursion plumbing.
pub fn resolve_session_refs_with_lookup_accumulator<F, Acc, OnVisit>(
    ctx: &mut Context,
    expr: ExprId,
    lookup: &mut F,
    mut acc: Acc,
    mut on_visit_acc: OnVisit,
) -> Result<(ExprId, Acc), ResolveError>
where
    F: FnMut(EntryId) -> Option<EntryKind>,
    OnVisit: FnMut(&mut Acc, EntryId),
{
    let mut on_visit = |id: EntryId| on_visit_acc(&mut acc, id);
    let resolved = resolve_session_refs_with_lookup_on_visit(ctx, expr, lookup, &mut on_visit)?;
    Ok((resolved, acc))
}

/// Resolve all session references and then apply environment substitution.
pub fn resolve_all_with_lookup_and_env<F>(
    ctx: &mut Context,
    expr: ExprId,
    lookup: &mut F,
    env: &crate::env::Environment,
) -> Result<ExprId, ResolveError>
where
    F: FnMut(EntryId) -> Option<EntryKind>,
{
    let expr_with_refs = resolve_session_refs_with_lookup(ctx, expr, lookup)?;
    Ok(crate::env::substitute(ctx, env, expr_with_refs))
}

/// Return only entry ids from cache-hit traces, preserving traversal order.
pub fn cache_hit_entry_ids<RequiredItem>(hits: &[CacheHitTrace<RequiredItem>]) -> Vec<EntryId> {
    hits.iter().map(|h| h.entry_id).collect()
}

/// Build inherited diagnostics from resolved requirement items.
pub fn inherited_diagnostics_from_requires<RequiredItem, Diagnostics, FPushRequired>(
    requires: Vec<RequiredItem>,
    mut diagnostics: Diagnostics,
    mut push_required: FPushRequired,
) -> Diagnostics
where
    FPushRequired: FnMut(&mut Diagnostics, RequiredItem),
{
    for item in requires {
        push_required(&mut diagnostics, item);
    }
    diagnostics
}

/// Resolve all session references with an additional visit hook.
///
/// `on_visit` is called once per entry traversal (before descending into its
/// stored expression/equation), and can be used to accumulate metadata.
pub fn resolve_session_refs_with_lookup_on_visit<F, V>(
    ctx: &mut Context,
    expr: ExprId,
    lookup: &mut F,
    on_visit: &mut V,
) -> Result<ExprId, ResolveError>
where
    F: FnMut(EntryId) -> Option<EntryKind>,
    V: FnMut(EntryId),
{
    let mut cache: HashMap<EntryId, ExprId> = HashMap::new();
    let mut visiting: HashSet<EntryId> = HashSet::new();
    if let Some(id) = root_session_ref_id(ctx, expr) {
        return resolve_session_id_with_lookup(
            ctx,
            id,
            lookup,
            on_visit,
            &mut cache,
            &mut visiting,
        );
    }
    resolve_with_lookup_recursive(ctx, expr, lookup, on_visit, &mut cache, &mut visiting)
}

/// Cache payload view used by mode-based resolution.
#[derive(Debug, Clone)]
pub struct ModeCacheEntry<CacheKey, RequiredItem> {
    pub key: CacheKey,
    pub expr: ExprId,
    pub requires: Vec<RequiredItem>,
}

/// Entry view used by mode-based resolution.
#[derive(Debug, Clone)]
pub struct ModeEntry<CacheKey, RequiredItem> {
    pub kind: EntryKind,
    pub requires: Vec<RequiredItem>,
    pub cache: Option<ModeCacheEntry<CacheKey, RequiredItem>>,
}

/// Config for mode-based resolution + env substitution.
#[derive(Debug, Clone, Copy)]
pub struct ModeResolveConfig<'a, CacheKey> {
    pub mode: crate::types::RefMode,
    pub cache_key: &'a CacheKey,
    pub env: &'a crate::env::Environment,
}

/// Resolve session refs with mode selection (prefer cache vs raw entry).
///
/// The caller provides a lookup callback that returns cloned entry data and
/// two callbacks to deduplicate requirements and mark propagated origin.
pub fn resolve_session_refs_with_mode_lookup<CacheKey, RequiredItem, Lookup, SameReq, MarkReq>(
    ctx: &mut Context,
    expr: ExprId,
    mode: crate::types::RefMode,
    cache_key: &CacheKey,
    lookup: &mut Lookup,
    same_requirement: &mut SameReq,
    mark_session_propagated: &mut MarkReq,
) -> Result<ResolvedExpr<RequiredItem>, ResolveError>
where
    CacheKey: PartialEq,
    RequiredItem: Clone,
    Lookup: FnMut(EntryId) -> Option<ModeEntry<CacheKey, RequiredItem>>,
    SameReq: FnMut(&RequiredItem, &RequiredItem) -> bool,
    MarkReq: FnMut(&mut RequiredItem),
{
    let mut memo: HashMap<ExprId, ExprId> = HashMap::new();
    let mut visiting: Vec<EntryId> = Vec::new();
    let mut requires: Vec<RequiredItem> = Vec::new();
    let mut used_cache = false;
    let mut ref_chain: smallvec::SmallVec<[EntryId; 4]> = smallvec::SmallVec::new();
    let mut seen_hits: HashSet<EntryId> = HashSet::new();
    let mut cache_hits: Vec<CacheHitTrace<RequiredItem>> = Vec::new();

    let resolved = resolve_with_mode_recursive(
        ctx,
        expr,
        mode,
        cache_key,
        lookup,
        same_requirement,
        mark_session_propagated,
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

/// Resolve session refs with mode selection and then apply environment substitution.
pub fn resolve_all_with_mode_lookup_and_env<CacheKey, RequiredItem, Lookup, SameReq, MarkReq>(
    ctx: &mut Context,
    expr: ExprId,
    config: ModeResolveConfig<'_, CacheKey>,
    lookup: &mut Lookup,
    same_requirement: &mut SameReq,
    mark_session_propagated: &mut MarkReq,
) -> Result<ResolvedExpr<RequiredItem>, ResolveError>
where
    CacheKey: PartialEq,
    RequiredItem: Clone,
    Lookup: FnMut(EntryId) -> Option<ModeEntry<CacheKey, RequiredItem>>,
    SameReq: FnMut(&RequiredItem, &RequiredItem) -> bool,
    MarkReq: FnMut(&mut RequiredItem),
{
    let mut resolved = resolve_session_refs_with_mode_lookup(
        ctx,
        expr,
        config.mode,
        config.cache_key,
        lookup,
        same_requirement,
        mark_session_propagated,
    )?;
    resolved.expr = crate::env::substitute(ctx, config.env, resolved.expr);
    Ok(resolved)
}

/// Resolve refs with mode + env and immediately build inherited diagnostics
/// and cache-hit entry ids.
#[allow(clippy::too_many_arguments)]
pub fn resolve_mode_with_env_and_diagnostics<
    CacheKey,
    RequiredItem,
    Diagnostics,
    Lookup,
    SameReq,
    MarkReq,
    FPushRequired,
>(
    ctx: &mut Context,
    expr: ExprId,
    config: ModeResolveConfig<'_, CacheKey>,
    lookup: &mut Lookup,
    same_requirement: &mut SameReq,
    mark_session_propagated: &mut MarkReq,
    diagnostics: Diagnostics,
    push_required: FPushRequired,
) -> Result<(ExprId, Diagnostics, Vec<EntryId>), ResolveError>
where
    CacheKey: PartialEq,
    RequiredItem: Clone,
    Lookup: FnMut(EntryId) -> Option<ModeEntry<CacheKey, RequiredItem>>,
    SameReq: FnMut(&RequiredItem, &RequiredItem) -> bool,
    MarkReq: FnMut(&mut RequiredItem),
    FPushRequired: FnMut(&mut Diagnostics, RequiredItem),
{
    let resolved = resolve_all_with_mode_lookup_and_env(
        ctx,
        expr,
        config,
        lookup,
        same_requirement,
        mark_session_propagated,
    )?;
    let diagnostics =
        inherited_diagnostics_from_requires(resolved.requires, diagnostics, push_required);
    Ok((
        resolved.expr,
        diagnostics,
        cache_hit_entry_ids(&resolved.cache_hits),
    ))
}

#[allow(clippy::too_many_arguments)]
fn resolve_with_mode_recursive<CacheKey, RequiredItem, Lookup, SameReq, MarkReq>(
    ctx: &mut Context,
    expr: ExprId,
    mode: crate::types::RefMode,
    cache_key: &CacheKey,
    lookup: &mut Lookup,
    same_requirement: &mut SameReq,
    mark_session_propagated: &mut MarkReq,
    memo: &mut HashMap<ExprId, ExprId>,
    visiting: &mut Vec<EntryId>,
    requires: &mut Vec<RequiredItem>,
    used_cache: &mut bool,
    ref_chain: &mut smallvec::SmallVec<[EntryId; 4]>,
    seen_hits: &mut HashSet<EntryId>,
    cache_hits: &mut Vec<CacheHitTrace<RequiredItem>>,
) -> Result<ExprId, ResolveError>
where
    CacheKey: PartialEq,
    RequiredItem: Clone,
    Lookup: FnMut(EntryId) -> Option<ModeEntry<CacheKey, RequiredItem>>,
    SameReq: FnMut(&RequiredItem, &RequiredItem) -> bool,
    MarkReq: FnMut(&mut RequiredItem),
{
    if let Some(&cached) = memo.get(&expr) {
        return Ok(cached);
    }

    if let Some(id) = root_session_ref_id(ctx, expr) {
        let result = resolve_mode_entry(
            ctx,
            expr,
            id,
            mode,
            cache_key,
            lookup,
            same_requirement,
            mark_session_propagated,
            memo,
            visiting,
            requires,
            used_cache,
            ref_chain,
            seen_hits,
            cache_hits,
        )?;
        memo.insert(expr, result);
        return Ok(result);
    }

    let result = rewrite_session_refs(ctx, expr, &mut |ctx, ref_expr_id, id| {
        resolve_mode_entry(
            ctx,
            ref_expr_id,
            id,
            mode,
            cache_key,
            lookup,
            same_requirement,
            mark_session_propagated,
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
fn resolve_mode_entry<CacheKey, RequiredItem, Lookup, SameReq, MarkReq>(
    ctx: &mut Context,
    ref_expr_id: ExprId,
    id: EntryId,
    mode: crate::types::RefMode,
    cache_key: &CacheKey,
    lookup: &mut Lookup,
    same_requirement: &mut SameReq,
    mark_session_propagated: &mut MarkReq,
    memo: &mut HashMap<ExprId, ExprId>,
    visiting: &mut Vec<EntryId>,
    requires: &mut Vec<RequiredItem>,
    used_cache: &mut bool,
    ref_chain: &mut smallvec::SmallVec<[EntryId; 4]>,
    seen_hits: &mut HashSet<EntryId>,
    cache_hits: &mut Vec<CacheHitTrace<RequiredItem>>,
) -> Result<ExprId, ResolveError>
where
    CacheKey: PartialEq,
    RequiredItem: Clone,
    Lookup: FnMut(EntryId) -> Option<ModeEntry<CacheKey, RequiredItem>>,
    SameReq: FnMut(&RequiredItem, &RequiredItem) -> bool,
    MarkReq: FnMut(&mut RequiredItem),
{
    if visiting.contains(&id) {
        return Err(ResolveError::CircularReference(id));
    }

    let entry = lookup(id).ok_or(ResolveError::NotFound(id))?;
    ref_chain.push(id);

    if mode == crate::types::RefMode::PreferSimplified {
        if let Some(cache) = &entry.cache {
            if cache.key == *cache_key {
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

    for item in &entry.requires {
        if !requires.iter().any(|r| same_requirement(r, item)) {
            let mut new_item = item.clone();
            mark_session_propagated(&mut new_item);
            requires.push(new_item);
        }
    }

    let resolved = match entry.kind {
        EntryKind::Expr(stored_expr) => resolve_with_mode_recursive(
            ctx,
            stored_expr,
            mode,
            cache_key,
            lookup,
            same_requirement,
            mark_session_propagated,
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
                ctx,
                lhs,
                mode,
                cache_key,
                lookup,
                same_requirement,
                mark_session_propagated,
                memo,
                visiting,
                requires,
                used_cache,
                ref_chain,
                seen_hits,
                cache_hits,
            )?;
            let resolved_rhs = resolve_with_mode_recursive(
                ctx,
                rhs,
                mode,
                cache_key,
                lookup,
                same_requirement,
                mark_session_propagated,
                memo,
                visiting,
                requires,
                used_cache,
                ref_chain,
                seen_hits,
                cache_hits,
            )?;
            ctx.add(Expr::Sub(resolved_lhs, resolved_rhs))
        }
    };

    visiting.pop();
    Ok(resolved)
}

fn resolve_with_lookup_recursive<F, V>(
    ctx: &mut Context,
    expr: ExprId,
    lookup: &mut F,
    on_visit: &mut V,
    cache: &mut HashMap<EntryId, ExprId>,
    visiting: &mut HashSet<EntryId>,
) -> Result<ExprId, ResolveError>
where
    F: FnMut(EntryId) -> Option<EntryKind>,
    V: FnMut(EntryId),
{
    rewrite_session_refs(ctx, expr, &mut |ctx, _ref_expr_id, id| {
        resolve_session_id_with_lookup(ctx, id, lookup, on_visit, cache, visiting)
    })
}

fn resolve_session_id_with_lookup<F, V>(
    ctx: &mut Context,
    id: EntryId,
    lookup: &mut F,
    on_visit: &mut V,
    cache: &mut HashMap<EntryId, ExprId>,
    visiting: &mut HashSet<EntryId>,
) -> Result<ExprId, ResolveError>
where
    F: FnMut(EntryId) -> Option<EntryKind>,
    V: FnMut(EntryId),
{
    // Memoized hit
    if let Some(&resolved) = cache.get(&id) {
        return Ok(resolved);
    }

    // Cycle detection
    if visiting.contains(&id) {
        return Err(ResolveError::CircularReference(id));
    }

    // Fetch entry
    let kind = lookup(id).ok_or(ResolveError::NotFound(id))?;
    on_visit(id);

    // Resolve entry recursively
    visiting.insert(id);
    let substitution = match kind {
        EntryKind::Expr(stored_expr) => {
            resolve_with_lookup_recursive(ctx, stored_expr, lookup, on_visit, cache, visiting)?
        }
        EntryKind::Eq { lhs, rhs } => {
            let resolved_lhs =
                resolve_with_lookup_recursive(ctx, lhs, lookup, on_visit, cache, visiting)?;
            let resolved_rhs =
                resolve_with_lookup_recursive(ctx, rhs, lookup, on_visit, cache, visiting)?;
            ctx.add(Expr::Sub(resolved_lhs, resolved_rhs))
        }
    };
    visiting.remove(&id);

    cache.insert(id, substitution);
    Ok(substitution)
}

fn rewrite_binary<E, F, Ctor>(
    ctx: &mut Context,
    expr: ExprId,
    left: ExprId,
    right: ExprId,
    ctor: Ctor,
    resolver: &mut F,
) -> Result<ExprId, E>
where
    F: FnMut(&mut Context, ExprId, EntryId) -> Result<ExprId, E>,
    Ctor: Fn(ExprId, ExprId) -> Expr,
{
    let new_left = rewrite_session_refs(ctx, left, resolver)?;
    let new_right = rewrite_session_refs(ctx, right, resolver)?;
    if new_left == left && new_right == right {
        Ok(expr)
    } else {
        Ok(ctx.add(ctor(new_left, new_right)))
    }
}

#[cfg(test)]
mod tests;
