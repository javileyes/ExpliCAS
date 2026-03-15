//! Helpers for `DivExpandToCancelRule` extraction.
//!
//! These routines keep structural/polynomial guards in `cas_math` while the
//! engine keeps orchestration details (recursion guards and simplifier calls).

use crate::expandable_pattern_support::contains_expandable_small_depth;
use crate::expr_complexity::node_count_tree;
use crate::expr_destructure::as_div;
use crate::expr_nary::{add_terms_no_sign, build_balanced_add};
use crate::fraction_factors::{
    build_fraction_from_factor_vectors, decompose_fraction_like_factors,
};
use crate::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};
use crate::opaque_atoms::{
    collect_function_calls_with_pow_limit, extract_opaque_rational_power_atom,
    extract_opaque_reciprocal_power_base, extract_opaque_signed_rational_power_atom,
};
use crate::opaque_function_calls_support::match_shared_calls_structural;
use crate::substitute::{substitute_power_aware, SubstituteOptions};
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::Zero;
use std::cell::Cell;
use std::cmp::Ordering;

/// Intermediate state after replacing shared opaque calls with temporary vars.
#[derive(Debug, Clone)]
pub struct OpaqueSubstitutionPlan {
    pub substituted_num: ExprId,
    pub substituted_den: ExprId,
    /// Pairs of `(original_call, temp_var)`.
    pub temp_vars: Vec<(ExprId, ExprId)>,
}

#[derive(Clone)]
pub struct ExpandedPair {
    pub context: Context,
    pub expanded_num: ExprId,
    pub expanded_den: ExprId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivExpandToCancelKind {
    OpaqueSubstitution,
    ExpandedEquality,
}

#[derive(Debug, Clone, Copy)]
pub struct DivExpandToCancelRewrite {
    pub rewritten: ExprId,
    pub kind: DivExpandToCancelKind,
}

fn mk_pow_u32(ctx: &mut Context, base: ExprId, exp: u32) -> ExprId {
    if exp == 1 {
        base
    } else {
        let exp_expr = ctx.num(exp as i64);
        ctx.add(Expr::Pow(base, exp_expr))
    }
}

fn mk_pow_i32(ctx: &mut Context, base: ExprId, exp: i32) -> ExprId {
    if exp == 1 {
        base
    } else {
        let exp_expr = ctx.num(exp as i64);
        ctx.add(Expr::Pow(base, exp_expr))
    }
}

fn canonical_root_family_atom(ctx: &mut Context, root_base: ExprId, root_index: u32) -> ExprId {
    if root_index == 1 {
        return root_base;
    }

    let exp = ctx.rational(1, i64::from(root_index));
    ctx.add(Expr::Pow(root_base, exp))
}

fn extract_root_family_base(ctx: &Context, expr: ExprId) -> Option<(ExprId, u32)> {
    if let Some((base, root_index)) = extract_opaque_reciprocal_power_base(ctx, expr) {
        return Some((base, root_index));
    }

    let base = crate::root_forms::extract_square_root_base(ctx, expr)?;
    Some((base, 2))
}

fn extract_root_family_signature(ctx: &Context, expr: ExprId) -> Option<(ExprId, u32)> {
    if let Some((base, root_index)) = extract_root_family_base(ctx, expr) {
        return Some((base, root_index));
    }
    let (base, _numer, denom) = extract_opaque_signed_rational_power_atom(ctx, expr)?;
    if denom >= 2 {
        Some((base, denom))
    } else {
        None
    }
}

fn replace_root_family_with_temp(
    ctx: &mut Context,
    expr: ExprId,
    root_base: ExprId,
    root_index: u32,
    temp_var: ExprId,
) -> ExprId {
    if compare_expr(ctx, expr, root_base) == Ordering::Equal {
        return mk_pow_u32(ctx, temp_var, root_index);
    }

    if let Some((base_expr, numer, denom)) = extract_opaque_signed_rational_power_atom(ctx, expr) {
        if denom == root_index && compare_expr(ctx, base_expr, root_base) == Ordering::Equal {
            return mk_pow_i32(ctx, temp_var, numer);
        }
    }

    if let Some((base_expr, numer, denom)) = extract_opaque_rational_power_atom(ctx, expr) {
        if denom == root_index && compare_expr(ctx, base_expr, root_base) == Ordering::Equal {
            return mk_pow_u32(ctx, temp_var, numer);
        }
    }

    if let Some(rewritten) =
        try_replace_collapsed_root_add_terms_with_temp(ctx, expr, root_base, root_index, temp_var)
    {
        return rewritten;
    }

    match ctx.get(expr).clone() {
        Expr::Add(l, r) => {
            let new_l = replace_root_family_with_temp(ctx, l, root_base, root_index, temp_var);
            let new_r = replace_root_family_with_temp(ctx, r, root_base, root_index, temp_var);
            if new_l == l && new_r == r {
                expr
            } else {
                ctx.add(Expr::Add(new_l, new_r))
            }
        }
        Expr::Sub(l, r) => {
            let new_l = replace_root_family_with_temp(ctx, l, root_base, root_index, temp_var);
            let new_r = replace_root_family_with_temp(ctx, r, root_base, root_index, temp_var);
            if new_l == l && new_r == r {
                expr
            } else {
                ctx.add(Expr::Sub(new_l, new_r))
            }
        }
        Expr::Mul(l, r) => {
            let new_l = replace_root_family_with_temp(ctx, l, root_base, root_index, temp_var);
            let new_r = replace_root_family_with_temp(ctx, r, root_base, root_index, temp_var);
            if new_l == l && new_r == r {
                expr
            } else {
                ctx.add(Expr::Mul(new_l, new_r))
            }
        }
        Expr::Div(l, r) => {
            let new_l = replace_root_family_with_temp(ctx, l, root_base, root_index, temp_var);
            let new_r = replace_root_family_with_temp(ctx, r, root_base, root_index, temp_var);
            if new_l == l && new_r == r {
                expr
            } else {
                ctx.add(Expr::Div(new_l, new_r))
            }
        }
        Expr::Pow(base, exp) => {
            let new_base =
                replace_root_family_with_temp(ctx, base, root_base, root_index, temp_var);
            let new_exp = replace_root_family_with_temp(ctx, exp, root_base, root_index, temp_var);
            if new_base == base && new_exp == exp {
                expr
            } else {
                ctx.add(Expr::Pow(new_base, new_exp))
            }
        }
        Expr::Neg(inner) => {
            let new_inner =
                replace_root_family_with_temp(ctx, inner, root_base, root_index, temp_var);
            if new_inner == inner {
                expr
            } else {
                ctx.add(Expr::Neg(new_inner))
            }
        }
        Expr::Function(kind, args) => {
            let new_args: Vec<_> = args
                .iter()
                .map(|&arg| {
                    replace_root_family_with_temp(ctx, arg, root_base, root_index, temp_var)
                })
                .collect();
            if new_args == args {
                expr
            } else {
                ctx.add(Expr::Function(kind, new_args))
            }
        }
        _ => expr,
    }
}

fn try_replace_collapsed_root_add_terms_with_temp(
    ctx: &mut Context,
    expr: ExprId,
    root_base: ExprId,
    root_index: u32,
    temp_var: ExprId,
) -> Option<ExprId> {
    let expr_terms = add_terms_no_sign(ctx, expr);
    let root_terms = add_terms_no_sign(ctx, root_base);
    if expr_terms.len() < 2 || root_terms.len() < 2 {
        return None;
    }

    let mut root_numeric_sum = BigRational::from_integer(0.into());
    let mut root_non_numeric = Vec::new();
    for term in root_terms {
        match ctx.get(term) {
            Expr::Number(n) => root_numeric_sum += n.clone(),
            _ => root_non_numeric.push(term),
        }
    }
    if root_non_numeric.is_empty() {
        return None;
    }

    let mut expr_numeric_sum = BigRational::from_integer(0.into());
    let mut expr_non_numeric = Vec::new();
    let mut expr_has_numeric = false;
    for term in expr_terms {
        match ctx.get(term) {
            Expr::Number(n) => {
                expr_numeric_sum += n.clone();
                expr_has_numeric = true;
            }
            _ => expr_non_numeric.push(term),
        }
    }
    if !expr_has_numeric {
        return None;
    }

    let mut remaining_non_numeric = expr_non_numeric;
    for required in &root_non_numeric {
        let matched_idx = remaining_non_numeric
            .iter()
            .position(|candidate| compare_expr(ctx, *candidate, *required) == Ordering::Equal)?;
        remaining_non_numeric.remove(matched_idx);
    }

    let constant_delta = expr_numeric_sum - root_numeric_sum;
    let temp_pow = mk_pow_u32(ctx, temp_var, root_index);
    let collapsed_root = if constant_delta.is_zero() {
        temp_pow
    } else {
        let delta_expr = ctx.add(Expr::Number(constant_delta));
        build_balanced_add(ctx, &[temp_pow, delta_expr])
    };

    let mut rebuilt_terms = Vec::with_capacity(remaining_non_numeric.len() + 1);
    rebuilt_terms.push(collapsed_root);
    for term in remaining_non_numeric {
        rebuilt_terms.push(replace_root_family_with_temp(
            ctx, term, root_base, root_index, temp_var,
        ));
    }
    Some(build_balanced_add(ctx, &rebuilt_terms))
}

fn is_leaf_like(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_)
    )
}

fn as_fraction_like_num_den(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    if let Some(pair) = as_div(ctx, expr) {
        return Some(pair);
    }
    let (num_factors, den_factors) = decompose_fraction_like_factors(ctx, expr)?;
    let rebuilt = build_fraction_from_factor_vectors(ctx, &num_factors, &den_factors);
    as_div(ctx, rebuilt)
}

/// True when both numerator and denominator are simple leaf-like atoms.
pub fn both_sides_leaf_like(ctx: &Context, num: ExprId, den: ExprId) -> bool {
    is_leaf_like(ctx, num) && is_leaf_like(ctx, den)
}

/// Guard used by expand-to-cancel: require expandable structure on at least one side.
pub fn has_expandable_product_on_either_side(ctx: &Context, num: ExprId, den: ExprId) -> bool {
    contains_expandable_small_depth(ctx, num) || contains_expandable_small_depth(ctx, den)
}

/// Node-budget guard for expensive expand-to-cancel strategies.
pub fn within_div_expand_cancel_budget(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    max_total_nodes: usize,
) -> bool {
    node_count_tree(ctx, num) + node_count_tree(ctx, den) <= max_total_nodes
}

pub fn default_div_expand_cancel_poly_budget() -> PolyBudget {
    PolyBudget {
        max_terms: 200,
        max_total_degree: 12,
        max_pow_exp: 6,
    }
}

/// Discover shared opaque function calls between numerator and denominator.
pub fn find_shared_opaque_calls(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    depth_limit: usize,
    shared_limit: usize,
) -> Vec<(ExprId, ExprId)> {
    let num_calls = collect_function_calls_with_pow_limit(ctx, num, depth_limit, 18);
    let den_calls = collect_function_calls_with_pow_limit(ctx, den, depth_limit, 18);
    let mut shared = match_shared_calls_structural(ctx, &num_calls, &den_calls, shared_limit);
    if shared.len() > shared_limit {
        return Vec::new();
    }

    for &num_call in &num_calls {
        let Some((num_base, num_root_index)) = extract_root_family_signature(ctx, num_call) else {
            continue;
        };
        for &den_call in &den_calls {
            let Some((den_base, den_root_index)) = extract_root_family_signature(ctx, den_call)
            else {
                continue;
            };
            if num_root_index != den_root_index {
                continue;
            }
            if compare_expr(ctx, num_base, den_base) != Ordering::Equal {
                continue;
            }
            let already = shared.iter().any(|(existing_num, existing_den)| {
                compare_expr(ctx, *existing_num, num_call) == Ordering::Equal
                    && compare_expr(ctx, *existing_den, den_call) == Ordering::Equal
            });
            if already {
                continue;
            }
            shared.push((num_call, den_call));
            if shared.len() > shared_limit {
                return Vec::new();
            }
        }
    }

    shared
}

/// Try polynomial equivalence if both sides can be lowered into multipolys.
///
/// - `Some(true)`: both lowered and are equal.
/// - `Some(false)`: both lowered and differ.
/// - `None`: lowering failed on at least one side.
pub fn poly_equality_if_convertible(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    budget: &PolyBudget,
) -> Option<bool> {
    let p_num = multipoly_from_expr(ctx, num, budget).ok()?;
    let p_den = multipoly_from_expr(ctx, den, budget).ok()?;
    Some(p_num == p_den)
}

fn try_exact_poly_quotient_expr(ctx: &mut Context, num: ExprId, den: ExprId) -> Option<ExprId> {
    let budget = default_div_expand_cancel_poly_budget();
    let p_num = multipoly_from_expr(ctx, num, &budget).ok()?;
    let p_den = multipoly_from_expr(ctx, den, &budget).ok()?;
    let quotient = p_num.div_exact(&p_den)?;
    Some(multipoly_to_expr(&quotient, ctx))
}

/// Replace matched shared function calls with fresh temporary variables.
pub fn prepare_opaque_shared_substitution(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    shared: &[(ExprId, ExprId)],
) -> OpaqueSubstitutionPlan {
    let mut substituted_num = num;
    let mut substituted_den = den;
    let mut temp_vars = Vec::with_capacity(shared.len());

    for (i, (num_call, den_call)) in shared.iter().enumerate() {
        let temp_name = format!("__opq{}", i);
        let temp_var = ctx.var(&temp_name);
        let shared_root_family =
            extract_root_family_signature(ctx, *num_call).and_then(|(num_base, num_root_index)| {
                let (den_base, den_root_index) = extract_root_family_signature(ctx, *den_call)?;
                if num_root_index == den_root_index
                    && compare_expr(ctx, num_base, den_base) == Ordering::Equal
                {
                    Some((num_base, num_root_index))
                } else {
                    None
                }
            });

        if let Some((root_base, root_index)) = shared_root_family {
            let opts = SubstituteOptions {
                power_aware: true,
                ..Default::default()
            };
            if extract_root_family_base(ctx, *num_call).is_some() {
                substituted_num =
                    substitute_power_aware(ctx, substituted_num, *num_call, temp_var, opts);
            }
            if extract_root_family_base(ctx, *den_call).is_some() {
                substituted_den =
                    substitute_power_aware(ctx, substituted_den, *den_call, temp_var, opts);
            }
            substituted_num = replace_root_family_with_temp(
                ctx,
                substituted_num,
                root_base,
                root_index,
                temp_var,
            );
            substituted_den = replace_root_family_with_temp(
                ctx,
                substituted_den,
                root_base,
                root_index,
                temp_var,
            );
            let representative = canonical_root_family_atom(ctx, root_base, root_index);
            temp_vars.push((representative, temp_var));
            continue;
        }

        let opts = SubstituteOptions {
            power_aware: true,
            ..Default::default()
        };
        substituted_num = substitute_power_aware(ctx, substituted_num, *num_call, temp_var, opts);
        substituted_den = substitute_power_aware(ctx, substituted_den, *den_call, temp_var, opts);
        temp_vars.push((*num_call, temp_var));
    }

    OpaqueSubstitutionPlan {
        substituted_num,
        substituted_den,
        temp_vars,
    }
}

/// Restore temporary opaque variables back to original calls.
pub fn substitute_back_opaque_temps(
    ctx: &mut Context,
    mut expr: ExprId,
    temp_vars: &[(ExprId, ExprId)],
) -> ExprId {
    let opts = SubstituteOptions::default();
    for (call_id, temp_var) in temp_vars {
        expr = substitute_power_aware(ctx, expr, *temp_var, *call_id, opts);
    }
    expr
}

/// Strategy-0 helper: perform opaque substitution cancel attempt using a
/// caller-provided simplification callback.
///
/// The callback receives `(context_after_substitution, substituted_fraction)`
/// and must return `(updated_context, simplified_expr)` when simplification
/// succeeds.
pub fn try_opaque_substitution_cancel_with<FSim>(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    depth_limit: usize,
    shared_limit: usize,
    mut simplify_sub_fraction: FSim,
) -> Option<(Context, ExprId)>
where
    FSim: FnMut(&Context, ExprId) -> Option<(Context, ExprId)>,
{
    let shared = find_shared_opaque_calls(ctx, num, den, depth_limit, shared_limit);
    if shared.is_empty() {
        return None;
    }

    let mut local_ctx = ctx.clone();
    let plan = prepare_opaque_shared_substitution(&mut local_ctx, num, den, &shared);
    if both_sides_leaf_like(&local_ctx, plan.substituted_num, plan.substituted_den) {
        return None;
    }

    let sub_frac = local_ctx.add(Expr::Div(plan.substituted_num, plan.substituted_den));
    let (mut simplified_ctx, simplified) = simplify_sub_fraction(&local_ctx, sub_frac)?;
    if let Some(quotient_expr) = try_exact_poly_quotient_expr(
        &mut simplified_ctx,
        plan.substituted_num,
        plan.substituted_den,
    ) {
        let final_result =
            substitute_back_opaque_temps(&mut simplified_ctx, quotient_expr, &plan.temp_vars);
        return Some((simplified_ctx, final_result));
    }

    let simplified_div_parts = match simplified_ctx.get(simplified) {
        Expr::Div(simplified_num, simplified_den) => Some((*simplified_num, *simplified_den)),
        _ => None,
    };
    if let Some((simplified_num, simplified_den)) = simplified_div_parts {
        if let Some(quotient_expr) =
            try_exact_poly_quotient_expr(&mut simplified_ctx, simplified_num, simplified_den)
        {
            let final_result =
                substitute_back_opaque_temps(&mut simplified_ctx, quotient_expr, &plan.temp_vars);
            return Some((simplified_ctx, final_result));
        }
    }

    if matches!(simplified_ctx.get(simplified), Expr::Div(_, _)) {
        return None;
    }

    let final_result =
        substitute_back_opaque_temps(&mut simplified_ctx, simplified, &plan.temp_vars);
    Some((simplified_ctx, final_result))
}

/// Compare expanded forms using polynomial equality when possible, otherwise
/// fall back to structural ordering equality.
pub fn expanded_forms_cancel_equivalent(
    ctx: &Context,
    simplified_num: ExprId,
    simplified_den: ExprId,
    budget: &PolyBudget,
) -> bool {
    match poly_equality_if_convertible(ctx, simplified_num, simplified_den, budget) {
        Some(true) => true,
        Some(false) => false,
        None => compare_expr(ctx, simplified_num, simplified_den) == Ordering::Equal,
    }
}

/// Clone context, expand numerator and denominator, and return only when at
/// least one side changed after expansion.
pub fn expand_div_sides_if_changed_with<FExpand>(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    mut expand: FExpand,
) -> Option<ExpandedPair>
where
    FExpand: FnMut(&mut Context, ExprId) -> ExprId,
{
    let mut ctx_clone = ctx.clone();
    let expanded_num = expand(&mut ctx_clone, num);
    let expanded_den = expand(&mut ctx_clone, den);
    let num_changed = compare_expr(&ctx_clone, num, expanded_num) != Ordering::Equal;
    let den_changed = compare_expr(&ctx_clone, den, expanded_den) != Ordering::Equal;
    if !num_changed && !den_changed {
        return None;
    }

    Some(ExpandedPair {
        context: ctx_clone,
        expanded_num,
        expanded_den,
    })
}

/// Strategy-2 helper: expand both sides, simplify each expanded side via callback,
/// and compare by poly/structural equivalence.
pub fn try_expand_then_compare_cancel_with<FExpand, FSimplify>(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    expand: FExpand,
    mut simplify_expanded: FSimplify,
) -> Option<bool>
where
    FExpand: FnMut(&mut Context, ExprId) -> ExprId,
    FSimplify: FnMut(Context, ExprId, ExprId) -> Option<(Context, ExprId, ExprId)>,
{
    let expanded = expand_div_sides_if_changed_with(ctx, num, den, expand)?;
    let (simplified_ctx, simplified_num, simplified_den) = simplify_expanded(
        expanded.context,
        expanded.expanded_num,
        expanded.expanded_den,
    )?;
    let budget = default_div_expand_cancel_poly_budget();
    Some(expanded_forms_cancel_equivalent(
        &simplified_ctx,
        simplified_num,
        simplified_den,
        &budget,
    ))
}

/// End-to-end orchestration for `DivExpandToCancelRule` with injected
/// simplification callbacks for Strategy 0 and Strategy 2.
#[allow(clippy::too_many_arguments)]
pub fn try_rewrite_div_expand_to_cancel_expr_with<FStrategy0, FExpand, FStrategy2>(
    ctx: &mut Context,
    expr: ExprId,
    mut strategy0_simplify_sub_fraction: FStrategy0,
    expand: FExpand,
    mut strategy2_simplify_expanded: FStrategy2,
) -> Option<DivExpandToCancelRewrite>
where
    FStrategy0: FnMut(&Context, ExprId) -> Option<(Context, ExprId)>,
    FExpand: FnMut(&mut Context, ExprId) -> ExprId,
    FStrategy2: FnMut(Context, ExprId, ExprId) -> Option<(Context, ExprId, ExprId)>,
{
    let (num, den) = as_fraction_like_num_den(ctx, expr)?;

    if both_sides_leaf_like(ctx, num, den) {
        return None;
    }

    if let Some((new_ctx, final_result)) =
        try_opaque_substitution_cancel_with(ctx, num, den, 4, 3, |base_ctx, sub_frac| {
            strategy0_simplify_sub_fraction(base_ctx, sub_frac)
        })
    {
        *ctx = new_ctx;
        return Some(DivExpandToCancelRewrite {
            rewritten: final_result,
            kind: DivExpandToCancelKind::OpaqueSubstitution,
        });
    }

    if !has_expandable_product_on_either_side(ctx, num, den) {
        return None;
    }
    if !within_div_expand_cancel_budget(ctx, num, den, 150) {
        return None;
    }

    let budget = default_div_expand_cancel_poly_budget();
    if let Some(poly_equal) = poly_equality_if_convertible(ctx, num, den, &budget) {
        if poly_equal {
            return Some(DivExpandToCancelRewrite {
                rewritten: ctx.num(1),
                kind: DivExpandToCancelKind::ExpandedEquality,
            });
        }
        return None;
    }

    if try_expand_then_compare_cancel_with(
        ctx,
        num,
        den,
        expand,
        |expanded_ctx, expanded_num, expanded_den| {
            strategy2_simplify_expanded(expanded_ctx, expanded_num, expanded_den)
        },
    ) == Some(true)
    {
        return Some(DivExpandToCancelRewrite {
            rewritten: ctx.num(1),
            kind: DivExpandToCancelKind::ExpandedEquality,
        });
    }

    None
}

/// End-to-end orchestration equivalent to
/// [`try_rewrite_div_expand_to_cancel_expr_with`] with built-in recursion guards
/// for strategy callbacks.
///
/// This prevents accidental recursive re-entry of the expensive strategy hooks
/// when callers route the callbacks through a full simplifier pipeline.
#[allow(clippy::too_many_arguments)]
pub fn try_rewrite_div_expand_to_cancel_expr_with_thread_guards<FStrategy0, FExpand, FStrategy2>(
    ctx: &mut Context,
    expr: ExprId,
    mut strategy0_simplify_sub_fraction: FStrategy0,
    expand: FExpand,
    mut strategy2_simplify_expanded: FStrategy2,
) -> Option<DivExpandToCancelRewrite>
where
    FStrategy0: FnMut(&Context, ExprId) -> Option<(Context, ExprId)>,
    FExpand: FnMut(&mut Context, ExprId) -> ExprId,
    FStrategy2: FnMut(Context, ExprId, ExprId) -> Option<(Context, ExprId, ExprId)>,
{
    thread_local! {
        static OPAQUE_SUB_DEPTH: Cell<u32> = const { Cell::new(0) };
    }
    thread_local! {
        static EXPAND_CANCEL_DEPTH: Cell<u32> = const { Cell::new(0) };
    }

    let mut guarded_strategy0 = |base_ctx: &Context, sub_frac: ExprId| {
        let depth = OPAQUE_SUB_DEPTH.with(|c| c.get());
        if depth > 0 {
            return None;
        }
        OPAQUE_SUB_DEPTH.with(|c| c.set(depth + 1));
        let out = strategy0_simplify_sub_fraction(base_ctx, sub_frac);
        OPAQUE_SUB_DEPTH.with(|c| c.set(depth));
        out
    };

    let mut guarded_strategy2 =
        |expanded_ctx: Context, expanded_num: ExprId, expanded_den: ExprId| {
            let depth = EXPAND_CANCEL_DEPTH.with(|c| c.get());
            if depth > 0 {
                return None;
            }
            EXPAND_CANCEL_DEPTH.with(|c| c.set(depth + 1));
            let out = strategy2_simplify_expanded(expanded_ctx, expanded_num, expanded_den);
            EXPAND_CANCEL_DEPTH.with(|c| c.set(depth));
            out
        };

    try_rewrite_div_expand_to_cancel_expr_with(
        ctx,
        expr,
        &mut guarded_strategy0,
        expand,
        &mut guarded_strategy2,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::ordering::compare_expr;
    use cas_formatter::render_expr;
    use cas_parser::parse;

    #[test]
    fn leaf_like_guard_detects_atom_division() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse x");
        let y = parse("y", &mut ctx).expect("parse y");
        assert!(both_sides_leaf_like(&ctx, x, y));
    }

    #[test]
    fn poly_equality_detects_equivalent_forms() {
        let mut ctx = Context::new();
        let num = parse("(x + 1) * (x + 1)", &mut ctx).expect("parse num");
        let den = parse("x^2 + 2*x + 1", &mut ctx).expect("parse den");
        let budget = PolyBudget {
            max_terms: 200,
            max_total_degree: 12,
            max_pow_exp: 6,
        };
        assert_eq!(
            poly_equality_if_convertible(&ctx, num, den, &budget),
            Some(true)
        );
    }

    #[test]
    fn opaque_substitution_roundtrip_restores_call() {
        let mut ctx = Context::new();
        let num = parse("sin(x) + 2", &mut ctx).expect("parse num");
        let den = parse("sin(x) + 1", &mut ctx).expect("parse den");
        let call = parse("sin(x)", &mut ctx).expect("parse call");
        let shared = vec![(call, call)];

        let plan = prepare_opaque_shared_substitution(&mut ctx, num, den, &shared);
        assert_eq!(plan.temp_vars.len(), 1);
        assert!(!both_sides_leaf_like(
            &ctx,
            plan.substituted_num,
            plan.substituted_den
        ));

        let restored = substitute_back_opaque_temps(&mut ctx, plan.temp_vars[0].1, &plan.temp_vars);
        assert_eq!(render_expr(&ctx, restored), "sin(x)");
    }

    #[test]
    fn expanded_forms_fallback_structural_equivalence() {
        let mut ctx = Context::new();
        let num = parse("x + 1", &mut ctx).expect("parse num");
        let den = parse("x + 1", &mut ctx).expect("parse den");
        let budget = default_div_expand_cancel_poly_budget();
        assert!(expanded_forms_cancel_equivalent(&ctx, num, den, &budget));
    }

    #[test]
    fn expand_div_sides_returns_none_when_unchanged() {
        let mut ctx = Context::new();
        let num = parse("x", &mut ctx).expect("parse num");
        let den = parse("y", &mut ctx).expect("parse den");
        let expanded = expand_div_sides_if_changed_with(&ctx, num, den, |_c, e| e);
        assert!(expanded.is_none());
    }

    #[test]
    fn find_shared_opaque_calls_detects_common_sin_call() {
        let mut ctx = Context::new();
        let num = parse("sin(x) + 2", &mut ctx).expect("parse num");
        let den = parse("sin(x) + 1", &mut ctx).expect("parse den");
        let shared = find_shared_opaque_calls(&ctx, num, den, 4, 3);
        assert_eq!(shared.len(), 1);
    }

    #[test]
    fn opaque_substitution_strategy_returns_none_when_simplified_is_still_div() {
        let mut ctx = Context::new();
        let num = parse("sin(x) + 2", &mut ctx).expect("parse num");
        let den = parse("sin(x) + 1", &mut ctx).expect("parse den");
        let result = try_opaque_substitution_cancel_with(&ctx, num, den, 4, 3, |base_ctx, frac| {
            Some((base_ctx.clone(), frac))
        });
        assert!(result.is_none());
    }

    #[test]
    fn expand_then_compare_reports_equivalence_when_callback_preserves_equal_forms() {
        let mut ctx = Context::new();
        let num = parse("x + 1", &mut ctx).expect("parse num");
        let den = parse("x + 1", &mut ctx).expect("parse den");
        let result = try_expand_then_compare_cancel_with(
            &ctx,
            num,
            den,
            |_c, e| e,
            |local_ctx, left, right| Some((local_ctx, left, right)),
        );
        assert_eq!(result, None);

        let mut ctx2 = Context::new();
        let num2 = parse("(x+1)^2", &mut ctx2).expect("parse num2");
        let den2 = parse("x^2 + 2*x + 1", &mut ctx2).expect("parse den2");
        let result2 = try_expand_then_compare_cancel_with(
            &ctx2,
            num2,
            den2,
            |_c, e| if e == num2 { den2 } else { e },
            |local_ctx, left, right| Some((local_ctx, left, right)),
        );
        assert_eq!(result2, Some(true));
    }

    #[test]
    fn orchestration_rejects_non_div_expression() {
        let mut ctx = Context::new();
        let expr = parse("x + 1", &mut ctx).expect("parse expr");
        let result = try_rewrite_div_expand_to_cancel_expr_with(
            &mut ctx,
            expr,
            |_c, _e| None,
            |_c, e| e,
            |_ctx, _l, _r| None,
        );
        assert!(result.is_none());
    }

    #[test]
    fn orchestration_uses_strategy0_when_available() {
        let mut ctx = Context::new();
        let expr = parse("(sin(x) + 2)/(sin(x) + 1)", &mut ctx).expect("parse expr");
        let result = try_rewrite_div_expand_to_cancel_expr_with(
            &mut ctx,
            expr,
            |base_ctx, _sub| {
                let mut local = base_ctx.clone();
                let five = local.num(5);
                Some((local, five))
            },
            |_c, e| e,
            |_ctx, _l, _r| None,
        )
        .expect("expected strategy0 rewrite");
        assert_eq!(result.kind, DivExpandToCancelKind::OpaqueSubstitution);
        assert_eq!(render_expr(&ctx, result.rewritten), "5");
    }

    #[test]
    fn orchestration_uses_strategy1_poly_equality() {
        let mut ctx = Context::new();
        let expr = parse("((x+1)^2)/(x^2 + 2*x + 1)", &mut ctx).expect("parse expr");
        let result = try_rewrite_div_expand_to_cancel_expr_with(
            &mut ctx,
            expr,
            |_c, _e| None,
            |_c, e| e,
            |_ctx, _l, _r| None,
        )
        .expect("expected poly equality rewrite");
        assert_eq!(result.kind, DivExpandToCancelKind::ExpandedEquality);
        assert_eq!(render_expr(&ctx, result.rewritten), "1");
    }

    #[test]
    fn opaque_substitution_strategy_extracts_exact_poly_quotient() {
        let mut ctx = Context::new();
        let num = parse("((x^2 + 1)^(1/2))^2 + 2*(x^2 + 1)^(1/2)", &mut ctx).expect("parse num");
        let den = parse("(x^2 + 1)^(1/2) + 2", &mut ctx).expect("parse den");
        let result = try_opaque_substitution_cancel_with(&ctx, num, den, 4, 3, |base_ctx, frac| {
            Some((base_ctx.clone(), frac))
        })
        .expect("opaque quotient");
        assert_eq!(render_expr(&result.0, result.1), "(x^2 + 1)^(1/2)");
    }

    #[test]
    fn opaque_substitution_strategy_extracts_exact_poly_quotient_from_collapsed_root_base() {
        let mut ctx = Context::new();
        let num = parse("x^2 + 1 + 2*(x^2 + 1)^(1/2)", &mut ctx).expect("parse num");
        let den = parse("(x^2 + 1)^(1/2) + 2", &mut ctx).expect("parse den");
        let result = try_opaque_substitution_cancel_with(&ctx, num, den, 4, 3, |base_ctx, frac| {
            Some((base_ctx.clone(), frac))
        })
        .expect("opaque quotient");
        assert_eq!(render_expr(&result.0, result.1), "(x^2 + 1)^(1/2)");
    }

    #[test]
    fn opaque_substitution_strategy_extracts_exact_poly_quotient_from_builtin_sqrt_base() {
        let mut ctx = Context::new();
        let num = parse("sqrt(x^2 + 1)^2 + 2*sqrt(x^2 + 1)", &mut ctx).expect("parse num");
        let den = parse("sqrt(x^2 + 1) + 2", &mut ctx).expect("parse den");
        let result = try_opaque_substitution_cancel_with(&ctx, num, den, 4, 3, |base_ctx, frac| {
            Some((base_ctx.clone(), frac))
        })
        .expect("opaque quotient");
        assert_eq!(render_expr(&result.0, result.1), "(x^2 + 1)^(1/2)");
    }

    #[test]
    fn find_shared_opaque_calls_detects_collapsed_root_power_atoms() {
        let mut ctx = Context::new();
        let num = parse("x^2 + 1 + 2*(x^2 + 1)^(1/2)", &mut ctx).expect("parse num");
        let den = parse("(x^2 + 1)^(1/2) + 2", &mut ctx).expect("parse den");
        let shared = find_shared_opaque_calls(&ctx, num, den, 4, 3);
        assert_eq!(shared.len(), 1);
    }

    #[test]
    fn prepare_opaque_shared_substitution_canonical_root_cube_uses_temp_power() {
        let mut ctx = Context::new();
        let num = parse("(x^2 + 1)^(3/2) - 1", &mut ctx).expect("parse num");
        let den = parse("sqrt(x^2 + 1) - 1", &mut ctx).expect("parse den");
        let shared = find_shared_opaque_calls(&ctx, num, den, 4, 3);
        assert_eq!(shared.len(), 1);
        let plan = prepare_opaque_shared_substitution(&mut ctx, num, den, &shared);
        let expected_num = parse("__opq0^3 - 1", &mut ctx).expect("expected num");
        let expected_den = parse("__opq0 - 1", &mut ctx).expect("expected den");
        assert_eq!(
            compare_expr(&ctx, plan.substituted_num, expected_num),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            compare_expr(&ctx, plan.substituted_den, expected_den),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn substitute_back_opaque_temps_restores_canonical_root_atom_for_family_powers() {
        let mut ctx = Context::new();
        let num = parse("sqrt(x^2 + 1)^5", &mut ctx).expect("parse num");
        let den = parse("sqrt(x^2 + 1)^3", &mut ctx).expect("parse den");
        let shared = find_shared_opaque_calls(&ctx, num, den, 4, 3);
        assert_eq!(shared.len(), 1);

        let plan = prepare_opaque_shared_substitution(&mut ctx, num, den, &shared);
        let quotient = parse("__opq0^2", &mut ctx).expect("parse quotient");
        let restored = substitute_back_opaque_temps(&mut ctx, quotient, &plan.temp_vars);
        assert_eq!(render_expr(&ctx, restored), "(x^2 + 1)^(1/2)^2");
    }

    #[test]
    fn end_to_end_div_expand_cancel_rewrites_collapsed_root_quotient() {
        let mut ctx = Context::new();
        let expr = parse(
            "(x^2 + 1 + 2*(x^2 + 1)^(1/2))/((x^2 + 1)^(1/2) + 2)",
            &mut ctx,
        )
        .expect("parse expr");
        let rewrite = try_rewrite_div_expand_to_cancel_expr_with(
            &mut ctx,
            expr,
            |base_ctx, sub_frac| Some((base_ctx.clone(), sub_frac)),
            |_expand_ctx, expand_expr| expand_expr,
            |expanded_ctx, expanded_num, expanded_den| {
                Some((expanded_ctx, expanded_num, expanded_den))
            },
        )
        .expect("rewrite");
        assert_eq!(render_expr(&ctx, rewrite.rewritten), "(x^2 + 1)^(1/2)");
    }

    #[test]
    fn end_to_end_div_expand_cancel_rewrites_same_root_family_power_quotient() {
        let mut ctx = Context::new();
        let expr = parse("sqrt(x^2 + 1)^5/sqrt(x^2 + 1)^3", &mut ctx).expect("parse expr");
        let rewrite = try_rewrite_div_expand_to_cancel_expr_with(
            &mut ctx,
            expr,
            |base_ctx, sub_frac| Some((base_ctx.clone(), sub_frac)),
            |_expand_ctx, expand_expr| expand_expr,
            |expanded_ctx, expanded_num, expanded_den| {
                Some((expanded_ctx, expanded_num, expanded_den))
            },
        )
        .expect("rewrite");
        assert_eq!(render_expr(&ctx, rewrite.rewritten), "(x^2 + 1)^(1/2)^2");
    }

    #[test]
    fn end_to_end_div_expand_cancel_rewrites_collapsed_root_quotient_plus_one() {
        let mut ctx = Context::new();
        let expr = parse(
            "(u^2 + 2*(u^2 + 1)^(1/2) + 2)/((u^2 + 1)^(1/2) + 1)",
            &mut ctx,
        )
        .expect("parse expr");
        let rewrite = try_rewrite_div_expand_to_cancel_expr_with(
            &mut ctx,
            expr,
            |base_ctx, sub_frac| Some((base_ctx.clone(), sub_frac)),
            |_expand_ctx, expand_expr| expand_expr,
            |expanded_ctx, expanded_num, expanded_den| {
                Some((expanded_ctx, expanded_num, expanded_den))
            },
        )
        .expect("rewrite");
        assert_eq!(render_expr(&ctx, rewrite.rewritten), "(u^2 + 1)^(1/2) + 1");
    }

    #[test]
    fn end_to_end_div_expand_cancel_rewrites_collapsed_root_cube_quotient() {
        let mut ctx = Context::new();
        let expr =
            parse("(sqrt(u^2 + 1)^3 - 1)/(sqrt(u^2 + 1) - 1)", &mut ctx).expect("parse expr");
        let rewrite = try_rewrite_div_expand_to_cancel_expr_with(
            &mut ctx,
            expr,
            |base_ctx, sub_frac| Some((base_ctx.clone(), sub_frac)),
            |_expand_ctx, expand_expr| expand_expr,
            |expanded_ctx, expanded_num, expanded_den| {
                Some((expanded_ctx, expanded_num, expanded_den))
            },
        )
        .expect("rewrite");
        assert_eq!(
            render_expr(&ctx, rewrite.rewritten),
            "(u^2 + 1)^(1/2)^2 + (u^2 + 1)^(1/2) + 1"
        );
    }

    #[test]
    fn end_to_end_div_expand_cancel_rewrites_canonical_root_cube_quotient() {
        let mut ctx = Context::new();
        let expr =
            parse("((u^2 + 1)^(3/2) - 1)/(sqrt(u^2 + 1) - 1)", &mut ctx).expect("parse expr");
        let rewrite = try_rewrite_div_expand_to_cancel_expr_with(
            &mut ctx,
            expr,
            |base_ctx, sub_frac| Some((base_ctx.clone(), sub_frac)),
            |_expand_ctx, expand_expr| expand_expr,
            |expanded_ctx, expanded_num, expanded_den| {
                Some((expanded_ctx, expanded_num, expanded_den))
            },
        )
        .expect("rewrite");
        assert_eq!(
            render_expr(&ctx, rewrite.rewritten),
            "(u^2 + 1)^(1/2)^2 + (u^2 + 1)^(1/2) + 1"
        );
    }
}
