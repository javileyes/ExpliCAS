//! Helpers for `DivExpandToCancelRule` extraction.
//!
//! These routines keep structural/polynomial guards in `cas_math` while the
//! engine keeps orchestration details (recursion guards and simplifier calls).

use crate::expandable_pattern_support::contains_expandable_small_depth;
use crate::expr_complexity::node_count_tree;
use crate::expr_destructure::as_div;
use crate::multipoly::{multipoly_from_expr, PolyBudget};
use crate::opaque_function_calls_support::{
    collect_function_calls_limited, match_shared_calls_structural,
};
use crate::substitute::{substitute_power_aware, SubstituteOptions};
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
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

fn is_leaf_like(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_)
    )
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
    let num_calls = collect_function_calls_limited(ctx, num, depth_limit);
    let den_calls = collect_function_calls_limited(ctx, den, depth_limit);
    let shared = match_shared_calls_structural(ctx, &num_calls, &den_calls, shared_limit);
    if shared.len() > shared_limit {
        Vec::new()
    } else {
        shared
    }
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
    let (num, den) = as_div(ctx, expr)?;

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
}
