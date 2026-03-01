//! Finite summation and product rules.
//!
//! Contains `SumRule`, `ProductRule`, and their helper functions:
//! - Telescoping sum/product detection
//! - Factorizable product detection
//! - Variable substitution

use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{Context, ExprId};
use cas_math::summation_support::{
    render_product_evaluation_desc_with, render_sum_evaluation_desc_with,
    try_plan_finite_product_evaluation, try_plan_finite_sum_evaluation,
};

// =============================================================================
// SUM RULE: Evaluate finite summations
// =============================================================================
// Syntax: sum(expr, var, start, end)
// Example: sum(k, k, 1, 10) → 55
// Example: sum(k^2, k, 1, 5) → 1 + 4 + 9 + 16 + 25 = 55

define_rule!(SumRule, "Finite Summation", |ctx, expr| {
    let plan = try_plan_finite_sum_evaluation(ctx, expr, 1000)?;
    let result = simplify_with_engine_rules(ctx, plan.candidate);
    let desc = render_sum_evaluation_desc_with(&plan.kind, &plan.call, |id| {
        format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
    });
    Some(Rewrite::new(result).desc(desc))
});

// =============================================================================
// PRODUCT RULE: Evaluate finite products (productorio)
// =============================================================================
// Syntax: product(expr, var, start, end)
// Example: product(k, k, 1, 5) → 120  (5!)
// Example: product((k+1)/k, k, 1, n) → n+1  (telescoping)

define_rule!(ProductRule, "Finite Product", |ctx, expr| {
    let plan = try_plan_finite_product_evaluation(ctx, expr, 1000)?;
    let result = simplify_with_engine_rules(ctx, plan.candidate);
    let desc = render_product_evaluation_desc_with(&plan.kind, &plan.call, |id| {
        format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
    });
    Some(Rewrite::new(result).desc(desc))
});

fn simplify_with_engine_rules(ctx: &mut Context, expr: ExprId) -> ExprId {
    let mut simplifier = crate::Simplifier::with_default_rules();
    simplifier.context = ctx.clone();
    let (simplified, _) = simplifier.simplify(expr);
    *ctx = simplifier.context;
    simplified
}
