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
    try_plan_finite_product_evaluation, try_plan_finite_sum_evaluation, FiniteAggregateCall,
    ProductEvaluationKind, SumEvaluationKind,
};

// =============================================================================
// SUM RULE: Evaluate finite summations
// =============================================================================
// Syntax: sum(expr, var, start, end)
// Example: sum(k, k, 1, 10) → 55
// Example: sum(k^2, k, 1, 5) → 1 + 4 + 9 + 16 + 25 = 55

define_rule!(
    SumRule,
    "Finite Summation",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let plan = try_plan_finite_sum_evaluation(ctx, expr, 1000)?;
        let result = simplify_with_engine_rules(ctx, plan.candidate);
        let desc = render_sum_evaluation_desc(&plan.kind, &plan.call, |id| {
            format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
        });
        Some(Rewrite::new(result).desc(desc))
    }
);

// =============================================================================
// PRODUCT RULE: Evaluate finite products (productorio)
// =============================================================================
// Syntax: product(expr, var, start, end)
// Example: product(k, k, 1, 5) → 120  (5!)
// Example: product((k+1)/k, k, 1, n) → n+1  (telescoping)

define_rule!(
    ProductRule,
    "Finite Product",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let plan = try_plan_finite_product_evaluation(ctx, expr, 1000)?;
        let result = simplify_with_engine_rules(ctx, plan.candidate);
        let desc = render_product_evaluation_desc(&plan.kind, &plan.call, |id| {
            format!("{}", cas_formatter::DisplayExpr { context: ctx, id })
        });
        Some(Rewrite::new(result).desc(desc))
    }
);

fn render_sum_evaluation_desc<F>(
    kind: &SumEvaluationKind,
    call: &FiniteAggregateCall,
    mut render_expr: F,
) -> String
where
    F: FnMut(ExprId) -> String,
{
    match kind {
        SumEvaluationKind::Telescoping => format!(
            "Telescoping sum: Σ({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        SumEvaluationKind::SumOfFirstIntegers => format!(
            "Sum of first integers: Σ({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        SumEvaluationKind::SumOfSquares => format!(
            "Sum of squares: Σ({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        SumEvaluationKind::SumOfCubes => format!(
            "Sum of cubes: Σ({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        SumEvaluationKind::SumOfConstant => format!(
            "Sum of constant term: Σ({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        SumEvaluationKind::GeometricPower => format!(
            "Geometric sum: Σ({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        SumEvaluationKind::FiniteDirect { start, end } => format!(
            "sum({}, {}, {}, {})",
            render_expr(call.term),
            call.var_name,
            start,
            end
        ),
    }
}

fn render_product_evaluation_desc<F>(
    kind: &ProductEvaluationKind,
    call: &FiniteAggregateCall,
    mut render_expr: F,
) -> String
where
    F: FnMut(ExprId) -> String,
{
    match kind {
        ProductEvaluationKind::Telescoping => format!(
            "Telescoping product: Π({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        ProductEvaluationKind::FactorizedTelescoping => format!(
            "Factorized telescoping product: Π({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        ProductEvaluationKind::ProductOfFirstIntegers => format!(
            "Product of first integers: Π({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        ProductEvaluationKind::ProductOfPowers => format!(
            "Product of powers: Π({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        ProductEvaluationKind::ProductOfConstant => format!(
            "Product of constant factor: Π({}, {}) from {} to {}",
            render_expr(call.term),
            call.var_name,
            render_expr(call.start_expr),
            render_expr(call.end_expr)
        ),
        ProductEvaluationKind::FiniteDirect { start, end } => format!(
            "product({}, {}, {}, {})",
            render_expr(call.term),
            call.var_name,
            start,
            end
        ),
    }
}

fn simplify_with_engine_rules(ctx: &mut Context, expr: ExprId) -> ExprId {
    let mut simplifier = crate::Simplifier::with_default_rules();
    simplifier.context = ctx.clone();
    let (simplified, _) = simplifier.simplify(expr);
    *ctx = simplifier.context;
    simplified
}

#[cfg(test)]
mod tests {
    use super::{ProductRule, SumRule};
    use crate::rule::Rule;
    use cas_ast::target_kind::TargetKind;

    #[test]
    fn finite_aggregate_rules_target_function_calls_only() {
        for (name, targets) in [
            (
                "SumRule",
                SumRule
                    .target_types()
                    .expect("SumRule should be structurally targeted"),
            ),
            (
                "ProductRule",
                ProductRule
                    .target_types()
                    .expect("ProductRule should be structurally targeted"),
            ),
        ] {
            assert!(
                targets.contains(TargetKind::Function),
                "{name} should target function calls"
            );
            assert!(
                !targets.contains(TargetKind::Add),
                "{name} should not target Add"
            );
            assert!(
                !targets.contains(TargetKind::Sub),
                "{name} should not target Sub"
            );
            assert!(
                !targets.contains(TargetKind::Mul),
                "{name} should not target Mul"
            );
            assert!(
                !targets.contains(TargetKind::Div),
                "{name} should not target Div"
            );
            assert!(
                !targets.contains(TargetKind::Pow),
                "{name} should not target Pow"
            );
        }
    }
}
