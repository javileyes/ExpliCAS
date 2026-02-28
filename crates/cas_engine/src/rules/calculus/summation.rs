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
    try_plan_finite_product_evaluation, try_plan_finite_sum_evaluation, ProductEvaluationKind,
    SumEvaluationKind,
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
    let call = plan.call;
    let term = call.term;
    let var_name = call.var_name;
    let start_expr = call.start_expr;
    let end_expr = call.end_expr;

    match plan.kind {
        SumEvaluationKind::Telescoping => Some(Rewrite::new(result).desc_lazy(|| {
            format!(
                "Telescoping sum: Σ({}, {}) from {} to {}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: term
                },
                var_name,
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: start_expr
                },
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: end_expr
                }
            )
        })),
        SumEvaluationKind::FiniteDirect { start, end } => {
            Some(Rewrite::new(result).desc_lazy(|| {
                format!(
                    "sum({}, {}, {}, {})",
                    cas_formatter::DisplayExpr {
                        context: ctx,
                        id: term
                    },
                    var_name,
                    start,
                    end
                )
            }))
        }
    }
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
    let call = plan.call;
    let term = call.term;
    let var_name = call.var_name;
    let start_expr = call.start_expr;
    let end_expr = call.end_expr;

    match plan.kind {
        ProductEvaluationKind::Telescoping => Some(Rewrite::new(result).desc_lazy(|| {
            format!(
                "Telescoping product: Π({}, {}) from {} to {}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: term
                },
                var_name,
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: start_expr
                },
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: end_expr
                }
            )
        })),
        ProductEvaluationKind::FactorizedTelescoping => {
            Some(Rewrite::new(result).desc_lazy(|| {
                format!(
                    "Factorized telescoping product: Π({}, {}) from {} to {}",
                    cas_formatter::DisplayExpr {
                        context: ctx,
                        id: term
                    },
                    var_name,
                    cas_formatter::DisplayExpr {
                        context: ctx,
                        id: start_expr
                    },
                    cas_formatter::DisplayExpr {
                        context: ctx,
                        id: end_expr
                    }
                )
            }))
        }
        ProductEvaluationKind::FiniteDirect { start, end } => {
            Some(Rewrite::new(result).desc_lazy(|| {
                format!(
                    "product({}, {}, {}, {})",
                    cas_formatter::DisplayExpr {
                        context: ctx,
                        id: term
                    },
                    var_name,
                    start,
                    end
                )
            }))
        }
    }
});

fn simplify_with_engine_rules(ctx: &mut Context, expr: ExprId) -> ExprId {
    let mut simplifier = crate::Simplifier::with_default_rules();
    simplifier.context = ctx.clone();
    let (simplified, _) = simplifier.simplify(expr);
    *ctx = simplifier.context;
    simplified
}
