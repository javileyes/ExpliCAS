//! Rule wrapper for exact polynomial GCD over ℚ[x1,...,xn].

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::gcd_exact::{
    try_rewrite_poly_gcd_exact_function_expr, GcdExactBudget, GcdExactLayer,
};

// Rule for poly_gcd_exact(a, b) function.
// Computes algebraic GCD of two polynomial expressions over ℚ.
define_rule!(
    PolyGcdExactRule,
    "Polynomial GCD Exact",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    priority: 200, // High priority to evaluate early
    |ctx, expr| {
        let rewrite =
            try_rewrite_poly_gcd_exact_function_expr(ctx, expr, &GcdExactBudget::default())?;
        let desc = format_poly_gcd_exact_desc(ctx, rewrite.lhs, rewrite.rhs, rewrite.layer_used);
        Some(Rewrite::simple(rewrite.gcd, desc))
    }
);

fn format_poly_gcd_exact_desc(
    ctx: &cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
    layer_used: GcdExactLayer,
) -> String {
    format!(
        "poly_gcd_exact({}, {}) [{:?}]",
        cas_formatter::render_expr(ctx, lhs),
        cas_formatter::render_expr(ctx, rhs),
        layer_used
    )
}
