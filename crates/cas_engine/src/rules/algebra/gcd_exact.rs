//! Rule wrapper for exact polynomial GCD over ℚ[x1,...,xn].

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_formatter::DisplayExpr;
use cas_math::gcd_exact::{try_rewrite_poly_gcd_exact_function_expr, GcdExactBudget};

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
        Some(Rewrite::simple(
            rewrite.gcd,
            format!(
                "poly_gcd_exact({}, {}) [{:?}]",
                DisplayExpr {
                    context: ctx,
                    id: rewrite.lhs
                },
                DisplayExpr {
                    context: ctx,
                    id: rewrite.rhs
                },
                rewrite.layer_used
            ),
        ))
    }
);
