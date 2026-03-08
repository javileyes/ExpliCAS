//! Rule wrapper for exact polynomial GCD over ℚ[x1,...,xn].

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::gcd_exact::{
    format_poly_gcd_exact_desc_with, try_rewrite_poly_gcd_exact_function_expr, GcdExactBudget,
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
        let desc = format_poly_gcd_exact_desc_with(
            rewrite.lhs,
            rewrite.rhs,
            rewrite.layer_used,
            |id| cas_formatter::render_expr(ctx, id),
        );
        Some(Rewrite::simple(rewrite.gcd, desc))
    }
);
