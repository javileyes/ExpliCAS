//! Rule wrapper for exact polynomial GCD over ℚ[x1,...,xn].

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::gcd_exact::rewrite_poly_gcd_exact_function_expr_default_with;

// Rule for poly_gcd_exact(a, b) function.
// Computes algebraic GCD of two polynomial expressions over ℚ.
define_rule!(
    PolyGcdExactRule,
    "Polynomial GCD Exact",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    priority: 200, // High priority to evaluate early
    |ctx, expr| {
        let rewritten = rewrite_poly_gcd_exact_function_expr_default_with(
            ctx,
            expr,
            cas_formatter::render_expr,
        )?;
        Some(Rewrite::simple(rewritten.0, rewritten.1))
    }
);
