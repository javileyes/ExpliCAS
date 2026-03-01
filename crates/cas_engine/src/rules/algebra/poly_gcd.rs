//! Polynomial GCD structural rule.
//!
//! Implements `poly_gcd(a, b)` which finds the structural GCD of two expressions
//! by collecting multiplicative factors and intersecting them.
//!
//! Example:
//! ```text
//! poly_gcd((1+x)^3 * (2+y), (1+x)^2 * (3+z)) = (1+x)^2
//! poly_gcd(a*g, b*g) = g
//! ```
//!
//! This allows Mathematica/Symbolica-style polynomial GCD without expanding.

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::poly_gcd_dispatch::rewrite_user_poly_gcd_call_held_with_expand_eval;

// =============================================================================
// REPL function rule
// =============================================================================

// Rule for `poly_gcd(a, b)` function.
// Computes structural GCD of two polynomial expressions.
define_rule!(
    PolyGcdRule,
    "Polynomial GCD",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    priority: 200,
    |ctx, expr| {
        let rewritten = rewrite_user_poly_gcd_call_held_with_expand_eval(
            ctx,
            expr,
            crate::expand::eval_expand_off,
            cas_formatter::render_expr,
        )?;
        Some(Rewrite::simple(rewritten.0, rewritten.1))
    }
);
