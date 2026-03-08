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
use cas_math::poly_gcd_dispatch::{compute_poly_gcd_unified_with, pre_evaluate_for_gcd_with};
use cas_math::poly_gcd_mode::{try_parse_poly_gcd_call, GcdGoal};

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
        let parsed = try_parse_poly_gcd_call(ctx, expr)?;
        let (result, desc) = compute_poly_gcd_unified_with(
            ctx,
            parsed.lhs,
            parsed.rhs,
            GcdGoal::UserPolyGcd,
            parsed.mode,
            parsed.modp_preset,
            parsed.modp_main_var,
            |core_ctx, id| {
                pre_evaluate_for_gcd_with(core_ctx, id, crate::expand::eval_expand_off)
            },
            cas_formatter::render_expr,
        );
        Some(Rewrite::simple(cas_ast::hold::wrap_hold(ctx, result), desc))
    }
);
