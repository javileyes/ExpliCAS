//! poly_gcd_modp and poly_eq_modp REPL functions.
//!
//! Exposes Zippel mod-p GCD to REPL for fast polynomial verification.

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::poly_modp_calls::{
    rewrite_poly_eq_modp_call_default_silent_with, rewrite_poly_gcd_modp_call_default_silent_with,
};

// Rule for poly_gcd_modp(a, b [, p]) function.
// Computes Zippel GCD of two polynomial expressions mod p.
define_rule!(
    PolyGcdModpRule,
    "Polynomial GCD mod p",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    priority: 200, // High priority to evaluate early
    |ctx, expr| {
        let rewritten =
            rewrite_poly_gcd_modp_call_default_silent_with(ctx, expr, cas_formatter::render_expr)?;
        Some(Rewrite::simple(rewritten.0, rewritten.1))
    }
);

// Rule for poly_eq_modp(a, b [, p]) function.
// Returns 1 if polynomials are equal mod p, 0 otherwise.
define_rule!(
    PolyEqModpRule,
    "Polynomial equality mod p",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    priority: 200,
    |ctx, expr| {
        let rewritten =
            rewrite_poly_eq_modp_call_default_silent_with(ctx, expr, cas_formatter::render_expr)?;
        Some(Rewrite::simple(rewritten.0, rewritten.1))
    }
);

#[cfg(test)]
mod tests {
    use cas_ast::Context;
    use cas_math::poly_modp_conv::{check_poly_equal_modp_expr, DEFAULT_PRIME};
    use cas_parser::parse;

    #[test]
    fn test_poly_eq_modp_same() {
        let mut ctx = Context::new();
        let a = parse("x + 1", &mut ctx).unwrap();
        let b = parse("1 + x", &mut ctx).unwrap();

        let result = check_poly_equal_modp_expr(&ctx, a, b, DEFAULT_PRIME).unwrap();
        assert!(result);
    }

    #[test]
    fn test_poly_eq_modp_different() {
        let mut ctx = Context::new();
        let a = parse("x + 1", &mut ctx).unwrap();
        let b = parse("x + 2", &mut ctx).unwrap();

        let result = check_poly_equal_modp_expr(&ctx, a, b, DEFAULT_PRIME).unwrap();
        assert!(!result);
    }
}
