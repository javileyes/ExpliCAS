//! poly_gcd_modp and poly_eq_modp REPL functions.
//!
//! Exposes Zippel mod-p GCD to REPL for fast polynomial verification.

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{Context, ExprId};
use cas_formatter::DisplayExpr;
use cas_math::poly_modp_calls::{
    eager_eval_poly_gcd_calls_with, format_poly_eq_modp_desc_with, format_poly_gcd_modp_desc_with,
    try_eval_poly_eq_modp_call_with_error_policy, try_eval_poly_gcd_modp_call_with_error_policy,
};
use cas_math::poly_modp_conv::DEFAULT_PRIME as INTERNAL_DEFAULT_PRIME;

const DEFAULT_PRIME: u64 = INTERNAL_DEFAULT_PRIME;

/// Eager evaluation pass for poly_gcd_modp calls.
///
/// This function traverses the expression tree TOP-DOWN and evaluates
/// poly_gcd_modp calls BEFORE the normal simplification pipeline.
///
/// CRITICAL: When we find poly_gcd_modp, we do NOT descend into its children.
/// This prevents the expensive symbolic expansion of huge arguments.
pub(crate) fn eager_eval_poly_gcd_calls(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> (ExprId, Vec<crate::Step>) {
    eager_eval_poly_gcd_calls_with(ctx, expr, collect_steps, |core_ctx, before, after| {
        crate::Step::new(
            "Eager eval poly_gcd_modp (bypass simplifier)",
            "Polynomial GCD mod p",
            before,
            after,
            Vec::new(),
            Some(core_ctx),
        )
    })
}

// Rule for poly_gcd_modp(a, b [, p]) function.
// Computes Zippel GCD of two polynomial expressions mod p.
define_rule!(
    PolyGcdModpRule,
    "Polynomial GCD mod p",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    priority: 200, // High priority to evaluate early
    |ctx, expr| {
        let call = try_eval_poly_gcd_modp_call_with_error_policy(ctx, expr, DEFAULT_PRIME, |e| {
                eprintln!("poly_gcd_modp error: {}", e);
            })?;
        let desc = format_poly_gcd_modp_desc_with(call.a_expr, call.b_expr, call.path, |id| {
            format!("{}", DisplayExpr { context: ctx, id })
        });
        Some(Rewrite::simple(call.held_expr, desc))
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
        let call = try_eval_poly_eq_modp_call_with_error_policy(ctx, expr, DEFAULT_PRIME, |e| {
                eprintln!("poly_eq_modp error: {}", e);
            })?;
        Some(Rewrite::simple(
            call.indicator_expr,
            format_poly_eq_modp_desc_with(call.a_expr, call.b_expr, call.equal, |id| {
                format!("{}", DisplayExpr { context: ctx, id })
            }),
        ))
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use cas_math::poly_modp_conv::check_poly_equal_modp_expr;
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
