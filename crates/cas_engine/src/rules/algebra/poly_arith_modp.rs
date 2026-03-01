//! Polynomial Arithmetic on __hold operands (mod p).
//!
//! This module provides rules for arithmetic operations between __hold-wrapped
//! polynomial expressions. When both sides of a Sub/Add are __hold(polynomial),
//! we convert to MultiPolyModP and compute in that domain.
//!
//! Key use case: `poly_gcd(expand(a*g), expand(b*g), modp) - expand(g)` → 0
//!
//! The problem: expand() returns __hold(giant_polynomial) to prevent simplifier
//! explosion, but Sub doesn't "enter" __hold by design. This rule handles it.

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::poly_modp_calls::try_rewrite_hold_poly_sub_to_zero_default;

// PolySubModpRule: handle __hold(P) - __hold(Q) in polynomial domain
define_rule!(
    PolySubModpRule,
    "Polynomial Subtraction (mod p)",
    Some(crate::target_kind::TargetKindSet::SUB),
    PhaseMask::CORE | PhaseMask::POST,
    |ctx, expr| {
        let zero = try_rewrite_hold_poly_sub_to_zero_default(ctx, expr)?;
        Some(
            Rewrite::new(zero)
                .desc("__hold(P) - __hold(Q) = 0 (equal polynomials mod p, up to scalar)"),
        )
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parent_context::ParentContext;
    use crate::rule::Rule;
    use cas_ast::{Context, Expr};
    use num_traits::Zero;

    #[test]
    fn test_hold_subtraction_equal_polys() {
        let mut ctx = Context::new();

        // Create two identical held polynomials: __hold(x + 1) - __hold(x + 1)
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_plus_1 = ctx.add(Expr::Add(x, one));
        let hold_a = cas_ast::hold::wrap_hold(&mut ctx, x_plus_1);
        let hold_b = cas_ast::hold::wrap_hold(&mut ctx, x_plus_1);
        let sub_expr = ctx.add(Expr::Sub(hold_a, hold_b));

        // Apply rule
        let parent = ParentContext::root();
        let result = PolySubModpRule.apply(&mut ctx, sub_expr, &parent);

        assert!(
            result.is_some(),
            "Rule should fire for identical held polynomials"
        );
        let rewrite = result.unwrap();

        // Should be 0
        if let Expr::Number(n) = ctx.get(rewrite.new_expr) {
            assert!(n.is_zero(), "Result should be 0");
        } else {
            panic!("Expected number 0");
        }
    }

    #[test]
    fn test_ignores_non_hold_operands() {
        let mut ctx = Context::new();

        // Just x - y (no __hold)
        let x = ctx.var("x");
        let y = ctx.var("y");
        let sub_expr = ctx.add(Expr::Sub(x, y));

        let parent = ParentContext::root();
        let result = PolySubModpRule.apply(&mut ctx, sub_expr, &parent);

        assert!(
            result.is_none(),
            "Rule should NOT fire for non-hold operands"
        );
    }
}
