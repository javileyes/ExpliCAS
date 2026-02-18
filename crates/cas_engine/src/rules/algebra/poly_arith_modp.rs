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
use crate::poly_modp_conv::{
    expr_to_poly_modp_with_store as expr_to_poly_modp, strip_hold, PolyModpBudget, VarTable,
};
use crate::rule::Rewrite;
use crate::rules::algebra::gcd_modp::DEFAULT_PRIME;
use cas_ast::{BuiltinFn, Expr};

/// Check if expression is wrapped in __hold
fn is_hold(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Hold) && args.len() == 1)
}

// PolySubModpRule: handle __hold(P) - __hold(Q) in polynomial domain
define_rule!(
    PolySubModpRule,
    "Polynomial Subtraction (mod p)",
    Some(crate::target_kind::TargetKindSet::SUB),
    PhaseMask::CORE | PhaseMask::POST,
    |ctx, expr| {
        let (a, b) = match ctx.get(expr) {
            Expr::Sub(a, b) => (*a, *b),
            _ => return None,
        };

        // Only activate if at least one operand is __hold
        // This avoids contaminating standard arithmetic
        let a_is_hold = is_hold(ctx, a);
        let b_is_hold = is_hold(ctx, b);

        if !(a_is_hold || b_is_hold) {
            return None;
        }

        // Use default budget and prime for mod-p operations
        let budget = PolyModpBudget::default();
        let p = DEFAULT_PRIME;
        let mut vars = VarTable::new();

        // Strip __hold wrappers before conversion
        let a_inner = strip_hold(ctx, a);
        let b_inner = strip_hold(ctx, b);

        // Convert both to MultiPolyModP
        let mut pa = expr_to_poly_modp(ctx, a_inner, p, &budget, &mut vars).ok()?;
        let mut pb = expr_to_poly_modp(ctx, b_inner, p, &budget, &mut vars).ok()?;

        // Normalize both to monic for comparison
        // (poly_gcd returns monic, but expand(g) may not be)
        pa.make_monic();
        pb.make_monic();

        // Check if polynomials are equal (after normalization)
        if pa == pb {
            // P - P = 0 (up to scalar multiple)
            let zero = ctx.num(0);
            return Some(
                Rewrite::new(zero)
                    .desc("__hold(P) - __hold(Q) = 0 (equal polynomials mod p, up to scalar)"),
            );
        }

        // If not equal, skip (conservative behavior)
        None
    }
);

/// Register polynomial arithmetic rules
pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(PolySubModpRule));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parent_context::ParentContext;
    use crate::rule::Rule;
    use cas_ast::Context;
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
