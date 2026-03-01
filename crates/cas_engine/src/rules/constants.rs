//! Algebraic constant rules - simplification rules for mathematical constants like φ (phi)
//!
//! φ (phi) is the golden ratio, defined as (1+√5)/2, and satisfies:
//! - φ² = φ + 1 (characteristic equation)
//! - 1/φ = φ - 1 (reciprocal identity)
//!
//! These rules normalize φ expressions to simpler forms.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::constants_support::{
    try_rewrite_phi_reciprocal_expr, try_rewrite_phi_squared_expr, try_rewrite_recognize_phi_expr,
};

#[cfg(test)]
use cas_ast::{Constant, Expr};

// Rule 1: Recognize (1 + √5)/2 as φ
// Matches both:
// - Div(Add(1, Sqrt(5)), 2)
// - Mul(1/2, Add(1, Sqrt(5)))
define_rule!(
    RecognizePhiRule,
    "Recognize Phi",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_recognize_phi_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// Rule 2: φ² → φ + 1 (from characteristic equation φ² - φ - 1 = 0)
define_rule!(
    PhiSquaredRule,
    "Phi Squared",
    importance: crate::step::ImportanceLevel::Medium,
    |ctx, expr| {
        let rewrite = try_rewrite_phi_squared_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// Rule 3: 1/φ → φ - 1 (reciprocal identity)
define_rule!(
    PhiReciprocalRule,
    "Phi Reciprocal",
    importance: crate::step::ImportanceLevel::Medium,
    |ctx, expr| {
        let rewrite = try_rewrite_phi_reciprocal_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// Note: is_one checks now route directly to cas_math::expr_predicates::is_one_expr.

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(RecognizePhiRule));
    simplifier.add_rule(Box::new(PhiSquaredRule));
    simplifier.add_rule(Box::new(PhiReciprocalRule));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;

    #[test]
    fn test_recognize_phi_div_form() {
        // (1 + sqrt(5)) / 2 -> phi
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let five = ctx.num(5);
        let half_exp = ctx.rational(1, 2);
        let sqrt5 = ctx.add(Expr::Pow(five, half_exp));
        let sum = ctx.add(Expr::Add(one, sqrt5));
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Div(sum, two));

        let rule = RecognizePhiRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(rewrite.is_some(), "Should recognize (1+√5)/2 as phi");
        assert!(matches!(
            ctx.get(rewrite.unwrap().new_expr),
            Expr::Constant(Constant::Phi)
        ));
    }

    #[test]
    fn test_recognize_phi_mul_form() {
        // 1/2 * (1 + sqrt(5)) -> phi
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let five = ctx.num(5);
        let half_exp = ctx.rational(1, 2);
        let sqrt5 = ctx.add(Expr::Pow(five, half_exp));
        let sum = ctx.add(Expr::Add(one, sqrt5));
        let half = ctx.rational(1, 2);
        let expr = ctx.add(Expr::Mul(half, sum));

        let rule = RecognizePhiRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(rewrite.is_some(), "Should recognize 1/2*(1+√5) as phi");
    }

    #[test]
    fn test_phi_squared() {
        // phi^2 -> phi + 1
        let mut ctx = Context::new();
        let phi = ctx.add(Expr::Constant(Constant::Phi));
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Pow(phi, two));

        let rule = PhiSquaredRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(rewrite.is_some(), "Should simplify phi^2");
        let result = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.unwrap().new_expr
            }
        );
        assert!(
            result.contains("phi") && result.contains("1"),
            "Result should be phi + 1: {}",
            result
        );
    }

    #[test]
    fn test_phi_reciprocal() {
        // 1/phi -> phi - 1
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let phi = ctx.add(Expr::Constant(Constant::Phi));
        let expr = ctx.add(Expr::Div(one, phi));

        let rule = PhiReciprocalRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(rewrite.is_some(), "Should simplify 1/phi");
        let result = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.unwrap().new_expr
            }
        );
        assert!(
            result.contains("phi"),
            "Result should contain phi: {}",
            result
        );
    }

    #[test]
    fn test_phi_stays_phi() {
        // phi should remain phi (no inverse rule)
        let mut ctx = Context::new();
        let phi = ctx.add(Expr::Constant(Constant::Phi));

        let rule1 = RecognizePhiRule;
        let rule2 = PhiSquaredRule;
        let rule3 = PhiReciprocalRule;

        assert!(rule1
            .apply(&mut ctx, phi, &crate::parent_context::ParentContext::root())
            .is_none());
        assert!(rule2
            .apply(&mut ctx, phi, &crate::parent_context::ParentContext::root())
            .is_none());
        assert!(rule3
            .apply(&mut ctx, phi, &crate::parent_context::ParentContext::root())
            .is_none());
    }
}
