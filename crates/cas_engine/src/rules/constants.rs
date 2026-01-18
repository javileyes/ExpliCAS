//! Algebraic constant rules - simplification rules for mathematical constants like φ (phi)
//!
//! φ (phi) is the golden ratio, defined as (1+√5)/2, and satisfies:
//! - φ² = φ + 1 (characteristic equation)
//! - 1/φ = φ - 1 (reciprocal identity)
//!
//! These rules normalize φ expressions to simpler forms.

use crate::define_rule;
use crate::rule::Rewrite;
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
        // Pattern 1: Div(Add(1, Pow(5, 1/2)), 2)
        if let Expr::Div(num, den) = ctx.get(expr) {
            // Check denominator is 2
            if let Expr::Number(d) = ctx.get(*den) {
                if d != &num_rational::BigRational::from_integer(2.into()) {
                    return None;
                }
            } else {
                return None;
            }

            // Check numerator is Add(1, sqrt(5)) or Add(sqrt(5), 1)
            if let Expr::Add(l, r) = ctx.get(*num) {
                let (one_id, sqrt5_id) = if crate::helpers::is_one(ctx, *l) && is_sqrt5(ctx, *r) {
                    (*l, *r)
                } else if is_sqrt5(ctx, *l) && crate::helpers::is_one(ctx, *r) {
                    (*r, *l)
                } else {
                    return None;
                };
                let _ = (one_id, sqrt5_id); // suppress unused warning

                let phi = ctx.add(Expr::Constant(Constant::Phi));
                return Some(Rewrite::new(phi).desc("(1 + √5)/2 = φ"));
            }
        }

        // Pattern 2: Mul(1/2, Add(1, sqrt(5))) or Mul(Add(1, sqrt(5)), 1/2)
        if let Expr::Mul(l, r) = ctx.get(expr) {
            let (half_id, sum_id) = if is_half(ctx, *l) {
                (*l, *r)
            } else if is_half(ctx, *r) {
                (*r, *l)
            } else {
                return None;
            };
            let _ = half_id; // suppress unused warning

            if let Expr::Add(a, b) = ctx.get(sum_id) {
                if (crate::helpers::is_one(ctx, *a) && is_sqrt5(ctx, *b)) || (is_sqrt5(ctx, *a) && crate::helpers::is_one(ctx, *b))
                {
                    let phi = ctx.add(Expr::Constant(Constant::Phi));
                    return Some(Rewrite::new(phi).desc("(1 + √5)/2 = φ"));
                }
            }
        }

        None
    }
);

// Rule 2: φ² → φ + 1 (from characteristic equation φ² - φ - 1 = 0)
define_rule!(
    PhiSquaredRule,
    "Phi Squared",
    importance: crate::step::ImportanceLevel::Medium,
    |ctx, expr| {
        if let Expr::Pow(base, exp) = ctx.get(expr) {
            // Check base is φ
            if !matches!(ctx.get(*base), Expr::Constant(Constant::Phi)) {
                return None;
            }

            // Check exponent is 2
            if let Expr::Number(n) = ctx.get(*exp) {
                if n == &num_rational::BigRational::from_integer(2.into()) {
                    // φ² = φ + 1
                    let phi = ctx.add(Expr::Constant(Constant::Phi));
                    let one = ctx.num(1);
                    let result = ctx.add(Expr::Add(phi, one));
                    return Some(Rewrite::new(result).desc("φ² = φ + 1"));
                }
            }
        }
        None
    }
);

// Rule 3: 1/φ → φ - 1 (reciprocal identity)
define_rule!(
    PhiReciprocalRule,
    "Phi Reciprocal",
    importance: crate::step::ImportanceLevel::Medium,
    |ctx, expr| {
        if let Expr::Div(num, den) = ctx.get(expr) {
            // Check numerator is 1
            if !crate::helpers::is_one(ctx, *num) {
                return None;
            }

            // Check denominator is φ
            if !matches!(ctx.get(*den), Expr::Constant(Constant::Phi)) {
                return None;
            }

            // 1/φ = φ - 1
            let phi = ctx.add(Expr::Constant(Constant::Phi));
            let one = ctx.num(1);
            let neg_one = ctx.add(Expr::Neg(one));
            let result = ctx.add(Expr::Add(phi, neg_one));
            return Some(Rewrite::new(result).desc("1/φ = φ - 1"));
        }
        None
    }
);

// Note: is_one uses crate::helpers::is_one (canonical)

// Helper: check if expression is 1/2
fn is_half(ctx: &cas_ast::Context, id: cas_ast::ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(id) {
        n == &num_rational::BigRational::new(1.into(), 2.into())
    } else {
        false
    }
}

// Helper: check if expression is √5 (Pow(5, 1/2) or Function("sqrt", [5]))
fn is_sqrt5(ctx: &cas_ast::Context, id: cas_ast::ExprId) -> bool {
    let five = num_rational::BigRational::from_integer(5.into());
    let half = num_rational::BigRational::new(1.into(), 2.into());

    match ctx.get(id) {
        // Pow(5, 1/2)
        Expr::Pow(base, exp) => {
            if let Expr::Number(b) = ctx.get(*base) {
                if let Expr::Number(e) = ctx.get(*exp) {
                    return b == &five && e == &half;
                }
            }
            false
        }
        // Function("sqrt", [5])
        Expr::Function(name, args) => {
            if name == "sqrt" && args.len() == 1 {
                if let Expr::Number(n) = ctx.get(args[0]) {
                    return n == &five;
                }
            }
            false
        }
        _ => false,
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(RecognizePhiRule));
    simplifier.add_rule(Box::new(PhiSquaredRule));
    simplifier.add_rule(Box::new(PhiReciprocalRule));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::{Context, DisplayExpr};

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
