//! Infinity arithmetic rules for the extended real line ℝ ∪ {+∞, −∞}.
//!
//! Implements safe, conservative rules for infinity operations.
//! Only collapses when all non-infinity terms are "finite literal" (numbers, known constants).
//!
//! # Covered operations
//! - `finite + ∞ → ∞` (absorption)
//! - `finite / ∞ → 0`
//! - `∞ + (-∞) → Undefined` (indeterminate)
//! - `0 · ∞ → Undefined` (indeterminate)

use crate::rule::Rewrite;
use cas_ast::{Constant, Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;

// ============================================================
// HELPERS
// ============================================================

/// Sign of infinity: positive (+∞) or negative (−∞).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InfSign {
    Pos,
    Neg,
}

/// Detect if an expression is ±∞.
///
/// Recognizes:
/// - `Constant::Infinity` → `InfSign::Pos`
/// - `Neg(Constant::Infinity)` → `InfSign::Neg`
pub fn inf_sign(ctx: &Context, id: ExprId) -> Option<InfSign> {
    match ctx.get(id) {
        Expr::Constant(Constant::Infinity) => Some(InfSign::Pos),
        Expr::Neg(inner) => match ctx.get(*inner) {
            Expr::Constant(Constant::Infinity) => Some(InfSign::Neg),
            _ => None,
        },
        _ => None,
    }
}

/// Construct ±∞.
pub fn mk_infinity(ctx: &mut Context, sign: InfSign) -> ExprId {
    let inf = ctx.add(Expr::Constant(Constant::Infinity));
    match sign {
        InfSign::Pos => inf,
        InfSign::Neg => ctx.add(Expr::Neg(inf)),
    }
}

/// Construct Undefined (for indeterminate forms).
pub fn mk_undefined(ctx: &mut Context) -> ExprId {
    ctx.add(Expr::Constant(Constant::Undefined))
}

/// Check if an expression is a "finite literal".
///
/// Conservative policy: only true for expressions we KNOW are finite:
/// - Numbers (BigRational)
/// - Constants that are not Infinity or Undefined (π, e, i)
///
/// This prevents unsound simplifications like `f(x) + ∞ → ∞` where f(x) might be −∞.
pub fn is_finite_literal(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(_) => true,
        Expr::Constant(c) => !matches!(c, Constant::Infinity | Constant::Undefined),
        _ => false,
    }
}

/// Check if expression is zero - uses the canonical helper.
fn is_zero(ctx: &Context, id: ExprId) -> bool {
    crate::helpers::is_zero(ctx, id)
}

/// Create zero as ExprId.
fn mk_zero(ctx: &mut Context) -> ExprId {
    ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(0))))
}

// ============================================================
// RULES
// ============================================================

/// Rule: Infinity absorption in addition.
///
/// - `finite + ∞ → ∞`
/// - `finite + (-∞) → -∞`
/// - `∞ + (-∞) → Undefined` (indeterminate)
///
/// Only applies when ALL non-infinity terms are finite literals (conservative).
pub fn add_infinity_absorption(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    // Collect all additive terms
    let mut terms = Vec::new();
    collect_add_terms_with_sign(ctx, expr, true, &mut terms);

    let mut has_pos_inf = false;
    let mut has_neg_inf = false;

    for &(term, is_positive) in &terms {
        if let Some(s) = inf_sign(ctx, term) {
            // Combine term sign with infinity sign
            let total = match (is_positive, s) {
                (true, InfSign::Pos) => InfSign::Pos,
                (false, InfSign::Pos) => InfSign::Neg,
                (true, InfSign::Neg) => InfSign::Neg,
                (false, InfSign::Neg) => InfSign::Pos,
            };
            match total {
                InfSign::Pos => has_pos_inf = true,
                InfSign::Neg => has_neg_inf = true,
            }
        } else {
            // Conservative: only absorb if all other terms are finite literals
            if !is_finite_literal(ctx, term) {
                return None;
            }
        }
    }

    let (new_expr, description) = match (has_pos_inf, has_neg_inf) {
        (true, true) => (mk_undefined(ctx), "∞ + (-∞) is indeterminate".to_string()),
        (true, false) => (mk_infinity(ctx, InfSign::Pos), "finite + ∞ → ∞".to_string()),
        (false, true) => (
            mk_infinity(ctx, InfSign::Neg),
            "finite + (-∞) → -∞".to_string(),
        ),
        (false, false) => return None,
    };

    Some(Rewrite {
        new_expr,
        description,
        before_local: None,
        after_local: None,
        domain_assumption: None,
    })
}

/// Collect additive terms with their signs (iterative, handles Sub).
fn collect_add_terms_with_sign(
    ctx: &Context,
    id: ExprId,
    is_positive: bool,
    terms: &mut Vec<(ExprId, bool)>,
) {
    let mut stack = vec![(id, is_positive)];

    while let Some((current, sign)) = stack.pop() {
        match ctx.get(current) {
            Expr::Add(l, r) => {
                stack.push((*r, sign));
                stack.push((*l, sign));
            }
            Expr::Sub(l, r) => {
                stack.push((*r, !sign)); // Right side gets inverted sign
                stack.push((*l, sign));
            }
            Expr::Neg(inner) => {
                stack.push((*inner, !sign));
            }
            _ => terms.push((current, sign)),
        }
    }
}

/// Rule: Division by infinity.
///
/// `finite / ∞ → 0`
/// `finite / (-∞) → 0`
///
/// Only applies when numerator is a finite literal.
pub fn div_by_infinity(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    if inf_sign(ctx, den).is_some() && is_finite_literal(ctx, num) {
        return Some(Rewrite {
            new_expr: mk_zero(ctx),
            description: "finite / ∞ → 0".to_string(),
            before_local: None,
            after_local: None,
            domain_assumption: None,
        });
    }
    None
}

/// Rule: Zero times infinity is indeterminate.
///
/// `0 · ∞ → Undefined`
/// `∞ · 0 → Undefined`
pub fn mul_zero_infinity(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let Expr::Mul(a, b) = ctx.get(expr).clone() else {
        return None;
    };

    let a_inf = inf_sign(ctx, a).is_some();
    let b_inf = inf_sign(ctx, b).is_some();

    let a_zero = is_zero(ctx, a);
    let b_zero = is_zero(ctx, b);

    if (a_zero && b_inf) || (b_zero && a_inf) {
        return Some(Rewrite {
            new_expr: mk_undefined(ctx),
            description: "0 · ∞ is indeterminate".to_string(),
            before_local: None,
            after_local: None,
            domain_assumption: None,
        });
    }
    None
}

// ============================================================
// RULE STRUCTS (for pipeline registration)
// ============================================================

use crate::define_rule;

define_rule!(
    AddInfinityRule,
    "Infinity Absorption in Addition",
    |ctx, expr| { add_infinity_absorption(ctx, expr) }
);

define_rule!(DivByInfinityRule, "Division by Infinity", |ctx, expr| {
    div_by_infinity(ctx, expr)
});

define_rule!(
    MulZeroInfinityRule,
    "Zero Times Infinity Indeterminate",
    |ctx, expr| { mul_zero_infinity(ctx, expr) }
);

/// Register infinity arithmetic rules with the simplifier.
///
/// These rules should be registered early in the pipeline (with CORE rules)
/// to handle infinity operations before other simplifications.
pub fn register(simplifier: &mut crate::Simplifier) {
    // Indeterminate forms first (highest priority)
    simplifier.add_rule(Box::new(MulZeroInfinityRule));
    // Then absorption rules
    simplifier.add_rule(Box::new(AddInfinityRule));
    simplifier.add_rule(Box::new(DivByInfinityRule));
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;
    use num_traits::Zero;

    fn parse_expr(ctx: &mut Context, s: &str) -> ExprId {
        parse(s, ctx).expect("parse failed")
    }

    #[allow(dead_code)]
    fn simplify_once(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
        if let Some(r) = add_infinity_absorption(ctx, expr) {
            return Some(r.new_expr);
        }
        if let Some(r) = div_by_infinity(ctx, expr) {
            return Some(r.new_expr);
        }
        if let Some(r) = mul_zero_infinity(ctx, expr) {
            return Some(r.new_expr);
        }
        None
    }

    // ========== REGRESSION TESTS ==========

    /// Test 118: Infinity addition absorption
    /// 1000 + infinity → infinity
    #[test]
    fn test_118_infinity_addition() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "1000 + infinity");

        let result = add_infinity_absorption(&mut ctx, expr);
        assert!(result.is_some(), "Test 118: Should absorb 1000 + ∞ → ∞");

        let new_expr = result.unwrap().new_expr;
        assert!(
            matches!(ctx.get(new_expr), Expr::Constant(Constant::Infinity)),
            "Test 118: Result should be infinity"
        );
    }

    /// Test 119: Division by infinity
    /// 5 / infinity → 0
    #[test]
    fn test_119_infinity_division() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "5 / infinity");

        let result = div_by_infinity(&mut ctx, expr);
        assert!(result.is_some(), "Test 119: Should simplify 5/∞ → 0");

        let new_expr = result.unwrap().new_expr;
        if let Expr::Number(n) = ctx.get(new_expr) {
            assert!(n.is_zero(), "Test 119: Result should be 0");
        } else {
            panic!("Test 119: Result should be Number(0)");
        }
    }

    /// Test: Infinity plus negative infinity is indeterminate
    /// infinity + -infinity → undefined
    #[test]
    fn test_infinity_plus_neg_infinity_indeterminate() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "infinity + -infinity");

        let result = add_infinity_absorption(&mut ctx, expr);
        assert!(result.is_some(), "Should detect ∞ + (-∞) indeterminate");

        let new_expr = result.unwrap().new_expr;
        assert!(
            matches!(ctx.get(new_expr), Expr::Constant(Constant::Undefined)),
            "Result should be Undefined"
        );
    }

    /// Test: Infinity minus infinity is indeterminate
    /// infinity - infinity → undefined
    #[test]
    fn test_infinity_minus_infinity_indeterminate() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "infinity - infinity");

        let result = add_infinity_absorption(&mut ctx, expr);
        assert!(result.is_some(), "Should detect ∞ - ∞ indeterminate");

        let new_expr = result.unwrap().new_expr;
        assert!(
            matches!(ctx.get(new_expr), Expr::Constant(Constant::Undefined)),
            "Result should be Undefined"
        );
    }

    /// Test: Zero times infinity is indeterminate
    /// 0 * infinity → undefined
    #[test]
    fn test_zero_times_infinity_indeterminate() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "0 * infinity");

        let result = mul_zero_infinity(&mut ctx, expr);
        assert!(result.is_some(), "Should detect 0·∞ indeterminate");

        let new_expr = result.unwrap().new_expr;
        assert!(
            matches!(ctx.get(new_expr), Expr::Constant(Constant::Undefined)),
            "Result should be Undefined"
        );
    }

    /// Test: Infinity times zero (commutative)
    /// infinity * 0 → undefined
    #[test]
    fn test_infinity_times_zero_indeterminate() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "infinity * 0");

        let result = mul_zero_infinity(&mut ctx, expr);
        assert!(result.is_some(), "Should detect ∞·0 indeterminate");

        let new_expr = result.unwrap().new_expr;
        assert!(
            matches!(ctx.get(new_expr), Expr::Constant(Constant::Undefined)),
            "Result should be Undefined"
        );
    }

    /// Test: Addition with negative infinity
    /// 1000 + -infinity → -infinity
    #[test]
    fn test_addition_with_negative_infinity() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "1000 + -infinity");

        let result = add_infinity_absorption(&mut ctx, expr);
        assert!(result.is_some(), "Should absorb 1000 + (-∞) → -∞");

        let new_expr = result.unwrap().new_expr;
        // Should be Neg(Infinity)
        if let Expr::Neg(inner) = ctx.get(new_expr) {
            assert!(
                matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity)),
                "Result should be -infinity"
            );
        } else {
            panic!("Result should be Neg(Infinity)");
        }
    }

    /// Test: Division by negative infinity also gives 0
    /// 5 / -infinity → 0
    #[test]
    fn test_division_by_negative_infinity() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "5 / -infinity");

        let result = div_by_infinity(&mut ctx, expr);
        assert!(result.is_some(), "Should simplify 5/(-∞) → 0");

        let new_expr = result.unwrap().new_expr;
        if let Expr::Number(n) = ctx.get(new_expr) {
            assert!(n.is_zero(), "Result should be 0");
        } else {
            panic!("Result should be Number(0)");
        }
    }

    /// Test: Conservative policy - variable + infinity should NOT simplify
    #[test]
    fn test_conservative_variable_plus_infinity() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x + infinity");

        let result = add_infinity_absorption(&mut ctx, expr);
        assert!(
            result.is_none(),
            "Conservative: x + ∞ should NOT simplify (x could be -∞)"
        );
    }
}
