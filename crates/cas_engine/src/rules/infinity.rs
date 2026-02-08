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

// ============================================================
// FINITENESS CLASSIFICATION (for limit support)
// ============================================================

/// Classification of an expression's finiteness.
///
/// This is a conservative classification:
/// - `FiniteLiteral`: We KNOW the expression is a finite value
/// - `Infinity(sign)`: We KNOW the expression is ±∞
/// - `Unknown`: Expression could be finite, infinite, or undefined
///
/// Example usage for limits:
/// ```ignore
/// match classify_finiteness(ctx, expr) {
///     Finiteness::FiniteLiteral => { /* can safely absorb into infinity */ },
///     Finiteness::Infinity(sign) => { /* propagate infinity with sign */ },
///     Finiteness::Unknown => { /* cannot simplify - may need limit rules */ },
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finiteness {
    /// Expression is a known finite value (number, π, e, i)
    FiniteLiteral,
    /// Expression is ±∞ with known sign
    Infinity(InfSign),
    /// Unknown finiteness - could be finite, infinite, or undefined
    /// This includes: variables, functions, complex expressions
    Unknown,
}

/// Classify an expression's finiteness.
///
/// Returns:
/// - `FiniteLiteral`: Numbers and finite constants (π, e, i)
/// - `Infinity(Pos)`: Expression is +∞
/// - `Infinity(Neg)`: Expression is -∞
/// - `Unknown`: Variables, functions, or complex expressions
///
/// Note: `Undefined` is classified as `Unknown` since it represents
/// an indeterminate form, not a known value.
pub fn classify_finiteness(ctx: &Context, id: ExprId) -> Finiteness {
    // Check for infinity first
    if let Some(sign) = inf_sign(ctx, id) {
        return Finiteness::Infinity(sign);
    }

    // Check for finite literals
    if is_finite_literal(ctx, id) {
        return Finiteness::FiniteLiteral;
    }

    // Everything else is unknown (conservative)
    Finiteness::Unknown
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

    Some(Rewrite::new(new_expr).desc(description))
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
    let (num, den) = if let Expr::Div(num, den) = ctx.get(expr) {
        (*num, *den)
    } else {
        return None;
    };

    if inf_sign(ctx, den).is_some() && is_finite_literal(ctx, num) {
        return Some(Rewrite::new(mk_zero(ctx)).desc("finite / ∞ → 0"));
    }
    None
}

/// Rule: Zero times infinity is indeterminate.
///
/// `0 · ∞ → Undefined`
/// `∞ · 0 → Undefined`
pub fn mul_zero_infinity(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let (a, b) = if let Expr::Mul(a, b) = ctx.get(expr) {
        (*a, *b)
    } else {
        return None;
    };

    let a_inf = inf_sign(ctx, a).is_some();
    let b_inf = inf_sign(ctx, b).is_some();

    let a_zero = is_zero(ctx, a);
    let b_zero = is_zero(ctx, b);

    if (a_zero && b_inf) || (b_zero && a_inf) {
        return Some(Rewrite::new(mk_undefined(ctx)).desc("0 · ∞ is indeterminate"));
    }
    None
}

/// Check if expression is a positive finite literal (for sign determination).
fn is_positive_literal(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(n) => {
            use num_traits::Signed;
            n.is_positive()
        }
        Expr::Constant(c) => matches!(c, Constant::Pi | Constant::E), // π and e are positive
        _ => false,
    }
}

/// Check if expression is a negative finite literal.
fn is_negative_literal(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(n) => {
            use num_traits::Signed;
            n.is_negative()
        }
        Expr::Neg(inner) => is_positive_literal(ctx, *inner),
        _ => false,
    }
}

/// Rule: Finite (non-zero) times infinity.
///
/// `finite * ∞ → ±∞` (sign depends on finite's sign)
/// - `3 * infinity → infinity`
/// - `(-2) * infinity → -infinity`
/// - `x * infinity → no simplification` (conservative)
pub fn mul_finite_infinity(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let (a, b) = if let Expr::Mul(a, b) = ctx.get(expr) {
        (*a, *b)
    } else {
        return None;
    };

    // Case 1: a is inf, b is finite non-zero literal
    if let Some(inf_s) = inf_sign(ctx, a) {
        if is_finite_literal(ctx, b) && !is_zero(ctx, b) {
            let b_negative = is_negative_literal(ctx, b);
            let result_sign = if b_negative {
                match inf_s {
                    InfSign::Pos => InfSign::Neg,
                    InfSign::Neg => InfSign::Pos,
                }
            } else {
                inf_s
            };
            return Some(
                Rewrite::new(mk_infinity(ctx, result_sign))
                    .desc_lazy(|| format!("finite * ∞ → {:?}∞", result_sign)),
            );
        }
    }

    // Case 2: b is inf, a is finite non-zero literal
    if let Some(inf_s) = inf_sign(ctx, b) {
        if is_finite_literal(ctx, a) && !is_zero(ctx, a) {
            let a_negative = is_negative_literal(ctx, a);
            let result_sign = if a_negative {
                match inf_s {
                    InfSign::Pos => InfSign::Neg,
                    InfSign::Neg => InfSign::Pos,
                }
            } else {
                inf_s
            };
            return Some(
                Rewrite::new(mk_infinity(ctx, result_sign))
                    .desc_lazy(|| format!("finite * ∞ → {:?}∞", result_sign)),
            );
        }
    }

    None
}

/// Rule: Infinity divided by finite (non-zero).
///
/// `∞ / finite → ±∞` (sign depends on finite's sign)
/// - `infinity / 2 → infinity`
/// - `infinity / (-3) → -infinity`
/// - `-infinity / 2 → -infinity`
pub fn inf_div_finite(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let (num, den) = if let Expr::Div(num, den) = ctx.get(expr) {
        (*num, *den)
    } else {
        return None;
    };

    let inf_s = inf_sign(ctx, num)?;

    // Denominator must be finite non-zero literal
    if !is_finite_literal(ctx, den) || is_zero(ctx, den) {
        return None;
    }

    let den_negative = is_negative_literal(ctx, den);
    let result_sign = if den_negative {
        match inf_s {
            InfSign::Pos => InfSign::Neg,
            InfSign::Neg => InfSign::Pos,
        }
    } else {
        inf_s
    };

    Some(
        Rewrite::new(mk_infinity(ctx, result_sign))
            .desc_lazy(|| format!("∞ / finite → {:?}∞", result_sign)),
    )
}

// ============================================================
// RULE STRUCTS (for pipeline registration)
// ============================================================

use crate::define_rule;

define_rule!(
    AddInfinityRule,
    "Infinity Absorption in Addition",
    Some(crate::target_kind::TargetKindSet::ADD),
    |ctx, expr| { add_infinity_absorption(ctx, expr) }
);

define_rule!(
    DivByInfinityRule,
    "Division by Infinity",
    Some(crate::target_kind::TargetKindSet::DIV),
    |ctx, expr| { div_by_infinity(ctx, expr) }
);

define_rule!(
    MulZeroInfinityRule,
    "Zero Times Infinity Indeterminate",
    Some(crate::target_kind::TargetKindSet::MUL),
    |ctx, expr| { mul_zero_infinity(ctx, expr) }
);

define_rule!(
    MulInfinityRule,
    "Finite Times Infinity",
    Some(crate::target_kind::TargetKindSet::MUL),
    |ctx, expr| { mul_finite_infinity(ctx, expr) }
);

define_rule!(
    InfDivFiniteRule,
    "Infinity Divided by Finite",
    Some(crate::target_kind::TargetKindSet::DIV),
    |ctx, expr| { inf_div_finite(ctx, expr) }
);

/// Register infinity arithmetic rules with the simplifier.
///
/// These rules should be registered early in the pipeline (with CORE rules)
/// to handle infinity operations before other simplifications.
pub fn register(simplifier: &mut crate::Simplifier) {
    // Indeterminate forms first (highest priority)
    simplifier.add_rule(Box::new(MulZeroInfinityRule));
    // Then absorption/computation rules
    simplifier.add_rule(Box::new(MulInfinityRule));
    simplifier.add_rule(Box::new(AddInfinityRule));
    simplifier.add_rule(Box::new(DivByInfinityRule));
    simplifier.add_rule(Box::new(InfDivFiniteRule));
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

    // ========== EXTENDED RULES TESTS ==========

    /// Test: Finite times infinity
    /// 3 * infinity → infinity
    #[test]
    fn test_finite_times_infinity() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "3 * infinity");

        let result = mul_finite_infinity(&mut ctx, expr);
        assert!(result.is_some(), "Should simplify 3 * ∞ → ∞");

        let new_expr = result.unwrap().new_expr;
        assert!(
            matches!(ctx.get(new_expr), Expr::Constant(Constant::Infinity)),
            "Result should be infinity"
        );
    }

    /// Test: Negative finite times infinity
    /// (-2) * infinity → -infinity
    #[test]
    fn test_negative_finite_times_infinity() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "(-2) * infinity");

        let result = mul_finite_infinity(&mut ctx, expr);
        assert!(result.is_some(), "Should simplify (-2) * ∞ → -∞");

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

    /// Test: Infinity times finite (commutative)
    /// infinity * 5 → infinity
    #[test]
    fn test_infinity_times_finite() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "infinity * 5");

        let result = mul_finite_infinity(&mut ctx, expr);
        assert!(result.is_some(), "Should simplify ∞ * 5 → ∞");

        let new_expr = result.unwrap().new_expr;
        assert!(
            matches!(ctx.get(new_expr), Expr::Constant(Constant::Infinity)),
            "Result should be infinity"
        );
    }

    /// Test: Infinity divided by finite
    /// infinity / 2 → infinity
    #[test]
    fn test_infinity_div_finite() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "infinity / 2");

        let result = inf_div_finite(&mut ctx, expr);
        assert!(result.is_some(), "Should simplify ∞ / 2 → ∞");

        let new_expr = result.unwrap().new_expr;
        assert!(
            matches!(ctx.get(new_expr), Expr::Constant(Constant::Infinity)),
            "Result should be infinity"
        );
    }

    /// Test: Infinity divided by negative finite
    /// infinity / (-3) → -infinity
    #[test]
    fn test_infinity_div_negative_finite() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "infinity / (-3)");

        let result = inf_div_finite(&mut ctx, expr);
        assert!(result.is_some(), "Should simplify ∞ / (-3) → -∞");

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

    /// Test: Negative infinity divided by negative finite
    /// -infinity / (-2) → infinity (negative * negative = positive)
    #[test]
    fn test_neg_infinity_div_neg_finite() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "-infinity / (-2)");

        let result = inf_div_finite(&mut ctx, expr);
        assert!(result.is_some(), "Should simplify -∞ / (-2) → ∞");

        let new_expr = result.unwrap().new_expr;
        assert!(
            matches!(ctx.get(new_expr), Expr::Constant(Constant::Infinity)),
            "Result should be positive infinity"
        );
    }

    /// Test: Symmetric - (-infinity) - (-infinity) is indeterminate
    #[test]
    fn test_neg_inf_minus_neg_inf_indeterminate() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "-infinity - (-infinity)");

        let result = add_infinity_absorption(&mut ctx, expr);
        assert!(result.is_some(), "Should detect -∞ - (-∞) indeterminate");

        let new_expr = result.unwrap().new_expr;
        assert!(
            matches!(ctx.get(new_expr), Expr::Constant(Constant::Undefined)),
            "Result should be Undefined"
        );
    }

    /// Test: Conservative policy - variable * infinity should NOT simplify
    #[test]
    fn test_conservative_variable_times_infinity() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x * infinity");

        let result = mul_finite_infinity(&mut ctx, expr);
        assert!(result.is_none(), "Conservative: x * ∞ should NOT simplify");
    }

    // ========== CLASSIFY_FINITENESS TESTS ==========

    /// Test: classify_finiteness for numbers
    #[test]
    fn test_classify_number_is_finite() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "42");

        assert_eq!(
            classify_finiteness(&ctx, expr),
            Finiteness::FiniteLiteral,
            "Numbers should be FiniteLiteral"
        );
    }

    /// Test: classify_finiteness for pi
    #[test]
    fn test_classify_pi_is_finite() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "pi");

        assert_eq!(
            classify_finiteness(&ctx, expr),
            Finiteness::FiniteLiteral,
            "Pi should be FiniteLiteral"
        );
    }

    /// Test: classify_finiteness for infinity
    #[test]
    fn test_classify_infinity() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "infinity");

        assert_eq!(
            classify_finiteness(&ctx, expr),
            Finiteness::Infinity(InfSign::Pos),
            "infinity should be Infinity(Pos)"
        );
    }

    /// Test: classify_finiteness for negative infinity
    #[test]
    fn test_classify_neg_infinity() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "-infinity");

        assert_eq!(
            classify_finiteness(&ctx, expr),
            Finiteness::Infinity(InfSign::Neg),
            "-infinity should be Infinity(Neg)"
        );
    }

    /// Test: classify_finiteness for variables is Unknown
    #[test]
    fn test_classify_variable_is_unknown() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x");

        assert_eq!(
            classify_finiteness(&ctx, expr),
            Finiteness::Unknown,
            "Variables should be Unknown"
        );
    }

    /// Test: classify_finiteness for undefined is Unknown
    #[test]
    fn test_classify_undefined_is_unknown() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "undefined");

        assert_eq!(
            classify_finiteness(&ctx, expr),
            Finiteness::Unknown,
            "Undefined should be Unknown (not a known value)"
        );
    }
}
