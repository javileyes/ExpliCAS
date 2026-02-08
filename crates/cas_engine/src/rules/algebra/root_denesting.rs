//! Root denesting rules.
//!
//! Contains `CubicConjugateTrapRule` and its helper functions.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Expr};
use num_traits::Zero;

// =============================================================================
// CUBIC CONJUGATE TRAP RULE
// Simplifies ∛(m+t) + ∛(m-t) when the result is a rational number.
// =============================================================================

/// Try to split an expression as `m ± t` where m is rational and t is a surd.
/// Returns (m, t, sign) where sign = +1 means m+t, sign = -1 means m-t.
/// Handles different orderings from parser canonicalization.
fn split_as_m_plus_t(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, i32)> {
    // Helper to check if an expression contains a sqrt (surd-like)
    fn is_surd_like(ctx: &cas_ast::Context, e: cas_ast::ExprId) -> bool {
        match ctx.get(e) {
            Expr::Function(fn_id, args)
                if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
            {
                true
            }
            Expr::Pow(_, exp) => {
                if let Expr::Number(n) = ctx.get(*exp) {
                    *n.numer() == 1.into() && *n.denom() == 2.into()
                } else {
                    false
                }
            }
            Expr::Mul(l, r) => is_surd_like(ctx, *l) || is_surd_like(ctx, *r),
            Expr::Neg(inner) => is_surd_like(ctx, *inner),
            _ => false,
        }
    }

    fn is_numeric(ctx: &cas_ast::Context, e: cas_ast::ExprId) -> bool {
        matches!(ctx.get(e), Expr::Number(_))
    }

    // Extract terms and effective sign
    // For Add(a, b): sign of t is +1
    // For Sub(a, b): sign of t is -1 (when t is the right operand)
    // For Add(a, Neg(b)): sign of t is -1 (when t is b)
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            // Check for Add(a, Neg(b)) = a - b
            if let Expr::Neg(neg_inner) = ctx.get(*r) {
                // a + (-b) where b = neg_inner
                if is_numeric(ctx, *l) && is_surd_like(ctx, *neg_inner) {
                    // m + (-t) = m - t
                    return Some((*l, *neg_inner, -1));
                }
                if is_surd_like(ctx, *l) && is_numeric(ctx, *neg_inner) {
                    // t + (-m) = t - m = -(m - t) => this represents -(m - t)
                    // Actually we should not match this as it's not (m ± t)
                    return None;
                }
            }

            // Regular Add(a, b)
            if is_numeric(ctx, *l) && is_surd_like(ctx, *r) {
                // m + t
                return Some((*l, *r, 1));
            }
            if is_surd_like(ctx, *l) && is_numeric(ctx, *r) {
                // t + m = m + t (commutative)
                return Some((*r, *l, 1));
            }
            None
        }
        Expr::Sub(l, r) => {
            if is_numeric(ctx, *l) && is_surd_like(ctx, *r) {
                // m - t
                return Some((*l, *r, -1));
            }
            if is_surd_like(ctx, *l) && is_numeric(ctx, *r) {
                // t - m: This is NOT (m ± t), it's -(m - t)
                // We could represent it as m - t with sign flipped externally,
                // but for clarity, we only match when m is first in Sub
                return None;
            }
            None
        }
        _ => None,
    }
}

/// Check if two expressions are conjugates: base1 = m + t, base2 = m - t (or vice versa).
/// Returns Some((m, t)) if they are conjugates.
fn is_conjugate_pair(
    ctx: &cas_ast::Context,
    base1: cas_ast::ExprId,
    base2: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    use crate::ordering::compare_expr;
    use std::cmp::Ordering;

    let (m1, t1, sign1) = split_as_m_plus_t(ctx, base1)?;
    let (m2, t2, sign2) = split_as_m_plus_t(ctx, base2)?;

    // Must have same m and same t
    if compare_expr(ctx, m1, m2) != Ordering::Equal {
        return None;
    }
    if compare_expr(ctx, t1, t2) != Ordering::Equal {
        return None;
    }

    // One must be +1 (addition), one must be -1 (subtraction)
    // i.e., sign1 * sign2 == -1
    if sign1 + sign2 != 0 {
        return None;
    }

    Some((m1, t1))
}

/// Extract exponent from Pow(base, exp) and check if it equals 1/3.
fn is_cube_root_pow(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> Option<cas_ast::ExprId> {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*exp) {
            // Check if exponent is 1/3
            if *n.numer() == 1.into() && *n.denom() == 3.into() {
                return Some(*base);
            }
        }
    }
    None
}

/// Compute cube root of a rational number (real cube root, handles negatives).
/// Returns Some(result) if input is a perfect cube.
fn rational_cbrt(r: &num_rational::BigRational) -> Option<num_rational::BigRational> {
    use num_traits::Signed;

    let neg = r.is_negative();
    let abs_r = if neg { -r.clone() } else { r.clone() };

    if abs_r.is_zero() {
        return Some(num_rational::BigRational::from_integer(0.into()));
    }

    let numer = abs_r.numer().clone();
    let denom = abs_r.denom().clone();

    // Check if numerator is a perfect cube
    let numer_cbrt = numer.cbrt();
    if &numer_cbrt * &numer_cbrt * &numer_cbrt != numer {
        return None;
    }

    // Check if denominator is a perfect cube
    let denom_cbrt = denom.cbrt();
    if &denom_cbrt * &denom_cbrt * &denom_cbrt != denom {
        return None;
    }

    let result = num_rational::BigRational::new(numer_cbrt, denom_cbrt);
    if neg {
        Some(-result)
    } else {
        Some(result)
    }
}

/// Find a rational root of depressed cubic: x³ + px + q = 0
/// Uses Rational Root Theorem correctly for rational coefficients.
/// After clearing denominators: a·x³ + b·x + c = 0
/// Candidates are ±(divisors of |c|) / (divisors of |a|)
fn find_rational_root_depressed_cubic(
    p: &num_rational::BigRational,
    q: &num_rational::BigRational,
) -> Option<num_rational::BigRational> {
    use num_bigint::BigInt;
    use num_traits::{Signed, Zero};

    if q.is_zero() {
        // x³ + px = 0 => x(x² + p) = 0 => x = 0 is always a root
        return Some(num_rational::BigRational::zero());
    }

    // Clear denominators: multiply by LCM of all denominators
    // x³ + (p_n/p_d)x + (q_n/q_d) = 0
    // Multiply by LCM(p_d, q_d): LCM·x³ + (p_n·...)*x + (q_n·...) = 0
    let lcm_denom = num_integer::lcm(p.denom().clone(), q.denom().clone());

    // After clearing, we have: L·x³ + P'·x + Q' = 0
    // where L = lcm_denom, P' = p * L, Q' = q * L
    let leading_coef = lcm_denom.clone(); // coefficient of x³
    let constant_coef = q * num_rational::BigRational::from_integer(lcm_denom.clone());
    let constant_int = constant_coef.to_integer();

    // RRT: x = ±d/e where d divides |constant| and e divides |leading|
    let c_abs = if constant_int.is_negative() {
        -constant_int.clone()
    } else {
        constant_int.clone()
    };
    let a_abs = if leading_coef.is_negative() {
        -leading_coef.clone()
    } else {
        leading_coef.clone()
    };

    // Find divisors (limit to reasonable size for puzzles)
    fn small_divisors(n: &BigInt, limit: i64) -> Vec<BigInt> {
        let mut divs = Vec::new();
        if n.is_zero() {
            return vec![BigInt::from(1)];
        }
        let n_abs = if n.is_negative() {
            -n.clone()
        } else {
            n.clone()
        };
        for d in 1..=limit {
            let bd = BigInt::from(d);
            if &n_abs % &bd == BigInt::zero() {
                divs.push(bd.clone());
                let quotient = &n_abs / &bd;
                if !divs.contains(&quotient) {
                    divs.push(quotient);
                }
            }
        }
        if divs.is_empty() {
            divs.push(BigInt::from(1));
        }
        divs
    }

    let c_divisors = small_divisors(&c_abs, 50); // divisors of constant term
    let a_divisors = small_divisors(&a_abs, 20); // divisors of leading coef

    // Test candidates ±d/e
    for d in &c_divisors {
        for e in &a_divisors {
            for sign in &[1i32, -1i32] {
                let candidate = if *sign == 1 {
                    num_rational::BigRational::new(d.clone(), e.clone())
                } else {
                    -num_rational::BigRational::new(d.clone(), e.clone())
                };

                // Evaluate x³ + px + q at candidate
                let x2 = &candidate * &candidate;
                let x3 = &x2 * &candidate;
                let val = &x3 + p * &candidate + q;

                if val.is_zero() {
                    return Some(candidate);
                }
            }
        }
    }

    None
}

define_rule!(
    CubicConjugateTrapRule,
    "Cubic Conjugate Identity",
    None,
    crate::phase::PhaseMask::TRANSFORM,
    |ctx, expr| {
        use num_traits::Zero;

        // Match Add(Pow(A, 1/3), Pow(B, 1/3))
        let (left, right) = match ctx.get(expr) {
            Expr::Add(l, r) => (*l, *r),
            _ => return None,
        };

        // Extract cube root bases
        let base_a = is_cube_root_pow(ctx, left)?;
        let base_b = is_cube_root_pow(ctx, right)?;

        // Check if A and B are conjugates (m + t) and (m - t)
        let (m, t) = is_conjugate_pair(ctx, base_a, base_b)?;

        // Compute S = A + B = 2m (directly, without simplify)
        // Since A = m + t and B = m - t, A + B = 2m
        let two = num_rational::BigRational::from_integer(2.into());

        // m must be a rational number for this to work
        let m_val = if let Expr::Number(n) = ctx.get(m) {
            n.clone()
        } else {
            return None; // m is not numeric, can't apply
        };

        let s_val = &two * &m_val; // S = 2m

        // Compute AB = m² - t² (directly)
        // t must also allow us to compute t² as rational
        // For t = sqrt(d) or k*sqrt(d), t² is rational
        let t_squared_val = compute_t_squared(ctx, t)?;

        let ab_val = &m_val * &m_val - &t_squared_val; // AB = m² - t²

        // P = ∛(AB) must be rational (perfect cube)
        let p_val = rational_cbrt(&ab_val)?;

        // Form depressed cubic: x³ + px + q = 0
        // where p_coef = -3P and q_coef = -S
        // x³ - 3Px - S = 0  =>  x³ + (-3P)x + (-S) = 0
        let three = num_rational::BigRational::from_integer(3.into());
        let p_coef = -&three * &p_val; // coefficient of x
        let q_coef = -&s_val; // constant term

        // Guard: if p_coef > 0, cubic is strictly increasing => unique real root
        // This ensures we can trust the RRT result
        if p_coef <= num_rational::BigRational::zero() {
            return None; // Multiple real roots possible, skip
        }

        // Find rational root via RRT
        let root = find_rational_root_depressed_cubic(&p_coef, &q_coef)?;

        // Success! Return the root as the result
        let result = ctx.add(Expr::Number(root.clone()));

        Some(
            Rewrite::new(result)
                .desc_lazy(|| format!("Cubic conjugate identity: ∛(m+t) + ∛(m-t) = {}", root)),
        )
    }
);

/// Compute t² where t may be:
/// - A number: t² = t * t
/// - sqrt(d) or d^(1/2): t² = d  
/// - k * sqrt(d): t² = k² * d
fn compute_t_squared(
    ctx: &cas_ast::Context,
    t: cas_ast::ExprId,
) -> Option<num_rational::BigRational> {
    match ctx.get(t) {
        // Direct number
        Expr::Number(n) => Some(n * n),

        // sqrt(d) function
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            if let Expr::Number(d) = ctx.get(args[0]) {
                Some(d.clone())
            } else {
                None
            }
        }

        // d^(1/2) power form
        Expr::Pow(base, exp) => {
            if let Expr::Number(e) = ctx.get(*exp) {
                // Check if exponent is 1/2
                if *e.numer() == 1.into() && *e.denom() == 2.into() {
                    if let Expr::Number(d) = ctx.get(*base) {
                        return Some(d.clone());
                    }
                }
            }
            None
        }

        // k * sqrt(d) product form
        Expr::Mul(l, r) => {
            // Try both orderings
            let try_extract = |coef: cas_ast::ExprId,
                               surd: cas_ast::ExprId|
             -> Option<num_rational::BigRational> {
                let k = if let Expr::Number(n) = ctx.get(coef) {
                    n.clone()
                } else {
                    return None;
                };

                let d = match ctx.get(surd) {
                    Expr::Function(fn_id, args)
                        if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
                    {
                        if let Expr::Number(n) = ctx.get(args[0]) {
                            n.clone()
                        } else {
                            return None;
                        }
                    }
                    Expr::Pow(base, exp) => {
                        if let Expr::Number(e) = ctx.get(*exp) {
                            if *e.numer() == 1.into() && *e.denom() == 2.into() {
                                if let Expr::Number(n) = ctx.get(*base) {
                                    n.clone()
                                } else {
                                    return None;
                                }
                            } else {
                                return None;
                            }
                        } else {
                            return None;
                        }
                    }
                    _ => return None,
                };

                // t = k * sqrt(d), so t² = k² * d
                Some(&k * &k * &d)
            };

            try_extract(*l, *r).or_else(|| try_extract(*r, *l))
        }

        _ => None,
    }
}

#[cfg(test)]
mod cubic_conjugate_tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn test_cubic_conjugate_basic() {
        let mut ctx = Context::new();
        let expr = parse("(2 + 5^(1/2))^(1/3) + (2 - 5^(1/2))^(1/3)", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "1");
    }

    #[test]
    fn test_cubic_conjugate_commuted() {
        let mut ctx = Context::new();
        // Reversed order
        let expr = parse("(2 - 5^(1/2))^(1/3) + (2 + 5^(1/2))^(1/3)", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "1");
    }

    #[test]
    fn test_cubic_conjugate_no_match_different_surd() {
        let mut ctx = Context::new();
        // Different surds: sqrt(5) vs sqrt(6)
        let expr = parse("(2 + 5^(1/2))^(1/3) + (2 - 6^(1/2))^(1/3)", &mut ctx).unwrap();

        let rule = CubicConjugateTrapRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match different surds");
    }

    #[test]
    fn test_cubic_conjugate_no_match_different_exp() {
        let mut ctx = Context::new();
        // Different exponents: 1/3 vs 1/5
        let expr = parse("(2 + 5^(1/2))^(1/3) + (2 - 5^(1/2))^(1/5)", &mut ctx).unwrap();

        let rule = CubicConjugateTrapRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match different exponents");
    }

    #[test]
    fn test_prerequisite_negative_cube_root() {
        // Prerequisite: (-1)^(1/3) must equal -1 for the rule to work
        let mut ctx = Context::new();
        let expr = parse("(-1)^(1/3)", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "-1");
    }

    #[test]
    fn test_prerequisite_negative_8_cube_root() {
        // (-8)^(1/3) = -2
        let mut ctx = Context::new();
        let expr = parse("(-8)^(1/3)", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "-2");
    }

    #[test]
    fn test_cubic_conjugate_sqrt_function_form() {
        // Test with sqrt() function instead of ^(1/2)
        let mut ctx = Context::new();
        let expr = parse("(2 + sqrt(5))^(1/3) + (2 - sqrt(5))^(1/3)", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "1");
    }

    #[test]
    fn test_cubic_conjugate_no_match_not_sum() {
        // Subtraction instead of sum - should not match
        let mut ctx = Context::new();
        let expr = parse("(2 + 5^(1/2))^(1/3) - (2 - 5^(1/2))^(1/3)", &mut ctx).unwrap();

        let rule = CubicConjugateTrapRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match subtraction");
    }

    #[test]
    fn test_cubic_conjugate_no_match_same_signs() {
        // Both addends have same sign: (m+t) + (m+t) style
        let mut ctx = Context::new();
        let expr = parse("(2 + 5^(1/2))^(1/3) + (2 + 5^(1/2))^(1/3)", &mut ctx).unwrap();

        let rule = CubicConjugateTrapRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(
            rewrite.is_none(),
            "Should not match when both have same sign"
        );
    }

    #[test]
    fn test_cubic_conjugate_no_match_irrational_root() {
        // (1 + √2)^(1/3) + (1 - √2)^(1/3)
        // AB = 1 - 2 = -1 is a cube, but cubic x³ + 3x - 2 = 0 has no rational root
        let mut ctx = Context::new();
        let expr = parse("(1 + 2^(1/2))^(1/3) + (1 - 2^(1/2))^(1/3)", &mut ctx).unwrap();

        let rule = CubicConjugateTrapRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        // The rule should not find a rational root (root is ~0.596)
        assert!(
            rewrite.is_none(),
            "Should not match when no rational root exists"
        );
    }

    #[test]
    fn test_cubic_conjugate_no_match_different_m() {
        // Different m values: (2+√5)^(1/3) + (3-√5)^(1/3)
        let mut ctx = Context::new();
        let expr = parse("(2 + 5^(1/2))^(1/3) + (3 - 5^(1/2))^(1/3)", &mut ctx).unwrap();

        let rule = CubicConjugateTrapRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match different m values");
    }
}
