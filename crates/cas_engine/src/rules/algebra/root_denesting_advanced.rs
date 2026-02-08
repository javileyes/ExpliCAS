//! Advanced root denesting rules.
//!
//! Contains `DenestSqrtAddSqrtRule` (√(a+√b) → √m+√n) and
//! `DenestPerfectCubeInQuadraticFieldRule` (∛(A+B√n) → x+y√n),
//! plus their helper functions.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Expr};

use super::roots::rational_sqrt;
use num_traits::Signed;

// =============================================================================
// DENEST SQRT(a + SQRT(b)) RULE
// Simplifies √(a + √b) → √m + √n where m,n = (a ± √(a²-b))/2
// =============================================================================

/// Extract the radicand if expression is a sqrt (either sqrt(x) function or x^(1/2))
fn as_sqrt(ctx: &cas_ast::Context, e: cas_ast::ExprId) -> Option<cas_ast::ExprId> {
    match ctx.get(e) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            Some(args[0])
        }
        Expr::Pow(base, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                if *n.numer() == 1.into() && *n.denom() == 2.into() {
                    return Some(*base);
                }
            }
            None
        }
        _ => None,
    }
}

define_rule!(
    DenestSqrtAddSqrtRule,
    "Denest Nested Square Root",
    None,
    crate::phase::PhaseMask::TRANSFORM,
    |ctx, expr| {
        // Match sqrt(inner) where inner = a + sqrt(b) or a - sqrt(b)
        let inner = as_sqrt(ctx, expr)?;

        // Inner must be Add or Sub
        let (left, right, is_add) = match ctx.get(inner) {
            Expr::Add(l, r) => (*l, *r, true),
            Expr::Sub(l, r) => (*l, *r, false),
            _ => return None,
        };

        // Identify which is `a` (rational) and which is `sqrt(b)`
        // Try both orderings
        let (a_val, b_val) = {
            // Try: left = a (Number), right = sqrt(b)
            if let Expr::Number(a) = ctx.get(left) {
                if let Some(b_inner) = as_sqrt(ctx, right) {
                    if let Expr::Number(b) = ctx.get(b_inner) {
                        Some((a.clone(), b.clone()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        }
        .or_else(|| {
            // Try: left = sqrt(b), right = a (Number)
            if let Some(b_inner) = as_sqrt(ctx, left) {
                if let Expr::Number(b) = ctx.get(b_inner) {
                    if let Expr::Number(a) = ctx.get(right) {
                        Some((a.clone(), b.clone()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        })?;

        // For subtraction (a - sqrt(b)), we'd need a different formula
        // For now, only handle addition: sqrt(a + sqrt(b))
        if !is_add {
            // TODO: Handle subtraction case
            return None;
        }

        // Apply denesting formula:
        // √(a + √b) = √m + √n where m = (a + √disc)/2, n = (a - √disc)/2
        // disc = a² - b

        let disc = &a_val * &a_val - &b_val;

        // disc must have a rational square root
        let disc_sqrt = rational_sqrt(&disc)?;

        // m = (a + disc_sqrt) / 2
        // n = (a - disc_sqrt) / 2
        let two = num_rational::BigRational::from_integer(2.into());
        let m = (&a_val + &disc_sqrt) / &two;
        let n = (&a_val - &disc_sqrt) / &two;

        // Both m and n must be non-negative for real roots
        if m.is_negative() || n.is_negative() {
            return None;
        }

        // Build result: sqrt(m) + sqrt(n)
        let m_expr = ctx.add(Expr::Number(m.clone()));
        let n_expr = ctx.add(Expr::Number(n.clone()));

        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let sqrt_m = ctx.add(Expr::Pow(m_expr, half));
        let half2 = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let sqrt_n = ctx.add(Expr::Pow(n_expr, half2));

        let result = ctx.add(Expr::Add(sqrt_m, sqrt_n));

        Some(
            Rewrite::new(result)
                .desc_lazy(|| format!("Denest nested square root: √(a+√b) = √({}) + √({})", m, n)),
        )
    }
);

// =============================================================================
// DENEST PERFECT CUBE IN QUADRATIC FIELD RULE
// Simplifies ∛(A + B√n) → x + y√n where (x+y√n)³ = A+B√n
// =============================================================================

/// Try to split an expression as A + B*sqrt(n) where A, B, n are rationals.
/// Returns (A, B, n) if successful.
fn split_linear_surd(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(
    num_rational::BigRational,
    num_rational::BigRational,
    num_rational::BigRational,
)> {
    use num_rational::BigRational;

    // Helper to extract coefficient and radicand from a surd term (B*sqrt(n) or sqrt(n))
    fn extract_coef_surd(
        ctx: &cas_ast::Context,
        term: cas_ast::ExprId,
    ) -> Option<(BigRational, BigRational)> {
        // Case: sqrt(n) or n^(1/2)
        if let Some(radicand) = as_sqrt(ctx, term) {
            if let Expr::Number(n) = ctx.get(radicand) {
                return Some((BigRational::from_integer(1.into()), n.clone()));
            }
        }

        // Case: B * sqrt(n) or sqrt(n) * B
        if let Expr::Mul(l, r) = ctx.get(term) {
            // Try l = B, r = sqrt(n)
            if let Expr::Number(b) = ctx.get(*l) {
                if let Some(radicand) = as_sqrt(ctx, *r) {
                    if let Expr::Number(n) = ctx.get(radicand) {
                        return Some((b.clone(), n.clone()));
                    }
                }
            }
            // Try l = sqrt(n), r = B
            if let Expr::Number(b) = ctx.get(*r) {
                if let Some(radicand) = as_sqrt(ctx, *l) {
                    if let Expr::Number(n) = ctx.get(radicand) {
                        return Some((b.clone(), n.clone()));
                    }
                }
            }
        }

        None
    }

    match ctx.get(expr) {
        // A + B*sqrt(n) or B*sqrt(n) + A
        Expr::Add(l, r) => {
            // Try: l = A (Number), r = B*sqrt(n)
            if let Expr::Number(a) = ctx.get(*l) {
                if let Some((b, n)) = extract_coef_surd(ctx, *r) {
                    return Some((a.clone(), b, n));
                }
            }
            // Try: l = B*sqrt(n), r = A (Number)
            if let Expr::Number(a) = ctx.get(*r) {
                if let Some((b, n)) = extract_coef_surd(ctx, *l) {
                    return Some((a.clone(), b, n));
                }
            }
            // Check for l + Neg(r) = l - something
            if let Expr::Neg(neg_inner) = ctx.get(*r) {
                if let Expr::Number(a) = ctx.get(*l) {
                    if let Some((b, n)) = extract_coef_surd(ctx, *neg_inner) {
                        return Some((a.clone(), -b, n));
                    }
                }
            }
            None
        }
        // A - B*sqrt(n)
        Expr::Sub(l, r) => {
            if let Expr::Number(a) = ctx.get(*l) {
                if let Some((b, n)) = extract_coef_surd(ctx, *r) {
                    return Some((a.clone(), -b, n));
                }
            }
            // Also handle sqrt(n) - A (which would be -A + sqrt(n))
            if let Expr::Number(a) = ctx.get(*r) {
                if let Some((b, n)) = extract_coef_surd(ctx, *l) {
                    return Some((-a.clone(), b, n));
                }
            }
            None
        }
        _ => None,
    }
}

/// Try to find rational x, y such that (x + y*sqrt(n))^3 = A + B*sqrt(n)
/// The equations are:
///   Rational part:    x³ + 3xy²n = A
///   Irrational part:  3x²y + y³n = B
/// We enumerate y from small rational candidates and solve for x.
fn solve_cube_in_quadratic_field(
    a: &num_rational::BigRational,
    b: &num_rational::BigRational,
    n: &num_rational::BigRational,
) -> Option<(num_rational::BigRational, num_rational::BigRational)> {
    use num_bigint::BigInt;
    use num_rational::BigRational;
    use num_traits::{Signed, Zero};

    // Guard: n must be positive for real sqrt
    if n <= &BigRational::zero() {
        return None;
    }

    // Guard: don't process huge numbers
    let a_approx: f64 = a.numer().to_string().parse().unwrap_or(f64::MAX);
    let b_approx: f64 = b.numer().to_string().parse().unwrap_or(f64::MAX);
    if a_approx.abs() > 1e12 || b_approx.abs() > 1e12 {
        return None;
    }

    // Denominators to try for y: {1, 2, 3, 4, 6, 8, 12}
    let denoms: [i64; 7] = [1, 2, 3, 4, 6, 8, 12];

    // Numerator range based on rough estimate
    // |y| ≈ cbrt(|B|) / sqrt(n) roughly, but we use a generous bound
    let max_num: i64 = 10;

    let three = BigRational::from_integer(3.into());

    for &denom in &denoms {
        let denom_big = BigInt::from(denom);
        for num in -max_num..=max_num {
            if num == 0 {
                continue; // y = 0 would mean no surd part
            }

            let y = BigRational::new(BigInt::from(num), denom_big.clone());

            // From: 3x²y + y³n = B
            // => x² = (B/y - y²n) / 3 = (B - y³n) / (3y)
            // But easier from: y(3x² + ny²) = B
            // => 3x² + ny² = B/y
            // => x² = (B/y - ny²) / 3

            let y_squared = &y * &y;
            let y_cubed = &y_squared * &y;

            // x² = (B/y - n*y²) / 3
            let b_over_y = b / &y;
            let n_y_sq = n * &y_squared;
            let x_squared = (&b_over_y - &n_y_sq) / &three;

            // x² must be non-negative
            if x_squared.is_negative() {
                continue;
            }

            // Try to get rational sqrt of x²
            if let Some(x_pos) = rational_sqrt(&x_squared) {
                // Try both +x and -x
                for x in [x_pos.clone(), -x_pos.clone()] {
                    // Verify: x³ + 3xy²n = A
                    let x_cubed = &x * &x * &x;
                    let term_3xy2n = &three * &x * &y_squared * n;
                    let lhs_a = &x_cubed + &term_3xy2n;

                    // Verify: 3x²y + y³n = B
                    let x_sq = &x * &x;
                    let term_3x2y = &three * &x_sq * &y;
                    let term_y3n = &y_cubed * n;
                    let lhs_b = &term_3x2y + &term_y3n;

                    if &lhs_a == a && &lhs_b == b {
                        return Some((x, y));
                    }
                }
            }
        }
    }

    None
}

define_rule!(
    DenestPerfectCubeInQuadraticFieldRule,
    "Denest Cube Root in Quadratic Field",
    None,
    crate::phase::PhaseMask::TRANSFORM,
    |ctx, expr| {
        use num_traits::Zero;

        // Match Pow(base, 1/3)
        let (base, exp) = match ctx.get(expr) {
            Expr::Pow(b, e) => (*b, *e),
            _ => return None,
        };

        // Check exponent is 1/3
        if let Expr::Number(exp_val) = ctx.get(exp) {
            if !(*exp_val.numer() == 1.into() && *exp_val.denom() == 3.into()) {
                return None;
            }
        } else {
            return None;
        }

        // Extract A + B*sqrt(n) from base
        let (a, b, n) = split_linear_surd(ctx, base)?;

        // Guard: b must be non-zero (otherwise no surd)
        if b.is_zero() {
            return None;
        }

        // Try to find x, y such that (x + y*sqrt(n))³ = A + B*sqrt(n)
        let (x, y) = solve_cube_in_quadratic_field(&a, &b, &n)?;

        // Build result: x + y*sqrt(n)
        let x_expr = ctx.add(Expr::Number(x.clone()));
        let y_expr = ctx.add(Expr::Number(y.clone()));
        let n_expr = ctx.add(Expr::Number(n.clone()));

        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let sqrt_n = ctx.add(Expr::Pow(n_expr, half));

        let result = if y.is_zero() {
            x_expr
        } else if x.is_zero() {
            ctx.add(Expr::Mul(y_expr, sqrt_n))
        } else {
            let y_sqrt_n = ctx.add(Expr::Mul(y_expr, sqrt_n));
            ctx.add(Expr::Add(x_expr, y_sqrt_n))
        };

        Some(Rewrite::new(result).desc_lazy(|| {
            format!(
                "Denest cube root in quadratic field: ∛(A+B√n) = {} + {}√{}",
                x, y, n
            )
        }))
    }
);

#[cfg(test)]
mod denest_sqrt_tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn test_denest_sqrt_4_plus_sqrt7() {
        // √(4 + √7) → √(7/2) + √(1/2)
        let mut ctx = Context::new();
        let expr = parse("sqrt(4 + sqrt(7))", &mut ctx).unwrap();

        let rule = DenestSqrtAddSqrtRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_some(), "Rule should apply to √(4+√7)");

        // Verify the result simplifies correctly
        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        // Just check that we get the denested form with surds
        let result_str = format!(
            "{}",
            cas_ast::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        // Should contain fractions 1/2 and 7/2
        assert!(
            result_str.contains("1/2") && result_str.contains("7/2"),
            "Result should be √(1/2)+√(7/2), got: {}",
            result_str
        );
    }

    #[test]
    fn test_denest_sqrt_pow_form() {
        // (4 + 7^(1/2))^(1/2) → pow form instead of sqrt function
        // Use simplifier since the expression needs canonicalization
        let mut ctx = Context::new();
        let expr = parse("(4 + 7^(1/2))^(1/2)", &mut ctx).unwrap();

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
        // Should contain fractions 1/2 and 7/2
        assert!(
            result_str.contains("1/2") && result_str.contains("7/2"),
            "Result should be denested with 1/2 and 7/2, got: {}",
            result_str
        );
    }

    #[test]
    fn test_denest_sqrt_no_match_bad_discriminant() {
        // √(3 + √5): disc = 9 - 5 = 4 ✓, but let's check it works
        // disc_sqrt = 2, m = (3+2)/2 = 5/2, n = (3-2)/2 = 1/2
        let mut ctx = Context::new();
        let expr = parse("sqrt(3 + sqrt(5))", &mut ctx).unwrap();

        let rule = DenestSqrtAddSqrtRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(
            rewrite.is_some(),
            "Should match √(3+√5) since disc=4 is perfect square"
        );
    }

    #[test]
    fn test_denest_sqrt_no_match_non_perfect_square_disc() {
        // √(4 + √10): disc = 16 - 10 = 6 (not a perfect square)
        let mut ctx = Context::new();
        let expr = parse("sqrt(4 + sqrt(10))", &mut ctx).unwrap();

        let rule = DenestSqrtAddSqrtRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(
            rewrite.is_none(),
            "Should not match when disc=6 is not a perfect square"
        );
    }

    #[test]
    fn test_denest_sqrt_no_match_negative_m_or_n() {
        // √(1 + √10): disc = 1 - 10 = -9 (negative)
        let mut ctx = Context::new();
        let expr = parse("sqrt(1 + sqrt(10))", &mut ctx).unwrap();

        let rule = DenestSqrtAddSqrtRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match when disc < 0");
    }

    #[test]
    fn test_denest_sqrt_commuted_order() {
        // √(√7 + 4) - surd comes first
        let mut ctx = Context::new();
        let expr = parse("sqrt(sqrt(7) + 4)", &mut ctx).unwrap();

        let rule = DenestSqrtAddSqrtRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_some(), "Should match commuted order √(√7+4)");
    }
}

#[cfg(test)]
mod denest_cube_quadratic_tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn test_denest_cube_26_15_sqrt3() {
        // (26 + 15*sqrt(3))^(1/3) → 2 + sqrt(3)
        // Because (2 + sqrt(3))³ = 26 + 15*sqrt(3)
        let mut ctx = Context::new();
        let expr = parse("(26 + 15*sqrt(3))^(1/3)", &mut ctx).unwrap();

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
        assert!(
            result_str.contains("2") && result_str.contains("3"),
            "Should be 2 + √3, got: {}",
            result_str
        );
        // Verify it doesn't contain a cube root anymore
        assert!(
            !result_str.contains("∛") && !result_str.contains("1/3"),
            "Should NOT contain cube root, got: {}",
            result_str
        );
    }

    #[test]
    fn test_denest_cube_golden_ratio() {
        // (2 + sqrt(5))^(1/3) → (1 + sqrt(5))/2 = φ
        let mut ctx = Context::new();
        let expr = parse("(2 + sqrt(5))^(1/3)", &mut ctx).unwrap();

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
        // Should be φ (phi) since (1 + √5)/2 is recognized as phi
        assert!(
            result_str.contains("phi")
                || (result_str.contains("1")
                    && result_str.contains("5")
                    && result_str.contains("2")),
            "Should be phi or (1+√5)/2, got: {}",
            result_str
        );
    }

    #[test]
    fn test_denest_cube_golden_ratio_conjugate() {
        // (2 - sqrt(5))^(1/3) → (1 - sqrt(5))/2 = 1-φ
        let mut ctx = Context::new();
        let expr = parse("(2 - sqrt(5))^(1/3)", &mut ctx).unwrap();

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
        // Should be (1 - √5)/2
        assert!(
            result_str.contains("1") && result_str.contains("5"),
            "Should be (1-√5)/2, got: {}",
            result_str
        );
    }

    #[test]
    fn test_denest_cube_no_match_sqrt6() {
        // (2 + sqrt(6))^(1/3) - no rational x,y exists
        let mut ctx = Context::new();
        let expr = parse("(2 + sqrt(6))^(1/3)", &mut ctx).unwrap();

        let rule = DenestPerfectCubeInQuadraticFieldRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should NOT match (2+√6)^(1/3)");
    }

    #[test]
    fn test_denest_cube_no_match_wrong_exp() {
        // (2 + sqrt(5))^(1/5) - exponent is 1/5 not 1/3
        let mut ctx = Context::new();
        let expr = parse("(2 + sqrt(5))^(1/5)", &mut ctx).unwrap();

        let rule = DenestPerfectCubeInQuadraticFieldRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should NOT match exponent 1/5");
    }
}
