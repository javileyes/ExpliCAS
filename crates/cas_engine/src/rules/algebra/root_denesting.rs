//! Root denesting rules.
//!
//! Contains `CubicConjugateTrapRule`, `DenestSqrtAddSqrtRule`,
//! `DenestPerfectCubeInQuadraticFieldRule` and their helper functions.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Expr};
use num_traits::{Signed, Zero};

use super::roots::rational_sqrt;

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

        Some(Rewrite::new(result).desc(format!(
            "Cubic conjugate identity: ∛(m+t) + ∛(m-t) = {}",
            root
        )))
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

        Some(Rewrite::new(result).desc(format!(
            "Denest nested square root: √(a+√b) = √({}) + √({})",
            m, n
        )))
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

        Some(Rewrite::new(result).desc(format!(
            "Denest cube root in quadratic field: ∛(A+B√n) = {} + {}√{}",
            x, y, n
        )))
    }
);

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
