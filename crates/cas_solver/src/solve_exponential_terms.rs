//! Pure parsing utilities for exponential terms `coeff · base^(affine·x)`.
//!
//! Extracted from `solve_backend_local` (a growing god file): these are structural, side-effect-free
//! matchers over `&Context` — no solve-infrastructure dependencies — shared by the exponential
//! equation handlers (`try_solve_exponential_reciprocal_polynomial`,
//! `try_solve_two_different_base_exponential_equation`). Owning them here keeps the "how do we read an
//! exponential term" vocabulary in one place; the handlers that *use* the vocabulary stay with the
//! dispatch.

use cas_ast::{Context, Expr, ExprId};

/// Accumulate the Laurent-in-`base^x` structure of `expr` into `map` (`integer exponent k → rational
/// coefficient` of `base^(k·x)`), walking `Add`/`Sub`/`Neg` for the sign and matching each leaf as
/// `coeff · base^(affine in x)` or a `var`-free constant (`k = 0`). Returns `None` if any leaf is not
/// such a term (a second exponential base, a `var` outside the exponent, an irrational `base^m`
/// coefficient) — i.e. the expression is not a Laurent polynomial in `base^x`.
pub(crate) fn collect_exp_laurent_terms(
    ctx: &Context,
    expr: ExprId,
    base: ExprId,
    var: &str,
    positive: bool,
    map: &mut std::collections::BTreeMap<i64, num_rational::BigRational>,
) -> Option<()> {
    use num_rational::BigRational;
    use num_traits::Zero;
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            collect_exp_laurent_terms(ctx, l, base, var, positive, map)?;
            collect_exp_laurent_terms(ctx, r, base, var, positive, map)
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            collect_exp_laurent_terms(ctx, l, base, var, positive, map)?;
            collect_exp_laurent_terms(ctx, r, base, var, !positive, map)
        }
        Expr::Neg(inner) => collect_exp_laurent_terms(ctx, *inner, base, var, !positive, map),
        _ => {
            let (k, coeff) = exp_laurent_leaf(ctx, expr, base, var)?;
            let signed = if positive { coeff } else { -coeff };
            *map.entry(k).or_insert_with(BigRational::zero) += signed;
            Some(())
        }
    }
}

/// Match a single leaf term as `coeff · base^(k·x + m)` and return `(k, coeff·base^m)` (with `k` an
/// integer, `coeff·base^m` rational), or a `var`-free constant as `(0, value)`. `None` if the leaf is
/// not of this shape (different base, non-integer exponent slope, or an irrational `base^m`).
pub(crate) fn exp_laurent_leaf(
    ctx: &Context,
    expr: ExprId,
    base: ExprId,
    var: &str,
) -> Option<(i64, num_rational::BigRational)> {
    use cas_ast::ordering::compare_expr;
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    // A `var`-free leaf is a constant term (`k = 0`).
    if !contains_var(ctx, expr, var) {
        return Some((0, as_rational_const(ctx, expr)?));
    }
    match ctx.get(expr) {
        // `base^(affine in x)`: slope is the exponent `k`, intercept `m` folds into the coefficient.
        Expr::Pow(b, e) => {
            if compare_expr(ctx, *b, base) != std::cmp::Ordering::Equal {
                return None;
            }
            let (slope, intercept) = affine_integer_exponent(ctx, *e, var)?;
            let coeff = base_pow_integer_rational(ctx, base, intercept)?;
            Some((slope, coeff))
        }
        // `c / base^(affine)`: reciprocal ⇒ negate the exponent and invert the folded coefficient.
        Expr::Div(n, d) => {
            let (n, d) = (*n, *d);
            if contains_var(ctx, n, var) {
                return None;
            }
            let c = as_rational_const(ctx, n)?;
            let (k, den_coeff) = exp_laurent_leaf(ctx, d, base, var)?;
            if num_traits::Zero::is_zero(&den_coeff) {
                return None;
            }
            Some((-k, c / den_coeff))
        }
        // `c · (exp leaf)` with one `var`-free numeric factor.
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            if !contains_var(ctx, l, var) {
                let c = as_rational_const(ctx, l)?;
                let (k, cc) = exp_laurent_leaf(ctx, r, base, var)?;
                Some((k, c * cc))
            } else if !contains_var(ctx, r, var) {
                let c = as_rational_const(ctx, r)?;
                let (k, cc) = exp_laurent_leaf(ctx, l, base, var)?;
                Some((k, c * cc))
            } else {
                None
            }
        }
        Expr::Neg(inner) => {
            let (k, c) = exp_laurent_leaf(ctx, *inner, base, var)?;
            Some((k, -c))
        }
        _ => None,
    }
}

/// Parse `e` as an affine `slope·x + intercept` with INTEGER slope and intercept, returning
/// `(slope, intercept)`, or `None` if it is not affine in `var` or a coefficient is not an integer.
pub(crate) fn affine_integer_exponent(ctx: &Context, e: ExprId, var: &str) -> Option<(i64, i64)> {
    use cas_math::polynomial::Polynomial;
    use num_traits::ToPrimitive;
    let poly = Polynomial::from_expr(ctx, e, var).ok()?;
    if poly.degree() > 1 {
        return None;
    }
    let int_of = |r: &num_rational::BigRational| -> Option<i64> {
        if r.is_integer() {
            r.to_integer().to_i64()
        } else {
            None
        }
    };
    // `coeffs[i]` is the coefficient of `xⁱ` (absent ⇒ 0). Degree ≤ 1 guaranteed above.
    let zero = num_rational::BigRational::from_integer(0.into());
    let slope = int_of(poly.coeffs.get(1).unwrap_or(&zero))?;
    let intercept = int_of(poly.coeffs.first().unwrap_or(&zero))?;
    Some((slope, intercept))
}

/// Compute `base^m` as a rational (for the affine-exponent intercept fold): `m = 0 → 1` for ANY base;
/// otherwise `base` must be a positive rational so `base^m` is rational (`2^1 = 2` for `2^(1−x)`).
/// Declines an irrational `base^m` (e.g. `e^1`), keeping the handler in the rational scope.
pub(crate) fn base_pow_integer_rational(
    ctx: &Context,
    base: ExprId,
    m: i64,
) -> Option<num_rational::BigRational> {
    use cas_math::numeric_eval::as_rational_const;
    use num_traits::One;
    if m == 0 {
        return Some(num_rational::BigRational::one());
    }
    let b = as_rational_const(ctx, base)?;
    let mag = b.pow(m.unsigned_abs() as i32);
    if m > 0 {
        Some(mag)
    } else {
        if num_traits::Zero::is_zero(&mag) {
            return None;
        }
        Some(mag.recip())
    }
}
