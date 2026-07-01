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

/// Factor an integer `n ≥ 2` as a single prime power `p^k`, or `None` if it has more than one distinct
/// prime factor (`6 = 2·3` → None). Trial division; the bases here are small literals.
pub(crate) fn integer_prime_power(n: &num_bigint::BigInt) -> Option<(num_bigint::BigInt, u32)> {
    use num_bigint::BigInt;
    use num_traits::{One, Zero};
    let mut m = n.clone();
    if m < BigInt::from(2) {
        return None;
    }
    let mut p = BigInt::from(2);
    while &p * &p <= m {
        if (&m % &p).is_zero() {
            break;
        }
        p += 1;
    }
    if &p * &p > m {
        p = m.clone(); // m itself is prime
    }
    let mut k = 0u32;
    while (&m % &p).is_zero() {
        m /= &p;
        k += 1;
    }
    m.is_one().then_some((p, k))
}

/// Collect the DISTINCT integer bases `m ≥ 2` of every `m^(…)` subterm whose EXPONENT carries the
/// variable (`4^x`, `2^x`, `9^x`), so a mixed-base exponential can be normalized to a common base.
pub(crate) fn collect_exp_integer_bases(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    out: &mut Vec<num_bigint::BigInt>,
) {
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    use num_traits::One;
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        let (base, exp) = (*base, *exp);
        if contains_var(ctx, exp, var) && !contains_var(ctx, base, var) {
            if let Some(b) = as_rational_const(ctx, base) {
                if b.is_integer() && b > num_rational::BigRational::one() {
                    let bi = b.to_integer();
                    if !out.contains(&bi) {
                        out.push(bi);
                    }
                }
            }
        }
    }
    match ctx.get(expr).clone() {
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            collect_exp_integer_bases(ctx, a, var, out);
            collect_exp_integer_bases(ctx, b, var, out);
        }
        Expr::Neg(a) | Expr::Hold(a) => collect_exp_integer_bases(ctx, a, var, out),
        Expr::Function(_, args) => {
            for a in args {
                collect_exp_integer_bases(ctx, a, var, out);
            }
        }
        _ => {}
    }
}

/// Rewrite every `m^g` (integer `m = p^k`, exponent `g` carrying the variable) to `p^(k·g)`, mapping a
/// mixed-base exponential onto the common prime base `p` (`4^x → 2^(2x)`).
pub(crate) fn rewrite_exp_bases_to_prime(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    p: &num_bigint::BigInt,
) -> ExprId {
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    use num_rational::BigRational;
    if let Expr::Pow(base, exp) = ctx.get(expr).clone() {
        if contains_var(ctx, exp, var) && !contains_var(ctx, base, var) {
            if let Some(b) = as_rational_const(ctx, base) {
                if b.is_integer() {
                    if let Some((q, k)) = integer_prime_power(&b.to_integer()) {
                        if &q == p {
                            let g = rewrite_exp_bases_to_prime(ctx, exp, var, p);
                            if k == 1 {
                                let p_node = ctx.add(Expr::Number(BigRational::from(p.clone())));
                                return ctx.add(Expr::Pow(p_node, g));
                            }
                            let p_node = ctx.add(Expr::Number(BigRational::from(p.clone())));
                            let k_node = ctx
                                .add(Expr::Number(BigRational::from(num_bigint::BigInt::from(k))));
                            let new_exp = ctx.add(Expr::Mul(k_node, g));
                            return ctx.add(Expr::Pow(p_node, new_exp));
                        }
                    }
                }
            }
        }
    }
    match ctx.get(expr).clone() {
        Expr::Add(a, b) => {
            let (a, b) = (
                rewrite_exp_bases_to_prime(ctx, a, var, p),
                rewrite_exp_bases_to_prime(ctx, b, var, p),
            );
            ctx.add(Expr::Add(a, b))
        }
        Expr::Sub(a, b) => {
            let (a, b) = (
                rewrite_exp_bases_to_prime(ctx, a, var, p),
                rewrite_exp_bases_to_prime(ctx, b, var, p),
            );
            ctx.add(Expr::Sub(a, b))
        }
        Expr::Mul(a, b) => {
            let (a, b) = (
                rewrite_exp_bases_to_prime(ctx, a, var, p),
                rewrite_exp_bases_to_prime(ctx, b, var, p),
            );
            ctx.add(Expr::Mul(a, b))
        }
        Expr::Div(a, b) => {
            let (a, b) = (
                rewrite_exp_bases_to_prime(ctx, a, var, p),
                rewrite_exp_bases_to_prime(ctx, b, var, p),
            );
            ctx.add(Expr::Div(a, b))
        }
        Expr::Pow(a, b) => {
            let (a, b) = (
                rewrite_exp_bases_to_prime(ctx, a, var, p),
                rewrite_exp_bases_to_prime(ctx, b, var, p),
            );
            ctx.add(Expr::Pow(a, b))
        }
        Expr::Neg(a) => {
            let a = rewrite_exp_bases_to_prime(ctx, a, var, p);
            ctx.add(Expr::Neg(a))
        }
        _ => expr,
    }
}

/// Match a single leaf as `coeff · base^(k·x + m)` and return its EFFECTIVE base `base^k` (the base of
/// `^x`) and folded coefficient `coeff · base^m`, both rational. `None` for a non-exponential leaf, a
/// non-rational base, or a `base^0` (constant slope) term.
pub(crate) fn exponential_base_and_coeff(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(num_rational::BigRational, num_rational::BigRational)> {
    use cas_math::numeric_eval::as_rational_const;
    use cas_solver_core::isolation_utils::contains_var;
    if !contains_var(ctx, expr, var) {
        return None; // a constant term ⇒ not a pure two-exponential form
    }
    match ctx.get(expr) {
        Expr::Pow(b, e) => {
            let base = as_rational_const(ctx, *b)?;
            // A valid exponential base is a positive rational ≠ 1.
            if base <= num_traits::Zero::zero() || base == num_traits::One::one() {
                return None;
            }
            let (slope, intercept) = affine_integer_exponent(ctx, *e, var)?;
            if slope == 0 {
                return None;
            }
            let eff_base = base_pow_integer_rational(ctx, *b, slope)?;
            let coeff = base_pow_integer_rational(ctx, *b, intercept)?;
            Some((eff_base, coeff))
        }
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            if !contains_var(ctx, l, var) {
                let c = as_rational_const(ctx, l)?;
                let (base, coeff) = exponential_base_and_coeff(ctx, r, var)?;
                Some((base, c * coeff))
            } else if !contains_var(ctx, r, var) {
                let c = as_rational_const(ctx, r)?;
                let (base, coeff) = exponential_base_and_coeff(ctx, l, var)?;
                Some((base, c * coeff))
            } else {
                None
            }
        }
        Expr::Div(n, d) => {
            let (n, d) = (*n, *d);
            if contains_var(ctx, n, var) {
                return None;
            }
            let c = as_rational_const(ctx, n)?;
            let (base, coeff) = exponential_base_and_coeff(ctx, d, var)?;
            if num_traits::Zero::is_zero(&base) || num_traits::Zero::is_zero(&coeff) {
                return None;
            }
            Some((base.recip(), c / coeff))
        }
        Expr::Neg(inner) => {
            let (base, coeff) = exponential_base_and_coeff(ctx, *inner, var)?;
            Some((base, -coeff))
        }
        _ => None,
    }
}

/// Collect `expr` as `Σ coeffᵢ · baseᵢ^x` into `terms` (effective base + rational coefficient), walking
/// `Add`/`Sub`/`Neg` for the sign. `None` if any leaf is not `c · b^(affine·x)` (a nonzero constant, a
/// non-rational base, or other var structure) — i.e. the expression is not a pure sum of exponentials.
pub(crate) fn collect_exponential_base_terms(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    positive: bool,
    terms: &mut Vec<(num_rational::BigRational, num_rational::BigRational)>,
) -> Option<()> {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            collect_exponential_base_terms(ctx, l, var, positive, terms)?;
            collect_exponential_base_terms(ctx, r, var, positive, terms)
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            collect_exponential_base_terms(ctx, l, var, positive, terms)?;
            collect_exponential_base_terms(ctx, r, var, !positive, terms)
        }
        Expr::Neg(inner) => collect_exponential_base_terms(ctx, *inner, var, !positive, terms),
        _ => {
            if let Some((base, coeff)) = exponential_base_and_coeff(ctx, expr, var) {
                let signed = if positive { coeff } else { -coeff };
                terms.push((base, signed));
                return Some(());
            }
            // A var-free ZERO constant (the moved-over `= 0` RHS) contributes nothing; a NONZERO
            // constant is a different form (`4^x − 9^x = 1`) and is out of scope.
            let c = cas_math::numeric_eval::as_rational_const(ctx, expr)?;
            if num_traits::Zero::is_zero(&c) {
                Some(())
            } else {
                None
            }
        }
    }
}
