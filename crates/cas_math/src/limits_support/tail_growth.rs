//! `limits_support`: familia `tail_growth`.
//!
//! Ver la cabecera de `limits_support.rs` para el contexto.

use super::*;

pub(super) fn finite_argument_tail_after_limit(
    ctx: &mut Context,
    argument: ExprId,
    argument_limit: ExprId,
) -> ExprId {
    match ctx.get(argument).clone() {
        Expr::Add(lhs, rhs) => {
            if structurally_equal_expr(ctx, lhs, argument_limit) {
                return rhs;
            }
            if structurally_equal_expr(ctx, rhs, argument_limit) {
                return lhs;
            }
        }
        Expr::Sub(lhs, rhs) => {
            if structurally_equal_expr(ctx, lhs, argument_limit) {
                return ctx.add(Expr::Neg(rhs));
            }
            if structurally_equal_expr(ctx, rhs, argument_limit) {
                return lhs;
            }
        }
        _ => {}
    }

    let neg_limit = ctx.add(Expr::Neg(argument_limit));
    ctx.add(Expr::Add(argument, neg_limit))
}

/// Asymptotic leading term `coeff * x^exp` (`coeff != 0`) of `factor` as
/// `x -> +inf`, for a polynomial factor or a `scale * sqrt(polynomial)` factor
/// with a rational leading square root. `exp` is rational (a sqrt factor halves
/// the degree). Declines anything else.
pub(super) fn factor_leading_term_at_pos_inf(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    if let Ok(poly) = Polynomial::from_expr(ctx, expr, var_name) {
        let deg = poly.degree();
        let lead = poly.coeffs.get(deg)?.clone();
        if lead.is_zero() {
            return None;
        }
        return Some((lead, BigRational::from_integer(BigInt::from(deg as i64))));
    }
    let (scale, radicand) = scaled_square_root_base(ctx, expr)?;
    if !scale.is_positive() {
        return None;
    }
    let poly = Polynomial::from_expr(ctx, radicand, var_name).ok()?;
    let deg = poly.degree();
    let lead = poly.coeffs.get(deg)?.clone();
    let sqrt_lead = rational_sqrt(&lead)?;
    Some((
        scale * sqrt_lead,
        BigRational::new(BigInt::from(deg as i64), BigInt::from(2)),
    ))
}

/// Asymptotic leading term `coeff * x^exp` of a conjugate radical difference as
/// `x -> +inf`, when the leading radical terms cancel so the difference decays.
/// Flattens the additive terms and partitions them into `scale*sqrt(polynomial)`
/// terms and a polynomial remainder. One sqrt term + linear remainder is the
/// `sqrt(quadratic) - linear` form (any orientation, any split linear tail); two
/// sqrt terms with zero remainder is `sqrt(P) - sqrt(Q)`. None otherwise.
pub(super) fn radical_difference_asymptotic_at_pos_inf(
    ctx: &Context,
    diff: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    let mut terms: Vec<(ExprId, bool)> = Vec::new();
    collect_signed_add_terms(ctx, diff, true, &mut terms);
    let zero = BigRational::from_integer(BigInt::from(0));
    let mut sqrt_terms: Vec<(BigRational, ExprId)> = Vec::new(); // (signed scale, radicand)
    let mut remainder: Vec<BigRational> = Vec::new(); // polynomial remainder coeffs
    for (term, positive) in terms {
        if let Some((scale, radicand)) = scaled_square_root_base(ctx, term) {
            sqrt_terms.push((if positive { scale } else { -scale }, radicand));
        } else {
            let poly = Polynomial::from_expr(ctx, term, var_name).ok()?;
            for (i, c) in poly.coeffs.iter().enumerate() {
                if remainder.len() <= i {
                    remainder.resize(i + 1, zero.clone());
                }
                remainder[i] = if positive {
                    &remainder[i] + c
                } else {
                    &remainder[i] - c
                };
            }
        }
    }
    match sqrt_terms.len() {
        1 => sqrt_quadratic_plus_remainder_asymptotic(ctx, &sqrt_terms[0], &remainder, var_name),
        2 => {
            if remainder.iter().any(|c| !c.is_zero()) {
                return None; // a sqrt(P)-sqrt(Q) difference carries no polynomial tail here
            }
            two_sqrt_difference_asymptotic(ctx, &sqrt_terms[0], &sqrt_terms[1], var_name)
        }
        _ => None,
    }
}

/// Asymptotic of `s*sqrt(Q) + R` with `Q` quadratic and `R` the (linear)
/// polynomial remainder, via the conjugate `s*sqrt(Q) - R`. The leading radical
/// term must cancel `R`'s linear term (`s*sqrt(a) + r1 == 0`); then the
/// rationalized numerator `s^2 Q - R^2` over the conjugate sum `~ 2 s sqrt(a) x`
/// gives the decay rate — a nonzero constant, or `K/x` when the constant cancels.
fn sqrt_quadratic_plus_remainder_asymptotic(
    ctx: &Context,
    sqrt_term: &(BigRational, ExprId),
    remainder: &[BigRational],
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    let (scale, radicand) = sqrt_term;
    let q = Polynomial::from_expr(ctx, *radicand, var_name).ok()?;
    if q.degree() != 2 {
        return None;
    }
    let zero = BigRational::from_integer(BigInt::from(0));
    let a = q.coeffs.get(2)?.clone();
    if !a.is_positive() {
        return None;
    }
    let b = q.coeffs.get(1).cloned().unwrap_or_else(|| zero.clone());
    let c = q.coeffs.first().cloned().unwrap_or_else(|| zero.clone());
    // Remainder must be at most linear.
    if remainder.iter().skip(2).any(|coeff| !coeff.is_zero()) {
        return None;
    }
    let r1 = remainder.get(1).cloned().unwrap_or_else(|| zero.clone());
    let r0 = remainder.first().cloned().unwrap_or_else(|| zero.clone());
    let sqrt_a = rational_sqrt(&a)?;
    let lead_sqrt = scale * &sqrt_a; // leading coeff of s*sqrt(Q)
                                     // Leading cancellation: s*sqrt(a) + r1 == 0.
    if &lead_sqrt + &r1 != zero {
        return None;
    }
    let den_lead = &lead_sqrt - &r1; // conjugate sum leading coeff = 2 s sqrt(a) != 0
    if den_lead.is_zero() {
        return None;
    }
    // N = s^2 Q - R^2 = (s^2 b - 2 r1 r0) x + (s^2 c - r0^2); x^2 cancels.
    let two = BigRational::from_integer(BigInt::from(2));
    let scale_sq = scale * scale;
    let n1 = &scale_sq * &b - &two * &r1 * &r0;
    let n0 = &scale_sq * &c - &r0 * &r0;
    if !n1.is_zero() {
        Some((n1 / den_lead, zero)) // difference -> nonzero constant
    } else if !n0.is_zero() {
        Some((n0 / den_lead, BigRational::from_integer(BigInt::from(-1)))) // difference ~ K/x
    } else {
        None // difference asymptotically 0; decline (degenerate input)
    }
}

/// Asymptotic of `s1*sqrt(P) + s2*sqrt(Q)` (a sqrt-sqrt difference) via the
/// conjugate `s1*sqrt(P) - s2*sqrt(Q)`, requiring equal degree `n` in `{1,2}`
/// and a leading cancellation `s1*sqrt(lead_P) + s2*sqrt(lead_Q) == 0`. The decay
/// rate is the leading term of `s1^2 P - s2^2 Q` over `~ 2 s1 sqrt(lead_P) x^(n/2)`.
fn two_sqrt_difference_asymptotic(
    ctx: &Context,
    left: &(BigRational, ExprId),
    right: &(BigRational, ExprId),
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    let (scale_l, rad_l) = left;
    let (scale_r, rad_r) = right;
    let p = Polynomial::from_expr(ctx, *rad_l, var_name).ok()?;
    let q = Polynomial::from_expr(ctx, *rad_r, var_name).ok()?;
    let n = p.degree();
    if n == 0 || n > 2 || q.degree() != n {
        return None;
    }
    let lead_p = p.coeffs.get(n)?.clone();
    let lead_q = q.coeffs.get(n)?.clone();
    if !lead_p.is_positive() || !lead_q.is_positive() {
        return None;
    }
    let sqrt_lead_p = rational_sqrt(&lead_p)?;
    let sqrt_lead_q = rational_sqrt(&lead_q)?;
    let zero = BigRational::from_integer(BigInt::from(0));
    // Leading cancellation: s1*sqrt(lead_P) + s2*sqrt(lead_Q) == 0.
    if scale_l * &sqrt_lead_p + scale_r * &sqrt_lead_q != zero {
        return None;
    }
    let den_lead = scale_l * &sqrt_lead_p - scale_r * &sqrt_lead_q; // = 2 s1 sqrt(lead_P)
    if den_lead.is_zero() {
        return None;
    }
    // N = s1^2 P - s2^2 Q; the x^n term cancels, take the next nonzero term.
    let scale_l_sq = scale_l * scale_l;
    let scale_r_sq = scale_r * scale_r;
    let mut n_lead = zero.clone();
    let mut n_deg: i64 = -1;
    for k in (0..n).rev() {
        let pk = p.coeffs.get(k).cloned().unwrap_or_else(|| zero.clone());
        let qk = q.coeffs.get(k).cloned().unwrap_or_else(|| zero.clone());
        let coeff_k = &scale_l_sq * &pk - &scale_r_sq * &qk;
        if !coeff_k.is_zero() {
            n_lead = coeff_k;
            n_deg = k as i64;
            break;
        }
    }
    if n_deg < 0 {
        return None; // difference asymptotically 0; decline
    }
    let n_half = BigRational::new(BigInt::from(n as i64), BigInt::from(2));
    let exp = BigRational::from_integer(BigInt::from(n_deg)) - n_half;
    Some((n_lead / den_lead, exp))
}

/// Asymptotic leading term `coeff * x^exp` of a cube-root conjugate difference
/// `s*cbrt(P) + R` (with `R` the polynomial remainder) as `x -> +inf`, when the
/// leading terms cancel so it decays. Rationalizing by `A^2 + A L + L^2` (with
/// `A = s*cbrt(P)`, `L = -R`): `A - L = (A^3 - L^3)/(A^2 + A L + L^2) =
/// (s^3 P - L^3)/(~ 3 d^2 x^2)`, where the leading cancellation forces
/// `s*cbrt(a) = d` (`a` the leading coeff of the cubic `P`, `d` the slope of L).
pub(super) fn cbrt_difference_asymptotic_at_pos_inf(
    ctx: &Context,
    diff: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    let mut terms: Vec<(ExprId, bool)> = Vec::new();
    collect_signed_add_terms(ctx, diff, true, &mut terms);
    let zero = BigRational::from_integer(BigInt::from(0));
    let mut cbrt_part: Option<(BigRational, ExprId)> = None;
    let mut remainder: Vec<BigRational> = Vec::new();
    for (term, positive) in terms {
        if let Some((scale, radicand)) = scaled_cube_root_base(ctx, term) {
            if cbrt_part.is_some() {
                return None; // only a single cube-root term is supported
            }
            cbrt_part = Some((if positive { scale } else { -scale }, radicand));
        } else {
            let poly = Polynomial::from_expr(ctx, term, var_name).ok()?;
            for (i, c) in poly.coeffs.iter().enumerate() {
                if remainder.len() <= i {
                    remainder.resize(i + 1, zero.clone());
                }
                remainder[i] = if positive {
                    &remainder[i] + c
                } else {
                    &remainder[i] - c
                };
            }
        }
    }
    let (scale, radicand) = cbrt_part?;
    let p = Polynomial::from_expr(ctx, radicand, var_name).ok()?;
    if p.degree() != 3 {
        return None;
    }
    let a = p.coeffs.get(3)?.clone();
    if !a.is_positive() {
        return None;
    }
    // Remainder must be at most linear.
    if remainder.iter().skip(2).any(|coeff| !coeff.is_zero()) {
        return None;
    }
    let r1 = remainder.get(1).cloned().unwrap_or_else(|| zero.clone());
    let r0 = remainder.first().cloned().unwrap_or_else(|| zero.clone());
    let cbrt_a = rational_cbrt_exact(&a)?;
    let lead = &scale * &cbrt_a; // leading coeff of s*cbrt(P)
                                 // Leading cancellation: s*cbrt(a) + r1 == 0.
    if &lead + &r1 != zero {
        return None;
    }
    // L = -R = d x + e with d = -r1 = lead (nonzero), e = -r0.
    let d = lead;
    if d.is_zero() {
        return None;
    }
    let e = -r0;
    // N = s^3 P - L^3 = s^3 P - (d x + e)^3; the x^3 term cancels, so N has
    // degree <= 2 and N / (3 d^2 x^2) gives the decay: N's x^2 term -> a nonzero
    // constant (exp 0), its x term -> K/x (exp -1), its constant -> K/x^2 (exp -2).
    let three = BigRational::from_integer(BigInt::from(3));
    let scale_cubed = &scale * &scale * &scale;
    let c2 = p.coeffs.get(2).cloned().unwrap_or_else(|| zero.clone());
    let c1 = p.coeffs.get(1).cloned().unwrap_or_else(|| zero.clone());
    let c0 = p.coeffs.first().cloned().unwrap_or_else(|| zero.clone());
    let n2 = &scale_cubed * &c2 - &three * &d * &d * &e;
    let n1 = &scale_cubed * &c1 - &three * &d * &e * &e;
    let n0 = &scale_cubed * &c0 - &e * &e * &e;
    let den_lead = &three * &d * &d; // conjugate sum ~ 3 d^2 x^2
    if !n2.is_zero() {
        Some((n2 / den_lead, zero)) // difference -> nonzero constant
    } else if !n1.is_zero() {
        Some((n1 / den_lead, BigRational::from_integer(BigInt::from(-1)))) // ~ K/x
    } else if !n0.is_zero() {
        Some((n0 / den_lead, BigRational::from_integer(BigInt::from(-2)))) // ~ K/x^2
    } else {
        None // difference asymptotically 0; decline
    }
}

/// Asymptotic leading term `coeff * x^exp` of a general `n`-th-root conjugate
/// difference `s*(P)^(1/n) + R` (R the polynomial remainder, P degree n) as
/// `x -> +inf`, when the leading terms cancel. Rationalizing by the n-term
/// conjugate `a^(n-1)+...+b^(n-1)` (leading `n d^(n-1) x^(n-1)`) gives the decay
/// from `N = s^n P - L^n` (L = -R, degree <= n-1 once `x^n` cancels). Generalizes
/// the sqrt (n=2) and cbrt (n=3) rules to the Pow form `(P)^(1/n)`.
pub(super) fn nth_root_difference_asymptotic_at_pos_inf(
    ctx: &Context,
    diff: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    let mut terms: Vec<(ExprId, bool)> = Vec::new();
    collect_signed_add_terms(ctx, diff, true, &mut terms);
    let zero = BigRational::from_integer(BigInt::from(0));
    let mut root_part: Option<(BigRational, ExprId, u32)> = None; // (signed scale, P, n)
    let mut remainder: Vec<BigRational> = Vec::new();
    for (term, positive) in terms {
        if let Some((scale, radicand, n)) = scaled_nth_root_pow_base(ctx, term) {
            if root_part.is_some() {
                return None;
            }
            root_part = Some((if positive { scale } else { -scale }, radicand, n));
        } else {
            let poly = Polynomial::from_expr(ctx, term, var_name).ok()?;
            for (i, c) in poly.coeffs.iter().enumerate() {
                if remainder.len() <= i {
                    remainder.resize(i + 1, zero.clone());
                }
                remainder[i] = if positive {
                    &remainder[i] + c
                } else {
                    &remainder[i] - c
                };
            }
        }
    }
    let (scale, radicand, n) = root_part?;
    let p = Polynomial::from_expr(ctx, radicand, var_name).ok()?;
    let n_usize = n as usize;
    if p.degree() != n_usize {
        return None;
    }
    let a = p.coeffs.get(n_usize)?.clone();
    if !a.is_positive() {
        return None;
    }
    // L = -R must be at most linear (R linear).
    if remainder.iter().skip(2).any(|coeff| !coeff.is_zero()) {
        return None;
    }
    let r1 = remainder.get(1).cloned().unwrap_or_else(|| zero.clone());
    let r0 = remainder.first().cloned().unwrap_or_else(|| zero.clone());
    let root_a = rational_nth_root(&a, n)?;
    let lead = &scale * &root_a; // leading coeff of s*(P)^(1/n)
                                 // Leading cancellation: s*a^(1/n) + r1 == 0.
    if &lead + &r1 != zero {
        return None;
    }
    let d = lead; // = -r1 (slope of L), nonzero since a > 0
    if d.is_zero() {
        return None;
    }
    let e = -r0;
    // N = s^n P - L^n with L = d x + e; the x^n term cancels. Read N's coefficients
    // directly: N_k = s^n P_k - C(n,k) d^k e^(n-k), for k from n-1 down to 0.
    let scale_n = pow_rational(&scale, n);
    let mut n_lead = zero.clone();
    let mut n_deg: i64 = -1;
    for k in (0..n_usize).rev() {
        let pk = p.coeffs.get(k).cloned().unwrap_or_else(|| zero.clone());
        let l_k = binomial_rational(n, k as u32)
            * pow_rational(&d, k as u32)
            * pow_rational(&e, n - k as u32);
        let coeff_k = &scale_n * &pk - &l_k;
        if !coeff_k.is_zero() {
            n_lead = coeff_k;
            n_deg = k as i64;
            break;
        }
    }
    if n_deg < 0 {
        return None;
    }
    let den_lead = BigRational::from_integer(BigInt::from(n as i64)) * pow_rational(&d, n - 1);
    let exp = BigRational::from_integer(BigInt::from(n_deg))
        - BigRational::from_integer(BigInt::from((n - 1) as i64));
    Some((n_lead / den_lead, exp))
}
