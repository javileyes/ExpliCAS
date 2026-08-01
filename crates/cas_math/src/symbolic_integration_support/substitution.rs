//! `symbolic_integration_support`: familia `substitution`.
//!
//! Ver la cabecera de `symbolic_integration_support.rs` para el contexto.

use super::*;

/// Shared tail of the `u = sqrt(x)` substitution. Given the delegate integrand
/// `u_integrand = u * g(u^2)` (already expressed in the original variable, where
/// `g` is the original integrand), integrate it, back-substitute `u -> sqrt(x)`,
/// fold the nested numeric powers the back-substitution introduces, and scale by
/// 2 (the `dx = 2u du` factor). `sqrt_arg` is the `sqrt(x)` expression to
/// substitute back. Self-gates to an honest residual if the delegated integral
/// does not resolve.
pub(super) fn complete_sqrt_substitution(
    ctx: &mut Context,
    u_integrand: ExprId,
    sqrt_arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    let h = integrate_symbolic_expr(ctx, u_integrand, var)?;

    let var_expr = ctx.var(var);
    let h_sub = crate::substitute::substitute_power_aware(
        ctx,
        h,
        var_expr,
        sqrt_arg,
        crate::substitute::SubstituteOptions::exact(),
    );
    // The substitution turns x^k into (sqrt(x))^k = Pow(Pow(x,1/2),k); fold those
    // nested numeric powers so the result never displays the ambiguous
    // `x^(1/2)^2` (which re-parses as x^(1/4), not x).
    let h_folded = fold_nested_numeric_powers(ctx, h_sub);
    Some(scale_rational_term(
        ctx,
        BigRational::from_integer(2.into()),
        h_folded,
    ))
}

/// Integrate `scale·sin(arg)^p·cos(arg)^q` (arg affine in `var`, ONE of `p,q` a positive ODD integer,
/// the other ANY integer) by the `u = cos` / `u = sin` substitution:
/// `∫ sin^p cos^q dx = ∓(1/a)∫ (1−u²)^((odd−1)/2) · u^companion du`, integrated termwise by the power
/// rule (`u^j → u^(j+1)/(j+1)`, `ln|u|` for `j = −1`), then `u → cos/sin(arg)`. Fills the reciprocal
/// gap `sin(x)/cos(x)^n → 1/((n−1)cos^(n−1)x)` (n ≥ 4) the polynomial-only odd-power owner misses.
/// Last-resort fallback, so the working `sec`/`csc`/`tan²` forms keep their dedicated owners.
pub(super) fn trig_odd_power_companion_substitution_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (scale, arg, sin_pow, cos_pow) = extract_sin_cos_power_monomial(ctx, expr, var)?;
    let (u_is_cos, odd_pow, companion_pow) = if sin_pow > 0 && sin_pow % 2 == 1 {
        (true, sin_pow, cos_pow)
    } else if cos_pow > 0 && cos_pow % 2 == 1 {
        (false, cos_pow, sin_pow)
    } else {
        return None;
    };
    let (a_expr, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a_expr)?;
    if a.is_zero() {
        return None;
    }

    let k = (odd_pow - 1) / 2;
    let sign = if u_is_cos {
        -BigRational::one()
    } else {
        BigRational::one()
    };
    let outer = scale * sign / a;
    let u_builtin = if u_is_cos {
        BuiltinFn::Cos
    } else {
        BuiltinFn::Sin
    };
    let u = ctx.call_builtin(u_builtin, vec![arg]);

    let mut acc: Option<ExprId> = None;
    for i in 0..=k {
        let binom = binomial_i64(k, i);
        let mut c = BigRational::from_integer(binom.into());
        if i % 2 != 0 {
            c = -c;
        }
        let exponent = companion_pow + 2 * i;
        let piece = if exponent == -1 {
            let abs_u = ctx.call_builtin(BuiltinFn::Abs, vec![u]);
            let ln = ctx.call_builtin(BuiltinFn::Ln, vec![abs_u]);
            scale_rational_term(ctx, &outer * &c, ln)
        } else {
            let next = exponent + 1;
            let pow = build_signed_integer_power(ctx, u, next);
            scale_rational_term(
                ctx,
                &outer * &c / BigRational::from_integer(next.into()),
                pow,
            )
        };
        acc = Some(match acc {
            None => piece,
            Some(previous) => ctx.add(Expr::Add(previous, piece)),
        });
    }
    acc
}

pub fn integrate_symbolic_is_hyperbolic_quotient_substitution_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            hyperbolic_log_derivative_ratio_antiderivative(ctx, num, den, var).is_some()
                || hyperbolic_tanh_reciprocal_log_sinh_antiderivative(ctx, num, den, var).is_some()
                || hyperbolic_reciprocal_square_antiderivative(ctx, num, den, var).is_some()
        }
        Expr::Mul(left, right) => {
            (!contains_named_var(ctx, left, var)
                && integrate_symbolic_is_hyperbolic_quotient_substitution_target(ctx, right, var))
                || (!contains_named_var(ctx, right, var)
                    && integrate_symbolic_is_hyperbolic_quotient_substitution_target(
                        ctx, left, var,
                    ))
        }
        Expr::Neg(inner) => {
            integrate_symbolic_is_hyperbolic_quotient_substitution_target(ctx, inner, var)
        }
        _ => false,
    }
}

pub fn integrate_symbolic_is_trig_quotient_substitution_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            trig_log_derivative_ratio_antiderivative(ctx, num, den, var).is_some()
                || nested_trig_log_derivative_antiderivative(ctx, num, den, var).is_some()
                || polynomial_reciprocal_trig_square_antiderivative(ctx, num, den, var).is_some()
                || polynomial_trig_reciprocal_derivative_antiderivative(ctx, num, den, var)
                    .is_some()
                || polynomial_trig_reciprocal_factor_antiderivative(ctx, num, den, var).is_some()
                || trig_tan_fourth_quotient_antiderivative(ctx, num, den, var).is_some()
                || trig_cot_fourth_quotient_antiderivative(ctx, num, den, var).is_some()
                || trig_tan_sixth_quotient_antiderivative(ctx, num, den, var).is_some()
                || trig_cot_sixth_quotient_antiderivative(ctx, num, den, var).is_some()
                || trig_tan_eighth_quotient_antiderivative(ctx, num, den, var).is_some()
                || trig_cot_eighth_quotient_antiderivative(ctx, num, den, var).is_some()
                || trig_sec_sixth_quotient_antiderivative(ctx, num, den, var).is_some()
                || trig_csc_sixth_quotient_antiderivative(ctx, num, den, var).is_some()
                || trig_sec_eighth_quotient_antiderivative(ctx, num, den, var).is_some()
                || trig_csc_eighth_quotient_antiderivative(ctx, num, den, var).is_some()
        }
        Expr::Mul(left, right) => {
            (!contains_named_var(ctx, left, var)
                && integrate_symbolic_is_trig_quotient_substitution_target(ctx, right, var))
                || (!contains_named_var(ctx, right, var)
                    && integrate_symbolic_is_trig_quotient_substitution_target(ctx, left, var))
        }
        Expr::Neg(inner) => {
            integrate_symbolic_is_trig_quotient_substitution_target(ctx, inner, var)
        }
        _ => false,
    }
}

pub fn integrate_symbolic_is_trig_log_substitution_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    if let Some(inner) = constant_scaled_integrand_inner(ctx, expr, var) {
        return integrate_symbolic_is_trig_log_substitution_target(ctx, inner, var);
    }

    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let Some(builtin) = ctx.builtin_of(fn_id) else {
                return false;
            };
            trig_log_antiderivative(ctx, builtin, args[0], var).is_some()
        }
        Expr::Div(num, den) => {
            polynomial_reciprocal_trig_log_antiderivative(ctx, num, den, var).is_some()
                || reciprocal_trig_log_antiderivative(ctx, num, den, var).is_some()
        }
        _ => false,
    }
}

pub(super) fn arctan_unary_derivative_substitution_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    if let Expr::Neg(inner) = ctx.get(num).clone() {
        let integral = arctan_unary_derivative_substitution_antiderivative(ctx, inner, den, var)?;
        return Some(ctx.add(Expr::Neg(integral)));
    }

    let arg = positive_one_plus_square_arg(ctx, den)?;
    let (fn_id, args) = match ctx.get(arg).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return None,
    };
    let inner = args[0];
    if !contains_named_var(ctx, inner, var) {
        return None;
    }

    let (companion_builtin, derivative_sign) = match ctx.builtin_of(fn_id)? {
        BuiltinFn::Sin => (BuiltinFn::Cos, BigRational::one()),
        BuiltinFn::Cos => (BuiltinFn::Sin, -BigRational::one()),
        BuiltinFn::Sinh => (BuiltinFn::Cosh, BigRational::one()),
        BuiltinFn::Cosh => (BuiltinFn::Sinh, BigRational::one()),
        _ => return None,
    };

    let factors = mul_leaves(ctx, num);
    let (companion_index, _) = factors.iter().enumerate().find(|(_, factor)| {
        unary_builtin_arg(ctx, **factor, companion_builtin)
            .is_some_and(|companion_arg| compare_expr(ctx, companion_arg, inner) == Ordering::Equal)
    })?;
    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != companion_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let inner_poly = Polynomial::from_expr(ctx, inner, var).ok()?;
    let derivative = inner_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)? * derivative_sign;
    if scale.is_zero() {
        return None;
    }

    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arg]);
    if scale.is_one() {
        return Some(arctan);
    }
    if scale == -BigRational::one() {
        return Some(ctx.add(Expr::Neg(arctan)));
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, arctan))
}

/// `p(x) * sin(g(x))^2` / `p(x) * cos(g(x))^2` with a NON-affine inner `g` (the
/// affine case is owned by `polynomial_times_trig_square_antiderivative`). Same
/// half-angle reduction `sin^2(g) = (1 - cos(2g))/2`, distributed by the
/// cofactor; the resulting `p*cos(2g)` terms integrate only when the cofactor
/// supplies the substitution derivative -- `x*sin(x^2)^2 -> x/2 - (x/2)cos(2x^2)`
/// is elementary via `u = x^2`. Delegation to `integrate_symbolic_expr`
/// self-gates: a cofactor that is not substitution-amenable (`x^2*sin(x^2)^2` is
/// non-elementary / Fresnel) leaves a term unintegrable, so the Add/Sub linearity
/// returns None via `?` and the product stays an honest residual.
pub(super) fn polynomial_times_trig_square_substitution_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }
    let mut trig: Option<(usize, BuiltinFn, ExprId)> = None;
    for (i, factor) in factors.iter().enumerate() {
        let Expr::Pow(base, exp) = ctx.get(*factor).clone() else {
            continue;
        };
        if !is_number(ctx, exp, 2) {
            continue;
        }
        let Expr::Function(fn_id, args) = ctx.get(base).clone() else {
            continue;
        };
        if args.len() != 1 {
            continue;
        }
        let builtin = match ctx.builtin_of(fn_id) {
            Some(b @ (BuiltinFn::Sin | BuiltinFn::Cos)) => b,
            _ => continue,
        };
        // Non-affine inner only: an affine argument with a nonzero rational slope
        // is owned by the affine rule (which runs first).
        if !contains_named_var(ctx, args[0], var) {
            continue;
        }
        if let Some((slope, _)) = get_linear_coeffs(ctx, args[0], var) {
            if rational_constant_value(ctx, slope).is_some_and(|s| !s.is_zero()) {
                continue;
            }
        }
        if trig.is_some() {
            return None; // more than one trig-square factor is out of scope
        }
        trig = Some((i, builtin, args[0]));
    }
    let (trig_idx, builtin, g) = trig?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != trig_idx)
        .map(|(_, f)| *f)
        .collect();
    let cofactor = build_balanced_mul(ctx, &cofactor_factors);
    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    if cofactor_poly.degree() == 0 {
        return None; // a constant cofactor is the bare (non-elementary) case
    }

    let two = ctx.num(2);
    let two_g = mul2_raw(ctx, two, g);
    let cos_2g = ctx.call_builtin(BuiltinFn::Cos, vec![two_g]);
    let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let half_cofactor = mul2_raw(ctx, half, cofactor);
    let cofactor_cos = mul2_raw(ctx, cofactor, cos_2g);
    let half_cofactor_cos = mul2_raw(ctx, half, cofactor_cos);
    let rewritten = match builtin {
        BuiltinFn::Sin => ctx.add(Expr::Sub(half_cofactor, half_cofactor_cos)),
        BuiltinFn::Cos => ctx.add(Expr::Add(half_cofactor, half_cofactor_cos)),
        _ => return None,
    };
    integrate_symbolic_expr(ctx, rewritten, var)
}

pub(super) fn exponential_rational_substitution_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let mut slopes: Vec<BigRational> = Vec::new();
    if !collect_exponential_rational_slopes(ctx, expr, var, &mut slopes) || slopes.is_empty() {
        return None;
    }
    let mut numer_gcd = slopes[0].numer().clone();
    let mut denom_lcm = slopes[0].denom().clone();
    for slope in &slopes[1..] {
        numer_gcd = num_integer::Integer::gcd(&numer_gcd, slope.numer());
        denom_lcm = num_integer::Integer::lcm(&denom_lcm, slope.denom());
    }
    if numer_gcd.is_zero() {
        return None;
    }
    let c = BigRational::new(numer_gcd, denom_lcm);

    let used = cas_ast::collect_variables(ctx, expr);
    let u_name = ["u", "u_", "u_sub"]
        .iter()
        .find(|candidate| !used.contains(**candidate) && *candidate != &var)?
        .to_string();

    let (num, mut den) = exponential_rational_function_parts(ctx, expr, var, &c, &u_name)?;
    // Divide by c*u (du = c*u dx).
    let mut cu = Polynomial::zero(u_name.clone());
    cu.coeffs = vec![BigRational::zero(), c.clone()];
    den = den.mul(&cu);
    if num.degree() > 10 || den.degree() > 10 || den.is_zero() {
        return None;
    }
    let numerator_expr = polynomial_to_expr(ctx, &num, &u_name);
    let denominator_expr = polynomial_to_expr(ctx, &den, &u_name);
    let integrand_u = ctx.add(Expr::Div(numerator_expr, denominator_expr));
    let integral_u = integrate_symbolic_expr(ctx, integrand_u, &u_name)?;
    // Partial-fraction owners wrap in an internal hold; unwrap so the
    // back-substituted antiderivative stays differentiable downstream.
    let integral_u = cas_ast::hold::unwrap_internal_hold(ctx, integral_u);

    let var_expr = ctx.var(var);
    let scaled_var = scale_rational_term(ctx, c, var_expr);
    let e_const = ctx.add(Expr::Constant(cas_ast::Constant::E));
    let replacement = ctx.add(Expr::Pow(e_const, scaled_var));
    let target = ctx.var(&u_name);
    let substituted = crate::substitute::substitute_power_aware(
        ctx,
        integral_u,
        target,
        replacement,
        crate::substitute::SubstituteOptions::exact(),
    );
    Some(strip_redundant_exponential_abs(ctx, substituted))
}

/// Quartic denominators (a x^2 + b)/(x^4 + 1) via the symmetric
/// substitution. Split a x^2 + b = c1 (x^2+1) + c2 (x^2-1); the
/// (x^2+1) piece divides by x^2 to (1+1/x^2)/((x-1/x)^2+2) so u=x-1/x
/// gives int du/(u^2+2) -> arctan, and the (x^2-1) piece gives
/// u=x+1/x, int du/(u^2-2) -> the irrational-root log form. Covers the
/// famous 1/(x^4+1), x^2/(x^4+1), (x^2+1)/(x^4+1), (x^2-1)/(x^4+1)
/// without irrational-coefficient factorization. Ordered after the
/// rational owners (x^4+4 etc. factor rationally and keep theirs).
pub(super) fn quartic_symmetric_substitution_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let den_poly = Polynomial::from_expr(ctx, den, var).ok()?;
    // Denominator exactly x^4 + 1.
    if den_poly.degree() != 4 {
        return None;
    }
    let expected = [
        BigRational::from_integer(1.into()),
        BigRational::zero(),
        BigRational::zero(),
        BigRational::zero(),
        BigRational::from_integer(1.into()),
    ];
    if den_poly.coeffs.len() != 5 || den_poly.coeffs != expected {
        return None;
    }
    // Numerator a x^2 + b (only even powers, degree <= 2).
    let num_poly = Polynomial::from_expr(ctx, num, var).ok()?;
    if num_poly.degree() > 2 {
        return None;
    }
    let coeff = |i: usize| {
        num_poly
            .coeffs
            .get(i)
            .cloned()
            .unwrap_or_else(BigRational::zero)
    };
    if !coeff(1).is_zero() {
        return None;
    }
    let a = coeff(2);
    let b = coeff(0);
    if a.is_zero() && b.is_zero() {
        return None;
    }
    let two = BigRational::from_integer(2.into());
    let c1 = (&a + &b) / &two; // weight of the (x^2+1) -> arctan piece
    let c2 = (&a - &b) / &two; // weight of the (x^2-1) -> log piece

    let used = cas_ast::collect_variables(ctx, expr);
    let u_name = ["u", "u_", "u_sub"]
        .iter()
        .find(|candidate| !used.contains(**candidate) && *candidate != &var)?
        .to_string();
    let var_expr = ctx.var(var);
    let x_squared = {
        let two_expr = ctx.num(2);
        ctx.add(Expr::Pow(var_expr, two_expr))
    };

    let mut pieces: Vec<ExprId> = Vec::new();

    if !c1.is_zero() {
        // u = x - 1/x = (x^2 - 1)/x ; int 1/(u^2 + 2) du.
        let u = ctx.var(&u_name);
        let u_sq = {
            let two_expr = ctx.num(2);
            ctx.add(Expr::Pow(u, two_expr))
        };
        let two_const = ctx.num(2);
        let denom = ctx.add(Expr::Add(u_sq, two_const));
        let one = ctx.num(1);
        let integrand = ctx.add(Expr::Div(one, denom));
        let integral_u = integrate_symbolic_expr(ctx, integrand, &u_name)?;
        let integral_u = cas_ast::hold::unwrap_internal_hold(ctx, integral_u);
        let one_b = ctx.num(1);
        let numerator = ctx.add(Expr::Sub(x_squared, one_b));
        let replacement = ctx.add(Expr::Div(numerator, var_expr));
        let target = ctx.var(&u_name);
        let substituted = crate::substitute::substitute_power_aware(
            ctx,
            integral_u,
            target,
            replacement,
            crate::substitute::SubstituteOptions::exact(),
        );
        pieces.push(scale_rational_term(ctx, c1, substituted));
    }

    if !c2.is_zero() {
        // u = x + 1/x = (x^2 + 1)/x ; int 1/(u^2 - 2) du.
        let u = ctx.var(&u_name);
        let u_sq = {
            let two_expr = ctx.num(2);
            ctx.add(Expr::Pow(u, two_expr))
        };
        let two_const = ctx.num(2);
        let denom = ctx.add(Expr::Sub(u_sq, two_const));
        let one = ctx.num(1);
        let integrand = ctx.add(Expr::Div(one, denom));
        let integral_u = integrate_symbolic_expr(ctx, integrand, &u_name)?;
        let integral_u = cas_ast::hold::unwrap_internal_hold(ctx, integral_u);
        let one_b = ctx.num(1);
        let numerator = ctx.add(Expr::Add(x_squared, one_b));
        let replacement = ctx.add(Expr::Div(numerator, var_expr));
        let target = ctx.var(&u_name);
        let substituted = crate::substitute::substitute_power_aware(
            ctx,
            integral_u,
            target,
            replacement,
            crate::substitute::SubstituteOptions::exact(),
        );
        pieces.push(scale_rational_term(ctx, c2, substituted));
    }

    if pieces.is_empty() {
        return None;
    }
    Some(build_balanced_add(ctx, &pieces))
}

/// Products sin(k x)^m cos(k x)^n sharing ONE linear argument (rational
/// k != 0, zero offset) with at least one ODD power: substitute
/// u = sin(k x) when n is odd (du = k cos dx, cos^n = cos (1-u^2)^((n-1)/2))
/// or u = cos(k x) when m is odd, giving a polynomial integrand in u
/// delegated to the polynomial integrator. Covers sin^2 cos^3,
/// sin^4 cos^3, sin^3 cos^2, sin^5 cos^2 and shared multiples like
/// sin(2x)^3 cos(2x)^2. The f^n f' single-factor cases (sin^3 cos),
/// the both-even power-reduction cases (sin^2 cos^2) and pure single
/// powers (cos^3) keep their existing owners - this route is ordered
/// AFTER them and only sees the genuinely residual mixed products.
pub(super) fn mixed_trig_power_substitution_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let mut sin_power: i64 = 0;
    let mut cos_power: i64 = 0;
    let mut slope: Option<BigRational> = None;
    if !collect_mixed_trig_powers(ctx, expr, var, &mut sin_power, &mut cos_power, &mut slope) {
        return None;
    }
    let k = slope?;
    // Both factors must carry power >= 2: products with a power-1
    // companion factor (sin^m cos, sin cos^n) are the f^n f' pattern
    // and keep their owners; single powers and both-even products are
    // owned too. This route claims only the genuinely residual mixed
    // products with at least one ODD power.
    if sin_power < 2 || cos_power < 2 {
        return None;
    }
    let sin_odd = sin_power % 2 == 1;
    let cos_odd = cos_power % 2 == 1;
    if !(sin_odd || cos_odd) {
        return None;
    }

    let used = cas_ast::collect_variables(ctx, expr);
    let u_name = ["u", "u_", "u_sub"]
        .iter()
        .find(|candidate| !used.contains(**candidate) && *candidate != &var)?
        .to_string();

    // Prefer substituting against the odd power. When BOTH are odd,
    // u = sin (cos carries the spare factor) keeps displays compact.
    let (substitute_sine, kept_power, spare_half) = if cos_odd {
        (true, sin_power, (cos_power - 1) / 2)
    } else {
        (false, cos_power, (sin_power - 1) / 2)
    };

    // integrand in u: u^kept_power * (1 - u^2)^spare_half, scaled by
    // +1/k (u = sin) or -1/k (u = cos).
    let mut poly = vec![BigRational::zero(); (kept_power + 2 * spare_half + 1) as usize];
    // (1 - u^2)^spare_half = sum_j C(spare_half, j) (-1)^j u^(2j).
    let mut binom = BigRational::from_integer(1.into());
    for j in 0..=spare_half {
        let exponent = (kept_power + 2 * j) as usize;
        let sign = if j % 2 == 0 {
            BigRational::from_integer(1.into())
        } else {
            BigRational::from_integer((-1).into())
        };
        poly[exponent] += &binom * &sign;
        // Update binomial coefficient C(spare_half, j+1).
        if j < spare_half {
            binom = &binom * BigRational::from_integer((spare_half - j).into())
                / BigRational::from_integer((j + 1).into());
        }
    }
    let scale = if substitute_sine {
        BigRational::from_integer(1.into()) / &k
    } else {
        -BigRational::from_integer(1.into()) / &k
    };
    for coeff in &mut poly {
        *coeff *= &scale;
    }

    let mut poly_struct = Polynomial::zero(u_name.clone());
    poly_struct.coeffs = poly;
    let integrand_u = polynomial_to_expr(ctx, &poly_struct, &u_name);
    let integral_u = integrate_symbolic_expr(ctx, integrand_u, &u_name)?;
    let integral_u = cas_ast::hold::unwrap_internal_hold(ctx, integral_u);

    let var_expr = ctx.var(var);
    let arg = scale_rational_term(ctx, k, var_expr);
    let replacement = if substitute_sine {
        ctx.call_builtin(BuiltinFn::Sin, vec![arg])
    } else {
        ctx.call_builtin(BuiltinFn::Cos, vec![arg])
    };
    let target = ctx.var(&u_name);
    Some(crate::substitute::substitute_power_aware(
        ctx,
        integral_u,
        target,
        replacement,
        crate::substitute::SubstituteOptions::exact(),
    ))
}

/// Rational functions of sin(k x) and cos(k x) sharing ONE linear
/// argument (rational k != 0, zero offset) with rational coefficients:
/// Weierstrass substitution t = tan(k x / 2), sin = 2t/(1+t^2),
/// cos = (1-t^2)/(1+t^2), dx = 2 dt / (k (1+t^2)). Builds the SINGLE
/// flattened quotient via Polynomial arithmetic and delegates to the
/// rational owners. Covers 1/(2+cos x), 1/(1+sin x), 1/(sin x + cos x),
/// sin(x)/(1+sin x), 1/(3+2 cos x). Mixed multiples (sin x with
/// cos 2x), phase offsets, tan/sec atoms and trig-polynomial mixes
/// decline. Ordered AFTER the specialized trig owners so their pinned
/// displays (sec, csc, odd/even powers) survive.
pub(super) fn weierstrass_rational_substitution_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let mut slopes: Vec<BigRational> = Vec::new();
    if !collect_weierstrass_rational_slopes(ctx, expr, var, &mut slopes) || slopes.is_empty() {
        return None;
    }
    let k = slopes[0].clone();
    if slopes[1..].iter().any(|slope| *slope != k) {
        return None;
    }

    let used = cas_ast::collect_variables(ctx, expr);
    let u_name = ["u", "u_", "u_sub"]
        .iter()
        .find(|candidate| !used.contains(**candidate) && *candidate != &var)?
        .to_string();

    let (num, den) = weierstrass_rational_function_parts(ctx, expr, var, &u_name)?;
    // dx = 2 dt / (k (1 + t^2)).
    let num = scale_polynomial_rational(&num, &BigRational::from_integer(2.into()));
    let mut one_plus_t2 = Polynomial::zero(u_name.clone());
    one_plus_t2.coeffs = vec![
        BigRational::from_integer(1.into()),
        BigRational::zero(),
        BigRational::from_integer(1.into()),
    ];
    let mut den = scale_polynomial_rational(&den.mul(&one_plus_t2), &k);
    let mut num = num;
    // Weierstrass atoms share (1+t^2) denominators, so num/den are
    // routinely non-coprime (sin/(1+sin) -> 4t(1+t^2)/((1+t^2)^2(t+1)^2))
    // and the rational owners do not cancel: divide out the gcd, monic
    // renormalized (Polynomial::gcd returns arbitrary rational scale).
    let gcd = num.gcd(&den);
    if gcd.degree() >= 1 {
        let lead = gcd.coeffs.last()?.clone();
        if lead.is_zero() {
            return None;
        }
        let monic_gcd =
            scale_polynomial_rational(&gcd, &(BigRational::from_integer(1.into()) / lead));
        let (num_q, num_r) = num.div_rem(&monic_gcd).ok()?;
        let (den_q, den_r) = den.div_rem(&monic_gcd).ok()?;
        if num_r.is_zero() && den_r.is_zero() {
            num = num_q;
            den = den_q;
        }
    }
    if num.degree() > 10 || den.degree() > 10 || den.is_zero() {
        return None;
    }
    let integrand_u = if den.degree() == 0 {
        let scaled = scale_polynomial_rational(
            &num,
            &(BigRational::from_integer(1.into()) / &den.coeffs[0]),
        );
        polynomial_to_expr(ctx, &scaled, &u_name)
    } else {
        let numerator_expr = polynomial_to_expr(ctx, &num, &u_name);
        let denominator_expr = polynomial_to_expr(ctx, &den, &u_name);
        ctx.add(Expr::Div(numerator_expr, denominator_expr))
    };
    let integral_u = integrate_symbolic_expr(ctx, integrand_u, &u_name).or_else(|| {
        // Strictly positive quadratic denominators (1/(2+cos x) ->
        // 2/(t^2+3)) live in the algorithmic backend, not the support
        // owners. Accept backend results ONLY when unconditional: this
        // route has no channel to surface required conditions.
        let config = crate::general_integration_backend::AlgorithmicIntegrationBackendConfig::residual_fallback();
        let candidate = crate::general_integration_backend::try_algorithmic_integration_backend(
            ctx, integrand_u, &u_name, config,
        );
        if !candidate.required_conditions.is_empty() {
            return None;
        }
        candidate.fallback_antiderivative(config)
    })?;
    // Partial-fraction owners wrap in an internal hold; unwrap so the
    // back-substituted antiderivative stays differentiable downstream.
    let integral_u = cas_ast::hold::unwrap_internal_hold(ctx, integral_u);

    let var_expr = ctx.var(var);
    let half_k = k / BigRational::from_integer(2.into());
    let half_angle = scale_rational_term(ctx, half_k, var_expr);
    let replacement = ctx.call_builtin(BuiltinFn::Tan, vec![half_angle]);
    let target = ctx.var(&u_name);
    Some(crate::substitute::substitute_power_aware(
        ctx,
        integral_u,
        target,
        replacement,
        crate::substitute::SubstituteOptions::exact(),
    ))
}

/// Rational functions of x and sqrt(a x + b) with rational a != 0, b:
/// substitute u = sqrt(a x + b), so x = (u^2 - b)/a and dx = (2u/a) du,
/// build the SINGLE flattened quotient num(u)/den(u) via Polynomial
/// arithmetic, integrate in u with the mature rational owners, and
/// back-substitute. Covers x*sqrt(x+1), x^2*sqrt(x+1), x*sqrt(2x-1),
/// sqrt(x)/(1+x), sqrt(x+1)/x and the rationalized 1/(sqrt(x)+1)
/// surface (sqrt(x)-1)/(x-1). Non-rational cofactors (e^sqrt(x)) and
/// mixed radicands decline. Ordered AFTER the specialized Div owners
/// so their pinned displays survive.
pub(super) fn linear_radical_substitution_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let mut radicands: Vec<(ExprId, BigRational, BigRational)> = Vec::new();
    if !collect_linear_radical_radicands(ctx, expr, var, &mut radicands) || radicands.is_empty() {
        return None;
    }
    let (radicand_expr, slope, offset) = radicands[0].clone();
    if slope.is_zero() {
        return None;
    }
    if radicands[1..]
        .iter()
        .any(|(_, a, b)| *a != slope || *b != offset)
    {
        return None;
    }

    let used = cas_ast::collect_variables(ctx, expr);
    let u_name = ["u", "u_", "u_sub"]
        .iter()
        .find(|candidate| !used.contains(**candidate) && *candidate != &var)?
        .to_string();

    let (num, den) =
        linear_radical_rational_function_parts(ctx, expr, var, &slope, &offset, &u_name)?;
    // dx = (2u/a) du.
    let mut two_u = Polynomial::zero(u_name.clone());
    two_u.coeffs = vec![BigRational::zero(), BigRational::from_integer(2.into())];
    let num = num.mul(&two_u);
    let den = scale_polynomial_rational(&den, &slope);
    if num.degree() > 10 || den.degree() > 10 || den.is_zero() {
        return None;
    }
    let integrand_u = if den.degree() == 0 {
        // Constant denominator: hand the owners a plain polynomial, the
        // degenerate Div(p(u), c) quotient has no rational owner.
        let scaled = scale_polynomial_rational(
            &num,
            &(BigRational::from_integer(1.into()) / &den.coeffs[0]),
        );
        polynomial_to_expr(ctx, &scaled, &u_name)
    } else {
        let numerator_expr = polynomial_to_expr(ctx, &num, &u_name);
        let denominator_expr = polynomial_to_expr(ctx, &den, &u_name);
        ctx.add(Expr::Div(numerator_expr, denominator_expr))
    };
    let integral_u = integrate_symbolic_expr(ctx, integrand_u, &u_name)?;
    // Partial-fraction owners wrap in an internal hold; unwrap so the
    // back-substituted antiderivative stays differentiable downstream.
    let integral_u = cas_ast::hold::unwrap_internal_hold(ctx, integral_u);

    let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let replacement = ctx.add(Expr::Pow(radicand_expr, half));
    let target = ctx.var(&u_name);
    let substituted = crate::substitute::substitute_power_aware(
        ctx,
        integral_u,
        target,
        replacement,
        crate::substitute::SubstituteOptions::exact(),
    );
    Some(strip_redundant_sqrt_abs(ctx, substituted))
}

fn polynomial_substitution_kernel_antiderivative(
    ctx: &mut Context,
    kernel: PolynomialSubstitutionKernel,
    arg: ExprId,
    kernel_factor: ExprId,
) -> ExprId {
    if let Some(antiderivative) =
        elementary_polynomial_substitution_kernel_antiderivative(ctx, kernel, arg, kernel_factor)
    {
        return antiderivative;
    }

    match kernel {
        PolynomialSubstitutionKernel::Sec => {
            let one = ctx.num(1);
            sec_csc_log_antiderivative(ctx, BuiltinFn::Sec, arg, one).unwrap_or(kernel_factor)
        }
        PolynomialSubstitutionKernel::Csc => {
            let one = ctx.num(1);
            sec_csc_log_antiderivative(ctx, BuiltinFn::Csc, arg, one).unwrap_or(kernel_factor)
        }
        _ => kernel_factor,
    }
}

fn polynomial_power_substitution_from_base(
    ctx: &mut Context,
    cofactor: ExprId,
    base: ExprId,
    exponent: BigRational,
    var: &str,
) -> Option<ExprId> {
    let negative_one = BigRational::from_integer((-1).into());
    if exponent == negative_one {
        return None;
    }

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let base_poly = Polynomial::from_expr(ctx, base, var).ok()?;
    let derivative = base_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let new_exponent = exponent + BigRational::one();
    if new_exponent.is_zero() {
        return None;
    }

    let coefficient = scale / new_exponent.clone();
    build_scaled_power_antiderivative(ctx, coefficient, base, new_exponent)
}

/// Shared tail for the u-du power routes: build `coefficient · base^new_exponent`,
/// routing negative exponents through the reciprocal-power presentation.
fn build_scaled_power_antiderivative(
    ctx: &mut Context,
    coefficient: BigRational,
    base: ExprId,
    new_exponent: BigRational,
) -> Option<ExprId> {
    if new_exponent < BigRational::zero() {
        return Some(rational_coefficient_times_reciprocal_power(
            ctx,
            coefficient,
            base,
            -new_exponent,
        ));
    }

    let power_exp = ctx.add(Expr::Number(new_exponent));
    let power = ctx.add(Expr::Pow(base, power_exp));
    if coefficient.is_one() {
        return Some(power);
    }

    let coefficient_expr = ctx.add(Expr::Number(coefficient));
    Some(mul2_raw(ctx, coefficient_expr, power))
}

/// Normaliza un factor para la comparación cofactor ≡ s·u′ de las rutas u-du:
/// la derivada CRUDA de `differentiate_symbolic_expr` trae la aritmética de
/// exponentes sin plegar (`u^(2-1)`), y `u^1 ≠ u` estructuralmente. Pliega el
/// exponente racional y quita el `^1`; el resto queda intacto.
fn normalize_power_factor(ctx: &mut Context, factor: ExprId) -> ExprId {
    let Expr::Pow(base, exp) = ctx.get(factor) else {
        return factor;
    };
    let (base, exp) = (*base, *exp);
    // numeric_eval, no views: la derivada cruda trae `2-1` como Sub sin
    // plegar, y el extractor estructural devuelve None sobre eso (lección ya
    // escrita en la skill; mordida aquí una vez más antes de releerla).
    let Some(value) = crate::numeric_eval::as_rational_const(ctx, exp) else {
        return factor;
    };
    if value == BigRational::one() {
        return base;
    }
    if matches!(ctx.get(exp), Expr::Number(_)) {
        return factor;
    }
    let folded = ctx.add(Expr::Number(value));
    ctx.add(Expr::Pow(base, folded))
}

fn normalize_power_factors(ctx: &mut Context, factors: &mut [ExprId]) {
    for f in factors.iter_mut() {
        *f = normalize_power_factor(ctx, *f);
    }
}

/// Tabla u-du con u SIMBÓLICA: `∫ s·u′·F(u) dx = s·G(u)` para
/// F ∈ {exp, sin, cos, sinh, cosh} y `u` NO polinómica (los kernels
/// polinómicos tienen dueño en `polynomial_derivative_table`).
///
/// El caso que la motiva: `∫cos(x)·cos(sin(x))` — la regla product-to-sum
/// trataba `sin(x)` como ángulo independiente, destrozaba la forma u-du y el
/// residual quedaba con el integrando transformado. Esta ruta se cuelga en el
/// router ANTES de ese destructor. Mismo criterio exacto que las hermanas:
/// cofactor ≡ s·u′ por multiconjunto de factores, o declina.
pub(super) fn symbolic_derivative_table_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }
    for (table_index, factor) in factors.iter().enumerate() {
        // exp(u) vive canonizado como Pow(E, u): las dos formas cuentan como
        // exterior Exp (misma dualidad que ya cubría el narrador — la ruta se
        // quedó ciega al wrapper y ∫cos·sin·e^(sin²) caía a residual).
        let outer_inner = match ctx.get(*factor) {
            Expr::Function(fn_id, args) if args.len() == 1 => {
                ctx.builtin_of(*fn_id).map(|b| (b, args[0]))
            }
            Expr::Pow(base, exp) if matches!(ctx.get(*base), Expr::Constant(Constant::E)) => {
                Some((BuiltinFn::Exp, *exp))
            }
            _ => None,
        };
        let Some((builtin, inner)) = outer_inner else {
            continue;
        };
        if !matches!(
            builtin,
            BuiltinFn::Exp | BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Sinh | BuiltinFn::Cosh
        ) {
            continue;
        }
        if !contains_named_var(ctx, inner, var) {
            continue;
        }
        if Polynomial::from_expr(ctx, inner, var).is_ok() {
            continue;
        }
        if crate::expr_complexity::node_count_tree(ctx, inner) > 64 {
            continue;
        }

        let cofactor_factors: Vec<ExprId> = factors
            .iter()
            .enumerate()
            .filter_map(|(idx, f)| (idx != table_index).then_some(*f))
            .collect();
        let cofactor = if cofactor_factors.is_empty() {
            ctx.num(1)
        } else {
            build_balanced_mul(ctx, &cofactor_factors)
        };

        let Some(derivative) =
            crate::symbolic_differentiation_support::differentiate_symbolic_expr(ctx, inner, var)
        else {
            continue;
        };
        let Some((cof_factors, cof_coef)) =
            crate::trig_power_identity_support::extract_as_product(ctx, cofactor)
        else {
            continue;
        };
        let Some((der_factors, der_coef)) =
            crate::trig_power_identity_support::extract_as_product(ctx, derivative)
        else {
            continue;
        };
        if der_coef.is_zero() || cof_factors.len() != der_factors.len() {
            continue;
        }
        let mut cof_sorted = cof_factors;
        let mut der_sorted = der_factors;
        normalize_power_factors(ctx, &mut cof_sorted);
        normalize_power_factors(ctx, &mut der_sorted);
        cof_sorted.sort_by(|a, b| compare_expr(ctx, *a, *b));
        der_sorted.sort_by(|a, b| compare_expr(ctx, *a, *b));
        if cof_sorted
            .iter()
            .zip(der_sorted.iter())
            .any(|(c, d)| compare_expr(ctx, *c, *d) != Ordering::Equal)
        {
            continue;
        }
        let scale = cof_coef / der_coef;

        let (anti_builtin, negate) = match builtin {
            BuiltinFn::Exp => (BuiltinFn::Exp, false),
            BuiltinFn::Cos => (BuiltinFn::Sin, false),
            BuiltinFn::Sin => (BuiltinFn::Cos, true),
            BuiltinFn::Sinh => (BuiltinFn::Cosh, false),
            BuiltinFn::Cosh => (BuiltinFn::Sinh, false),
            _ => unreachable!(),
        };
        let coefficient = if negate { -scale } else { scale };
        let anti = if anti_builtin == BuiltinFn::Exp {
            let e_const = ctx.add(Expr::Constant(Constant::E));
            ctx.add(Expr::Pow(e_const, inner))
        } else {
            ctx.call_builtin(anti_builtin, vec![inner])
        };
        if coefficient == BigRational::one() {
            return Some(anti);
        }
        let minus_one = BigRational::from_integer((-1).into());
        if coefficient == minus_one {
            return Some(ctx.add(Expr::Neg(anti)));
        }
        let coefficient_expr = ctx.add(Expr::Number(coefficient));
        return Some(mul2_raw(ctx, coefficient_expr, anti));
    }
    None
}

/// Potencia con exponente racional viendo a través de `Neg`: el parser
/// guarda `u^(-3)` como `Pow(u, Neg(3))` y `polynomial_power_factor` (que
/// solo casa `Number`) devolvía None — por eso `∫cos·(sin+2)^(-3)` ni entraba
/// en la ruta y caía al molino Weierstrass.
fn signed_power_factor(ctx: &Context, expr: ExprId) -> Option<(ExprId, BigRational)> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let exponent = cas_ast::views::as_rational_const(ctx, *exp, 12)?;
    Some((*base, exponent))
}

/// u-du sobre formas `Div`: `∫ s·u′/uᵐ dx` con `u` NO polinómica.
///
/// `m>1` → `s·u^{1−m}/(1−m)` (misma cola compartida, presentación recíproca);
/// `m=1` → `s·ln(|u|)`. Triple cerrojo para no robar a los dueños existentes:
/// bases función-desnuda (∫cos/sin y familia, con sus rutas propias) y bases
/// polinómicas declinan, y el cofactor debe ser `s·u′` EXACTO.
pub(super) fn symbolic_power_substitution_div_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (base, m) = match signed_power_factor(ctx, den) {
        Some((b, e)) if e > BigRational::zero() => (b, e),
        Some(_) => return None,
        None => (den, BigRational::one()),
    };
    if matches!(ctx.get(base), Expr::Function(_, _)) {
        return None;
    }
    if !contains_named_var(ctx, base, var) {
        return None;
    }
    if Polynomial::from_expr(ctx, base, var).is_ok() {
        return None;
    }
    if crate::expr_complexity::node_count_tree(ctx, base) > 64 {
        return None;
    }

    let derivative =
        crate::symbolic_differentiation_support::differentiate_symbolic_expr(ctx, base, var)?;
    let (cof_factors, cof_coef) = crate::trig_power_identity_support::extract_as_product(ctx, num)?;
    let (der_factors, der_coef) =
        crate::trig_power_identity_support::extract_as_product(ctx, derivative)?;
    if der_coef.is_zero() || cof_factors.len() != der_factors.len() {
        return None;
    }
    let mut cof_sorted = cof_factors;
    let mut der_sorted = der_factors;
    normalize_power_factors(ctx, &mut cof_sorted);
    normalize_power_factors(ctx, &mut der_sorted);
    cof_sorted.sort_by(|a, b| compare_expr(ctx, *a, *b));
    der_sorted.sort_by(|a, b| compare_expr(ctx, *a, *b));
    if cof_sorted
        .iter()
        .zip(der_sorted.iter())
        .any(|(c, d)| compare_expr(ctx, *c, *d) != Ordering::Equal)
    {
        return None;
    }
    let scale = cof_coef / der_coef;

    if m == BigRational::one() {
        let abs = ctx.call_builtin(BuiltinFn::Abs, vec![base]);
        let ln = ctx.call_builtin(BuiltinFn::Ln, vec![abs]);
        if scale == BigRational::one() {
            return Some(ln);
        }
        let scale_expr = ctx.add(Expr::Number(scale));
        return Some(mul2_raw(ctx, scale_expr, ln));
    }

    let new_exponent = BigRational::one() - m;
    let coefficient = scale / new_exponent.clone();
    build_scaled_power_antiderivative(ctx, coefficient, base, new_exponent)
}

/// u-du power fallback when the base is NOT a polynomial in `var`:
/// `∫ s·u'·uⁿ dx = s·u^{n+1}/(n+1)` for any `u` the symbolic differentiator
/// can handle, with the cofactor required to equal `s·u'` EXACTLY (rational
/// `s`, structural factor-multiset comparison — conservative, never lossy).
///
/// This is what catches `∫cos(x)·(sin(x)+1)² dx` — the affine-shifted trig
/// power that the polynomial route rejects (`sin(x)+1` no baja a polinomio) y
/// que sin esta vía caía al carril Weierstrass patológico (ledger L16).
fn symbolic_power_substitution_from_base(
    ctx: &mut Context,
    cofactor: ExprId,
    base: ExprId,
    exponent: BigRational,
    var: &str,
) -> Option<ExprId> {
    let negative_one = BigRational::from_integer((-1).into());
    if exponent == negative_one {
        return None;
    }
    if !contains_named_var(ctx, base, var) {
        return None;
    }
    // Cota de protección: la diferenciación simbólica es capaz pero no gratis.
    if crate::expr_complexity::node_count_tree(ctx, base) > 64 {
        return None;
    }

    let derivative =
        crate::symbolic_differentiation_support::differentiate_symbolic_expr(ctx, base, var)?;

    let (cof_factors, cof_coef) =
        crate::trig_power_identity_support::extract_as_product(ctx, cofactor)?;
    let (der_factors, der_coef) =
        crate::trig_power_identity_support::extract_as_product(ctx, derivative)?;
    if der_coef.is_zero() || cof_factors.len() != der_factors.len() {
        return None;
    }

    let mut cof_sorted = cof_factors;
    let mut der_sorted = der_factors;
    normalize_power_factors(ctx, &mut cof_sorted);
    normalize_power_factors(ctx, &mut der_sorted);
    cof_sorted.sort_by(|a, b| compare_expr(ctx, *a, *b));
    der_sorted.sort_by(|a, b| compare_expr(ctx, *a, *b));
    if cof_sorted
        .iter()
        .zip(der_sorted.iter())
        .any(|(c, d)| compare_expr(ctx, *c, *d) != Ordering::Equal)
    {
        return None;
    }

    let scale = cof_coef / der_coef;
    let new_exponent = exponent + BigRational::one();
    if new_exponent.is_zero() {
        return None;
    }

    let coefficient = scale / new_exponent.clone();
    build_scaled_power_antiderivative(ctx, coefficient, base, new_exponent)
}

pub(super) fn polynomial_power_substitution_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (power_index, factor) in factors.iter().enumerate() {
        let poly_pf = polynomial_power_factor(ctx, *factor);
        let from_polynomial_route = poly_pf.is_some();
        let Some((base, exponent)) = poly_pf.or_else(|| signed_power_factor(ctx, *factor)) else {
            continue;
        };

        if !contains_named_var(ctx, base, var) {
            continue;
        }

        let cofactor_factors: Vec<ExprId> = factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != power_index).then_some(*factor))
            .collect();
        let cofactor = if cofactor_factors.is_empty() {
            ctx.num(1)
        } else {
            build_balanced_mul(ctx, &cofactor_factors)
        };

        if from_polynomial_route {
            if let Some(integral) =
                polynomial_power_substitution_from_base(ctx, cofactor, base, exponent.clone(), var)
            {
                return Some(integral);
            }
        }

        if let Some(integral) =
            symbolic_power_substitution_from_base(ctx, cofactor, base, exponent, var)
        {
            return Some(integral);
        }
    }

    None
}

pub fn integrate_symbolic_is_bounded_negative_syntactic_denominator_power_substitution_target(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    max_abs_power: i64,
) -> bool {
    bounded_negative_denominator_power_substitution_target_parts(ctx, expr, var, max_abs_power)
        .is_some()
}

pub fn integrate_symbolic_is_bounded_reciprocal_quotient_denominator_power_substitution_target(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    max_abs_power: i64,
) -> bool {
    let Some((_, exponent)) =
        reciprocal_quotient_denominator_power_substitution_target_parts(ctx, expr, var)
    else {
        return false;
    };
    let bound = BigRational::from_integer(max_abs_power.into());
    exponent >= -bound.clone() && exponent <= bound
}

pub(super) fn polynomial_denominator_power_substitution_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (base, denominator_exponent, denominator_scale) =
        polynomial_denominator_power_parts(ctx, den, var)?;
    if denominator_scale.is_zero() {
        return None;
    }

    let adjusted_num = if denominator_scale.is_one() {
        num
    } else {
        let reciprocal_scale = BigRational::one() / denominator_scale;
        let reciprocal_scale = ctx.add(Expr::Number(reciprocal_scale));
        mul2_raw(ctx, reciprocal_scale, num)
    };

    polynomial_power_substitution_from_base(ctx, adjusted_num, base, -denominator_exponent, var)
}

pub fn integrate_symbolic_is_fractional_denominator_power_substitution_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    if let Expr::Mul(left, right) = ctx.get(expr).clone() {
        if rational_constant_value(ctx, left).is_some() {
            return integrate_symbolic_is_fractional_denominator_power_substitution_target(
                ctx, right, var,
            );
        }
        if rational_constant_value(ctx, right).is_some() {
            return integrate_symbolic_is_fractional_denominator_power_substitution_target(
                ctx, left, var,
            );
        }
    }

    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return false,
    };
    let Some((_, exponent, _)) = polynomial_denominator_power_parts(ctx, den, var) else {
        return false;
    };
    !exponent.is_integer()
        && polynomial_denominator_power_substitution_antiderivative(ctx, num, den, var).is_some()
}

pub(super) fn polynomial_negative_denominator_power_substitution_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (base, numerator_exponent, denominator_scale) =
        negative_syntactic_polynomial_denominator_power_parts(ctx, den, var)?;
    if denominator_scale.is_zero() {
        return None;
    }

    let adjusted_num = if denominator_scale.is_one() {
        num
    } else {
        let reciprocal_scale = BigRational::one() / denominator_scale;
        let reciprocal_scale = ctx.add(Expr::Number(reciprocal_scale));
        mul2_raw(ctx, reciprocal_scale, num)
    };

    polynomial_power_substitution_from_base(ctx, adjusted_num, base, numerator_exponent, var)
}

pub(super) fn polynomial_reciprocal_quotient_denominator_power_substitution_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (base, numerator_exponent, denominator_scale) =
        reciprocal_quotient_polynomial_denominator_power_parts(ctx, den, var)?;
    if denominator_scale.is_zero() {
        return None;
    }

    let adjusted_num = if denominator_scale.is_one() {
        num
    } else {
        let reciprocal_scale = BigRational::one() / denominator_scale;
        let reciprocal_scale = ctx.add(Expr::Number(reciprocal_scale));
        mul2_raw(ctx, reciprocal_scale, num)
    };

    polynomial_power_substitution_from_base(ctx, adjusted_num, base, numerator_exponent, var)
}

pub(super) fn constant_scaled_denominator_power_substitution_antiderivative(
    ctx: &mut Context,
    scale: ExprId,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let scale = rational_constant_value(ctx, scale)?;
    if scale.is_zero() {
        return None;
    }

    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let (num, den) = (*num, *den);
    let scale_expr = ctx.add(Expr::Number(scale));
    let scaled_num = mul2_raw(ctx, scale_expr, num);

    polynomial_denominator_power_substitution_antiderivative(ctx, scaled_num, den, var)
        .or_else(|| {
            polynomial_negative_denominator_power_substitution_antiderivative(
                ctx, scaled_num, den, var,
            )
        })
        .or_else(|| {
            polynomial_reciprocal_quotient_denominator_power_substitution_antiderivative(
                ctx, scaled_num, den, var,
            )
        })
}

pub(super) fn polynomial_denominator_power_substitution_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };
    let (base, _, _) = polynomial_denominator_power_parts(ctx, den, var)?;
    polynomial_denominator_power_substitution_antiderivative(ctx, num, den, var)?;
    Some(base)
}

pub(super) fn polynomial_fractional_denominator_power_substitution_required_positive(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };
    let (base, exponent, _) = polynomial_denominator_power_parts(ctx, den, var)?;
    if exponent.is_integer() {
        return None;
    }
    polynomial_denominator_power_substitution_antiderivative(ctx, num, den, var)?;
    Some(base)
}

pub(super) fn polynomial_negative_denominator_power_substitution_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };
    let (base, _, _) = negative_syntactic_polynomial_denominator_power_parts(ctx, den, var)?;
    polynomial_negative_denominator_power_substitution_antiderivative(ctx, num, den, var)?;
    Some(base)
}

pub(super) fn polynomial_reciprocal_quotient_denominator_power_substitution_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };
    let (base, _, _) = reciprocal_quotient_polynomial_denominator_power_parts(ctx, den, var)?;
    polynomial_reciprocal_quotient_denominator_power_substitution_antiderivative(
        ctx, num, den, var,
    )?;
    Some(base)
}

fn polynomial_log_product_substitution_from_base(
    ctx: &mut Context,
    cofactor: ExprId,
    base: ExprId,
    log_factor: ExprId,
    log_derivative_correction: ExprId,
    distribute_correction: bool,
    var: &str,
) -> Option<ExprId> {
    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let base_poly = Polynomial::from_expr(ctx, base, var).ok()?;
    if base_poly.degree() == 0 {
        return None;
    }

    let derivative = base_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let integral = if distribute_correction {
        let log_term = mul2_raw(ctx, base, log_factor);
        let correction_term =
            multiply_log_derivative_correction(ctx, base, log_derivative_correction);
        ctx.add(Expr::Sub(log_term, correction_term))
    } else {
        let log_minus_correction = ctx.add(Expr::Sub(log_factor, log_derivative_correction));
        mul2_raw(ctx, base, log_minus_correction)
    };
    if scale.is_one() {
        return Some(integral);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, integral))
}

pub(super) fn polynomial_log_product_substitution_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() >= 2 {
        for (log_index, factor) in factors.iter().enumerate() {
            let Some((log_expr, base, log_derivative_correction, distribute_correction)) =
                log_product_substitution_factor_parts(ctx, *factor)
            else {
                continue;
            };
            if !contains_named_var(ctx, base, var) {
                continue;
            }

            let cofactor_factors: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| (idx != log_index).then_some(*factor))
                .collect();
            let cofactor = if cofactor_factors.is_empty() {
                ctx.num(1)
            } else {
                build_balanced_mul(ctx, &cofactor_factors)
            };

            if let Some(integral) = polynomial_log_product_substitution_from_base(
                ctx,
                cofactor,
                base,
                log_expr,
                log_derivative_correction,
                distribute_correction,
                var,
            ) {
                return Some(integral);
            }
        }
    }

    let (log_expr, log_base, power, cofactor) = additive_common_log_power_cofactor(ctx, expr)?;
    if power != 1 {
        return None;
    }
    let log_arg = natural_log_argument(ctx, log_expr)?;
    if extract_abs_argument_view(ctx, log_arg).is_some()
        || compare_expr(ctx, log_arg, log_base) != Ordering::Equal
    {
        return None;
    }

    let one = ctx.num(1);
    polynomial_log_product_substitution_from_base(
        ctx, cofactor, log_base, log_expr, one, false, var,
    )
}

pub fn integrate_symbolic_is_log_product_substitution_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    polynomial_log_product_substitution_antiderivative(ctx, expr, var).is_some()
}

fn build_polynomial_log_power_product_substitution_integral(
    ctx: &mut Context,
    log_expr: ExprId,
    log_base: ExprId,
    correction: ExprId,
    power: u32,
    cofactor: ExprId,
    var: &str,
) -> Option<ExprId> {
    if !matches!(power, 2..=5) {
        return None;
    }

    let base_poly = Polynomial::from_expr(ctx, log_base, var).ok()?;
    if base_poly.degree() == 0 {
        return None;
    }
    if power >= 4 && !is_positive_leading_quadratic(&base_poly) {
        return None;
    }
    let derivative = base_poly.derivative();

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let base = base_poly.to_expr(ctx);
    let integral = polynomial_log_power_by_parts_integral_with_correction(
        ctx, base, log_expr, correction, power,
    );
    if scale.is_one() {
        return Some(integral);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, integral))
}

pub(super) fn polynomial_log_power_product_substitution_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.is_empty() {
        return None;
    }

    for (log_index, factor) in factors.iter().enumerate() {
        let Some((log_expr, log_base, correction, power)) =
            log_power_substitution_factor_parts(ctx, *factor)
        else {
            continue;
        };

        let cofactor_factors: Vec<ExprId> = factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != log_index).then_some(*factor))
            .collect();
        let cofactor = if cofactor_factors.is_empty() {
            ctx.num(1)
        } else {
            build_balanced_mul(ctx, &cofactor_factors)
        };

        if let Some(integral) = build_polynomial_log_power_product_substitution_integral(
            ctx, log_expr, log_base, correction, power, cofactor, var,
        ) {
            return Some(integral);
        }
    }

    let (log_expr, log_base, correction, power, cofactor) =
        additive_common_log_power_cofactor_with_correction(ctx, expr)?;
    build_polynomial_log_power_product_substitution_integral(
        ctx, log_expr, log_base, correction, power, cofactor, var,
    )
}

fn polynomial_log_product_substitution_power(ctx: &mut Context, expr: ExprId) -> Option<u32> {
    let factors = mul_leaves(ctx, expr);
    for factor in factors {
        if let Some((_, _, _, power)) = log_power_substitution_factor_parts(ctx, factor) {
            return Some(power);
        }
    }

    additive_common_log_power_cofactor_with_correction(ctx, expr).map(|(_, _, _, power, _)| power)
}

pub fn integrate_symbolic_is_log_cube_product_substitution_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    if polynomial_log_product_substitution_power(ctx, expr) != Some(3) {
        return false;
    }

    polynomial_log_power_product_substitution_antiderivative(ctx, expr, var).is_some()
}

pub fn integrate_symbolic_is_log_power_product_substitution_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    polynomial_log_power_product_substitution_antiderivative(ctx, expr, var).is_some()
}

pub fn integrate_symbolic_is_high_log_power_product_substitution_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    matches!(
        polynomial_log_product_substitution_power(ctx, expr),
        Some(4 | 5)
    ) && polynomial_log_power_product_substitution_antiderivative(ctx, expr, var).is_some()
}

pub fn integrate_symbolic_is_verifiable_log_power_product_substitution_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    matches!(
        polynomial_log_product_substitution_power(ctx, expr),
        Some(2..=5)
    ) && polynomial_log_power_product_substitution_antiderivative(ctx, expr, var).is_some()
}

pub(super) fn polynomial_substitution_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    // Normalization artifact: c*u' * e^(-u) arrives as Div(c*u', e^u);
    // rebuild as a product with the reciprocal exponential and reuse the
    // same route. Nonlinear arguments only: linear exponential quotients
    // belong to the reciprocal/by-parts/cyclic family and its narrators.
    if let Expr::Div(numerator, denominator) = ctx.get(expr).clone() {
        let (kernel, kernel_arg) = polynomial_substitution_kernel(ctx, denominator)?;
        if !matches!(kernel, PolynomialSubstitutionKernel::Exp) {
            return None;
        }
        let arg_poly = Polynomial::from_expr(ctx, kernel_arg, var).ok()?;
        if arg_poly.degree() < 2 {
            return None;
        }
        let negated = ctx.add(Expr::Neg(kernel_arg));
        let e = ctx.add(Expr::Constant(Constant::E));
        let reciprocal = ctx.add(Expr::Pow(e, negated));
        let product = mul2_raw(ctx, numerator, reciprocal);
        return polynomial_substitution_antiderivative(ctx, product, var);
    }

    let factors = mul_leaves(ctx, expr);
    let (kernel_index, kernel, kernel_arg) =
        factors.iter().enumerate().find_map(|(idx, factor)| {
            polynomial_substitution_kernel(ctx, *factor).map(|(kernel, arg)| (idx, kernel, arg))
        })?;

    if !contains_named_var(ctx, kernel_arg, var) {
        return None;
    }

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != kernel_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        if matches!(
            kernel,
            PolynomialSubstitutionKernel::Tan | PolynomialSubstitutionKernel::Cot
        ) {
            return None;
        }
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, kernel_arg, var).ok()?;
    if matches!(
        kernel,
        PolynomialSubstitutionKernel::Sec | PolynomialSubstitutionKernel::Csc
    ) && arg_poly.degree() <= 1
    {
        return None;
    }
    let derivative_poly = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative_poly)?;
    if scale.is_zero() {
        return None;
    }

    let antiderivative = polynomial_substitution_kernel_antiderivative(
        ctx,
        kernel,
        kernel_arg,
        factors[kernel_index],
    );
    Some(scale_rational_term(ctx, scale, antiderivative))
}

pub fn integrate_symbolic_is_polynomial_derivative_substitution_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    polynomial_substitution_antiderivative(ctx, expr, var).is_some()
}

fn has_trig_polynomial_substitution_kernel(ctx: &Context, expr: ExprId, var: &str) -> bool {
    mul_leaves(ctx, expr).into_iter().any(|factor| {
        let Some((kernel, arg)) = polynomial_substitution_kernel(ctx, factor) else {
            return false;
        };
        matches!(
            kernel,
            PolynomialSubstitutionKernel::Sin
                | PolynomialSubstitutionKernel::Cos
                | PolynomialSubstitutionKernel::Tan
                | PolynomialSubstitutionKernel::Cot
                | PolynomialSubstitutionKernel::Sec
                | PolynomialSubstitutionKernel::Csc
        ) && contains_named_var(ctx, arg, var)
    })
}

pub fn integrate_symbolic_is_trig_polynomial_substitution_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    if additive_common_trig_polynomial_substitution_antiderivative(ctx, expr, var).is_some() {
        return true;
    }

    if has_trig_polynomial_substitution_kernel(ctx, expr, var)
        && polynomial_substitution_antiderivative(ctx, expr, var).is_some()
    {
        return true;
    }

    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return false,
    };
    trig_log_derivative_ratio_scale(ctx, num, den, var).is_some()
}

pub(super) fn additive_common_trig_polynomial_substitution_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let add_view = AddView::from_expr(ctx, expr);
    if add_view.terms.len() < 2 {
        return None;
    }

    let mut common: Option<(PolynomialSubstitutionKernel, ExprId, ExprId)> = None;
    let mut cofactor_terms = Vec::with_capacity(add_view.terms.len());

    for (term, sign) in add_view.terms {
        let factors = mul_leaves(ctx, term);
        let mut term_cofactor = None;

        for (kernel_index, factor) in factors.iter().enumerate() {
            let Some((kernel, arg)) = polynomial_substitution_kernel(ctx, *factor) else {
                continue;
            };
            if !matches!(
                kernel,
                PolynomialSubstitutionKernel::Sin | PolynomialSubstitutionKernel::Cos
            ) {
                continue;
            }

            if let Some((common_kernel, common_arg, _)) = common {
                if kernel != common_kernel || compare_expr(ctx, arg, common_arg) != Ordering::Equal
                {
                    continue;
                }
            } else {
                common = Some((kernel, arg, *factor));
            }

            let cofactor_factors: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| (idx != kernel_index).then_some(*factor))
                .collect();
            let cofactor = if cofactor_factors.is_empty() {
                ctx.num(1)
            } else {
                build_balanced_mul(ctx, &cofactor_factors)
            };
            term_cofactor = Some(signed_term(ctx, cofactor, sign));
            break;
        }

        cofactor_terms.push(term_cofactor?);
    }

    let (kernel, arg, kernel_factor) = common?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let cofactor = build_balanced_add(ctx, &cofactor_terms);
    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative_poly = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative_poly)?;
    if scale.is_zero() {
        return None;
    }

    let antiderivative =
        polynomial_substitution_kernel_antiderivative(ctx, kernel, arg, kernel_factor);
    Some(scale_rational_term(ctx, scale, antiderivative))
}

pub(super) fn arctan_polynomial_substitution_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let Some((arg_poly, offset_square)) =
        exact_polynomial_square_plus_positive_constant(&denominator)
    else {
        return arctan_scaled_quadratic_antiderivative(ctx, &numerator, &denominator);
    };
    let Some(offset) = exact_rational_sqrt(&offset_square) else {
        return arctan_surd_offset_antiderivative(ctx, &numerator, &arg_poly, &offset_square);
    };
    if offset.is_zero() {
        return None;
    }

    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let arg = arg_poly.to_expr(ctx);
    let arctan_arg = if offset.is_one() {
        arg
    } else {
        let offset_expr = ctx.add(Expr::Number(offset.clone()));
        ctx.add(Expr::Div(arg, offset_expr))
    };
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arctan_arg]);
    let scaled = scale / offset;
    if scaled.is_one() {
        return Some(arctan);
    }

    let scale_expr = ctx.add(Expr::Number(scaled));
    Some(mul2_raw(ctx, scale_expr, arctan))
}

pub(super) fn atanh_polynomial_substitution_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let (arg_poly, offset_square) = exact_positive_constant_minus_polynomial_square(&denominator)?;
    let Some(offset) = exact_rational_sqrt(&offset_square) else {
        return atanh_surd_offset_antiderivative(ctx, &numerator, &arg_poly, &offset_square);
    };
    if offset.is_zero() {
        return None;
    }

    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let arg = arg_poly.to_expr(ctx);
    let atanh_arg = if offset.is_one() {
        arg
    } else {
        let offset_expr = ctx.add(Expr::Number(offset.clone()));
        ctx.add(Expr::Div(arg, offset_expr))
    };
    let atanh = ctx.call_builtin(BuiltinFn::Atanh, vec![atanh_arg]);
    let scaled = scale / offset;
    if scaled.is_one() {
        return Some(atanh);
    }
    if scaled == -BigRational::one() {
        return Some(ctx.add(Expr::Neg(atanh)));
    }

    let scale_expr = ctx.add(Expr::Number(scaled));
    Some(mul2_raw(ctx, scale_expr, atanh))
}

fn arcsin_polynomial_substitution_from_radicand(
    ctx: &mut Context,
    numerator: ExprId,
    radicand: ExprId,
    var: &str,
) -> Option<ExprId> {
    if let Some(integral) =
        arcsin_symbolic_radius_substitution_from_radicand(ctx, numerator, radicand, var)
    {
        return Some(integral);
    }

    let numerator = Polynomial::from_expr(ctx, numerator, var).ok()?;
    let radicand = Polynomial::from_expr(ctx, radicand, var).ok()?;
    if let Some((arg_poly, offset_square)) =
        exact_positive_constant_minus_polynomial_square(&radicand)
    {
        return arcsin_polynomial_substitution_from_parts(
            ctx,
            &numerator,
            arg_poly,
            offset_square,
            None,
        );
    }

    let (arg_poly, offset_square, radicand_scale) =
        exact_positive_constant_minus_scaled_polynomial_square(&radicand)?;
    arcsin_polynomial_substitution_from_parts(
        ctx,
        &numerator,
        arg_poly,
        offset_square,
        Some(radicand_scale),
    )
}

pub(super) fn arcsin_symbolic_radius_substitution_from_radicand(
    ctx: &mut Context,
    numerator: ExprId,
    radicand: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (radius_square, arg) = symbolic_radius_minus_square_arg(ctx, radicand, var)
        .or_else(|| symbolic_radius_minus_expanded_square_arg(ctx, radicand, var))?;
    symbolic_radius_inverse_sqrt_primitive(
        ctx,
        numerator,
        radius_square,
        arg,
        var,
        BuiltinFn::Arcsin,
    )
}

fn asinh_symbolic_radius_substitution_from_radicand(
    ctx: &mut Context,
    numerator: ExprId,
    radicand: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (radius_square, arg) = symbolic_radius_plus_square_arg(ctx, radicand, var)
        .or_else(|| symbolic_radius_plus_expanded_square_arg(ctx, radicand, var))?;
    symbolic_radius_inverse_sqrt_primitive(
        ctx,
        numerator,
        radius_square,
        arg,
        var,
        BuiltinFn::Asinh,
    )
}

fn asinh_polynomial_substitution_from_radicand(
    ctx: &mut Context,
    numerator: ExprId,
    radicand: ExprId,
    var: &str,
) -> Option<ExprId> {
    if let Some(integral) =
        asinh_symbolic_radius_substitution_from_radicand(ctx, numerator, radicand, var)
    {
        return Some(integral);
    }

    let numerator = Polynomial::from_expr(ctx, numerator, var).ok()?;
    let radicand = Polynomial::from_expr(ctx, radicand, var).ok()?;
    let (arg_poly, offset_square) = exact_polynomial_square_plus_positive_constant(&radicand)?;

    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let (arg_poly, offset_square) = normalize_surd_ratio_arg(arg_poly, offset_square);
    let offset_expr = positive_rational_sqrt_expr(ctx, &offset_square)?;

    let raw_arg = arg_poly.to_expr(ctx);
    let arg = compact_single_power_polynomial_arg(ctx, raw_arg);
    let asinh_arg = if offset_square.is_one() {
        arg
    } else {
        ctx.add(Expr::Div(arg, offset_expr))
    };
    let asinh = ctx.call_builtin(BuiltinFn::Asinh, vec![asinh_arg]);
    if scale.is_one() {
        return Some(asinh);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, asinh))
}

fn acosh_polynomial_substitution_from_radicand_with_domain_sample(
    ctx: &mut Context,
    numerator: ExprId,
    radicand: ExprId,
    var: &str,
    domain_sample: Option<&BigRational>,
) -> Option<ExprId> {
    let (radicand, radicand_scale) =
        split_positive_rational_content_from_sqrt_radicand(ctx, radicand, var)?;
    let numerator = Polynomial::from_expr(ctx, numerator, var).ok()?;
    let (arg_poly, offset_square, scale) =
        acosh_polynomial_substitution_oriented_arg(ctx, &numerator, radicand, var, domain_sample)?;
    let offset_expr = positive_rational_sqrt_expr(ctx, &offset_square)?;

    let raw_arg = arg_poly.to_expr(ctx);
    let arg = compact_single_power_polynomial_arg(ctx, raw_arg);
    let acosh_arg = if offset_square.is_one() {
        arg
    } else {
        ctx.add(Expr::Div(arg, offset_expr))
    };
    let acosh = ctx.call_builtin(BuiltinFn::Acosh, vec![acosh_arg]);
    let scaled = if scale.is_one() {
        acosh
    } else {
        let scale_expr = ctx.add(Expr::Number(scale));
        mul2_raw(ctx, scale_expr, acosh)
    };

    Some(if let Some(radicand_scale) = radicand_scale {
        divide_by_sqrt_product_denominator_scale(ctx, scaled, radicand_scale)
    } else {
        scaled
    })
}

fn acosh_polynomial_substitution_oriented_arg(
    ctx: &Context,
    numerator: &Polynomial,
    radicand: ExprId,
    var: &str,
    domain_sample: Option<&BigRational>,
) -> Option<(Polynomial, BigRational, BigRational)> {
    let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
    let (mut arg_poly, offset_square) =
        exact_positive_constant_minus_polynomial_square(&radicand_poly.neg())?;

    let derivative = arg_poly.derivative();
    let mut scale = constant_polynomial_ratio(numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }
    if arg_poly.degree() == 1 && scale.is_negative() {
        let factored_domain_sample = domain_sample
            .cloned()
            .or_else(|| positive_linear_factor_domain_sample(ctx, radicand, var));
        let should_flip = factored_domain_sample
            .as_ref()
            .map(|sample| arg_poly.eval(sample).is_negative())
            .unwrap_or(true);
        if should_flip {
            arg_poly = arg_poly.neg();
            scale = -scale;
        }
    }

    Some((arg_poly, offset_square, scale))
}

fn sqrt_derivative_substitution_from_radicand(
    ctx: &mut Context,
    numerator: ExprId,
    radicand: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, numerator, var).ok()?;
    let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
    let derivative = radicand_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let two = BigRational::from_integer(2.into());
    let scaled = scale * two;
    if scaled.is_one() {
        return Some(sqrt_radicand);
    }

    let scale_expr = ctx.add(Expr::Number(scaled));
    Some(mul2_raw(ctx, scale_expr, sqrt_radicand))
}

fn sqrt_product_derivative_substitution_from_radicand(
    ctx: &mut Context,
    numerator: ExprId,
    radicand: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, numerator, var).ok()?;
    let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
    let derivative = radicand_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let half_exp = ctx.rational(3, 2);
    let power = ctx.add(Expr::Pow(radicand, half_exp));
    let two = BigRational::from_integer(2.into());
    let three_r = BigRational::from_integer(3.into());
    let scaled = scale * two / three_r;
    if scaled.is_one() {
        return Some(power);
    }

    let scale_expr = ctx.add(Expr::Number(scaled));
    Some(mul2_raw(ctx, scale_expr, power))
}

pub(super) fn sqrt_derivative_substitution_div_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    if let Some(radicand) = sqrt_like_radicand(ctx, den) {
        return sqrt_derivative_substitution_from_radicand(ctx, num, radicand, var);
    }

    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let factors = mul_leaves(ctx, num);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        let radicand = sqrt_like_radicand(ctx, *factor)?;
        let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
        (radicand_poly == denominator).then_some((idx, radicand))
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    sqrt_derivative_substitution_from_radicand(ctx, cofactor, radicand, var)
}

pub(super) fn sqrt_derivative_substitution_product_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        reciprocal_sqrt_like_radicand(ctx, *factor).map(|radicand| (idx, radicand))
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    sqrt_derivative_substitution_from_radicand(ctx, cofactor, radicand, var)
}

pub(super) fn sqrt_product_derivative_substitution_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        sqrt_like_radicand(ctx, *factor).map(|radicand| (idx, radicand))
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    sqrt_product_derivative_substitution_from_radicand(ctx, cofactor, radicand, var)
}

pub(super) fn arcsin_polynomial_substitution_div_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    if let Some(radicand) = sqrt_like_radicand(ctx, den) {
        if let Some((scale, cofactor)) = split_variable_free_scale_from_product(ctx, num, var) {
            let integral =
                arcsin_polynomial_substitution_from_radicand(ctx, cofactor, radicand, var)?;
            return Some(multiply_constant_integral_result(ctx, scale, integral));
        }
        return arcsin_polynomial_substitution_from_radicand(ctx, num, radicand, var);
    }

    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let factors = mul_leaves(ctx, num);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        let radicand = sqrt_like_radicand(ctx, *factor)?;
        let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
        (radicand_poly == denominator).then_some((idx, radicand))
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    arcsin_polynomial_substitution_from_radicand(ctx, cofactor, radicand, var)
}

pub(super) fn asinh_polynomial_substitution_div_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    if let Some(radicand) = sqrt_like_radicand(ctx, den) {
        if let Some((scale, cofactor)) = split_variable_free_scale_from_product(ctx, num, var) {
            let integral =
                asinh_polynomial_substitution_from_radicand(ctx, cofactor, radicand, var)?;
            return Some(multiply_constant_integral_result(ctx, scale, integral));
        }
        return asinh_polynomial_substitution_from_radicand(ctx, num, radicand, var);
    }

    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let factors = mul_leaves(ctx, num);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        let radicand = sqrt_like_radicand(ctx, *factor)?;
        let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
        (radicand_poly == denominator).then_some((idx, radicand))
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    asinh_polynomial_substitution_from_radicand(ctx, cofactor, radicand, var)
}

pub(super) fn acosh_polynomial_substitution_div_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let domain_sample = explicit_sqrt_linear_domain_sample(ctx, den, var);
    if let Some(radicand) = sqrt_like_radicand(ctx, den) {
        return acosh_polynomial_substitution_from_radicand_with_domain_sample(
            ctx,
            num,
            radicand,
            var,
            domain_sample.as_ref(),
        );
    }
    if let Some((radicand, denominator_scale)) = sqrt_product_denominator_radicand(ctx, den, var) {
        let antiderivative = acosh_polynomial_substitution_from_radicand_with_domain_sample(
            ctx,
            num,
            radicand,
            var,
            domain_sample.as_ref(),
        )?;
        return Some(divide_by_sqrt_product_denominator_scale(
            ctx,
            antiderivative,
            denominator_scale,
        ));
    }

    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let factors = mul_leaves(ctx, num);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        let radicand = sqrt_like_radicand(ctx, *factor)?;
        let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
        (radicand_poly == denominator).then_some((idx, radicand))
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    acosh_polynomial_substitution_from_radicand_with_domain_sample(
        ctx,
        cofactor,
        radicand,
        var,
        domain_sample.as_ref(),
    )
}

pub(super) fn arcsin_polynomial_substitution_product_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        reciprocal_sqrt_like_radicand(ctx, *factor).map(|radicand| (idx, radicand))
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    arcsin_polynomial_substitution_from_radicand(ctx, cofactor, radicand, var)
}

pub(super) fn asinh_polynomial_substitution_product_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        reciprocal_sqrt_like_radicand(ctx, *factor).map(|radicand| (idx, radicand))
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    asinh_polynomial_substitution_from_radicand(ctx, cofactor, radicand, var)
}

pub(super) fn acosh_polynomial_substitution_product_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let domain_sample = explicit_sqrt_linear_domain_sample(ctx, expr, var);
    let factors = mul_leaves(ctx, expr);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        reciprocal_sqrt_like_radicand(ctx, *factor).map(|radicand| (idx, radicand))
    })?;

    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
        .collect();
    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };

    acosh_polynomial_substitution_from_radicand_with_domain_sample(
        ctx,
        cofactor,
        radicand,
        var,
        domain_sample.as_ref(),
    )
}

pub(super) fn sqrt_derivative_substitution_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            if let Some(radicand) = sqrt_like_radicand(ctx, den) {
                sqrt_derivative_substitution_from_radicand(ctx, num, radicand, var)?;
                return Some(radicand);
            }

            let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
            let factors = mul_leaves(ctx, num);
            let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
                let radicand = sqrt_like_radicand(ctx, *factor)?;
                let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
                (radicand_poly == denominator).then_some((idx, radicand))
            })?;

            let cofactor_factors: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
                .collect();
            let cofactor = if cofactor_factors.is_empty() {
                ctx.num(1)
            } else {
                build_balanced_mul(ctx, &cofactor_factors)
            };
            sqrt_derivative_substitution_from_radicand(ctx, cofactor, radicand, var)?;
            Some(radicand)
        }
        Expr::Mul(_, _) => {
            let factors = mul_leaves(ctx, expr);
            let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
                reciprocal_sqrt_like_radicand(ctx, *factor).map(|radicand| (idx, radicand))
            })?;

            let cofactor_factors: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
                .collect();
            let cofactor = if cofactor_factors.is_empty() {
                ctx.num(1)
            } else {
                build_balanced_mul(ctx, &cofactor_factors)
            };
            sqrt_derivative_substitution_from_radicand(ctx, cofactor, radicand, var)?;
            Some(radicand)
        }
        _ => None,
    }
}

pub fn integrate_symbolic_is_sqrt_derivative_substitution_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    sqrt_derivative_substitution_radicand(ctx, expr, var).is_some()
}

pub(super) fn arcsin_polynomial_substitution_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            if let Some(radicand) = sqrt_like_radicand(ctx, den) {
                arcsin_polynomial_substitution_from_radicand(ctx, num, radicand, var)?;
                return Some(radicand);
            }

            let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
            let factors = mul_leaves(ctx, num);
            let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
                let radicand = sqrt_like_radicand(ctx, *factor)?;
                let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
                (radicand_poly == denominator).then_some((idx, radicand))
            })?;

            let cofactor_factors: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
                .collect();
            let cofactor = if cofactor_factors.is_empty() {
                ctx.num(1)
            } else {
                build_balanced_mul(ctx, &cofactor_factors)
            };
            arcsin_polynomial_substitution_from_radicand(ctx, cofactor, radicand, var)?;
            Some(radicand)
        }
        Expr::Mul(_, _) => {
            let factors = mul_leaves(ctx, expr);
            let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
                reciprocal_sqrt_like_radicand(ctx, *factor).map(|radicand| (idx, radicand))
            })?;

            let cofactor_factors: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
                .collect();
            let cofactor = if cofactor_factors.is_empty() {
                ctx.num(1)
            } else {
                build_balanced_mul(ctx, &cofactor_factors)
            };
            arcsin_polynomial_substitution_from_radicand(ctx, cofactor, radicand, var)?;
            Some(radicand)
        }
        _ => None,
    }
}

pub(super) fn arcsin_inverse_sqrt_product_substitution_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);

    if let Some(radicand) = reciprocal_sqrt_like_radicand(ctx, expr) {
        if mul_leaves(ctx, radicand).len() >= 2 {
            let one = ctx.num(1);
            arcsin_polynomial_substitution_from_radicand(ctx, one, radicand, var)?;
            return Some(radicand);
        }
    }

    let mut numerator_factors = Vec::new();
    let mut denominator_factors = Vec::new();
    collect_fraction_factors_for_inverse_sqrt_product(
        ctx,
        expr,
        false,
        &mut numerator_factors,
        &mut denominator_factors,
    );

    let mut radicands = Vec::new();
    let mut remaining_denominator_factors = Vec::new();
    for factor in denominator_factors {
        if let Some(radicand) = sqrt_like_radicand(ctx, factor) {
            radicands.push(radicand);
        } else {
            remaining_denominator_factors.push(factor);
        }
    }
    if radicands.len() < 2 {
        return None;
    }

    let numerator = if numerator_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &numerator_factors)
    };
    let cofactor = if remaining_denominator_factors.is_empty() {
        numerator
    } else {
        let denominator = build_balanced_mul(ctx, &remaining_denominator_factors);
        ctx.add(Expr::Div(numerator, denominator))
    };

    let combined_radicand = build_balanced_mul(ctx, &radicands);
    let validation_cofactor =
        strip_variable_free_factors_from_arcsin_product_cofactor(ctx, cofactor, var);
    arcsin_polynomial_substitution_from_radicand(ctx, validation_cofactor, combined_radicand, var)?;
    Some(combined_radicand)
}

pub fn integrate_symbolic_is_acosh_polynomial_substitution_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    if let Some(inner) = constant_scaled_integrand_inner(ctx, expr, var) {
        return integrate_symbolic_is_acosh_polynomial_substitution_target(ctx, inner, var);
    }

    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            acosh_polynomial_substitution_div_antiderivative(ctx, num, den, var).is_some()
        }
        Expr::Mul(_, _) => {
            acosh_polynomial_substitution_product_antiderivative(ctx, expr, var).is_some()
        }
        _ => false,
    }
}

pub fn integrate_symbolic_is_asinh_polynomial_substitution_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            asinh_polynomial_substitution_div_antiderivative(ctx, num, den, var).is_some()
        }
        _ => asinh_polynomial_substitution_product_antiderivative(ctx, expr, var).is_some(),
    }
}

pub fn integrate_symbolic_is_arcsin_polynomial_substitution_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            arcsin_polynomial_substitution_div_antiderivative(ctx, num, den, var).is_some()
        }
        _ => arcsin_polynomial_substitution_product_antiderivative(ctx, expr, var).is_some(),
    }
}

pub(super) fn acosh_polynomial_substitution_positive_conditions(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Vec<ExprId> {
    let Some((radicand, arg_poly, offset_square)) =
        acosh_polynomial_substitution_oriented_radicand_arg(ctx, expr, var)
    else {
        return vec![];
    };

    let Some(offset_expr) = positive_rational_sqrt_expr(ctx, &offset_square) else {
        return vec![radicand];
    };

    let arg = arg_poly.to_expr(ctx);
    let lower_domain = ctx.add(Expr::Sub(arg, offset_expr));
    vec![radicand, lower_domain]
}

fn acosh_polynomial_substitution_oriented_radicand_arg(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, Polynomial, BigRational)> {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            if let Some(radicand) = sqrt_like_radicand(ctx, den) {
                let (radicand, _) =
                    split_positive_rational_content_from_sqrt_radicand(ctx, radicand, var)?;
                let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
                let domain_sample = explicit_sqrt_linear_domain_sample(ctx, den, var);
                let (arg, offset, _) = acosh_polynomial_substitution_oriented_arg(
                    ctx,
                    &numerator,
                    radicand,
                    var,
                    domain_sample.as_ref(),
                )?;
                return Some((radicand, arg, offset));
            }
            if let Some((radicand, _denominator_scale)) =
                sqrt_product_denominator_radicand(ctx, den, var)
            {
                let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
                let domain_sample = explicit_sqrt_linear_domain_sample(ctx, den, var);
                let (arg, offset, _) = acosh_polynomial_substitution_oriented_arg(
                    ctx,
                    &numerator,
                    radicand,
                    var,
                    domain_sample.as_ref(),
                )?;
                return Some((radicand, arg, offset));
            }

            let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
            let factors = mul_leaves(ctx, num);
            let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
                let radicand = sqrt_like_radicand(ctx, *factor)?;
                let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
                (radicand_poly == denominator).then_some((idx, radicand))
            })?;

            let cofactor_factors: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
                .collect();
            let cofactor = if cofactor_factors.is_empty() {
                ctx.num(1)
            } else {
                build_balanced_mul(ctx, &cofactor_factors)
            };
            let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
            let domain_sample = explicit_sqrt_linear_domain_sample(ctx, den, var);
            let (arg, offset, _) = acosh_polynomial_substitution_oriented_arg(
                ctx,
                &cofactor_poly,
                radicand,
                var,
                domain_sample.as_ref(),
            )?;
            Some((radicand, arg, offset))
        }
        Expr::Mul(_, _) => {
            let domain_sample = explicit_sqrt_linear_domain_sample(ctx, expr, var);
            let factors = mul_leaves(ctx, expr);
            let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
                reciprocal_sqrt_like_radicand(ctx, *factor).map(|radicand| (idx, radicand))
            })?;
            let (radicand, _) =
                split_positive_rational_content_from_sqrt_radicand(ctx, radicand, var)?;

            let cofactor_factors: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter_map(|(idx, factor)| (idx != sqrt_index).then_some(*factor))
                .collect();
            let cofactor = if cofactor_factors.is_empty() {
                ctx.num(1)
            } else {
                build_balanced_mul(ctx, &cofactor_factors)
            };
            let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
            let (arg, offset, _) = acosh_polynomial_substitution_oriented_arg(
                ctx,
                &cofactor_poly,
                radicand,
                var,
                domain_sample.as_ref(),
            )?;
            Some((radicand, arg, offset))
        }
        _ => {
            let radicand = reciprocal_sqrt_like_radicand(ctx, expr)?;
            let (radicand, _) =
                split_positive_rational_content_from_sqrt_radicand(ctx, radicand, var)?;
            let one = ctx.num(1);
            let numerator = Polynomial::from_expr(ctx, one, var).ok()?;
            let (arg, offset, _) =
                acosh_polynomial_substitution_oriented_arg(ctx, &numerator, radicand, var, None)?;
            Some((radicand, arg, offset))
        }
    }
}

pub(super) fn atanh_polynomial_substitution_denominator(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    atanh_polynomial_substitution_antiderivative(ctx, num, den, var)?;
    Some(den)
}

fn atanh_polynomial_substitution_arg_degree(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<usize> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };

    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let (arg_poly, offset_square) = exact_positive_constant_minus_polynomial_square(&denominator)?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() || offset_square.is_zero() {
        return None;
    }

    Some(arg_poly.degree())
}

pub fn integrate_symbolic_is_atanh_polynomial_substitution_target(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> bool {
    if let Some(inner) = constant_scaled_integrand_inner(ctx, expr, var) {
        return integrate_symbolic_is_atanh_polynomial_substitution_target(ctx, inner, var);
    }

    atanh_polynomial_substitution_target_parts(ctx, expr, var).is_some()
}

fn arctan_polynomial_substitution_arg_degree(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<usize> {
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let denominator = Polynomial::from_expr(ctx, den, var).ok()?;
    let (arg_poly, offset_square) = exact_polynomial_square_plus_positive_constant(&denominator)?;
    if offset_square.is_zero() {
        return None;
    }

    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    Some(arg_poly.degree())
}

fn inverse_sqrt_nested_polynomial_substitution_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            let radicand = sqrt_like_radicand(ctx, den)?;
            if arcsin_polynomial_substitution_div_antiderivative(ctx, num, den, var).is_some()
                || asinh_polynomial_substitution_div_antiderivative(ctx, num, den, var).is_some()
                || acosh_polynomial_substitution_div_antiderivative(ctx, num, den, var).is_some()
            {
                return Some(radicand);
            }
            None
        }
        Expr::Mul(_, _) => {
            let factors = mul_leaves(ctx, expr);
            let radicand = factors
                .iter()
                .find_map(|factor| reciprocal_sqrt_like_radicand(ctx, *factor))?;
            if arcsin_polynomial_substitution_product_antiderivative(ctx, expr, var).is_some()
                || asinh_polynomial_substitution_product_antiderivative(ctx, expr, var).is_some()
                || acosh_polynomial_substitution_product_antiderivative(ctx, expr, var).is_some()
            {
                return Some(radicand);
            }
            None
        }
        _ => None,
    }
}

pub fn integrate_symbolic_is_nested_inverse_polynomial_substitution_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    if let Some(inner) = constant_scaled_integrand_inner(ctx, expr, var) {
        return integrate_symbolic_is_nested_inverse_polynomial_substitution_target(
            ctx, inner, var,
        );
    }

    if let Some(radicand) = inverse_sqrt_nested_polynomial_substitution_radicand(ctx, expr, var) {
        return Polynomial::from_expr(ctx, radicand, var).is_ok_and(|poly| poly.degree() > 2);
    }

    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return false;
    };

    arctan_polynomial_substitution_arg_degree(ctx, num, den, var).is_some_and(|degree| degree > 1)
        || atanh_polynomial_substitution_arg_degree(ctx, expr, var).is_some_and(|degree| degree > 1)
}

pub fn integrate_symbolic_is_polynomial_base_substitution_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    if let Some(inner) = constant_scaled_integrand_inner(ctx, expr, var) {
        return integrate_symbolic_is_polynomial_base_substitution_target(ctx, inner, var);
    }

    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            polynomial_log_derivative_antiderivative(ctx, num, den, var).is_some()
                || polynomial_log_derivative_power_antiderivative(ctx, num, den, var).is_some()
                || polynomial_log_reciprocal_derivative_antiderivative(ctx, num, den, var).is_some()
                || polynomial_denominator_power_substitution_antiderivative(ctx, num, den, var)
                    .is_some()
                || polynomial_negative_denominator_power_substitution_antiderivative(
                    ctx, num, den, var,
                )
                .is_some()
                || polynomial_reciprocal_quotient_denominator_power_substitution_antiderivative(
                    ctx, num, den, var,
                )
                .is_some()
                || sqrt_derivative_substitution_div_antiderivative(ctx, num, den, var).is_some()
        }
        _ => {
            polynomial_power_substitution_antiderivative(ctx, expr, var).is_some()
                || sqrt_derivative_substitution_product_antiderivative(ctx, expr, var).is_some()
                || sqrt_product_derivative_substitution_antiderivative(ctx, expr, var).is_some()
        }
    }
}

/// Integrate a transcendental chain `c·g'(x)·f(g(x))` by GUESS-AND-VERIFY u-substitution:
/// enumerate the `f(g)` subexpressions whose outer `f` has a closed elementary antiderivative
/// `F` (exp/sin/cos/sinh/cosh), build the candidate `F(g(x))`, and ACCEPT it only when
/// `d/dx F(g) == integrand` exactly (up to a global sign). Sound by construction — the
/// differentiation IS the verifier, so a wrong guess is rejected. Covers `cos(x)·e^(sin x)`,
/// `sin(x)·e^(cos x)`, `cos(ln x)/x`, `e^x·cos(e^x)`, etc.
pub(super) fn transcendental_chain_substitution_antiderivative(
    ctx: &mut Context,
    integrand: ExprId,
    var: &str,
) -> Option<ExprId> {
    if !contains_named_var(ctx, integrand, var) {
        return None;
    }
    let mut candidates = Vec::new();
    let mut seen = std::collections::HashSet::new();
    collect_chain_substitution_candidates(ctx, integrand, var, &mut candidates, &mut seen);

    // Normalized integrand (rational coefficient stripped) for the exact transcendental compare.
    let reduced_integrand = reduce_ln_e_and_unit_products(ctx, integrand);
    let (integrand_coeff, integrand_core) = strip_rational_coefficient(ctx, reduced_integrand);
    if integrand_coeff.is_zero() {
        return None;
    }

    for (builtin, inner) in candidates {
        let Some(antiderivative) = unary_chain_antiderivative(ctx, builtin, inner) else {
            continue;
        };
        // SOUNDNESS GATE: accept `F(g)` only when `d/dx F(g) == k·integrand` for a NONZERO rational
        // constant `k` — then `∫ integrand = F(g)/k`. The differentiation is trusted; the comparison
        // folds the `ln(E)`(=1) artefact the general power rule leaves on `Pow(E, g)`, strips a
        // rational coefficient from BOTH the derivative and the integrand (covering the sign AND any
        // scale `≠ ±1` a linear inner like `cos(2x)` introduces via the chain rule), and requires the
        // coefficient-free cores to be exactly equal. The scale `integrand_coeff / derivative_coeff`
        // is exact, so the returned `scale · F(g)` differentiates back to the integrand by construction.
        let Some(raw_derivative) =
            crate::symbolic_differentiation_support::differentiate_symbolic_expr(
                ctx,
                antiderivative,
                var,
            )
        else {
            continue;
        };
        let derivative = reduce_ln_e_and_unit_products(ctx, raw_derivative);
        let (derivative_coeff, derivative_core) = strip_rational_coefficient(ctx, derivative);
        if derivative_coeff.is_zero() {
            continue;
        }
        if crate::semantic_equality::SemanticEqualityChecker::new(ctx)
            .are_equal(derivative_core, integrand_core)
        {
            let scale = integrand_coeff.clone() / derivative_coeff;
            return Some(scale_expr_by_rational(ctx, antiderivative, scale));
        }
    }
    None
}

/// Collect `(f, g)` candidates: subexpressions `f(g)` where `f` is a unary builtin with a
/// closed elementary antiderivative and `g` depends on the variable. `e^g` (`Pow(E, g)`) is
/// reported as `(Exp, g)`.
fn collect_chain_substitution_candidates(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    out: &mut Vec<(BuiltinFn, ExprId)>,
    seen: &mut std::collections::HashSet<ExprId>,
) {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            let inner = args[0];
            if let Some(builtin) = ctx.builtin_of(fn_id) {
                if chain_antiderivative_supported(builtin)
                    && contains_named_var(ctx, inner, var)
                    && seen.insert(expr)
                {
                    out.push((builtin, inner));
                }
            }
            collect_chain_substitution_candidates(ctx, inner, var, out, seen);
        }
        Expr::Pow(base, exponent) => {
            if matches!(ctx.get(base), Expr::Constant(cas_ast::Constant::E))
                && contains_named_var(ctx, exponent, var)
                && seen.insert(expr)
            {
                out.push((BuiltinFn::Exp, exponent));
            }
            collect_chain_substitution_candidates(ctx, base, var, out, seen);
            collect_chain_substitution_candidates(ctx, exponent, var, out, seen);
        }
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
            collect_chain_substitution_candidates(ctx, a, var, out, seen);
            collect_chain_substitution_candidates(ctx, b, var, out, seen);
        }
        Expr::Neg(inner) => collect_chain_substitution_candidates(ctx, inner, var, out, seen),
        _ => {}
    }
}

#[cfg(test)]
mod symbolic_table_tests {
    use super::*;

    #[test]
    fn nested_symbolic_u_matches_through_raw_derivative_powers() {
        // Regresión doble: (a) exp(u) canonizado como Pow(E,u) cuenta como
        // exterior Exp también en la RUTA (no solo en el narrador); (b) la
        // derivada CRUDA trae `u^(2-1)` sin plegar y la comparación debe
        // plegar con numeric_eval::as_rational_const (views:: devuelve None
        // sobre Sub — lección de la skill, mordida aquí antes de releerla).
        let mut ctx = Context::new();
        let integrand =
            cas_parser::parse("cos(x)*sin(x)*exp(sin(x)^2)", &mut ctx).expect("integrand");
        let out = symbolic_derivative_table_antiderivative(&mut ctx, integrand, "x")
            .expect("la tabla u-du debe casar cofactor = (1/2)·d(sin^2)/dx");
        let rendered = cas_formatter::render_expr(&ctx, out);
        assert!(
            rendered.contains("1/2") && rendered.contains("sin(x)^2"),
            "esperaba (1/2)·e^(sin(x)^2), got {rendered}"
        );
    }
}
