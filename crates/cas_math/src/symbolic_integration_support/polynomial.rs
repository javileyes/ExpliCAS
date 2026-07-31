//! `symbolic_integration_support`: familia `polynomial`.
//!
//! Ver la cabecera de `symbolic_integration_support.rs` para el contexto.

use super::*;

pub(super) fn compact_single_power_polynomial_arg(ctx: &mut Context, arg: ExprId) -> ExprId {
    let factored = factor(ctx, arg);
    if factored == arg {
        return arg;
    }

    match ctx.get(factored) {
        Expr::Pow(_, exp) if is_integer_power_at_least_two(ctx, *exp) => factored,
        _ => arg,
    }
}

fn is_integer_power_at_least_two(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(n)
            if n.denom().is_one() && *n >= BigRational::from_integer(2.into())
    )
}

pub(super) fn var_power(ctx: &Context, expr: ExprId, var: &str) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var => Some(BigRational::one()),
        Expr::Pow(base, exp) if is_var(ctx, *base, var) => rational_constant_value(ctx, *exp),
        _ => None,
    }
}

pub(super) fn positive_linear_polynomial_coeffs(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(BigRational, BigRational)> {
    let poly = Polynomial::from_expr(ctx, expr, var).ok()?;
    if poly.degree() != 1 {
        return None;
    }

    let offset = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let slope = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    (slope.is_positive() && offset.is_positive()).then_some((slope, offset))
}

fn non_integrating_parameter_square(ctx: &Context, expr: ExprId, var: &str) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if rational_constant_value(ctx, *exp) != Some(BigRational::from_integer(2.into())) {
        return None;
    }
    if contains_named_var(ctx, *base, var) {
        return None;
    }
    match ctx.get(*base) {
        Expr::Variable(_) => Some(*base),
        _ => None,
    }
}

pub(super) fn var_plus_symbolic_parameter_square(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };
    if is_var(ctx, *left, var) {
        return non_integrating_parameter_square(ctx, *right, var);
    }
    if is_var(ctx, *right, var) {
        return non_integrating_parameter_square(ctx, *left, var);
    }
    None
}

pub(super) fn numeric_square_scaled_var_plus_symbolic_parameter_square(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, BigRational)> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };

    let match_scaled_var = |term: ExprId, square_term: ExprId| {
        let (scale, power) = scaled_var_power_term(ctx, term, var)?;
        if power != BigRational::one() {
            return None;
        }
        let scale_root = exact_rational_sqrt(&scale)?;
        if scale_root.is_zero() {
            return None;
        }
        let parameter = non_integrating_parameter_square(ctx, square_term, var)?;
        Some((parameter, scale_root))
    };

    match_scaled_var(*left, *right).or_else(|| match_scaled_var(*right, *left))
}

fn symbolic_parameter_square_times_var(ctx: &Context, expr: ExprId, var: &str) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    let mut scale = BigRational::one();
    let mut saw_var = false;
    let mut parameter = None;

    for factor in factors {
        if is_var(ctx, factor, var) {
            if saw_var {
                return None;
            }
            saw_var = true;
        } else if let Some(square_parameter) = non_integrating_parameter_square(ctx, factor, var) {
            if parameter.is_some() {
                return None;
            }
            parameter = Some(square_parameter);
        } else {
            scale *= rational_constant_value(ctx, factor)?;
        }
    }

    (saw_var && scale.is_one()).then_some(parameter?)
}

pub(super) fn symbolic_parameter_square_times_var_plus_one(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };
    if is_number(ctx, *left, 1) {
        return symbolic_parameter_square_times_var(ctx, *right, var);
    }
    if is_number(ctx, *right, 1) {
        return symbolic_parameter_square_times_var(ctx, *left, var);
    }
    None
}

pub(super) fn affine_polynomial(ctx: &Context, expr: ExprId, var: &str) -> Option<Polynomial> {
    let poly = Polynomial::from_expr(ctx, expr, var).ok()?;
    (poly.degree() == 1).then_some(poly)
}

pub(super) fn positive_constant_difference(
    left: &Polynomial,
    right: &Polynomial,
) -> Option<BigRational> {
    let diff = left.sub(right);
    if diff.degree() != 0 {
        return None;
    }
    let constant = diff
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    constant.is_positive().then_some(constant)
}

pub(super) fn single_affine_factor(
    ctx: &Context,
    factors: &[ExprId],
    var: &str,
) -> Option<Polynomial> {
    let [factor] = factors else {
        return None;
    };
    affine_polynomial(ctx, *factor, var)
}

pub(super) fn affine_radicand_polynomial(
    ctx: &Context,
    radicand: ExprId,
    var: &str,
) -> Option<Polynomial> {
    let poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
    (poly.degree() == 1).then_some(poly)
}

pub(super) fn polynomial_ratio_to_expr_factor(
    ctx: &Context,
    target: &Polynomial,
    factor: ExprId,
    var: &str,
) -> Option<BigRational> {
    let factor_poly = Polynomial::from_expr(ctx, factor, var).ok()?;
    constant_polynomial_ratio(target, &factor_poly)
}

fn is_var_square(ctx: &Context, expr: ExprId, var: &str) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => is_var(ctx, *base, var) && is_number(ctx, *exp, 2),
        _ => false,
    }
}

pub(super) fn is_var_square_plus_one(ctx: &Context, expr: ExprId, var: &str) -> bool {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            (is_var_square(ctx, *l, var) && is_number(ctx, *r, 1))
                || (is_number(ctx, *l, 1) && is_var_square(ctx, *r, var))
        }
        _ => false,
    }
}

/// `int |a*x + b| dx = (a*x + b) * |a*x + b| / (2a)` for an affine argument.
///
/// Over the reals `|h| = sqrt(h^2)`, and `d/dx[(h)|h|/(2a)] = |h|` because
/// `h * sign(h) = |h|`. Covers `int sqrt(x^2) dx = int |x| dx = x|x|/2` (the
/// integrand `sqrt(x^2)` is canonicalized to `|x|` before integration). Declines
/// non-affine arguments, whose antiderivative is piecewise across the roots.
pub(super) fn abs_affine_antiderivative(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    let polynomial = Polynomial::from_expr(ctx, arg, var).ok()?;
    if polynomial.degree() != 1 {
        return None;
    }
    let slope = polynomial.leading_coeff();
    if slope.is_zero() {
        return None;
    }
    let abs_arg = ctx.call_builtin(BuiltinFn::Abs, vec![arg]);
    let product = mul2_raw(ctx, arg, abs_arg);
    let scale = BigRational::new(1.into(), 2.into()) / slope;
    Some(scale_rational_term(ctx, scale, product))
}

/// `int sign(a*x + b) dx = |a*x + b| / a` for an affine argument. The companion
/// of `d/dx |a*x+b| = a*sign(a*x+b)`; `int sign(x) dx = |x|`. Declines non-affine
/// arguments.
pub(super) fn sign_affine_antiderivative(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    let polynomial = Polynomial::from_expr(ctx, arg, var).ok()?;
    if polynomial.degree() != 1 {
        return None;
    }
    let slope = polynomial.leading_coeff();
    if slope.is_zero() {
        return None;
    }
    let abs_arg = ctx.call_builtin(BuiltinFn::Abs, vec![arg]);
    let scale = BigRational::one() / slope;
    Some(scale_rational_term(ctx, scale, abs_arg))
}

/// Whether `expr` is `x^(-1/2)` (the reciprocal square root of the bare variable).
pub(super) fn is_negative_half_power_of_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => is_var(ctx, *base, var) && is_negative_half(ctx, *exp),
        _ => false,
    }
}

/// Fold `Pow(Pow(b, p), q) -> Pow(b, p*q)` for numeric `p, q` (and `Pow(b, 1) ->
/// b`) throughout an expression. Used to collapse `(sqrt(x))^k = x^(k/2)` left by
/// a `u = sqrt(x)` back-substitution, which the generic normalizer keeps nested
/// for fractional inner exponents.
pub(super) fn fold_nested_numeric_powers(ctx: &mut Context, expr: ExprId) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) => {
            let base = fold_nested_numeric_powers(ctx, base);
            let exp = fold_nested_numeric_powers(ctx, exp);
            if let Expr::Pow(inner_base, inner_exp) = ctx.get(base).clone() {
                // Accept either a literal Number or a folded rational such as
                // Div(1, 2) for the exponents.
                if let (Some(p), Some(q)) = (
                    crate::numeric_eval::as_rational_const(ctx, inner_exp),
                    crate::numeric_eval::as_rational_const(ctx, exp),
                ) {
                    let product = &p * &q;
                    if product.is_one() {
                        return inner_base;
                    }
                    let new_exp = ctx.add(Expr::Number(product));
                    return ctx.add(Expr::Pow(inner_base, new_exp));
                }
            }
            ctx.add(Expr::Pow(base, exp))
        }
        Expr::Add(l, r) => {
            let l = fold_nested_numeric_powers(ctx, l);
            let r = fold_nested_numeric_powers(ctx, r);
            ctx.add(Expr::Add(l, r))
        }
        Expr::Sub(l, r) => {
            let l = fold_nested_numeric_powers(ctx, l);
            let r = fold_nested_numeric_powers(ctx, r);
            ctx.add(Expr::Sub(l, r))
        }
        Expr::Mul(l, r) => {
            let l = fold_nested_numeric_powers(ctx, l);
            let r = fold_nested_numeric_powers(ctx, r);
            mul2_raw(ctx, l, r)
        }
        Expr::Div(l, r) => {
            let l = fold_nested_numeric_powers(ctx, l);
            let r = fold_nested_numeric_powers(ctx, r);
            ctx.add(Expr::Div(l, r))
        }
        Expr::Neg(inner) => {
            let inner = fold_nested_numeric_powers(ctx, inner);
            ctx.add(Expr::Neg(inner))
        }
        Expr::Function(fn_id, args) => {
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|&a| fold_nested_numeric_powers(ctx, a))
                .collect();
            ctx.add(Expr::Function(fn_id, new_args))
        }
        _ => expr,
    }
}

pub(super) fn powered_unary_builtin_arg(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
    power: i64,
) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if !is_number(ctx, *exp, power) {
        return None;
    }
    unary_builtin_arg(ctx, *base, builtin)
}

pub(super) fn squared_unary_builtin_arg(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<ExprId> {
    powered_unary_builtin_arg(ctx, expr, builtin, 2)
}

pub(super) fn bounded_positive_integer_power(
    ctx: &Context,
    expr: ExprId,
    min: i64,
    max: i64,
) -> Option<i64> {
    let Expr::Number(value) = ctx.get(expr) else {
        return None;
    };
    if !value.denom().is_one() {
        return None;
    }
    let value = value.to_integer().to_i64()?;
    (min..=max).contains(&value).then_some(value)
}

/// An integer power `u^e` of an already-built base: `u^0 = 1`, `u^1 = u`, else `Pow(u, e)`.
pub(super) fn build_signed_integer_power(ctx: &mut Context, u: ExprId, e: i64) -> ExprId {
    match e {
        0 => ctx.num(1),
        1 => u,
        _ => {
            let exp = ctx.add(Expr::Number(BigRational::from_integer(e.into())));
            ctx.add(Expr::Pow(u, exp))
        }
    }
}

pub(super) fn neg_linear_part(ctx: &mut Context, part: ExprId) -> ExprId {
    if is_number(ctx, part, 0) {
        ctx.num(0)
    } else if let Expr::Number(value) = ctx.get(part) {
        ctx.add(Expr::Number(-value.clone()))
    } else {
        ctx.add(Expr::Neg(part))
    }
}

pub(super) fn polynomial_or_symbolic_linear_cofactor_scale(
    ctx: &mut Context,
    cofactor: ExprId,
    arg: ExprId,
    var: &str,
) -> Option<(BigRational, bool)> {
    if let (Ok(cofactor_poly), Ok(arg_poly)) = (
        Polynomial::from_expr(ctx, cofactor, var),
        Polynomial::from_expr(ctx, arg, var),
    ) {
        let derivative = arg_poly.derivative();
        return constant_polynomial_ratio(&cofactor_poly, &derivative).map(|scale| (scale, false));
    }

    if let Some(dependent_arg) = additive_var_dependent_part(ctx, arg, var) {
        if let (Ok(cofactor_poly), Ok(arg_poly)) = (
            Polynomial::from_expr(ctx, cofactor, var),
            Polynomial::from_expr(ctx, dependent_arg, var),
        ) {
            let derivative = arg_poly.derivative();
            return constant_polynomial_ratio(&cofactor_poly, &derivative)
                .map(|scale| (scale, false));
        }
    }

    let (slope, _) = get_linear_coeffs(ctx, arg, var)?;
    if contains_named_var(ctx, slope, var) {
        return None;
    }

    if compare_expr(ctx, cofactor, slope) == Ordering::Equal {
        return Some((BigRational::one(), true));
    }

    if let Expr::Neg(inner) = ctx.get(cofactor).clone() {
        if compare_expr(ctx, inner, slope) == Ordering::Equal {
            return Some((-BigRational::one(), true));
        }
    }

    if let Expr::Neg(inner) = ctx.get(slope).clone() {
        if compare_expr(ctx, cofactor, inner) == Ordering::Equal {
            return Some((-BigRational::one(), true));
        }
    }

    None
}

pub(super) fn symbolic_linear_cofactor_scale_expr(
    ctx: &mut Context,
    cofactor: ExprId,
    arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    if let Some((scale, _)) = polynomial_or_symbolic_linear_cofactor_scale(ctx, cofactor, arg, var)
    {
        return Some(ctx.add(Expr::Number(scale)));
    }

    if let Some(scale) = polynomial_symbolic_cofactor_scale_expr(ctx, cofactor, arg, var) {
        return Some(scale);
    }

    if let Expr::Neg(inner) = ctx.get(cofactor).clone() {
        let scale = symbolic_linear_cofactor_scale_expr(ctx, inner, arg, var)?;
        return Some(negate_scalar_expr(ctx, scale));
    }

    let (slope, _) = get_linear_coeffs(ctx, arg, var)?;
    if contains_named_var(ctx, slope, var) {
        return None;
    }

    let factors = mul_leaves(ctx, cofactor);
    for (idx, factor) in factors.iter().enumerate() {
        let Some(negative) = linear_slope_factor_sign(ctx, *factor, slope) else {
            continue;
        };

        let scale_factors: Vec<ExprId> = factors
            .iter()
            .enumerate()
            .filter_map(|(factor_idx, factor)| (factor_idx != idx).then_some(*factor))
            .collect();
        if scale_factors
            .iter()
            .any(|factor| contains_named_var(ctx, *factor, var))
        {
            return None;
        }

        let scale = if scale_factors.is_empty() {
            ctx.num(1)
        } else {
            build_balanced_mul(ctx, &scale_factors)
        };
        return Some(if negative {
            negate_scalar_expr(ctx, scale)
        } else {
            scale
        });
    }

    None
}

fn polynomial_symbolic_cofactor_scale_expr(
    ctx: &mut Context,
    cofactor: ExprId,
    arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    let derivative_arg = additive_var_dependent_part(ctx, arg, var).unwrap_or(arg);
    let arg_poly = Polynomial::from_expr(ctx, derivative_arg, var).ok()?;
    if arg_poly.degree() == 0 {
        return None;
    }
    let derivative = arg_poly.derivative();

    let factors = mul_leaves(ctx, cofactor);
    let mut dependent_factors = Vec::new();
    let mut scale_factors = Vec::new();
    for factor in factors {
        if contains_named_var(ctx, factor, var) {
            dependent_factors.push(factor);
        } else {
            scale_factors.push(factor);
        }
    }
    if dependent_factors.is_empty() || scale_factors.is_empty() {
        return None;
    }

    let dependent = if dependent_factors.len() == 1 {
        dependent_factors[0]
    } else {
        build_balanced_mul(ctx, &dependent_factors)
    };
    let dependent_poly = Polynomial::from_expr(ctx, dependent, var).ok()?;
    let rational_scale = constant_polynomial_ratio(&dependent_poly, &derivative)?;
    if rational_scale.is_zero() {
        return None;
    }

    let mut numeric_scale = rational_scale;
    let mut symbolic_scale_factors = Vec::new();
    for factor in scale_factors {
        if let Some(value) = rational_constant_value(ctx, factor) {
            numeric_scale *= value;
        } else {
            symbolic_scale_factors.push(factor);
        }
    }

    let scale = if symbolic_scale_factors.is_empty() {
        ctx.num(1)
    } else if symbolic_scale_factors.len() == 1 {
        symbolic_scale_factors[0]
    } else {
        build_balanced_mul(ctx, &symbolic_scale_factors)
    };
    Some(scale_rational_term(ctx, numeric_scale, scale))
}

fn linear_slope_factor_sign(ctx: &mut Context, factor: ExprId, slope: ExprId) -> Option<bool> {
    if compare_expr(ctx, factor, slope) == Ordering::Equal {
        return Some(false);
    }

    if let Expr::Neg(inner) = ctx.get(factor).clone() {
        if compare_expr(ctx, inner, slope) == Ordering::Equal {
            return Some(true);
        }
    }

    if let Expr::Neg(inner) = ctx.get(slope).clone() {
        if compare_expr(ctx, factor, inner) == Ordering::Equal {
            return Some(true);
        }
    }

    None
}

pub(super) fn same_structural_or_linear_arg(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    var: &str,
) -> bool {
    if compare_expr(ctx, left, right) == Ordering::Equal {
        return true;
    }

    let (Some((left_slope, left_intercept)), Some((right_slope, right_intercept))) = (
        get_linear_coeffs(ctx, left, var),
        get_linear_coeffs(ctx, right, var),
    ) else {
        return false;
    };

    compare_expr(ctx, left_slope, right_slope) == Ordering::Equal
        && compare_expr(ctx, left_intercept, right_intercept) == Ordering::Equal
}

pub(super) fn polynomial_product_from_factors(
    ctx: &mut Context,
    factors: &[ExprId],
    var: &str,
) -> Option<Polynomial> {
    if factors.is_empty() {
        return Some(Polynomial::one(var.to_string()));
    }

    let product = if factors.len() == 1 {
        factors[0]
    } else {
        build_balanced_mul(ctx, factors)
    };
    Polynomial::from_expr(ctx, product, var).ok()
}

pub(super) fn positive_square_constant_plus_square_arg(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let Expr::Add(left, right) = ctx.get(expr).clone() else {
        return None;
    };

    if let Some(root) = positive_constant_like_root(ctx, left, var) {
        if let Some(arg) = square_base(ctx, right) {
            return Some((arg, root));
        }
    }
    if let Some(root) = positive_constant_like_root(ctx, right, var) {
        if let Some(arg) = square_base(ctx, left) {
            return Some((arg, root));
        }
    }

    None
}

fn positive_symbolic_square_radius_plus_square_arg(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let Expr::Add(left, right) = ctx.get(expr).clone() else {
        return None;
    };

    let left_base = square_base(ctx, left)?;
    let right_base = square_base(ctx, right)?;
    let left_depends_on_var = contains_named_var(ctx, left_base, var);
    let right_depends_on_var = contains_named_var(ctx, right_base, var);

    match (left_depends_on_var, right_depends_on_var) {
        (true, false) if rational_constant_value(ctx, right_base).is_none() => {
            Some((left_base, right_base))
        }
        (false, true) if rational_constant_value(ctx, left_base).is_none() => {
            Some((right_base, left_base))
        }
        _ => None,
    }
}

pub(super) fn positive_one_plus_square_arg(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let (arg, radius) = positive_square_constant_plus_square_arg(ctx, expr, "")?;
    is_number(ctx, radius, 1).then_some(arg)
}

pub(super) fn positive_square_constant_plus_expanded_square_arg(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let terms = AddView::from_expr(ctx, expr).terms;

    for (constant_index, (constant_term, constant_sign)) in terms.iter().enumerate() {
        if *constant_sign != Sign::Pos {
            continue;
        }
        let Some(constant_root) = positive_constant_like_root(ctx, *constant_term, var) else {
            continue;
        };

        let mut remainder_terms = Vec::new();
        for (term_index, (term, sign)) in terms.iter().enumerate() {
            if term_index == constant_index {
                continue;
            }
            remainder_terms.push(match sign {
                Sign::Pos => *term,
                Sign::Neg => ctx.add(Expr::Neg(*term)),
            });
        }

        if remainder_terms.len() != 3 {
            continue;
        }

        let square = build_balanced_add(ctx, &remainder_terms);
        let Some((left, right, is_sub)) =
            crate::perfect_square_support::try_match_perfect_square_trinomial(ctx, square)
        else {
            continue;
        };
        let arg = if is_sub {
            ctx.add(Expr::Sub(left, right))
        } else {
            ctx.add(Expr::Add(left, right))
        };
        return Some((arg, constant_root));
    }

    None
}

fn positive_symbolic_square_radius_plus_expanded_square_arg(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let mut remainder_terms = Vec::new();
    let mut radius = None;

    for (term, sign) in AddView::from_expr(ctx, expr).terms {
        if sign == Sign::Pos && radius.is_none() {
            if let Some(base) = square_base(ctx, term) {
                if !contains_named_var(ctx, base, var)
                    && rational_constant_value(ctx, base).is_none()
                {
                    radius = Some(base);
                    continue;
                }
            }
        }
        remainder_terms.push(match sign {
            Sign::Pos => term,
            Sign::Neg => ctx.add(Expr::Neg(term)),
        });
    }

    let radius = radius?;
    if remainder_terms.len() != 3 {
        return None;
    }

    let square = build_balanced_add(ctx, &remainder_terms);
    let (left, right, is_sub) =
        crate::perfect_square_support::try_match_perfect_square_trinomial(ctx, square)?;
    let arg = if is_sub {
        ctx.add(Expr::Sub(left, right))
    } else {
        ctx.add(Expr::Add(left, right))
    };
    Some((arg, radius))
}

fn positive_one_plus_expanded_square_arg(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let mut remainder_terms = Vec::new();
    let mut found_one = false;

    for (term, sign) in AddView::from_expr(ctx, expr).terms {
        if sign == Sign::Pos && is_number(ctx, term, 1) && !found_one {
            found_one = true;
            continue;
        }
        remainder_terms.push(match sign {
            Sign::Pos => term,
            Sign::Neg => ctx.add(Expr::Neg(term)),
        });
    }

    if !found_one || remainder_terms.len() != 3 {
        return None;
    }

    let square = build_balanced_add(ctx, &remainder_terms);
    let (left, right, is_sub) =
        crate::perfect_square_support::try_match_perfect_square_trinomial(ctx, square)?;
    Some(if is_sub {
        ctx.add(Expr::Sub(left, right))
    } else {
        ctx.add(Expr::Add(left, right))
    })
}

fn symbolic_scaled_linear_arg_and_slope(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let (arg, slope) = nonzero_linear_arg_and_slope(ctx, arg, var)?;
    if rational_constant_value(ctx, slope).is_some() {
        return None;
    }

    Some((arg, slope))
}

pub(super) fn symbolic_scaled_linear_square_arg_and_slope(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    if let Some((arg, radius)) = positive_symbolic_square_radius_plus_square_arg(ctx, expr, var) {
        let (arg, slope) = nonzero_linear_arg_and_slope(ctx, arg, var)?;
        return Some(arctan_positive_quadratic_arg_and_scale_from_linear(
            ctx, arg, radius, slope,
        ));
    }

    if let Some((arg, radius)) =
        positive_symbolic_square_radius_plus_expanded_square_arg(ctx, expr, var)
    {
        let (arg, slope) = nonzero_linear_arg_and_slope(ctx, arg, var)?;
        return Some(arctan_positive_quadratic_arg_and_scale_from_linear(
            ctx, arg, radius, slope,
        ));
    }

    if let Some((arg, radius)) = positive_square_constant_plus_square_arg(ctx, expr, var) {
        return arctan_positive_quadratic_arg_and_scale(ctx, arg, radius, var);
    }

    if let Some((arg, radius)) = positive_square_constant_plus_expanded_square_arg(ctx, expr, var) {
        return arctan_positive_quadratic_arg_and_scale(ctx, arg, radius, var);
    }

    if let Some(arg) = positive_one_plus_expanded_square_arg(ctx, expr) {
        return symbolic_scaled_linear_arg_and_slope(ctx, arg, var);
    }

    let square_term = positive_one_plus_non_one_term(ctx, expr)?;
    let factors = mul_leaves(ctx, square_term);
    let mut linear_base = None;
    let mut linear_slope = None;
    let mut scale_bases = Vec::new();

    for factor in factors {
        let base = square_base(ctx, factor)?;
        if contains_named_var(ctx, base, var) {
            if linear_base.is_some() {
                return None;
            }
            let (slope, _) = get_linear_coeffs(ctx, base, var)?;
            if contains_named_var(ctx, slope, var) || is_number(ctx, slope, 0) {
                return None;
            }
            linear_base = Some(base);
            linear_slope = Some(slope);
        } else {
            scale_bases.push(base);
        }
    }

    if linear_base.is_none() || scale_bases.is_empty() {
        return None;
    }

    let scale = build_balanced_mul(ctx, &scale_bases);
    if rational_constant_value(ctx, scale).is_some() {
        return None;
    }

    let linear_base = linear_base?;
    let slope = linear_slope?;
    let arg = mul2_raw(ctx, scale, linear_base);
    let derivative_scale = if is_number(ctx, slope, 1) {
        scale
    } else {
        mul2_raw(ctx, scale, slope)
    };
    Some((arg, derivative_scale))
}

fn square_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    is_number(ctx, *exp, 2).then_some(*base)
}

pub(super) fn polynomial_antiderivative(poly: &Polynomial) -> Polynomial {
    let mut coeffs = Vec::with_capacity(poly.coeffs.len() + 1);
    coeffs.push(BigRational::zero());
    for (idx, coeff) in poly.coeffs.iter().enumerate() {
        let power = BigRational::from_integer(((idx as i64) + 1).into());
        coeffs.push(coeff / power);
    }
    Polynomial::new(coeffs, poly.var.clone())
}

pub(super) fn polynomial_antiderivative_expr(ctx: &mut Context, poly: &Polynomial) -> ExprId {
    let primitive = polynomial_antiderivative(poly);
    let nonzero_terms: Vec<_> = primitive
        .coeffs
        .iter()
        .enumerate()
        .filter(|(_, coeff)| !coeff.is_zero())
        .collect();
    if nonzero_terms.len() != 1 {
        return primitive.to_expr(ctx);
    }

    let (power, coeff) = nonzero_terms[0];
    if power == 0 || coeff.is_one() || coeff == &-BigRational::one() {
        return primitive.to_expr(ctx);
    }

    let var_expr = ctx.var(&primitive.var);
    let monomial = if power == 1 {
        var_expr
    } else {
        let exponent = ctx.num(power as i64);
        ctx.add(Expr::Pow(var_expr, exponent))
    };

    if coeff.numer().abs().is_one() {
        let denominator = ctx.add(Expr::Number(BigRational::from_integer(
            coeff.denom().clone(),
        )));
        let quotient = ctx.add(Expr::Div(monomial, denominator));
        if coeff.is_negative() {
            ctx.add(Expr::Neg(quotient))
        } else {
            quotient
        }
    } else {
        primitive.to_expr(ctx)
    }
}

pub(super) fn affine_args_are_opposite(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
    var: &str,
) -> bool {
    let Ok(left_poly) = Polynomial::from_expr(ctx, left, var) else {
        return false;
    };
    let Ok(right_poly) = Polynomial::from_expr(ctx, right, var) else {
        return false;
    };
    left_poly == right_poly.neg()
}

pub(super) fn unit_minus_square(ctx: &mut Context, arg: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let arg_sq = ctx.add(Expr::Pow(arg, two));
    ctx.add(Expr::Sub(one, arg_sq))
}

pub(super) fn scaled_unit_minus_square_linear_radicand(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
    coeff: &BigRational,
) -> Option<ExprId> {
    if coeff.is_zero() {
        return None;
    }
    if coeff.is_one() {
        return Some(unit_minus_square(ctx, arg));
    }

    let raw = unit_minus_square(ctx, arg);
    let poly = Polynomial::from_expr(ctx, raw, var).ok()?;
    let coeff_square = coeff * coeff;
    Some(poly.div_scalar(&coeff_square).to_expr(ctx))
}

pub(super) fn odd_power_times_quadratic_function_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() != 2 {
        return None;
    }
    // One factor is x^(2k+1); the other is f(c x^2). Try both orders.
    let (power, func_factor) =
        if let Some(p) = bare_var_odd_power_at_least_three(ctx, factors[0], var) {
            (p, factors[1])
        } else if let Some(p) = bare_var_odd_power_at_least_three(ctx, factors[1], var) {
            (p, factors[0])
        } else {
            return None;
        };
    if !is_elementary_function_of_pure_quadratic(ctx, func_factor, var) {
        return None;
    }
    let k = (power - 1) / 2; // power = 2k+1

    // Fresh substitution variable u, not colliding with the integrand's vars.
    let used = cas_ast::collect_variables(ctx, expr);
    let u_name = ["u", "u_", "u_sub"]
        .iter()
        .find(|candidate| !used.contains(**candidate) && *candidate != &var)?
        .to_string();
    let u_var = ctx.var(&u_name);
    let two = ctx.num(2);
    let x_squared = {
        let var_expr = ctx.var(var);
        ctx.add(Expr::Pow(var_expr, two))
    };

    // Build u^k * f(c u) by replacing x^2 with u inside f(c x^2).
    let func_in_u = crate::substitute::substitute_power_aware(
        ctx,
        func_factor,
        x_squared,
        u_var,
        crate::substitute::SubstituteOptions::exact(),
    );
    let u_power = {
        let k_expr = ctx.num(k);
        ctx.add(Expr::Pow(u_var, k_expr))
    };
    let integrand_u = ctx.add(Expr::Mul(u_power, func_in_u));
    let integral_u = integrate_symbolic_expr(ctx, integrand_u, &u_name)?;
    let integral_u = cas_ast::hold::unwrap_internal_hold(ctx, integral_u);

    // Back-substitute u = x^2 and apply the 1/2 from x dx = du/2.
    let back = crate::substitute::substitute_power_aware(
        ctx,
        integral_u,
        u_var,
        x_squared,
        crate::substitute::SubstituteOptions::exact(),
    );
    let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    Some(ctx.add(Expr::Mul(half, back)))
}

/// `Pow(var, p)` with `p` an odd integer `>= 3` -> `p`.
fn bare_var_odd_power_at_least_three(ctx: &Context, expr: ExprId, var: &str) -> Option<i64> {
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    if !is_var(ctx, base, var) {
        return None;
    }
    let value = crate::numeric_eval::as_rational_const(ctx, exp)?;
    if !value.is_integer() {
        return None;
    }
    let p = value.to_integer().to_i64()?;
    if p >= 3 && p % 2 == 1 {
        Some(p)
    } else {
        None
    }
}

pub(super) fn polynomial_to_expr(ctx: &mut Context, poly: &Polynomial, var: &str) -> ExprId {
    let var_expr = ctx.var(var);
    let mut terms = Vec::new();
    for (degree, coeff) in poly.coeffs.iter().enumerate() {
        if coeff.is_zero() {
            continue;
        }
        let term = match degree {
            0 => ctx.num(1),
            1 => var_expr,
            _ => {
                let exponent = ctx.num(degree as i64);
                ctx.add(Expr::Pow(var_expr, exponent))
            }
        };
        terms.push(scale_rational_term(ctx, coeff.clone(), term));
    }
    if terms.is_empty() {
        return ctx.num(0);
    }
    build_balanced_add(ctx, &terms)
}

pub(super) fn polynomial_power_factor(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            let exponent = rational_constant_value(ctx, *exp)?;
            Some((*base, exponent))
        }
        _ => None,
    }
}

pub(super) fn is_positive_quadratic_polynomial(poly: &Polynomial) -> bool {
    if poly.degree() != 2 {
        return false;
    }

    let a = poly
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if a <= BigRational::zero() {
        return false;
    }
    let b = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let four = BigRational::from_integer(4.into());
    four * a * c - b.clone() * b > BigRational::zero()
}

pub(super) fn polynomial_nonzero_term_count(poly: &Polynomial) -> usize {
    poly.coeffs.iter().filter(|coeff| !coeff.is_zero()).count()
}

pub(super) fn positive_integer_power_value(ctx: &Context, expr: ExprId) -> Option<u32> {
    match ctx.get(expr) {
        Expr::Number(n) if n.denom().is_one() && n.is_positive() => n.to_integer().to_u32(),
        _ => None,
    }
}

pub(super) fn nonzero_linear_polynomial_slope(poly: &Polynomial) -> Option<BigRational> {
    if poly.degree() != 1 {
        return None;
    }
    let slope = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    (!slope.is_zero()).then_some(slope)
}

pub(super) fn nonzero_linear_polynomial_from_expr(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Result<Option<(Polynomial, BigRational)>, ()> {
    let poly = Polynomial::from_expr(ctx, expr, var).map_err(|_| ())?;
    Ok(nonzero_linear_polynomial_slope(&poly).map(|slope| (poly, slope)))
}

pub(super) fn polynomial_cofactor_excluding_index(
    ctx: &mut Context,
    factors: &[ExprId],
    excluded_index: usize,
    var: &str,
) -> Result<Polynomial, ()> {
    let cofactor = factor_product_excluding_index(ctx, factors, excluded_index);
    Polynomial::from_expr(ctx, cofactor, var).map_err(|_| ())
}

/// ∫ p(x)·a^x dx for a positive rational base `a ≠ 1` and a polynomial cofactor `p`
/// (degree ≥ 1), by repeated integration by parts:
///   ∫ p(x)·a^x dx = a^x · Σ_{k=0}^{deg p} (-1)^k p^(k)(x) / (ln a)^(k+1).
/// Since `a^x = e^(x ln a)`, the by-parts slope is the SYMBOLIC `ln a`; the rational-slope
/// `e^(cx)` kernel above cannot be reused, but the antiderivative has the same series shape.
pub(super) fn polynomial_times_constant_base_power_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    use num_traits::{One, Signed, Zero};

    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    // Locate the unique constant-base power factor `a^x = Pow(base, x)`.
    let mut power_index = None;
    let mut base_expr = None;
    let mut exponent_slope = None;
    for (index, &factor) in factors.iter().enumerate() {
        let Expr::Pow(base, exponent) = ctx.get(factor) else {
            continue;
        };
        let (base, exponent) = (*base, *exponent);
        // The exponent must be AFFINE in the variable, `m·x + c` with a
        // nonzero rational slope `m`. For `a^(m·x+c) = a^c·(a^m)^x` the
        // effective by-parts slope of the series is `m·ln(a)`; the constant
        // `a^c` is already carried inside the untouched power factor, so
        // only the slope enters the formula. Bare `a^x` is the `m=1, c=0`
        // special case (previously the ONLY accepted shape).
        let Ok(exp_poly) = Polynomial::from_expr(ctx, exponent, var) else {
            continue;
        };
        if exp_poly.degree() != 1 {
            continue;
        }
        let Some(slope) = exp_poly.coeffs.get(1).cloned() else {
            continue;
        };
        if slope.is_zero() {
            continue;
        }
        let Some(base_value) = crate::numeric_eval::as_rational_const(ctx, base) else {
            continue;
        };
        if !base_value.is_positive() || base_value.is_one() || base_value.is_zero() {
            continue;
        }
        if power_index.is_some() {
            return None; // more than one a^(m·x+c) factor — out of scope
        }
        power_index = Some(index);
        base_expr = Some(base);
        exponent_slope = Some(slope);
    }
    let power_index = power_index?;
    let base = base_expr?;
    let exponent_slope = exponent_slope?;
    let power_factor = factors[power_index];

    // The remaining factors must form a polynomial cofactor of degree ≥ 1.
    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(index, &factor)| (index != power_index).then_some(factor))
        .collect();
    let cofactor = if cofactor_factors.len() == 1 {
        cofactor_factors[0]
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };
    let poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let degree = poly.degree();
    if degree == 0 || degree > MAX_EXP_POLYNOMIAL_BY_PARTS_DEGREE {
        return None;
    }

    // inner = Σ_{k=0}^{deg} (-1)^k p^(k)(x) / (m·ln a)^(k+1), where the
    // effective by-parts slope is `m·ln a` for the affine exponent `m·x+c`
    // (m = 1 recovers the bare `a^x` denominator `ln a`).
    let ln_base = {
        let ln = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
        if exponent_slope.is_one() {
            ln
        } else {
            let slope_expr = ctx.add(Expr::Number(exponent_slope));
            ctx.add(Expr::Mul(slope_expr, ln))
        }
    };
    let mut derivative = poly.clone();
    let mut inner: Option<ExprId> = None;
    for k in 0..=degree {
        if derivative.is_zero() {
            break;
        }
        let numerator = derivative.to_expr(ctx);
        let power = ctx.num((k + 1) as i64);
        let denominator = ctx.add(Expr::Pow(ln_base, power));
        let term = ctx.add(Expr::Div(numerator, denominator));
        inner = Some(match inner {
            None if k % 2 == 0 => term,
            None => negate_scalar_expr(ctx, term),
            Some(acc) if k % 2 == 0 => ctx.add(Expr::Add(acc, term)),
            Some(acc) => ctx.add(Expr::Sub(acc, term)),
        });
        derivative = derivative.derivative();
    }
    let inner = inner?;
    Some(mul2_raw(ctx, power_factor, inner))
}

pub(super) fn add_polynomial_term(
    acc: Polynomial,
    term: &Polynomial,
    positive: bool,
) -> Polynomial {
    if positive {
        acc.add(term)
    } else {
        acc.sub(term)
    }
}

pub(super) fn is_polynomial_times_linear_function_target<F>(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    min_degree: usize,
    max_degree: usize,
    detector: F,
) -> bool
where
    F: Fn(&Context, ExprId) -> Option<(BuiltinFn, ExprId, Sign, ExprId)>,
{
    let (outer_sign, factors) = signed_mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return false;
    }

    for (function_index, factor) in factors.iter().enumerate() {
        if signed_linear_function_factor_parts(ctx, *factor, var, &detector).is_none() {
            continue;
        }

        let cofactor =
            signed_factor_product_excluding_index(ctx, &factors, function_index, outer_sign);
        let Ok(cofactor_poly) = Polynomial::from_expr(ctx, cofactor, var) else {
            continue;
        };
        if (min_degree..=max_degree).contains(&cofactor_poly.degree()) {
            return true;
        }
    }

    false
}

pub(super) fn integer_polynomial_content(poly: &Polynomial) -> Option<i64> {
    let mut content = 0_i64;
    for coeff in &poly.coeffs {
        if coeff.is_zero() {
            continue;
        }
        if !coeff.denom().is_one() {
            return None;
        }
        let value = coeff.to_integer().to_i64()?.checked_abs()?;
        content = if content == 0 {
            value
        } else {
            gcd_i64(content, value)
        };
    }
    (content > 1).then_some(content)
}

pub(super) fn constant_polynomial_value(poly: &Polynomial) -> Option<BigRational> {
    if poly.is_zero() {
        return Some(BigRational::zero());
    }
    if poly.degree() != 0 {
        return None;
    }

    poly.coeffs.first().cloned()
}

pub(super) fn exact_polynomial_square_plus_positive_constant(
    poly: &Polynomial,
) -> Option<(Polynomial, BigRational)> {
    if poly.is_zero() {
        return None;
    }

    let degree = poly.degree();
    if degree == 0 || !degree.is_multiple_of(2) {
        return None;
    }

    let root_degree = degree / 2;
    let leading = poly
        .coeffs
        .get(degree)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let leading_root = exact_rational_sqrt(&leading)?;
    if leading_root.is_zero() {
        return None;
    }

    let mut root_coeffs = vec![BigRational::zero(); root_degree + 1];
    root_coeffs[root_degree] = leading_root.clone();
    let two = BigRational::from_integer(2.into());

    for k in (0..root_degree).rev() {
        let target_degree = root_degree + k;
        let target = poly
            .coeffs
            .get(target_degree)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let mut known = BigRational::zero();

        for i in 0..=root_degree {
            if let Some(j) = target_degree.checked_sub(i) {
                if j <= root_degree && i != k && j != k {
                    known += root_coeffs[i].clone() * root_coeffs[j].clone();
                }
            }
        }

        root_coeffs[k] = (target - known) / (two.clone() * leading_root.clone());
    }

    let root = Polynomial::new(root_coeffs, poly.var.clone());
    let square = root.mul(&root);
    let len = poly.coeffs.len().max(square.coeffs.len());

    for idx in 1..len {
        let left = poly
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let right = square
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if left != right {
            return None;
        }
    }

    let constant = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero)
        - square
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(BigRational::zero);
    if constant > BigRational::zero() {
        Some((root, constant))
    } else {
        None
    }
}

pub(super) fn compact_polynomial_square_arg(
    ctx: &mut Context,
    poly: &Polynomial,
) -> Option<ExprId> {
    let root = exact_polynomial_square_root(poly)?;
    let base = root.to_expr(ctx);
    let exp = ctx.num(2);
    Some(ctx.add(Expr::Pow(base, exp)))
}

pub(super) fn exact_positive_constant_minus_scaled_polynomial_square(
    poly: &Polynomial,
) -> Option<(Polynomial, BigRational, BigRational)> {
    let leading = poly.leading_coeff();
    if !leading.is_negative() {
        return None;
    }

    let radicand_scale = -leading;
    if radicand_scale.is_zero() || radicand_scale.is_one() {
        return None;
    }

    let normalized = poly.div_scalar(&radicand_scale);
    let (arg_poly, offset_square) = exact_positive_constant_minus_polynomial_square(&normalized)?;
    Some((arg_poly, offset_square, radicand_scale))
}

pub(super) fn polynomial_exprs_equal(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
    var: &str,
) -> bool {
    let Ok(left_poly) = Polynomial::from_expr(ctx, left, var) else {
        return false;
    };
    let Ok(right_poly) = Polynomial::from_expr(ctx, right, var) else {
        return false;
    };

    left_poly.sub(&right_poly).is_zero()
}

pub(super) fn positive_quadratic_square_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    if numerator.is_zero() {
        return Some(ctx.num(0));
    }

    let base = match ctx.get(den) {
        Expr::Pow(base, exp) if is_number(ctx, *exp, 2) => *base,
        _ => return None,
    };

    let mut denominator = Polynomial::from_expr(ctx, base, var).ok()?;
    if denominator.degree() != 2 {
        return None;
    }
    if denominator.leading_coeff().is_negative() {
        denominator = denominator.neg();
    }

    if numerator.degree() > 2 {
        return None;
    }

    let a = denominator
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if a <= BigRational::zero() {
        return None;
    }
    let b = denominator
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = denominator
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);

    let two = BigRational::from_integer(2.into());
    let four = BigRational::from_integer(4.into());
    let discriminant_gap = four * a.clone() * c - b.clone() * b.clone();
    if discriminant_gap <= BigRational::zero() {
        return None;
    }

    let leading_numerator = numerator
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let quotient_scale = leading_numerator / a.clone();
    let quotient_poly = Polynomial::new(vec![quotient_scale.clone()], denominator.var.clone());
    let remainder = numerator.sub(&denominator.mul(&quotient_poly));
    let linear_remainder = remainder
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let derivative_scale = linear_remainder / (two.clone() * a.clone());
    let constant_remainder = remainder
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero)
        - derivative_scale.clone() * b.clone();

    if !quotient_scale.is_zero() || !derivative_scale.is_zero() {
        let mut terms = Vec::new();
        let arctan_scale = quotient_scale
            + two.clone() * a.clone() * constant_remainder.clone() / discriminant_gap.clone();
        if !arctan_scale.is_zero() {
            let arctan_numerator = Polynomial::new(vec![arctan_scale], denominator.var.clone());
            terms.push(arctan_scaled_quadratic_antiderivative(
                ctx,
                &arctan_numerator,
                &denominator,
            )?);
        }

        let rational_numerator = Polynomial::new(
            vec![
                constant_remainder.clone() * b / discriminant_gap.clone() - derivative_scale,
                constant_remainder * two * a / discriminant_gap,
            ],
            denominator.var.clone(),
        );
        if !rational_numerator.is_zero() {
            terms.push(positive_quadratic_over_positive_quadratic_rational_term(
                ctx,
                &rational_numerator,
                &denominator,
            ));
        }

        return match terms.len() {
            0 => None,
            1 => Some(terms[0]),
            _ => {
                let sum = build_balanced_add(ctx, &terms);
                Some(cas_ast::hold::wrap_hold(ctx, sum))
            }
        };
    }

    positive_quadratic_square_constant_antiderivative(ctx, constant_remainder, &denominator)
}

pub(super) fn positive_quadratic_cube_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    if numerator.is_zero() {
        return Some(ctx.num(0));
    }

    let base = match ctx.get(den) {
        Expr::Pow(base, exp) if is_number(ctx, *exp, 3) => *base,
        _ => return None,
    };

    let mut denominator = Polynomial::from_expr(ctx, base, var).ok()?;
    if denominator.degree() != 2 {
        return None;
    }
    if denominator.leading_coeff().is_negative() {
        denominator = denominator.neg();
    }

    let a = denominator
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if a <= BigRational::zero() {
        return None;
    }
    let b = denominator
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = denominator
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);

    let two = BigRational::from_integer(2.into());
    let three = BigRational::from_integer(3.into());
    let four = BigRational::from_integer(4.into());
    let discriminant_gap = four * a.clone() * c - b.clone() * b.clone();
    if discriminant_gap <= BigRational::zero() {
        return None;
    }

    if numerator.degree() > 2 {
        if numerator.degree() > 4 {
            return None;
        }

        let (quotient, remainder) = numerator.div_rem(&denominator).ok()?;
        if quotient.degree() > 2 {
            return None;
        }

        let mut terms = Vec::new();
        if !quotient.is_zero() {
            let quotient_expr = quotient.to_expr(ctx);
            let square_exp = ctx.num(2);
            let square_den = ctx.add(Expr::Pow(base, square_exp));
            terms.push(positive_quadratic_square_antiderivative(
                ctx,
                quotient_expr,
                square_den,
                var,
            )?);
        }
        if !remainder.is_zero() {
            let remainder_expr = remainder.to_expr(ctx);
            terms.push(positive_quadratic_cube_antiderivative(
                ctx,
                remainder_expr,
                den,
                var,
            )?);
        }

        return match terms.len() {
            0 => Some(ctx.num(0)),
            1 => Some(terms[0]),
            _ => {
                let sum = build_balanced_add(ctx, &terms);
                Some(cas_ast::hold::wrap_hold(ctx, sum))
            }
        };
    }

    let leading_numerator = numerator
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let quotient_scale = leading_numerator / a.clone();
    let quotient_poly = Polynomial::new(vec![quotient_scale.clone()], denominator.var.clone());
    let remainder = numerator.sub(&denominator.mul(&quotient_poly));
    let linear_remainder = remainder
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let derivative_scale = linear_remainder / (two.clone() * a.clone());
    let constant_remainder = remainder
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero)
        - derivative_scale.clone() * b.clone();

    let derivative_poly =
        Polynomial::new(vec![b, two.clone() * a.clone()], denominator.var.clone());
    let square_rational_numerator =
        scale_polynomial(&derivative_poly.mul(&denominator), quotient_scale.clone())
            .div_scalar(&discriminant_gap);
    let derivative_rational_numerator = Polynomial::new(
        vec![-derivative_scale / two.clone()],
        denominator.var.clone(),
    );
    let cube_direct_rational_numerator =
        scale_polynomial(&derivative_poly, constant_remainder.clone())
            .div_scalar(&(two.clone() * discriminant_gap.clone()));
    let cube_square_rational_scale = constant_remainder.clone() * three.clone() * a.clone()
        / (discriminant_gap.clone() * discriminant_gap.clone());
    let cube_square_rational_numerator = scale_polynomial(
        &derivative_poly.mul(&denominator),
        cube_square_rational_scale,
    );
    let rational_numerator = square_rational_numerator
        .add(&derivative_rational_numerator)
        .add(&cube_direct_rational_numerator)
        .add(&cube_square_rational_numerator);

    let mut terms = Vec::new();
    let arctan_scale = two.clone() * a.clone() * quotient_scale / discriminant_gap.clone()
        + constant_remainder * three * two * a.clone() * a
            / (discriminant_gap.clone() * discriminant_gap);
    if !arctan_scale.is_zero() {
        let arctan_numerator = Polynomial::new(vec![arctan_scale], denominator.var.clone());
        terms.push(arctan_scaled_quadratic_antiderivative(
            ctx,
            &arctan_numerator,
            &denominator,
        )?);
    }

    if !rational_numerator.is_zero() {
        terms.push(positive_quadratic_power_rational_term(
            ctx,
            &rational_numerator,
            &denominator,
            2,
        ));
    }

    match terms.len() {
        0 => Some(ctx.num(0)),
        1 => Some(terms[0]),
        _ => {
            let sum = build_balanced_add(ctx, &terms);
            Some(cas_ast::hold::wrap_hold(ctx, sum))
        }
    }
}

fn integrate_symbolic_is_positive_quadratic_power_target(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    power: i64,
    max_numerator_degree: usize,
) -> bool {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return false;
    };
    let Ok(numerator) = Polynomial::from_expr(ctx, *num, var) else {
        return false;
    };
    if numerator.degree() > max_numerator_degree {
        return false;
    }

    let base = match ctx.get(*den) {
        Expr::Pow(base, exp) if is_number(ctx, *exp, power) => *base,
        _ => return false,
    };
    let Ok(mut denominator) = Polynomial::from_expr(ctx, base, var) else {
        return false;
    };
    if denominator.degree() != 2 {
        return false;
    }
    if denominator.leading_coeff().is_negative() {
        denominator = denominator.neg();
    }

    let a = denominator
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if a <= BigRational::zero() {
        return false;
    }
    let b = denominator
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = denominator
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let four = BigRational::from_integer(4.into());

    four * a * c - b.clone() * b > BigRational::zero()
}

pub fn integrate_symbolic_is_positive_quadratic_cube_target(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> bool {
    integrate_symbolic_is_positive_quadratic_power_target(ctx, expr, var, 3, 4)
}

pub fn integrate_symbolic_is_positive_quadratic_square_target(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> bool {
    integrate_symbolic_is_positive_quadratic_power_target(ctx, expr, var, 2, 2)
}

pub fn integrate_symbolic_positive_quadratic_square_constant_reduction_expr(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };

    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    if numerator.degree() != 0 {
        return None;
    }
    let scale = numerator
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if scale.is_zero() {
        return None;
    }

    let base = match ctx.get(den) {
        Expr::Pow(base, exp) if is_number(ctx, *exp, 2) => *base,
        _ => return None,
    };
    let mut denominator = Polynomial::from_expr(ctx, base, var).ok()?;
    if denominator.degree() != 2 {
        return None;
    }
    if denominator.leading_coeff().is_negative() {
        denominator = denominator.neg();
    }

    let a = denominator
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if a <= BigRational::zero() {
        return None;
    }
    let b = denominator
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = denominator
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);

    let two = BigRational::from_integer(2.into());
    let four = BigRational::from_integer(4.into());
    let discriminant_gap = four * a.clone() * c - b.clone() * b;
    if discriminant_gap <= BigRational::zero() {
        return None;
    }

    let arctan_integrand_scale = two * a * scale.clone() / discriminant_gap;
    let arctan_integrand_numerator = Polynomial::new(
        vec![arctan_integrand_scale.clone()],
        denominator.var.clone(),
    );
    let derivative_integrand_numerator = Polynomial::new(vec![scale], denominator.var.clone())
        .sub(&scale_polynomial(&denominator, arctan_integrand_scale));

    if derivative_integrand_numerator.is_zero() {
        return None;
    }

    let arctan_integrand = positive_quadratic_over_positive_quadratic_rational_term(
        ctx,
        &arctan_integrand_numerator,
        &denominator,
    );
    let derivative_integrand = positive_quadratic_power_rational_term(
        ctx,
        &derivative_integrand_numerator,
        &denominator,
        2,
    );
    Some(build_balanced_add(
        ctx,
        &[arctan_integrand, derivative_integrand],
    ))
}

pub fn integrate_symbolic_positive_quadratic_cube_constant_reduction_expr(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };

    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    if numerator.degree() != 0 {
        return None;
    }
    let scale = numerator
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if scale.is_zero() {
        return None;
    }

    let base = match ctx.get(den) {
        Expr::Pow(base, exp) if is_number(ctx, *exp, 3) => *base,
        _ => return None,
    };
    let mut denominator = Polynomial::from_expr(ctx, base, var).ok()?;
    if denominator.degree() != 2 {
        return None;
    }
    if denominator.leading_coeff().is_negative() {
        denominator = denominator.neg();
    }

    let a = denominator
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if a <= BigRational::zero() {
        return None;
    }
    let b = denominator
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = denominator
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);

    let six = BigRational::from_integer(6.into());
    let four = BigRational::from_integer(4.into());
    let discriminant_gap = four * a.clone() * c - b.clone() * b;
    if discriminant_gap <= BigRational::zero() {
        return None;
    }

    let arctan_integrand_scale =
        six * a.clone() * a * scale.clone() / (discriminant_gap.clone() * discriminant_gap);
    let arctan_integrand_numerator = Polynomial::new(
        vec![arctan_integrand_scale.clone()],
        denominator.var.clone(),
    );
    let q_squared = denominator.mul(&denominator);
    let derivative_integrand_numerator = Polynomial::new(vec![scale], denominator.var.clone())
        .sub(&scale_polynomial(&q_squared, arctan_integrand_scale));

    if derivative_integrand_numerator.is_zero() {
        return None;
    }

    let arctan_integrand = positive_quadratic_over_positive_quadratic_rational_term(
        ctx,
        &arctan_integrand_numerator,
        &denominator,
    );
    let derivative_integrand = positive_quadratic_power_rational_term(
        ctx,
        &derivative_integrand_numerator,
        &denominator,
        3,
    );
    Some(build_balanced_add(
        ctx,
        &[arctan_integrand, derivative_integrand],
    ))
}

fn positive_quadratic_square_constant_antiderivative(
    ctx: &mut Context,
    scale: BigRational,
    denominator: &Polynomial,
) -> Option<ExprId> {
    if scale.is_zero() {
        return Some(ctx.num(0));
    }

    let a = denominator
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if a <= BigRational::zero() {
        return None;
    }
    let b = denominator
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = denominator
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);

    let two = BigRational::from_integer(2.into());
    let four = BigRational::from_integer(4.into());
    let discriminant_gap = four * a.clone() * c - b.clone() * b.clone();
    if discriminant_gap <= BigRational::zero() {
        return None;
    }

    let linear_numerator = Polynomial::new(
        vec![scale.clone() * b, scale.clone() * two.clone() * a.clone()],
        denominator.var.clone(),
    );
    let linear_numerator = linear_numerator.div_scalar(&discriminant_gap);
    let rational_term = positive_quadratic_over_positive_quadratic_rational_term(
        ctx,
        &linear_numerator,
        denominator,
    );

    let arctan_numerator = Polynomial::new(
        vec![two * a * scale / discriminant_gap],
        denominator.var.clone(),
    );
    let scaled_arctan =
        arctan_scaled_quadratic_antiderivative(ctx, &arctan_numerator, denominator)?;

    let sum = ctx.add(Expr::Add(rational_term, scaled_arctan));
    Some(cas_ast::hold::wrap_hold(ctx, sum))
}

pub(super) fn polynomial_pow(poly: &Polynomial, exponent: usize) -> Polynomial {
    let mut result = Polynomial::one(poly.var.clone());
    for _ in 0..exponent {
        result = result.mul(poly);
    }
    result
}

pub(super) fn polynomial_has_only_negative_terms(poly: &Polynomial) -> bool {
    let mut saw_nonzero = false;
    for coeff in poly.coeffs.iter().filter(|coeff| !coeff.is_zero()) {
        if !coeff.is_negative() {
            return false;
        }
        saw_nonzero = true;
    }
    saw_nonzero
}

fn is_definite_quadratic_polynomial(poly: &Polynomial) -> bool {
    if poly.degree() != 2 {
        return false;
    }

    let a = poly
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if a.is_zero() {
        return false;
    }
    let b = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let four = BigRational::from_integer(4.into());
    four * a * c - b.clone() * b > BigRational::zero()
}

pub(super) fn linear_positive_quadratic_factors(
    denominator: &Polynomial,
) -> Option<(Polynomial, usize, Polynomial)> {
    if denominator.degree() < 3 || denominator.degree() > 5 {
        return None;
    }

    for factor in denominator.factor_rational_roots() {
        let Ok(linear_factor) = normalized_partial_fraction_linear_factor(&factor).ok_or(()) else {
            continue;
        };

        let mut multiplicity = 0;
        let mut remaining = denominator.clone();
        loop {
            let (quotient, remainder) = remaining.div_rem(&linear_factor).ok()?;
            if !remainder.is_zero() {
                break;
            }
            multiplicity += 1;
            remaining = quotient;
        }

        if (1..=3).contains(&multiplicity) && is_definite_quadratic_polynomial(&remaining) {
            return Some((linear_factor, multiplicity, remaining));
        }
    }

    None
}

pub(super) fn multi_linear_positive_quadratic_factors(
    denominator: &Polynomial,
) -> Option<(Vec<PartialFractionLinearFactor>, Polynomial)> {
    const MAX_MIXED_PARTIAL_FRACTION_DEGREE: usize = 6;

    if denominator.degree() < 4 || denominator.degree() > MAX_MIXED_PARTIAL_FRACTION_DEGREE {
        return None;
    }

    let mut remaining = denominator.clone();
    let mut groups: Vec<PartialFractionLinearFactor> = Vec::new();
    for factor in denominator.factor_rational_roots() {
        let Some(factor) = normalized_partial_fraction_linear_factor(&factor) else {
            continue;
        };
        if groups.iter().any(|group| group.factor == factor) {
            continue;
        }

        let mut multiplicity = 0;
        loop {
            let (quotient, remainder) = remaining.div_rem(&factor).ok()?;
            if !remainder.is_zero() {
                break;
            }
            multiplicity += 1;
            remaining = quotient;
        }

        if multiplicity > 0 {
            groups.push(PartialFractionLinearFactor {
                factor,
                multiplicity,
            });
        }
    }

    if groups.len() < 2 || !is_definite_quadratic_polynomial(&remaining) {
        return None;
    }

    let linear_degree: usize = groups.iter().map(|group| group.multiplicity).sum();
    (linear_degree + remaining.degree() == denominator.degree()).then_some((groups, remaining))
}

fn square_power_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp)
            if rational_constant_value(ctx, *exp) == Some(BigRational::from_integer(2.into())) =>
        {
            Some(*base)
        }
        _ => None,
    }
}

fn symbolic_radius_base_from_square(
    ctx: &Context,
    radius_square: ExprId,
    var: &str,
) -> Option<ExprId> {
    if contains_named_var(ctx, radius_square, var)
        || rational_constant_value(ctx, radius_square).is_some()
    {
        return None;
    }
    let radius_base = square_power_base(ctx, radius_square)?;
    if contains_named_var(ctx, radius_base, var)
        || rational_constant_value(ctx, radius_base).is_some()
    {
        return None;
    }
    Some(radius_base)
}

fn symbolic_radius_square_arg_from_terms(
    ctx: &Context,
    radius_square: ExprId,
    arg_square: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    symbolic_radius_base_from_square(ctx, radius_square, var)?;
    let arg = square_power_base(ctx, arg_square)?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }
    Some((radius_square, arg))
}

fn symbolic_radius_arg_from_square_terms(
    ctx: &mut Context,
    square_terms: &[ExprId],
    var: &str,
) -> Option<ExprId> {
    if square_terms.len() != 3 {
        return None;
    }

    let square = build_balanced_add(ctx, square_terms);
    let (left, right, is_sub) =
        crate::perfect_square_support::try_match_perfect_square_trinomial(ctx, square)?;
    let arg = if is_sub {
        ctx.add(Expr::Sub(left, right))
    } else {
        ctx.add(Expr::Add(left, right))
    };
    if !contains_named_var(ctx, arg, var) {
        return None;
    }
    Some(arg)
}

pub(super) fn symbolic_radius_minus_square_arg(
    ctx: &Context,
    radicand: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let Expr::Sub(radius_square, arg_square) = ctx.get(radicand) else {
        return None;
    };
    symbolic_radius_square_arg_from_terms(ctx, *radius_square, *arg_square, var)
}

pub(super) fn symbolic_radius_minus_expanded_square_arg(
    ctx: &mut Context,
    radicand: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let terms = AddView::from_expr(ctx, radicand).terms;
    for (radius_index, (radius_square, sign)) in terms.iter().enumerate() {
        if *sign != Sign::Pos
            || symbolic_radius_base_from_square(ctx, *radius_square, var).is_none()
        {
            continue;
        }

        let mut square_terms = Vec::new();
        for (term_index, (term, term_sign)) in terms.iter().enumerate() {
            if term_index == radius_index {
                continue;
            }
            square_terms.push(match term_sign {
                Sign::Pos => ctx.add(Expr::Neg(*term)),
                Sign::Neg => *term,
            });
        }
        let Some(arg) = symbolic_radius_arg_from_square_terms(ctx, &square_terms, var) else {
            continue;
        };
        return Some((*radius_square, arg));
    }

    None
}

pub(super) fn symbolic_radius_plus_square_arg(
    ctx: &Context,
    radicand: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let Expr::Add(left, right) = ctx.get(radicand) else {
        return None;
    };
    symbolic_radius_plus_square_arg_from_terms(ctx, *left, *right, var)
        .or_else(|| symbolic_radius_plus_square_arg_from_terms(ctx, *right, *left, var))
}

fn symbolic_radius_plus_square_arg_from_terms(
    ctx: &Context,
    radius_square: ExprId,
    arg_square: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    symbolic_radius_square_arg_from_terms(ctx, radius_square, arg_square, var)
}

pub(super) fn symbolic_radius_plus_expanded_square_arg(
    ctx: &mut Context,
    radicand: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let terms = AddView::from_expr(ctx, radicand).terms;
    for (radius_index, (radius_square, sign)) in terms.iter().enumerate() {
        if *sign != Sign::Pos
            || symbolic_radius_base_from_square(ctx, *radius_square, var).is_none()
        {
            continue;
        }

        let mut square_terms = Vec::new();
        for (term_index, (term, term_sign)) in terms.iter().enumerate() {
            if term_index == radius_index {
                continue;
            }
            if *term_sign != Sign::Pos {
                square_terms.clear();
                break;
            }
            square_terms.push(*term);
        }
        let Some(arg) = symbolic_radius_arg_from_square_terms(ctx, &square_terms, var) else {
            continue;
        };
        return Some((*radius_square, arg));
    }

    None
}

pub(super) fn positive_linear_factor_domain_sample(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<BigRational> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    let mut lower: Option<BigRational> = None;
    let mut upper: Option<BigRational> = None;
    for factor in factors {
        let poly = Polynomial::from_expr(ctx, factor, var).ok()?;
        if poly.degree() != 1 {
            return None;
        }
        let constant = poly
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let slope = poly
            .coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if slope.is_zero() {
            return None;
        }

        let bound = -constant / slope.clone();
        if slope.is_positive() {
            lower = Some(match lower {
                Some(current) if current > bound => current,
                _ => bound,
            });
        } else {
            upper = Some(match upper {
                Some(current) if current < bound => current,
                _ => bound,
            });
        }
    }

    if let (Some(low), Some(high)) = (&lower, &upper) {
        if low >= high {
            return None;
        }
        return Some((low.clone() + high.clone()) / BigRational::from_integer(2.into()));
    }

    let one = BigRational::one();
    if let Some(low) = lower {
        return Some(low + one);
    }
    if let Some(high) = upper {
        return Some(high - one);
    }

    None
}

pub(super) fn constant_scaled_integrand_inner(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Neg(inner) => Some(*inner),
        Expr::Mul(left, right) => {
            let left_depends_on_var = contains_named_var(ctx, *left, var);
            let right_depends_on_var = contains_named_var(ctx, *right, var);
            match (left_depends_on_var, right_depends_on_var) {
                (false, true) => Some(*right),
                (true, false) => Some(*left),
                _ => None,
            }
        }
        _ => None,
    }
}

pub(super) fn multiply_constant_integral_result(
    ctx: &mut Context,
    scale: ExprId,
    integral: ExprId,
) -> ExprId {
    if let Expr::Div(num, den) = ctx.get(integral).clone() {
        if integral_factors_match(ctx, scale, den) {
            return num;
        }
        if let Some(scaled) = multiply_rational_fraction_integral_result(ctx, scale, num, den) {
            return scaled;
        }

        let scaled_num = multiply_rational_factor_if_possible(ctx, scale, num)
            .unwrap_or_else(|| mul2_raw(ctx, scale, num));
        if let Some(cancelled) = cancel_matching_factor_from_product(ctx, scaled_num, den) {
            return cancelled;
        }

        return ctx.add(Expr::Div(scaled_num, den));
    }

    multiply_rational_factor_if_possible(ctx, scale, integral)
        .unwrap_or_else(|| mul2_raw(ctx, scale, integral))
}

pub(super) fn table_reused_constant_integration_candidate(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> AlgorithmicIntegrationCandidate {
    let var_expr = ctx.var(var);
    let antiderivative = mul2_raw(ctx, expr, var_expr);
    AlgorithmicIntegrationCandidate::verified_table_reused(expr, var, antiderivative)
}

pub(super) fn multiply_linear_part(ctx: &mut Context, factor: ExprId, part: ExprId) -> ExprId {
    if is_number(ctx, part, 0) {
        ctx.num(0)
    } else if is_number(ctx, factor, 1) {
        part
    } else {
        mul2_raw(ctx, factor, part)
    }
}

pub(super) fn divide_linear_part(ctx: &mut Context, part: ExprId, denominator: ExprId) -> ExprId {
    if is_number(ctx, part, 0) {
        ctx.num(0)
    } else if is_number(ctx, denominator, 1) {
        part
    } else {
        ctx.add(Expr::Div(part, denominator))
    }
}
