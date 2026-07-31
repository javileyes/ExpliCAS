//! `symbolic_integration_support`: familia `inverse_trig`.
//!
//! Ver la cabecera de `symbolic_integration_support.rs` para el contexto.

use super::*;

pub(super) fn arctan_sqrt_var_reciprocal_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = arctan_sqrt_var_positive_linear_parts(ctx, num, den, var)?;
    arctan_sqrt_var_reciprocal_antiderivative_from_parts(
        ctx,
        parts.scale,
        parts.slope,
        parts.offset,
        var,
    )
}

pub(super) fn arctan_sqrt_var_symbolic_square_shift_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = arctan_sqrt_var_symbolic_square_shift_parts(ctx, num, den, var)?;
    arctan_sqrt_var_symbolic_square_shift_antiderivative_from_parts(
        ctx,
        parts.scale,
        parts.parameter,
        parts.argument,
        parts.argument_scale,
        var,
    )
}

fn arctan_sqrt_var_unit_shift_square_antiderivative_from_scale(
    ctx: &mut Context,
    scale: BigRational,
    offset: BigRational,
    offset_root: BigRational,
    var: &str,
) -> Option<ExprId> {
    if scale.is_zero() || !offset.is_positive() || !offset_root.is_positive() {
        return None;
    }

    let var_expr = ctx.var(var);
    let sqrt_var = ctx.call_builtin(BuiltinFn::Sqrt, vec![var_expr]);
    if offset.is_one() && offset_root.is_one() {
        let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![sqrt_var]);
        let one = ctx.num(1);
        let unit_shift = ctx.add(Expr::Add(var_expr, one));
        let quotient = ctx.add(Expr::Div(sqrt_var, unit_shift));
        let primitive = build_balanced_add(ctx, &[arctan, quotient]);
        return Some(scale_factor(ctx, scale, primitive));
    }

    let arctan_arg = if offset_root.is_one() {
        sqrt_var
    } else if offset_root < BigRational::one() {
        scale_factor(ctx, BigRational::one() / offset_root.clone(), sqrt_var)
    } else {
        let offset_root_expr = ctx.add(Expr::Number(offset_root.clone()));
        ctx.add(Expr::Div(sqrt_var, offset_root_expr))
    };
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arctan_arg]);
    let offset_expr = ctx.add(Expr::Number(offset.clone()));
    let shifted = ctx.add(Expr::Add(var_expr, offset_expr));
    let quotient_term = if offset.is_one() {
        ctx.add(Expr::Div(sqrt_var, shifted))
    } else if offset < BigRational::one() {
        let scaled_numerator = scale_factor(ctx, scale.clone() / offset.clone(), sqrt_var);
        ctx.add(Expr::Div(scaled_numerator, shifted))
    } else {
        let offset_expr = ctx.add(Expr::Number(offset.clone()));
        let offset_denominator = build_balanced_mul(ctx, &[offset_expr, shifted]);
        let quotient = ctx.add(Expr::Div(sqrt_var, offset_denominator));
        scale_factor(ctx, scale.clone(), quotient)
    };
    let arctan_term = scale_factor(ctx, scale.clone() / (offset.clone() * offset_root), arctan);
    Some(build_balanced_add(ctx, &[arctan_term, quotient_term]))
}

pub(super) fn arctan_sqrt_var_unit_shift_square_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = arctan_sqrt_var_unit_shift_square_parts(ctx, num, den, var)?;
    arctan_sqrt_var_unit_shift_square_antiderivative_from_scale(
        ctx,
        parts.scale,
        parts.offset,
        parts.offset_root,
        var,
    )
}

pub(super) fn arctan_sqrt_var_reciprocal_required_positive_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let matches_positive_linear_form =
        arctan_sqrt_var_positive_linear_parts(ctx, *num, *den, var).is_some();
    let matches_unit_shift_square_form =
        arctan_sqrt_var_unit_shift_square_parts(ctx, *num, *den, var).is_some();
    let matches_symbolic_square_shift_form =
        arctan_sqrt_var_symbolic_square_shift_parts(ctx, *num, *den, var).is_some();
    if matches_positive_linear_form
        || matches_unit_shift_square_form
        || matches_symbolic_square_shift_form
    {
        return Some(ctx.var(var));
    }
    None
}

pub(super) fn arctan_sqrt_var_symbolic_square_shift_required_nonzero_parameter(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    arctan_sqrt_var_symbolic_square_shift_parts(ctx, *num, *den, var).map(|parts| parts.parameter)
}

pub(super) fn arctan_sqrt_affine_output_scale(
    radicand: &Polynomial,
    kernel_scale: BigRational,
    denominator_root: &BigRational,
) -> Option<BigRational> {
    if kernel_scale.is_zero() || denominator_root.is_zero() {
        return None;
    }
    let slope = radicand.coeffs.get(1).cloned()?;
    if slope.is_zero() {
        return None;
    }
    Some(kernel_scale * BigRational::from_integer(2.into()) / (slope * denominator_root))
}

pub(super) fn arctan_sqrt_affine_derivative_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = arctan_sqrt_affine_derivative_parts(ctx, num, den, var)?;
    let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![parts.radicand]);
    let arg = scale_factor(ctx, parts.argument_scale, sqrt);
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arg]);
    Some(scale_factor(ctx, parts.scale, arctan))
}

fn arctan_sqrt_affine_derivative_required_positive_radicand(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    arctan_sqrt_affine_derivative_parts(ctx, *num, *den, var).map(|parts| parts.radicand)
}

pub fn integrate_symbolic_is_arctan_sqrt_affine_derivative_target(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return false;
    };
    arctan_sqrt_affine_derivative_parts(ctx, *num, *den, var).is_some()
}

pub fn integrate_symbolic_is_arctan_sqrt_var_reciprocal_target(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return false;
    };
    arctan_sqrt_var_positive_linear_parts(ctx, *num, *den, var).is_some()
        || arctan_sqrt_var_symbolic_square_shift_parts(ctx, *num, *den, var).is_some()
}

pub fn integrate_symbolic_is_arctan_sqrt_var_unit_shift_square_target(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> bool {
    integrate_symbolic_arctan_sqrt_var_unit_shift_square_key(ctx, expr, var)
        .is_some_and(|(scale, _)| !scale.is_zero())
}

pub fn integrate_symbolic_arctan_sqrt_var_unit_shift_square_key(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(BigRational, BigRational)> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let parts = arctan_sqrt_var_unit_shift_square_parts(ctx, *num, *den, var)?;
    (!parts.scale.is_zero()).then_some((parts.scale, parts.offset))
}

/// `int inv(x)/x^n dx` (integer `n >= 2`) for an inverse-trig `inv` of the bare
/// variable, by parts (`u = inv(x)`, `dv = x^-n dx`, `v = -1/((n-1) x^(n-1))`):
///
/// ```text
/// int inv(x)/x^n dx = -inv(x)/((n-1) x^(n-1)) + 1/(n-1) int inv'(x)/x^(n-1) dx
/// ```
///
/// The lower-degree tail is delegated to the integrator: for `arctan` it is
/// `int 1/(x^(n-1)(1+x^2))` and for `arcsin`/`arccos` it is
/// `int 1/(x^(n-1) sqrt(1-x^2))`. If the tail does not resolve the whole rule
/// self-gates to an honest residual.
pub(super) fn inverse_trig_over_power_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    // Denominator must be x^n with n >= 2 a small integer.
    let Expr::Pow(base, exp) = ctx.get(den).clone() else {
        return None;
    };
    if !is_var(ctx, base, var) {
        return None;
    }
    let n = match ctx.get(exp) {
        Expr::Number(value) if value.is_integer() => value.to_integer().to_i64()?,
        _ => return None,
    };
    if !(2..=8).contains(&n) {
        return None;
    }

    // Numerator must be a single inverse-trig of the bare variable.
    let Expr::Function(fn_id, args) = ctx.get(num).clone() else {
        return None;
    };
    if args.len() != 1 || !is_var(ctx, args[0], var) {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    if !matches!(
        builtin,
        BuiltinFn::Arcsin
            | BuiltinFn::Asin
            | BuiltinFn::Arccos
            | BuiltinFn::Acos
            | BuiltinFn::Arctan
            | BuiltinFn::Atan
    ) {
        return None;
    }

    // x^(n-1) (just x when n == 2).
    let lower_power = if n == 2 {
        ctx.var(var)
    } else {
        let var_expr = ctx.var(var);
        let exp_expr = ctx.num(n - 1);
        ctx.add(Expr::Pow(var_expr, exp_expr))
    };
    let inv_scale = BigRational::new(1.into(), (n - 1).into()); // 1/(n-1)

    // head = -1/(n-1) * inv(x)/x^(n-1)
    let head_div = ctx.add(Expr::Div(num, lower_power));
    let head = scale_rational_term(ctx, -inv_scale.clone(), head_div);

    // tail integrand = inv'(x)/x^(n-1), flattened to a single fraction (the
    // internal integrator does not pre-simplify a nested `(N/D)/x^(n-1)`).
    let derivative =
        crate::symbolic_differentiation_support::differentiate_symbolic_expr(ctx, num, var)?;
    let tail_integrand = match ctx.get(derivative).clone() {
        Expr::Div(num_d, den_d) => {
            let denom = ctx.add(Expr::Mul(den_d, lower_power));
            ctx.add(Expr::Div(num_d, denom))
        }
        _ => ctx.add(Expr::Div(derivative, lower_power)),
    };
    let tail = integrate_symbolic_expr(ctx, tail_integrand, var)?;
    let scaled_tail = scale_rational_term(ctx, inv_scale, tail);

    Some(ctx.add(Expr::Add(head, scaled_tail)))
}

/// Antiderivative of `arcsin(a x + b)^2` and `arccos(a x + b)^2`.
///
/// Both are elementary (integration by parts, twice). With `u = a x + b`,
///
/// ```text
/// G(u) = u·arcsin(u)^2 − 2u + 2·sqrt(1 − u²)·arcsin(u)        (arcsin)
/// G(u) = u·arccos(u)^2 − 2u − 2·sqrt(1 − u²)·arccos(u)        (arccos)
/// ```
///
/// and the antiderivative in `x` is `(1/a)·G(a x + b)`, since
/// `d/dx[(1/a)·G(a x + b)] = G'(a x + b) = inv(a x + b)^2`.  The only branch
/// difference between the two is the sign of the `sqrt` term.
///
/// `arctan(x)^2` is deliberately NOT handled: it is non-elementary (it reduces
/// to `∫ln(cos θ) dθ`), so it must stay an honest residual.
pub(super) fn inverse_trig_square_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    if !is_number(ctx, exp, 2) {
        return None;
    }

    let (fn_id, args) = match ctx.get(base).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return None,
    };

    let builtin = ctx.builtin_of(fn_id)?;
    if !matches!(
        builtin,
        BuiltinFn::Arcsin | BuiltinFn::Asin | BuiltinFn::Arccos | BuiltinFn::Acos
    ) {
        return None;
    }

    let arg = args[0];
    let (a_expr, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a_expr)?;
    if a.is_zero() {
        return None;
    }

    // head = arg·inv(arg)^2
    let inv = ctx.call_builtin(builtin, vec![arg]);
    let two = ctx.num(2);
    let inv_sq = ctx.add(Expr::Pow(inv, two));
    let head = mul2_raw(ctx, arg, inv_sq);

    // linear = −2·arg  (the −2/a·u part of G scaled back by 1/a gives −2x; the
    // constant offset folds into the integration constant)
    let linear = scale_rational_term(ctx, -BigRational::from_integer(2.into()), arg);

    // sqrt_signed = ±2·sqrt(1 − arg²)·inv(arg)
    let radicand = unit_minus_square(ctx, arg);
    let sqrt_term = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let sqrt_product = mul2_raw(ctx, sqrt_term, inv);
    let sqrt_scale = match builtin {
        BuiltinFn::Arcsin | BuiltinFn::Asin => BigRational::from_integer(2.into()),
        _ => -BigRational::from_integer(2.into()),
    };
    let sqrt_signed = scale_rational_term(ctx, sqrt_scale, sqrt_product);

    let head_plus_linear = ctx.add(Expr::Add(head, linear));
    let g = ctx.add(Expr::Add(head_plus_linear, sqrt_signed));

    let scale = BigRational::one() / a;
    Some(scale_rational_term(ctx, scale, g))
}

pub(super) fn arctan_positive_quadratic_arg_and_scale(
    ctx: &mut Context,
    arg: ExprId,
    radius: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let (arg, slope) = nonzero_linear_arg_and_slope(ctx, arg, var)?;
    Some(arctan_positive_quadratic_arg_and_scale_from_linear(
        ctx, arg, radius, slope,
    ))
}

pub(super) fn arctan_positive_quadratic_arg_and_scale_from_linear(
    ctx: &mut Context,
    arg: ExprId,
    radius: ExprId,
    slope: ExprId,
) -> (ExprId, ExprId) {
    if is_number(ctx, radius, 1) {
        return (arg, slope);
    }

    let scaled_arg = match ctx.get(radius).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Sqrt) =>
        {
            if let Some(radicand) = rational_constant_value(ctx, args[0]) {
                let numerator = mul2_raw(ctx, radius, arg);
                let denominator = ctx.add(Expr::Number(radicand));
                ctx.add(Expr::Div(numerator, denominator))
            } else {
                ctx.add(Expr::Div(arg, radius))
            }
        }
        _ => ctx.add(Expr::Div(arg, radius)),
    };
    let scale = mul2_raw(ctx, slope, radius);
    (scaled_arg, scale)
}

pub fn integrate_symbolic_is_positive_rational_quadratic_arctan_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return false,
    };

    if !matches!(ctx.get(num), Expr::Number(n) if n.is_one()) {
        return false;
    }

    symbolic_scaled_linear_square_arg_and_slope(ctx, den, var).is_some()
}

pub(super) fn arctan_symbolic_scaled_variable_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    if !is_number(ctx, num, 1) {
        return None;
    }

    let (arg, scale) = symbolic_scaled_linear_square_arg_and_slope(ctx, den, var)?;
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arg]);
    Some(ctx.add(Expr::Div(arctan, scale)))
}

pub(super) fn arctan_symbolic_scaled_variable_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    if !is_number(ctx, num, 1) {
        return None;
    }

    let (_, scale) = symbolic_scaled_linear_square_arg_and_slope(ctx, den, var)?;
    Some(scale)
}

pub(super) fn arctan_scaled_variable_antiderivative(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (coeff, offset) = get_linear_coeffs(ctx, arg, var)?;
    let coeff = rational_constant_value(ctx, coeff)?;
    if coeff.is_zero() {
        return None;
    }

    let zero_offset = is_number(ctx, offset, 0);
    let presentation_arg = if zero_offset && coeff.is_one() {
        arg
    } else {
        cas_ast::hold::wrap_hold(ctx, arg)
    };
    let arctan_arg = ctx.call_builtin(BuiltinFn::Arctan, vec![arg]);
    let leading_linear = if zero_offset {
        ctx.var(var)
    } else {
        let scale = BigRational::one() / coeff.clone();
        scale_rational_term(ctx, scale, presentation_arg)
    };
    let leading_term = mul2_raw(ctx, leading_linear, arctan_arg);

    let two = ctx.num(2);
    let arg_sq = ctx.add(Expr::Pow(presentation_arg, two));
    let one = ctx.num(1);
    let log_arg = ctx.add(Expr::Add(arg_sq, one));
    let log_term = ctx.call_builtin(BuiltinFn::Ln, vec![log_arg]);

    let two_coeff = coeff * BigRational::from_integer(2.into());
    let log_scale = BigRational::one() / two_coeff;
    let signed_log_term = scale_rational_term(ctx, -log_scale, log_term);
    let leading_term = cas_ast::hold::wrap_hold(ctx, leading_term);
    let signed_log_term = cas_ast::hold::wrap_hold(ctx, signed_log_term);
    Some(ctx.add(Expr::Add(leading_term, signed_log_term)))
}

pub fn integrate_symbolic_is_arctan_scaled_variable_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let (fn_id, args) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return false,
    };

    if !matches!(
        ctx.builtin_of(fn_id),
        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
    ) {
        return false;
    }

    arctan_scaled_variable_antiderivative(ctx, args[0], var).is_some()
}

fn orient_arctan_terms_to_argument(
    ctx: &mut Context,
    expr: ExprId,
    target_arg: ExprId,
    var: &str,
) -> ExprId {
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(fn_id),
                    Some(BuiltinFn::Arctan | BuiltinFn::Atan)
                )
                && compare_expr(ctx, args[0], target_arg) != Ordering::Equal
                && affine_args_are_opposite(ctx, args[0], target_arg, var) =>
        {
            let oriented = ctx.call_builtin(BuiltinFn::Arctan, vec![target_arg]);
            ctx.add(Expr::Neg(oriented))
        }
        Expr::Add(left, right) => {
            let left = orient_arctan_terms_to_argument(ctx, left, target_arg, var);
            let right = orient_arctan_terms_to_argument(ctx, right, target_arg, var);
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left = orient_arctan_terms_to_argument(ctx, left, target_arg, var);
            let right = orient_arctan_terms_to_argument(ctx, right, target_arg, var);
            ctx.add(Expr::Sub(left, right))
        }
        Expr::Mul(left, right) => {
            let left = orient_arctan_terms_to_argument(ctx, left, target_arg, var);
            let right = orient_arctan_terms_to_argument(ctx, right, target_arg, var);
            ctx.add(Expr::Mul(left, right))
        }
        Expr::Div(left, right) => {
            let left = orient_arctan_terms_to_argument(ctx, left, target_arg, var);
            let right = orient_arctan_terms_to_argument(ctx, right, target_arg, var);
            ctx.add(Expr::Div(left, right))
        }
        Expr::Pow(base, exp) => {
            let base = orient_arctan_terms_to_argument(ctx, base, target_arg, var);
            let exp = orient_arctan_terms_to_argument(ctx, exp, target_arg, var);
            ctx.add(Expr::Pow(base, exp))
        }
        Expr::Neg(inner) => {
            let inner = orient_arctan_terms_to_argument(ctx, inner, target_arg, var);
            ctx.add(Expr::Neg(inner))
        }
        Expr::Function(fn_id, args) => {
            let args = args
                .into_iter()
                .map(|arg| orient_arctan_terms_to_argument(ctx, arg, target_arg, var))
                .collect();
            ctx.add(Expr::Function(fn_id, args))
        }
        Expr::Matrix { rows, cols, data } => {
            let data = data
                .into_iter()
                .map(|entry| orient_arctan_terms_to_argument(ctx, entry, target_arg, var))
                .collect();
            ctx.add(Expr::Matrix { rows, cols, data })
        }
        Expr::Hold(inner) => {
            let inner = orient_arctan_terms_to_argument(ctx, inner, target_arg, var);
            ctx.add(Expr::Hold(inner))
        }
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => expr,
    }
}

fn arctan_factor_matches_target(
    ctx: &Context,
    expr: ExprId,
    arctan: ExprId,
    target_arg: ExprId,
    var: &str,
) -> Option<bool> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if expr == arctan {
        return Some(true);
    }

    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(*fn_id),
                    Some(BuiltinFn::Arctan | BuiltinFn::Atan)
                ) =>
        {
            let Ok(arg_poly) = Polynomial::from_expr(ctx, args[0], var) else {
                return None;
            };
            let Ok(target_poly) = Polynomial::from_expr(ctx, target_arg, var) else {
                return None;
            };
            (arg_poly == target_poly).then_some(false)
        }
        _ => None,
    }
}

fn matching_arctan_term_scale(
    ctx: &Context,
    expr: ExprId,
    arctan: ExprId,
    target_arg: ExprId,
    var: &str,
) -> Option<(BigRational, bool)> {
    if let Some(exact_match) = arctan_factor_matches_target(ctx, expr, arctan, target_arg, var) {
        return Some((BigRational::one(), exact_match));
    }

    match ctx.get(expr) {
        Expr::Neg(inner) => matching_arctan_term_scale(ctx, *inner, arctan, target_arg, var)
            .map(|(scale, exact_match)| (-scale, exact_match)),
        Expr::Mul(left, right) => {
            if let Some(exact_match) =
                arctan_factor_matches_target(ctx, *right, arctan, target_arg, var)
            {
                return rational_constant_value(ctx, *left).map(|scale| (scale, exact_match));
            }
            if let Some(exact_match) =
                arctan_factor_matches_target(ctx, *left, arctan, target_arg, var)
            {
                return rational_constant_value(ctx, *right).map(|scale| (scale, exact_match));
            }
            None
        }
        Expr::Div(num, den) => {
            let denominator = rational_constant_value(ctx, *den)?;
            if denominator.is_zero() {
                return None;
            }
            matching_arctan_term_scale(ctx, *num, arctan, target_arg, var)
                .map(|(scale, exact_match)| (scale / denominator, exact_match))
        }
        _ => None,
    }
}

fn arctan_times_polynomial_with_rational_content_factored(
    ctx: &mut Context,
    arctan: ExprId,
    poly: &Polynomial,
) -> ExprId {
    let mut denominator_lcm = num_bigint::BigInt::one();
    for coeff in &poly.coeffs {
        denominator_lcm = denominator_lcm.lcm(coeff.denom());
    }

    if denominator_lcm.is_one() {
        let inner = poly.to_expr(ctx);
        return mul2_raw(ctx, inner, arctan);
    }

    let denominator = BigRational::from_integer(denominator_lcm);
    let scaled_poly = poly.mul(&Polynomial::new(
        vec![denominator.clone()],
        poly.var.clone(),
    ));
    let inner = scaled_poly.to_expr(ctx);
    let numerator = mul2_raw(ctx, inner, arctan);
    let denominator_expr = ctx.add(Expr::Number(denominator));
    ctx.add(Expr::Div(numerator, denominator_expr))
}

pub(super) fn arctan_polynomial_minus_remainder_with_rational_content_factored(
    ctx: &mut Context,
    arctan: ExprId,
    arctan_poly: &Polynomial,
    remainder_poly: &Polynomial,
) -> ExprId {
    if remainder_poly.is_zero() {
        return arctan_times_polynomial_with_rational_content_factored(ctx, arctan, arctan_poly);
    }

    let mut denominator_lcm = num_bigint::BigInt::one();
    for coeff in arctan_poly
        .coeffs
        .iter()
        .chain(remainder_poly.coeffs.iter())
    {
        denominator_lcm = denominator_lcm.lcm(coeff.denom());
    }

    if denominator_lcm.is_one() {
        let arctan_inner = arctan_poly.to_expr(ctx);
        let arctan_inner = cas_ast::hold::wrap_hold(ctx, arctan_inner);
        let arctan_term = mul2_raw(ctx, arctan_inner, arctan);
        let remainder = remainder_poly.to_expr(ctx);
        return ctx.add(Expr::Sub(arctan_term, remainder));
    }

    let denominator = BigRational::from_integer(denominator_lcm);
    let denominator_poly = Polynomial::new(vec![denominator.clone()], arctan_poly.var.clone());
    let scaled_arctan_poly = arctan_poly.mul(&denominator_poly);
    let scaled_remainder_poly = remainder_poly.mul(&denominator_poly);

    let arctan_inner = scaled_arctan_poly.to_expr(ctx);
    let arctan_inner = cas_ast::hold::wrap_hold(ctx, arctan_inner);
    let arctan_term = mul2_raw(ctx, arctan_inner, arctan);
    let remainder = scaled_remainder_poly.to_expr(ctx);
    let numerator = ctx.add(Expr::Sub(arctan_term, remainder));
    let denominator_expr = ctx.add(Expr::Number(denominator));
    ctx.add(Expr::Div(numerator, denominator_expr))
}

pub(super) fn arctan_polynomial_minus_expr_with_rational_content_factored(
    ctx: &mut Context,
    arctan: ExprId,
    arctan_poly: &Polynomial,
    remainder: ExprId,
) -> ExprId {
    let mut denominator_lcm = num_bigint::BigInt::one();
    for coeff in &arctan_poly.coeffs {
        denominator_lcm = denominator_lcm.lcm(coeff.denom());
    }

    let arctan_term = if denominator_lcm.is_one() {
        let arctan_inner = arctan_poly.to_expr(ctx);
        let arctan_inner = cas_ast::hold::wrap_hold(ctx, arctan_inner);
        mul2_raw(ctx, arctan_inner, arctan)
    } else {
        let denominator = BigRational::from_integer(denominator_lcm);
        let denominator_poly = Polynomial::new(vec![denominator.clone()], arctan_poly.var.clone());
        let scaled_arctan_poly = arctan_poly.mul(&denominator_poly);
        let arctan_inner = scaled_arctan_poly.to_expr(ctx);
        let arctan_inner = cas_ast::hold::wrap_hold(ctx, arctan_inner);
        let numerator = mul2_raw(ctx, arctan_inner, arctan);
        let denominator_expr = ctx.add(Expr::Number(denominator));
        ctx.add(Expr::Div(numerator, denominator_expr))
    };

    let arctan_term = cas_ast::hold::wrap_hold(ctx, arctan_term);
    let remainder = cas_ast::hold::wrap_hold(ctx, remainder);
    ctx.add(Expr::Sub(arctan_term, remainder))
}

pub(super) fn split_arctan_part_when_subtracting(
    ctx: &mut Context,
    expr: ExprId,
    arctan: ExprId,
    target_arg: ExprId,
    var: &str,
) -> Option<(ExprId, BigRational, bool)> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let view = AddView::from_expr(ctx, expr);
    for (index, (term, sign)) in view.terms.iter().copied().enumerate() {
        let Some((scale, exact_match)) =
            matching_arctan_term_scale(ctx, term, arctan, target_arg, var)
        else {
            continue;
        };

        let signed_scale = match sign {
            Sign::Pos => scale,
            Sign::Neg => -scale,
        };
        let adjustment = -signed_scale;
        let mut remainder_terms = Vec::new();
        for (candidate_index, (candidate, candidate_sign)) in view.terms.iter().copied().enumerate()
        {
            if candidate_index == index {
                continue;
            }
            let signed_candidate = match candidate_sign {
                Sign::Pos => candidate,
                Sign::Neg => ctx.add(Expr::Neg(candidate)),
            };
            remainder_terms.push(signed_candidate);
        }

        let remainder = build_balanced_add(ctx, &remainder_terms);
        return Some((remainder, adjustment, exact_match));
    }

    None
}

pub(super) fn polynomial_times_arctan_affine_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (arctan_index, factor) in factors.iter().enumerate() {
        let Expr::Function(fn_id, args) = ctx.get(*factor).clone() else {
            continue;
        };
        if args.len() != 1
            || !matches!(
                ctx.builtin_of(fn_id),
                Some(BuiltinFn::Arctan | BuiltinFn::Atan)
            )
        {
            continue;
        }

        let arg = args[0];
        let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
        if arg_poly.degree() != 1 {
            continue;
        }
        let arg_slope = arg_poly
            .coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if arg_slope.is_zero() {
            continue;
        }

        let cofactor_factors: Vec<ExprId> = factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != arctan_index).then_some(*factor))
            .collect();
        let cofactor = if cofactor_factors.is_empty() {
            ctx.num(1)
        } else {
            build_balanced_mul(ctx, &cofactor_factors)
        };
        let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
        if !(1..=MAX_ARCTAN_BY_PARTS_POLY_DEGREE).contains(&cofactor_poly.degree()) {
            continue;
        }

        let v_poly = polynomial_antiderivative(&cofactor_poly);
        if v_poly.is_zero() {
            continue;
        }

        let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arg]);
        let v_expr = v_poly.to_expr(ctx);
        let leading = mul2_raw(ctx, v_expr, arctan);

        let denominator = arg_poly
            .mul(&arg_poly)
            .add(&Polynomial::one(arg_poly.var.clone()));
        let numerator = v_poly.mul(&Polynomial::new(
            vec![arg_slope.clone()],
            v_poly.var.clone(),
        ));
        let numerator_expr = numerator.to_expr(ctx);
        let denominator_expr = denominator.to_expr(ctx);
        let rational_integral = positive_quadratic_linear_numerator_antiderivative(
            ctx,
            numerator_expr,
            denominator_expr,
            var,
        )?;
        let rational_integral = if arg_slope.is_negative() {
            orient_arctan_terms_to_argument(ctx, rational_integral, arg, var)
        } else {
            rational_integral
        };
        if let Some(compact) =
            compact_arctan_by_parts_subtraction(ctx, &v_poly, arctan, arg, var, rational_integral)
        {
            return Some(compact);
        }

        let leading = cas_ast::hold::wrap_hold(ctx, leading);
        let rational_integral = cas_ast::hold::wrap_hold(ctx, rational_integral);
        return Some(ctx.add(Expr::Sub(leading, rational_integral)));
    }

    None
}

pub(super) fn arctan_reciprocal_affine_variable_antiderivative(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (denominator, coeff, has_zero_offset) =
        reciprocal_affine_variable_denominator(ctx, arg, var)?;

    let arctan_arg = ctx.call_builtin(BuiltinFn::Arctan, vec![arg]);
    let leading_linear = if has_zero_offset {
        ctx.var(var)
    } else {
        let scale = BigRational::one() / coeff.clone();
        if scale.is_one() {
            denominator
        } else {
            let scale = ctx.add(Expr::Number(scale));
            mul2_raw(ctx, scale, denominator)
        }
    };
    let leading_term = mul2_raw(ctx, leading_linear, arctan_arg);

    let two = ctx.num(2);
    let denominator_sq = ctx.add(Expr::Pow(denominator, two));
    let one = ctx.num(1);
    let log_arg = ctx.add(Expr::Add(denominator_sq, one));
    let log_term = ctx.call_builtin(BuiltinFn::Ln, vec![log_arg]);

    let two_coeff = coeff * BigRational::from_integer(2.into());
    let log_scale = BigRational::one() / two_coeff;
    let scaled_log_term = if log_scale.is_one() {
        log_term
    } else {
        let scale = ctx.add(Expr::Number(log_scale));
        mul2_raw(ctx, scale, log_term)
    };

    Some(ctx.add(Expr::Add(leading_term, scaled_log_term)))
}

pub fn integrate_symbolic_is_arctan_reciprocal_affine_variable_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let (fn_id, args) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return false,
    };

    match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Arctan | BuiltinFn::Atan) => {
            reciprocal_affine_variable_denominator(ctx, args[0], var)
                .is_some_and(|(_, _, has_zero_offset)| !has_zero_offset)
        }
        Some(BuiltinFn::Arccot | BuiltinFn::Acot) => {
            let Some((coeff, offset)) = get_linear_coeffs(ctx, args[0], var) else {
                return false;
            };
            if is_number(ctx, offset, 0) {
                return false;
            }
            rational_constant_value(ctx, coeff).is_some_and(|coeff| !coeff.is_zero())
        }
        _ => false,
    }
}

fn bounded_inverse_trig_scaled_sqrt_term(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
    coeff: &BigRational,
) -> Option<(ExprId, ExprId)> {
    if coeff.is_negative() {
        let radicand = unit_minus_square(ctx, arg);
        let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
        let abs_reciprocal = -BigRational::one() / coeff.clone();
        let sqrt_term = if abs_reciprocal.is_one() {
            sqrt
        } else {
            let scale = ctx.add(Expr::Number(abs_reciprocal));
            mul2_raw(ctx, scale, sqrt)
        };
        return Some((radicand, sqrt_term));
    }

    let radicand = scaled_unit_minus_square_linear_radicand(ctx, arg, var, coeff)?;
    let sqrt_term = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    Some((radicand, sqrt_term))
}

pub(super) fn monomial_times_bounded_inverse_trig_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    // c*x * arcsin(x) / arccos(x): the classic by-parts pair
    // (2x^2 - 1)/4 * inv(x) +/- x*sqrt(1 - x^2)/4 (sign by family).
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }
    for (inverse_index, factor) in factors.iter().enumerate() {
        let Expr::Function(fn_id, args) = ctx.get(*factor).clone() else {
            continue;
        };
        if args.len() != 1 {
            continue;
        }
        let Some(builtin) = ctx.builtin_of(fn_id) else {
            continue;
        };
        if !matches!(
            builtin,
            BuiltinFn::Arcsin | BuiltinFn::Asin | BuiltinFn::Arccos | BuiltinFn::Acos
        ) {
            continue;
        }
        let (slope_expr, offset) = get_linear_coeffs(ctx, args[0], var)?;
        let slope = rational_constant_value(ctx, slope_expr)?;
        if slope.is_zero() {
            return None;
        }
        let zero_offset = is_number(ctx, offset, 0);
        if !zero_offset && rational_constant_value(ctx, offset).is_none() {
            return None;
        }
        let cofactor_factors: Vec<ExprId> = factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != inverse_index).then_some(*factor))
            .collect();
        if cofactor_factors.is_empty() {
            return None;
        }
        let cofactor = build_balanced_mul(ctx, &cofactor_factors);
        let (scale, power) = scaled_var_power_term(ctx, cofactor, var)?;
        if !power.denom().is_one() || !power.is_positive() {
            return None;
        }
        let n = usize::try_from(i64::try_from(&power.to_integer()).ok()?).ok()?;
        if n > 5 {
            return None;
        }
        if n != 1 || !slope.is_one() || !zero_offset {
            // General by parts: x^(n+1)/(n+1) * inv(u) -/+ k/(n+1) *
            // integral of x^(n+1)/sqrt(1 - u^2) with u = kx + b, the
            // tail delegated to the radical reduction families (pure
            // radicands) or the Hermite split (shifted radicands).
            let inverse = *factor;
            let var_expr = ctx.var(var);
            let head_exponent = ctx.num((n + 1) as i64);
            let head_monomial = ctx.add(Expr::Pow(var_expr, head_exponent));
            let head_raw = mul2_raw(ctx, head_monomial, inverse);
            let over_n_plus_one =
                BigRational::one() / BigRational::from_integer(((n + 1) as i64).into());
            let head = scale_rational_term(ctx, over_n_plus_one.clone(), head_raw);

            let one = ctx.num(1);
            let two = ctx.num(2);
            let arg_squared = ctx.add(Expr::Pow(args[0], two));
            let radicand = ctx.add(Expr::Sub(one, arg_squared));
            let tail = if zero_offset {
                let slope_square = &slope * &slope;
                monomial_over_sqrt_reduction(
                    ctx,
                    n + 1,
                    &BigRational::one(),
                    &slope_square,
                    radicand,
                    var,
                )?
            } else {
                let tail_exponent = ctx.num((n + 1) as i64);
                let tail_monomial = ctx.add(Expr::Pow(var_expr, tail_exponent));
                let sqrt_term = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
                let tail_integrand = ctx.add(Expr::Div(tail_monomial, sqrt_term));
                integrate_symbolic_expr(ctx, tail_integrand, var)?
            };
            let tail_signed = match builtin {
                BuiltinFn::Arcsin | BuiltinFn::Asin => -(&slope * &over_n_plus_one),
                _ => &slope * &over_n_plus_one,
            };
            let tail_term = scale_rational_term(ctx, tail_signed, tail);
            let primitive = ctx.add(Expr::Add(head, tail_term));
            return Some(scale_rational_term(ctx, scale, primitive));
        }

        let var_expr = ctx.var(var);
        let two = ctx.num(2);
        let one = ctx.num(1);
        let x_squared = ctx.add(Expr::Pow(var_expr, two));
        let doubled = scale_rational_term(ctx, BigRational::from_integer(2.into()), x_squared);
        let quad = ctx.add(Expr::Sub(doubled, one));
        let inverse = *factor;
        let inverse_term_raw = mul2_raw(ctx, quad, inverse);
        let radicand = unit_minus_square(ctx, var_expr);
        let sqrt_term = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
        let sqrt_product = mul2_raw(ctx, var_expr, sqrt_term);
        let quarter = BigRational::new(1.into(), 4.into());
        let inverse_term = scale_rational_term(ctx, quarter.clone(), inverse_term_raw);
        let sqrt_signed = match builtin {
            BuiltinFn::Arcsin | BuiltinFn::Asin => quarter,
            _ => -quarter,
        };
        let sqrt_scaled = scale_rational_term(ctx, sqrt_signed, sqrt_product);
        let primitive = ctx.add(Expr::Add(inverse_term, sqrt_scaled));
        return Some(scale_rational_term(ctx, scale, primitive));
    }
    None
}

pub(super) fn bounded_inverse_trig_linear_antiderivative(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (coeff_expr, offset) = get_linear_coeffs(ctx, arg, var)?;
    let coeff = rational_constant_value(ctx, coeff_expr)?;
    if coeff.is_zero() {
        return None;
    }

    let zero_offset = is_number(ctx, offset, 0);
    let inverse = ctx.call_builtin(builtin, vec![arg]);
    if !zero_offset && coeff.is_negative() {
        let product = mul2_raw(ctx, arg, inverse);
        let radicand = scaled_unit_minus_square_linear_radicand(ctx, arg, var, &coeff)?;
        let coeff_square = &coeff * &coeff;
        let factored_radicand = if coeff_square.is_one() {
            radicand
        } else {
            let coeff_square_expr = ctx.add(Expr::Number(coeff_square));
            mul2_raw(ctx, coeff_square_expr, radicand)
        };
        let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
        let sqrt_term = ctx.add(Expr::Pow(factored_radicand, half));

        if matches!(builtin, BuiltinFn::Arccos | BuiltinFn::Acos) {
            let sqrt_term = cas_ast::hold::wrap_hold(ctx, sqrt_term);
            let product = cas_ast::hold::wrap_hold(ctx, product);
            let primitive = ctx.add(Expr::Sub(sqrt_term, product));
            let scale = -BigRational::one() / coeff;
            if scale.is_one() {
                return Some(primitive);
            }
            let scaled_sqrt = scale_rational_term(ctx, scale.clone(), sqrt_term);
            let scaled_product = scale_rational_term(ctx, -scale, product);
            return Some(ctx.add(Expr::Add(scaled_sqrt, scaled_product)));
        }

        if !matches!(builtin, BuiltinFn::Arcsin | BuiltinFn::Asin) {
            return None;
        }
        let scale = BigRational::one() / coeff;
        let scaled_product = scale_rational_term(ctx, scale.clone(), product);
        let scaled_sqrt = scale_rational_term(ctx, scale, sqrt_term);
        return Some(ctx.add(Expr::Add(scaled_product, scaled_sqrt)));
    }

    let (product, sqrt_term) = if zero_offset {
        let leading_linear = ctx.var(var);
        let product = mul2_raw(ctx, leading_linear, inverse);
        let (_, sqrt_term) = bounded_inverse_trig_scaled_sqrt_term(ctx, arg, var, &coeff)?;
        (product, sqrt_term)
    } else {
        if !coeff.is_positive() {
            return None;
        }
        let product = mul2_raw(ctx, arg, inverse);
        let radicand = unit_minus_square(ctx, arg);
        let sqrt_term = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
        (product, sqrt_term)
    };

    let primitive = match builtin {
        BuiltinFn::Arcsin | BuiltinFn::Asin if coeff.is_positive() => {
            ctx.add(Expr::Add(product, sqrt_term))
        }
        BuiltinFn::Arcsin | BuiltinFn::Asin => ctx.add(Expr::Sub(product, sqrt_term)),
        BuiltinFn::Arccos | BuiltinFn::Acos if coeff.is_positive() => {
            ctx.add(Expr::Sub(product, sqrt_term))
        }
        BuiltinFn::Arccos | BuiltinFn::Acos => ctx.add(Expr::Add(product, sqrt_term)),
        _ => return None,
    };

    if zero_offset || coeff.is_one() {
        Some(primitive)
    } else {
        let scale = BigRational::one() / coeff;
        let scaled_product = scale_rational_term(ctx, scale.clone(), product);
        let sqrt_scale = match builtin {
            BuiltinFn::Arcsin | BuiltinFn::Asin => scale,
            BuiltinFn::Arccos | BuiltinFn::Acos => -scale,
            _ => return None,
        };
        let scaled_sqrt = scale_rational_term(ctx, sqrt_scale, sqrt_term);
        Some(ctx.add(Expr::Add(scaled_product, scaled_sqrt)))
    }
}

pub(super) fn bounded_inverse_trig_linear_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return None,
    };
    let builtin = ctx.builtin_of(fn_id)?;
    if !matches!(
        builtin,
        BuiltinFn::Arcsin | BuiltinFn::Asin | BuiltinFn::Arccos | BuiltinFn::Acos
    ) {
        return None;
    }
    let (coeff_expr, offset) = get_linear_coeffs(ctx, args[0], var)?;
    let coeff = rational_constant_value(ctx, coeff_expr)?;
    if coeff.is_zero() {
        return None;
    }

    if !is_number(ctx, offset, 0) {
        bounded_inverse_trig_linear_antiderivative(ctx, builtin, args[0], var)?;
        if coeff.is_negative() {
            return scaled_unit_minus_square_linear_radicand(ctx, args[0], var, &coeff);
        }
        return Some(unit_minus_square(ctx, args[0]));
    }

    let (radicand, _) = bounded_inverse_trig_scaled_sqrt_term(ctx, args[0], var, &coeff)?;
    Some(radicand)
}

pub fn integrate_symbolic_is_bounded_inverse_trig_variable_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    bounded_inverse_trig_linear_radicand(ctx, expr, var).is_some()
}

pub fn integrate_symbolic_is_polynomial_times_arctan_affine_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    polynomial_times_arctan_affine_antiderivative(ctx, expr, var).is_some()
}

pub(super) fn arctan_surd_offset_antiderivative(
    ctx: &mut Context,
    numerator: &Polynomial,
    arg_poly: &Polynomial,
    offset_square: &BigRational,
) -> Option<ExprId> {
    let derivative = arg_poly.derivative();
    let mut scale = constant_polynomial_ratio(numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let mut arg_poly = arg_poly.clone();
    let mut offset_square = offset_square.clone();
    if let Some((normalized_arg, normalized_offset_square, denominator_scale)) =
        reduce_surd_offset_by_square_denominator(&arg_poly, &offset_square)
    {
        arg_poly = normalized_arg;
        offset_square = normalized_offset_square;
        scale *= denominator_scale;
    }
    if let Some((reduced_arg, reduced_offset_square, common_factor)) =
        reduce_surd_offset_by_common_square_factor(&arg_poly, &offset_square)
    {
        arg_poly = reduced_arg;
        offset_square = reduced_offset_square;
        scale /= common_factor;
    }

    let offset_expr = positive_rational_sqrt_expr(ctx, &offset_square)?;
    let arg = arg_poly.to_expr(ctx);
    let arctan_arg = ctx.add(Expr::Div(arg, offset_expr));
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arctan_arg]);
    let numerator = if scale.is_one() {
        arctan
    } else {
        let scale_num = ctx.add(Expr::Number(scale));
        mul2_raw(ctx, scale_num, arctan)
    };
    Some(ctx.add(Expr::Div(numerator, offset_expr)))
}

pub(super) fn arctan_scaled_quadratic_antiderivative(
    ctx: &mut Context,
    numerator: &Polynomial,
    denominator: &Polynomial,
) -> Option<ExprId> {
    if denominator.degree() != 2 {
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
    let discriminant = four.clone() * a.clone() * c - b.clone() * b.clone();
    if discriminant <= BigRational::zero() {
        return None;
    }

    let arg_poly = Polynomial::new(vec![b, two * a.clone()], denominator.var.clone());
    let Some(offset) = exact_rational_sqrt(&discriminant) else {
        return arctan_scaled_quadratic_surd_antiderivative(
            ctx,
            numerator,
            &arg_poly,
            &discriminant,
        );
    };
    if offset.is_zero() {
        return None;
    }

    let derivative = arg_poly.derivative();
    let denominator_scale = offset.clone() / (four * a);
    let scaled_derivative = Polynomial::new(
        derivative
            .coeffs
            .iter()
            .map(|coeff| coeff.clone() * denominator_scale.clone())
            .collect(),
        denominator.var.clone(),
    );
    let scale = constant_polynomial_ratio(numerator, &scaled_derivative)?;
    if scale.is_zero() {
        return None;
    }

    let arg_over_offset = Polynomial::new(
        arg_poly
            .coeffs
            .iter()
            .map(|coeff| coeff.clone() / offset.clone())
            .collect(),
        denominator.var.clone(),
    )
    .to_expr(ctx);
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arg_over_offset]);
    if scale.is_one() {
        return Some(arctan);
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, arctan))
}

fn arctan_scaled_quadratic_surd_antiderivative(
    ctx: &mut Context,
    numerator: &Polynomial,
    arg_poly: &Polynomial,
    discriminant: &BigRational,
) -> Option<ExprId> {
    let numerator_constant = constant_polynomial_value(numerator)?;
    if numerator_constant.is_zero() {
        return None;
    }

    let mut scale = BigRational::from_integer(2.into()) * numerator_constant;
    let mut arg_poly = arg_poly.clone();
    let mut offset_square = discriminant.clone();
    if let Some((normalized_arg, normalized_offset_square, denominator_scale)) =
        reduce_surd_offset_by_square_denominator(&arg_poly, &offset_square)
    {
        arg_poly = normalized_arg;
        offset_square = normalized_offset_square;
        scale *= denominator_scale;
    }
    if let Some((reduced_arg, reduced_offset_square, common_factor)) =
        reduce_surd_offset_by_common_square_factor(&arg_poly, &offset_square)
    {
        arg_poly = reduced_arg;
        offset_square = reduced_offset_square;
        scale /= common_factor;
    }

    let offset_expr = positive_rational_sqrt_expr(ctx, &offset_square)?;
    let arg = arg_poly.to_expr(ctx);
    let arctan_arg = ctx.add(Expr::Div(arg, offset_expr));
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arctan_arg]);

    let numerator = if scale.is_one() {
        arctan
    } else {
        let scale_num = ctx.add(Expr::Number(scale));
        mul2_raw(ctx, scale_num, arctan)
    };
    Some(ctx.add(Expr::Div(numerator, offset_expr)))
}

fn shifted_sqrt_arcsin_inverse_product_cofactor(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);

    if let Some(radicand) = reciprocal_sqrt_like_radicand(ctx, expr) {
        if is_var_times_sqrt_var_minus_var_radicand(ctx, radicand, var) {
            return Some(ctx.num(1));
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
    if radicands.len() != 2 {
        return None;
    }

    let combined_radicand = build_balanced_mul(ctx, &radicands);
    if !is_var_times_sqrt_var_minus_var_radicand(ctx, combined_radicand, var) {
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
    (!contains_named_var(ctx, cofactor, var)).then_some(cofactor)
}

pub(super) fn shifted_sqrt_arcsin_inverse_product_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let cofactor = shifted_sqrt_arcsin_inverse_product_cofactor(ctx, expr, var)?;
    let var_expr = ctx.var(var);
    let sqrt_var = ctx.call_builtin(BuiltinFn::Sqrt, vec![var_expr]);
    let two = ctx.num(2);
    let one = ctx.num(1);
    let shifted_sqrt = mul2_raw(ctx, two, sqrt_var);
    let arg = ctx.add(Expr::Sub(shifted_sqrt, one));
    let arcsin = ctx.call_builtin(BuiltinFn::Arcsin, vec![arg]);
    if let Some(scale) = rational_constant_value(ctx, cofactor) {
        return Some(scale_rational_term(
            ctx,
            scale * BigRational::from_integer(2.into()),
            arcsin,
        ));
    }

    let two = ctx.num(2);
    let doubled_cofactor = mul2_raw(ctx, two, cofactor);
    Some(mul2_raw(ctx, doubled_cofactor, arcsin))
}

pub(super) fn shifted_sqrt_arcsin_inverse_product_positive_conditions(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Vec<ExprId> {
    if shifted_sqrt_arcsin_inverse_product_cofactor(ctx, expr, var).is_none() {
        return vec![];
    }

    let var_expr = ctx.var(var);
    let sqrt_var = ctx.call_builtin(BuiltinFn::Sqrt, vec![var_expr]);
    let var_expr = ctx.var(var);
    vec![var_expr, ctx.add(Expr::Sub(sqrt_var, var_expr))]
}

pub(super) fn strip_variable_free_factors_from_arcsin_product_cofactor(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> ExprId {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    if !contains_named_var(ctx, expr, var) {
        return ctx.num(1);
    }

    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            strip_variable_free_factors_from_arcsin_product_cofactor(ctx, inner, var)
        }
        Expr::Mul(left, right) => {
            let left_depends = contains_named_var(ctx, left, var);
            let right_depends = contains_named_var(ctx, right, var);
            match (left_depends, right_depends) {
                (false, false) => ctx.num(1),
                (false, true) => {
                    strip_variable_free_factors_from_arcsin_product_cofactor(ctx, right, var)
                }
                (true, false) => {
                    strip_variable_free_factors_from_arcsin_product_cofactor(ctx, left, var)
                }
                (true, true) => {
                    let left =
                        strip_variable_free_factors_from_arcsin_product_cofactor(ctx, left, var);
                    let right =
                        strip_variable_free_factors_from_arcsin_product_cofactor(ctx, right, var);
                    mul2_raw(ctx, left, right)
                }
            }
        }
        Expr::Div(num, den) => {
            let num_depends = contains_named_var(ctx, num, var);
            let den_depends = contains_named_var(ctx, den, var);
            match (num_depends, den_depends) {
                (false, false) => ctx.num(1),
                (false, true) => {
                    let one = ctx.num(1);
                    let den =
                        strip_variable_free_factors_from_arcsin_product_cofactor(ctx, den, var);
                    ctx.add(Expr::Div(one, den))
                }
                (true, false) => {
                    strip_variable_free_factors_from_arcsin_product_cofactor(ctx, num, var)
                }
                (true, true) => {
                    let num =
                        strip_variable_free_factors_from_arcsin_product_cofactor(ctx, num, var);
                    let den =
                        strip_variable_free_factors_from_arcsin_product_cofactor(ctx, den, var);
                    ctx.add(Expr::Div(num, den))
                }
            }
        }
        _ => expr,
    }
}

pub fn integrate_symbolic_is_arcsin_inverse_sqrt_product_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    arcsin_inverse_sqrt_product_substitution_radicand(ctx, expr, var).is_some()
        || shifted_sqrt_arcsin_inverse_product_cofactor(ctx, expr, var).is_some()
}

pub(super) fn arctan_sqrt_affine_derivative_required_positive_radicand_from_mut_context(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    arctan_sqrt_affine_derivative_required_positive_radicand(ctx, expr, var)
}

pub(super) fn table_reused_arctan_kernel_integration_candidate(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> AlgorithmicIntegrationCandidate {
    let var_expr = ctx.var(var);
    let antiderivative = ctx.call_builtin(BuiltinFn::Arctan, vec![var_expr]);
    let mut candidate =
        AlgorithmicIntegrationCandidate::unverified_table_reused(expr, var, antiderivative);
    verify_antiderivative_by_differentiation(ctx, &mut candidate);
    candidate
}
