//! `symbolic_integration_support`: familia `support`.
//!
//! Ver la cabecera de `symbolic_integration_support.rs` para el contexto.

use super::*;

pub(super) fn is_number(ctx: &Context, expr: ExprId, value: i64) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if *n == BigRational::from_integer(value.into()))
}

pub(super) fn scale_rational_term(ctx: &mut Context, scale: BigRational, term: ExprId) -> ExprId {
    if let Expr::Neg(inner) = ctx.get(term).clone() {
        return scale_rational_term(ctx, -scale, inner);
    }

    if scale.is_one() {
        term
    } else if scale == BigRational::from_integer((-1).into()) {
        ctx.add(Expr::Neg(term))
    } else {
        let scale = ctx.add(Expr::Number(scale));
        mul2_raw(ctx, scale, term)
    }
}

pub(super) fn sqrt_like_radicand(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Sqrt) =>
        {
            Some(args[0])
        }
        Expr::Pow(base, exp) if is_positive_half(ctx, *exp) => Some(*base),
        _ => None,
    }
}

pub(super) fn reciprocal_sqrt_like_radicand(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) if is_negative_half(ctx, *exp) => Some(*base),
        _ => None,
    }
}

pub(super) fn scaled_var_power_term(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(BigRational, BigRational)> {
    let factors = mul_leaves(ctx, expr);
    let mut scale = BigRational::one();
    let mut power = None;

    for factor in factors {
        if let Some(factor_power) = var_power(ctx, factor, var) {
            if power.is_some() {
                return None;
            }
            power = Some(factor_power);
        } else {
            scale *= rational_constant_value(ctx, factor)?;
        }
    }

    Some((scale, power?))
}

pub(super) fn unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(builtin) =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

pub(super) fn negate_scalar_expr(ctx: &mut Context, expr: ExprId) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Number(value) => ctx.add(Expr::Number(-value)),
        Expr::Neg(inner) => inner,
        _ => ctx.add(Expr::Neg(expr)),
    }
}

pub(super) fn scale_expr_reciprocal_integration_result(
    ctx: &mut Context,
    scale: ExprId,
    expr: ExprId,
) -> ExprId {
    if let Expr::Number(value) = ctx.get(scale).clone() {
        return scale_reciprocal_integration_result(ctx, value, expr);
    }
    if is_number(ctx, scale, 1) {
        return expr;
    }

    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let negative_scale = negate_scalar_expr(ctx, scale);
            scale_expr_reciprocal_integration_result(ctx, negative_scale, inner)
        }
        Expr::Div(num, den) => {
            let scaled_num = if is_number(ctx, num, 1) {
                scale
            } else {
                mul2_raw(ctx, scale, num)
            };
            ctx.add(Expr::Div(scaled_num, den))
        }
        _ => mul2_raw(ctx, scale, expr),
    }
}

pub(super) fn scale_reciprocal_integration_result_preserving_presentation(
    ctx: &mut Context,
    scale: BigRational,
    expr: ExprId,
) -> ExprId {
    if scale.is_one() {
        return expr;
    }
    if scale == -BigRational::one() {
        return negate_integration_result(ctx, expr);
    }
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        return scale_reciprocal_integration_result_preserving_presentation(ctx, -scale, inner);
    }

    let numerator_scale = BigRational::from_integer(scale.numer().clone());
    let denominator_scale = BigRational::from_integer(scale.denom().clone());
    let numerator = if numerator_scale.is_one() {
        expr
    } else if numerator_scale == BigRational::from_integer((-1).into()) {
        negate_integration_result(ctx, expr)
    } else {
        let numerator_scale = ctx.add(Expr::Number(numerator_scale));
        multiply_rational_factor_if_possible(ctx, numerator_scale, expr)
            .unwrap_or_else(|| mul2_raw(ctx, numerator_scale, expr))
    };

    let scaled = if denominator_scale.is_one() {
        numerator
    } else {
        let denominator_scale = ctx.add(Expr::Number(denominator_scale));
        ctx.add(Expr::Div(numerator, denominator_scale))
    };
    cas_ast::hold::wrap_hold(ctx, scaled)
}

pub(super) fn nonzero_linear_arg_and_slope(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let (slope, _) = get_linear_coeffs(ctx, arg, var)?;
    if contains_named_var(ctx, slope, var) || is_number(ctx, slope, 0) {
        return None;
    }

    Some((arg, slope))
}

pub(super) fn rational_constant_value(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(n) => Some(n.clone()),
        Expr::Add(l, r) => {
            let left = rational_constant_value(ctx, *l)?;
            let right = rational_constant_value(ctx, *r)?;
            Some(left + right)
        }
        Expr::Sub(l, r) => {
            let left = rational_constant_value(ctx, *l)?;
            let right = rational_constant_value(ctx, *r)?;
            Some(left - right)
        }
        Expr::Mul(l, r) => {
            let left = rational_constant_value(ctx, *l)?;
            let right = rational_constant_value(ctx, *r)?;
            Some(left * right)
        }
        Expr::Div(num, den) => {
            let numerator = rational_constant_value(ctx, *num)?;
            let denominator = rational_constant_value(ctx, *den)?;
            if denominator.is_zero() {
                None
            } else {
                Some(numerator / denominator)
            }
        }
        Expr::Neg(inner) => rational_constant_value(ctx, *inner).map(|value| -value),
        _ => None,
    }
}

pub(super) fn scale_factor(ctx: &mut Context, scale: BigRational, expr: ExprId) -> ExprId {
    if scale.is_one() {
        return expr;
    }
    let scale_expr = ctx.add(Expr::Number(scale));
    mul2_raw(ctx, scale_expr, expr)
}

pub(super) fn signed_linear_function_factor_parts<F>(
    ctx: &mut Context,
    factor: ExprId,
    var: &str,
    detector: F,
) -> Option<SignedLinearFunctionFactorParts>
where
    F: Fn(&Context, ExprId) -> Option<(BuiltinFn, ExprId, Sign, ExprId)>,
{
    let (builtin, arg, sign, source_factor) = detector(ctx, factor)?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    if arg_poly.degree() != 1 {
        return None;
    }

    let arg_slope = arg_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if arg_slope.is_zero() {
        return None;
    }

    Some(SignedLinearFunctionFactorParts {
        builtin,
        arg,
        sign,
        factor: source_factor,
        arg_slope,
    })
}

pub(super) fn exact_rational_sqrt(value: &BigRational) -> Option<BigRational> {
    if value < &BigRational::zero() {
        return None;
    }

    let sqrt_num = value.numer().sqrt();
    let sqrt_den = value.denom().sqrt();
    if &sqrt_num * &sqrt_num == value.numer().clone()
        && &sqrt_den * &sqrt_den == value.denom().clone()
    {
        Some(BigRational::new(sqrt_num, sqrt_den))
    } else {
        None
    }
}

pub(super) fn positive_rational_sqrt_expr(
    ctx: &mut Context,
    value: &BigRational,
) -> Option<ExprId> {
    if value <= &BigRational::zero() {
        return None;
    }

    if let Some(root) = exact_rational_sqrt(value) {
        return Some(ctx.add(Expr::Number(root)));
    }

    let radicand = ctx.add(Expr::Number(value.clone()));
    Some(ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]))
}

pub(super) fn exact_positive_constant_minus_polynomial_square(
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
    let leading_root = exact_rational_sqrt(&(-leading))?;
    if leading_root.is_zero() {
        return None;
    }

    let mut root_coeffs = vec![BigRational::zero(); root_degree + 1];
    root_coeffs[root_degree] = leading_root.clone();
    let two = BigRational::from_integer(2.into());

    for k in (0..root_degree).rev() {
        let target_degree = root_degree + k;
        let target = -poly
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
        if left != -right {
            return None;
        }
    }

    let constant = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero)
        + square
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

pub(super) fn scale_polynomial(poly: &Polynomial, scale: BigRational) -> Polynomial {
    poly.mul(&Polynomial::new(vec![scale], poly.var.clone()))
}

pub(super) fn signed_mul_leaves(ctx: &Context, expr: ExprId) -> (Sign, Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Neg(inner) => (Sign::Neg, mul_leaves(ctx, *inner).into_iter().collect()),
        _ => (Sign::Pos, mul_leaves(ctx, expr).into_iter().collect()),
    }
}

/// Integrate `expr` with respect to `var` using a small set of symbolic rules.
pub fn integrate_symbolic_expr(ctx: &mut Context, expr: ExprId, var: &str) -> Option<ExprId> {
    // cbrt is kept as Function(Cbrt) globally, but for integration it is x^(1/3):
    // lower the cube roots and recurse through the ordinary power rule.
    if let Some(lowered) = lower_cbrt_for_integration(ctx, expr) {
        return integrate_symbolic_expr(ctx, lowered, var);
    }
    // Extract variant info in one borrow, then process with owned ExprId values.
    enum IntKind {
        Add(ExprId, ExprId),
        Sub(ExprId, ExprId),
        Neg(ExprId),
        Mul(ExprId, ExprId),
        Pow(ExprId, ExprId),
        Variable(usize),
        Div(ExprId, ExprId),
        Function(usize, Vec<ExprId>),
        Other,
    }
    let kind = match ctx.get(expr) {
        Expr::Add(l, r) => IntKind::Add(*l, *r),
        Expr::Sub(l, r) => IntKind::Sub(*l, *r),
        Expr::Neg(inner) => IntKind::Neg(*inner),
        Expr::Mul(l, r) => IntKind::Mul(*l, *r),
        Expr::Pow(b, e) => IntKind::Pow(*b, *e),
        Expr::Variable(s) => IntKind::Variable(*s),
        Expr::Div(n, d) => IntKind::Div(*n, *d),
        Expr::Function(f, args) => IntKind::Function(*f, args.clone()),
        _ => IntKind::Other,
    };

    if real_domain_is_empty_or_nonfinite_for_integration(ctx, expr) {
        return Some(ctx.add(Expr::Constant(Constant::Undefined)));
    }

    if matches!(kind, IntKind::Add(_, _) | IntKind::Sub(_, _)) {
        if let Some(integral) =
            additive_common_trig_polynomial_substitution_antiderivative(ctx, expr, var)
        {
            return Some(integral);
        }

        if let Some(integral) = polynomial_log_product_substitution_antiderivative(ctx, expr, var) {
            return Some(integral);
        }

        if let Some(integral) =
            polynomial_log_power_product_substitution_antiderivative(ctx, expr, var)
        {
            return Some(integral);
        }

        if let Some(integral) =
            additive_quadratic_times_affine_ln_by_parts_antiderivative(ctx, expr, var)
        {
            return Some(integral);
        }

        if let Some(integral) =
            additive_positive_quadratic_ln_by_parts_antiderivative(ctx, expr, var)
        {
            return Some(integral);
        }

        if let Some(integral) =
            additive_linear_times_affine_ln_by_parts_antiderivative(ctx, expr, var)
        {
            return Some(integral);
        }

        if let Some(integral) = additive_polynomial_times_trig_linear_antiderivative(ctx, expr, var)
        {
            return Some(integral);
        }

        if let Some(integral) =
            additive_polynomial_times_hyperbolic_linear_antiderivative(ctx, expr, var)
        {
            return Some(integral);
        }
    }

    if let IntKind::Add(l, r) = kind {
        let int_l = integrate_symbolic_expr(ctx, l, var)?;
        let int_r = integrate_symbolic_expr(ctx, r, var)?;
        return Some(ctx.add(Expr::Add(int_l, int_r)));
    }

    if let IntKind::Sub(l, r) = kind {
        let int_l = integrate_symbolic_expr(ctx, l, var)?;
        let int_r = integrate_symbolic_expr(ctx, r, var)?;
        return Some(ctx.add(Expr::Sub(int_l, int_r)));
    }

    if let IntKind::Neg(inner) = kind {
        let inner_integral = integrate_symbolic_expr(ctx, inner, var)?;
        return Some(negate_integration_result(ctx, inner_integral));
    }

    if let Some(integral) = inverse_hyperbolic_sqrt_reciprocal_antiderivative(
        ctx,
        expr,
        var,
        InverseHyperbolicSqrtReciprocalKind::Asinh,
    ) {
        return Some(integral);
    }

    if let Some(integral) = inverse_hyperbolic_sqrt_reciprocal_antiderivative(
        ctx,
        expr,
        var,
        InverseHyperbolicSqrtReciprocalKind::Atanh,
    ) {
        return Some(integral);
    }

    if let Some(integral) = polynomial_substitution_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = trig_sine_cosine_same_affine_product_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = trig_power_times_derivative_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = trig_fourth_power_affine_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = trig_sixth_power_affine_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = trig_eighth_power_affine_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = trig_ratio_power_reciprocal_square_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = hyperbolic_power_times_derivative_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = hyperbolic_square_product_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = sqrt_derivative_substitution_product_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = affine_sqrt_product_derivative_product_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = sqrt_product_derivative_substitution_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = sqrt_trig_reciprocal_derivative_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = sqrt_trig_log_derivative_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = sqrt_reciprocal_trig_log_derivative_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = sqrt_hyperbolic_log_derivative_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = sqrt_hyperbolic_reciprocal_square_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = sqrt_hyperbolic_reciprocal_derivative_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = polynomial_power_substitution_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = polynomial_log_product_substitution_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = hyperbolic_tanh_log_cosh_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = monomial_times_ln_var_by_parts_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = quadratic_times_affine_ln_by_parts_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = positive_quadratic_ln_by_parts_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) =
        low_degree_times_positive_quadratic_ln_by_parts_antiderivative(ctx, expr, var)
    {
        return Some(integral);
    }

    if let Some(integral) = linear_times_affine_ln_by_parts_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = polynomial_log_power_product_substitution_antiderivative(ctx, expr, var)
    {
        return Some(integral);
    }

    if let Some(integral) = polynomial_times_arctan_affine_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = monomial_times_bounded_inverse_trig_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = monomial_over_sqrt_negative_quadratic_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = linear_over_sqrt_shifted_quadratic_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = polynomial_over_sqrt_quadratic_hermite_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = radical_numerator_polynomial_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = polynomial_times_exp_linear_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = polynomial_times_constant_base_power_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = exp_trig_same_linear_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = hyperbolic_transcendental_product_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = trig_of_log_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = linear_times_exp_linear_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = polynomial_times_trig_linear_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = linear_times_trig_linear_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = polynomial_times_hyperbolic_linear_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = linear_times_hyperbolic_linear_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = shifted_sqrt_arcsin_inverse_product_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = arcsin_polynomial_substitution_product_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = asinh_polynomial_substitution_product_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = acosh_polynomial_substitution_product_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let IntKind::Mul(l, r) = kind {
        // H(sqrt(x)) * x^(-1/2) (the normalized form of H(sqrt(x))/sqrt(x)) via
        // u = sqrt(x): 2 int H(u) du.
        if let Some(integral) = function_over_sqrt_antiderivative(ctx, l, r, var) {
            return Some(integral);
        }

        // Tabla u-du simbólica (∫s·u′·F(u), F ∈ {exp,sin,cos,sinh,cosh}, u no
        // polinómica): ANTES de product-to-sum, que trata el u interno como
        // ángulo independiente y destroza la forma (∫cos·cos(sin(x)) quedaba
        // residual con el integrando transformado).
        if let Some(integral) = symbolic_derivative_table_antiderivative(ctx, expr, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_product_to_sum_antiderivative(ctx, l, r, var) {
            return Some(integral);
        }
        if !contains_named_var(ctx, l, var) {
            if let Some(integral) =
                constant_scaled_trig_reciprocal_derivative_antiderivative(ctx, l, r, var)
            {
                return Some(integral);
            }
            if let Some(integral) =
                constant_scaled_hyperbolic_reciprocal_square_antiderivative(ctx, l, r, var)
            {
                return Some(integral);
            }
            if let Some(integral) =
                constant_scaled_denominator_power_substitution_antiderivative(ctx, l, r, var)
            {
                return Some(integral);
            }
            if let Some(int_r) = integrate_symbolic_expr(ctx, r, var) {
                return Some(multiply_constant_integral_result(ctx, l, int_r));
            }
        }
        if !contains_named_var(ctx, r, var) {
            if let Some(integral) =
                constant_scaled_trig_reciprocal_derivative_antiderivative(ctx, r, l, var)
            {
                return Some(integral);
            }
            if let Some(integral) =
                constant_scaled_hyperbolic_reciprocal_square_antiderivative(ctx, r, l, var)
            {
                return Some(integral);
            }
            if let Some(integral) =
                constant_scaled_denominator_power_substitution_antiderivative(ctx, r, l, var)
            {
                return Some(integral);
            }
            if let Some(int_l) = integrate_symbolic_expr(ctx, l, var) {
                return Some(multiply_constant_integral_result(ctx, r, int_l));
            }
        }

        // `p(x)·(q)^(-1/2)` with a SUM numerator (`(x+1)/√(x²+1)` normalizes to
        // `(x²+1)^(-1/2)·(x+1)`): distribute over the radical so each `term/√q` hits the existing
        // sqrt-quadratic antiderivatives, then sum by linearity. Last in the chain, so single-term
        // numerators keep their dedicated owners and only currently-declining sums are caught.
        if let Some(integral) =
            linear_numerator_over_reciprocal_sqrt_quadratic_antiderivative(ctx, l, r, var)
        {
            return Some(integral);
        }
    }

    if !contains_named_var(ctx, expr, var) {
        return table_reused_constant_integration_candidate(ctx, expr, var).public_antiderivative();
    }

    if let IntKind::Pow(base, exp) = kind {
        // e^sqrt(x) = Pow(E, sqrt(x)) via u = sqrt(x): 2 int u e^u du.
        if let Some(integral) = exp_of_sqrt_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if is_negative_half(ctx, exp) && is_var_square_plus_one(ctx, base, var) {
            let var_expr = ctx.var(var);
            return Some(ctx.call_builtin(BuiltinFn::Asinh, vec![var_expr]));
        }

        if is_negative_half(ctx, exp) {
            let one = ctx.num(1);
            if let Some(integral) =
                arcsin_symbolic_radius_substitution_from_radicand(ctx, one, base, var)
            {
                return Some(integral);
            }
        }

        if let Some(integral) = trig_square_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = inverse_trig_square_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_ratio_square_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_tan_cot_odd_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_tan_fourth_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_cot_fourth_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_tan_sixth_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_cot_sixth_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_tan_eighth_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_cot_eighth_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_sec_third_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_csc_third_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_sec_fifth_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_csc_fifth_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_sec_fourth_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_csc_fourth_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_sec_sixth_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_csc_sixth_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_sec_eighth_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_csc_eighth_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = hyperbolic_square_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) = hyperbolic_tanh_even_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some(integral) =
            hyperbolic_odd_power_limited_affine_antiderivative(ctx, base, exp, var)
        {
            return Some(integral);
        }

        if let Some(integral) = trig_odd_power_affine_antiderivative(ctx, base, exp, var) {
            return Some(integral);
        }

        if let Some((a, _)) = get_linear_coeffs(ctx, base, var) {
            if !contains_named_var(ctx, exp, var) {
                if let Expr::Number(n) = ctx.get(exp) {
                    if *n == BigRational::from_integer((-1).into()) {
                        let ln_u = ln_abs(ctx, base);
                        return Some(ctx.add(Expr::Div(ln_u, a)));
                    }
                }

                let new_exp = if let Expr::Number(n) = ctx.get(exp) {
                    ctx.add(Expr::Number(n + BigRational::one()))
                } else {
                    let one = ctx.num(1);
                    ctx.add(Expr::Add(exp, one))
                };

                let is_a_one = if let Expr::Number(n) = ctx.get(a) {
                    n.is_one()
                } else {
                    false
                };
                let new_denom = if is_a_one {
                    new_exp
                } else {
                    mul2_raw(ctx, a, new_exp)
                };

                let pow_expr = ctx.add(Expr::Pow(base, new_exp));
                return Some(ctx.add(Expr::Div(pow_expr, new_denom)));
            }
        }

        if !contains_named_var(ctx, base, var) {
            if let Some((a, _)) = get_linear_coeffs(ctx, exp, var) {
                let is_a_one = if let Expr::Number(n) = ctx.get(a) {
                    n.is_one()
                } else {
                    false
                };

                let is_e = if let Expr::Constant(c) = ctx.get(base) {
                    c == &cas_ast::Constant::E
                } else {
                    false
                };

                if is_e {
                    if is_a_one {
                        return Some(expr);
                    }
                    return Some(ctx.add(Expr::Div(expr, a)));
                }

                let ln_c = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
                let denom = if is_a_one {
                    ln_c
                } else {
                    mul2_raw(ctx, a, ln_c)
                };
                return Some(ctx.add(Expr::Div(expr, denom)));
            }
        }
    }

    if let IntKind::Variable(sym_id) = kind {
        if ctx.sym_name(sym_id) == var {
            let var_expr = ctx.var(var);
            let two = ctx.num(2);
            let pow_expr = ctx.add(Expr::Pow(var_expr, two));
            return Some(ctx.add(Expr::Div(pow_expr, two)));
        }
    }

    if let IntKind::Div(num, den) = kind {
        // H(sqrt(x))/sqrt(x) via u = sqrt(x): 2 int H(u) du (literal-Div form, e.g.
        // on recursive entry before the A/sqrt(x) -> A*x^(-1/2) normalization).
        if let Some(integral) = function_over_sqrt_div_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = inverse_trig_over_power_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = reciprocal_exp_linear_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = div_exp_linear_by_parts_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = arctan_sqrt_var_reciprocal_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) =
            arctan_sqrt_var_symbolic_square_shift_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = arctan_sqrt_var_unit_shift_square_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = arctan_sqrt_affine_derivative_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        // u-du simbólica sobre Div (∫s·u′/uᵐ con u compuesta no polinómica):
        // ANTES del clúster trig, que para bases desplazadas produce formas de
        // medio ángulo válidas pero ilegibles (o directamente muele). Los
        // dueños de bases función-desnuda y polinómicas conservan los suyos
        // por los cerrojos internos de la ruta.
        if let Some(integral) = symbolic_power_substitution_div_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = nested_trig_log_derivative_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = polynomial_reciprocal_trig_log_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = reciprocal_trig_log_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_reciprocal_derivative_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) =
            polynomial_trig_reciprocal_derivative_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = polynomial_trig_reciprocal_factor_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = hyperbolic_log_derivative_ratio_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_log_derivative_ratio_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_ratio_square_quotient_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_tan_cot_odd_quotient_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_tan_fourth_quotient_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_cot_fourth_quotient_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_tan_sixth_quotient_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_cot_sixth_quotient_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_tan_eighth_quotient_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_cot_eighth_quotient_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = sine_multiple_angle_ratio_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_sec_third_quotient_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_csc_third_quotient_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_sec_fifth_quotient_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_csc_fifth_quotient_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_sec_fourth_quotient_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_csc_fourth_quotient_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_sec_sixth_quotient_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_csc_sixth_quotient_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_sec_eighth_quotient_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = trig_csc_eighth_quotient_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) =
            hyperbolic_tanh_reciprocal_log_sinh_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = hyperbolic_reciprocal_square_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) =
            arctan_unary_derivative_substitution_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = sqrt_derivative_substitution_div_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) =
            affine_sqrt_product_derivative_div_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) =
            arcsin_polynomial_substitution_div_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = asinh_polynomial_substitution_div_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = acosh_polynomial_substitution_div_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = atanh_polynomial_substitution_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) =
            polynomial_denominator_power_substitution_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) =
            polynomial_negative_denominator_power_substitution_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) =
            polynomial_reciprocal_quotient_denominator_power_substitution_antiderivative(
                ctx, num, den, var,
            )
        {
            return Some(integral);
        }

        if let Some(integral) =
            polynomial_square_minus_constant_log_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) =
            polynomial_log_reciprocal_derivative_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = polynomial_log_derivative_power_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = positive_quadratic_square_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = positive_quadratic_cube_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = arctan_polynomial_substitution_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) = arctan_symbolic_scaled_variable_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Expr::Number(n) = ctx.get(num) {
            if n.is_one() {
                if is_var_square_plus_one(ctx, den, var) {
                    return table_reused_arctan_kernel_integration_candidate(ctx, expr, var)
                        .public_antiderivative();
                }

                if let Some(integral) = reciprocal_trig_square_antiderivative(ctx, den, var) {
                    return Some(integral);
                }

                if let Some((a, _)) = get_linear_coeffs(ctx, den, var) {
                    let ln_den = ln_abs(ctx, den);
                    return Some(ctx.add(Expr::Div(ln_den, a)));
                }
            }
        }

        if let Some(integral) = polynomial_reciprocal_trig_square_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = polynomial_log_derivative_antiderivative(ctx, num, den, var) {
            return Some(integral);
        }

        if let Some(integral) =
            positive_quadratic_linear_numerator_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) = rational_linear_partial_fraction_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) =
            rational_linear_positive_quadratic_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }

        if let Some(integral) =
            rational_multi_linear_positive_quadratic_antiderivative(ctx, num, den, var)
        {
            return Some(integral);
        }
    }

    if let Some(integral) = exponential_rational_substitution_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = odd_power_times_quadratic_function_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = polynomial_times_trig_square_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = polynomial_times_trig_square_substitution_antiderivative(ctx, expr, var)
    {
        return Some(integral);
    }

    if let Some(integral) = polynomial_times_higher_even_trig_power_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = linear_radical_substitution_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = weierstrass_rational_substitution_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = mixed_trig_power_substitution_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = quartic_symmetric_substitution_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = quadratic_radical_over_monomial_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let Some(integral) = transcendental_chain_substitution_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    if let IntKind::Function(fn_id, args) = kind {
        if ctx.builtin_of(fn_id) == Some(BuiltinFn::Log) && args.len() == 2 {
            let base = args[0];
            let arg = args[1];
            let base_ln = valid_constant_log_base_ln(ctx, base)?;
            let log_expr = if base_ln.is_none() {
                ctx.call_builtin(BuiltinFn::Ln, vec![arg])
            } else {
                expr
            };
            return affine_constant_base_log_antiderivative(ctx, log_expr, arg, base_ln, var);
        }

        if args.len() == 1 {
            let arg = args[0];
            // int |a*x+b| dx = (a*x+b)|a*x+b|/(2a); int sqrt(x^2) = int |x| = x|x|/2.
            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Abs) {
                if let Some(integral) = abs_affine_antiderivative(ctx, arg, var) {
                    return Some(integral);
                }
            }
            // int sign(a*x+b) dx = |a*x+b|/a; int sign(x) = |x|.
            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Sign) {
                if let Some(integral) = sign_affine_antiderivative(ctx, arg, var) {
                    return Some(integral);
                }
            }

            if let Some(
                builtin @ (BuiltinFn::Tan | BuiltinFn::Cot | BuiltinFn::Sec | BuiltinFn::Csc),
            ) = ctx.builtin_of(fn_id)
            {
                return trig_log_antiderivative(ctx, builtin, arg, var);
            }

            if matches!(
                ctx.builtin_of(fn_id),
                Some(BuiltinFn::Arctan | BuiltinFn::Atan)
            ) {
                if let Some(integral) =
                    arctan_reciprocal_affine_variable_antiderivative(ctx, arg, var)
                {
                    return Some(integral);
                }
                if let Some(integral) = arctan_scaled_variable_antiderivative(ctx, arg, var) {
                    return Some(integral);
                }
            }

            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Asinh) {
                if let Some(integral) = asinh_affine_antiderivative(ctx, arg, var) {
                    return Some(integral);
                }
            }

            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Atanh) {
                if let Some(integral) = atanh_affine_antiderivative(ctx, arg, var) {
                    return Some(integral);
                }
            }

            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Acosh) {
                if let Some(integral) = acosh_affine_antiderivative(ctx, arg, var) {
                    return Some(integral);
                }
            }

            if let Some(
                builtin @ (BuiltinFn::Arcsin
                | BuiltinFn::Asin
                | BuiltinFn::Arccos
                | BuiltinFn::Acos
                | BuiltinFn::Arctan
                | BuiltinFn::Atan
                | BuiltinFn::Sin
                | BuiltinFn::Cos
                | BuiltinFn::Sinh
                | BuiltinFn::Cosh),
            ) = ctx.builtin_of(fn_id)
            {
                if let Some(integral) = function_of_sqrt_antiderivative(ctx, builtin, arg, var) {
                    return Some(integral);
                }
            }

            if let Some(
                builtin @ (BuiltinFn::Arcsin
                | BuiltinFn::Asin
                | BuiltinFn::Arccos
                | BuiltinFn::Acos),
            ) = ctx.builtin_of(fn_id)
            {
                if let Some(integral) =
                    bounded_inverse_trig_linear_antiderivative(ctx, builtin, arg, var)
                {
                    return Some(integral);
                }
            }

            if let Some((a, _)) = get_linear_coeffs(ctx, arg, var) {
                let is_a_one = if let Expr::Number(n) = ctx.get(a) {
                    n.is_one()
                } else {
                    false
                };

                match ctx.builtin_of(fn_id) {
                    Some(BuiltinFn::Log2) => {
                        let base_ln = positive_integer_constant_log_base_ln(ctx, 2);
                        return affine_constant_base_log_antiderivative(
                            ctx,
                            expr,
                            arg,
                            Some(base_ln),
                            var,
                        );
                    }
                    Some(BuiltinFn::Log10) => {
                        let base_ln = positive_integer_constant_log_base_ln(ctx, 10);
                        return affine_constant_base_log_antiderivative(
                            ctx,
                            expr,
                            arg,
                            Some(base_ln),
                            var,
                        );
                    }
                    Some(BuiltinFn::Sin) => {
                        let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                        let integral = ctx.add(Expr::Neg(cos_arg));
                        return Some(scale_by_reciprocal_linear_coeff(ctx, integral, a));
                    }
                    Some(BuiltinFn::Cos) => {
                        let integral = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
                        return Some(scale_by_reciprocal_linear_coeff(ctx, integral, a));
                    }
                    Some(BuiltinFn::Exp) => {
                        return Some(scale_by_reciprocal_linear_coeff(ctx, expr, a));
                    }
                    Some(BuiltinFn::Ln) => {
                        if is_a_one && is_var(ctx, arg, var) {
                            let product = mul2_raw(ctx, arg, expr);
                            return Some(ctx.add(Expr::Sub(product, arg)));
                        }

                        let one = ctx.num(1);
                        let log_minus_one = ctx.add(Expr::Sub(expr, one));
                        let integral = mul2_raw(ctx, arg, log_minus_one);
                        if is_a_one {
                            return Some(integral);
                        }
                        return Some(ctx.add(Expr::Div(integral, a)));
                    }
                    _ => {}
                }
            }
        }
    }

    // Last resort: `sin^p·cos^q` with an odd numerator power and ANY (incl. negative) companion
    // power, via the `u=cos`/`u=sin` substitution. Fills `sin(x)/cos(x)^n` (n ≥ 4) and friends.
    if let Some(integral) = trig_odd_power_companion_substitution_antiderivative(ctx, expr, var) {
        return Some(integral);
    }

    None
}

/// Returns `(a, b)` such that `expr = a*var + b`.
pub fn get_linear_coeffs(ctx: &mut Context, expr: ExprId, var: &str) -> Option<(ExprId, ExprId)> {
    if !contains_named_var(ctx, expr, var) {
        return Some((ctx.num(0), expr));
    }

    match ctx.get(expr) {
        Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var => Some((ctx.num(1), ctx.num(0))),
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            if !contains_named_var(ctx, l, var) && is_var(ctx, r, var) {
                return Some((l, ctx.num(0)));
            }
            if !contains_named_var(ctx, l, var) {
                let (a, b) = get_linear_coeffs(ctx, r, var)?;
                if !contains_named_var(ctx, a, var) && !contains_named_var(ctx, b, var) {
                    return Some((
                        multiply_linear_part(ctx, l, a),
                        multiply_linear_part(ctx, l, b),
                    ));
                }
            }
            if is_var(ctx, l, var) && !contains_named_var(ctx, r, var) {
                return Some((r, ctx.num(0)));
            }
            if !contains_named_var(ctx, r, var) {
                let (a, b) = get_linear_coeffs(ctx, l, var)?;
                if !contains_named_var(ctx, a, var) && !contains_named_var(ctx, b, var) {
                    return Some((
                        multiply_linear_part(ctx, r, a),
                        multiply_linear_part(ctx, r, b),
                    ));
                }
            }
            None
        }
        Expr::Div(num, den) => {
            let (num, den) = (*num, *den);
            if contains_named_var(ctx, den, var) {
                return None;
            }
            let (a, b) = get_linear_coeffs(ctx, num, var)?;
            if !contains_named_var(ctx, a, var) && !contains_named_var(ctx, b, var) {
                return Some((
                    divide_linear_part(ctx, a, den),
                    divide_linear_part(ctx, b, den),
                ));
            }
            None
        }
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            let l_coeffs = get_linear_coeffs(ctx, l, var);
            let r_coeffs = get_linear_coeffs(ctx, r, var);

            if let (Some((a1, b1)), Some((a2, b2))) = (l_coeffs, r_coeffs) {
                if !contains_named_var(ctx, a1, var) && !contains_named_var(ctx, a2, var) {
                    let a = add_linear_parts(ctx, a1, a2);
                    let b = add_linear_parts(ctx, b1, b2);
                    return Some((a, b));
                }
            }
            None
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            let l_coeffs = get_linear_coeffs(ctx, l, var);
            let r_coeffs = get_linear_coeffs(ctx, r, var);
            if let (Some((a1, b1)), Some((a2, b2))) = (l_coeffs, r_coeffs) {
                if !contains_named_var(ctx, a1, var) && !contains_named_var(ctx, a2, var) {
                    let a = sub_linear_parts(ctx, a1, a2);
                    let b = sub_linear_parts(ctx, b1, b2);
                    return Some((a, b));
                }
            }
            None
        }
        Expr::Neg(inner) => {
            let (a, b) = get_linear_coeffs(ctx, *inner, var)?;
            let neg_a = neg_linear_part(ctx, a);
            let neg_b = neg_linear_part(ctx, b);
            Some((neg_a, neg_b))
        }
        _ => None,
    }
}

pub(super) fn is_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    if let Expr::Variable(sym_id) = ctx.get(expr) {
        ctx.sym_name(*sym_id) == var
    } else {
        false
    }
}
