//! `symbolic_integration_support`: familia `by_parts`.
//!
//! Ver la cabecera de `symbolic_integration_support.rs` para el contexto.

use super::*;

pub(super) fn sqrt_chain_argument_derivative_parts(
    ctx: &Context,
    arg: ExprId,
    var: &str,
) -> Option<(ExprId, BigRational)> {
    if let Some(radicand) = sqrt_like_radicand(ctx, arg) {
        return Some((radicand, BigRational::one()));
    }

    match ctx.get(arg) {
        Expr::Add(left, right) => {
            if !contains_named_var(ctx, *left, var) {
                return signed_sqrt_like_radicand(ctx, *right);
            }
            if !contains_named_var(ctx, *right, var) {
                return signed_sqrt_like_radicand(ctx, *left);
            }
            None
        }
        Expr::Sub(left, right) => {
            if !contains_named_var(ctx, *left, var) {
                let (radicand, sign) = signed_sqrt_like_radicand(ctx, *right)?;
                return Some((radicand, -sign));
            }
            if !contains_named_var(ctx, *right, var) {
                return signed_sqrt_like_radicand(ctx, *left);
            }
            None
        }
        _ => None,
    }
}

pub(super) fn symbolic_square_shift_argument_parts(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, SymbolicSquareShiftArgument, BigRational)> {
    if let Some(parameter) = var_plus_symbolic_parameter_square(ctx, expr, var) {
        return Some((
            parameter,
            SymbolicSquareShiftArgument::DivideByParameter,
            BigRational::one(),
        ));
    }
    if let Some((parameter, argument_scale)) =
        numeric_square_scaled_var_plus_symbolic_parameter_square(ctx, expr, var)
    {
        return Some((
            parameter,
            SymbolicSquareShiftArgument::DivideByParameter,
            argument_scale,
        ));
    }
    symbolic_parameter_square_times_var_plus_one(ctx, expr, var).map(|parameter| {
        (
            parameter,
            SymbolicSquareShiftArgument::MultiplyByParameter,
            BigRational::one(),
        )
    })
}

fn reciprocal_sqrt_var_over_symbolic_square_shift_parts(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<SymbolicSquareShiftDenominator> {
    let (parameter, argument, argument_scale) =
        symbolic_square_shift_argument_parts(ctx, den, var)?;
    let factors = mul_leaves(ctx, num);
    let mut scale = BigRational::one();
    let mut saw_reciprocal_sqrt_var = false;

    for factor in factors {
        if reciprocal_sqrt_like_radicand(ctx, factor)
            .is_some_and(|radicand| is_var(ctx, radicand, var))
        {
            if saw_reciprocal_sqrt_var {
                return None;
            }
            saw_reciprocal_sqrt_var = true;
        } else {
            scale *= rational_constant_value(ctx, factor)?;
        }
    }

    saw_reciprocal_sqrt_var.then_some(SymbolicSquareShiftDenominator {
        scale,
        parameter,
        argument,
        argument_scale,
    })
}

pub(super) fn arctan_sqrt_var_symbolic_square_shift_parts(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<SymbolicSquareShiftDenominator> {
    if let Some(parts) = sqrt_var_times_symbolic_square_shift_denominator(ctx, den, var) {
        let numerator_scale = rational_constant_value(ctx, num)?;
        return Some(SymbolicSquareShiftDenominator {
            scale: numerator_scale / parts.scale,
            parameter: parts.parameter,
            argument: parts.argument,
            argument_scale: parts.argument_scale,
        });
    }

    reciprocal_sqrt_var_over_symbolic_square_shift_parts(ctx, num, den, var)
}

fn sqrt_var_times_positive_linear_parts(
    ctx: &Context,
    den: ExprId,
    var: &str,
) -> Option<SqrtLinearDenominator> {
    sqrt_var_times_positive_linear_denominator(ctx, den, var)
        .or_else(|| expanded_sqrt_var_times_positive_linear_denominator(ctx, den, var))
}

fn reciprocal_sqrt_var_over_positive_linear_parts(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<SqrtLinearDenominator> {
    let (slope, offset) = positive_linear_polynomial_coeffs(ctx, den, var)?;

    let factors = mul_leaves(ctx, num);
    let mut scale = BigRational::one();
    let mut saw_reciprocal_sqrt_var = false;

    for factor in factors {
        if reciprocal_sqrt_like_radicand(ctx, factor)
            .is_some_and(|radicand| is_var(ctx, radicand, var))
        {
            if saw_reciprocal_sqrt_var {
                return None;
            }
            saw_reciprocal_sqrt_var = true;
        } else {
            scale *= rational_constant_value(ctx, factor)?;
        }
    }

    saw_reciprocal_sqrt_var.then_some(SqrtLinearDenominator {
        scale,
        slope,
        offset,
    })
}

pub(super) fn arctan_sqrt_var_positive_linear_parts(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<SqrtLinearDenominator> {
    if let Some(parts) = sqrt_var_times_positive_linear_parts(ctx, den, var) {
        if parts.scale.is_zero() {
            return None;
        }
        let numerator_scale = rational_constant_value(ctx, num)?;
        return Some(SqrtLinearDenominator {
            scale: numerator_scale / parts.scale,
            slope: parts.slope,
            offset: parts.offset,
        });
    }

    reciprocal_sqrt_var_over_positive_linear_parts(ctx, num, den, var)
}

fn positive_square_shift_linear_square_parts(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(BigRational, BigRational)> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if rational_constant_value(ctx, *exp) != Some(BigRational::from_integer(2.into())) {
        return None;
    }
    let (slope, offset) = positive_linear_polynomial_coeffs(ctx, *base, var)?;
    if !slope.is_one() || !offset.is_positive() {
        return None;
    }
    let offset_root = exact_rational_sqrt(&offset)?;
    Some((offset, offset_root))
}

fn positive_square_shift_linear_square_denominator_parts(
    ctx: &Context,
    den: ExprId,
    var: &str,
) -> Option<PositiveSquareShiftDenominator> {
    let factors = mul_leaves(ctx, den);
    let mut scale = BigRational::one();
    let mut square_offset = None;
    let mut square_offset_root = None;

    for factor in factors {
        if let Some((offset, offset_root)) =
            positive_square_shift_linear_square_parts(ctx, factor, var)
        {
            if square_offset.is_some() {
                return None;
            }
            square_offset = Some(offset);
            square_offset_root = Some(offset_root);
        } else {
            scale *= rational_constant_value(ctx, factor)?;
        }
    }

    Some(PositiveSquareShiftDenominator {
        scale,
        offset: square_offset?,
        offset_root: square_offset_root?,
    })
}

fn sqrt_var_times_positive_square_shift_linear_square_denominator_parts(
    ctx: &Context,
    den: ExprId,
    var: &str,
) -> Option<PositiveSquareShiftDenominator> {
    let factors = mul_leaves(ctx, den);
    let mut scale = BigRational::one();
    let mut saw_sqrt_var = false;
    let mut square_offset = None;
    let mut square_offset_root = None;

    for factor in factors {
        if sqrt_like_radicand(ctx, factor).is_some_and(|radicand| is_var(ctx, radicand, var)) {
            if saw_sqrt_var {
                return None;
            }
            saw_sqrt_var = true;
        } else if let Some((offset, offset_root)) =
            positive_square_shift_linear_square_parts(ctx, factor, var)
        {
            if square_offset.is_some() {
                return None;
            }
            square_offset = Some(offset);
            square_offset_root = Some(offset_root);
        } else {
            scale *= rational_constant_value(ctx, factor)?;
        }
    }

    saw_sqrt_var.then_some(PositiveSquareShiftDenominator {
        scale,
        offset: square_offset?,
        offset_root: square_offset_root?,
    })
}

fn reciprocal_sqrt_var_over_positive_square_shift_linear_square_parts(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<PositiveSquareShiftDenominator> {
    let denominator = positive_square_shift_linear_square_denominator_parts(ctx, den, var)?;
    let factors = mul_leaves(ctx, num);
    let mut numerator_scale = BigRational::one();
    let mut saw_reciprocal_sqrt_var = false;

    for factor in factors {
        if reciprocal_sqrt_like_radicand(ctx, factor)
            .is_some_and(|radicand| is_var(ctx, radicand, var))
        {
            if saw_reciprocal_sqrt_var {
                return None;
            }
            saw_reciprocal_sqrt_var = true;
        } else {
            numerator_scale *= rational_constant_value(ctx, factor)?;
        }
    }

    saw_reciprocal_sqrt_var.then_some(PositiveSquareShiftDenominator {
        scale: numerator_scale / denominator.scale,
        offset: denominator.offset,
        offset_root: denominator.offset_root,
    })
}

fn sqrt_var_over_var_times_positive_square_shift_linear_square_parts(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<PositiveSquareShiftDenominator> {
    let numerator_factors = mul_leaves(ctx, num);
    let mut numerator_scale = BigRational::one();
    let mut saw_sqrt_var = false;
    for factor in numerator_factors {
        if sqrt_like_radicand(ctx, factor).is_some_and(|radicand| is_var(ctx, radicand, var)) {
            if saw_sqrt_var {
                return None;
            }
            saw_sqrt_var = true;
        } else {
            numerator_scale *= rational_constant_value(ctx, factor)?;
        }
    }
    if !saw_sqrt_var {
        return None;
    }

    let denominator_factors = mul_leaves(ctx, den);
    let mut denominator_scale = BigRational::one();
    let mut saw_var = false;
    let mut square_offset = None;
    let mut square_offset_root = None;
    for factor in denominator_factors {
        if is_var(ctx, factor, var) {
            if saw_var {
                return None;
            }
            saw_var = true;
        } else if let Some((offset, offset_root)) =
            positive_square_shift_linear_square_parts(ctx, factor, var)
        {
            if square_offset.is_some() {
                return None;
            }
            square_offset = Some(offset);
            square_offset_root = Some(offset_root);
        } else {
            denominator_scale *= rational_constant_value(ctx, factor)?;
        }
    }

    saw_var.then_some(PositiveSquareShiftDenominator {
        scale: numerator_scale / denominator_scale,
        offset: square_offset?,
        offset_root: square_offset_root?,
    })
}

pub(super) fn arctan_sqrt_var_unit_shift_square_parts(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<PositiveSquareShiftDenominator> {
    if let Some(denominator) =
        sqrt_var_times_positive_square_shift_linear_square_denominator_parts(ctx, den, var)
    {
        if denominator.scale.is_zero() {
            return None;
        }
        let numerator_scale = rational_constant_value(ctx, num)?;
        return Some(PositiveSquareShiftDenominator {
            scale: numerator_scale / denominator.scale,
            offset: denominator.offset,
            offset_root: denominator.offset_root,
        });
    }

    reciprocal_sqrt_var_over_positive_square_shift_linear_square_parts(ctx, num, den, var).or_else(
        || sqrt_var_over_var_times_positive_square_shift_linear_square_parts(ctx, num, den, var),
    )
}

pub(super) fn arctan_sqrt_var_reciprocal_antiderivative_from_parts(
    ctx: &mut Context,
    scale: BigRational,
    slope: BigRational,
    offset: BigRational,
    var: &str,
) -> Option<ExprId> {
    if scale.is_zero() {
        return None;
    }
    if !slope.is_positive() || !offset.is_positive() {
        return None;
    }

    let product = slope.clone() * offset.clone();
    let product_root = exact_rational_sqrt(&product)?;
    let ratio = slope / offset;
    let ratio_root = exact_rational_sqrt(&ratio)?;

    let var_expr = ctx.var(var);
    let sqrt_var = ctx.call_builtin(BuiltinFn::Sqrt, vec![var_expr]);
    let arctan_arg = scale_factor(ctx, ratio_root, sqrt_var);
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arctan_arg]);
    Some(scale_factor(
        ctx,
        scale * BigRational::from_integer(2.into()) / product_root,
        arctan,
    ))
}

pub(super) fn arctan_sqrt_var_symbolic_square_shift_antiderivative_from_parts(
    ctx: &mut Context,
    scale: BigRational,
    parameter: ExprId,
    argument: SymbolicSquareShiftArgument,
    argument_scale: BigRational,
    var: &str,
) -> Option<ExprId> {
    if scale.is_zero() || argument_scale.is_zero() || contains_named_var(ctx, parameter, var) {
        return None;
    }

    let var_expr = ctx.var(var);
    let sqrt_var = ctx.call_builtin(BuiltinFn::Sqrt, vec![var_expr]);
    let scaled_sqrt_var = scale_factor(ctx, argument_scale.clone(), sqrt_var);
    let arctan_arg = match argument {
        SymbolicSquareShiftArgument::DivideByParameter => {
            ctx.add(Expr::Div(scaled_sqrt_var, parameter))
        }
        SymbolicSquareShiftArgument::MultiplyByParameter => {
            mul2_raw(ctx, parameter, scaled_sqrt_var)
        }
    };
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arctan_arg]);
    let coefficient = scale * BigRational::from_integer(2.into()) / argument_scale;
    if coefficient.is_one() {
        return Some(ctx.add(Expr::Div(arctan, parameter)));
    }
    if coefficient == BigRational::from_integer((-1).into()) {
        let quotient = ctx.add(Expr::Div(arctan, parameter));
        return Some(ctx.add(Expr::Neg(quotient)));
    }
    let coefficient = rational_over_expr(ctx, coefficient, parameter);
    Some(mul2_raw(ctx, coefficient, arctan))
}

fn asinh_normalized_sqrt_reciprocal_parts(
    ctx: &Context,
    sqrt_radicand: ExprId,
    polynomial_factors: &[ExprId],
    num_scale: BigRational,
    den_scale: BigRational,
    var: &str,
) -> Option<InverseHyperbolicSqrtReciprocalParts> {
    if polynomial_factors.len() != 2 || den_scale.is_zero() {
        return None;
    }
    let sqrt_poly = affine_polynomial(ctx, sqrt_radicand, var)?;

    for base_idx in 0..2 {
        let base = affine_polynomial(ctx, polynomial_factors[base_idx], var)?;
        let constant = positive_constant_difference(&sqrt_poly, &base)?;
        let offset = base
            .coeffs
            .first()
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if offset.is_negative() {
            continue;
        }

        let radicand_factor =
            Polynomial::from_expr(ctx, polynomial_factors[1 - base_idx], var).ok()?;
        let radicand_ratio = constant_polynomial_ratio(&sqrt_poly, &radicand_factor)?;
        let slope = base.coeffs.get(1).cloned()?;
        if slope.is_zero() {
            continue;
        }
        let kernel_scale = num_scale.clone() * radicand_ratio / den_scale.clone();
        let scale_without_root = -BigRational::from_integer(2.into()) * kernel_scale / slope;
        let (scale, scale_sqrt_factor) =
            inverse_hyperbolic_scale_over_constant_root(scale_without_root, &constant)?;
        let (base, constant) = normalize_asinh_reciprocal_display_base(base, constant);
        return Some(InverseHyperbolicSqrtReciprocalParts {
            base,
            constant,
            scale,
            scale_sqrt_factor,
        });
    }

    None
}

fn atanh_normalized_sqrt_reciprocal_parts(
    ctx: &Context,
    sqrt_radicand: ExprId,
    polynomial_factors: &[ExprId],
    num_scale: BigRational,
    den_scale: BigRational,
    var: &str,
) -> Option<InverseHyperbolicSqrtReciprocalParts> {
    if polynomial_factors.len() != 2 || den_scale.is_zero() {
        return None;
    }
    let base = affine_polynomial(ctx, sqrt_radicand, var)?;
    let base_slope = base.coeffs.get(1).cloned()?;
    if base_slope.is_zero() {
        return None;
    }

    for base_idx in 0..2 {
        let Some(base_factor) = Polynomial::from_expr(ctx, polynomial_factors[base_idx], var).ok()
        else {
            continue;
        };
        let Some(base_ratio) = constant_polynomial_ratio(&base, &base_factor) else {
            continue;
        };
        let Some(gap_factor) =
            Polynomial::from_expr(ctx, polynomial_factors[1 - base_idx], var).ok()
        else {
            continue;
        };
        let Some((constant, gap_alignment)) = atanh_gap_constant_and_alignment(&base, &gap_factor)
        else {
            continue;
        };
        let constant_root = exact_rational_sqrt(&constant)?;
        let kernel_scale = num_scale.clone() * base_ratio * gap_alignment / den_scale.clone();
        let scale = -BigRational::from_integer(2.into()) * kernel_scale
            / (constant_root * base_slope.clone());
        return Some(InverseHyperbolicSqrtReciprocalParts {
            base,
            constant,
            scale,
            scale_sqrt_factor: None,
        });
    }

    None
}

fn scaled_asinh_sqrt_reciprocal_parts(
    sqrt_poly: &Polynomial,
    base: &Polynomial,
    num_scale: BigRational,
    den_scale: BigRational,
) -> Option<InverseHyperbolicSqrtReciprocalParts> {
    let base_slope = base.coeffs.get(1).cloned()?;
    let sqrt_slope = sqrt_poly.coeffs.get(1).cloned()?;
    if base_slope.is_zero() || sqrt_slope.is_zero() || den_scale.is_zero() {
        return None;
    }

    let constant = sqrt_slope / base_slope.clone();
    if !constant.is_positive() {
        return None;
    }

    let normalized_sqrt = sqrt_poly.div_scalar(&constant);
    let gap = normalized_sqrt.sub(base);
    if gap.degree() != 0 {
        return None;
    }
    let gap_constant = gap
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if gap_constant != constant {
        return None;
    }

    let kernel_scale = num_scale / den_scale;
    let scale =
        -BigRational::from_integer(2.into()) * kernel_scale / (constant.clone() * base_slope);
    Some(InverseHyperbolicSqrtReciprocalParts {
        base: base.clone(),
        constant,
        scale,
        scale_sqrt_factor: None,
    })
}

fn atanh_expanded_sqrt_var_reciprocal_parts(
    ctx: &Context,
    num_scale: BigRational,
    den: ExprId,
    var: &str,
) -> Option<InverseHyperbolicSqrtReciprocalParts> {
    let (left, right, right_sign) = match ctx.get(den) {
        Expr::Add(left, right) => (*left, *right, BigRational::one()),
        Expr::Sub(left, right) => (*left, *right, BigRational::from_integer((-1).into())),
        _ => return None,
    };

    let (left_scale, left_power) = scaled_var_power_term(ctx, left, var)?;
    let (right_scale, right_power) = scaled_var_power_term(ctx, right, var)?;
    let right_scale = right_scale * right_sign;
    let half = BigRational::new(1.into(), 2.into());
    let three_halves = BigRational::new(3.into(), 2.into());
    let (linear_scale, root_scale) = if left_power == three_halves && right_power == half {
        (left_scale, right_scale)
    } else if left_power == half && right_power == three_halves {
        (right_scale, left_scale)
    } else {
        return None;
    };
    if linear_scale.is_zero() {
        return None;
    }

    let constant = -root_scale / linear_scale.clone();
    if !constant.is_positive() {
        return None;
    }
    let constant_root = exact_rational_sqrt(&constant)?;
    let kernel_scale = num_scale / linear_scale;
    let scale = -BigRational::from_integer(2.into()) * kernel_scale / constant_root;

    Some(InverseHyperbolicSqrtReciprocalParts {
        base: Polynomial::new(
            vec![BigRational::zero(), BigRational::one()],
            var.to_string(),
        ),
        constant,
        scale,
        scale_sqrt_factor: None,
    })
}

pub(super) fn inverse_hyperbolic_sqrt_reciprocal_parts(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    kind: InverseHyperbolicSqrtReciprocalKind,
) -> Option<InverseHyperbolicSqrtReciprocalParts> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };

    let mut num_scale = BigRational::one();
    let mut numerator_sqrt_radicand = None;
    for factor in mul_leaves(ctx, *num) {
        if let Some(radicand) = sqrt_like_radicand(ctx, factor) {
            if numerator_sqrt_radicand.is_some() {
                return None;
            }
            numerator_sqrt_radicand = Some(radicand);
        } else {
            num_scale *= rational_constant_value(ctx, factor)?;
        }
    }

    if numerator_sqrt_radicand.is_none()
        && matches!(kind, InverseHyperbolicSqrtReciprocalKind::Atanh)
    {
        if let Some(parts) =
            atanh_expanded_sqrt_var_reciprocal_parts(ctx, num_scale.clone(), *den, var)
        {
            return Some(parts);
        }
    }

    let mut den_scale = BigRational::one();
    let mut sqrt_radicand = None;
    let mut polynomial_factors = Vec::new();
    for factor in mul_leaves(ctx, *den) {
        if let Some(radicand) = sqrt_like_radicand(ctx, factor) {
            if sqrt_radicand.is_some() {
                return None;
            }
            sqrt_radicand = Some(radicand);
        } else if let Some(scale) = rational_constant_value(ctx, factor) {
            den_scale *= scale;
        } else {
            polynomial_factors.push(factor);
        }
    }

    if den_scale.is_zero() {
        return None;
    }

    if let Some(radicand) = numerator_sqrt_radicand {
        if sqrt_radicand.is_none() && matches!(kind, InverseHyperbolicSqrtReciprocalKind::Asinh) {
            return asinh_normalized_sqrt_reciprocal_parts(
                ctx,
                radicand,
                &polynomial_factors,
                num_scale,
                den_scale,
                var,
            );
        }
        if sqrt_radicand.is_none() && matches!(kind, InverseHyperbolicSqrtReciprocalKind::Atanh) {
            return atanh_normalized_sqrt_reciprocal_parts(
                ctx,
                radicand,
                &polynomial_factors,
                num_scale,
                den_scale,
                var,
            );
        }
        return None;
    }

    let sqrt_poly = affine_polynomial(ctx, sqrt_radicand?, var)?;
    let (base, constant) = match kind {
        InverseHyperbolicSqrtReciprocalKind::Asinh => {
            let other_poly = single_affine_factor(ctx, &polynomial_factors, var)?;
            let offset = other_poly
                .coeffs
                .first()
                .cloned()
                .unwrap_or_else(BigRational::zero);
            if offset.is_negative() {
                return None;
            }
            if let Some(constant) = positive_constant_difference(&sqrt_poly, &other_poly) {
                (other_poly, constant)
            } else if offset.is_positive() {
                if let Some(parts) = scaled_asinh_sqrt_reciprocal_parts(
                    &sqrt_poly,
                    &other_poly,
                    num_scale.clone(),
                    den_scale.clone(),
                ) {
                    return Some(parts);
                }
                return None;
            } else {
                return None;
            }
        }
        InverseHyperbolicSqrtReciprocalKind::Atanh => {
            let gap_factor = single_affine_factor(ctx, &polynomial_factors, var)?;
            let (constant, gap_alignment) =
                atanh_gap_constant_and_alignment(&sqrt_poly, &gap_factor)?;
            let slope = sqrt_poly.coeffs.get(1).cloned()?;
            if slope.is_zero() {
                return None;
            }
            let constant_root = exact_rational_sqrt(&constant)?;
            let kernel_scale = num_scale * gap_alignment / den_scale;
            let scale =
                -BigRational::from_integer(2.into()) * kernel_scale / (constant_root * slope);
            let (base, constant) = normalize_atanh_reciprocal_display_base(sqrt_poly, constant);
            return Some(InverseHyperbolicSqrtReciprocalParts {
                base,
                constant,
                scale,
                scale_sqrt_factor: None,
            });
        }
    };

    let slope = base.coeffs.get(1).cloned()?;
    if slope.is_zero() {
        return None;
    }
    let kernel_scale = num_scale / den_scale;
    let scale_without_root = -BigRational::from_integer(2.into()) * kernel_scale / slope;
    let (scale, scale_sqrt_factor) =
        inverse_hyperbolic_scale_over_constant_root(scale_without_root, &constant)?;
    let (base, constant) = if matches!(kind, InverseHyperbolicSqrtReciprocalKind::Asinh) {
        normalize_asinh_reciprocal_display_base(base, constant)
    } else {
        (base, constant)
    };

    Some(InverseHyperbolicSqrtReciprocalParts {
        base,
        constant,
        scale,
        scale_sqrt_factor,
    })
}

pub(super) fn arctan_sqrt_affine_derivative_parts(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ArctanSqrtAffineDerivativeParts> {
    let mut num_scale = BigRational::one();
    let mut numerator_sqrt_radicand = None;

    for factor in mul_leaves(ctx, num) {
        if let Some(radicand) = sqrt_like_radicand(ctx, factor) {
            if numerator_sqrt_radicand.is_some() {
                return None;
            }
            numerator_sqrt_radicand = Some(radicand);
        } else {
            num_scale *= rational_constant_value(ctx, factor)?;
        }
    }

    let mut den_scale = BigRational::one();
    let mut denominator_sqrt_radicand = None;
    let mut polynomial_factors = Vec::new();

    for factor in mul_leaves(ctx, den) {
        if let Some(radicand) = sqrt_like_radicand(ctx, factor) {
            if denominator_sqrt_radicand.is_some() {
                return None;
            }
            denominator_sqrt_radicand = Some(radicand);
        } else if let Some(scale) = rational_constant_value(ctx, factor) {
            den_scale *= scale;
        } else {
            polynomial_factors.push(factor);
        }
    }

    if den_scale.is_zero() {
        return None;
    }

    if let Some(radicand) = numerator_sqrt_radicand {
        if denominator_sqrt_radicand.is_some() || polynomial_factors.len() != 2 {
            return None;
        }
        return arctan_sqrt_affine_derivative_parts_from_normalized_factors(
            ctx,
            radicand,
            &polynomial_factors,
            num_scale,
            den_scale,
            var,
        );
    }

    let radicand = denominator_sqrt_radicand?;
    if polynomial_factors.len() != 1 {
        return None;
    }
    arctan_sqrt_affine_derivative_parts_from_direct_factors(
        ctx,
        radicand,
        polynomial_factors[0],
        num_scale,
        den_scale,
        var,
    )
}

fn arctan_sqrt_affine_derivative_parts_from_direct_factors(
    ctx: &Context,
    radicand: ExprId,
    gap_factor: ExprId,
    num_scale: BigRational,
    den_scale: BigRational,
    var: &str,
) -> Option<ArctanSqrtAffineDerivativeParts> {
    let radicand_poly = affine_radicand_polynomial(ctx, radicand, var)?;
    let gap = arctan_sqrt_affine_gap_parts(ctx, &radicand_poly, gap_factor, var)?;
    let kernel_scale = num_scale * gap.kernel_scale_factor.clone() / den_scale;
    let scale =
        arctan_sqrt_affine_output_scale(&radicand_poly, kernel_scale, &gap.denominator_root)?;
    Some(ArctanSqrtAffineDerivativeParts {
        radicand,
        scale,
        argument_scale: gap.argument_scale,
    })
}

fn arctan_sqrt_affine_derivative_parts_from_normalized_factors(
    ctx: &Context,
    radicand: ExprId,
    factors: &[ExprId],
    num_scale: BigRational,
    den_scale: BigRational,
    var: &str,
) -> Option<ArctanSqrtAffineDerivativeParts> {
    let radicand_poly = affine_radicand_polynomial(ctx, radicand, var)?;
    let (radicand_ratio, gap) =
        normalized_affine_radicand_and_gap_parts(ctx, &radicand_poly, factors, var)?;
    let kernel_scale = num_scale * radicand_ratio * gap.kernel_scale_factor.clone() / den_scale;
    let scale =
        arctan_sqrt_affine_output_scale(&radicand_poly, kernel_scale, &gap.denominator_root)?;
    Some(ArctanSqrtAffineDerivativeParts {
        radicand,
        scale,
        argument_scale: gap.argument_scale,
    })
}

fn normalized_affine_radicand_and_gap_parts(
    ctx: &Context,
    radicand: &Polynomial,
    factors: &[ExprId],
    var: &str,
) -> Option<(BigRational, ArctanSqrtAffineGapParts)> {
    if factors.len() != 2 {
        return None;
    }

    polynomial_ratio_to_expr_factor(ctx, radicand, factors[0], var)
        .and_then(|radicand_ratio| {
            arctan_sqrt_affine_gap_parts(ctx, radicand, factors[1], var)
                .map(|gap| (radicand_ratio, gap))
        })
        .or_else(|| {
            polynomial_ratio_to_expr_factor(ctx, radicand, factors[1], var).and_then(
                |radicand_ratio| {
                    arctan_sqrt_affine_gap_parts(ctx, radicand, factors[0], var)
                        .map(|gap| (radicand_ratio, gap))
                },
            )
        })
}

fn arctan_sqrt_affine_gap_parts(
    ctx: &Context,
    radicand: &Polynomial,
    gap_factor: ExprId,
    var: &str,
) -> Option<ArctanSqrtAffineGapParts> {
    let gap = Polynomial::from_expr(ctx, gap_factor, var).ok()?;
    if gap.degree() != 1 {
        return None;
    }
    let radicand_slope = radicand.coeffs.get(1).cloned()?;
    if radicand_slope.is_zero() {
        return None;
    }
    let radicand_constant = radicand
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let gap_slope = gap.coeffs.get(1).cloned().unwrap_or_else(BigRational::zero);
    let gap_constant = gap
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let mut radicand_coefficient = gap_slope / radicand_slope;
    let mut offset = gap_constant - radicand_coefficient.clone() * radicand_constant;
    let kernel_scale_factor = if radicand_coefficient.is_negative() && offset.is_negative() {
        radicand_coefficient = -radicand_coefficient;
        offset = -offset;
        -BigRational::one()
    } else {
        BigRational::one()
    };
    if !radicand_coefficient.is_positive() || !offset.is_positive() {
        return None;
    }

    let argument_scale = exact_rational_sqrt(&(radicand_coefficient.clone() / offset.clone()))?;
    let denominator_root = exact_rational_sqrt(&(radicand_coefficient * offset))?;
    Some(ArctanSqrtAffineGapParts {
        argument_scale,
        denominator_root,
        kernel_scale_factor,
    })
}

/// If `h_expr` is `H(sqrt(x))` for `H = exp` (`Pow(E, .)`) or a unary builtin,
/// return `(H(u) rebuilt in the original variable, the sqrt(x) argument)`.
pub(super) fn function_of_sqrt_parts(
    ctx: &mut Context,
    h_expr: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let arg_is_sqrt_var = |ctx: &mut Context, a: ExprId| {
        sqrt_like_radicand(ctx, a).is_some_and(|r| is_var(ctx, r, var))
    };
    let var_expr = ctx.var(var);
    match ctx.get(h_expr).clone() {
        Expr::Pow(base, exp)
            if matches!(ctx.get(base), Expr::Constant(Constant::E))
                && arg_is_sqrt_var(ctx, exp) =>
        {
            let e = ctx.add(Expr::Constant(Constant::E));
            Some((ctx.add(Expr::Pow(e, var_expr)), exp))
        }
        Expr::Function(fn_id, args) if args.len() == 1 && arg_is_sqrt_var(ctx, args[0]) => {
            let builtin = ctx.builtin_of(fn_id)?;
            Some((ctx.call_builtin(builtin, vec![var_expr]), args[0]))
        }
        _ => None,
    }
}

pub(super) fn trig_ratio_square_antiderivative_from_parts(
    ctx: &mut Context,
    parts: TrigRatioSquareParts,
    var: &str,
) -> ExprId {
    let primitive = match parts.builtin {
        BuiltinFn::Tan => ctx.call_builtin(BuiltinFn::Tan, vec![parts.arg]),
        BuiltinFn::Cot => {
            let cot_arg = ctx.call_builtin(BuiltinFn::Cot, vec![parts.arg]);
            ctx.add(Expr::Neg(cot_arg))
        }
        _ => unreachable!("only tan/cot ratio squares have a primitive"),
    };
    let scaled_primitive = if parts.a.is_one() {
        primitive
    } else {
        let a = ctx.add(Expr::Number(parts.a));
        let scaled = ctx.add(Expr::Div(primitive, a));
        if matches!(parts.builtin, BuiltinFn::Cot) {
            cas_ast::hold::wrap_hold(ctx, scaled)
        } else {
            scaled
        }
    };
    let var_expr = ctx.var(var);
    ctx.add(Expr::Sub(scaled_primitive, var_expr))
}

pub(super) fn trig_tan_third_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    a: BigRational,
) -> ExprId {
    // d/du [tan^2(u)/2 + ln|cos(u)|] = tan^3(u).
    let tan_arg = ctx.call_builtin(BuiltinFn::Tan, vec![arg]);
    let two = ctx.num(2);
    let tan_squared = ctx.add(Expr::Pow(tan_arg, two));
    let square = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / (BigRational::from_integer(2.into()) * a.clone()),
        tan_squared,
    );
    let log_raw = trig_abs_log_term(ctx, BuiltinFn::Cos, arg);
    let log_term = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / a,
        log_raw,
    );
    ctx.add(Expr::Add(square, log_term))
}

pub(super) fn trig_cot_third_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    a: BigRational,
) -> ExprId {
    // d/du [-cot^2(u)/2 - ln|sin(u)|] = cot^3(u).
    let cot_arg = ctx.call_builtin(BuiltinFn::Cot, vec![arg]);
    let two = ctx.num(2);
    let cot_squared = ctx.add(Expr::Pow(cot_arg, two));
    let square = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / (BigRational::from_integer(2.into()) * a.clone()),
        cot_squared,
    );
    let log_raw = trig_abs_log_term(ctx, BuiltinFn::Sin, arg);
    let log_term = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / a,
        log_raw,
    );
    ctx.add(Expr::Add(square, log_term))
}

pub(super) fn trig_tan_fifth_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    a: BigRational,
) -> ExprId {
    // d/du [tan^4(u)/4 - tan^2(u)/2 - ln|cos(u)|] = tan^5(u).
    // Built in expanded sin/cos form: Pow(Tan, 4) sends the
    // post-integration rewrite into a non-terminating tan <-> sin/cos
    // ping-pong (cot^4 forms do not loop).
    let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
    let four = ctx.num(4);
    let two = ctx.num(2);
    let sin_fourth = ctx.add(Expr::Pow(sin_arg, four));
    let cos_fourth = ctx.add(Expr::Pow(cos_arg, four));
    let sin_squared = ctx.add(Expr::Pow(sin_arg, two));
    let cos_squared = ctx.add(Expr::Pow(cos_arg, two));
    let tan_fourth = ctx.add(Expr::Div(sin_fourth, cos_fourth));
    let tan_squared = ctx.add(Expr::Div(sin_squared, cos_squared));
    let quartic = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / (BigRational::from_integer(4.into()) * a.clone()),
        tan_fourth,
    );
    let square = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / (BigRational::from_integer(2.into()) * a.clone()),
        tan_squared,
    );
    let log_raw = trig_abs_log_term(ctx, BuiltinFn::Cos, arg);
    let log_term = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / a,
        log_raw,
    );
    let head = ctx.add(Expr::Add(quartic, square));
    ctx.add(Expr::Add(head, log_term))
}

pub(super) fn trig_cot_fifth_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    a: BigRational,
) -> ExprId {
    // d/du [-cot^4(u)/4 + cot^2(u)/2 + ln|sin(u)|] = cot^5(u).
    let cot_arg = ctx.call_builtin(BuiltinFn::Cot, vec![arg]);
    let four = ctx.num(4);
    let two = ctx.num(2);
    let cot_fourth = ctx.add(Expr::Pow(cot_arg, four));
    let cot_squared = ctx.add(Expr::Pow(cot_arg, two));
    let quartic = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / (BigRational::from_integer(4.into()) * a.clone()),
        cot_fourth,
    );
    let square = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / (BigRational::from_integer(2.into()) * a.clone()),
        cot_squared,
    );
    let log_raw = trig_abs_log_term(ctx, BuiltinFn::Sin, arg);
    let log_term = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / a,
        log_raw,
    );
    let head = ctx.add(Expr::Add(quartic, square));
    ctx.add(Expr::Add(head, log_term))
}

pub(super) fn trig_tan_fourth_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    a: BigRational,
    var: &str,
) -> ExprId {
    let tan_arg = ctx.call_builtin(BuiltinFn::Tan, vec![arg]);
    let three = ctx.num(3);
    let tan_cubed = ctx.add(Expr::Pow(tan_arg, three));
    let cubic = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / (BigRational::from_integer(3.into()) * a.clone()),
        tan_cubed,
    );
    let linear_tan = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / a,
        tan_arg,
    );
    let variable = ctx.var(var);
    let tan_terms = ctx.add(Expr::Add(cubic, linear_tan));
    ctx.add(Expr::Add(tan_terms, variable))
}

pub(super) fn trig_cot_fourth_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    a: BigRational,
    var: &str,
) -> ExprId {
    let cot_arg = ctx.call_builtin(BuiltinFn::Cot, vec![arg]);
    let three = ctx.num(3);
    let cot_cubed = ctx.add(Expr::Pow(cot_arg, three));
    let variable = ctx.var(var);

    if a.is_positive() {
        let linear = scale_reciprocal_integration_result_preserving_presentation(
            ctx,
            BigRational::one() / a.clone(),
            cot_arg,
        );
        let cubic = scale_reciprocal_integration_result_preserving_presentation(
            ctx,
            BigRational::one() / (BigRational::from_integer(3.into()) * a),
            cot_cubed,
        );
        let variable_plus_linear = ctx.add(Expr::Add(variable, linear));
        return ctx.add(Expr::Sub(variable_plus_linear, cubic));
    }

    let linear = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / a.clone(),
        cot_arg,
    );
    let cubic = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / (BigRational::from_integer(3.into()) * a),
        cot_cubed,
    );
    let variable_minus_linear = ctx.add(Expr::Sub(variable, linear));
    ctx.add(Expr::Add(variable_minus_linear, cubic))
}

pub(super) fn trig_tan_sixth_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    a: BigRational,
    var: &str,
) -> ExprId {
    let tan_arg = ctx.call_builtin(BuiltinFn::Tan, vec![arg]);
    let three = ctx.num(3);
    let five = ctx.num(5);
    let tan_cubed = ctx.add(Expr::Pow(tan_arg, three));
    let tan_fifth = ctx.add(Expr::Pow(tan_arg, five));
    let fifth = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / (BigRational::from_integer(5.into()) * a.clone()),
        tan_fifth,
    );
    let cubic = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / (BigRational::from_integer(3.into()) * a.clone()),
        tan_cubed,
    );
    let linear = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / a,
        tan_arg,
    );
    let first_two = ctx.add(Expr::Add(fifth, cubic));
    let tan_terms = ctx.add(Expr::Add(first_two, linear));
    let variable = ctx.var(var);
    ctx.add(Expr::Sub(tan_terms, variable))
}

pub(super) fn trig_tan_eighth_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    a: BigRational,
    var: &str,
) -> ExprId {
    let tan_arg = ctx.call_builtin(BuiltinFn::Tan, vec![arg]);
    let three = ctx.num(3);
    let five = ctx.num(5);
    let seven = ctx.num(7);
    let tan_cubed = ctx.add(Expr::Pow(tan_arg, three));
    let tan_fifth = ctx.add(Expr::Pow(tan_arg, five));
    let tan_seventh = ctx.add(Expr::Pow(tan_arg, seven));
    let seventh = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / (BigRational::from_integer(7.into()) * a.clone()),
        tan_seventh,
    );
    let fifth = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / (BigRational::from_integer(5.into()) * a.clone()),
        tan_fifth,
    );
    let cubic = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / (BigRational::from_integer(3.into()) * a.clone()),
        tan_cubed,
    );
    let linear = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / a,
        tan_arg,
    );
    let first_two = ctx.add(Expr::Add(seventh, fifth));
    let first_three = ctx.add(Expr::Add(first_two, cubic));
    let tan_terms = ctx.add(Expr::Add(first_three, linear));
    let variable = ctx.var(var);
    ctx.add(Expr::Add(tan_terms, variable))
}

pub(super) fn trig_cot_sixth_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    a: BigRational,
    var: &str,
) -> ExprId {
    let cot_arg = ctx.call_builtin(BuiltinFn::Cot, vec![arg]);
    let three = ctx.num(3);
    let five = ctx.num(5);
    let cot_cubed = ctx.add(Expr::Pow(cot_arg, three));
    let cot_fifth = ctx.add(Expr::Pow(cot_arg, five));
    let fifth = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / (BigRational::from_integer(5.into()) * a.clone()),
        cot_fifth,
    );
    let cubic = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / (BigRational::from_integer(3.into()) * a.clone()),
        cot_cubed,
    );
    let linear = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / a,
        cot_arg,
    );
    let first_two = ctx.add(Expr::Add(fifth, cubic));
    let cot_terms = ctx.add(Expr::Add(first_two, linear));
    let variable = ctx.var(var);
    ctx.add(Expr::Sub(cot_terms, variable))
}

pub(super) fn trig_cot_eighth_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    a: BigRational,
    var: &str,
) -> ExprId {
    let cot_arg = ctx.call_builtin(BuiltinFn::Cot, vec![arg]);
    let three = ctx.num(3);
    let five = ctx.num(5);
    let seven = ctx.num(7);
    let cot_cubed = ctx.add(Expr::Pow(cot_arg, three));
    let cot_fifth = ctx.add(Expr::Pow(cot_arg, five));
    let cot_seventh = ctx.add(Expr::Pow(cot_arg, seven));
    let seventh = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / (BigRational::from_integer(7.into()) * a.clone()),
        cot_seventh,
    );
    let fifth = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / (BigRational::from_integer(5.into()) * a.clone()),
        cot_fifth,
    );
    let cubic = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / (BigRational::from_integer(3.into()) * a.clone()),
        cot_cubed,
    );
    let linear = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / a,
        cot_arg,
    );
    let first_two = ctx.add(Expr::Add(seventh, fifth));
    let first_three = ctx.add(Expr::Add(first_two, cubic));
    let cot_terms = ctx.add(Expr::Add(first_three, linear));
    let variable = ctx.var(var);
    ctx.add(Expr::Add(cot_terms, variable))
}

pub(super) fn trig_sec_third_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    a: BigRational,
) -> ExprId {
    // d/du [sec(u)tan(u) + ln|sec(u)+tan(u)|] = 2 sec^3(u).
    let sec_arg = ctx.call_builtin(BuiltinFn::Sec, vec![arg]);
    let tan_arg = ctx.call_builtin(BuiltinFn::Tan, vec![arg]);
    let product = ctx.add(Expr::Mul(sec_arg, tan_arg));
    let log_term = reciprocal_trig_abs_log_term(ctx, BuiltinFn::Sec, arg);
    let sum = ctx.add(Expr::Add(product, log_term));
    scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / (BigRational::from_integer(2.into()) * a),
        sum,
    )
}

pub(super) fn trig_csc_third_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    a: BigRational,
) -> ExprId {
    // d/du [-csc(u)cot(u) + ln|csc(u)-cot(u)|] = 2 csc^3(u).
    let csc_arg = ctx.call_builtin(BuiltinFn::Csc, vec![arg]);
    let cot_arg = ctx.call_builtin(BuiltinFn::Cot, vec![arg]);
    let product_raw = ctx.add(Expr::Mul(csc_arg, cot_arg));
    let product = ctx.add(Expr::Neg(product_raw));
    let log_term = reciprocal_trig_abs_log_term(ctx, BuiltinFn::Csc, arg);
    let sum = ctx.add(Expr::Add(product, log_term));
    scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / (BigRational::from_integer(2.into()) * a),
        sum,
    )
}

pub(super) fn trig_sec_fifth_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    a: BigRational,
) -> ExprId {
    // sec^5 reduction: sec^3(u)tan(u)/4 + (3/4) * integral of sec^3.
    let sec_arg = ctx.call_builtin(BuiltinFn::Sec, vec![arg]);
    let tan_arg = ctx.call_builtin(BuiltinFn::Tan, vec![arg]);
    let three = ctx.num(3);
    let sec_cubed = ctx.add(Expr::Pow(sec_arg, three));
    let product_raw = ctx.add(Expr::Mul(sec_cubed, tan_arg));
    let leading = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / (BigRational::from_integer(4.into()) * a.clone()),
        product_raw,
    );
    let lower = trig_sec_third_antiderivative_from_parts(ctx, arg, a);
    let scaled_lower = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::new(3.into(), 4.into()),
        lower,
    );
    ctx.add(Expr::Add(leading, scaled_lower))
}

pub(super) fn trig_csc_fifth_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    a: BigRational,
) -> ExprId {
    // csc^5 reduction: -csc^3(u)cot(u)/4 + (3/4) * integral of csc^3.
    let csc_arg = ctx.call_builtin(BuiltinFn::Csc, vec![arg]);
    let cot_arg = ctx.call_builtin(BuiltinFn::Cot, vec![arg]);
    let three = ctx.num(3);
    let csc_cubed = ctx.add(Expr::Pow(csc_arg, three));
    let product_raw = ctx.add(Expr::Mul(csc_cubed, cot_arg));
    let product = ctx.add(Expr::Neg(product_raw));
    let leading = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / (BigRational::from_integer(4.into()) * a.clone()),
        product,
    );
    let lower = trig_csc_third_antiderivative_from_parts(ctx, arg, a);
    let scaled_lower = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::new(3.into(), 4.into()),
        lower,
    );
    ctx.add(Expr::Add(leading, scaled_lower))
}

pub(super) fn trig_sec_fourth_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    a: BigRational,
) -> ExprId {
    let tan_arg = ctx.call_builtin(BuiltinFn::Tan, vec![arg]);
    let three = ctx.num(3);
    let tan_cubed = ctx.add(Expr::Pow(tan_arg, three));
    let linear = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / a.clone(),
        tan_arg,
    );
    let cubic = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / (BigRational::from_integer(3.into()) * a),
        tan_cubed,
    );
    ctx.add(Expr::Add(linear, cubic))
}

pub(super) fn trig_sec_sixth_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    a: BigRational,
) -> ExprId {
    let tan_arg = ctx.call_builtin(BuiltinFn::Tan, vec![arg]);
    let three = ctx.num(3);
    let five = ctx.num(5);
    let tan_cubed = ctx.add(Expr::Pow(tan_arg, three));
    let tan_fifth = ctx.add(Expr::Pow(tan_arg, five));
    let linear = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / a.clone(),
        tan_arg,
    );
    let cubic = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::from_integer(2.into()) / (BigRational::from_integer(3.into()) * a.clone()),
        tan_cubed,
    );
    let fifth = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / (BigRational::from_integer(5.into()) * a),
        tan_fifth,
    );
    let first_two = ctx.add(Expr::Add(linear, cubic));
    ctx.add(Expr::Add(first_two, fifth))
}

pub(super) fn trig_sec_eighth_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    a: BigRational,
) -> ExprId {
    let tan_arg = ctx.call_builtin(BuiltinFn::Tan, vec![arg]);
    let three = ctx.num(3);
    let five = ctx.num(5);
    let seven = ctx.num(7);
    let tan_cubed = ctx.add(Expr::Pow(tan_arg, three));
    let tan_fifth = ctx.add(Expr::Pow(tan_arg, five));
    let tan_seventh = ctx.add(Expr::Pow(tan_arg, seven));
    let linear = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / a.clone(),
        tan_arg,
    );
    let cubic = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / a.clone(),
        tan_cubed,
    );
    let fifth = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::from_integer(3.into()) / (BigRational::from_integer(5.into()) * a.clone()),
        tan_fifth,
    );
    let seventh = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        BigRational::one() / (BigRational::from_integer(7.into()) * a),
        tan_seventh,
    );
    let first_two = ctx.add(Expr::Add(linear, cubic));
    let first_three = ctx.add(Expr::Add(first_two, fifth));
    ctx.add(Expr::Add(first_three, seventh))
}

pub(super) fn reciprocal_trig_power_affine_parts(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
    builtin: BuiltinFn,
    power: i64,
) -> Option<ReciprocalTrigPowerAffineParts> {
    if !is_number(ctx, exp, power) {
        return None;
    }
    let arg = reciprocal_trig_power_affine_arg(ctx, base, var, builtin)?;
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    (!a.is_zero()).then_some(ReciprocalTrigPowerAffineParts { arg, a })
}

pub(super) fn trig_csc_fourth_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    a: BigRational,
) -> ExprId {
    let cot_arg = ctx.call_builtin(BuiltinFn::Cot, vec![arg]);
    let three = ctx.num(3);
    let cot_cubed = ctx.add(Expr::Pow(cot_arg, three));
    if a.is_positive() {
        let linear = scale_reciprocal_integration_result_preserving_presentation(
            ctx,
            BigRational::one() / a.clone(),
            cot_arg,
        );
        let cubic = scale_reciprocal_integration_result_preserving_presentation(
            ctx,
            -BigRational::one() / (BigRational::from_integer(3.into()) * a),
            cot_cubed,
        );
        return ctx.add(Expr::Sub(cubic, linear));
    }

    let linear = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / a.clone(),
        cot_arg,
    );
    let cubic = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / (BigRational::from_integer(3.into()) * a),
        cot_cubed,
    );
    ctx.add(Expr::Add(linear, cubic))
}

pub(super) fn trig_csc_sixth_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    a: BigRational,
) -> ExprId {
    let cot_arg = ctx.call_builtin(BuiltinFn::Cot, vec![arg]);
    let three = ctx.num(3);
    let five = ctx.num(5);
    let cot_cubed = ctx.add(Expr::Pow(cot_arg, three));
    let cot_fifth = ctx.add(Expr::Pow(cot_arg, five));
    let linear = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / a.clone(),
        cot_arg,
    );
    let cubic = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::from_integer(2.into()) / (BigRational::from_integer(3.into()) * a.clone()),
        cot_cubed,
    );
    let fifth = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / (BigRational::from_integer(5.into()) * a),
        cot_fifth,
    );
    let first_two = ctx.add(Expr::Add(linear, cubic));
    ctx.add(Expr::Add(first_two, fifth))
}

pub(super) fn trig_csc_eighth_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    a: BigRational,
) -> ExprId {
    let cot_arg = ctx.call_builtin(BuiltinFn::Cot, vec![arg]);
    let three = ctx.num(3);
    let five = ctx.num(5);
    let seven = ctx.num(7);
    let cot_cubed = ctx.add(Expr::Pow(cot_arg, three));
    let cot_fifth = ctx.add(Expr::Pow(cot_arg, five));
    let cot_seventh = ctx.add(Expr::Pow(cot_arg, seven));
    let linear = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / a.clone(),
        cot_arg,
    );
    let cubic = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / a.clone(),
        cot_cubed,
    );
    let fifth = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::from_integer(3.into()) / (BigRational::from_integer(5.into()) * a.clone()),
        cot_fifth,
    );
    let seventh = scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        -BigRational::one() / (BigRational::from_integer(7.into()) * a),
        cot_seventh,
    );
    let first_two = ctx.add(Expr::Add(linear, cubic));
    let first_three = ctx.add(Expr::Add(first_two, fifth));
    ctx.add(Expr::Add(first_three, seventh))
}

pub(super) fn trig_ratio_square_quotient_parts(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<TrigRatioSquareParts> {
    let matched = if let (Some(num_arg), Some(den_arg)) = (
        squared_unary_builtin_arg(ctx, num, BuiltinFn::Sin),
        squared_unary_builtin_arg(ctx, den, BuiltinFn::Cos),
    ) {
        (BuiltinFn::Tan, num_arg, den_arg)
    } else if let (Some(num_arg), Some(den_arg)) = (
        squared_unary_builtin_arg(ctx, num, BuiltinFn::Cos),
        squared_unary_builtin_arg(ctx, den, BuiltinFn::Sin),
    ) {
        (BuiltinFn::Cot, num_arg, den_arg)
    } else {
        return None;
    };

    let (builtin, num_arg, den_arg) = matched;
    if compare_expr(ctx, num_arg, den_arg) != Ordering::Equal {
        return None;
    }
    let (a, _) = get_linear_coeffs(ctx, num_arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    (!a.is_zero()).then_some(TrigRatioSquareParts {
        builtin,
        arg: num_arg,
        a,
    })
}

pub(super) fn trig_tan_fourth_quotient_parts(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<TrigPowerQuotientParts> {
    trig_power_quotient_parts(ctx, num, den, var, BuiltinFn::Sin, BuiltinFn::Cos, 4)
}

pub(super) fn trig_cot_fourth_quotient_parts(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<TrigPowerQuotientParts> {
    trig_power_quotient_parts(ctx, num, den, var, BuiltinFn::Cos, BuiltinFn::Sin, 4)
}

pub(super) fn trig_tan_sixth_quotient_parts(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<TrigPowerQuotientParts> {
    trig_power_quotient_parts(ctx, num, den, var, BuiltinFn::Sin, BuiltinFn::Cos, 6)
}

pub(super) fn trig_tan_eighth_quotient_parts(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<TrigPowerQuotientParts> {
    trig_power_quotient_parts(ctx, num, den, var, BuiltinFn::Sin, BuiltinFn::Cos, 8)
}

pub(super) fn trig_cot_sixth_quotient_parts(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<TrigPowerQuotientParts> {
    trig_power_quotient_parts(ctx, num, den, var, BuiltinFn::Cos, BuiltinFn::Sin, 6)
}

pub(super) fn trig_cot_eighth_quotient_parts(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<TrigPowerQuotientParts> {
    trig_power_quotient_parts(ctx, num, den, var, BuiltinFn::Cos, BuiltinFn::Sin, 8)
}

pub(super) fn trig_power_quotient_parts(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
    numerator_builtin: BuiltinFn,
    denominator_builtin: BuiltinFn,
    power: i64,
) -> Option<TrigPowerQuotientParts> {
    let num_arg = powered_unary_builtin_arg(ctx, num, numerator_builtin, power)?;
    let den_arg = powered_unary_builtin_arg(ctx, den, denominator_builtin, power)?;
    if compare_expr(ctx, num_arg, den_arg) != Ordering::Equal {
        return None;
    }
    let (a, _) = get_linear_coeffs(ctx, num_arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    (!a.is_zero()).then_some(TrigPowerQuotientParts { arg: num_arg, a })
}

pub(super) fn reciprocal_trig_power_quotient_parts(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
    denominator_builtin: BuiltinFn,
    power: i64,
) -> Option<ReciprocalTrigPowerQuotientParts> {
    let arg = reciprocal_trig_power_quotient_arg(ctx, num, den, var, denominator_builtin, power)?;
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    (!a.is_zero()).then_some(ReciprocalTrigPowerQuotientParts { arg, a })
}

pub(super) fn trig_ratio_power_reciprocal_square_parts(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
) -> Option<(BigRational, BuiltinFn, ExprId, i64, BigRational)> {
    if let Some((den_builtin, den_arg)) = reciprocal_trig_square_parts(ctx, den) {
        let (num_builtin, primitive_builtin, sign) = match den_builtin {
            BuiltinFn::Cos => (BuiltinFn::Tan, BuiltinFn::Tan, BigRational::one()),
            BuiltinFn::Sin => (BuiltinFn::Cot, BuiltinFn::Cot, -BigRational::one()),
            _ => return None,
        };
        let (scale, num_arg, power) = trig_ratio_power_factor(ctx, num, num_builtin, 1, 5)?;
        if compare_expr(ctx, den_arg, num_arg) != Ordering::Equal {
            return None;
        }
        return Some((scale, primitive_builtin, den_arg, power, sign));
    }

    let (den_builtin, den_arg, den_power) = trig_power_base(ctx, den, 3, 7)?;
    let (num_builtin, primitive_builtin, sign) = match den_builtin {
        BuiltinFn::Cos => (BuiltinFn::Sin, BuiltinFn::Tan, BigRational::one()),
        BuiltinFn::Sin => (BuiltinFn::Cos, BuiltinFn::Cot, -BigRational::one()),
        _ => return None,
    };
    let (scale, num_arg, num_power) = trig_ratio_power_factor(ctx, num, num_builtin, 1, 5)?;
    if den_power != num_power + 2 {
        return None;
    }
    if compare_expr(ctx, den_arg, num_arg) != Ordering::Equal {
        return None;
    }

    Some((scale, primitive_builtin, den_arg, num_power, sign))
}

pub(super) fn hyperbolic_log_derivative_ratio_parts(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let (den_builtin, arg) = match ctx.get(den) {
        Expr::Function(fn_id, args) if args.len() == 1 => (ctx.builtin_of(*fn_id)?, args[0]),
        _ => return None,
    };
    let numerator_builtin = match den_builtin {
        BuiltinFn::Cosh => BuiltinFn::Sinh,
        BuiltinFn::Sinh => BuiltinFn::Cosh,
        _ => return None,
    };
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let cofactor = product_cofactor_excluding_unary_builtin_arg(
        ctx,
        num,
        numerator_builtin,
        |ctx, numerator_arg| compare_expr(ctx, numerator_arg, arg) == Ordering::Equal,
    )?;

    let scale = symbolic_linear_cofactor_scale_expr(ctx, cofactor, arg, var)?;

    Some((den, scale))
}

pub(super) fn hyperbolic_log_derivative_ratio_antiderivative_from_parts(
    ctx: &mut Context,
    den: ExprId,
    scale: ExprId,
) -> ExprId {
    let log_arg = cas_ast::hold::wrap_hold(ctx, den);
    let log_abs = ln_abs(ctx, log_arg);
    scale_expr_reciprocal_integration_result(ctx, scale, log_abs)
}

pub(super) fn hyperbolic_tanh_log_cosh_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let factors = mul_leaves(ctx, expr);
    let (tanh_index, arg) = indexed_hyperbolic_tangent_factor_arg(ctx, &factors)?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let cofactor = factor_product_excluding_index(ctx, &factors, tanh_index);
    let scale = symbolic_linear_cofactor_scale_expr(ctx, cofactor, arg, var)?;
    if is_number(ctx, scale, 0) {
        return None;
    }

    Some((arg, scale))
}

pub(super) fn hyperbolic_tanh_reciprocal_log_sinh_parts(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let arg = hyperbolic_tangent_arg(ctx, den)?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let scale = symbolic_linear_cofactor_scale_expr(ctx, num, arg, var)?;
    if is_number(ctx, scale, 0) {
        return None;
    }

    Some((arg, scale))
}

pub(super) fn hyperbolic_tanh_reciprocal_log_sinh_antiderivative_from_parts(
    ctx: &mut Context,
    arg: ExprId,
    scale: ExprId,
) -> ExprId {
    let sinh_arg = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
    let log_abs = ln_abs(ctx, sinh_arg);
    scale_expr_reciprocal_integration_result(ctx, scale, log_abs)
}

pub(super) fn hyperbolic_reciprocal_derivative_parts(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<HyperbolicReciprocalDerivativeParts> {
    let (den_builtin, arg) = reciprocal_hyperbolic_square_parts(ctx, den)?;
    let policy = hyperbolic_reciprocal_derivative_policy(den_builtin)?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let cofactor = product_cofactor_excluding_unary_builtin_arg(
        ctx,
        num,
        policy.numerator_builtin,
        |ctx, numerator_arg| same_structural_or_linear_arg(ctx, numerator_arg, arg, var),
    )?;

    let scale = symbolic_linear_cofactor_scale_expr(ctx, cofactor, arg, var)?;
    if is_number(ctx, scale, 0) {
        return None;
    }

    Some(HyperbolicReciprocalDerivativeParts {
        denominator_builtin: den_builtin,
        arg,
        scale,
    })
}

pub(super) fn hyperbolic_reciprocal_derivative_antiderivative_from_parts(
    ctx: &mut Context,
    parts: HyperbolicReciprocalDerivativeParts,
) -> Option<ExprId> {
    let policy = hyperbolic_reciprocal_derivative_policy(parts.denominator_builtin)?;
    let integral = build_hyperbolic_reciprocal_derivative_integral(ctx, policy, parts.arg);
    Some(scale_expr_reciprocal_integration_result(
        ctx,
        parts.scale,
        integral,
    ))
}

pub(super) fn polynomial_reciprocal_trig_square_required_nonzero_from_parts(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    build_reciprocal_trig_denominator_nonzero_condition(ctx, builtin, arg)
}

pub(super) fn add_linear_parts(ctx: &mut Context, left: ExprId, right: ExprId) -> ExprId {
    if is_number(ctx, left, 0) {
        right
    } else if is_number(ctx, right, 0) {
        left
    } else if let (Expr::Number(left_value), Expr::Number(right_value)) =
        (ctx.get(left), ctx.get(right))
    {
        ctx.add(Expr::Number(left_value.clone() + right_value.clone()))
    } else {
        ctx.add(Expr::Add(left, right))
    }
}

pub(super) fn sub_linear_parts(ctx: &mut Context, left: ExprId, right: ExprId) -> ExprId {
    if is_number(ctx, right, 0) {
        left
    } else if let (Expr::Number(left_value), Expr::Number(right_value)) =
        (ctx.get(left), ctx.get(right))
    {
        ctx.add(Expr::Number(left_value.clone() - right_value.clone()))
    } else {
        ctx.add(Expr::Sub(left, right))
    }
}

pub(super) fn polynomial_trig_reciprocal_derivative_parts(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<(BuiltinFn, ExprId, ExprId)> {
    let (den_builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    let policy = reciprocal_trig_derivative_policy(den_builtin)?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let cofactor =
        polynomial_trig_reciprocal_derivative_cofactor(ctx, num, policy.numerator_builtin(), arg)?;
    Some((den_builtin, arg, cofactor))
}

pub(super) fn polynomial_trig_reciprocal_derivative_antiderivative_from_parts(
    ctx: &mut Context,
    den_builtin: BuiltinFn,
    arg: ExprId,
    cofactor: ExprId,
    var: &str,
) -> Option<ExprId> {
    let integral = trig_reciprocal_derivative_base_integral(ctx, den_builtin, arg)?;
    let preserve_shifted_symbolic_presentation =
        additive_var_dependent_part(ctx, arg, var).is_some();

    if let Some((scale, preserve_symbolic_linear_presentation)) =
        trig_reciprocal_derivative_cofactor_scale(ctx, cofactor, arg, var)
    {
        if scale.is_zero() {
            return None;
        }

        return Some(scale_reciprocal_integration_result_with_unit_presentation(
            ctx,
            scale,
            integral,
            preserve_symbolic_linear_presentation || preserve_shifted_symbolic_presentation,
        ));
    }

    let scale = symbolic_linear_cofactor_scale_expr(ctx, cofactor, arg, var)?;
    if is_number(ctx, scale, 0) {
        return None;
    }
    Some(
        scale_expr_reciprocal_integration_result_preserving_presentation(
            ctx, scale, integral, true,
        ),
    )
}

pub(super) fn sqrt_trig_reciprocal_derivative_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<SqrtTrigReciprocalDerivativeParts> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let mut parts = sqrt_trig_reciprocal_derivative_parts(ctx, inner, var)?;
            parts.scale = negate_scalar_expr(ctx, parts.scale);
            return Some(parts);
        }
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };

    let numerator_factors = mul_leaves(ctx, num);
    let denominator_factors = mul_leaves(ctx, den);

    if let Some((den_builtin, arg, reciprocal_index, derivative_index, derivative_sign)) =
        sqrt_trig_reciprocal_derivative_raw_numerator_parts(ctx, &numerator_factors)
    {
        let remaining_numerator =
            factors_excluding_two_indices(&numerator_factors, reciprocal_index, derivative_index);
        let scale = sqrt_polynomial_derivative_quotient_scale_expr(
            ctx,
            &remaining_numerator,
            &denominator_factors,
            arg,
            var,
        )?;
        let scale = scale_rational_term(ctx, derivative_sign, scale);
        return finish_sqrt_trig_reciprocal_derivative_parts(ctx, den_builtin, arg, scale, var);
    }

    let (denominator_index, (den_builtin, arg)) =
        indexed_reciprocal_trig_square_parts(ctx, &denominator_factors)?;
    let policy = reciprocal_trig_derivative_policy(den_builtin)?;

    let (numerator_index, numerator_sign) = indexed_signed_matching_unary_factor(
        ctx,
        &numerator_factors,
        policy.numerator_builtin(),
        arg,
    )?;

    let remaining_numerator = factors_excluding_index(&numerator_factors, numerator_index);
    let remaining_denominator = factors_excluding_index(&denominator_factors, denominator_index);
    let scale = sqrt_polynomial_derivative_quotient_scale_expr(
        ctx,
        &remaining_numerator,
        &remaining_denominator,
        arg,
        var,
    )?;
    let scale = scale_rational_term(ctx, numerator_sign, scale);
    finish_sqrt_trig_reciprocal_derivative_parts(ctx, den_builtin, arg, scale, var)
}

fn finish_sqrt_trig_reciprocal_derivative_parts(
    ctx: &mut Context,
    denominator_builtin: BuiltinFn,
    arg: ExprId,
    scale: ExprId,
    var: &str,
) -> Option<SqrtTrigReciprocalDerivativeParts> {
    if is_number(ctx, scale, 0) {
        return None;
    }

    let (radicand, _) = sqrt_chain_argument_derivative_parts(ctx, arg, var)?;
    Some(SqrtTrigReciprocalDerivativeParts {
        denominator_builtin,
        arg,
        radicand,
        scale,
    })
}

fn sqrt_trig_reciprocal_derivative_raw_numerator_parts(
    ctx: &Context,
    numerator_factors: &[ExprId],
) -> Option<(BuiltinFn, ExprId, usize, usize, BigRational)> {
    for (reciprocal_index, factor) in numerator_factors.iter().enumerate() {
        let (policy, arg) = match ctx.get(*factor) {
            Expr::Function(fn_id, args) if args.len() == 1 => {
                let Some(policy) = ctx
                    .builtin_of(*fn_id)
                    .and_then(reciprocal_trig_derivative_policy_from_reciprocal)
                else {
                    continue;
                };
                (policy, args[0])
            }
            _ => continue,
        };

        for (derivative_index, factor) in numerator_factors.iter().enumerate() {
            if derivative_index == reciprocal_index {
                continue;
            }
            let Some((derivative_arg, derivative_sign)) =
                signed_unary_builtin_arg(ctx, *factor, policy.derivative_builtin())
            else {
                continue;
            };
            if compare_expr(ctx, derivative_arg, arg) == Ordering::Equal {
                return Some((
                    policy.denominator_builtin(),
                    arg,
                    reciprocal_index,
                    derivative_index,
                    derivative_sign,
                ));
            }
        }
    }

    None
}

pub(super) fn sqrt_trig_log_derivative_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<SqrtTrigLogDerivativeParts> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let mut parts = sqrt_trig_log_derivative_parts(ctx, inner, var)?;
            parts.scale = -parts.scale;
            return Some(parts);
        }
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };

    let numerator_factors = mul_leaves(ctx, num);
    let denominator_factors = mul_leaves(ctx, den);
    if let Some((den_builtin, arg, numerator_index, numerator_sign)) =
        indexed_trig_log_derivative_raw_numerator_factor(ctx, &numerator_factors)
    {
        let remaining_numerator = factors_excluding_index(&numerator_factors, numerator_index);
        let scale = sqrt_polynomial_derivative_quotient_scale(
            ctx,
            &remaining_numerator,
            &denominator_factors,
            arg,
            var,
        )? * numerator_sign;
        return finish_sqrt_trig_log_derivative_parts(ctx, den_builtin, arg, scale, var);
    }

    let (denominator_index, (den_builtin, arg)) =
        indexed_reciprocal_trig_denominator_call(ctx, &denominator_factors)?;
    let numerator_index =
        indexed_trig_log_derivative_numerator_factor(ctx, &numerator_factors, den_builtin, arg)?;

    let remaining_numerator = factors_excluding_index(&numerator_factors, numerator_index);
    let remaining_denominator = factors_excluding_index(&denominator_factors, denominator_index);
    let scale = sqrt_polynomial_derivative_quotient_scale(
        ctx,
        &remaining_numerator,
        &remaining_denominator,
        arg,
        var,
    )?;
    finish_sqrt_trig_log_derivative_parts(ctx, den_builtin, arg, scale, var)
}

fn finish_sqrt_trig_log_derivative_parts(
    ctx: &mut Context,
    denominator_builtin: BuiltinFn,
    arg: ExprId,
    scale: BigRational,
    var: &str,
) -> Option<SqrtTrigLogDerivativeParts> {
    if scale.is_zero() {
        return None;
    }

    let (radicand, _) = sqrt_chain_argument_derivative_parts(ctx, arg, var)?;
    Some(SqrtTrigLogDerivativeParts {
        denominator_builtin,
        arg,
        radicand,
        scale,
    })
}

pub(super) fn sqrt_reciprocal_trig_log_derivative_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<SqrtReciprocalTrigLogDerivativeParts> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let mut parts = sqrt_reciprocal_trig_log_derivative_parts(ctx, inner, var)?;
            parts.scale = -parts.scale;
            return Some(parts);
        }
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };

    let numerator_factors = mul_leaves(ctx, num);
    let denominator_factors = mul_leaves(ctx, den);
    let (denominator_index, (den_builtin, arg)) =
        indexed_reciprocal_trig_denominator_call(ctx, &denominator_factors)?;

    let radicand = sqrt_like_radicand(ctx, arg)?;
    let remaining_denominator = factors_excluding_index(&denominator_factors, denominator_index);
    let scale = sqrt_polynomial_derivative_quotient_scale(
        ctx,
        &numerator_factors,
        &remaining_denominator,
        arg,
        var,
    )?;
    if scale.is_zero() {
        return None;
    }

    Some(SqrtReciprocalTrigLogDerivativeParts {
        denominator_builtin: den_builtin,
        arg,
        radicand,
        scale,
    })
}

pub(super) fn sqrt_hyperbolic_log_derivative_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<SqrtHyperbolicLogDerivativeParts> {
    let (numerator_factors, denominator_factors) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (mul_leaves(ctx, num), mul_leaves(ctx, den)),
        _ => (mul_leaves(ctx, expr), Default::default()),
    };

    if let Some((idx, arg)) = indexed_hyperbolic_tangent_factor_arg(ctx, &numerator_factors) {
        let remaining_numerator = factors_excluding_index(&numerator_factors, idx);
        let scale = sqrt_polynomial_derivative_quotient_scale(
            ctx,
            &remaining_numerator,
            &denominator_factors,
            arg,
            var,
        )?;
        return finish_sqrt_hyperbolic_log_derivative_parts(ctx, BuiltinFn::Cosh, arg, scale, var);
    }

    if let Some((idx, arg)) = indexed_hyperbolic_tangent_factor_arg(ctx, &denominator_factors) {
        let remaining_denominator = factors_excluding_index(&denominator_factors, idx);
        let scale = sqrt_polynomial_derivative_quotient_scale(
            ctx,
            &numerator_factors,
            &remaining_denominator,
            arg,
            var,
        )?;
        return finish_sqrt_hyperbolic_log_derivative_parts(ctx, BuiltinFn::Sinh, arg, scale, var);
    }

    None
}

fn finish_sqrt_hyperbolic_log_derivative_parts(
    ctx: &mut Context,
    log_builtin: BuiltinFn,
    arg: ExprId,
    scale: BigRational,
    var: &str,
) -> Option<SqrtHyperbolicLogDerivativeParts> {
    if scale.is_zero() {
        return None;
    }

    let (radicand, _) = sqrt_chain_argument_derivative_parts(ctx, arg, var)?;
    Some(SqrtHyperbolicLogDerivativeParts {
        log_builtin,
        arg,
        radicand,
        scale,
    })
}

pub(super) fn sqrt_hyperbolic_reciprocal_square_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<SqrtHyperbolicReciprocalSquareParts> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };

    let numerator_factors = mul_leaves(ctx, num);
    let denominator_factors = mul_leaves(ctx, den);
    let (denominator_index, (den_builtin, arg)) =
        indexed_reciprocal_hyperbolic_square_parts(ctx, &denominator_factors)?;

    let remaining_denominator = factors_excluding_index(&denominator_factors, denominator_index);
    let scale = sqrt_polynomial_derivative_quotient_scale_expr(
        ctx,
        &numerator_factors,
        &remaining_denominator,
        arg,
        var,
    )?;
    finish_sqrt_hyperbolic_reciprocal_parts(ctx, den_builtin, arg, scale, var)
}

pub(super) fn sqrt_hyperbolic_reciprocal_derivative_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<SqrtHyperbolicReciprocalDerivativeParts> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };

    let numerator_factors = mul_leaves(ctx, num);
    let denominator_factors = mul_leaves(ctx, den);
    let (denominator_index, (den_builtin, arg)) =
        indexed_reciprocal_hyperbolic_square_parts(ctx, &denominator_factors)?;

    let numerator_index = indexed_hyperbolic_reciprocal_derivative_numerator_factor(
        ctx,
        &numerator_factors,
        den_builtin,
        arg,
    )?;

    let remaining_numerator = factors_excluding_index(&numerator_factors, numerator_index);
    let remaining_denominator = factors_excluding_index(&denominator_factors, denominator_index);
    let scale = sqrt_polynomial_derivative_quotient_scale_expr(
        ctx,
        &remaining_numerator,
        &remaining_denominator,
        arg,
        var,
    )?;
    finish_sqrt_hyperbolic_reciprocal_parts(ctx, den_builtin, arg, scale, var)
}

fn finish_sqrt_hyperbolic_reciprocal_parts(
    ctx: &mut Context,
    den_builtin: BuiltinFn,
    arg: ExprId,
    scale: ExprId,
    var: &str,
) -> Option<SqrtHyperbolicReciprocalParts> {
    if is_number(ctx, scale, 0) {
        return None;
    }

    let (radicand, _) = sqrt_chain_argument_derivative_parts(ctx, arg, var)?;
    Some(SqrtHyperbolicReciprocalParts {
        denominator_builtin: den_builtin,
        arg,
        radicand,
        scale,
    })
}

pub(super) fn polynomial_trig_reciprocal_factor_derivative_parts(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<(BuiltinFn, ExprId, BigRational)> {
    let (den_builtin, arg) = reciprocal_trig_denominator_call(ctx, den)?;
    let policy = reciprocal_trig_derivative_policy(den_builtin)?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let cofactor = product_cofactor_excluding_unary_builtin_arg(
        ctx,
        num,
        policy.derivative_builtin(),
        |ctx, numerator_arg| compare_expr(ctx, numerator_arg, arg) == Ordering::Equal,
    )?;

    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    Some((den_builtin, arg, scale))
}

pub(super) fn polynomial_trig_reciprocal_derivative_required_nonzero_from_parts(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, arg, cofactor) =
        polynomial_trig_reciprocal_derivative_parts(ctx, num, den, var)?;
    trig_reciprocal_derivative_cofactor_is_nonzero(ctx, cofactor, arg, var)?;

    build_reciprocal_trig_denominator_nonzero_condition(ctx, den_builtin, arg)
}

pub(super) fn sqrt_hyperbolic_reciprocal_parts_required_nonzero(
    ctx: &mut Context,
    parts: &SqrtHyperbolicReciprocalParts,
) -> Option<ExprId> {
    build_hyperbolic_denominator_nonzero_condition(ctx, parts.denominator_builtin, parts.arg)
}

pub(crate) fn positive_constant_radius_quadratic_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<PositiveConstantRadiusQuadraticParts> {
    let (arg, radius) = positive_square_constant_plus_square_arg(ctx, expr, var)
        .or_else(|| positive_square_constant_plus_expanded_square_arg(ctx, expr, var))?;
    let (linear_arg, slope) = nonzero_linear_arg_and_slope(ctx, arg, var)?;
    let (arctan_arg, arctan_scale) =
        arctan_positive_quadratic_arg_and_scale_from_linear(ctx, linear_arg, radius, slope);

    Some(PositiveConstantRadiusQuadraticParts {
        linear_arg,
        slope,
        arctan_arg,
        arctan_scale,
    })
}

pub(super) fn compact_arctan_by_parts_subtraction(
    ctx: &mut Context,
    v_poly: &Polynomial,
    arctan: ExprId,
    target_arg: ExprId,
    var: &str,
    rational_integral: ExprId,
) -> Option<ExprId> {
    let (remainder, scale, exact_match) =
        split_arctan_part_when_subtracting(ctx, rational_integral, arctan, target_arg, var)?;
    if scale.is_zero() {
        return None;
    }

    let combined_poly = v_poly.add(&Polynomial::new(vec![scale], v_poly.var.clone()));
    if combined_poly.is_zero() {
        return None;
    }

    if let Ok(remainder_poly) = Polynomial::from_expr(ctx, remainder, &v_poly.var) {
        return Some(
            arctan_polynomial_minus_remainder_with_rational_content_factored(
                ctx,
                arctan,
                &combined_poly,
                &remainder_poly,
            ),
        );
    }
    let _ = exact_match;
    Some(arctan_polynomial_minus_expr_with_rational_content_factored(
        ctx,
        arctan,
        &combined_poly,
        remainder,
    ))
}

/// Build (num, den) Polynomials in t = tan(k x / 2) for a rational
/// expression over sin/cos atoms: sin -> 2t/(1+t^2),
/// cos -> (1-t^2)/(1+t^2).
pub(super) fn weierstrass_rational_function_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    u_name: &str,
) -> Option<(Polynomial, Polynomial)> {
    let one = Polynomial::one(u_name.to_string());
    if let Some((_, is_sine)) = weierstrass_trig_atom(ctx, expr, var) {
        let mut den = Polynomial::zero(u_name.to_string());
        den.coeffs = vec![
            BigRational::from_integer(1.into()),
            BigRational::zero(),
            BigRational::from_integer(1.into()),
        ];
        let mut num = Polynomial::zero(u_name.to_string());
        if is_sine {
            num.coeffs = vec![BigRational::zero(), BigRational::from_integer(2.into())];
        } else {
            num.coeffs = vec![
                BigRational::from_integer(1.into()),
                BigRational::zero(),
                BigRational::from_integer((-1).into()),
            ];
        }
        return Some((num, den));
    }
    if !contains_named_var(ctx, expr, var) {
        let value = crate::numeric_eval::as_rational_const(ctx, expr)?;
        let mut constant = Polynomial::zero(u_name.to_string());
        constant.coeffs = vec![value];
        return Some((constant, one));
    }
    match ctx.get(expr).clone() {
        Expr::Add(l, r) => {
            let (n1, d1) = weierstrass_rational_function_parts(ctx, l, var, u_name)?;
            let (n2, d2) = weierstrass_rational_function_parts(ctx, r, var, u_name)?;
            Some((n1.mul(&d2).add(&n2.mul(&d1)), d1.mul(&d2)))
        }
        Expr::Sub(l, r) => {
            let (n1, d1) = weierstrass_rational_function_parts(ctx, l, var, u_name)?;
            let (n2, d2) = weierstrass_rational_function_parts(ctx, r, var, u_name)?;
            let neg_n2 = scale_polynomial_rational(&n2, &-BigRational::from_integer(1.into()));
            Some((n1.mul(&d2).add(&neg_n2.mul(&d1)), d1.mul(&d2)))
        }
        Expr::Mul(l, r) => {
            let (n1, d1) = weierstrass_rational_function_parts(ctx, l, var, u_name)?;
            let (n2, d2) = weierstrass_rational_function_parts(ctx, r, var, u_name)?;
            Some((n1.mul(&n2), d1.mul(&d2)))
        }
        Expr::Div(l, r) => {
            let (n1, d1) = weierstrass_rational_function_parts(ctx, l, var, u_name)?;
            let (n2, d2) = weierstrass_rational_function_parts(ctx, r, var, u_name)?;
            if n2.is_zero() {
                return None;
            }
            Some((n1.mul(&d2), d1.mul(&n2)))
        }
        Expr::Neg(inner) => {
            let (n, d) = weierstrass_rational_function_parts(ctx, inner, var, u_name)?;
            Some((
                scale_polynomial_rational(&n, &-BigRational::from_integer(1.into())),
                d,
            ))
        }
        Expr::Pow(base, exponent) => {
            let value = crate::numeric_eval::as_rational_const(ctx, exponent)?;
            if !value.is_integer() {
                return None;
            }
            let p = i64::try_from(&value.to_integer()).ok()?;
            let (n, d) = weierstrass_rational_function_parts(ctx, base, var, u_name)?;
            let times = usize::try_from(p.unsigned_abs()).ok()?;
            let mut acc_n = Polynomial::one(u_name.to_string());
            let mut acc_d = Polynomial::one(u_name.to_string());
            for _ in 0..times {
                acc_n = acc_n.mul(&n);
                acc_d = acc_d.mul(&d);
            }
            if p >= 0 {
                Some((acc_n, acc_d))
            } else {
                if acc_n.is_zero() {
                    return None;
                }
                Some((acc_d, acc_n))
            }
        }
        _ => None,
    }
}

/// Build (num, den) Polynomials in u = sqrt(a x + b) for a rational
/// expression over x and half-integer powers of the radicand, with
/// x = (u^2 - b)/a. Negative u-powers land in the denominator.
pub(super) fn linear_radical_rational_function_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    slope: &BigRational,
    offset: &BigRational,
    u_name: &str,
) -> Option<(Polynomial, Polynomial)> {
    let one = Polynomial::one(u_name.to_string());
    if let Some(atom) = linear_radical_atom(ctx, expr, var) {
        let mut mono = Polynomial::zero(u_name.to_string());
        let degree = usize::try_from(atom.half_power.unsigned_abs()).ok()?;
        mono.coeffs = vec![BigRational::zero(); degree + 1];
        mono.coeffs[degree] = BigRational::from_integer(1.into());
        return if atom.half_power >= 0 {
            Some((mono, one))
        } else {
            Some((one, mono))
        };
    }
    if !contains_named_var(ctx, expr, var) {
        let value = crate::numeric_eval::as_rational_const(ctx, expr)?;
        let mut constant = Polynomial::zero(u_name.to_string());
        constant.coeffs = vec![value];
        return Some((constant, one));
    }
    if matches!(ctx.get(expr), Expr::Variable(sym) if ctx.sym_name(*sym) == var) {
        // x = (u^2 - b)/a.
        let mut x_poly = Polynomial::zero(u_name.to_string());
        x_poly.coeffs = vec![
            -offset / slope,
            BigRational::zero(),
            BigRational::from_integer(1.into()) / slope,
        ];
        return Some((x_poly, one));
    }
    match ctx.get(expr).clone() {
        Expr::Add(l, r) => {
            let (n1, d1) =
                linear_radical_rational_function_parts(ctx, l, var, slope, offset, u_name)?;
            let (n2, d2) =
                linear_radical_rational_function_parts(ctx, r, var, slope, offset, u_name)?;
            Some((n1.mul(&d2).add(&n2.mul(&d1)), d1.mul(&d2)))
        }
        Expr::Sub(l, r) => {
            let (n1, d1) =
                linear_radical_rational_function_parts(ctx, l, var, slope, offset, u_name)?;
            let (n2, d2) =
                linear_radical_rational_function_parts(ctx, r, var, slope, offset, u_name)?;
            let neg_n2 = scale_polynomial_rational(&n2, &-BigRational::from_integer(1.into()));
            Some((n1.mul(&d2).add(&neg_n2.mul(&d1)), d1.mul(&d2)))
        }
        Expr::Mul(l, r) => {
            let (n1, d1) =
                linear_radical_rational_function_parts(ctx, l, var, slope, offset, u_name)?;
            let (n2, d2) =
                linear_radical_rational_function_parts(ctx, r, var, slope, offset, u_name)?;
            Some((n1.mul(&n2), d1.mul(&d2)))
        }
        Expr::Div(l, r) => {
            let (n1, d1) =
                linear_radical_rational_function_parts(ctx, l, var, slope, offset, u_name)?;
            let (n2, d2) =
                linear_radical_rational_function_parts(ctx, r, var, slope, offset, u_name)?;
            if n2.is_zero() {
                return None;
            }
            Some((n1.mul(&d2), d1.mul(&n2)))
        }
        Expr::Neg(inner) => {
            let (n, d) =
                linear_radical_rational_function_parts(ctx, inner, var, slope, offset, u_name)?;
            Some((
                scale_polynomial_rational(&n, &-BigRational::from_integer(1.into())),
                d,
            ))
        }
        Expr::Pow(base, exponent) => {
            let value = crate::numeric_eval::as_rational_const(ctx, exponent)?;
            if !value.is_integer() {
                return None;
            }
            let p = i64::try_from(&value.to_integer()).ok()?;
            let (n, d) =
                linear_radical_rational_function_parts(ctx, base, var, slope, offset, u_name)?;
            let times = usize::try_from(p.unsigned_abs()).ok()?;
            let mut acc_n = Polynomial::one(u_name.to_string());
            let mut acc_d = Polynomial::one(u_name.to_string());
            for _ in 0..times {
                acc_n = acc_n.mul(&n);
                acc_d = acc_d.mul(&d);
            }
            if p >= 0 {
                Some((acc_n, acc_d))
            } else {
                if acc_n.is_zero() {
                    return None;
                }
                Some((acc_d, acc_n))
            }
        }
        _ => None,
    }
}

/// Build (num, den) Polynomials in u for a rational expression over
/// e^(k*var) atoms with RATIONAL constant coefficients. Negative
/// u-powers land in the denominator.
pub(super) fn exponential_rational_function_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    c: &BigRational,
    u_name: &str,
) -> Option<(Polynomial, Polynomial)> {
    let one = Polynomial::one(u_name.to_string());
    if let Some(slope) = exponential_atom_slope(ctx, expr, var) {
        let power = slope / c;
        if !power.is_integer() {
            return None;
        }
        let p = i64::try_from(&power.to_integer()).ok()?;
        let mut mono = Polynomial::zero(u_name.to_string());
        let degree = usize::try_from(p.unsigned_abs()).ok()?;
        mono.coeffs = vec![BigRational::zero(); degree + 1];
        mono.coeffs[degree] = BigRational::from_integer(1.into());
        return if p >= 0 {
            Some((mono, one))
        } else {
            Some((one, mono))
        };
    }
    if !contains_named_var(ctx, expr, var) {
        let value = crate::numeric_eval::as_rational_const(ctx, expr)?;
        let mut constant = Polynomial::zero(u_name.to_string());
        constant.coeffs = vec![value];
        return Some((constant, one));
    }
    match ctx.get(expr).clone() {
        Expr::Add(l, r) => {
            let (n1, d1) = exponential_rational_function_parts(ctx, l, var, c, u_name)?;
            let (n2, d2) = exponential_rational_function_parts(ctx, r, var, c, u_name)?;
            Some((n1.mul(&d2).add(&n2.mul(&d1)), d1.mul(&d2)))
        }
        Expr::Sub(l, r) => {
            let (n1, d1) = exponential_rational_function_parts(ctx, l, var, c, u_name)?;
            let (n2, d2) = exponential_rational_function_parts(ctx, r, var, c, u_name)?;
            let neg_n2 = scale_polynomial_rational(&n2, &-BigRational::from_integer(1.into()));
            Some((n1.mul(&d2).add(&neg_n2.mul(&d1)), d1.mul(&d2)))
        }
        Expr::Mul(l, r) => {
            let (n1, d1) = exponential_rational_function_parts(ctx, l, var, c, u_name)?;
            let (n2, d2) = exponential_rational_function_parts(ctx, r, var, c, u_name)?;
            Some((n1.mul(&n2), d1.mul(&d2)))
        }
        Expr::Div(l, r) => {
            let (n1, d1) = exponential_rational_function_parts(ctx, l, var, c, u_name)?;
            let (n2, d2) = exponential_rational_function_parts(ctx, r, var, c, u_name)?;
            if n2.is_zero() {
                return None;
            }
            Some((n1.mul(&d2), d1.mul(&n2)))
        }
        Expr::Neg(inner) => {
            let (n, d) = exponential_rational_function_parts(ctx, inner, var, c, u_name)?;
            Some((
                scale_polynomial_rational(&n, &-BigRational::from_integer(1.into())),
                d,
            ))
        }
        Expr::Pow(base, exponent) => {
            let value = crate::numeric_eval::as_rational_const(ctx, exponent)?;
            if !value.is_integer() {
                return None;
            }
            let p = i64::try_from(&value.to_integer()).ok()?;
            let (n, d) = exponential_rational_function_parts(ctx, base, var, c, u_name)?;
            let times = usize::try_from(p.unsigned_abs()).ok()?;
            let mut acc_n = Polynomial::one(u_name.to_string());
            let mut acc_d = Polynomial::one(u_name.to_string());
            for _ in 0..times {
                acc_n = acc_n.mul(&n);
                acc_d = acc_d.mul(&d);
            }
            if p >= 0 {
                Some((acc_n, acc_d))
            } else {
                if acc_n.is_zero() {
                    return None;
                }
                Some((acc_d, acc_n))
            }
        }
        _ => None,
    }
}

pub(super) fn polynomial_denominator_power_parts(
    ctx: &mut Context,
    den: ExprId,
    var: &str,
) -> Option<(ExprId, BigRational, BigRational)> {
    if let Expr::Pow(base, exp) = ctx.get(den) {
        let base = *base;
        let exp = *exp;
        let exponent = rational_constant_value(ctx, exp)?;
        let (base, exponent) = if let Some(radicand) = sqrt_like_radicand(ctx, base) {
            (radicand, exponent / BigRational::from_integer(2.into()))
        } else {
            (base, exponent)
        };
        if exponent <= BigRational::one() {
            return None;
        }
        if !contains_named_var(ctx, base, var) {
            return None;
        }

        return Some((base, exponent, BigRational::one()));
    }

    scaled_syntactic_polynomial_denominator_power_parts(ctx, den, var)
        .or_else(|| {
            expanded_square_denominator_base(ctx, den, var).map(|base| {
                (
                    base,
                    BigRational::from_integer(2.into()),
                    BigRational::one(),
                )
            })
        })
        .or_else(|| expanded_polynomial_denominator_power_parts(ctx, den, var))
}

fn scaled_syntactic_polynomial_denominator_power_parts(
    ctx: &mut Context,
    den: ExprId,
    var: &str,
) -> Option<(ExprId, BigRational, BigRational)> {
    let factors = mul_leaves(ctx, den);
    if factors.len() < 2 {
        return None;
    }

    let mut scale = BigRational::one();
    let mut power_part = None;

    for factor in factors {
        if power_part.is_none() {
            let candidate = match ctx.get(factor) {
                Expr::Pow(base, exp) => {
                    let exponent = rational_constant_value(ctx, *exp)?;
                    if let Some(radicand) = sqrt_like_radicand(ctx, *base) {
                        Some((radicand, exponent / BigRational::from_integer(2.into())))
                    } else {
                        Some((*base, exponent))
                    }
                }
                _ => None,
            };

            if let Some((base, exponent)) = candidate {
                if exponent > BigRational::one()
                    && contains_named_var(ctx, base, var)
                    && Polynomial::from_expr(ctx, base, var).is_ok()
                {
                    power_part = Some((base, exponent));
                    continue;
                }
            }
        }

        scale *= rational_constant_value(ctx, factor)?;
    }

    if scale.is_zero() {
        return None;
    }

    power_part.map(|(base, exponent)| (base, exponent, scale))
}

pub(super) fn negative_syntactic_polynomial_denominator_power_parts(
    ctx: &mut Context,
    den: ExprId,
    var: &str,
) -> Option<(ExprId, BigRational, BigRational)> {
    negative_syntactic_polynomial_denominator_power_parts_view(ctx, den, var)
}

fn negative_syntactic_polynomial_denominator_power_parts_view(
    ctx: &Context,
    den: ExprId,
    var: &str,
) -> Option<(ExprId, BigRational, BigRational)> {
    negative_syntactic_polynomial_denominator_power_parts_by(
        ctx,
        den,
        var,
        rational_constant_value,
        rational_constant_value,
    )
}

fn negative_syntactic_polynomial_denominator_power_parts_by<ExponentValue, ScaleValue>(
    ctx: &Context,
    den: ExprId,
    var: &str,
    exponent_value: ExponentValue,
    scale_value: ScaleValue,
) -> Option<(ExprId, BigRational, BigRational)>
where
    ExponentValue: Fn(&Context, ExprId) -> Option<BigRational>,
    ScaleValue: Fn(&Context, ExprId) -> Option<BigRational>,
{
    let factors = mul_leaves(ctx, den);
    let mut scale = BigRational::one();
    let mut power_part = None;

    for factor in factors {
        if power_part.is_none() {
            let candidate = match ctx.get(factor) {
                Expr::Pow(base, exp) => Some((*base, *exp)),
                _ => None,
            };

            if let Some((base, exp)) = candidate {
                let exponent = exponent_value(ctx, exp)?;
                if exponent.is_integer()
                    && exponent < BigRational::zero()
                    && contains_named_var(ctx, base, var)
                    && Polynomial::from_expr(ctx, base, var).is_ok()
                {
                    power_part = Some((base, -exponent));
                    continue;
                }
            }
        }

        scale *= scale_value(ctx, factor)?;
    }

    if scale.is_zero() {
        return None;
    }

    power_part.map(|(base, exponent)| (base, exponent, scale))
}

fn bounded_negative_syntactic_polynomial_denominator_power_parts_view(
    ctx: &Context,
    den: ExprId,
    var: &str,
    max_abs_power: i64,
) -> Option<(ExprId, BigRational, BigRational)> {
    let (base, exponent, scale) = negative_syntactic_polynomial_denominator_power_parts_by(
        ctx,
        den,
        var,
        |ctx, exp| {
            crate::numeric::as_i64(ctx, exp).map(|value| BigRational::from_integer(value.into()))
        },
        |ctx, factor| cas_ast::views::as_rational_const(ctx, factor, 4),
    )?;

    let bound = BigRational::from_integer(max_abs_power.into());
    if exponent > bound {
        return None;
    }

    Some((base, exponent, scale))
}

pub(super) fn bounded_negative_denominator_power_substitution_target_parts(
    ctx: &Context,
    expr: ExprId,
    var: &str,
    max_abs_power: i64,
) -> Option<(ExprId, BigRational)> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let (base, exponent, _) = bounded_negative_syntactic_polynomial_denominator_power_parts_view(
        ctx,
        *den,
        var,
        max_abs_power,
    )?;

    let Ok(numerator) = Polynomial::from_expr(ctx, *num, var) else {
        return None;
    };
    let Ok(base_poly) = Polynomial::from_expr(ctx, base, var) else {
        return None;
    };
    let derivative = base_poly.derivative();
    if constant_polynomial_ratio(&numerator, &derivative).is_none_or(|scale| scale.is_zero()) {
        return None;
    }

    Some((base, exponent))
}

fn reciprocal_quotient_polynomial_denominator_power_parts_view(
    ctx: &Context,
    den: ExprId,
    var: &str,
) -> Option<(ExprId, BigRational, BigRational)> {
    let (scale_expr, reciprocal_den) = match ctx.get(den) {
        Expr::Div(scale, reciprocal_den) => (*scale, *reciprocal_den),
        _ => return None,
    };
    let scale = rational_constant_value(ctx, scale_expr)?;
    if scale.is_zero() {
        return None;
    }

    let (base, exponent) = match ctx.get(reciprocal_den) {
        Expr::Pow(base, exp) => {
            let exponent = rational_constant_value(ctx, *exp)?;
            let negative_two = BigRational::from_integer((-2).into());
            if !exponent.is_integer() || (exponent < BigRational::one() && exponent > negative_two)
            {
                return None;
            }
            (*base, exponent)
        }
        _ => (reciprocal_den, BigRational::one()),
    };

    if !contains_named_var(ctx, base, var) || Polynomial::from_expr(ctx, base, var).is_err() {
        return None;
    }

    Some((base, exponent, scale))
}

pub(super) fn reciprocal_quotient_denominator_power_substitution_target_parts(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, BigRational)> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let (base, exponent, _) =
        reciprocal_quotient_polynomial_denominator_power_parts_view(ctx, *den, var)?;

    let Ok(numerator) = Polynomial::from_expr(ctx, *num, var) else {
        return None;
    };
    let Ok(base_poly) = Polynomial::from_expr(ctx, base, var) else {
        return None;
    };
    let derivative = base_poly.derivative();
    if constant_polynomial_ratio(&numerator, &derivative).is_none_or(|scale| scale.is_zero()) {
        return None;
    }

    Some((base, exponent))
}

pub(super) fn reciprocal_quotient_polynomial_denominator_power_parts(
    ctx: &mut Context,
    den: ExprId,
    var: &str,
) -> Option<(ExprId, BigRational, BigRational)> {
    reciprocal_quotient_polynomial_denominator_power_parts_view(ctx, den, var)
}

fn expanded_polynomial_denominator_power_parts(
    ctx: &mut Context,
    den: ExprId,
    var: &str,
) -> Option<(ExprId, BigRational, BigRational)> {
    const MAX_EXPANDED_DENOMINATOR_POWER: usize = 5;

    let den_poly = Polynomial::from_expr(ctx, den, var).ok()?;
    if den_poly.degree() < 2 {
        return None;
    }

    let derivative = den_poly.derivative();
    if derivative.is_zero() {
        return None;
    }

    let repeated_factor = den_poly.gcd(&derivative);
    if repeated_factor.is_zero() || repeated_factor.degree() == 0 {
        return None;
    }

    let (mut base_poly, remainder) = den_poly.div_rem(&repeated_factor).ok()?;
    if !remainder.is_zero() || base_poly.is_zero() || base_poly.degree() == 0 {
        return None;
    }

    let base_lc = base_poly.leading_coeff();
    if base_lc.is_zero() {
        return None;
    }
    base_poly = base_poly.div_scalar(&base_lc);

    let base_degree = base_poly.degree();
    if den_poly.degree() % base_degree != 0 {
        return None;
    }

    let exponent = den_poly.degree() / base_degree;
    if !(2..=MAX_EXPANDED_DENOMINATOR_POWER).contains(&exponent) {
        return None;
    }

    let mut reconstructed = Polynomial::one(var.to_string());
    for _ in 0..exponent {
        reconstructed = reconstructed.mul(&base_poly);
    }
    let denominator_scale = constant_polynomial_ratio(&den_poly, &reconstructed)?;
    if denominator_scale.is_zero() {
        return None;
    }

    let base = base_poly.to_expr(ctx);
    if !contains_named_var(ctx, base, var) {
        return None;
    }

    Some((
        base,
        BigRational::from_integer((exponent as i64).into()),
        denominator_scale,
    ))
}

pub(super) fn log_product_substitution_factor_parts(
    ctx: &mut Context,
    factor: ExprId,
) -> Option<(ExprId, ExprId, ExprId, bool)> {
    let Expr::Function(fn_id, args) = ctx.get(factor).clone() else {
        return None;
    };

    match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Ln) if args.len() == 1 => {
            let one = ctx.num(1);
            Some((factor, args[0], one, false))
        }
        Some(BuiltinFn::Log) if args.len() == 2 => {
            let base = args[0];
            let arg = args[1];
            let base_ln = valid_constant_log_base_ln(ctx, base)?;
            let distribute_correction = base_ln.is_some();
            let log_expr = if base_ln.is_none() {
                ctx.call_builtin(BuiltinFn::Ln, vec![arg])
            } else {
                factor
            };
            let correction = constant_base_log_derivative_correction(ctx, base_ln);
            Some((log_expr, arg, correction, distribute_correction))
        }
        Some(BuiltinFn::Log2) if args.len() == 1 => {
            let correction = positive_integer_constant_log_base_derivative_correction(ctx, 2);
            Some((factor, args[0], correction, true))
        }
        Some(BuiltinFn::Log10) if args.len() == 1 => {
            let correction = positive_integer_constant_log_base_derivative_correction(ctx, 10);
            Some((factor, args[0], correction, true))
        }
        _ => None,
    }
}

fn monomial_log_power_by_parts_integral(
    ctx: &mut Context,
    x_next: ExprId,
    log_expr: ExprId,
    next_power: &BigRational,
    log_power: u32,
) -> ExprId {
    let mut terms = Vec::new();
    for degree in (0..=log_power).rev() {
        let mut coeff = descending_factorial_ratio(log_power, degree)
            * positive_integer_power_rational(next_power, degree);
        if (log_power - degree) % 2 == 1 {
            coeff = -coeff;
        }

        if degree == 0 {
            terms.push(ctx.add(Expr::Number(coeff)));
        } else {
            let term = log_power_term(ctx, log_expr, degree);
            terms.push(scale_rational_term(ctx, coeff, term));
        }
    }

    let inner = build_balanced_add(ctx, &terms);
    let product_raw = mul2_raw(ctx, x_next, inner);
    let product = cas_ast::hold::wrap_hold(ctx, product_raw);
    let denominator = positive_integer_power_rational(next_power, log_power + 1);
    scale_rational_term(ctx, BigRational::one() / denominator, product)
}

pub(super) fn monomial_times_ln_var_by_parts_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    // Normalization artifact: ln(x)^m / (c*x^k) arrives as Div; rebuild
    // as a product with the reciprocal power and reuse the same route.
    if let Expr::Div(numerator, denominator) = ctx.get(expr).clone() {
        let (scale, power) = scaled_var_power_term(ctx, denominator, var)?;
        if scale.is_zero() {
            return None;
        }
        let negated = ctx.add(Expr::Number(-power));
        let var_expr = ctx.var(var);
        let reciprocal = ctx.add(Expr::Pow(var_expr, negated));
        let product = mul2_raw(ctx, numerator, reciprocal);
        let integral = monomial_times_ln_var_by_parts_antiderivative(ctx, product, var)?;
        return Some(scale_rational_term(
            ctx,
            BigRational::one() / scale,
            integral,
        ));
    }

    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (log_index, factor) in factors.iter().enumerate() {
        let Some((log_expr, log_base, log_power)) = natural_log_power_factor_parts(ctx, *factor)
        else {
            continue;
        };
        if !matches!(log_power, 1..=5) {
            continue;
        }
        let log_arg = natural_log_argument(ctx, log_expr)?;
        if !is_var(ctx, log_arg, var) || !is_var(ctx, log_base, var) {
            continue;
        }

        let cofactor_factors: Vec<ExprId> = factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != log_index).then_some(*factor))
            .collect();
        if cofactor_factors.is_empty() {
            return None;
        }
        let cofactor = build_balanced_mul(ctx, &cofactor_factors);
        let (scale, power) = scaled_var_power_term(ctx, cofactor, var)?;
        // Any rational power works in the closed by-parts form except
        // p = -1 (next_power 0): ln(x)/x belongs to the u-substitution
        // owner (ln(x)^2/2).
        let next_power = power + BigRational::one();
        if next_power.is_zero() {
            return None;
        }

        let var_expr = ctx.var(var);
        let next_power_expr = ctx.add(Expr::Number(next_power.clone()));
        let x_next = ctx.add(Expr::Pow(var_expr, next_power_expr));
        let integral =
            monomial_log_power_by_parts_integral(ctx, x_next, log_expr, &next_power, log_power);

        return Some(scale_rational_term(ctx, scale, integral));
    }

    None
}

pub fn integrate_symbolic_is_monomial_times_ln_var_by_parts_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    monomial_times_ln_var_by_parts_antiderivative(ctx, expr, var).is_some()
}

fn linear_affine_ln_term_parts(
    ctx: &mut Context,
    term: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId, BigRational, BigRational)> {
    let factors = mul_leaves(ctx, term);

    for (log_index, factor) in factors.iter().enumerate() {
        let log_arg = match ctx.get(*factor).clone() {
            Expr::Function(fn_id, args)
                if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Ln) =>
            {
                args[0]
            }
            _ => continue,
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
        let (cofactor_slope, cofactor_offset) = get_linear_coeffs(ctx, cofactor, var)?;
        let cofactor_slope = rational_constant_value(ctx, cofactor_slope)?;
        let cofactor_offset = rational_constant_value(ctx, cofactor_offset)?;

        return Some((log_arg, *factor, cofactor_slope, cofactor_offset));
    }

    None
}

pub(super) fn additive_linear_times_affine_ln_by_parts_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    let mut common_log_arg = None;
    let mut common_log_factor = None;
    let mut slope_sum = BigRational::zero();
    let mut offset_sum = BigRational::zero();

    for (term, sign) in view.terms {
        let (log_arg, log_factor, mut slope, mut offset) =
            linear_affine_ln_term_parts(ctx, term, var)?;
        if sign == Sign::Neg {
            slope = -slope;
            offset = -offset;
        }

        if let Some(existing_arg) = common_log_arg {
            if compare_expr(ctx, existing_arg, log_arg) != Ordering::Equal {
                return None;
            }
        } else {
            common_log_arg = Some(log_arg);
            common_log_factor = Some(log_factor);
        }

        slope_sum += slope;
        offset_sum += offset;
    }

    let cofactor = build_linear_expr_from_rationals(ctx, var, slope_sum, offset_sum)?;
    let combined_expr = mul2_raw(ctx, cofactor, common_log_factor?);
    linear_times_affine_ln_by_parts_antiderivative(ctx, combined_expr, var)
}

fn polynomial_affine_ln_term_parts(
    ctx: &mut Context,
    term: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId, Polynomial)> {
    let factors = mul_leaves(ctx, term);

    for (log_index, factor) in factors.iter().enumerate() {
        let log_arg = match ctx.get(*factor).clone() {
            Expr::Function(fn_id, args)
                if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Ln) =>
            {
                args[0]
            }
            _ => continue,
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
        let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;

        return Some((log_arg, *factor, cofactor_poly));
    }

    None
}

pub(super) fn additive_quadratic_times_affine_ln_by_parts_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    let mut common_log_arg = None;
    let mut common_log_factor = None;
    let mut cofactor_sum = Polynomial::zero(var.to_string());

    for (term, sign) in view.terms {
        let (log_arg, log_factor, mut cofactor) = polynomial_affine_ln_term_parts(ctx, term, var)?;
        if sign == Sign::Neg {
            cofactor = cofactor.neg();
        }

        if let Some(existing_arg) = common_log_arg {
            if compare_expr(ctx, existing_arg, log_arg) != Ordering::Equal {
                return None;
            }
        } else {
            common_log_arg = Some(log_arg);
            common_log_factor = Some(log_factor);
        }

        cofactor_sum = cofactor_sum.add(&cofactor);
    }

    if cofactor_sum.degree() < 2 || cofactor_sum.degree() > AFFINE_LN_BY_PARTS_MAX_COFACTOR_DEGREE {
        return None;
    }

    let cofactor = cofactor_sum.to_expr(ctx);
    let combined_expr = mul2_raw(ctx, cofactor, common_log_factor?);
    quadratic_times_affine_ln_by_parts_antiderivative(ctx, combined_expr, var)
}

pub(super) fn additive_positive_quadratic_ln_by_parts_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    let mut common_log_arg = None;
    let mut common_log_factor = None;
    let mut cofactor_sum = Polynomial::zero(var.to_string());

    for (term, sign) in view.terms {
        let (log_arg, log_factor, mut cofactor) = polynomial_affine_ln_term_parts(ctx, term, var)?;
        let log_arg_poly = Polynomial::from_expr(ctx, log_arg, var).ok()?;
        if !is_positive_quadratic_polynomial(&log_arg_poly) {
            return None;
        }
        if sign == Sign::Neg {
            cofactor = cofactor.neg();
        }

        if let Some(existing_arg) = common_log_arg {
            if compare_expr(ctx, existing_arg, log_arg) != Ordering::Equal {
                return None;
            }
        } else {
            common_log_arg = Some(log_arg);
            common_log_factor = Some(log_factor);
        }

        cofactor_sum = cofactor_sum.add(&cofactor);
    }

    if cofactor_sum.is_zero()
        || cofactor_sum.degree() > POSITIVE_QUADRATIC_LN_BY_PARTS_MAX_COFACTOR_DEGREE
    {
        return None;
    }

    let cofactor = cofactor_sum.to_expr(ctx);
    let combined_expr = mul2_raw(ctx, cofactor, common_log_factor?);
    low_degree_times_positive_quadratic_ln_by_parts_antiderivative(ctx, combined_expr, var)
}

pub(super) fn linear_times_affine_ln_by_parts_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    for (log_index, factor) in factors.iter().enumerate() {
        let log_arg = match ctx.get(*factor).clone() {
            Expr::Function(fn_id, args)
                if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Ln) =>
            {
                args[0]
            }
            _ => continue,
        };
        let (arg_slope, arg_offset) = get_linear_coeffs(ctx, log_arg, var)?;
        let arg_slope = rational_constant_value(ctx, arg_slope)?;
        if arg_slope.is_zero() {
            continue;
        }
        let arg_offset = rational_constant_value(ctx, arg_offset)?;

        let cofactor_factors: Vec<ExprId> = factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != log_index).then_some(*factor))
            .collect();
        if cofactor_factors.is_empty() {
            continue;
        }
        let cofactor = build_balanced_mul(ctx, &cofactor_factors);
        let (cofactor_slope, cofactor_offset) = get_linear_coeffs(ctx, cofactor, var)?;
        let cofactor_slope = rational_constant_value(ctx, cofactor_slope)?;
        let cofactor_offset = rational_constant_value(ctx, cofactor_offset)?;
        if cofactor_slope.is_zero() && cofactor_offset.is_zero() {
            continue;
        }

        if !cofactor_slope.is_zero()
            && cofactor_offset.clone() * arg_slope.clone()
                == cofactor_slope.clone() * arg_offset.clone()
        {
            let proportional_scale = cofactor_slope.clone() / arg_slope.clone();
            let two = ctx.num(2);
            let arg_squared = ctx.add(Expr::Pow(log_arg, two));
            let log_term = mul2_raw(ctx, arg_squared, *factor);

            let var_expr = ctx.var(var);
            let two = ctx.num(2);
            let x_squared = ctx.add(Expr::Pow(var_expr, two));
            let quadratic_coeff =
                -(arg_slope.clone() * arg_slope.clone()) / BigRational::from_integer(2.into());
            let quadratic_term = scale_rational_term(ctx, quadratic_coeff, x_squared);
            let linear_coeff = -(arg_slope.clone() * arg_offset.clone());
            let linear_term = scale_rational_term(ctx, linear_coeff, var_expr);
            let inner = build_balanced_add(ctx, &[log_term, quadratic_term, linear_term]);

            let scale =
                proportional_scale / (BigRational::from_integer(2.into()) * arg_slope.clone());
            return Some(scale_rational_term(ctx, scale, inner));
        }

        if !cofactor_slope.is_zero() && !cofactor_offset.is_zero() {
            let arg_slope_squared = arg_slope.clone() * arg_slope.clone();
            let u_coeff = cofactor_slope.clone() / arg_slope_squared.clone();
            let constant_coeff = (cofactor_offset.clone() * arg_slope.clone()
                - cofactor_slope.clone() * arg_offset.clone())
                / arg_slope_squared;

            let log_inner_shift =
                (BigRational::from_integer(2.into()) * constant_coeff.clone()) / u_coeff.clone();
            let log_inner = if log_inner_shift.is_zero() {
                log_arg
            } else {
                let shift = ctx.add(Expr::Number(log_inner_shift));
                ctx.add(Expr::Add(log_arg, shift))
            };
            let log_coeff = build_balanced_mul(ctx, &[log_arg, log_inner]);
            let log_coeff = scale_rational_term(
                ctx,
                u_coeff.clone() / BigRational::from_integer(2.into()),
                log_coeff,
            );
            let log_term = mul2_raw(ctx, log_coeff, *factor);

            let var_expr = ctx.var(var);
            let two = ctx.num(2);
            let x_squared = ctx.add(Expr::Pow(var_expr, two));
            let quadratic_term = scale_rational_term(
                ctx,
                -cofactor_slope.clone() / BigRational::from_integer(4.into()),
                x_squared,
            );

            let linear_coeff = cofactor_slope.clone() * arg_offset.clone()
                / (BigRational::from_integer(2.into()) * arg_slope.clone())
                - cofactor_offset.clone();
            let linear_term = scale_rational_term(ctx, linear_coeff, var_expr);

            return Some(build_balanced_add(
                ctx,
                &[log_term, quadratic_term, linear_term],
            ));
        }

        let mut integral_terms = Vec::new();

        if !cofactor_slope.is_zero() {
            let var_expr = ctx.var(var);
            let two = ctx.num(2);
            let x_squared = ctx.add(Expr::Pow(var_expr, two));
            let half_x_squared = ctx.add(Expr::Div(x_squared, two));

            let log_coeff = {
                let denominator =
                    BigRational::from_integer(2.into()) * arg_slope.clone() * arg_slope.clone();
                let offset_square = arg_offset.clone() * arg_offset.clone();
                let offset_term = offset_square / denominator;
                if offset_term.is_zero() {
                    half_x_squared
                } else {
                    let offset_term = ctx.add(Expr::Number(offset_term));
                    ctx.add(Expr::Sub(half_x_squared, offset_term))
                }
            };
            let log_term = mul2_raw(ctx, log_coeff, *factor);
            let negative_quarter_x_squared =
                scale_rational_term(ctx, BigRational::new((-1).into(), 4.into()), x_squared);

            let linear_scale =
                arg_offset.clone() / (BigRational::from_integer(2.into()) * arg_slope.clone());
            let x_integral = if linear_scale.is_zero() {
                ctx.add(Expr::Add(log_term, negative_quarter_x_squared))
            } else {
                let linear_term = scale_rational_term(ctx, linear_scale, var_expr);
                build_balanced_add(ctx, &[log_term, negative_quarter_x_squared, linear_term])
            };
            let x_integral = scale_rational_term(ctx, cofactor_slope, x_integral);
            integral_terms.push(cas_ast::hold::wrap_hold(ctx, x_integral));
        }

        if !cofactor_offset.is_zero() {
            let one = ctx.num(1);
            let log_minus_one = ctx.add(Expr::Sub(*factor, one));
            let affine_log_integral = mul2_raw(ctx, log_arg, log_minus_one);
            let affine_log_integral =
                scale_rational_term(ctx, BigRational::one() / arg_slope, affine_log_integral);
            let affine_log_integral =
                scale_rational_term(ctx, cofactor_offset, affine_log_integral);
            integral_terms.push(cas_ast::hold::wrap_hold(ctx, affine_log_integral));
        }

        return Some(match integral_terms.as_slice() {
            [single] => *single,
            _ => build_balanced_add(ctx, &integral_terms),
        });
    }

    None
}

pub fn integrate_symbolic_is_linear_times_affine_ln_by_parts_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    linear_times_affine_ln_by_parts_antiderivative(ctx, expr, var).is_some()
        || additive_linear_times_affine_ln_by_parts_antiderivative(ctx, expr, var).is_some()
}

pub(super) fn quadratic_times_affine_ln_by_parts_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    for (log_index, factor) in factors.iter().enumerate() {
        let log_factor = cas_ast::hold::unwrap_hold(ctx, *factor);
        let log_arg = match ctx.get(log_factor).clone() {
            Expr::Function(fn_id, args)
                if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Ln) =>
            {
                args[0]
            }
            _ => continue,
        };
        let (arg_slope, arg_offset) = get_linear_coeffs(ctx, log_arg, var)?;
        let arg_slope = rational_constant_value(ctx, arg_slope)?;
        if arg_slope.is_zero() {
            continue;
        }
        let arg_offset = rational_constant_value(ctx, arg_offset)?;

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
        let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
        if cofactor_poly.degree() < 2
            || cofactor_poly.degree() > AFFINE_LN_BY_PARTS_MAX_COFACTOR_DEGREE
        {
            continue;
        }

        let primitive_poly = polynomial_antiderivative(&cofactor_poly);
        let linear_divisor = Polynomial::new(
            vec![arg_offset.clone(), arg_slope.clone()],
            cofactor_poly.var.clone(),
        );
        let numerator = primitive_poly.mul(&Polynomial::new(
            vec![arg_slope.clone()],
            cofactor_poly.var.clone(),
        ));
        let (quotient, remainder) = numerator.div_rem(&linear_divisor).ok()?;

        let primitive_expr = primitive_poly.to_expr(ctx);
        let leading_product = mul2_raw(ctx, primitive_expr, *factor);
        let leading = cas_ast::hold::wrap_hold(ctx, leading_product);

        let mut residual_terms = Vec::new();
        if !quotient.is_zero() {
            residual_terms.push(polynomial_antiderivative_expr(ctx, &quotient));
        }
        if !remainder.is_zero() {
            let remainder_coeff = remainder
                .coeffs
                .first()
                .cloned()
                .unwrap_or_else(BigRational::zero);
            if !remainder_coeff.is_zero() {
                residual_terms.push(scale_rational_term(
                    ctx,
                    remainder_coeff / arg_slope,
                    *factor,
                ));
            }
        }

        if residual_terms.is_empty() {
            return Some(leading);
        }

        let residual = build_balanced_add(ctx, &residual_terms);
        if cofactor_poly.degree() >= AFFINE_LN_BY_PARTS_MAX_COFACTOR_DEGREE {
            return Some(compact_ln_by_parts_result(
                ctx,
                &primitive_poly,
                *factor,
                residual,
                var,
            ));
        }

        let residual = cas_ast::hold::wrap_hold(ctx, residual);
        return Some(ctx.add(Expr::Sub(leading, residual)));
    }

    None
}

fn compact_ln_by_parts_result(
    ctx: &mut Context,
    primitive_poly: &Polynomial,
    log_expr: ExprId,
    residual: ExprId,
    var: &str,
) -> ExprId {
    let target_arg_poly = match ctx.get(cas_ast::hold::unwrap_hold(ctx, log_expr)) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Ln) =>
        {
            Polynomial::from_expr(ctx, args[0], var).ok()
        }
        _ => None,
    };
    let Some(target_arg_poly) = target_arg_poly else {
        let primitive_expr = primitive_poly.to_expr(ctx);
        let leading_product = mul2_raw(ctx, primitive_expr, log_expr);
        let leading = cas_ast::hold::wrap_hold(ctx, leading_product);
        let residual = cas_ast::hold::wrap_hold(ctx, residual);
        return ctx.add(Expr::Sub(leading, residual));
    };

    let mut residual_terms = Vec::new();
    collect_additive_terms_signed(
        ctx,
        cas_ast::hold::unwrap_hold(ctx, residual),
        true,
        &mut residual_terms,
    );

    let mut log_coeff = BigRational::zero();
    let mut remaining_terms = Vec::new();
    for (term, is_positive) in residual_terms {
        if let Some(mut coeff) = scaled_matching_ln_coefficient(ctx, term, &target_arg_poly, var) {
            if !is_positive {
                coeff = -coeff;
            }
            log_coeff += coeff;
        } else {
            remaining_terms.push((term, is_positive));
        }
    }

    if log_coeff.is_zero() {
        let primitive_expr = primitive_poly.to_expr(ctx);
        let leading_product = mul2_raw(ctx, primitive_expr, log_expr);
        let leading = cas_ast::hold::wrap_hold(ctx, leading_product);
        let residual = cas_ast::hold::wrap_hold(ctx, residual);
        return ctx.add(Expr::Sub(leading, residual));
    }

    let adjusted_log_poly = primitive_poly.sub(&Polynomial::new(vec![log_coeff], var.to_string()));
    let adjusted_log_coeff = adjusted_log_poly.to_expr(ctx);
    let leading_product = mul2_raw(ctx, log_expr, adjusted_log_coeff);
    let leading = cas_ast::hold::wrap_hold(ctx, leading_product);

    if remaining_terms.is_empty() {
        return leading;
    }

    let mut flattened_terms = Vec::with_capacity(remaining_terms.len() + 1);
    flattened_terms.push(leading);
    for (term, is_positive) in remaining_terms {
        let term = if is_positive {
            negate_term_for_compact_integration_sum(ctx, term)
        } else {
            term
        };
        flattened_terms.push(term);
    }
    build_balanced_add(ctx, &flattened_terms)
}

pub(super) fn low_degree_times_positive_quadratic_ln_by_parts_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (log_index, factor) in factors.iter().enumerate() {
        let log_arg = match ctx.get(*factor).clone() {
            Expr::Function(fn_id, args)
                if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Ln) =>
            {
                args[0]
            }
            _ => continue,
        };

        let log_arg_poly = Polynomial::from_expr(ctx, log_arg, var).ok()?;
        if !is_positive_quadratic_polynomial(&log_arg_poly) {
            continue;
        }

        let cofactor_factors: Vec<ExprId> = factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != log_index).then_some(*factor))
            .collect();
        if cofactor_factors.is_empty() {
            continue;
        }
        let cofactor = build_balanced_mul(ctx, &cofactor_factors);
        let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
        if cofactor_poly.degree() > POSITIVE_QUADRATIC_LN_BY_PARTS_MAX_COFACTOR_DEGREE {
            continue;
        }

        let primitive_poly = polynomial_antiderivative(&cofactor_poly);
        let primitive_expr = primitive_poly.to_expr(ctx);
        let leading_product = mul2_raw(ctx, primitive_expr, *factor);
        let leading = cas_ast::hold::wrap_hold(ctx, leading_product);

        let residual_num = primitive_poly.mul(&log_arg_poly.derivative()).to_expr(ctx);
        let residual_ratio = ctx.add(Expr::Div(residual_num, log_arg));
        let residual = integrate_symbolic_expr(ctx, residual_ratio, var)?;
        if polynomial_nonzero_term_count(&cofactor_poly) >= 2 {
            return Some(compact_ln_by_parts_result(
                ctx,
                &primitive_poly,
                *factor,
                residual,
                var,
            ));
        }

        let residual = cas_ast::hold::wrap_hold(ctx, residual);
        return Some(ctx.add(Expr::Sub(leading, residual)));
    }

    None
}

pub(super) fn positive_quadratic_ln_by_parts_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let log_expr = cas_ast::hold::unwrap_hold(ctx, expr);
    let log_arg = match ctx.get(log_expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Ln) =>
        {
            args[0]
        }
        _ => return None,
    };

    let log_arg_poly = Polynomial::from_expr(ctx, log_arg, var).ok()?;
    if !is_positive_quadratic_polynomial(&log_arg_poly) {
        return None;
    }

    let cofactor_poly = Polynomial::one(var.to_string());
    let primitive_poly = polynomial_antiderivative(&cofactor_poly);
    let primitive_expr = primitive_poly.to_expr(ctx);
    let leading_product = mul2_raw(ctx, primitive_expr, log_expr);
    let leading = cas_ast::hold::wrap_hold(ctx, leading_product);

    let residual_num = primitive_poly.mul(&log_arg_poly.derivative()).to_expr(ctx);
    let residual_ratio = ctx.add(Expr::Div(residual_num, log_arg));
    let residual = integrate_symbolic_expr(ctx, residual_ratio, var)?;
    let residual = cas_ast::hold::wrap_hold(ctx, residual);

    Some(ctx.add(Expr::Sub(leading, residual)))
}

pub fn integrate_symbolic_is_quadratic_times_affine_ln_by_parts_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    // Historical public name: this recognizer now covers the same bounded
    // affine-log by-parts family through a small polynomial cofactor cap.
    quadratic_times_affine_ln_by_parts_antiderivative(ctx, expr, var).is_some()
        || additive_quadratic_times_affine_ln_by_parts_antiderivative(ctx, expr, var).is_some()
}

pub fn integrate_symbolic_is_quadratic_times_positive_quadratic_ln_by_parts_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    positive_quadratic_ln_by_parts_antiderivative(ctx, expr, var).is_some()
        || low_degree_times_positive_quadratic_ln_by_parts_antiderivative(ctx, expr, var).is_some()
        || additive_positive_quadratic_ln_by_parts_antiderivative(ctx, expr, var).is_some()
}

pub(super) fn natural_log_power_factor_parts(
    ctx: &Context,
    factor: ExprId,
) -> Option<(ExprId, ExprId, u32)> {
    let (log_expr, power) = match ctx.get(factor) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Ln) =>
        {
            (factor, 1)
        }
        Expr::Pow(base, exp) => {
            let power = positive_integer_power_value(ctx, *exp)?;
            let Expr::Function(fn_id, args) = ctx.get(*base) else {
                return None;
            };
            if args.len() != 1 || ctx.builtin_of(*fn_id) != Some(BuiltinFn::Ln) {
                return None;
            }
            (*base, power)
        }
        _ => return None,
    };

    let Expr::Function(_, args) = ctx.get(log_expr) else {
        return None;
    };
    let log_arg = args[0];
    let log_base = extract_abs_argument_view(ctx, log_arg).unwrap_or(log_arg);
    Some((log_expr, log_base, power))
}

fn log_power_function_parts(
    ctx: &mut Context,
    log_expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(log_expr).clone() else {
        return None;
    };

    match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Ln) if args.len() == 1 => {
            let arg = args[0];
            let one = ctx.num(1);
            Some((arg, arg, one))
        }
        Some(BuiltinFn::Log) if args.len() == 2 => {
            let base = args[0];
            let arg = args[1];
            let base_ln = valid_constant_log_base_ln(ctx, base)?;
            let correction = constant_base_log_derivative_correction(ctx, base_ln);
            Some((
                arg,
                extract_abs_argument_view(ctx, arg).unwrap_or(arg),
                correction,
            ))
        }
        Some(BuiltinFn::Log2) if args.len() == 1 => {
            let correction = positive_integer_constant_log_base_derivative_correction(ctx, 2);
            let arg = args[0];
            Some((
                arg,
                extract_abs_argument_view(ctx, arg).unwrap_or(arg),
                correction,
            ))
        }
        Some(BuiltinFn::Log10) if args.len() == 1 => {
            let correction = positive_integer_constant_log_base_derivative_correction(ctx, 10);
            let arg = args[0];
            Some((
                arg,
                extract_abs_argument_view(ctx, arg).unwrap_or(arg),
                correction,
            ))
        }
        _ => None,
    }
}

pub(super) fn log_power_substitution_factor_parts(
    ctx: &mut Context,
    factor: ExprId,
) -> Option<(ExprId, ExprId, ExprId, u32)> {
    let (log_expr, power) = match ctx.get(factor).clone() {
        Expr::Function(_, _) => (factor, 1),
        Expr::Pow(base, exp) => {
            let power = positive_integer_power_value(ctx, exp)?;
            (base, power)
        }
        _ => return None,
    };

    let (log_arg, log_base, correction) = log_power_function_parts(ctx, log_expr)?;
    if extract_abs_argument_view(ctx, log_arg).is_some() {
        return None;
    }
    Some((log_expr, log_base, correction, power))
}

fn polynomial_log_power_by_parts_integral(
    ctx: &mut Context,
    base: ExprId,
    log_expr: ExprId,
    power: u32,
) -> ExprId {
    let mut terms = Vec::new();
    for degree in (0..=power).rev() {
        let mut coeff = descending_factorial_ratio(power, degree);
        if (power - degree) % 2 == 1 {
            coeff = -coeff;
        }

        let term = if degree == 0 {
            ctx.num(1)
        } else {
            log_power_term(ctx, log_expr, degree)
        };
        terms.push(scale_rational_term(ctx, coeff, term));
    }

    let by_parts_factor = build_balanced_add(ctx, &terms);
    mul2_raw(ctx, base, by_parts_factor)
}

pub(super) fn polynomial_log_power_by_parts_integral_with_correction(
    ctx: &mut Context,
    base: ExprId,
    log_expr: ExprId,
    correction: ExprId,
    power: u32,
) -> ExprId {
    if is_number(ctx, correction, 1) {
        return polynomial_log_power_by_parts_integral(ctx, base, log_expr, power);
    }

    let mut terms = Vec::new();
    for degree in (0..=power).rev() {
        let mut coeff = descending_factorial_ratio(power, degree);
        if (power - degree) % 2 == 1 {
            coeff = -coeff;
        }

        let correction_degree = power - degree;
        let correction_term = log_derivative_correction_power(ctx, correction, correction_degree);
        let mut factors = Vec::new();
        if degree != 0 {
            factors.push(log_power_term(ctx, log_expr, degree));
        }
        if !is_number(ctx, correction_term, 1) {
            factors.push(correction_term);
        }

        let term = if factors.is_empty() {
            ctx.num(1)
        } else {
            build_balanced_mul(ctx, &factors)
        };
        terms.push(scale_log_power_term(ctx, coeff, term));
    }

    let by_parts_factor = build_balanced_add(ctx, &terms);
    mul2_raw(ctx, base, by_parts_factor)
}

/// integrate(p(x) / e^(a*x+b), x): the simplifier normalizes
/// p(x)*e^(-(a*x+b)) into this Div shape, so the by-parts exponential
/// family must be reachable from it. The integrand is rebuilt as the
/// equivalent product and delegated to the existing by-parts handlers.
pub(super) fn div_exp_linear_by_parts_antiderivative(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    var: &str,
) -> Option<ExprId> {
    let exponent = exp_like_arg(ctx, denominator)?;
    if !contains_named_var(ctx, numerator, var) {
        return None;
    }
    let poly = Polynomial::from_expr(ctx, exponent, var).ok()?;
    nonzero_linear_polynomial_slope(&poly)?;
    let negated_exponent = ctx.add(Expr::Neg(exponent));
    let exp_factor = ctx.call_builtin(BuiltinFn::Exp, vec![negated_exponent]);
    let product = mul2_raw(ctx, numerator, exp_factor);
    polynomial_times_exp_linear_antiderivative(ctx, product, var)
        .or_else(|| linear_times_exp_linear_antiderivative(ctx, product, var))
        .or_else(|| exp_trig_same_linear_antiderivative(ctx, product, var))
}

pub(super) fn linear_exp_factor_parts(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Result<Option<(ExprId, BigRational)>, ()> {
    let Some(exp_arg) = exp_like_arg(ctx, expr) else {
        return Ok(None);
    };
    let Some((_arg_poly, arg_slope)) = nonzero_linear_polynomial_from_expr(ctx, exp_arg, var)?
    else {
        return Ok(None);
    };
    Ok(Some((exp_arg, arg_slope)))
}

pub(super) fn linear_trig_factor_parts(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Result<Option<(BuiltinFn, ExprId, BigRational)>, ()> {
    let Some((builtin, trig_arg)) = trig_like_factor(ctx, expr) else {
        return Ok(None);
    };
    let Some((_arg_poly, arg_slope)) = nonzero_linear_polynomial_from_expr(ctx, trig_arg, var)?
    else {
        return Ok(None);
    };
    Ok(Some((builtin, trig_arg, arg_slope)))
}

fn linear_exp_polynomial_product_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    cofactor_failure: ExpByPartsCofactorFailure,
    mut cofactor_matches: impl FnMut(&Polynomial) -> bool,
) -> Result<Option<LinearExpPolynomialProductParts>, ()> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return Ok(None);
    }

    for (exp_index, factor) in factors.iter().enumerate() {
        let Some((exp_arg, arg_slope)) = linear_exp_factor_parts(ctx, *factor, var)? else {
            continue;
        };

        let cofactor_poly = match polynomial_cofactor_excluding_index(ctx, &factors, exp_index, var)
        {
            Ok(cofactor_poly) => cofactor_poly,
            Err(()) => match cofactor_failure {
                ExpByPartsCofactorFailure::Stop => return Err(()),
                ExpByPartsCofactorFailure::Skip => continue,
            },
        };

        if !cofactor_matches(&cofactor_poly) {
            continue;
        }

        return Ok(Some(LinearExpPolynomialProductParts {
            exp_factor: *factor,
            exp_arg,
            arg_slope,
            cofactor_poly,
        }));
    }

    Ok(None)
}

pub(super) fn linear_exp_linear_product_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Result<Option<LinearExpPolynomialProductParts>, ()> {
    linear_exp_polynomial_product_parts(
        ctx,
        expr,
        var,
        ExpByPartsCofactorFailure::Stop,
        |cofactor_poly| nonzero_linear_polynomial_slope(cofactor_poly).is_some(),
    )
}

fn polynomial_exp_by_parts_cofactor_degree(poly: &Polynomial) -> bool {
    (2..=MAX_EXP_POLYNOMIAL_BY_PARTS_DEGREE).contains(&poly.degree())
}

pub(super) fn polynomial_exp_linear_product_parts(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    cofactor_failure: ExpByPartsCofactorFailure,
) -> Result<Option<LinearExpPolynomialProductParts>, ()> {
    linear_exp_polynomial_product_parts(
        ctx,
        expr,
        var,
        cofactor_failure,
        polynomial_exp_by_parts_cofactor_degree,
    )
}

pub(super) fn polynomial_exp_by_parts_inner(
    poly: &Polynomial,
    arg_slope: &BigRational,
) -> Polynomial {
    let mut inner = Polynomial::zero(poly.var.clone());
    let mut derivative = poly.clone();

    for order in 0..=poly.degree() {
        if derivative.is_zero() {
            break;
        }
        let denominator = positive_integer_power_rational(arg_slope, (order + 1) as u32);
        let term = derivative.div_scalar(&denominator);
        inner = if order % 2 == 0 {
            inner.add(&term)
        } else {
            inner.sub(&term)
        };
        derivative = derivative.derivative();
    }

    inner
}

pub(super) fn polynomial_trig_by_parts_polys(
    poly: &Polynomial,
    builtin: BuiltinFn,
    arg_slope: &BigRational,
) -> Option<(Polynomial, Polynomial)> {
    let mut sin_poly = Polynomial::zero(poly.var.clone());
    let mut cos_poly = Polynomial::zero(poly.var.clone());
    let mut derivative = poly.clone();

    for order in 0..=poly.degree() {
        if derivative.is_zero() {
            break;
        }
        let denominator = positive_integer_power_rational(arg_slope, (order + 1) as u32);
        let term = derivative.div_scalar(&denominator);

        match builtin {
            BuiltinFn::Sin if order % 2 == 0 => {
                let positive = (order / 2) % 2 == 1;
                cos_poly = add_polynomial_term(cos_poly, &term, positive);
            }
            BuiltinFn::Sin => {
                let positive = ((order - 1) / 2) % 2 == 0;
                sin_poly = add_polynomial_term(sin_poly, &term, positive);
            }
            BuiltinFn::Cos if order % 2 == 0 => {
                let positive = (order / 2) % 2 == 0;
                sin_poly = add_polynomial_term(sin_poly, &term, positive);
            }
            BuiltinFn::Cos => {
                let positive = ((order - 1) / 2) % 2 == 0;
                cos_poly = add_polynomial_term(cos_poly, &term, positive);
            }
            _ => return None,
        }

        derivative = derivative.derivative();
    }

    Some((sin_poly, cos_poly))
}

pub(super) fn trig_by_parts_quotient_by_slope(
    ctx: &mut Context,
    numerator: ExprId,
    slope: &BigRational,
) -> ExprId {
    if slope.is_one() {
        numerator
    } else if slope.is_integer() {
        let slope_expr = ctx.add(Expr::Number(slope.clone()));
        ctx.add(Expr::Div(numerator, slope_expr))
    } else {
        scale_factor(ctx, BigRational::one() / slope.clone(), numerator)
    }
}

pub(super) fn polynomial_trig_linear_term_parts(
    ctx: &mut Context,
    term: ExprId,
    var: &str,
) -> Option<(BuiltinFn, ExprId, ExprId, Polynomial)> {
    let (outer_sign, factors) = signed_mul_leaves(ctx, term);

    for (trig_index, factor) in factors.iter().enumerate() {
        let Some(trig_parts) =
            signed_linear_function_factor_parts(ctx, *factor, var, signed_trig_like_factor)
        else {
            continue;
        };

        let effective_sign = combine_factor_signs(outer_sign, trig_parts.sign);
        let cofactor =
            signed_factor_product_excluding_index(ctx, &factors, trig_index, effective_sign);
        let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;

        return Some((
            trig_parts.builtin,
            trig_parts.arg,
            trig_parts.factor,
            cofactor_poly,
        ));
    }

    None
}

pub(super) fn polynomial_hyperbolic_linear_term_parts(
    ctx: &mut Context,
    term: ExprId,
    var: &str,
) -> Option<(BuiltinFn, ExprId, ExprId, Polynomial)> {
    let (outer_sign, factors) = signed_mul_leaves(ctx, term);

    for (hyperbolic_index, factor) in factors.iter().enumerate() {
        let Some(hyperbolic_parts) =
            signed_linear_function_factor_parts(ctx, *factor, var, signed_hyperbolic_like_factor)
        else {
            continue;
        };

        let effective_sign = combine_factor_signs(outer_sign, hyperbolic_parts.sign);
        let cofactor =
            signed_factor_product_excluding_index(ctx, &factors, hyperbolic_index, effective_sign);
        let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;

        return Some((
            hyperbolic_parts.builtin,
            hyperbolic_parts.arg,
            hyperbolic_parts.factor,
            cofactor_poly,
        ));
    }

    None
}

pub(super) fn arcsin_polynomial_substitution_from_parts(
    ctx: &mut Context,
    numerator: &Polynomial,
    arg_poly: Polynomial,
    offset_square: BigRational,
    radicand_scale: Option<BigRational>,
) -> Option<ExprId> {
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let (arg_poly, offset_square) = normalize_surd_ratio_arg(arg_poly, offset_square);
    let offset_expr = positive_rational_sqrt_expr(ctx, &offset_square)?;

    let raw_arg =
        compact_polynomial_square_arg(ctx, &arg_poly).unwrap_or_else(|| arg_poly.to_expr(ctx));
    let arg = compact_single_power_polynomial_arg(ctx, raw_arg);
    let arcsin_arg = if offset_square.is_one() {
        arg
    } else {
        ctx.add(Expr::Div(arg, offset_expr))
    };
    let arcsin = ctx.call_builtin(BuiltinFn::Arcsin, vec![arcsin_arg]);
    let scaled_arcsin = if scale.is_one() {
        arcsin
    } else {
        let scale_expr = ctx.add(Expr::Number(scale));
        mul2_raw(ctx, scale_expr, arcsin)
    };

    if let Some(radicand_scale) = radicand_scale {
        let scale_sqrt = positive_rational_sqrt_expr(ctx, &radicand_scale)?;
        return Some(ctx.add(Expr::Div(scaled_arcsin, scale_sqrt)));
    }

    Some(scaled_arcsin)
}

pub(super) fn affine_sqrt_product_derivative_from_parts(
    ctx: &mut Context,
    numerator: Polynomial,
    radicand: ExprId,
    var: &str,
) -> Option<ExprId> {
    let radicand_poly = Polynomial::from_expr(ctx, radicand, var).ok()?;
    let factor = affine_sqrt_product_derivative_solution(&radicand_poly, &numerator)?;
    let factor_expr = factor.to_expr(ctx);
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    Some(mul2_raw(ctx, factor_expr, sqrt_radicand))
}

pub(super) fn affine_sqrt_product_derivative_div_parts(
    ctx: &Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<(Polynomial, ExprId)> {
    let denominator_factors = mul_leaves(ctx, den);
    let (sqrt_index, radicand) =
        denominator_factors
            .iter()
            .enumerate()
            .find_map(|(idx, factor)| {
                sqrt_like_radicand(ctx, *factor).map(|radicand| (idx, radicand))
            })?;

    let mut denominator_scale = BigRational::one();
    for (idx, factor) in denominator_factors.iter().enumerate() {
        if idx == sqrt_index {
            continue;
        }
        denominator_scale *= rational_constant_value(ctx, *factor)?;
    }
    if denominator_scale.is_zero() {
        return None;
    }

    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let two = BigRational::from_integer(2.into());
    let numerator = scale_polynomial(&numerator.div_scalar(&denominator_scale), two);
    Some((numerator, radicand))
}

pub(super) fn affine_sqrt_product_derivative_product_parts(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(Polynomial, ExprId)> {
    let factors = mul_leaves(ctx, expr);
    let (sqrt_index, radicand) = factors.iter().enumerate().find_map(|(idx, factor)| {
        reciprocal_sqrt_like_radicand(ctx, *factor).map(|radicand| (idx, radicand))
    })?;

    let mut cofactor = Polynomial::one(var.to_string());
    for (idx, factor) in factors.iter().enumerate() {
        if idx == sqrt_index {
            continue;
        }
        cofactor = cofactor.mul(&Polynomial::from_expr(ctx, *factor, var).ok()?);
    }

    let two = BigRational::from_integer(2.into());
    Some((scale_polynomial(&cofactor, two), radicand))
}

pub(super) fn sqrt_hyperbolic_reciprocal_parts_radicand(
    parts: &SqrtHyperbolicReciprocalParts,
) -> ExprId {
    parts.radicand
}

pub(super) fn atanh_polynomial_substitution_target_parts(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<()> {
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

    Some(())
}
