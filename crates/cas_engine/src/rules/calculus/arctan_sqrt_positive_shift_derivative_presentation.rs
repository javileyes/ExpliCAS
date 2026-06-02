use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

use super::inverse_tangent_root_args::arctan_sqrt_radicand_arg;
use super::presentation_utils::{calculus_sqrt_like_radicand, unwrap_internal_hold_for_calculus};
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    split_numeric_scale_single_core, split_outer_numeric_mul_for_calculus_presentation,
};

pub(super) fn arctan_sqrt_plus_sqrt_over_x_plus_one_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (scale, offset) =
        if let Some(parts) = arctan_sqrt_positive_shift_primitive_parts(ctx, target, var_name) {
            (parts.primitive_scale, parts.offset)
        } else {
            (
                arctan_sqrt_plus_sqrt_over_x_plus_one_scale(ctx, target, var_name)?,
                BigRational::one(),
            )
        };
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&scale)?;

    let var = ctx.var(var_name);
    let neg_half = ctx.rational(-1, 2);
    let reciprocal_sqrt_var = ctx.add(Expr::Pow(var, neg_half));
    let offset_expr = rational_const_for_calculus_presentation(ctx, offset);
    let unit_shift = ctx.add(Expr::Add(var, offset_expr));
    let two = ctx.num(2);
    let unit_shift_square = ctx.add(Expr::Pow(unit_shift, two));
    let numerator = if numerator_coeff == BigRational::one() {
        reciprocal_sqrt_var
    } else {
        let numerator_coeff = rational_const_for_calculus_presentation(ctx, numerator_coeff);
        ctx.add_raw(Expr::Mul(numerator_coeff, reciprocal_sqrt_var))
    };
    let denominator = if denominator_coeff == BigRational::one() {
        unit_shift_square
    } else {
        let denominator_coeff = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        ctx.add_raw(Expr::Mul(denominator_coeff, unit_shift_square))
    };
    let result = ctx.add_raw(Expr::Div(numerator, denominator));
    let var = ctx.var(var_name);
    Some((result, vec![crate::ImplicitCondition::Positive(var)]))
}

pub(super) fn compact_sqrt_var_over_var_times_positive_shift_square_diff_result(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (num_scale, shift_square, denominator_scale) =
        sqrt_var_over_var_times_positive_shift_square_parts(ctx, expr, var_name)?;
    if denominator_scale.is_zero() {
        return None;
    }
    let coefficient = num_scale / denominator_scale;
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let var = ctx.var(var_name);
    let neg_half = ctx.rational(-1, 2);
    let reciprocal_sqrt_var = ctx.add(Expr::Pow(var, neg_half));
    let numerator = if numerator_coeff == BigRational::one() {
        reciprocal_sqrt_var
    } else {
        let numerator_coeff = rational_const_for_calculus_presentation(ctx, numerator_coeff);
        ctx.add_raw(Expr::Mul(numerator_coeff, reciprocal_sqrt_var))
    };
    let denominator = if denominator_coeff == BigRational::one() {
        shift_square
    } else {
        let denominator_coeff = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        ctx.add_raw(Expr::Mul(denominator_coeff, shift_square))
    };
    let result = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(result)
}

fn sqrt_var_over_var_times_positive_shift_square_parts(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, ExprId, BigRational)> {
    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        let (num_scale, num_core) = split_numeric_scale_single_core(ctx, num)?;
        let radicand = calculus_sqrt_like_radicand(ctx, num_core)?;
        if !is_calculus_var(ctx, radicand, var_name) {
            return None;
        }
        let (shift_square, denominator_scale) =
            var_times_positive_shift_square_denominator_parts(ctx, den, var_name)?;
        return Some((num_scale, shift_square, denominator_scale));
    }

    let mut num_scale = BigRational::one();
    let mut saw_sqrt_var = false;
    let mut denominator_factors = Vec::new();

    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            num_scale *= value;
        } else if calculus_sqrt_like_radicand(ctx, factor)
            .is_some_and(|radicand| is_calculus_var(ctx, radicand, var_name))
        {
            if saw_sqrt_var {
                return None;
            }
            saw_sqrt_var = true;
        } else if let Expr::Pow(base, exp) = ctx.get(factor) {
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                != Some(BigRational::new((-1).into(), 1.into()))
            {
                return None;
            }
            denominator_factors.push(*base);
        } else {
            return None;
        }
    }

    if !saw_sqrt_var || denominator_factors.is_empty() {
        return None;
    }
    let denominator_factors = denominator_factors
        .into_iter()
        .flat_map(|factor| cas_math::expr_nary::mul_leaves(ctx, factor))
        .collect::<Vec<_>>();
    let (shift_square, denominator_scale) =
        var_times_positive_shift_square_denominator_factor_parts(
            ctx,
            denominator_factors,
            var_name,
        )?;
    Some((num_scale, shift_square, denominator_scale))
}

fn var_times_positive_shift_square_denominator_parts(
    ctx: &Context,
    den: ExprId,
    var_name: &str,
) -> Option<(ExprId, BigRational)> {
    var_times_positive_shift_square_denominator_factor_parts(
        ctx,
        cas_math::expr_nary::mul_leaves(ctx, den).to_vec(),
        var_name,
    )
}

fn var_times_positive_shift_square_denominator_factor_parts(
    ctx: &Context,
    factors: Vec<ExprId>,
    var_name: &str,
) -> Option<(ExprId, BigRational)> {
    let mut denominator_scale = BigRational::one();
    let mut saw_var_factor = false;
    let mut shift_square = None;
    for factor in factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            denominator_scale *= value;
        } else if is_calculus_var(ctx, factor, var_name) {
            if saw_var_factor {
                return None;
            }
            saw_var_factor = true;
        } else if positive_shift_square_factor_for_calculus_presentation(ctx, factor, var_name)
            .is_some()
        {
            if shift_square.replace(factor).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }

    saw_var_factor.then_some((shift_square?, denominator_scale))
}

fn positive_shift_square_factor_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if cas_ast::views::as_rational_const(ctx, *exp, 8) != Some(BigRational::from_integer(2.into()))
    {
        return None;
    }
    positive_shift_denominator_scale(ctx, *base, var_name).map(|(_, offset)| offset)
}

struct ArctanSqrtPositiveShiftPrimitiveParts {
    primitive_scale: BigRational,
    offset: BigRational,
}

fn arctan_sqrt_positive_shift_primitive_parts(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ArctanSqrtPositiveShiftPrimitiveParts> {
    if let Some((outer_scale, core)) = scaled_nontrivial_core_for_calculus_presentation(ctx, target)
    {
        let inner = arctan_sqrt_positive_shift_primitive_parts(ctx, core, var_name)?;
        return Some(ArctanSqrtPositiveShiftPrimitiveParts {
            primitive_scale: outer_scale * inner.primitive_scale,
            offset: inner.offset,
        });
    }

    if let Some(parts) = arctan_sqrt_positive_shift_combined_quotient_parts(ctx, target, var_name) {
        return Some(parts);
    }

    let terms = cas_math::expr_nary::add_terms_signed(ctx, target);
    if terms.len() != 2 {
        return None;
    }

    let mut arctan_scale_and_offset = None;
    let mut quotient_scale_and_offset = None;
    for (term, sign) in terms {
        if sign != cas_math::expr_nary::Sign::Pos {
            return None;
        }
        if let Some(parts) = scaled_arctan_sqrt_positive_shift_term(ctx, term, var_name) {
            if arctan_scale_and_offset.replace(parts).is_some() {
                return None;
            }
        } else if let Some(parts) = scaled_sqrt_var_over_positive_shift_term(ctx, term, var_name) {
            if quotient_scale_and_offset.replace(parts).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }

    let (arctan_scale, offset, offset_root) = arctan_scale_and_offset?;
    let (quotient_scale, quotient_offset) = quotient_scale_and_offset?;
    if offset != quotient_offset {
        return None;
    }

    let primitive_scale_from_arctan = arctan_scale * offset.clone() * offset_root;
    let primitive_scale_from_quotient = quotient_scale * offset.clone();
    (primitive_scale_from_arctan == primitive_scale_from_quotient).then_some(
        ArctanSqrtPositiveShiftPrimitiveParts {
            primitive_scale: primitive_scale_from_arctan,
            offset,
        },
    )
}

fn arctan_sqrt_positive_shift_combined_quotient_parts(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ArctanSqrtPositiveShiftPrimitiveParts> {
    let target = unwrap_internal_hold_for_calculus(ctx, target);
    let Expr::Div(num, den) = ctx.get(target).clone() else {
        return None;
    };
    let (den_scale, denominator_offset) = positive_shift_denominator_scale(ctx, den, var_name)?;
    if den_scale.is_zero() {
        return None;
    }

    let terms = cas_math::expr_nary::add_terms_signed(ctx, num);
    if terms.len() != 2 {
        return None;
    }

    let mut sqrt_scale = None;
    let mut arctan_linear = None;
    for (term, sign) in terms {
        if sign != cas_math::expr_nary::Sign::Pos {
            return None;
        }
        if let Some(scale) = scaled_sqrt_var_term(ctx, term, var_name) {
            if sqrt_scale.replace(scale).is_some() {
                return None;
            }
        } else if let Some(parts) =
            scaled_positive_shift_times_arctan_sqrt_term(ctx, term, var_name)
        {
            if arctan_linear.replace(parts).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }

    let quotient_offset = denominator_offset;
    let quotient_scale = sqrt_scale? / den_scale.clone();
    let (arctan_linear_scale, arctan_offset, arctan_offset_root) = arctan_linear?;
    if quotient_offset != arctan_offset {
        return None;
    }
    let arctan_scale = arctan_linear_scale / den_scale;
    let primitive_scale_from_quotient = quotient_scale * quotient_offset.clone();
    let primitive_scale_from_arctan = arctan_scale * quotient_offset.clone() * arctan_offset_root;
    (primitive_scale_from_quotient == primitive_scale_from_arctan).then_some(
        ArctanSqrtPositiveShiftPrimitiveParts {
            primitive_scale: primitive_scale_from_quotient,
            offset: quotient_offset,
        },
    )
}

fn scaled_positive_shift_times_arctan_sqrt_term(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational, BigRational)> {
    let mut scale = BigRational::one();
    let mut offset = None;
    let mut offset_root = None;
    let mut saw_arctan = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else if let Some((factor_offset, factor_root)) =
            arctan_sqrt_positive_shift_arg(ctx, factor, var_name)
        {
            if saw_arctan {
                return None;
            }
            saw_arctan = true;
            offset_root = Some(factor_root);
            match &offset {
                Some(existing) if existing != &factor_offset => return None,
                Some(_) => {}
                None => offset = Some(factor_offset),
            }
        } else if let Some((linear_scale, linear_offset)) =
            positive_shift_denominator_scale(ctx, factor, var_name)
        {
            match &offset {
                Some(existing) if existing != &linear_offset => return None,
                Some(_) => {}
                None => offset = Some(linear_offset),
            }
            scale *= linear_scale;
        } else {
            return None;
        }
    }
    let offset = offset?;
    let offset_root = offset_root?;
    saw_arctan.then_some((scale, offset, offset_root))
}

fn scaled_arctan_sqrt_positive_shift_term(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational, BigRational)> {
    let (scale, core) = split_numeric_scale_single_core(ctx, expr)?;
    let (offset, offset_root) = arctan_sqrt_positive_shift_arg(ctx, core, var_name)?;
    Some((scale, offset, offset_root))
}

fn arctan_sqrt_positive_shift_arg(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(fn_id),
            Some(BuiltinFn::Atan | BuiltinFn::Arctan)
        )
    {
        return None;
    }
    let (sqrt_scale, sqrt_core) = split_numeric_scale_single_core(ctx, args[0])?;
    if !sqrt_scale.is_positive() {
        return None;
    }
    let radicand = calculus_sqrt_like_radicand(ctx, sqrt_core)?;
    if !is_calculus_var(ctx, radicand, var_name) {
        return None;
    }
    let offset_root = BigRational::one() / sqrt_scale;
    let offset = offset_root.clone() * offset_root.clone();
    Some((offset, offset_root))
}

fn scaled_sqrt_var_over_positive_shift_term(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    let (outer_scale, core) = split_outer_numeric_mul_for_calculus_presentation(ctx, expr)?;
    let core = unwrap_internal_hold_for_calculus(ctx, core);
    let Expr::Div(num, den) = ctx.get(core).clone() else {
        return None;
    };
    let (num_scale, num_core) = split_numeric_scale_single_core(ctx, num)?;
    let radicand = calculus_sqrt_like_radicand(ctx, num_core)?;
    if !is_calculus_var(ctx, radicand, var_name) {
        return None;
    }
    let (den_scale, offset) = positive_shift_denominator_scale(ctx, den, var_name)?;
    Some((outer_scale * num_scale / den_scale, offset))
}

fn positive_shift_denominator_scale(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(BigRational, BigRational)> {
    let mut scale = BigRational::one();
    let mut offset = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
            continue;
        }
        let poly = Polynomial::from_expr(ctx, factor, var_name).ok()?;
        if poly.degree() != 1 {
            return None;
        }
        let constant = poly.coeffs.first()?.clone();
        let slope = poly.coeffs.get(1)?;
        if !slope.is_positive() {
            return None;
        }
        let candidate_offset = constant / slope.clone();
        if !candidate_offset.is_positive() {
            return None;
        }
        scale *= slope.clone();
        if offset.replace(candidate_offset).is_some() {
            return None;
        }
    }
    Some((scale, offset?))
}

fn arctan_sqrt_plus_sqrt_over_x_plus_one_scale(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    if let Some((outer_scale, core)) = scaled_nontrivial_core_for_calculus_presentation(ctx, target)
    {
        if let Some(inner_scale) = arctan_sqrt_plus_sqrt_over_x_plus_one_scale(ctx, core, var_name)
        {
            return Some(outer_scale * inner_scale);
        }
    }

    if let Some(scale) =
        arctan_sqrt_plus_sqrt_over_x_plus_one_combined_quotient_scale(ctx, target, var_name)
    {
        return Some(scale);
    }

    let terms = cas_math::expr_nary::add_terms_signed(ctx, target);
    if terms.len() != 2 {
        return None;
    }

    let mut arctan_scale = None;
    let mut rational_scale = None;
    for (term, sign) in terms {
        if sign != cas_math::expr_nary::Sign::Pos {
            return None;
        }
        if let Some(scale) = scaled_arctan_sqrt_var_term(ctx, term, var_name) {
            if arctan_scale.replace(scale).is_some() {
                return None;
            }
        } else if let Some(scale) = scaled_sqrt_var_over_x_plus_one_term(ctx, term, var_name) {
            if rational_scale.replace(scale).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }

    let arctan_scale = arctan_scale?;
    let rational_scale = rational_scale?;
    (arctan_scale == rational_scale).then_some(arctan_scale)
}

fn scaled_nontrivial_core_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    let (scale, core) = split_numeric_scale_single_core(ctx, expr)?;
    (!scale.is_one() && core != expr).then_some((scale, core))
}

fn arctan_sqrt_plus_sqrt_over_x_plus_one_combined_quotient_scale(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let target = unwrap_internal_hold_for_calculus(ctx, target);
    let Expr::Div(num, den) = ctx.get(target).clone() else {
        return None;
    };
    let den_scale = x_plus_one_linear_scale_for_calculus_presentation(ctx, den, var_name)?;
    if den_scale.is_zero() {
        return None;
    }
    let num_scale =
        arctan_sqrt_plus_sqrt_over_x_plus_one_combined_numerator_scale(ctx, num, var_name)?;
    Some(num_scale / den_scale)
}

fn arctan_sqrt_plus_sqrt_over_x_plus_one_combined_numerator_scale(
    ctx: &mut Context,
    numerator: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, numerator);
    if terms.len() != 3 {
        return None;
    }

    let mut arctan_scale = None;
    let mut sqrt_scale = None;
    let mut x_arctan_scale = None;
    for (term, sign) in terms {
        if sign != cas_math::expr_nary::Sign::Pos {
            return None;
        }
        if let Some(scale) = scaled_arctan_sqrt_var_term(ctx, term, var_name) {
            if arctan_scale.replace(scale).is_some() {
                return None;
            }
        } else if let Some(scale) = scaled_sqrt_var_term(ctx, term, var_name) {
            if sqrt_scale.replace(scale).is_some() {
                return None;
            }
        } else if let Some(scale) = scaled_var_times_arctan_sqrt_var_term(ctx, term, var_name) {
            if x_arctan_scale.replace(scale).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }

    let arctan_scale = arctan_scale?;
    (sqrt_scale? == arctan_scale && x_arctan_scale? == arctan_scale).then_some(arctan_scale)
}

fn scaled_arctan_sqrt_var_term(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let (scale, core) = split_numeric_scale_single_core(ctx, expr)?;
    arctan_sqrt_radicand_arg(ctx, core)
        .is_some_and(|radicand| is_calculus_var(ctx, radicand, var_name))
        .then_some(scale)
}

fn scaled_sqrt_var_term(ctx: &mut Context, expr: ExprId, var_name: &str) -> Option<BigRational> {
    let (scale, core) = split_numeric_scale_single_core(ctx, expr)?;
    let radicand = calculus_sqrt_like_radicand(ctx, core)?;
    is_calculus_var(ctx, radicand, var_name).then_some(scale)
}

fn scaled_var_times_arctan_sqrt_var_term(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let mut scale = BigRational::one();
    let mut saw_var = false;
    let mut saw_arctan = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else if is_calculus_var(ctx, factor, var_name) {
            if saw_var {
                return None;
            }
            saw_var = true;
        } else if arctan_sqrt_radicand_arg(ctx, factor)
            .is_some_and(|radicand| is_calculus_var(ctx, radicand, var_name))
        {
            if saw_arctan {
                return None;
            }
            saw_arctan = true;
        } else {
            return None;
        }
    }
    (saw_var && saw_arctan).then_some(scale)
}

fn scaled_sqrt_var_over_x_plus_one_term(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let (outer_scale, core) = split_numeric_scale_single_core(ctx, expr)?;
    let core = unwrap_internal_hold_for_calculus(ctx, core);
    let Expr::Div(num, den) = ctx.get(core).clone() else {
        return None;
    };
    if !is_x_plus_one_for_calculus_presentation(ctx, den, var_name) {
        return None;
    }

    let (num_scale, num_core) = split_numeric_scale_single_core(ctx, num)?;
    let radicand = calculus_sqrt_like_radicand(ctx, num_core)?;
    is_calculus_var(ctx, radicand, var_name).then_some(outer_scale * num_scale)
}

fn x_plus_one_linear_scale_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<BigRational> {
    let poly = Polynomial::from_expr(ctx, expr, var_name).ok()?;
    if poly.degree() != 1 {
        return None;
    }
    let offset = poly.coeffs.first()?;
    let slope = poly.coeffs.get(1)?;
    (offset == slope).then_some(offset.clone())
}

fn is_calculus_var(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var_name)
}

fn is_x_plus_one_for_calculus_presentation(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    let poly = Polynomial::from_expr(ctx, expr, var_name).ok();
    poly.is_some_and(|poly| {
        poly.degree() == 1
            && poly
                .coeffs
                .first()
                .is_some_and(|offset| offset == &BigRational::one())
            && poly
                .coeffs
                .get(1)
                .is_some_and(|slope| slope == &BigRational::one())
    })
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::arctan_sqrt_plus_sqrt_over_x_plus_one_derivative_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn post_diff_presentation_compacts_sqrt_over_var_shift_square() {
        let mut ctx = Context::new();
        let target = parse("arctan(sqrt(x)) + sqrt(x)/(x+1)", &mut ctx).unwrap();
        let (direct, _required) =
            arctan_sqrt_plus_sqrt_over_x_plus_one_derivative_presentation(&mut ctx, target, "x")
                .unwrap();

        assert_eq!(rendered(&ctx, direct), "1 / ((x + 1)^2 * sqrt(x))");

        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(
            "diff(arctan(sqrt(x)) + sqrt(x)/(x+1), x)",
            &mut simplifier.context,
        )
        .unwrap();
        let (result, _steps) = simplifier.simplify(expr);

        assert_eq!(
            rendered(&simplifier.context, result),
            "1 / ((x + 1)^2 * sqrt(x))"
        );

        let expr = parse(
            "diff(8*arctan(2*sqrt(x)) + 4*sqrt(x)/(x+1/4), x)",
            &mut simplifier.context,
        )
        .unwrap();
        let (result, _steps) = simplifier.simplify(expr);

        assert_eq!(
            rendered(&simplifier.context, result),
            "1 / ((x + 1/4)^2 * sqrt(x))"
        );
    }
}
