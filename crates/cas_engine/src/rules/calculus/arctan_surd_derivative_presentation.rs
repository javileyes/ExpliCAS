use super::differentiation::differentiate;
use super::gap_presentation::squared_expr_for_compact_gap_presentation;
use super::polynomial_support::{
    polynomial_derivative_expr_for_calculus_presentation,
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::presentation_utils::multiply_by_sqrt_factor_for_calculus_presentation;
use super::result_presentation::scale_compact_derivative_by_rational;
use super::scalar_presentation::{
    rational_const_for_calculus_presentation, rational_scaled_single_factor,
    scale_expr_for_calculus_presentation, signed_numerator_for_calculus_presentation,
};
use super::surd_quotient_args::{
    arctan_self_normalized_surd_quotient_parts, arctan_self_normalized_surd_reciprocal_parts,
    atanh_arg_over_sqrt_parts,
};
use super::surd_quotient_presentation::compact_surd_quotient_polynomial_presentation_parts;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{Signed, Zero};

pub(super) fn constant_scaled_arctan_surd_quotient_scaled_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Expr::Div(numerator, outer_den) = ctx.get(target).clone() {
        let (scale, inner) = rational_scaled_single_factor(ctx, numerator)?;
        let base = ctx.add(Expr::Div(inner, outer_den));
        let derivative = arctan_surd_quotient_scaled_compact_derivative(ctx, base, var_name)?;
        return Some(scale_compact_derivative_by_rational(ctx, derivative, scale));
    }

    let Expr::Mul(_, _) = ctx.get(target).clone() else {
        return None;
    };

    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let derivative = arctan_surd_quotient_scaled_compact_derivative(ctx, inner, var_name)?;
    Some(scale_compact_derivative_by_rational(ctx, derivative, scale))
}

pub(super) fn arctan_surd_quotient_scaled_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (arctan_expr, outer_den) = match ctx.get(target).clone() {
        Expr::Div(arctan_expr, outer_den) => (arctan_expr, outer_den),
        _ => return None,
    };
    let outer_radicand = extract_square_root_base(ctx, outer_den)?;
    let outer_radicand_value = cas_ast::views::as_rational_const(ctx, outer_radicand, 8)?;
    if !outer_radicand_value.is_positive() {
        return None;
    }

    let (fn_id, args) = match ctx.get(arctan_expr).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if !matches!(
        ctx.builtin_of(fn_id),
        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
    ) || args.len() != 1
    {
        return None;
    }

    let (num, inner_radicand) = atanh_arg_over_sqrt_parts(ctx, args[0])?;
    if compare_expr(ctx, outer_radicand, inner_radicand) != std::cmp::Ordering::Equal {
        return None;
    }

    let (d_num, square_base) = if let Some(parts) =
        compact_surd_quotient_polynomial_presentation_parts(ctx, num, var_name)
    {
        parts
    } else {
        (differentiate(ctx, num, var_name)?, num)
    };
    if cas_ast::views::as_rational_const(ctx, d_num, 8).is_some_and(|value| value.is_zero()) {
        return Some(ctx.num(0));
    }
    let num_square = squared_expr_for_compact_gap_presentation(ctx, square_base);
    let denominator = ctx.add(Expr::Add(outer_radicand, num_square));
    Some(ctx.add(Expr::Div(d_num, denominator)))
}

pub(super) fn arctan_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if !matches!(
        ctx.builtin_of(fn_id),
        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
    ) || args.len() != 1
    {
        return None;
    }

    let (num, radicand) = atanh_arg_over_sqrt_parts(ctx, args[0])?;
    let radicand_value = cas_ast::views::as_rational_const(ctx, radicand, 8)?;
    if !radicand_value.is_positive() {
        return None;
    }

    let (d_num, square_base) = if let Some(parts) =
        compact_surd_quotient_polynomial_presentation_parts(ctx, num, var_name)
    {
        parts
    } else {
        let d_num = polynomial_derivative_expr_for_calculus_presentation(ctx, num, var_name)
            .or_else(|| differentiate(ctx, num, var_name))?;
        let (d_num_core, d_num_content) =
            split_polynomial_content_for_calculus_presentation(ctx, d_num);
        let d_num = signed_numerator_for_calculus_presentation(ctx, d_num_content, d_num_core);
        (d_num, num)
    };
    if cas_ast::views::as_rational_const(ctx, d_num, 8).is_some_and(|value| value.is_zero()) {
        return Some(ctx.num(0));
    }
    let num_square = squared_expr_for_compact_gap_presentation(ctx, square_base);
    let denominator = ctx.add(Expr::Add(radicand, num_square));
    let compact_radicand = rational_const_for_calculus_presentation(ctx, radicand_value);
    let numerator = multiply_by_sqrt_factor_for_calculus_presentation(ctx, d_num, compact_radicand);
    let compact = ctx.add(Expr::Div(numerator, denominator));

    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

pub(super) fn arctan_self_normalized_surd_quotient_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if !matches!(
        ctx.builtin_of(fn_id),
        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
    ) || args.len() != 1
    {
        return None;
    }

    let (num, radicand) = arctan_self_normalized_surd_quotient_parts(ctx, args[0])?;
    let num_poly = polynomial_radicand_for_calculus_presentation(ctx, num, var_name)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let gap_poly = radicand_poly.sub(&num_poly.mul(&num_poly));
    if gap_poly.degree() != 0 {
        return None;
    }
    let gap_constant = gap_poly.coeffs.first().cloned()?;
    if !gap_constant.is_positive() {
        return None;
    }

    let (d_num, square_base) =
        compact_surd_quotient_polynomial_presentation_parts(ctx, num, var_name)?;
    if cas_ast::views::as_rational_const(ctx, d_num, 8).is_some_and(|value| value.is_zero()) {
        return Some(ctx.num(0));
    }

    let numerator = scale_expr_for_calculus_presentation(ctx, gap_constant.clone(), d_num);
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let num_square = squared_expr_for_compact_gap_presentation(ctx, square_base);
    let doubled_square =
        scale_expr_for_calculus_presentation(ctx, BigRational::from_integer(2.into()), num_square);
    let gap = rational_const_for_calculus_presentation(ctx, gap_constant);
    let quadratic_factor = ctx.add(Expr::Add(doubled_square, gap));
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, quadratic_factor]);
    let compact = ctx.add(Expr::Div(numerator, denominator));

    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

pub(super) fn arctan_self_normalized_surd_reciprocal_compact_derivative(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, crate::ImplicitCondition)> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if !matches!(
        ctx.builtin_of(fn_id),
        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
    ) || args.len() != 1
    {
        return None;
    }

    let (denominator_arg, radicand) = arctan_self_normalized_surd_reciprocal_parts(ctx, args[0])?;
    let denominator_poly =
        polynomial_radicand_for_calculus_presentation(ctx, denominator_arg, var_name)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let gap_poly = radicand_poly.sub(&denominator_poly.mul(&denominator_poly));
    if gap_poly.degree() != 0 {
        return None;
    }
    let gap_constant = gap_poly.coeffs.first().cloned()?;
    if !gap_constant.is_positive() {
        return None;
    }

    let (d_denominator, square_base) =
        compact_surd_quotient_polynomial_presentation_parts(ctx, denominator_arg, var_name)?;
    if cas_ast::views::as_rational_const(ctx, d_denominator, 8).is_some_and(|value| value.is_zero())
    {
        return Some((
            ctx.num(0),
            crate::ImplicitCondition::NonZero(denominator_arg),
        ));
    }

    let numerator = scale_expr_for_calculus_presentation(ctx, -gap_constant.clone(), d_denominator);
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator_square = squared_expr_for_compact_gap_presentation(ctx, square_base);
    let doubled_square = scale_expr_for_calculus_presentation(
        ctx,
        BigRational::from_integer(2.into()),
        denominator_square,
    );
    let gap = rational_const_for_calculus_presentation(ctx, gap_constant);
    let quadratic_factor = ctx.add(Expr::Add(doubled_square, gap));
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, quadratic_factor]);
    let compact = ctx.add(Expr::Div(numerator, denominator));

    Some((
        cas_ast::hold::wrap_hold(ctx, compact),
        crate::ImplicitCondition::NonZero(denominator_arg),
    ))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::{
        arctan_self_normalized_surd_quotient_compact_derivative,
        arctan_self_normalized_surd_reciprocal_compact_derivative,
        arctan_surd_quotient_compact_derivative, arctan_surd_quotient_scaled_compact_derivative,
        constant_scaled_arctan_surd_quotient_scaled_compact_derivative,
    };

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn arctan_surd_quotient_scaled_compact_derivative_avoids_rationalized_route() {
        let mut ctx = Context::new();
        let expr = parse("arctan((2*x+2)/sqrt(6))/sqrt(6)", &mut ctx).unwrap();
        let derivative =
            arctan_surd_quotient_scaled_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, derivative), "2 / ((2 * x + 2)^2 + 6)");
    }

    #[test]
    fn constant_scaled_arctan_surd_quotient_scaled_derivative_reuses_compact_route() {
        let mut ctx = Context::new();
        let expr = parse("7*arctan((2*x+1)/sqrt(3))/sqrt(3)", &mut ctx).unwrap();
        let derivative =
            constant_scaled_arctan_surd_quotient_scaled_compact_derivative(&mut ctx, expr, "x")
                .unwrap();

        assert_eq!(rendered(&ctx, derivative), "7 / (2 * (x^2 + x + 1))");
    }

    #[test]
    fn arctan_surd_quotient_compact_derivative_normalizes_negative_polynomial() {
        let mut ctx = Context::new();
        let expr = parse("arctan(-(x^2+x+1)/sqrt(5))", &mut ctx).unwrap();
        let derivative = arctan_surd_quotient_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "-(2 * x + 1) * sqrt(5) / ((x^2 + x + 1)^2 + 5)"
        );
    }

    #[test]
    fn arctan_surd_quotient_scaled_compact_derivative_normalizes_negative_polynomial() {
        let mut ctx = Context::new();
        let expr = parse("arctan((1-x-x^2)/sqrt(5))/sqrt(5)", &mut ctx).unwrap();
        let derivative =
            arctan_surd_quotient_scaled_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "-(2 * x + 1) / ((x^2 + x - 1)^2 + 5)"
        );
    }

    #[test]
    fn arctan_self_normalized_surd_quotient_accepts_inverse_sqrt_product_arg() {
        let mut ctx = Context::new();
        let expr = parse("arctan((4*x^2+4*x+2)^(-1/2)*(2*x+1))", &mut ctx).unwrap();
        let derivative =
            arctan_self_normalized_surd_quotient_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "2 / (sqrt(4 * x^2 + 4 * x + 2) * (2 * (2 * x + 1)^2 + 1))"
        );
    }

    #[test]
    fn arctan_self_normalized_surd_reciprocal_accepts_inverse_denominator_arg() {
        let mut ctx = Context::new();
        let expr = parse("arctan((x^2+1)^(1/2)*x^(-1))", &mut ctx).unwrap();
        let (derivative, required_condition) =
            arctan_self_normalized_surd_reciprocal_compact_derivative(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, derivative),
            "-1 / (sqrt(x^2 + 1) * (2 * x^2 + 1))"
        );
        assert!(matches!(
            required_condition,
            crate::ImplicitCondition::NonZero(required) if rendered(&ctx, required) == "x"
        ));
    }
}
