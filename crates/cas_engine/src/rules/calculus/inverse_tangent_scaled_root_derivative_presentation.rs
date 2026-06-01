use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

use super::domain_checks::{
    positive_polynomial_radicand_and_nonzero_required_conditions,
    positive_polynomial_radicand_required_conditions,
};
use super::gap_presentation::squared_expr_for_compact_gap_presentation;
use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::presentation_utils::scaled_sqrt_argument_for_calculus_presentation;
use super::result_presentation::{
    cancel_denominator_content_with_numerator_for_calculus_presentation,
    scale_compact_derivative_by_rational,
};
use super::scalar_presentation::{
    add_one_for_calculus_presentation, nonzero_rational_parts,
    rational_const_for_calculus_presentation, rational_scaled_single_factor,
    scale_compact_fraction_numerator_by_rational_for_calculus_presentation,
    scale_expr_for_calculus_presentation, signed_numerator_for_calculus_presentation,
    subtract_expr_for_calculus_presentation,
};
use super::scaled_sqrt_args::{
    inverse_tangent_sqrt_over_symbolic_constant_arg_for_calculus_presentation,
    scaled_sqrt_over_symbolic_constant_arg_for_calculus_presentation,
    scaled_sqrt_polynomial_arg_for_calculus_presentation,
};

pub(super) fn inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    derivative_sign: BigRational,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    let builtin = ctx.builtin_of(fn_id)?;
    if args.len() != 1 {
        return None;
    }
    let expected_sign = match builtin {
        BuiltinFn::Atan | BuiltinFn::Arctan => BigRational::one(),
        BuiltinFn::Acot | BuiltinFn::Arccot => -BigRational::one(),
        _ => return None,
    };
    if derivative_sign != expected_sign {
        return None;
    }

    let (radicand, radicand_poly, sqrt_scale) = if let Some(parts) =
        scaled_sqrt_polynomial_arg_for_calculus_presentation(ctx, args[0], var_name)
    {
        parts
    } else {
        let (radicand, sqrt_scale) = scaled_sqrt_argument_for_calculus_presentation(ctx, args[0])?;
        let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
        (radicand, radicand_poly, sqrt_scale)
    };
    if sqrt_scale.is_zero() || sqrt_scale.is_one() {
        return None;
    }

    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_sign
        * sqrt_scale.clone()
        * derivative_content
        * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let scaled_radicand =
        scale_expr_for_calculus_presentation(ctx, sqrt_scale.clone() * sqrt_scale, radicand);
    let radicand_gap = add_one_for_calculus_presentation(ctx, scaled_radicand);
    let (numerator_coeff, denominator_coeff, radicand_gap) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            radicand_gap,
        );
    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&(numerator_coeff / denominator_coeff))?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, radicand_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    let derivative_sign = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Atan | BuiltinFn::Arctan) => BigRational::one(),
        Some(BuiltinFn::Acot | BuiltinFn::Arccot) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let (radicand, scale_denominator, argument_sign, sqrt_scale) =
        inverse_tangent_sqrt_over_symbolic_constant_arg_for_calculus_presentation(
            ctx, args[0], var_name,
        )?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_sign
        * argument_sign
        * sqrt_scale.clone()
        * derivative_content
        * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let derivative_core_is_one = cas_ast::views::as_rational_const(ctx, derivative_core, 8)
        .is_some_and(|value| value.is_one());
    let numerator_core = if derivative_core_is_one {
        scale_denominator
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &[scale_denominator, derivative_core])
    };

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let scale_square = squared_expr_for_compact_gap_presentation(ctx, scale_denominator);
    let scaled_radicand =
        scale_expr_for_calculus_presentation(ctx, sqrt_scale.clone() * sqrt_scale, radicand);
    let denominator_gap = ctx.add(Expr::Add(scale_square, scaled_radicand));
    let (numerator_coeff, denominator_coeff, denominator_gap) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            denominator_gap,
        );
    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&(numerator_coeff / denominator_coeff))?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, denominator_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Function(_, args) = ctx.get(target).clone() else {
        return None;
    };
    let [arg] = args.as_slice() else {
        return None;
    };

    let (radicand, scale_denominator, _, _) =
        inverse_tangent_sqrt_over_symbolic_constant_arg_for_calculus_presentation(
            ctx, *arg, var_name,
        )?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let result =
        inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(ctx, target, var_name)?;

    let required_conditions = positive_polynomial_radicand_and_nonzero_required_conditions(
        radicand,
        &radicand_poly,
        scale_denominator,
    );

    Some((cas_ast::hold::wrap_hold(ctx, result), required_conditions))
}

pub(super) fn atanh_sqrt_over_symbolic_constant_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target).clone() else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(fn_id, BuiltinFn::Atanh) {
        return None;
    }

    let (radicand, scale_denominator, argument_sign, sqrt_scale) =
        scaled_sqrt_over_symbolic_constant_arg_for_calculus_presentation(ctx, args[0], var_name)?;
    if sqrt_scale.abs().is_one() {
        return None;
    }

    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }

    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = argument_sign
        * sqrt_scale.clone()
        * derivative_content
        * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let derivative_core_is_one = cas_ast::views::as_rational_const(ctx, derivative_core, 8)
        .is_some_and(|value| value.is_one());
    let numerator_core = if derivative_core_is_one {
        scale_denominator
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &[scale_denominator, derivative_core])
    };

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let scale_square = squared_expr_for_compact_gap_presentation(ctx, scale_denominator);
    let scaled_radicand =
        scale_expr_for_calculus_presentation(ctx, sqrt_scale.clone() * sqrt_scale, radicand);
    let denominator_gap =
        subtract_expr_for_calculus_presentation(ctx, scale_square, scaled_radicand);
    let (numerator_coeff, denominator_coeff, denominator_gap) =
        cancel_denominator_content_with_numerator_for_calculus_presentation(
            ctx,
            numerator_coeff,
            denominator_coeff,
            denominator_gap,
        );
    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&(numerator_coeff / denominator_coeff))?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, denominator_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn atanh_sqrt_over_symbolic_constant_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Function(_, args) = ctx.get(target).clone() else {
        return None;
    };
    let [arg] = args.as_slice() else {
        return None;
    };

    let (radicand, scale_denominator, _, sqrt_scale) =
        scaled_sqrt_over_symbolic_constant_arg_for_calculus_presentation(ctx, *arg, var_name)?;
    if sqrt_scale.abs().is_one() {
        return None;
    }

    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let result = atanh_sqrt_over_symbolic_constant_derivative_presentation(ctx, target, var_name)?;
    let required_conditions = positive_polynomial_radicand_and_nonzero_required_conditions(
        radicand,
        &radicand_poly,
        scale_denominator,
    );

    Some((cas_ast::hold::wrap_hold(ctx, result), required_conditions))
}

pub(super) fn constant_scaled_inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let (derivative, required_conditions) =
        inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut(ctx, inner, var_name)?;
    let derivative = scale_compact_fraction_numerator_by_rational_for_calculus_presentation(
        ctx, derivative, scale,
    );

    Some((ctx.add(Expr::Hold(derivative)), required_conditions))
}

pub(super) fn constant_scaled_inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let derivative = inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
        ctx,
        inner,
        var_name,
        BigRational::one(),
    )
    .or_else(|| {
        inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
            ctx,
            inner,
            var_name,
            -BigRational::one(),
        )
    })?;

    Some(scale_compact_derivative_by_rational(ctx, derivative, scale))
}

pub(super) fn inverse_tangent_scaled_sqrt_polynomial_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let arg = match ctx.get(target).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(fn_id),
                    Some(BuiltinFn::Atan | BuiltinFn::Arctan | BuiltinFn::Acot | BuiltinFn::Arccot)
                ) =>
        {
            args[0]
        }
        _ => return None,
    };
    let (radicand, radicand_poly, sqrt_scale) = if let Some(parts) =
        scaled_sqrt_polynomial_arg_for_calculus_presentation(ctx, arg, var_name)
    {
        parts
    } else {
        let (radicand, sqrt_scale) = scaled_sqrt_argument_for_calculus_presentation(ctx, arg)?;
        let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
        (radicand, radicand_poly, sqrt_scale)
    };
    if sqrt_scale.is_zero() || sqrt_scale.abs() == BigRational::one() {
        return None;
    }

    let result = inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
        ctx,
        target,
        var_name,
        BigRational::one(),
    )
    .or_else(|| {
        inverse_tangent_scaled_sqrt_polynomial_derivative_presentation(
            ctx,
            target,
            var_name,
            -BigRational::one(),
        )
    })?;
    let required_conditions =
        positive_polynomial_radicand_required_conditions(radicand, &radicand_poly);
    Some((cas_ast::hold::wrap_hold(ctx, result), required_conditions))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::super::presentation_utils::unwrap_internal_hold_for_calculus;
    use super::{
        atanh_sqrt_over_symbolic_constant_derivative_shortcut,
        constant_scaled_inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut,
        inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation,
        inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut,
    };

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn inverse_tangent_sqrt_over_symbolic_constant_derivative_keeps_parameter_scale_compact() {
        let mut ctx = Context::new();
        let expr = parse("arctan(sqrt(x)/a)", &mut ctx).unwrap();
        let derivative = inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(rendered(&ctx, derivative), "a / (2 * sqrt(x) * (a^2 + x))");
    }

    #[test]
    fn inverse_tangent_sqrt_over_symbolic_constant_derivative_compacts_arccot_dual() {
        let mut ctx = Context::new();
        let expr = parse("arccot(sqrt(x)/a)", &mut ctx).unwrap();
        let derivative = inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(rendered(&ctx, derivative), "-a / (2 * sqrt(x) * (a^2 + x))");
    }

    #[test]
    fn inverse_tangent_sqrt_over_symbolic_constant_derivative_extracts_numerator_scale() {
        let mut ctx = Context::new();
        let expr = parse("arctan(2*sqrt(x)/a)", &mut ctx).unwrap();
        let derivative = inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(rendered(&ctx, derivative), "a / (sqrt(x) * (a^2 + 4 * x))");
    }

    #[test]
    fn inverse_tangent_sqrt_over_symbolic_constant_derivative_extracts_denominator_scale() {
        let mut ctx = Context::new();
        let expr = parse("arctan(sqrt(x)/(2*a))", &mut ctx).unwrap();
        let derivative = inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(rendered(&ctx, derivative), "a / (sqrt(x) * (4 * a^2 + x))");
    }

    #[test]
    fn inverse_tangent_sqrt_over_symbolic_constant_shortcut_keeps_affine_domain_minimal() {
        let mut ctx = Context::new();
        let expr = parse("arctan(sqrt(2*x+2)/(2*a))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut(&mut ctx, expr, "x")
                .unwrap();
        let derivative = unwrap_internal_hold_for_calculus(&mut ctx, derivative);

        assert_eq!(
            rendered(&ctx, derivative),
            "a / (sqrt(2 * x + 2) * (2 * a^2 + x + 1))"
        );
        assert_eq!(required_conditions.len(), 2);
        assert!(matches!(
            required_conditions[0],
            crate::ImplicitCondition::Positive(required) if rendered(&ctx, required) == "2 * x + 2"
        ));
        assert!(matches!(
            required_conditions[1],
            crate::ImplicitCondition::NonZero(required) if rendered(&ctx, required) == "a"
        ));
    }

    #[test]
    fn constant_scaled_inverse_tangent_sqrt_over_symbolic_constant_shortcut_scales_affine_numerator(
    ) {
        let mut ctx = Context::new();
        let expr = parse("3*arctan(sqrt(2*x+2)/(2*a))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            constant_scaled_inverse_tangent_sqrt_over_symbolic_constant_derivative_shortcut(
                &mut ctx, expr, "x",
            )
            .unwrap();
        let derivative = unwrap_internal_hold_for_calculus(&mut ctx, derivative);

        assert_eq!(
            rendered(&ctx, derivative),
            "3 * a / (sqrt(2 * x + 2) * (2 * a^2 + x + 1))"
        );
        assert_eq!(required_conditions.len(), 2);
        assert!(matches!(
            required_conditions[0],
            crate::ImplicitCondition::Positive(required) if rendered(&ctx, required) == "2 * x + 2"
        ));
        assert!(matches!(
            required_conditions[1],
            crate::ImplicitCondition::NonZero(required) if rendered(&ctx, required) == "a"
        ));
    }

    #[test]
    fn inverse_tangent_sqrt_over_symbolic_constant_derivative_extracts_fractional_denominator_scale(
    ) {
        let mut ctx = Context::new();
        let expr = parse("arctan(sqrt(x)/(a/2))", &mut ctx).unwrap();
        let derivative = inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(rendered(&ctx, derivative), "a / (sqrt(x) * (a^2 + 4 * x))");
    }

    #[test]
    fn inverse_tangent_sqrt_over_symbolic_constant_derivative_extracts_square_content() {
        let mut ctx = Context::new();
        let expr = parse("arctan(sqrt(4*x)/a)", &mut ctx).unwrap();
        let derivative = inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(rendered(&ctx, derivative), "a / (sqrt(x) * (a^2 + 4 * x))");
    }

    #[test]
    fn inverse_tangent_sqrt_over_symbolic_constant_derivative_accepts_external_scale() {
        let mut ctx = Context::new();
        let expr = parse("arctan(2*(sqrt(x)/a))", &mut ctx).unwrap();
        let derivative = inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(
            &mut ctx, expr, "x",
        )
        .unwrap();

        assert_eq!(rendered(&ctx, derivative), "a / (sqrt(x) * (a^2 + 4 * x))");
    }

    #[test]
    fn atanh_sqrt_over_symbolic_constant_derivative_compacts_exact_square_scale() {
        let mut ctx = Context::new();
        let expr = parse("atanh(2*(sqrt(x+1)/a))", &mut ctx).unwrap();
        let (derivative, required_conditions) =
            atanh_sqrt_over_symbolic_constant_derivative_shortcut(&mut ctx, expr, "x").unwrap();
        let derivative = unwrap_internal_hold_for_calculus(&mut ctx, derivative);

        assert_eq!(
            rendered(&ctx, derivative),
            "a / (sqrt(x + 1) * (a^2 - 4 * x - 4))"
        );
        assert_eq!(required_conditions.len(), 2);
        assert!(matches!(
            required_conditions[0],
            crate::ImplicitCondition::Positive(required) if rendered(&ctx, required) == "x + 1"
        ));
        assert!(matches!(
            required_conditions[1],
            crate::ImplicitCondition::NonZero(required) if rendered(&ctx, required) == "a"
        ));
    }

    #[test]
    fn inverse_tangent_sqrt_over_symbolic_constant_derivative_rejects_x_dependent_denominator() {
        let mut ctx = Context::new();
        let expr = parse("arctan(sqrt(x)/x)", &mut ctx).unwrap();

        assert!(
            inverse_tangent_sqrt_over_symbolic_constant_derivative_presentation(
                &mut ctx, expr, "x"
            )
            .is_none()
        );
    }
}
