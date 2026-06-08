//! Direct symbolic-calculus routes that run before general simplification.
//!
//! These routes are intentionally ordered and narrow. They protect verified
//! calculus frontiers from falling into broader post-calculus presentation or
//! generic simplification.

use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use cas_math::expr_predicates::contains_named_var;
use cas_math::polynomial::Polynomial;
use num_traits::Zero;

const DIRECT_CALCULUS_DOMAIN_PROOF_DEPTH: usize = 12;

pub(crate) type PreSimplifyCalculusResolution = (
    ExprId,
    Vec<crate::ImplicitCondition>,
    &'static str,
    &'static str,
);

pub(crate) fn try_resolve_direct_symbolic_calculus_before_general_simplify(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PreSimplifyCalculusResolution> {
    try_resolve_direct_diff_integral_positive_constant_radius_quadratic_linear_numerator_before_general_simplify(ctx, expr)
        .map(|(result, required_conditions)| {
            (
                result,
                required_conditions,
                "Symbolic Differentiation",
                "Symbolic Differentiation",
            )
        })
        .or_else(|| {
            try_resolve_direct_diff_integral_positive_quadratic_arctan_before_general_simplify(
                ctx, expr,
            )
            .map(|(result, required_conditions)| {
                (
                    result,
                    required_conditions,
                    "Symbolic Differentiation",
                    "Symbolic Differentiation",
                )
            })
        })
        .or_else(|| {
            try_resolve_direct_diff_integral_reciprocal_trig_derivative_product_before_general_simplify(
                ctx, expr,
            )
            .map(|(result, required_conditions)| {
                (
                    result,
                    required_conditions,
                    "Symbolic Differentiation",
                    "Symbolic Differentiation",
                )
            })
        })
        .or_else(|| {
            try_resolve_direct_diff_integral_hyperbolic_reciprocal_fourth_before_general_simplify(
                ctx, expr,
            )
            .map(|(result, required_conditions)| {
                (
                    result,
                    required_conditions,
                    "Symbolic Differentiation",
                    "Symbolic Differentiation",
                )
            })
        })
        .or_else(|| {
            try_resolve_direct_diff_integral_hyperbolic_sinh_square_before_general_simplify(
                ctx, expr,
            )
            .map(|(result, required_conditions)| {
                (
                    result,
                    required_conditions,
                    "Symbolic Differentiation",
                    "Symbolic Differentiation",
                )
            })
        })
        .or_else(|| {
            try_resolve_direct_diff_hyperbolic_coth_before_general_simplify(ctx, expr).map(
                |(result, required_conditions)| {
                    (
                        result,
                        required_conditions,
                        "Symbolic Differentiation",
                        "Symbolic Differentiation",
                    )
                },
            )
        })
        .or_else(|| {
            try_resolve_direct_diff_variable_power_quadratic_base_before_general_simplify(ctx, expr)
                .map(|(result, required_conditions)| {
                    (
                        result,
                        required_conditions,
                        "Calcular la derivada",
                        "Calcular la derivada",
                    )
                })
        })
        .or_else(|| {
            try_resolve_direct_post_calculus_before_general_simplify(ctx, expr).map(
                |(result, required_conditions)| {
                    (
                        result,
                        required_conditions,
                        "Calcular la derivada",
                        "Calcular la derivada",
                    )
                },
            )
        })
}

fn try_resolve_direct_diff_variable_power_quadratic_base_before_general_simplify(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, expr)?;
    let target = cas_ast::hold::strip_all_holds(ctx, call.target);
    let Expr::Pow(base, exponent) = ctx.get(target).clone() else {
        return None;
    };
    if !is_var_local(ctx, exponent, &call.var_name)
        || !contains_named_var(ctx, base, &call.var_name)
    {
        return None;
    }

    let base_poly = Polynomial::from_expr(ctx, base, &call.var_name).ok()?;
    if base_poly.degree() != 2 {
        return None;
    }

    if cas_math::calculus_domain_support::positive_condition_is_impossible_over_reals(
        ctx,
        base,
        DIRECT_CALCULUS_DOMAIN_PROOF_DEPTH,
    ) {
        let undefined = ctx.add(Expr::Constant(Constant::Undefined));
        return Some((undefined, Vec::new()));
    }

    let derivative_poly = base_poly.derivative();
    if derivative_poly.is_zero() {
        return None;
    }

    let x_poly = Polynomial::new(
        vec![
            num_rational::BigRational::zero(),
            num_rational::BigRational::from_integer(1.into()),
        ],
        call.var_name.clone(),
    );
    let x_times_base_derivative = derivative_poly.mul(&x_poly).to_expr(ctx);
    let ln_base = ctx.call_builtin(BuiltinFn::Ln, vec![base]);
    let base_times_ln = ctx.add(Expr::Mul(base, ln_base));
    let numerator = ctx.add(Expr::Add(base_times_ln, x_times_base_derivative));
    let power_times_numerator = ctx.add(Expr::Mul(target, numerator));
    let result = ctx.add(Expr::Div(power_times_numerator, base));

    let required_conditions =
        if cas_math::calculus_domain_support::positive_condition_is_proven_over_reals(
            ctx,
            base,
            DIRECT_CALCULUS_DOMAIN_PROOF_DEPTH,
        ) {
            Vec::new()
        } else {
            vec![crate::ImplicitCondition::Positive(base)]
        };
    Some((result, required_conditions))
}

fn is_var_local(ctx: &mut Context, expr: ExprId, var_name: &str) -> bool {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var_name)
}

fn try_resolve_direct_diff_integral_positive_constant_radius_quadratic_linear_numerator_before_general_simplify(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if integrate_call.var_name != diff_call.var_name {
        return None;
    }

    let integrand = cas_ast::hold::strip_all_holds(ctx, integrate_call.target);
    if !cas_math::symbolic_integration_support::integrate_symbolic_is_positive_constant_radius_quadratic_linear_numerator_target(
        ctx,
        integrand,
        &integrate_call.var_name,
    ) {
        return None;
    }

    let raw_required_conditions =
        super::integration_conditions::IntegrationRequiredConditions::from_target(
            ctx,
            integrand,
            &integrate_call.var_name,
        )
        .into_implicit_conditions()
        .collect::<Vec<_>>();
    if raw_required_conditions
        .iter()
        .any(|condition| !condition_is_proven_for_direct_integral_source(ctx, condition))
    {
        return None;
    }

    Some((integrand, Vec::new()))
}

fn try_resolve_direct_diff_integral_positive_quadratic_arctan_before_general_simplify(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if integrate_call.var_name != diff_call.var_name {
        return None;
    }

    let integrand = cas_ast::hold::strip_all_holds(ctx, integrate_call.target);
    if !cas_math::symbolic_integration_support::integrate_symbolic_is_positive_rational_quadratic_arctan_target(
        ctx,
        integrand,
        &integrate_call.var_name,
    ) {
        return None;
    }

    let raw_required_conditions =
        super::integration_conditions::IntegrationRequiredConditions::from_target(
            ctx,
            integrand,
            &integrate_call.var_name,
        )
        .into_implicit_conditions()
        .collect::<Vec<_>>();
    if raw_required_conditions
        .iter()
        .any(|condition| !condition_is_proven_for_direct_integral_source(ctx, condition))
    {
        return None;
    }

    Some((integrand, Vec::new()))
}

fn try_resolve_direct_diff_integral_reciprocal_trig_derivative_product_before_general_simplify(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if integrate_call.var_name != diff_call.var_name {
        return None;
    }

    let integrand = cas_ast::hold::strip_all_holds(ctx, integrate_call.target);
    let (source, required_nonzero) =
        super::reciprocal_trig_derivative_product_source::verified_reciprocal_trig_derivative_product_source_with_domain(
            ctx,
            integrand,
            &integrate_call.var_name,
        )?;

    Some((
        source,
        vec![crate::ImplicitCondition::NonZero(required_nonzero)],
    ))
}

fn try_resolve_direct_diff_integral_hyperbolic_reciprocal_fourth_before_general_simplify(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if integrate_call.var_name != diff_call.var_name {
        return None;
    }

    let integrand = cas_ast::hold::strip_all_holds(ctx, integrate_call.target);
    let Expr::Div(_, den) = ctx.get(integrand).clone() else {
        return None;
    };
    let (den_builtin, arg) = fourth_hyperbolic_builtin_arg(ctx, den)?;
    if !cas_math::symbolic_integration_support::integrate_symbolic_is_hyperbolic_quotient_substitution_target(
        ctx,
        integrand,
        &integrate_call.var_name,
    ) {
        return None;
    }

    let mut required_conditions =
        super::integration_conditions::IntegrationRequiredConditions::from_target(
            ctx,
            integrand,
            &integrate_call.var_name,
        )
        .into_implicit_conditions()
        .collect::<Vec<_>>();
    if den_builtin == BuiltinFn::Sinh {
        let sinh_arg = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
        required_conditions.push(crate::ImplicitCondition::NonZero(sinh_arg));
    }
    Some((integrand, required_conditions))
}

fn condition_is_proven_for_direct_integral_source(
    ctx: &mut Context,
    condition: &crate::ImplicitCondition,
) -> bool {
    match condition {
        crate::ImplicitCondition::Positive(expr) | crate::ImplicitCondition::NonZero(expr) => {
            cas_math::calculus_domain_support::positive_condition_is_proven_over_reals(
                ctx,
                *expr,
                DIRECT_CALCULUS_DOMAIN_PROOF_DEPTH,
            ) || expr_is_known_nonzero_for_direct_integral_source(ctx, *expr)
        }
        _ => false,
    }
}

fn expr_is_known_nonzero_for_direct_integral_source(ctx: &mut Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    if cas_ast::views::as_rational_const(ctx, expr, 8).is_some_and(|value| !value.is_zero()) {
        return true;
    }

    match ctx.get(expr).clone() {
        Expr::Constant(Constant::Pi | Constant::E | Constant::Phi) => true,
        Expr::Neg(inner) => expr_is_known_nonzero_for_direct_integral_source(ctx, inner),
        Expr::Mul(left, right) | Expr::Div(left, right) => {
            expr_is_known_nonzero_for_direct_integral_source(ctx, left)
                && expr_is_known_nonzero_for_direct_integral_source(ctx, right)
        }
        Expr::Pow(base, exp) => {
            cas_ast::views::as_rational_const(ctx, exp, 8).is_some_and(|value| !value.is_zero())
                && expr_is_known_nonzero_for_direct_integral_source(ctx, base)
        }
        Expr::Function(fn_id, args)
            if ctx.builtin_of(fn_id) == Some(BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            cas_math::calculus_domain_support::positive_condition_is_proven_over_reals(
                ctx,
                args[0],
                DIRECT_CALCULUS_DOMAIN_PROOF_DEPTH,
            )
        }
        _ => false,
    }
}

fn try_resolve_direct_post_calculus_before_general_simplify(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, expr)?;
    if super::diff_target_known_undefined_over_reals(ctx, call.target, &call.var_name) {
        let undefined = ctx.add(Expr::Constant(Constant::Undefined));
        return Some((undefined, Vec::new()));
    }

    if let Some(result) =
        cas_math::symbolic_differentiation_support::positive_constant_radius_quadratic_log_arctan_primitive_derivative(
            ctx,
            call.target,
            &call.var_name,
        )
    {
        return Some((result, Vec::new()));
    }
    if let Some(result) =
        cas_math::symbolic_differentiation_support::positive_constant_radius_quadratic_arctan_primitive_derivative(
            ctx,
            call.target,
            &call.var_name,
        )
    {
        return Some((result, Vec::new()));
    }

    if let Some((result, required_conditions)) =
        super::constant_scaled_reciprocal_trig_affine_derivative_presentation_with_domain(
            ctx,
            call.target,
            &call.var_name,
        )
    {
        return Some((result, required_conditions));
    }

    if let Some((result, required_conditions)) =
        super::constant_scaled_hyperbolic_reciprocal_derivative_quotient_presentation_with_domain(
            ctx,
            call.target,
            &call.var_name,
        )
    {
        return Some((result, required_conditions));
    }

    if let Some((result, required_conditions)) =
        super::reciprocal_trig_shifted_sqrt_derivative_presentation(
            ctx,
            call.target,
            &call.var_name,
        )
    {
        return Some((result, required_conditions));
    }

    if let Some(result) = super::affine_hyperbolic_odd_primitive_derivative_presentation(
        ctx,
        call.target,
        &call.var_name,
    ) {
        return Some((result, Vec::new()));
    }

    if let Some((result, radicand, mut required_conditions)) =
        super::sqrt_additive_tan_polynomial_derivative_inline_presentation(
            ctx,
            call.target,
            &call.var_name,
        )
    {
        required_conditions.insert(0, crate::ImplicitCondition::Positive(radicand));
        return Some((result, required_conditions));
    }

    let (mut result, radicand, mut required_conditions) =
        super::sqrt_additive_tan_polynomial_derivative_presentation(
            ctx,
            call.target,
            &call.var_name,
        )?;
    if let Some((inline_result, inline_radicand, inline_required_conditions)) =
        super::sqrt_additive_tan_polynomial_derivative_inline_presentation(
            ctx,
            call.target,
            &call.var_name,
        )
    {
        if inline_radicand == radicand {
            result = inline_result;
            required_conditions = inline_required_conditions;
        }
    }
    required_conditions.insert(0, crate::ImplicitCondition::Positive(radicand));
    Some((result, required_conditions))
}

fn try_resolve_direct_diff_hyperbolic_coth_before_general_simplify(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, expr)?;
    let result =
        cas_math::symbolic_differentiation_support::differentiate_symbolic_linear_times_hyperbolic_coth_linear_div_derivative(
            ctx,
            call.target,
            &call.var_name,
        )?;
    Some((result, Vec::new()))
}

fn try_resolve_direct_diff_integral_hyperbolic_sinh_square_before_general_simplify(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if integrate_call.var_name != diff_call.var_name {
        return None;
    }

    let integrand = cas_ast::hold::strip_all_holds(ctx, integrate_call.target);
    let Expr::Div(num, den) = ctx.get(integrand).clone() else {
        return None;
    };
    let one = num_rational::BigRational::from_integer(1.into());
    if cas_ast::views::as_rational_const(ctx, num, 8).is_none_or(|value| value != one) {
        return None;
    }

    let arg = squared_builtin_arg(ctx, den, BuiltinFn::Sinh)?;
    let arg_poly =
        cas_math::polynomial::Polynomial::from_expr(ctx, arg, &diff_call.var_name).ok()?;
    if arg_poly.degree() != 1 {
        return None;
    }
    let slope = arg_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(num_rational::BigRational::zero);
    if slope.is_zero() {
        return None;
    }

    let sinh_arg = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
    Some((integrand, vec![crate::ImplicitCondition::NonZero(sinh_arg)]))
}

fn fourth_hyperbolic_builtin_arg(ctx: &mut Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    let expected = num_rational::BigRational::from_integer(4.into());
    if cas_ast::views::as_rational_const(ctx, exp, 8).is_none_or(|value| value != expected) {
        return None;
    }

    let base = cas_ast::hold::strip_all_holds(ctx, base);
    let Expr::Function(fn_id, args) = ctx.get(base) else {
        return None;
    };
    let builtin = ctx.builtin_of(*fn_id)?;
    (args.len() == 1 && matches!(builtin, BuiltinFn::Cosh | BuiltinFn::Sinh))
        .then_some((builtin, args[0]))
}

fn squared_builtin_arg(ctx: &mut Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    let expected = num_rational::BigRational::from_integer(2.into());
    if cas_ast::views::as_rational_const(ctx, exp, 8).is_none_or(|value| value != expected) {
        return None;
    }

    let base = cas_ast::hold::strip_all_holds(ctx, base);
    let Expr::Function(fn_id, args) = ctx.get(base) else {
        return None;
    };
    if args.len() == 1 && ctx.is_builtin(*fn_id, builtin) {
        Some(args[0])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn direct_diff_integral_positive_quadratic_arctan_returns_source_before_general_simplify() {
        let mut ctx = Context::new();
        let expr = parse("diff(integrate(1/((2*x+3)^2+phi), x), x)", &mut ctx).unwrap();

        let (result, required_conditions) =
            try_resolve_direct_diff_integral_positive_quadratic_arctan_before_general_simplify(
                &mut ctx, expr,
            )
            .expect("direct positive quadratic arctan route");

        assert_eq!(rendered(&ctx, result), "1 / ((2 * x + 3)^2 + phi)");
        assert!(required_conditions.is_empty());
    }

    #[test]
    fn direct_diff_integral_positive_constant_radius_linear_numerator_returns_source() {
        let mut ctx = Context::new();
        let expr = parse(
            "diff(integrate((2*x+6)/(4*x^2+12*x+9+phi), x), x)",
            &mut ctx,
        )
        .unwrap();

        let (result, required_conditions) =
            try_resolve_direct_diff_integral_positive_constant_radius_quadratic_linear_numerator_before_general_simplify(
                &mut ctx, expr,
            )
            .expect("direct positive constant radius linear numerator route");

        assert_eq!(
            rendered(&ctx, result),
            "(2 * x + 6) / (4 * x^2 + 12 * x + 9 + phi)"
        );
        assert!(required_conditions.is_empty());
    }

    #[test]
    fn direct_diff_integral_positive_quadratic_arctan_defers_symbolic_slope_condition() {
        let mut ctx = Context::new();
        let expr = parse("diff(integrate(1/((a*x+b)^2+2), x), x)", &mut ctx).unwrap();

        assert!(
            try_resolve_direct_diff_integral_positive_quadratic_arctan_before_general_simplify(
                &mut ctx, expr,
            )
            .is_none()
        );
    }

    #[test]
    fn direct_diff_integral_reciprocal_trig_polynomial_product_returns_source_and_pole() {
        let mut ctx = Context::new();
        let expr = parse("diff(integrate(2*x*csc(x^2+b)*cot(x^2+b), x), x)", &mut ctx).unwrap();

        let (result, required_conditions) =
            try_resolve_direct_diff_integral_reciprocal_trig_derivative_product_before_general_simplify(
                &mut ctx, expr,
            )
            .expect("direct reciprocal trig derivative-product route");

        assert_eq!(
            rendered(&ctx, result),
            "2 * x * csc(x^2 + b) * cot(x^2 + b)"
        );
        let required_displays: Vec<_> = required_conditions
            .iter()
            .map(|condition| condition.display(&ctx).to_string())
            .collect();
        assert_eq!(required_displays, vec!["sin(x^2 + b) ≠ 0"]);
    }

    #[test]
    fn direct_diff_integral_reciprocal_trig_polynomial_product_still_requires_du() {
        let mut ctx = Context::new();
        let expr = parse("diff(integrate(csc(x^2+b)*cot(x^2+b), x), x)", &mut ctx).unwrap();

        assert!(
            try_resolve_direct_diff_integral_reciprocal_trig_derivative_product_before_general_simplify(
                &mut ctx, expr,
            )
            .is_none()
        );
    }

    #[test]
    fn direct_diff_integral_hyperbolic_cosh_fourth_polynomial_product_returns_source() {
        let mut ctx = Context::new();
        let expr = parse("diff(integrate(2*k*x/cosh(x^2+b)^4, x), x)", &mut ctx).unwrap();

        let (result, required_conditions) =
            try_resolve_direct_diff_integral_hyperbolic_reciprocal_fourth_before_general_simplify(
                &mut ctx, expr,
            )
            .expect("direct cosh fourth source route");

        assert_eq!(rendered(&ctx, result), "2 * k * x / cosh(x^2 + b)^4");
        assert!(required_conditions.is_empty());
    }

    #[test]
    fn direct_diff_integral_hyperbolic_cosh_fourth_still_requires_du() {
        let mut ctx = Context::new();
        let expr = parse("diff(integrate(1/cosh(x^2+b)^4, x), x)", &mut ctx).unwrap();

        assert!(
            try_resolve_direct_diff_integral_hyperbolic_reciprocal_fourth_before_general_simplify(
                &mut ctx, expr,
            )
            .is_none()
        );
    }

    #[test]
    fn direct_diff_integral_hyperbolic_sinh_fourth_polynomial_product_returns_source_and_pole() {
        let mut ctx = Context::new();
        let expr = parse("diff(integrate(2*k*x/sinh(x^2+b)^4, x), x)", &mut ctx).unwrap();

        let (result, required_conditions) =
            try_resolve_direct_diff_integral_hyperbolic_reciprocal_fourth_before_general_simplify(
                &mut ctx, expr,
            )
            .expect("direct sinh fourth source route");

        assert_eq!(rendered(&ctx, result), "2 * k * x / sinh(x^2 + b)^4");
        let required_displays: Vec<_> = required_conditions
            .iter()
            .map(|condition| condition.display(&ctx).to_string())
            .collect();
        assert_eq!(required_displays, vec!["sinh(x^2 + b) ≠ 0"]);
    }

    #[test]
    fn direct_diff_integral_hyperbolic_reciprocal_fourth_rejects_other_powers() {
        let mut ctx = Context::new();
        let expr = parse("diff(integrate(2*k*x/sinh(x^2+b)^2, x), x)", &mut ctx).unwrap();

        assert!(
            try_resolve_direct_diff_integral_hyperbolic_reciprocal_fourth_before_general_simplify(
                &mut ctx, expr,
            )
            .is_none()
        );
    }

    #[test]
    fn direct_diff_positive_constant_radius_quadratic_log_arctan_primitive_returns_source() {
        let mut ctx = Context::new();
        let expr = parse(
            "diff(1/4*ln(4*x^2+12*x+9+phi)+(3*atan(phi^(-1/2)*(2*x+3)))/(2*sqrt(phi)), x)",
            &mut ctx,
        )
        .unwrap();

        let (result, required_conditions) =
            try_resolve_direct_post_calculus_before_general_simplify(&mut ctx, expr)
                .expect("direct positive constant radius quadratic log-arctan primitive route");

        assert_eq!(
            rendered(&ctx, result),
            "(2 * x + 6) / (4 * x^2 + 12 * x + 9 + phi)"
        );
        assert!(required_conditions.is_empty());
    }

    #[test]
    fn direct_diff_positive_constant_radius_quadratic_arctan_primitive_returns_compact_source() {
        let mut ctx = Context::new();
        let expr = parse("diff(atan(phi^(-1/2)*(2*x+3))/(2*sqrt(phi)), x)", &mut ctx).unwrap();

        let (result, required_conditions) =
            try_resolve_direct_post_calculus_before_general_simplify(&mut ctx, expr)
                .expect("direct positive constant radius quadratic arctan primitive route");

        assert_eq!(rendered(&ctx, result), "1 / ((2 * x + 3)^2 + phi)");
        assert!(required_conditions.is_empty());
    }

    #[test]
    fn direct_diff_variable_power_quadratic_base_returns_compact_log_derivative() {
        let mut ctx = Context::new();
        let expr = parse("diff((x^2-1)^x, x)", &mut ctx).unwrap();

        let (result, required_conditions) =
            try_resolve_direct_diff_variable_power_quadratic_base_before_general_simplify(
                &mut ctx, expr,
            )
            .expect("direct quadratic variable-power route");

        assert_eq!(
            rendered(&ctx, result),
            "(x^2 - 1)^x * (ln(x^2 - 1) * (x^2 - 1) + 2 * x^2) / (x^2 - 1)"
        );
        assert_eq!(required_conditions.len(), 1);
        assert_eq!(
            required_conditions[0].display(&ctx).to_string(),
            "x < -1 or x > 1"
        );
    }

    #[test]
    fn direct_diff_variable_power_positive_quadratic_suppresses_domain_condition() {
        let mut ctx = Context::new();
        let expr = parse("diff((x^2+x+1)^x, x)", &mut ctx).unwrap();

        let (result, required_conditions) =
            try_resolve_direct_diff_variable_power_quadratic_base_before_general_simplify(
                &mut ctx, expr,
            )
            .expect("direct positive quadratic variable-power route");

        assert_eq!(
            rendered(&ctx, result),
            "(x^2 + x + 1)^x * (ln(x^2 + x + 1) * (x^2 + x + 1) + 2 * x^2 + x) / (x^2 + x + 1)"
        );
        assert!(required_conditions.is_empty());
    }
}
