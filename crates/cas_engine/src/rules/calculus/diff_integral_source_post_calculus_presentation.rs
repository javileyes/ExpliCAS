use super::sqrt_hyperbolic_log_integrand_presentation::compact_direct_sqrt_hyperbolic_log_derivative_integrand;
use super::sqrt_trig_log_integrand_presentation::compact_sqrt_trig_log_derivative_integrand;
use crate::symbolic_calculus_call_support::try_extract_integrate_call;
use cas_ast::{Context, ExprId};

pub(super) fn try_diff_integral_source_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    result: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    try_extract_integrate_call(ctx, target)?;
    compact_sqrt_trig_log_derivative_integrand(ctx, result, var_name)
        .or_else(|| compact_direct_sqrt_hyperbolic_log_derivative_integrand(ctx, result, var_name))
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::try_diff_integral_source_post_calculus_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn diff_integral_source_post_calculus_presentation_compacts_nested_integral_source() {
        let mut ctx = Context::new();
        let target = parse("integrate(tan(sqrt(3*x+1))*3/(2*sqrt(3*x+1)), x)", &mut ctx).unwrap();
        let result = parse(
            "((3*x+1)^(1/2) * sin((3*x+1)^(1/2)) * 3)/(cos((3*x+1)^(1/2)) * (6*x+2))",
            &mut ctx,
        )
        .unwrap();
        let compact =
            try_diff_integral_source_post_calculus_presentation(&mut ctx, target, result, "x")
                .unwrap();

        assert_eq!(
            rendered(&ctx, compact),
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))"
        );
    }
}
