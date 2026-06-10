//! General integration result pipeline.
//!
//! `IntegrateRule` owns route selection. This module owns the standard
//! integration path after the generic symbolic integrator has accepted the
//! source target: source/result preservation, result-implied conditions, and
//! final public presentation.

use super::integration::{
    integrate_rewrite_with_conditions, integrate_with_trace, IntegrationTraceKind,
};
use super::integration_arctan_by_parts_result_presentation::compact_polynomial_arctan_by_parts_result_for_integration_presentation;
use super::integration_conditions::IntegrationRequiredConditions;
use super::integration_final_result_presentation::apply_integration_final_presentation;
use super::integration_source_preservation::integration_source_preservation_gates;
use super::result_preservation::apply_integration_result_preservation;
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::NamedVarCall;
use cas_ast::{Context, ExprId};

pub(super) fn standard_integration_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
) -> Option<Rewrite> {
    let mut required_conditions =
        IntegrationRequiredConditions::from_target(ctx, call.target, &call.var_name);
    let result = integrate_with_result_preservation(
        ctx,
        call.target,
        &call.var_name,
        &mut required_conditions,
    )?;
    Some(integrate_rewrite_with_conditions(
        ctx,
        call,
        result,
        required_conditions.into_implicit_conditions(),
    ))
}

fn integrate_with_result_preservation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    required_conditions: &mut IntegrationRequiredConditions,
) -> Option<ExprId> {
    let source_preservation = integration_source_preservation_gates(ctx, target, var_name);
    let outcome = integrate_with_trace(ctx, target, var_name)?;
    let trace_kind = outcome.trace_kind;
    if trace_kind == IntegrationTraceKind::AlgorithmicBackendSummary {
        required_conditions.suppress_backend_positive_quadratic_source_condition(
            ctx,
            target,
            var_name,
            &outcome.required_conditions,
        );
    }
    required_conditions.include_backend_conditions(ctx, outcome.required_conditions);
    let mut result = outcome.result;
    let compact_polynomial_arctan_by_parts_result =
        compact_polynomial_arctan_by_parts_result_for_integration_presentation(
            ctx, var_name, result,
        );
    required_conditions.include_conditions_implied_by_result(ctx, result);
    result = apply_integration_result_preservation(
        ctx,
        result,
        var_name,
        required_conditions.has_positive(),
        &source_preservation,
        compact_polynomial_arctan_by_parts_result,
    );
    result = apply_integration_final_presentation(ctx, result, var_name);
    if trace_kind == IntegrationTraceKind::AlgorithmicBackendSummary {
        result = cas_ast::hold::wrap_hold(ctx, result);
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    fn rendered_required_conditions(ctx: &Context, rewrite: &Rewrite) -> Vec<String> {
        rewrite
            .required_conditions
            .iter()
            .map(|condition| condition.display(ctx))
            .collect()
    }

    #[test]
    fn standard_pipeline_returns_supported_integral_result() {
        let mut ctx = Context::new();
        let target = parse("1/(x+1)", &mut ctx).unwrap();
        let mut conditions = IntegrationRequiredConditions::from_target(&mut ctx, target, "x");

        let result =
            integrate_with_result_preservation(&mut ctx, target, "x", &mut conditions).unwrap();

        assert_eq!(rendered(&ctx, result), "ln(|x + 1|)");
    }

    #[test]
    fn standard_pipeline_rejects_unsupported_integrand() {
        let mut ctx = Context::new();
        let target = parse("exp(x^2)", &mut ctx).unwrap();
        let mut conditions = IntegrationRequiredConditions::from_target(&mut ctx, target, "x");

        assert!(
            integrate_with_result_preservation(&mut ctx, target, "x", &mut conditions).is_none()
        );
    }

    #[test]
    fn standard_rewrite_preserves_result_and_required_conditions() {
        let mut ctx = Context::new();
        let target = parse("1/(x+1)", &mut ctx).unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };

        let rewrite = standard_integration_rewrite(&mut ctx, &call).unwrap();

        assert_eq!(rendered(&ctx, rewrite.new_expr), "ln(|x + 1|)");
        assert_eq!(rendered_required_conditions(&ctx, &rewrite), vec!["x ≠ -1"]);
    }

    #[test]
    fn standard_rewrite_suppresses_backend_positive_quadratic_source_condition() {
        let mut ctx = Context::new();
        let target = parse("(m*(s*x+b)+c)/((s*x+b)^2+a)", &mut ctx).unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };

        let rewrite = standard_integration_rewrite(&mut ctx, &call).unwrap();
        let rendered = rendered(&ctx, rewrite.new_expr);

        assert!(!rendered.starts_with("integrate("));
        assert!(rendered.contains("ln((s * x + b)^2 + a)"));
        assert!(rendered.contains("arctan((s * x + b) / sqrt(a))"));
        let mut required = rendered_required_conditions(&ctx, &rewrite);
        required.sort();
        assert_eq!(required, vec!["a > 0", "s ≠ 0"]);
    }

    #[test]
    fn standard_rewrite_uses_backend_for_expanded_symbolic_slope_positive_quadratic() {
        let mut ctx = Context::new();
        let target = parse("(m*s*x+b*m+c)/(s^2*x^2+2*b*s*x+b^2+a)", &mut ctx).unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };

        let rewrite = standard_integration_rewrite(&mut ctx, &call).unwrap();
        let rendered = rendered(&ctx, rewrite.new_expr);

        assert!(!rendered.starts_with("integrate("));
        assert!(rendered.contains("ln((s * x + b)^2 + a)"));
        assert!(rendered.contains("arctan((s * x + b) / sqrt(a))"));
        let mut required = rendered_required_conditions(&ctx, &rewrite);
        required.sort();
        assert_eq!(required, vec!["a > 0", "s ≠ 0"]);
    }
}
