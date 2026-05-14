use crate::EvalResult;
use cas_api_models::{EnvelopeEvalOptions, ExprDto, OutputEnvelope};

use super::common::{build_request_info, display_expr};
use super::transparency::{build_transparency, TransparencyInput};

pub fn build_success_envelope(
    expr: &str,
    opts: &EnvelopeEvalOptions,
    ctx: &cas_ast::Context,
    output_view: &crate::EvalOutputView,
) -> OutputEnvelope {
    match &output_view.result {
        EvalResult::Expr(id) => {
            let result_display = display_expr(ctx, *id);
            OutputEnvelope::eval_success(
                build_request_info(expr, opts),
                ExprDto::from_display(result_display.clone()),
                build_transparency(TransparencyInput {
                    required_conditions: &output_view.required_conditions,
                    solver_assumptions: &output_view.solver_assumptions,
                    domain_warnings: &output_view.domain_warnings,
                    steps: &output_view.steps.0,
                    blocked_hints: &output_view.blocked_hints,
                    ctx,
                    raw_input: expr,
                    result_display: Some(result_display.as_str()),
                }),
            )
        }
        EvalResult::Set(v) if !v.is_empty() => {
            let result_display = display_expr(ctx, v[0]);
            OutputEnvelope::eval_success(
                build_request_info(expr, opts),
                ExprDto::from_display(result_display.clone()),
                build_transparency(TransparencyInput {
                    required_conditions: &output_view.required_conditions,
                    solver_assumptions: &output_view.solver_assumptions,
                    domain_warnings: &output_view.domain_warnings,
                    steps: &output_view.steps.0,
                    blocked_hints: &output_view.blocked_hints,
                    ctx,
                    raw_input: expr,
                    result_display: Some(result_display.as_str()),
                }),
            )
        }
        EvalResult::Bool(b) => OutputEnvelope::eval_success(
            build_request_info(expr, opts),
            ExprDto::from_display(b.to_string()),
            build_transparency(TransparencyInput {
                required_conditions: &output_view.required_conditions,
                solver_assumptions: &output_view.solver_assumptions,
                domain_warnings: &output_view.domain_warnings,
                steps: &output_view.steps.0,
                blocked_hints: &output_view.blocked_hints,
                ctx,
                raw_input: expr,
                result_display: None,
            }),
        ),
        EvalResult::Text { plain, .. } => OutputEnvelope::eval_success(
            build_request_info(expr, opts),
            ExprDto::from_display(plain.clone()),
            build_transparency(TransparencyInput {
                required_conditions: &output_view.required_conditions,
                solver_assumptions: &output_view.solver_assumptions,
                domain_warnings: &output_view.domain_warnings,
                steps: &output_view.steps.0,
                blocked_hints: &output_view.blocked_hints,
                ctx,
                raw_input: expr,
                result_display: None,
            }),
        ),
        EvalResult::SolutionSet(solution_set) => OutputEnvelope::eval_success(
            build_request_info(expr, opts),
            ExprDto::from_display(crate::display_solution_set(ctx, solution_set)),
            build_transparency(TransparencyInput {
                required_conditions: &output_view.required_conditions,
                solver_assumptions: &output_view.solver_assumptions,
                domain_warnings: &output_view.domain_warnings,
                steps: &output_view.steps.0,
                blocked_hints: &output_view.blocked_hints,
                ctx,
                raw_input: expr,
                result_display: None,
            }),
        ),
        _ => OutputEnvelope::eval_error(build_request_info(expr, opts), "No result expression"),
    }
}
