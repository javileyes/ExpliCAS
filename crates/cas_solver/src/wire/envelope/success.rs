use crate::EvalResult;
use cas_api_models::{EnvelopeEvalOptions, ExprDto, OutputEnvelope};

use super::common::{build_request_info, display_expr};
use super::transparency::build_transparency;

pub fn build_success_envelope(
    expr: &str,
    opts: &EnvelopeEvalOptions,
    ctx: &cas_ast::Context,
    output_view: &crate::EvalOutputView,
) -> OutputEnvelope {
    match &output_view.result {
        EvalResult::Expr(id) => OutputEnvelope::eval_success(
            build_request_info(expr, opts),
            ExprDto::from_display(display_expr(ctx, *id)),
            build_transparency(
                &output_view.required_conditions,
                &output_view.solver_assumptions,
                &output_view.domain_warnings,
                &output_view.blocked_hints,
                ctx,
            ),
        ),
        EvalResult::Set(v) if !v.is_empty() => OutputEnvelope::eval_success(
            build_request_info(expr, opts),
            ExprDto::from_display(display_expr(ctx, v[0])),
            build_transparency(
                &output_view.required_conditions,
                &output_view.solver_assumptions,
                &output_view.domain_warnings,
                &output_view.blocked_hints,
                ctx,
            ),
        ),
        EvalResult::Bool(b) => OutputEnvelope::eval_success(
            build_request_info(expr, opts),
            ExprDto::from_display(b.to_string()),
            build_transparency(
                &output_view.required_conditions,
                &output_view.solver_assumptions,
                &output_view.domain_warnings,
                &output_view.blocked_hints,
                ctx,
            ),
        ),
        EvalResult::SolutionSet(solution_set) => OutputEnvelope::eval_success(
            build_request_info(expr, opts),
            ExprDto::from_display(crate::display_solution_set(ctx, solution_set)),
            build_transparency(
                &output_view.required_conditions,
                &output_view.solver_assumptions,
                &output_view.domain_warnings,
                &output_view.blocked_hints,
                ctx,
            ),
        ),
        _ => OutputEnvelope::eval_error(build_request_info(expr, opts), "No result expression"),
    }
}
