use cas_api_models::{EnvelopeEvalOptions, ExprDto, OutputEnvelope};
use cas_solver::{EvalAction, EvalOptions, EvalRequest, EvalResult};

use crate::envelope_json_command_support::{
    build_request_info, build_transparency, display_expr, domain_mode_from_str,
    value_domain_from_str,
};

pub(crate) fn eval_str_to_output_envelope(
    expr: &str,
    opts: &EnvelopeEvalOptions,
) -> OutputEnvelope {
    let mut engine = cas_solver::Engine::new();
    let mut eval_options = EvalOptions::default();
    eval_options.shared.semantics.domain_mode = domain_mode_from_str(&opts.domain);
    eval_options.shared.semantics.value_domain = value_domain_from_str(&opts.value_domain);

    let parsed = match cas_parser::parse(expr, &mut engine.simplifier.context) {
        Ok(id) => id,
        Err(e) => {
            return OutputEnvelope::eval_error(
                build_request_info(expr, opts),
                format!("Parse error: {}", e),
            );
        }
    };

    let req = EvalRequest {
        raw_input: expr.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = match engine.eval_stateless(eval_options, req) {
        Ok(o) => o,
        Err(e) => return OutputEnvelope::eval_error(build_request_info(expr, opts), e.to_string()),
    };
    let output_view = cas_solver::eval_output_view(&output);

    match &output_view.result {
        EvalResult::Expr(id) => OutputEnvelope::eval_success(
            build_request_info(expr, opts),
            ExprDto::from_display(display_expr(&engine.simplifier.context, *id)),
            build_transparency(
                &output_view.required_conditions,
                &output_view.solver_assumptions,
                &output_view.domain_warnings,
                &output_view.blocked_hints,
                &engine.simplifier.context,
            ),
        ),
        EvalResult::Set(v) if !v.is_empty() => OutputEnvelope::eval_success(
            build_request_info(expr, opts),
            ExprDto::from_display(display_expr(&engine.simplifier.context, v[0])),
            build_transparency(
                &output_view.required_conditions,
                &output_view.solver_assumptions,
                &output_view.domain_warnings,
                &output_view.blocked_hints,
                &engine.simplifier.context,
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
                &engine.simplifier.context,
            ),
        ),
        _ => OutputEnvelope::eval_error(build_request_info(expr, opts), "No result expression"),
    }
}
