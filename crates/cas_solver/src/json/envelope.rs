use super::mappers::{map_assumptions_used, map_blocked_hints, map_required_conditions};
use crate::{
    AssumptionRecord, BlockedHint, DomainMode, DomainWarning, Engine, EvalAction, EvalOptions,
    EvalRequest, EvalResult, ImplicitCondition, ValueDomain,
};
use cas_api_models::{
    EnvelopeEvalOptions, ExprDto, OutputEnvelope, RequestInfo, RequestOptions, TransparencyDto,
};
use cas_formatter::DisplayExpr;

pub fn eval_str_to_output_envelope(expr: &str, opts: &EnvelopeEvalOptions) -> OutputEnvelope {
    let mut engine = Engine::new();
    let mut eval_options = EvalOptions::default();
    eval_options.shared.semantics.domain_mode = parse_domain_mode(&opts.domain);
    eval_options.shared.semantics.value_domain = parse_value_domain(&opts.value_domain);

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
    let output_view = crate::eval_output_view(&output);

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

fn display_expr(ctx: &cas_ast::Context, id: cas_ast::ExprId) -> String {
    DisplayExpr { context: ctx, id }.to_string()
}

fn build_request_info(expr: &str, opts: &EnvelopeEvalOptions) -> RequestInfo {
    RequestInfo::eval(
        expr,
        RequestOptions {
            domain_mode: opts.domain.clone(),
            value_domain: opts.value_domain.clone(),
            hints: true,
            explain: false,
        },
    )
}

fn build_transparency(
    required_conditions: &[ImplicitCondition],
    solver_assumptions: &[AssumptionRecord],
    domain_warnings: &[DomainWarning],
    blocked_hints: &[BlockedHint],
    ctx: &cas_ast::Context,
) -> TransparencyDto {
    let required_conditions = map_required_conditions(required_conditions, ctx);
    let assumptions_used = map_assumptions_used(solver_assumptions, domain_warnings);
    let blocked_hints = map_blocked_hints(blocked_hints);

    TransparencyDto {
        required_conditions,
        assumptions_used,
        blocked_hints,
    }
}

fn parse_domain_mode(domain: &str) -> DomainMode {
    match domain {
        "strict" => DomainMode::Strict,
        "assume" => DomainMode::Assume,
        _ => DomainMode::Generic,
    }
}

fn parse_value_domain(value_domain: &str) -> ValueDomain {
    match value_domain {
        "complex" => ValueDomain::ComplexEnabled,
        _ => ValueDomain::RealOnly,
    }
}
