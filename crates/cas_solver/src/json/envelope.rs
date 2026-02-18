use super::mappers::{map_assumptions_used, map_blocked_hints, map_required_conditions};
use cas_api_models::{ExprDto, OutputEnvelope, RequestInfo, RequestOptions, TransparencyDto};
use cas_formatter::DisplayExpr;

#[derive(Clone, Debug)]
pub struct EnvelopeEvalOptions {
    pub domain: String,
    pub value_domain: String,
}

impl Default for EnvelopeEvalOptions {
    fn default() -> Self {
        Self {
            domain: "generic".to_string(),
            value_domain: "real".to_string(),
        }
    }
}

pub fn eval_str_to_output_envelope(expr: &str, opts: &EnvelopeEvalOptions) -> OutputEnvelope {
    let mut engine = cas_engine::eval::Engine::new();
    let mut session = cas_session::SessionState::new();

    session.options_mut().shared.semantics.domain_mode = parse_domain_mode(&opts.domain);
    session.options_mut().shared.semantics.value_domain = parse_value_domain(&opts.value_domain);

    let parsed = match cas_parser::parse(expr, &mut engine.simplifier.context) {
        Ok(id) => id,
        Err(e) => {
            return OutputEnvelope::eval_error(
                build_request_info(expr, opts),
                format!("Parse error: {}", e),
            );
        }
    };

    let req = cas_engine::eval::EvalRequest {
        raw_input: expr.to_string(),
        parsed,
        action: cas_engine::eval::EvalAction::Simplify,
        auto_store: false,
    };

    let output = match engine.eval(&mut session, req) {
        Ok(o) => o,
        Err(e) => return OutputEnvelope::eval_error(build_request_info(expr, opts), e.to_string()),
    };

    match &output.result {
        cas_engine::eval::EvalResult::Expr(id) => OutputEnvelope::eval_success(
            build_request_info(expr, opts),
            ExprDto::from_display(display_expr(&engine.simplifier.context, *id)),
            build_transparency(&output, &engine.simplifier.context),
        ),
        cas_engine::eval::EvalResult::Set(v) if !v.is_empty() => OutputEnvelope::eval_success(
            build_request_info(expr, opts),
            ExprDto::from_display(display_expr(&engine.simplifier.context, v[0])),
            build_transparency(&output, &engine.simplifier.context),
        ),
        cas_engine::eval::EvalResult::Bool(b) => OutputEnvelope::eval_success(
            build_request_info(expr, opts),
            ExprDto::from_display(b.to_string()),
            build_transparency(&output, &engine.simplifier.context),
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
    output: &cas_engine::eval::EvalOutput,
    ctx: &cas_ast::Context,
) -> TransparencyDto {
    let required_conditions = map_required_conditions(&output.required_conditions, ctx);
    let assumptions_used =
        map_assumptions_used(&output.solver_assumptions, &output.domain_warnings);
    let blocked_hints = map_blocked_hints(&output.blocked_hints);

    TransparencyDto {
        required_conditions,
        assumptions_used,
        blocked_hints,
    }
}

fn parse_domain_mode(domain: &str) -> cas_engine::domain::DomainMode {
    match domain {
        "strict" => cas_engine::domain::DomainMode::Strict,
        "assume" => cas_engine::domain::DomainMode::Assume,
        _ => cas_engine::domain::DomainMode::Generic,
    }
}

fn parse_value_domain(value_domain: &str) -> cas_engine::semantics::ValueDomain {
    match value_domain {
        "complex" => cas_engine::semantics::ValueDomain::ComplexEnabled,
        _ => cas_engine::semantics::ValueDomain::RealOnly,
    }
}
