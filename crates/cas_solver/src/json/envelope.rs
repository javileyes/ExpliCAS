use super::session::JsonEvalSession;
use cas_api_models::{
    AssumptionDto, BlockedHintDto, ConditionDto, ExprDto, OutputEnvelope, RequestInfo,
    RequestOptions, TransparencyDto,
};
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
    let mut session = JsonEvalSession::new(cas_engine::options::EvalOptions::default());

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
    use cas_engine::implicit_domain::ImplicitCondition;

    let required_conditions = output
        .required_conditions
        .iter()
        .map(|cond| {
            let (kind, expr_id) = match cond {
                ImplicitCondition::NonNegative(e) => ("NonNegative", *e),
                ImplicitCondition::Positive(e) => ("Positive", *e),
                ImplicitCondition::NonZero(e) => ("NonZero", *e),
            };
            let expr_display = display_expr(ctx, expr_id);
            ConditionDto {
                kind: kind.to_string(),
                display: cond.display(ctx),
                expr_display: expr_display.clone(),
                expr_canonical: expr_display,
            }
        })
        .collect();

    let mut assumptions_used: Vec<AssumptionDto> = output
        .solver_assumptions
        .iter()
        .map(|a| AssumptionDto {
            kind: a.kind.clone(),
            display: a.message.clone(),
            expr_canonical: a.expr.clone(),
            rule: "solver".to_string(),
        })
        .collect();
    assumptions_used.extend(output.domain_warnings.iter().map(|w| AssumptionDto {
        kind: "domain_warning".to_string(),
        display: w.message.clone(),
        expr_canonical: String::new(),
        rule: w.rule_name.clone(),
    }));

    let blocked_hints = output
        .blocked_hints
        .iter()
        .map(|h| BlockedHintDto {
            rule: h.rule.clone(),
            requires: vec![h.key.condition_display().to_string()],
            tip: h.suggestion.to_string(),
        })
        .collect();

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
