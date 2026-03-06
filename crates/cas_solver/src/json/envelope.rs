use super::stateless_eval::evaluate_prepared_stateless_request;
use crate::{
    AssumptionRecord, BlockedHint, DomainWarning, Engine, EvalOptions, EvalResult,
    ImplicitCondition,
};
use cas_api_models::{
    AssumptionDto, BlockedHintDto, ConditionDto, EnvelopeEvalOptions, ExprDto, OutputEnvelope,
    RequestInfo, RequestOptions, TransparencyDto,
};

fn display_expr(ctx: &cas_ast::Context, id: cas_ast::ExprId) -> String {
    cas_formatter::DisplayExpr { context: ctx, id }.to_string()
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

fn map_required_conditions(
    required_conditions: &[ImplicitCondition],
    ctx: &cas_ast::Context,
) -> Vec<ConditionDto> {
    required_conditions
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
        .collect()
}

fn map_assumptions_used(
    assumptions: &[AssumptionRecord],
    warnings: &[DomainWarning],
) -> Vec<AssumptionDto> {
    let mut out: Vec<AssumptionDto> = assumptions
        .iter()
        .map(|a| AssumptionDto {
            kind: a.kind.clone(),
            display: a.message.clone(),
            expr_canonical: a.expr.clone(),
            rule: "solver".to_string(),
        })
        .collect();
    out.extend(warnings.iter().map(|w| AssumptionDto {
        kind: "domain_warning".to_string(),
        display: w.message.clone(),
        expr_canonical: String::new(),
        rule: w.rule_name.clone(),
    }));
    out
}

fn map_blocked_hints(blocked_hints: &[BlockedHint]) -> Vec<BlockedHintDto> {
    blocked_hints
        .iter()
        .map(|h| BlockedHintDto {
            rule: h.rule.clone(),
            requires: vec![h.key.condition_display().to_string()],
            tip: h.suggestion.to_string(),
        })
        .collect()
}

fn build_transparency(
    required_conditions: &[ImplicitCondition],
    solver_assumptions: &[AssumptionRecord],
    domain_warnings: &[DomainWarning],
    blocked_hints: &[BlockedHint],
    ctx: &cas_ast::Context,
) -> TransparencyDto {
    TransparencyDto {
        required_conditions: map_required_conditions(required_conditions, ctx),
        assumptions_used: map_assumptions_used(solver_assumptions, domain_warnings),
        blocked_hints: map_blocked_hints(blocked_hints),
    }
}

/// Stateless envelope evaluation used by application wrappers.
pub fn eval_str_to_output_envelope(expr: &str, opts: &EnvelopeEvalOptions) -> OutputEnvelope {
    let mut engine = Engine::new();
    let mut eval_options = EvalOptions::default();
    eval_options.shared.semantics.domain_mode =
        crate::eval_json_options::domain_mode_from_str(&opts.domain);
    eval_options.shared.semantics.value_domain =
        crate::eval_json_options::value_domain_from_str(&opts.value_domain);

    let prepared =
        match crate::build_eval_json_request_for_input(expr, &mut engine.simplifier.context, false)
        {
            Ok(request) => request,
            Err(e) => {
                return OutputEnvelope::eval_error(
                    build_request_info(expr, opts),
                    format!("Parse error: {}", e),
                );
            }
        };

    let output_view = match evaluate_prepared_stateless_request(&mut engine, eval_options, prepared)
    {
        Ok(view) => view,
        Err(e) => return OutputEnvelope::eval_error(build_request_info(expr, opts), e.to_string()),
    };

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
        EvalResult::SolutionSet(solution_set) => OutputEnvelope::eval_success(
            build_request_info(expr, opts),
            ExprDto::from_display(crate::display_solution_set(
                &engine.simplifier.context,
                solution_set,
            )),
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

/// Stateless CLI helper for `envelope-json` command.
pub fn evaluate_envelope_json_command(expr: &str, domain: &str, value_domain: &str) -> String {
    let opts = EnvelopeEvalOptions {
        domain: domain.to_string(),
        value_domain: value_domain.to_string(),
    };
    eval_str_to_output_envelope(expr, &opts).to_json_pretty()
}

#[cfg(test)]
mod tests {
    use super::evaluate_envelope_json_command;

    #[test]
    fn evaluate_envelope_json_command_returns_json_contract() {
        let payload = evaluate_envelope_json_command("x + x", "generic", "real");
        assert!(payload.contains("\"schema_version\": 1"));
        assert!(payload.contains("\"kind\": \"eval_result\""));
    }
}
