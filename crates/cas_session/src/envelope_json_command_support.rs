use cas_api_models::{
    AssumptionDto, BlockedHintDto, ConditionDto, EnvelopeEvalOptions, RequestInfo, RequestOptions,
    TransparencyDto,
};
use cas_solver::{AssumptionRecord, BlockedHint, DomainWarning, ImplicitCondition};

pub(crate) fn domain_mode_from_str(value: &str) -> cas_solver::DomainMode {
    match value {
        "strict" => cas_solver::DomainMode::Strict,
        "assume" => cas_solver::DomainMode::Assume,
        _ => cas_solver::DomainMode::Generic,
    }
}

pub(crate) fn value_domain_from_str(value: &str) -> cas_solver::ValueDomain {
    match value {
        "complex" => cas_solver::ValueDomain::ComplexEnabled,
        _ => cas_solver::ValueDomain::RealOnly,
    }
}

pub(crate) fn display_expr(ctx: &cas_ast::Context, id: cas_ast::ExprId) -> String {
    cas_formatter::DisplayExpr { context: ctx, id }.to_string()
}

pub(crate) fn build_request_info(expr: &str, opts: &EnvelopeEvalOptions) -> RequestInfo {
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

pub(crate) fn build_transparency(
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
            let expr_display = cas_formatter::DisplayExpr {
                context: ctx,
                id: expr_id,
            }
            .to_string();
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
