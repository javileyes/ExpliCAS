//! Canonical stateless substitute-json bridge for external frontends.

use cas_api_models::{
    EngineJsonError as ApiEngineJsonError, EngineJsonSubstep, SpanJson as ApiSpanJson,
    SubstituteEvalResult, SubstituteEvalStep, SubstituteJsonOptions, SubstituteJsonResponse,
    SubstituteOptionsInner, SubstituteOptionsJson, SubstituteRequestEcho,
};
use cas_ast::hold::strip_all_holds;
use cas_ast::{Context, ExprId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParseField {
    Expression,
    Target,
    Replacement,
}

#[derive(Clone, Debug)]
struct SubstituteParseIssue {
    field: ParseField,
    error: String,
    span: Option<ApiSpanJson>,
}

impl SubstituteParseIssue {
    fn to_json_error(&self) -> ApiEngineJsonError {
        let message = match self.field {
            ParseField::Expression => format!("Failed to parse expression: {}", self.error),
            ParseField::Target => format!("Failed to parse target: {}", self.error),
            ParseField::Replacement => format!("Failed to parse replacement: {}", self.error),
        };
        ApiEngineJsonError::parse(message, self.span)
    }
}

fn parse_component(
    input: &str,
    ctx: &mut Context,
    field: ParseField,
) -> Result<ExprId, SubstituteParseIssue> {
    cas_parser::parse(input, ctx).map_err(|e| SubstituteParseIssue {
        field,
        error: e.to_string(),
        span: e.span().map(|s| ApiSpanJson {
            start: s.start,
            end: s.end,
        }),
    })
}

fn parse_substitute_input(
    expr_str: &str,
    target_str: &str,
    replacement_str: &str,
) -> Result<(Context, ExprId, ExprId, ExprId), SubstituteParseIssue> {
    let mut ctx = Context::new();
    let expr = parse_component(expr_str, &mut ctx, ParseField::Expression)?;
    let target = parse_component(target_str, &mut ctx, ParseField::Target)?;
    let replacement = parse_component(replacement_str, &mut ctx, ParseField::Replacement)?;
    Ok((ctx, expr, target, replacement))
}

fn eval_substitute_impl(
    expr_str: &str,
    target_str: &str,
    replacement_str: &str,
    mode: &str,
    steps_enabled: bool,
) -> Result<SubstituteEvalResult, SubstituteParseIssue> {
    let (mut ctx, expr, target, replacement) =
        parse_substitute_input(expr_str, target_str, replacement_str)?;

    let sub_opts = match mode {
        "exact" => cas_solver::substitute::SubstituteOptions::exact(),
        _ => cas_solver::substitute::SubstituteOptions::power_aware_no_remainder(),
    };

    let sub_result = cas_solver::substitute::substitute_with_steps(
        &mut ctx,
        expr,
        target,
        replacement,
        sub_opts,
    );

    let clean_result = strip_all_holds(&mut ctx, sub_result.expr);
    let result = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: clean_result
        }
    );

    let steps = if steps_enabled {
        sub_result
            .steps
            .into_iter()
            .map(|step| SubstituteEvalStep {
                rule: step.rule,
                before: step.before,
                after: step.after,
                note: step.note,
            })
            .collect()
    } else {
        Vec::new()
    };

    Ok(SubstituteEvalResult { result, steps })
}

/// Stateless canonical substitute JSON entry point.
pub fn evaluate_substitute_json_canonical(
    expr: &str,
    target: &str,
    replacement: &str,
    opts_json: Option<&str>,
) -> String {
    let opts = SubstituteJsonOptions::parse_optional_json(opts_json);
    let request = SubstituteRequestEcho {
        expr: expr.to_string(),
        target: target.to_string(),
        with_expr: replacement.to_string(),
    };
    let options = SubstituteOptionsJson {
        substitute: SubstituteOptionsInner {
            mode: opts.mode.clone(),
            steps: opts.steps,
        },
    };

    let eval = match eval_substitute_impl(expr, target, replacement, &opts.mode, opts.steps) {
        Ok(eval) => eval,
        Err(issue) => {
            let resp = SubstituteJsonResponse::err(
                issue.to_json_error(),
                request.clone(),
                options.clone(),
            );
            return resp.to_json_with_pretty(opts.pretty);
        }
    };

    let json_steps: Vec<EngineJsonSubstep> = eval
        .steps
        .into_iter()
        .map(|step| EngineJsonSubstep {
            rule: step.rule,
            before: step.before,
            after: step.after,
            note: step.note,
        })
        .collect();
    let resp = SubstituteJsonResponse::ok(eval.result, request, options, json_steps);
    resp.to_json_with_pretty(opts.pretty)
}
