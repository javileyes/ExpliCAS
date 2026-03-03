use cas_api_models::{
    EngineJsonError as ApiEngineJsonError, EngineJsonSubstep, SpanJson as ApiSpanJson,
};
use cas_api_models::{
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
    with_str: &str,
) -> Result<(Context, ExprId, ExprId, ExprId), SubstituteParseIssue> {
    let mut ctx = Context::new();
    let expr = parse_component(expr_str, &mut ctx, ParseField::Expression)?;
    let target = parse_component(target_str, &mut ctx, ParseField::Target)?;
    let replacement = parse_component(with_str, &mut ctx, ParseField::Replacement)?;
    Ok((ctx, expr, target, replacement))
}

fn eval_substitute_impl(
    expr_str: &str,
    target_str: &str,
    with_str: &str,
    mode: &str,
    steps_enabled: bool,
) -> Result<SubstituteEvalResult, SubstituteParseIssue> {
    let (mut ctx, expr, target, replacement) =
        parse_substitute_input(expr_str, target_str, with_str)?;

    let sub_opts = match mode {
        "exact" => crate::substitute::SubstituteOptions::exact(),
        _ => crate::substitute::SubstituteOptions::power_aware_no_remainder(),
    };

    let sub_result =
        crate::substitute::substitute_with_steps(&mut ctx, expr, target, replacement, sub_opts);

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
            .map(|s| SubstituteEvalStep {
                rule: s.rule,
                before: s.before,
                after: s.after,
                note: s.note,
            })
            .collect()
    } else {
        vec![]
    };

    Ok(SubstituteEvalResult { result, steps })
}

fn substitute_str_to_json_impl(
    expr_str: &str,
    target_str: &str,
    with_str: &str,
    opts: SubstituteJsonOptions,
) -> String {
    let request = SubstituteRequestEcho {
        expr: expr_str.to_string(),
        target: target_str.to_string(),
        with_expr: with_str.to_string(),
    };

    let options = SubstituteOptionsJson {
        substitute: SubstituteOptionsInner {
            mode: opts.mode.clone(),
            steps: opts.steps,
        },
    };

    let eval = match eval_substitute_impl(expr_str, target_str, with_str, &opts.mode, opts.steps) {
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
        .map(|s| EngineJsonSubstep {
            rule: s.rule,
            before: s.before,
            after: s.after,
            note: s.note,
        })
        .collect();

    let resp = SubstituteJsonResponse::ok(eval.result, request, options, json_steps);
    resp.to_json_with_pretty(opts.pretty)
}

/// Substitute an expression and return JSON response.
///
/// This is the **solver-level canonical entry point** for JSON-returning
/// stateless substitution. Frontends should normally go through
/// `cas_session::evaluate_substitute_json_canonical`.
///
/// # Arguments
/// * `expr_str` - Expression string to substitute in
/// * `target_str` - Target expression to replace
/// * `with_str` - Replacement expression
/// * `opts_json` - Options JSON string (optional, see `SubstituteJsonOptions`)
///
/// # Returns
/// JSON string with `SubstituteJsonResponse` (schema v1).
/// Always returns valid JSON, even on errors.
pub fn substitute_str_to_json(
    expr_str: &str,
    target_str: &str,
    with_str: &str,
    opts_json: Option<&str>,
) -> String {
    let opts = SubstituteJsonOptions::parse_optional_json(opts_json);
    substitute_str_to_json_impl(expr_str, target_str, with_str, opts)
}

#[cfg(test)]
mod tests {
    use super::substitute_str_to_json;

    #[test]
    fn substitute_str_to_json_returns_ok_contract() {
        let out = substitute_str_to_json("x^2+1", "x", "y", Some(r#"{"mode":"exact"}"#));
        let json: serde_json::Value = serde_json::from_str(&out).expect("valid json");
        assert_eq!(json["ok"], true);
    }

    #[test]
    fn substitute_str_to_json_parse_error_contract() {
        let out = substitute_str_to_json("x^2 + 1", "invalid(((", "y", None);
        let json: serde_json::Value = serde_json::from_str(&out).expect("valid json");
        assert_eq!(json["ok"], false);
        assert_eq!(json["error"]["kind"], "ParseError");
    }
}
