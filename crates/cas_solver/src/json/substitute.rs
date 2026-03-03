use cas_api_models::{
    EngineJsonError as ApiEngineJsonError, EngineJsonSubstep, SpanJson as ApiSpanJson,
};
#[cfg(test)]
use cas_api_models::{SubstituteEvalError, SubstituteEvalMode};
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
    #[cfg(test)]
    fn to_text_error(&self) -> SubstituteEvalError {
        match self.field {
            ParseField::Expression => SubstituteEvalError::ParseExpression(format!(
                "Parse error in expression: {}",
                self.error
            )),
            ParseField::Target => {
                SubstituteEvalError::ParseTarget(format!("Parse error in target: {}", self.error))
            }
            ParseField::Replacement => SubstituteEvalError::ParseReplacement(format!(
                "Parse error in replacement: {}",
                self.error
            )),
        }
    }

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

/// Evaluate substitution from string inputs for text callers.
#[cfg(test)]
fn eval_substitute_from_str(
    expr_str: &str,
    target_str: &str,
    with_str: &str,
    mode: SubstituteEvalMode,
    steps_enabled: bool,
) -> Result<SubstituteEvalResult, SubstituteEvalError> {
    eval_substitute_impl(
        expr_str,
        target_str,
        with_str,
        mode.as_mode_str(),
        steps_enabled,
    )
    .map_err(|issue| issue.to_text_error())
}

/// Evaluate substitute subcommand in JSON mode.
#[cfg(test)]
fn evaluate_substitute_subcommand_json(
    expr: &str,
    target: &str,
    replacement: &str,
    mode: &str,
    steps: bool,
) -> String {
    let opts = SubstituteJsonOptions::from_mode_flags(mode, steps, true);
    substitute_str_to_json_impl(expr, target, replacement, opts)
}

/// Evaluate substitute subcommand in text mode and return output lines.
#[cfg(test)]
fn evaluate_substitute_subcommand_text_lines(
    expr: &str,
    target: &str,
    replacement: &str,
    mode: SubstituteEvalMode,
    steps_enabled: bool,
) -> Result<Vec<String>, String> {
    let output = eval_substitute_from_str(expr, target, replacement, mode, steps_enabled)
        .map_err(|error| error.to_string())?;
    Ok(format_substitute_subcommand_text_lines(
        &output,
        steps_enabled,
    ))
}

/// Evaluate substitute subcommand in text mode from mode string (`exact|power`).
#[cfg(test)]
fn evaluate_substitute_subcommand_text_lines_with_mode(
    expr: &str,
    target: &str,
    replacement: &str,
    mode: &str,
    steps_enabled: bool,
) -> Result<Vec<String>, String> {
    let parsed_mode = parse_substitute_eval_mode(mode);
    evaluate_substitute_subcommand_text_lines(expr, target, replacement, parsed_mode, steps_enabled)
}

/// Format text output lines for substitute subcommand.
#[cfg(test)]
fn format_substitute_subcommand_text_lines(
    output: &SubstituteEvalResult,
    steps_enabled: bool,
) -> Vec<String> {
    let mut lines = Vec::new();
    if steps_enabled && !output.steps.is_empty() {
        lines.push("Steps:".to_string());
        lines.extend(output.steps.iter().map(format_substitute_step_line));
    }
    lines.push(output.result.clone());
    lines
}

#[cfg(test)]
fn parse_substitute_eval_mode(mode: &str) -> SubstituteEvalMode {
    match mode {
        "exact" => SubstituteEvalMode::Exact,
        _ => SubstituteEvalMode::Power,
    }
}

#[cfg(test)]
fn format_substitute_step_line(step: &SubstituteEvalStep) -> String {
    match &step.note {
        Some(note) => format!(
            "  {} \u{2192} {} [{}] ({})",
            step.before, step.after, step.rule, note
        ),
        None => format!("  {} \u{2192} {} [{}]", step.before, step.after, step.rule),
    }
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
/// This is the **canonical entry point** for all JSON-returning substitution.
/// Both CLI and FFI should use this to ensure consistent behavior.
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
    use super::{
        eval_substitute_from_str, evaluate_substitute_subcommand_json,
        evaluate_substitute_subcommand_text_lines,
        evaluate_substitute_subcommand_text_lines_with_mode,
        format_substitute_subcommand_text_lines, SubstituteEvalError, SubstituteEvalMode,
        SubstituteEvalResult, SubstituteEvalStep,
    };

    #[test]
    fn eval_substitute_from_str_power_mode_returns_result() {
        let result =
            eval_substitute_from_str("x^4 + x^2 + 1", "x^2", "y", SubstituteEvalMode::Power, true)
                .expect("substitute should succeed");

        assert!(result.result.contains('y'));
        assert!(!result.steps.is_empty());
    }

    #[test]
    fn eval_substitute_from_str_parse_target_error_is_typed() {
        let err = eval_substitute_from_str(
            "x^2 + 1",
            "invalid(((",
            "y",
            SubstituteEvalMode::Power,
            false,
        )
        .expect_err("invalid target should fail");

        match err {
            SubstituteEvalError::ParseTarget(message) => {
                assert!(message.contains("Parse error in target"));
            }
            _ => panic!("expected ParseTarget error"),
        }
    }

    #[test]
    fn evaluate_substitute_subcommand_json_returns_ok_contract() {
        let out = evaluate_substitute_subcommand_json("x^2+1", "x", "y", "exact", false);
        let json: serde_json::Value = serde_json::from_str(&out).expect("valid json");
        assert_eq!(json["ok"], true);
    }

    #[test]
    fn evaluate_substitute_subcommand_text_lines_success() {
        let lines = evaluate_substitute_subcommand_text_lines(
            "x^4 + x^2 + 1",
            "x^2",
            "y",
            SubstituteEvalMode::Power,
            true,
        )
        .expect("text output");

        assert!(!lines.is_empty());
        assert_eq!(lines.first().map(String::as_str), Some("Steps:"));
        assert!(lines.last().is_some_and(|line| line.contains('y')));
    }

    #[test]
    fn evaluate_substitute_subcommand_text_lines_with_mode_exact() {
        let lines =
            evaluate_substitute_subcommand_text_lines_with_mode("x + 1", "x", "y", "exact", false)
                .expect("text output");
        assert!(lines.last().is_some_and(|line| line.contains('y')));
    }

    #[test]
    fn evaluate_substitute_subcommand_text_lines_parse_error_passthrough() {
        let err = evaluate_substitute_subcommand_text_lines(
            "x^2 + 1",
            "invalid(((",
            "y",
            SubstituteEvalMode::Power,
            false,
        )
        .expect_err("expected parse error");

        assert!(err.contains("Parse error in target"));
    }

    #[test]
    fn format_substitute_subcommand_text_lines_renders_steps_and_result() {
        let output = SubstituteEvalResult {
            result: "2 * y + 1".to_string(),
            steps: vec![
                SubstituteEvalStep {
                    rule: "subst".to_string(),
                    before: "x^2".to_string(),
                    after: "y".to_string(),
                    note: None,
                },
                SubstituteEvalStep {
                    rule: "simplify".to_string(),
                    before: "y + y".to_string(),
                    after: "2 * y".to_string(),
                    note: Some("combine like terms".to_string()),
                },
            ],
        };

        let lines = format_substitute_subcommand_text_lines(&output, true);
        assert_eq!(lines[0], "Steps:");
        assert_eq!(lines[1], "  x^2 \u{2192} y [subst]");
        assert_eq!(
            lines[2],
            "  y + y \u{2192} 2 * y [simplify] (combine like terms)"
        );
        assert_eq!(lines[3], "2 * y + 1");
    }
}
