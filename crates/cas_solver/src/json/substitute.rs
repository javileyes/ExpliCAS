use cas_api_models::{
    EngineJsonError as ApiEngineJsonError, EngineJsonSubstep, SpanJson as ApiSpanJson,
};
pub use cas_api_models::{
    SubstituteJsonOptions, SubstituteJsonResponse, SubstituteOptionsInner, SubstituteOptionsJson,
    SubstituteRequestEcho,
};
use cas_ast::hold::strip_all_holds;
use cas_ast::{Context, ExprId};

/// Substitution mode for typed non-JSON evaluation APIs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubstituteEvalMode {
    Exact,
    Power,
}

impl SubstituteEvalMode {
    fn as_mode_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Power => "power",
        }
    }
}

/// One substitution step for typed non-JSON evaluation APIs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubstituteEvalStep {
    pub rule: String,
    pub before: String,
    pub after: String,
    pub note: Option<String>,
}

/// Result payload for typed non-JSON substitution evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubstituteEvalResult {
    pub result: String,
    pub steps: Vec<SubstituteEvalStep>,
}

/// Parse-time errors produced by [`eval_substitute_from_str`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubstituteEvalError {
    ParseExpression(String),
    ParseTarget(String),
    ParseReplacement(String),
}

impl std::fmt::Display for SubstituteEvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseExpression(message)
            | Self::ParseTarget(message)
            | Self::ParseReplacement(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for SubstituteEvalError {}

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
pub fn eval_substitute_from_str(
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
            return if opts.pretty {
                resp.to_json_pretty()
            } else {
                resp.to_json()
            };
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

    if opts.pretty {
        resp.to_json_pretty()
    } else {
        resp.to_json()
    }
}

/// Typed helper for callers that already have substitute options available.
pub fn substitute_str_to_json_with_options(
    expr_str: &str,
    target_str: &str,
    with_str: &str,
    mode: &str,
    steps: bool,
    pretty: bool,
) -> String {
    let opts = SubstituteJsonOptions {
        mode: mode.to_string(),
        steps,
        pretty,
    };
    substitute_str_to_json_impl(expr_str, target_str, with_str, opts)
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
    // Parse options (with defaults)
    let opts: SubstituteJsonOptions = match opts_json {
        Some(json) => serde_json::from_str(json).unwrap_or_default(),
        None => SubstituteJsonOptions::default(),
    };
    substitute_str_to_json_impl(expr_str, target_str, with_str, opts)
}

#[cfg(test)]
mod tests {
    use super::{eval_substitute_from_str, SubstituteEvalError, SubstituteEvalMode};

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
}
