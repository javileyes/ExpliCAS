use cas_api_models::{
    EngineJsonError as ApiEngineJsonError, EngineJsonSubstep, SpanJson as ApiSpanJson,
};
pub use cas_api_models::{
    SubstituteJsonOptions, SubstituteJsonResponse, SubstituteOptionsInner, SubstituteOptionsJson,
    SubstituteRequestEcho,
};
use cas_engine::strip_all_holds;

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

    // Create context
    let mut ctx = cas_ast::Context::new();

    // Parse expressions
    let expr = match cas_parser::parse(expr_str, &mut ctx) {
        Ok(id) => id,
        Err(e) => {
            let resp = SubstituteJsonResponse::err(
                ApiEngineJsonError::parse(
                    format!("Failed to parse expression: {}", e),
                    e.span().map(|s| ApiSpanJson {
                        start: s.start,
                        end: s.end,
                    }),
                ),
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

    let target = match cas_parser::parse(target_str, &mut ctx) {
        Ok(id) => id,
        Err(e) => {
            let resp = SubstituteJsonResponse::err(
                ApiEngineJsonError::parse(
                    format!("Failed to parse target: {}", e),
                    e.span().map(|s| ApiSpanJson {
                        start: s.start,
                        end: s.end,
                    }),
                ),
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

    let replacement = match cas_parser::parse(with_str, &mut ctx) {
        Ok(id) => id,
        Err(e) => {
            let resp = SubstituteJsonResponse::err(
                ApiEngineJsonError::parse(
                    format!("Failed to parse replacement: {}", e),
                    e.span().map(|s| ApiSpanJson {
                        start: s.start,
                        end: s.end,
                    }),
                ),
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

    // Build substitute options
    let sub_opts = match opts.mode.as_str() {
        "exact" => crate::substitute::SubstituteOptions::exact(),
        _ => crate::substitute::SubstituteOptions::power_aware_no_remainder(),
    };

    // Perform substitution
    let sub_result =
        crate::substitute::substitute_with_steps(&mut ctx, expr, target, replacement, sub_opts);

    // Strip __hold from result
    let clean_result = strip_all_holds(&mut ctx, sub_result.expr);
    let result_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &ctx,
            id: clean_result
        }
    );

    // Convert steps to JSON format
    let json_steps: Vec<EngineJsonSubstep> = if opts.steps {
        sub_result
            .steps
            .into_iter()
            .map(|s| EngineJsonSubstep {
                rule: s.rule,
                before: s.before,
                after: s.after,
                note: s.note,
            })
            .collect()
    } else {
        vec![]
    };

    let resp = SubstituteJsonResponse::ok(result_str, request, options, json_steps);

    if opts.pretty {
        resp.to_json_pretty()
    } else {
        resp.to_json()
    }
}
