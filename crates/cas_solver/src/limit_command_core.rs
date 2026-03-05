use cas_api_models::{LimitEvalError, LimitEvalResult, LimitJsonResponse};
use cas_formatter::DisplayExpr;

use crate::limit_command_parse::parse_limit_command_input;
use crate::limit_command_types::{
    LimitCommandEvalError, LimitCommandEvalOutput, LimitSubcommandEvalError,
    LimitSubcommandEvalOutput,
};

fn eval_limit_from_str(
    expr: &str,
    var: &str,
    approach: crate::Approach,
    presimplify: crate::PreSimplifyMode,
) -> Result<LimitEvalResult, LimitEvalError> {
    let mut ctx = cas_ast::Context::new();
    let parsed = cas_parser::parse(expr, &mut ctx)
        .map_err(|e| LimitEvalError::Parse(format!("Parse error: {}", e)))?;

    let var_id = ctx.var(var);
    let mut budget = crate::Budget::new();
    let opts = crate::LimitOptions {
        presimplify,
        ..Default::default()
    };

    match crate::limit(&mut ctx, parsed, var_id, approach, &opts, &mut budget) {
        Ok(limit_result) => {
            let result = DisplayExpr {
                context: &ctx,
                id: limit_result.expr,
            }
            .to_string();
            Ok(LimitEvalResult {
                result,
                warning: limit_result.warning,
            })
        }
        Err(e) => Err(LimitEvalError::Limit(e.to_string())),
    }
}

fn limit_str_to_json(
    expr: &str,
    var: &str,
    approach: crate::Approach,
    presimplify: crate::PreSimplifyMode,
    pretty: bool,
) -> String {
    let response = match eval_limit_from_str(expr, var, approach, presimplify) {
        Ok(limit_result) => LimitJsonResponse::ok(limit_result.result, limit_result.warning),
        Err(LimitEvalError::Parse(message)) => LimitJsonResponse::parse_error(message),
        Err(LimitEvalError::Limit(message)) => LimitJsonResponse::limit_error(message),
    };

    response.to_json_with_pretty(pretty)
}

pub fn evaluate_limit_command_input(
    input: &str,
) -> Result<LimitCommandEvalOutput, LimitCommandEvalError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(LimitCommandEvalError::EmptyInput);
    }

    let parsed = parse_limit_command_input(trimmed);
    match eval_limit_from_str(parsed.expr, parsed.var, parsed.approach, parsed.presimplify) {
        Ok(limit_result) => Ok(LimitCommandEvalOutput {
            var: parsed.var.to_string(),
            approach: parsed.approach,
            result: limit_result.result,
            warning: limit_result.warning,
        }),
        Err(LimitEvalError::Parse(message)) => Err(LimitCommandEvalError::Parse(message)),
        Err(LimitEvalError::Limit(message)) => Err(LimitCommandEvalError::Limit(message)),
    }
}

pub fn evaluate_limit_subcommand_output(
    expr: &str,
    var: &str,
    approach: crate::Approach,
    presimplify: crate::PreSimplifyMode,
    json_output: bool,
) -> Result<LimitSubcommandEvalOutput, LimitSubcommandEvalError> {
    if json_output {
        return Ok(LimitSubcommandEvalOutput::Json(limit_str_to_json(
            expr,
            var,
            approach,
            presimplify,
            false,
        )));
    }

    match eval_limit_from_str(expr, var, approach, presimplify) {
        Ok(limit_result) => Ok(LimitSubcommandEvalOutput::Text {
            result: limit_result.result,
            warning: limit_result.warning,
        }),
        Err(LimitEvalError::Parse(message)) => Err(LimitSubcommandEvalError::Parse(message)),
        Err(LimitEvalError::Limit(message)) => Err(LimitSubcommandEvalError::Limit(message)),
    }
}

pub fn format_limit_subcommand_error(error: &LimitSubcommandEvalError) -> String {
    match error {
        LimitSubcommandEvalError::Parse(message) => message.clone(),
        LimitSubcommandEvalError::Limit(message) => format!("Error: {message}"),
    }
}
