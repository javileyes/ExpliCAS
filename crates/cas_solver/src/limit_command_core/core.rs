use cas_api_models::{LimitEvalError, LimitEvalResult, LimitWireResponse};
use cas_formatter::DisplayExpr;
use cas_math::limit_types::{Approach, LimitOptions, PreSimplifyMode};

pub(super) fn eval_limit_from_str(
    expr: &str,
    var: &str,
    approach: Approach,
    presimplify: PreSimplifyMode,
) -> Result<LimitEvalResult, LimitEvalError> {
    let mut ctx = cas_ast::Context::new();
    let parsed = cas_parser::parse(expr, &mut ctx)
        .map_err(|e| LimitEvalError::Parse(format!("Parse error: {}", e)))?;

    let var_id = ctx.var(var);
    let mut budget = crate::Budget::new();
    let opts = LimitOptions {
        presimplify,
        ..Default::default()
    };

    match crate::solver_entrypoints_eval::limit(
        &mut ctx,
        parsed,
        var_id,
        approach,
        &opts,
        &mut budget,
    ) {
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

pub(super) fn limit_str_to_wire(
    expr: &str,
    var: &str,
    approach: Approach,
    presimplify: PreSimplifyMode,
    pretty: bool,
) -> String {
    let response = match eval_limit_from_str(expr, var, approach, presimplify) {
        Ok(limit_result) => LimitWireResponse::ok(limit_result.result, limit_result.warning),
        Err(LimitEvalError::Parse(message)) => LimitWireResponse::parse_error(message),
        Err(LimitEvalError::Limit(message)) => LimitWireResponse::limit_error(message),
    };

    response.to_json_with_pretty(pretty)
}
