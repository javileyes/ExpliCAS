use crate::limit_command_parse_types::LimitCommandApproachSpec;
use cas_api_models::{LimitEvalError, LimitEvalResult, LimitWireResponse};
use cas_formatter::DisplayExpr;
use cas_math::limit_types::{Approach, LimitOptions, PreSimplifyMode};

/// Spec-level entry (F7, Fase 3): a finite approach point travels as source
/// text and parses HERE, in the same fresh context as the expression — the
/// unification that lets the REPL `limit` command take finite points.
pub(super) fn eval_limit_from_str_spec(
    expr: &str,
    var: &str,
    approach: LimitCommandApproachSpec<'_>,
    presimplify: PreSimplifyMode,
    complex_enabled: bool,
) -> Result<LimitEvalResult, LimitEvalError> {
    let mut ctx = cas_ast::Context::new();
    let parsed = cas_parser::parse(expr, &mut ctx)
        .map_err(|e| LimitEvalError::Parse(format!("Parse error: {}", e)))?;
    let approach = match approach {
        LimitCommandApproachSpec::PosInfinity => Approach::PosInfinity,
        LimitCommandApproachSpec::NegInfinity => Approach::NegInfinity,
        LimitCommandApproachSpec::Finite(point_src) => {
            let point = cas_parser::parse(point_src, &mut ctx)
                .map_err(|e| LimitEvalError::Parse(format!("Parse error: {}", e)))?;
            Approach::Finite(point)
        }
    };
    eval_limit_in_ctx(ctx, parsed, var, approach, presimplify, complex_enabled)
}

pub(super) fn eval_limit_from_str(
    expr: &str,
    var: &str,
    approach: Approach,
    presimplify: PreSimplifyMode,
    complex_enabled: bool,
) -> Result<LimitEvalResult, LimitEvalError> {
    let mut ctx = cas_ast::Context::new();
    let parsed = cas_parser::parse(expr, &mut ctx)
        .map_err(|e| LimitEvalError::Parse(format!("Parse error: {}", e)))?;
    eval_limit_in_ctx(ctx, parsed, var, approach, presimplify, complex_enabled)
}

fn eval_limit_in_ctx(
    mut ctx: cas_ast::Context,
    parsed: cas_ast::ExprId,
    var: &str,
    approach: Approach,
    presimplify: PreSimplifyMode,
    complex_enabled: bool,
) -> Result<LimitEvalResult, LimitEvalError> {
    let var_id = ctx.var(var);
    let mut budget = crate::Budget::new();
    let opts = LimitOptions {
        presimplify,
        complex_enabled,
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
    // The web wire has no value-domain axis yet: real view (named residual, F11).
    let response = match eval_limit_from_str(expr, var, approach, presimplify, false) {
        Ok(limit_result) => LimitWireResponse::ok(limit_result.result, limit_result.warning),
        Err(LimitEvalError::Parse(message)) => LimitWireResponse::parse_error(message),
        Err(LimitEvalError::Limit(message)) => LimitWireResponse::limit_error(message),
    };

    response.to_json_with_pretty(pretty)
}
