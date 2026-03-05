use cas_api_models::{
    EvalJsonOutput, EvalJsonOutputBuild, RequiredConditionJson, SolveStepJson, StepJson,
    TimingsJson, WarningJson,
};
use cas_ast::{Context, ExprId};
use cas_formatter::LaTeXExpr;

use crate::eval_json_finalize_wire::build_eval_wire_value;

#[allow(clippy::too_many_arguments)]
pub(crate) fn finalize_expr_like_eval_json_output(
    ctx: &Context,
    result_expr: ExprId,
    max_chars: usize,
    input: &str,
    input_latex: Option<String>,
    steps_mode: &str,
    steps: Vec<StepJson>,
    solve_steps: Vec<SolveStepJson>,
    warnings: Vec<WarningJson>,
    required_conditions: Vec<RequiredConditionJson>,
    required_display: Vec<String>,
    raw_steps_count: usize,
    budget_preset: &str,
    strict: bool,
    domain: &str,
    timings_us: TimingsJson,
    context_mode: &str,
    branch_mode: &str,
    expand_policy: &str,
    complex_mode: &str,
    const_fold: &str,
    value_domain: &str,
    complex_branch: &str,
    inv_trig: &str,
    assume_scope: &str,
) -> Result<EvalJsonOutput, String> {
    let (result_str, truncated, char_count) =
        crate::eval_json_stats::format_expr_limited_eval_json(ctx, result_expr, max_chars);
    let stats = crate::eval_json_stats::expr_stats_eval_json(ctx, result_expr);
    let hash = if truncated {
        Some(crate::eval_json_stats::expr_hash_eval_json(
            ctx,
            result_expr,
        ))
    } else {
        None
    };

    let result_latex = if !truncated {
        Some(
            LaTeXExpr {
                context: ctx,
                id: result_expr,
            }
            .to_latex(),
        )
    } else {
        None
    };

    let wire = build_eval_wire_value(
        &warnings,
        &required_display,
        &result_str,
        result_latex.as_deref(),
        raw_steps_count,
        steps_mode,
    );

    Ok(EvalJsonOutput::from_build(EvalJsonOutputBuild {
        input,
        input_latex,
        result_chars: char_count,
        result: result_str,
        result_truncated: truncated,
        result_latex,
        steps_mode,
        steps_count: raw_steps_count,
        steps,
        solve_steps,
        warnings,
        required_conditions,
        required_display,
        budget_preset,
        strict,
        domain,
        stats,
        hash,
        timings_us,
        context_mode,
        branch_mode,
        expand_policy,
        complex_mode,
        const_fold,
        value_domain,
        complex_branch,
        inv_trig,
        assume_scope,
        wire,
    }))
}
