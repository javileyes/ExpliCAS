//! Final assembly for session-backed `eval-json` outputs.

use cas_api_models::{
    wire::build_eval_wire_reply, EvalJsonOutput, EvalJsonOutputBuild, RequiredConditionJson,
    SolveStepJson, StepJson, TimingsJson, WarningJson,
};
use cas_ast::{Context, ExprId};
use cas_formatter::LaTeXExpr;

use crate::eval_json_presentation::{
    format_solution_set_eval_json, solution_set_to_latex_eval_json,
};

pub(crate) struct EvalJsonFinalizeInput<'a> {
    pub(crate) result: &'a cas_solver::EvalResult,
    pub(crate) ctx: &'a Context,
    pub(crate) max_chars: usize,
    pub(crate) input: &'a str,
    pub(crate) input_latex: Option<String>,
    pub(crate) steps_mode: &'a str,
    pub(crate) steps: Vec<StepJson>,
    pub(crate) solve_steps: Vec<SolveStepJson>,
    pub(crate) warnings: Vec<WarningJson>,
    pub(crate) required_conditions: Vec<RequiredConditionJson>,
    pub(crate) required_display: Vec<String>,
    pub(crate) raw_steps_count: usize,
    pub(crate) raw_solve_steps_count: usize,
    pub(crate) budget_preset: &'a str,
    pub(crate) strict: bool,
    pub(crate) domain: &'a str,
    pub(crate) timings_us: TimingsJson,
    pub(crate) context_mode: &'a str,
    pub(crate) branch_mode: &'a str,
    pub(crate) expand_policy: &'a str,
    pub(crate) complex_mode: &'a str,
    pub(crate) const_fold: &'a str,
    pub(crate) value_domain: &'a str,
    pub(crate) complex_branch: &'a str,
    pub(crate) inv_trig: &'a str,
    pub(crate) assume_scope: &'a str,
}

fn build_eval_wire_value(
    warnings: &[WarningJson],
    required_display: &[String],
    result: &str,
    result_latex: Option<&str>,
    steps_count: usize,
    steps_mode: &str,
) -> Option<serde_json::Value> {
    serde_json::to_value(build_eval_wire_reply(
        warnings,
        required_display,
        result,
        result_latex,
        steps_count,
        steps_mode,
    ))
    .ok()
}

pub(crate) fn finalize_eval_json_output(
    input: EvalJsonFinalizeInput<'_>,
) -> Result<EvalJsonOutput, String> {
    let EvalJsonFinalizeInput {
        result,
        ctx,
        max_chars,
        input,
        input_latex,
        steps_mode,
        steps,
        solve_steps,
        warnings,
        required_conditions,
        required_display,
        raw_steps_count,
        raw_solve_steps_count,
        budget_preset,
        strict,
        domain,
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
    } = input;

    match result {
        cas_solver::EvalResult::SolutionSet(solution_set) => {
            let result_str = format_solution_set_eval_json(ctx, solution_set);
            let result_latex = solution_set_to_latex_eval_json(ctx, solution_set);
            let wire = build_eval_wire_value(
                &warnings,
                &required_display,
                &result_str,
                Some(&result_latex),
                raw_steps_count + raw_solve_steps_count,
                steps_mode,
            );

            Ok(EvalJsonOutput::from_build(EvalJsonOutputBuild {
                input,
                input_latex,
                result_chars: result_str.len(),
                result: result_str,
                result_truncated: false,
                result_latex: Some(result_latex),
                steps_mode,
                steps_count: raw_steps_count + raw_solve_steps_count,
                steps,
                solve_steps,
                warnings,
                required_conditions,
                required_display,
                budget_preset,
                strict,
                domain,
                stats: Default::default(),
                hash: None,
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
        cas_solver::EvalResult::Bool(b) => {
            let result_str = b.to_string();
            let wire = build_eval_wire_value(
                &warnings,
                &required_display,
                &result_str,
                None,
                raw_steps_count,
                steps_mode,
            );

            Ok(EvalJsonOutput::from_build(EvalJsonOutputBuild {
                input,
                input_latex,
                result_chars: result_str.len(),
                result: result_str,
                result_truncated: false,
                result_latex: None,
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
                stats: Default::default(),
                hash: None,
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
        cas_solver::EvalResult::Expr(e) => finalize_expr_like_eval_json_output(
            ctx,
            *e,
            max_chars,
            input,
            input_latex,
            steps_mode,
            steps,
            solve_steps,
            warnings,
            required_conditions,
            required_display,
            raw_steps_count,
            budget_preset,
            strict,
            domain,
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
        ),
        cas_solver::EvalResult::Set(v) if !v.is_empty() => finalize_expr_like_eval_json_output(
            ctx,
            v[0],
            max_chars,
            input,
            input_latex,
            steps_mode,
            steps,
            solve_steps,
            warnings,
            required_conditions,
            required_display,
            raw_steps_count,
            budget_preset,
            strict,
            domain,
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
        ),
        _ => Err("No result expression".to_string()),
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize_expr_like_eval_json_output(
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
