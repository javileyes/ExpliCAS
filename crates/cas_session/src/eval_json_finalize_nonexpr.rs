use cas_api_models::{
    EvalJsonOutput, EvalJsonOutputBuild, RequiredConditionJson, SolveStepJson, StepJson,
    TimingsJson, WarningJson,
};
use cas_ast::{Context, SolutionSet};

use crate::eval_json_finalize_wire::build_eval_wire_value;
use crate::eval_json_presentation::{
    format_solution_set_eval_json, solution_set_to_latex_eval_json,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn finalize_solution_set_output(
    ctx: &Context,
    solution_set: &SolutionSet,
    input: &str,
    input_latex: Option<String>,
    steps_mode: &str,
    steps: Vec<StepJson>,
    solve_steps: Vec<SolveStepJson>,
    warnings: Vec<WarningJson>,
    required_conditions: Vec<RequiredConditionJson>,
    required_display: Vec<String>,
    steps_count: usize,
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
) -> EvalJsonOutput {
    let result_str = format_solution_set_eval_json(ctx, solution_set);
    let result_latex = solution_set_to_latex_eval_json(ctx, solution_set);
    let wire = build_eval_wire_value(
        &warnings,
        &required_display,
        &result_str,
        Some(&result_latex),
        steps_count,
        steps_mode,
    );

    EvalJsonOutput::from_build(EvalJsonOutputBuild {
        input,
        input_latex,
        result_chars: result_str.len(),
        result: result_str,
        result_truncated: false,
        result_latex: Some(result_latex),
        steps_mode,
        steps_count,
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
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finalize_bool_output(
    value: bool,
    input: &str,
    input_latex: Option<String>,
    steps_mode: &str,
    steps: Vec<StepJson>,
    solve_steps: Vec<SolveStepJson>,
    warnings: Vec<WarningJson>,
    required_conditions: Vec<RequiredConditionJson>,
    required_display: Vec<String>,
    steps_count: usize,
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
) -> EvalJsonOutput {
    let result_str = value.to_string();
    let wire = build_eval_wire_value(
        &warnings,
        &required_display,
        &result_str,
        None,
        steps_count,
        steps_mode,
    );

    EvalJsonOutput::from_build(EvalJsonOutputBuild {
        input,
        input_latex,
        result_chars: result_str.len(),
        result: result_str,
        result_truncated: false,
        result_latex: None,
        steps_mode,
        steps_count,
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
    })
}
