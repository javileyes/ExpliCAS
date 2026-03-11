use cas_api_models::{RequiredConditionWire, SolveStepWire, StepWire, TimingsWire, WarningWire};
use cas_solver_core::engine_events::EngineEvent;

use crate::eval_output_presentation::{
    collect_output_required_conditions, collect_output_required_display,
    collect_output_solve_steps, collect_output_warnings, format_output_input_latex,
};

use super::PreparedEvalRun;

pub(super) struct CollectedEvalArtifacts {
    pub(super) input_latex: Option<String>,
    pub(super) steps: Vec<StepWire>,
    pub(super) solve_steps: Vec<SolveStepWire>,
    pub(super) warnings: Vec<WarningWire>,
    pub(super) required_conditions: Vec<RequiredConditionWire>,
    pub(super) required_display: Vec<String>,
    pub(super) raw_steps_count: usize,
    pub(super) raw_solve_steps_count: usize,
    pub(super) timings_us: TimingsWire,
}

pub(super) fn collect_eval_artifacts<F>(
    ctx: &cas_ast::Context,
    steps_mode: &str,
    prepared: &PreparedEvalRun,
    total_us: u64,
    collect_steps: F,
) -> CollectedEvalArtifacts
where
    F: Fn(&[crate::Step], &[EngineEvent], &cas_ast::Context, &str) -> Vec<StepWire>,
{
    let input_latex = Some(format_output_input_latex(ctx, prepared.parsed_input));
    let steps_raw = prepared.output_view.steps.as_slice();
    let solve_steps_raw = prepared.output_view.solve_steps.as_slice();
    let steps = collect_steps(steps_raw, prepared.events.as_slice(), ctx, steps_mode);
    let solve_steps = collect_output_solve_steps(solve_steps_raw, ctx, steps_mode);
    let warnings = collect_output_warnings(&prepared.output_view.domain_warnings);
    let required_conditions_raw = prepared.output_view.required_conditions.as_slice();
    let required_conditions = collect_output_required_conditions(required_conditions_raw, ctx);
    let required_display = collect_output_required_display(required_conditions_raw, ctx);
    let timings_us = TimingsWire {
        parse_us: prepared.parse_us,
        simplify_us: prepared.simplify_us,
        total_us,
    };

    CollectedEvalArtifacts {
        input_latex,
        steps,
        solve_steps,
        warnings,
        required_conditions,
        required_display,
        raw_steps_count: steps_raw.len(),
        raw_solve_steps_count: solve_steps_raw.len(),
        timings_us,
    }
}
