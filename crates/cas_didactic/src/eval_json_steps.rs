mod build;
mod events;
mod prepare;

use cas_api_models::StepJson;
use cas_ast::Context;
use cas_solver::Step;
use cas_solver_core::engine_events::EngineEvent;

/// Convert engine steps to eval-json step payloads.
///
/// Keeps JSON step formatting behavior consistent with timeline rendering.
pub fn collect_eval_json_steps(steps: &[Step], ctx: &Context, steps_mode: &str) -> Vec<StepJson> {
    prepare::prepare_eval_json_steps(steps, ctx, steps_mode)
        .iter()
        .enumerate()
        .map(|(index, enriched)| build::build_step_json(ctx, index + 1, enriched))
        .collect()
}

/// Convert steps to eval-json payloads, falling back to engine events when
/// steps are not available but event capture is enabled.
pub fn collect_eval_json_steps_with_events(
    steps: &[Step],
    events: &[EngineEvent],
    ctx: &Context,
    steps_mode: &str,
) -> Vec<StepJson> {
    let collected = collect_eval_json_steps(steps, ctx, steps_mode);
    if !collected.is_empty() || steps_mode != "on" {
        return collected;
    }
    events::collect_event_json_steps(events, ctx)
}
