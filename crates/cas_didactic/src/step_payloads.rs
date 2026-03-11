mod build;
mod events;
mod prepare;

use crate::runtime::Step;
use cas_api_models::StepWire;
use cas_ast::Context;
use cas_solver_core::engine_events::EngineEvent;

/// Convert engine steps to typed step payload DTOs.
///
/// Keeps step formatting behavior consistent with timeline rendering.
pub fn collect_step_payloads(steps: &[Step], ctx: &Context, steps_mode: &str) -> Vec<StepWire> {
    prepare::prepare_step_payloads(steps, ctx, steps_mode)
        .iter()
        .enumerate()
        .map(|(index, enriched)| build::build_step_wire(ctx, index + 1, enriched))
        .collect()
}

/// Convert steps to typed step payload DTOs, falling back to engine events when
/// steps are not available but event capture is enabled.
pub fn collect_step_payloads_with_events(
    steps: &[Step],
    events: &[EngineEvent],
    ctx: &Context,
    steps_mode: &str,
) -> Vec<StepWire> {
    let collected = collect_step_payloads(steps, ctx, steps_mode);
    if !collected.is_empty() || steps_mode != "on" {
        return collected;
    }
    events::collect_event_step_payloads(events, ctx)
}
