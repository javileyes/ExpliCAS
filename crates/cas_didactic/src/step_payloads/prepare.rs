mod enrich;
mod filter;
mod mode;

use cas_ast::Context;
use cas_solver::Step;

pub(super) fn prepare_step_payloads(
    steps: &[Step],
    context: &Context,
    steps_mode: &str,
) -> Vec<crate::didactic::EnrichedStep> {
    if !mode::step_payloads_enabled(steps_mode) {
        return Vec::new();
    }

    let filtered =
        filter::filter_step_payloads(steps, crate::didactic::clone_steps_matching_visibility);
    if filtered.is_empty() {
        return Vec::new();
    }

    let Some(original_expr) = filter::infer_original_expr_for_filtered_steps(
        &filtered,
        crate::didactic::infer_original_expr_for_steps,
    ) else {
        return Vec::new();
    };

    enrich::enrich_step_payloads(
        context,
        original_expr,
        filtered,
        crate::didactic::enrich_steps,
    )
}
