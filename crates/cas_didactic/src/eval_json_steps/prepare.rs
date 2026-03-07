use cas_ast::Context;
use cas_solver::Step;

pub(super) fn prepare_eval_json_steps(
    steps: &[Step],
    context: &Context,
    steps_mode: &str,
) -> Vec<crate::didactic::EnrichedStep> {
    if steps_mode != "on" {
        return Vec::new();
    }

    let filtered = crate::didactic::clone_steps_matching_visibility(
        steps,
        crate::didactic::StepVisibility::MediumOrHigher,
    );
    if filtered.is_empty() {
        return Vec::new();
    }

    let Some(original_expr) = crate::didactic::infer_original_expr_for_steps(&filtered) else {
        return Vec::new();
    };

    crate::didactic::enrich_steps(context, original_expr, filtered)
}
