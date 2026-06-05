use crate::runtime::Step;
use cas_ast::ExprId;

pub(super) fn filter_step_payloads(
    steps: &[Step],
    clone_steps_matching_visibility: fn(&[Step], crate::didactic::StepVisibility) -> Vec<Step>,
) -> Vec<Step> {
    let filtered =
        clone_steps_matching_visibility(steps, crate::didactic::StepVisibility::MediumOrHigher);
    if filtered.len() == 1 && steps.len() > 1 && is_terminal_calculus_residual_step(&filtered[0]) {
        return clone_steps_matching_visibility(steps, crate::didactic::StepVisibility::All);
    }
    if !filtered.is_empty() || steps.is_empty() {
        return filtered;
    }

    // When the didactic visibility gate removes every engine step, prefer
    // falling back to the raw trace instead of serializing `steps_count > 0`
    // with an empty `steps` array.
    clone_steps_matching_visibility(steps, crate::didactic::StepVisibility::All)
}

fn is_terminal_calculus_residual_step(step: &Step) -> bool {
    matches!(
        step.rule_name.as_str(),
        "Conservar derivada residual" | "Conservar integral residual" | "Conservar límite residual"
    )
}

pub(super) fn infer_original_expr_for_filtered_steps(
    filtered: &[Step],
    infer_original_expr_for_steps: fn(&[Step]) -> Option<ExprId>,
) -> Option<ExprId> {
    infer_original_expr_for_steps(filtered)
}
