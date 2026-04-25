use super::StepVisibility;
use crate::runtime::Step;
use cas_ast::ExprId;

fn is_inverse_tan_preparatory_step(step: &Step) -> bool {
    matches!(
        (step.rule_name.as_str(), step.description.as_str()),
        ("Canonicalize Trig Function Names", "atan -> arctan")
            | ("Canonicalize", "Canonicalization")
    )
}

pub(super) fn should_absorb_preparatory_step_at(
    steps: &[Step],
    index: usize,
    visibility: StepVisibility,
) -> bool {
    let _ = visibility;

    let Some(step) = steps.get(index) else {
        return false;
    };
    if !is_inverse_tan_preparatory_step(step) {
        return false;
    }

    for next_step in &steps[index + 1..] {
        if is_inverse_tan_preparatory_step(next_step) {
            continue;
        }
        return matches!(
            next_step.rule_name.as_str(),
            "Inverse Tan Relations" | "Inverse Trig Composition"
        );
    }

    false
}

pub(super) fn clone_steps_matching_visibility(
    steps: &[Step],
    visibility: StepVisibility,
    step_matches_visibility: fn(&Step, StepVisibility) -> bool,
) -> Vec<Step> {
    steps
        .iter()
        .enumerate()
        .filter(|(index, step)| {
            step_matches_visibility(step, visibility)
                && !should_absorb_preparatory_step_at(steps, *index, visibility)
        })
        .map(|(_, step)| step.clone())
        .collect()
}

pub(super) fn infer_original_expr_for_steps(steps: &[Step]) -> Option<ExprId> {
    steps
        .first()
        .map(|step| step.global_before.unwrap_or(step.before))
}
