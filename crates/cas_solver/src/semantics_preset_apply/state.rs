use crate::{SemanticsPreset, SemanticsPresetState};

/// Build a preset-state snapshot from simplifier + eval options.
pub fn semantics_preset_state_from_options(
    simplify_options: &crate::SimplifyOptions,
    eval_options: &crate::EvalOptions,
) -> SemanticsPresetState {
    SemanticsPresetState {
        domain: simplify_options.shared.semantics.domain_mode,
        value: simplify_options.shared.semantics.value_domain,
        branch: simplify_options.shared.semantics.branch,
        inv_trig: simplify_options.shared.semantics.inv_trig,
        const_fold: eval_options.const_fold,
    }
}

/// Apply preset state to both simplifier options and runtime eval options.
pub fn apply_semantics_preset_state_to_options(
    next: SemanticsPresetState,
    simplify_options: &mut crate::SimplifyOptions,
    eval_options: &mut crate::EvalOptions,
) {
    simplify_options.shared.semantics.domain_mode = next.domain;
    simplify_options.shared.semantics.value_domain = next.value;
    simplify_options.shared.semantics.branch = next.branch;
    simplify_options.shared.semantics.inv_trig = next.inv_trig;

    eval_options.shared.semantics.domain_mode = next.domain;
    eval_options.shared.semantics.value_domain = next.value;
    eval_options.shared.semantics.branch = next.branch;
    eval_options.shared.semantics.inv_trig = next.inv_trig;

    eval_options.const_fold = next.const_fold;
}

pub(super) fn state_from_preset(preset: SemanticsPreset) -> SemanticsPresetState {
    SemanticsPresetState {
        domain: preset.domain,
        value: preset.value,
        branch: preset.branch,
        inv_trig: preset.inv_trig,
        const_fold: preset.const_fold,
    }
}
