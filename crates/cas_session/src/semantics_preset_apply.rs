use crate::semantics_preset_catalog::find_semantics_preset;
use crate::semantics_preset_format::{
    format_semantics_preset_application_lines, format_semantics_preset_help_lines,
    format_semantics_preset_list_lines,
};
use crate::semantics_preset_types::{
    SemanticsPreset, SemanticsPresetApplication, SemanticsPresetApplyError,
    SemanticsPresetCommandOutput, SemanticsPresetState,
};

/// Build a preset-state snapshot from simplifier + eval options.
pub fn semantics_preset_state_from_options(
    simplify_options: &cas_solver::SimplifyOptions,
    eval_options: &cas_solver::EvalOptions,
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
    simplify_options: &mut cas_solver::SimplifyOptions,
    eval_options: &mut cas_solver::EvalOptions,
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

fn state_from_preset(preset: SemanticsPreset) -> SemanticsPresetState {
    SemanticsPresetState {
        domain: preset.domain,
        value: preset.value,
        branch: preset.branch,
        inv_trig: preset.inv_trig,
        const_fold: preset.const_fold,
    }
}

pub fn apply_semantics_preset_by_name(
    name: &str,
) -> Result<SemanticsPresetApplication, SemanticsPresetApplyError> {
    let Some(preset) = find_semantics_preset(name) else {
        return Err(SemanticsPresetApplyError::UnknownPreset {
            name: name.to_string(),
        });
    };
    Ok(SemanticsPresetApplication {
        preset,
        next: state_from_preset(preset),
    })
}

/// Resolve and apply a semantics preset by name to runtime options.
pub fn apply_semantics_preset_by_name_to_options(
    name: &str,
    simplify_options: &mut cas_solver::SimplifyOptions,
    eval_options: &mut cas_solver::EvalOptions,
) -> Result<SemanticsPresetApplication, SemanticsPresetApplyError> {
    let application = apply_semantics_preset_by_name(name)?;
    apply_semantics_preset_state_to_options(application.next, simplify_options, eval_options);
    Ok(application)
}

/// Evaluate `semantics preset ...` args, mutating options on successful apply.
pub fn evaluate_semantics_preset_args_to_options(
    args: &[&str],
    simplify_options: &mut cas_solver::SimplifyOptions,
    eval_options: &mut cas_solver::EvalOptions,
) -> SemanticsPresetCommandOutput {
    match args.first().copied() {
        None => SemanticsPresetCommandOutput {
            lines: format_semantics_preset_list_lines(),
            applied: false,
        },
        Some("help") => SemanticsPresetCommandOutput {
            lines: format_semantics_preset_help_lines(args.get(1).copied()),
            applied: false,
        },
        Some(name) => {
            let current = semantics_preset_state_from_options(simplify_options, eval_options);
            match apply_semantics_preset_by_name_to_options(name, simplify_options, eval_options) {
                Ok(application) => SemanticsPresetCommandOutput {
                    lines: format_semantics_preset_application_lines(current, &application),
                    applied: true,
                },
                Err(SemanticsPresetApplyError::UnknownPreset { .. }) => {
                    SemanticsPresetCommandOutput {
                        lines: format_semantics_preset_help_lines(Some(name)),
                        applied: false,
                    }
                }
            }
        }
    }
}
