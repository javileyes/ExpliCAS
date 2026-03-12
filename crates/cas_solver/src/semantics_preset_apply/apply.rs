use crate::semantics_preset_catalog::find_semantics_preset;
use crate::semantics_preset_format::{
    format_semantics_preset_application_lines, format_semantics_preset_help_lines,
    format_semantics_preset_list_lines,
};
use cas_solver_core::semantics_preset_types::{
    SemanticsPresetApplication, SemanticsPresetApplyError, SemanticsPresetCommandOutput,
};

use super::state::{
    apply_semantics_preset_state_to_options, semantics_preset_state_from_options, state_from_preset,
};

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
    simplify_options: &mut crate::SimplifyOptions,
    eval_options: &mut crate::EvalOptions,
) -> Result<SemanticsPresetApplication, SemanticsPresetApplyError> {
    let application = apply_semantics_preset_by_name(name)?;
    apply_semantics_preset_state_to_options(application.next, simplify_options, eval_options);
    Ok(application)
}

/// Evaluate `semantics preset ...` args, mutating options on successful apply.
pub fn evaluate_semantics_preset_args_to_options(
    args: &[&str],
    simplify_options: &mut crate::SimplifyOptions,
    eval_options: &mut crate::EvalOptions,
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
