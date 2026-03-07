use crate::steps_command_types::{StepsCommandApplyEffects, StepsDisplayMode};

/// Apply `steps` update fields into eval options and return external effects.
pub fn apply_steps_command_update(
    set_steps_mode: Option<crate::StepsMode>,
    set_display_mode: Option<StepsDisplayMode>,
    eval_options: &mut crate::EvalOptions,
) -> StepsCommandApplyEffects {
    let mut changed_steps_mode = None;
    if let Some(mode) = set_steps_mode {
        if eval_options.steps_mode != mode {
            eval_options.steps_mode = mode;
            changed_steps_mode = Some(mode);
        }
    }

    StepsCommandApplyEffects {
        set_steps_mode: changed_steps_mode,
        set_display_mode,
    }
}
