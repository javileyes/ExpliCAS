use cas_api_models::{EvalStepsMode, StepsCommandApplyEffects, StepsDisplayMode};

/// Apply `steps` update fields into eval options and return external effects.
pub fn apply_steps_command_update(
    set_steps_mode: Option<EvalStepsMode>,
    set_display_mode: Option<StepsDisplayMode>,
    eval_options: &mut crate::EvalOptions,
) -> StepsCommandApplyEffects {
    let mut changed_steps_mode = None;
    if let Some(mode) = set_steps_mode {
        let runtime_mode = steps_mode_from_eval(mode);
        if eval_options.steps_mode != runtime_mode {
            eval_options.steps_mode = runtime_mode;
            changed_steps_mode = Some(mode);
        }
    }

    StepsCommandApplyEffects {
        set_steps_mode: changed_steps_mode,
        set_display_mode,
    }
}

fn steps_mode_from_eval(mode: EvalStepsMode) -> crate::StepsMode {
    match mode {
        EvalStepsMode::On => crate::StepsMode::On,
        EvalStepsMode::Off => crate::StepsMode::Off,
        EvalStepsMode::Compact => crate::StepsMode::Compact,
    }
}
