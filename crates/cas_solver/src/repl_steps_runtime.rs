use crate::{StepsCommandApplyEffects, StepsCommandState, StepsDisplayMode, StepsMode};
use cas_api_models::EvalStepsMode;

/// Runtime context needed by `steps` command orchestration.
pub trait ReplStepsRuntimeContext {
    fn steps_mode_current(&self) -> EvalStepsMode;
    fn apply_steps_effects_to_eval_options(
        &mut self,
        set_steps_mode: Option<EvalStepsMode>,
        set_display_mode: Option<StepsDisplayMode>,
    ) -> StepsCommandApplyEffects;
    fn set_simplifier_steps_mode(&mut self, mode: StepsMode);
}

/// Build `steps` command state from runtime context + display mode.
pub fn steps_command_state_for_runtime<C: ReplStepsRuntimeContext>(
    context: &C,
    display_mode: StepsDisplayMode,
) -> StepsCommandState {
    StepsCommandState {
        steps_mode: context.steps_mode_current(),
        display_mode,
    }
}

/// Apply `steps` command updates to runtime context.
pub fn apply_steps_command_update_on_runtime<C: ReplStepsRuntimeContext>(
    context: &mut C,
    set_steps_mode: Option<EvalStepsMode>,
    set_display_mode: Option<StepsDisplayMode>,
) -> StepsCommandApplyEffects {
    let effects = context.apply_steps_effects_to_eval_options(set_steps_mode, set_display_mode);
    if let Some(mode) = effects.set_steps_mode {
        context.set_simplifier_steps_mode(steps_mode_from_eval(mode));
    }
    effects
}

fn steps_mode_from_eval(mode: EvalStepsMode) -> StepsMode {
    match mode {
        EvalStepsMode::On => StepsMode::On,
        EvalStepsMode::Off => StepsMode::Off,
        EvalStepsMode::Compact => StepsMode::Compact,
    }
}
