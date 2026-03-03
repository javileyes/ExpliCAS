//! Runtime adapters to apply `steps` command effects on `ReplCore`.

use crate::{ReplCore, StepsCommandApplyEffects, StepsCommandState, StepsDisplayMode, StepsMode};

/// Build `steps` command state from REPL core + current display mode.
pub fn steps_command_state_for_repl_core(
    core: &ReplCore,
    display_mode: StepsDisplayMode,
) -> StepsCommandState {
    StepsCommandState {
        steps_mode: core.eval_options().steps_mode,
        display_mode,
    }
}

/// Apply `steps` command updates to REPL runtime state.
///
/// This synchronizes both eval options and simplifier step mode.
pub fn apply_steps_command_update_on_repl_core(
    core: &mut ReplCore,
    set_steps_mode: Option<StepsMode>,
    set_display_mode: Option<StepsDisplayMode>,
) -> StepsCommandApplyEffects {
    let effects = crate::apply_steps_command_update(
        set_steps_mode,
        set_display_mode,
        core.eval_options_mut(),
    );
    if let Some(mode) = effects.set_steps_mode {
        core.simplifier_mut().set_steps_mode(mode);
    }
    effects
}
