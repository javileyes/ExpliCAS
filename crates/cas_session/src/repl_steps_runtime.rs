//! Runtime adapters to apply `steps` command effects on `ReplCore`.

use crate::{ReplCore, StepsCommandApplyEffects, StepsCommandState, StepsDisplayMode, StepsMode};

/// Build `steps` command state from REPL core + current display mode.
pub fn steps_command_state_for_repl_core(
    core: &ReplCore,
    display_mode: StepsDisplayMode,
) -> StepsCommandState {
    StepsCommandState {
        steps_mode: core.state.options().steps_mode,
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
        core.state.options_mut(),
    );
    if let Some(mode) = effects.set_steps_mode {
        core.engine.simplifier.set_steps_mode(mode);
    }
    effects
}

#[cfg(test)]
mod tests {
    use super::{apply_steps_command_update_on_repl_core, steps_command_state_for_repl_core};

    #[test]
    fn steps_command_state_for_repl_core_reads_state() {
        let core = crate::ReplCore::new();
        let state = steps_command_state_for_repl_core(&core, crate::StepsDisplayMode::Normal);
        assert_eq!(state.display_mode, crate::StepsDisplayMode::Normal);
    }

    #[test]
    fn apply_steps_command_update_on_repl_core_updates_engine_and_options() {
        let mut core = crate::ReplCore::new();
        let effects = apply_steps_command_update_on_repl_core(
            &mut core,
            Some(crate::StepsMode::Compact),
            Some(crate::StepsDisplayMode::Succinct),
        );
        assert_eq!(effects.set_steps_mode, Some(crate::StepsMode::Compact));
        assert_eq!(
            effects.set_display_mode,
            Some(crate::StepsDisplayMode::Succinct)
        );
        assert_eq!(core.state.options().steps_mode, crate::StepsMode::Compact);
    }
}
