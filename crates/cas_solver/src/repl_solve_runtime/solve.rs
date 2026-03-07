use crate::{EvalSession, SetDisplayMode};

use super::{display_mode, ReplSolveRuntimeContext};

pub(super) fn evaluate_solve_command_message_on_runtime<C: ReplSolveRuntimeContext>(
    context: &mut C,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<String, String> {
    let debug_mode = context.debug_mode();
    context.with_state_and_simplifier_mut(|state, simplifier| {
        let options = state.options().clone();
        crate::evaluate_solve_command_message_with_session(
            simplifier,
            state,
            line,
            &options,
            display_mode::map_solve_display_mode(display_mode),
            debug_mode,
        )
    })
}
