//! Unary-command REPL adapters extracted from `repl_command_runtime`.

use crate::{ReplCore, SetDisplayMode};

/// Evaluate unary command invocation (`det`, `transpose`, `trace`) against REPL simplifier.
pub(crate) fn evaluate_unary_command_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
    function_name: &str,
    display_mode: SetDisplayMode,
    show_parsed: bool,
    clean_result: bool,
) -> Result<String, String> {
    crate::evaluate_unary_command_message(
        core.simplifier_mut(),
        line,
        function_name,
        display_mode,
        show_parsed,
        clean_result,
    )
}

/// Evaluate `det ...` invocation against REPL simplifier.
pub fn evaluate_det_command_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<String, String> {
    evaluate_unary_command_message_on_repl_core(core, line, "det", display_mode, true, true)
}

/// Evaluate `transpose ...` invocation against REPL simplifier.
pub fn evaluate_transpose_command_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<String, String> {
    evaluate_unary_command_message_on_repl_core(core, line, "transpose", display_mode, false, false)
}

/// Evaluate `trace ...` invocation against REPL simplifier.
pub fn evaluate_trace_command_message_on_repl_core(
    core: &mut ReplCore,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<String, String> {
    evaluate_unary_command_message_on_repl_core(core, line, "trace", display_mode, false, true)
}
