use crate::SetDisplayMode;

use super::ReplSimplifierRuntimeContext;

/// Evaluate unary command invocation (`det`, `transpose`, `trace`) against runtime simplifier.
pub fn evaluate_unary_command_message_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
    function_name: &str,
    display_mode: SetDisplayMode,
    show_parsed: bool,
    clean_result: bool,
) -> Result<String, String> {
    crate::evaluate_unary_command_message(
        context.simplifier_mut(),
        line,
        function_name,
        display_mode,
        show_parsed,
        clean_result,
    )
}

/// Evaluate `det ...` invocation against runtime simplifier.
pub fn evaluate_det_command_message_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<String, String> {
    evaluate_unary_command_message_on_runtime(context, line, "det", display_mode, true, true)
}

/// Evaluate `transpose ...` invocation against runtime simplifier.
pub fn evaluate_transpose_command_message_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<String, String> {
    evaluate_unary_command_message_on_runtime(
        context,
        line,
        "transpose",
        display_mode,
        false,
        false,
    )
}

/// Evaluate `trace ...` invocation against runtime simplifier.
pub fn evaluate_trace_command_message_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<String, String> {
    evaluate_unary_command_message_on_runtime(context, line, "trace", display_mode, false, true)
}
