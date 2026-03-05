use crate::{SetDisplayMode, Simplifier, VisualizeCommandOutput};

/// Runtime context that exposes mutable access to the active simplifier.
pub trait ReplSimplifierRuntimeContext {
    fn simplifier_mut(&mut self) -> &mut Simplifier;
}

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

/// Evaluate `weierstrass ...` invocation against runtime simplifier.
pub fn evaluate_weierstrass_invocation_message_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Result<String, String> {
    crate::evaluate_weierstrass_invocation_message(context.simplifier_mut(), line)
}

/// Evaluate `telescope ...` using runtime context.
pub fn evaluate_telescope_invocation_message_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Result<String, String> {
    crate::evaluate_telescope_invocation_message(&mut context.simplifier_mut().context, line)
}

/// Evaluate `expand_log ...` using runtime context.
pub fn evaluate_expand_log_invocation_message_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Result<String, String> {
    crate::evaluate_expand_log_invocation_message(&mut context.simplifier_mut().context, line)
}

/// Evaluate `solve_system ...` using runtime context.
pub fn evaluate_linear_system_command_message_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
) -> String {
    crate::evaluate_linear_system_command_message(&mut context.simplifier_mut().context, line)
}

/// Evaluate `visualize ...` using runtime context.
pub fn evaluate_visualize_invocation_output_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Result<VisualizeCommandOutput, String> {
    crate::evaluate_visualize_invocation_output(&mut context.simplifier_mut().context, line)
}

/// Evaluate `explain ...` using runtime context.
pub fn evaluate_explain_invocation_message_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Result<String, String> {
    crate::evaluate_explain_invocation_message(&mut context.simplifier_mut().context, line)
}

/// Evaluate `equiv ...` against runtime simplifier.
pub fn evaluate_equiv_invocation_message_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Result<String, String> {
    crate::evaluate_equiv_invocation_message(context.simplifier_mut(), line)
}

/// Evaluate `subst ...` against runtime simplifier.
pub fn evaluate_substitute_invocation_user_message_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
    display_mode: SetDisplayMode,
) -> Result<String, String> {
    crate::evaluate_substitute_invocation_user_message(context.simplifier_mut(), line, display_mode)
}

/// Evaluate `rationalize ...` against runtime simplifier.
pub fn evaluate_rationalize_command_lines_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
) -> Result<Vec<String>, String> {
    crate::evaluate_rationalize_command_lines(context.simplifier_mut(), line)
}

/// Apply `profile` command line to runtime simplifier profiler.
pub fn apply_profile_command_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
) -> String {
    crate::apply_profile_command(context.simplifier_mut(), line)
}

/// Evaluate `profile` command and return user-facing message.
pub fn evaluate_profile_command_message_on_runtime<C: ReplSimplifierRuntimeContext>(
    context: &mut C,
    line: &str,
) -> String {
    apply_profile_command_on_runtime(context, line)
}
