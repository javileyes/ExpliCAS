use super::ReplSimplifierRuntimeContext;

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
