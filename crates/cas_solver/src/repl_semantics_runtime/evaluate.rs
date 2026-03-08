use super::{
    apply_autoexpand_command_on_runtime, apply_context_command_on_runtime,
    apply_semantics_command_on_runtime, ReplSemanticsRuntimeContext,
};

/// Evaluate `context` command on runtime and optionally run a sync hook after rebuild.
pub fn evaluate_context_command_on_runtime<C: ReplSemanticsRuntimeContext>(
    line: &str,
    context: &mut C,
    on_rebuilt: impl FnOnce(&mut C),
) -> String {
    let applied = apply_context_command_on_runtime(line, context);
    if applied.rebuilt_simplifier {
        on_rebuilt(context);
    }
    applied.message
}

/// Evaluate `autoexpand` command on runtime and optionally run a sync hook after rebuild.
pub fn evaluate_autoexpand_command_on_runtime<C: ReplSemanticsRuntimeContext>(
    line: &str,
    context: &mut C,
    on_rebuilt: impl FnOnce(&mut C),
) -> String {
    let applied = apply_autoexpand_command_on_runtime(line, context);
    if applied.rebuilt_simplifier {
        on_rebuilt(context);
    }
    applied.message
}

/// Evaluate `semantics` command on runtime and optionally run a sync hook after rebuild.
pub fn evaluate_semantics_command_on_runtime<C: ReplSemanticsRuntimeContext>(
    line: &str,
    context: &mut C,
    on_rebuilt: impl FnOnce(&mut C),
) -> String {
    let out = apply_semantics_command_on_runtime(line, context);
    if out.sync_simplifier {
        on_rebuilt(context);
    }
    out.lines.join("\n")
}
