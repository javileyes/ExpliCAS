use crate::SemanticsCommandOutput;

use super::{ReplSemanticsApplyOutput, ReplSemanticsRuntimeContext};

/// Apply `context` command to runtime state.
pub fn apply_context_command_on_runtime<C: ReplSemanticsRuntimeContext>(
    line: &str,
    context: &mut C,
) -> ReplSemanticsApplyOutput {
    let applied = crate::context_command_eval::evaluate_and_apply_context_command(
        line,
        context.eval_options_mut(),
    );
    if applied.rebuild_simplifier {
        context.rebuild_simplifier_from_profile();
    }
    ReplSemanticsApplyOutput {
        message: applied.message,
        rebuilt_simplifier: applied.rebuild_simplifier,
    }
}

/// Apply `autoexpand` command to runtime state.
pub fn apply_autoexpand_command_on_runtime<C: ReplSemanticsRuntimeContext>(
    line: &str,
    context: &mut C,
) -> ReplSemanticsApplyOutput {
    let applied = crate::autoexpand_command_eval::evaluate_and_apply_autoexpand_command(
        line,
        context.eval_options_mut(),
    );
    if applied.rebuild_simplifier {
        context.rebuild_simplifier_from_profile();
    }
    ReplSemanticsApplyOutput {
        message: applied.message,
        rebuilt_simplifier: applied.rebuild_simplifier,
    }
}

/// Apply `semantics` command to runtime state.
pub fn apply_semantics_command_on_runtime<C: ReplSemanticsRuntimeContext>(
    line: &str,
    context: &mut C,
) -> SemanticsCommandOutput {
    let out = context.with_simplify_and_eval_options_mut(|simplify_options, eval_options| {
        crate::evaluate_semantics_command_line(line, simplify_options, eval_options)
    });
    if out.sync_simplifier {
        context.rebuild_simplifier_from_profile();
    }
    out
}
