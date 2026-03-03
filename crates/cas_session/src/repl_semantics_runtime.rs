//! Runtime adapters to apply semantics commands directly on `ReplCore`.

use crate::{CasConfig, ReplCore};

/// Result of applying a semantics command over REPL runtime state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplSemanticsApplyOutput {
    pub message: String,
    pub rebuilt_simplifier: bool,
}

/// Apply `context` command to REPL core state.
pub fn apply_context_command_on_repl_core(
    line: &str,
    core: &mut ReplCore,
) -> ReplSemanticsApplyOutput {
    let applied = crate::evaluate_and_apply_context_command(line, core.eval_options_mut());
    if applied.rebuild_simplifier {
        core.rebuild_simplifier_from_profile();
    }
    ReplSemanticsApplyOutput {
        message: applied.message,
        rebuilt_simplifier: applied.rebuild_simplifier,
    }
}

/// Evaluate `context` command on REPL core and synchronize config toggles after rebuild.
pub fn evaluate_context_command_on_repl(
    line: &str,
    core: &mut ReplCore,
    config: &CasConfig,
) -> String {
    let applied = apply_context_command_on_repl_core(line, core);
    if applied.rebuilt_simplifier {
        crate::sync_simplifier_with_cas_config(core.simplifier_mut(), config);
    }
    applied.message
}

/// Apply `autoexpand` command to REPL core state.
pub fn apply_autoexpand_command_on_repl_core(
    line: &str,
    core: &mut ReplCore,
) -> ReplSemanticsApplyOutput {
    let applied = crate::evaluate_and_apply_autoexpand_command(line, core.eval_options_mut());
    if applied.rebuild_simplifier {
        core.rebuild_simplifier_from_profile();
    }
    ReplSemanticsApplyOutput {
        message: applied.message,
        rebuilt_simplifier: applied.rebuild_simplifier,
    }
}

/// Evaluate `autoexpand` command on REPL core and synchronize config toggles after rebuild.
pub fn evaluate_autoexpand_command_on_repl(
    line: &str,
    core: &mut ReplCore,
    config: &CasConfig,
) -> String {
    let applied = apply_autoexpand_command_on_repl_core(line, core);
    if applied.rebuilt_simplifier {
        crate::sync_simplifier_with_cas_config(core.simplifier_mut(), config);
    }
    applied.message
}

/// Apply `semantics` command to REPL core state.
pub fn apply_semantics_command_on_repl_core(
    line: &str,
    core: &mut ReplCore,
) -> crate::SemanticsCommandOutput {
    let out = core.with_simplify_and_eval_options_mut(|simplify_options, eval_options| {
        crate::evaluate_semantics_command_line(line, simplify_options, eval_options)
    });
    if out.sync_simplifier {
        core.rebuild_simplifier_from_profile();
    }
    out
}

/// Evaluate `semantics` command on REPL core and synchronize config toggles after rebuild.
pub fn evaluate_semantics_command_on_repl(
    line: &str,
    core: &mut ReplCore,
    config: &CasConfig,
) -> String {
    let out = apply_semantics_command_on_repl_core(line, core);
    if out.sync_simplifier {
        crate::sync_simplifier_with_cas_config(core.simplifier_mut(), config);
    }
    out.lines.join("\n")
}
