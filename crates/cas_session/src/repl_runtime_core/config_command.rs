use crate::{CasConfig, ReplCore};

/// Apply a config command against REPL config + core runtime.
///
/// If the command changes solver toggles, this also syncs them into the
/// active simplifier.
pub fn evaluate_and_apply_config_command_on_repl(
    line: &str,
    config: &mut CasConfig,
    core: &mut ReplCore,
) -> String {
    cas_solver::evaluate_and_apply_config_command_on_runtime(
        line,
        config,
        core,
        crate::evaluate_and_apply_config_command,
        crate::sync_simplifier_with_cas_config,
    )
}
