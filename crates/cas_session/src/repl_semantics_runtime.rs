//! Runtime adapters that synchronize semantics-related REPL changes with `CasConfig`.

use crate::{CasConfig, ReplCore};

/// Evaluate `context` command on REPL core and synchronize config toggles after rebuild.
pub fn evaluate_context_command_on_repl(
    line: &str,
    core: &mut ReplCore,
    config: &CasConfig,
) -> String {
    cas_solver::evaluate_context_command_with_config_sync_on_runtime(
        line,
        core,
        config,
        |core, config| crate::sync_simplifier_with_cas_config(core.simplifier_mut(), config),
    )
}

/// Evaluate `autoexpand` command on REPL core and synchronize config toggles after rebuild.
pub fn evaluate_autoexpand_command_on_repl(
    line: &str,
    core: &mut ReplCore,
    config: &CasConfig,
) -> String {
    cas_solver::evaluate_autoexpand_command_with_config_sync_on_runtime(
        line,
        core,
        config,
        |core, config| crate::sync_simplifier_with_cas_config(core.simplifier_mut(), config),
    )
}

/// Evaluate `semantics` command on REPL core and synchronize config toggles after rebuild.
pub fn evaluate_semantics_command_on_repl(
    line: &str,
    core: &mut ReplCore,
    config: &CasConfig,
) -> String {
    cas_solver::evaluate_semantics_command_with_config_sync_on_runtime(
        line,
        core,
        config,
        |core, config| crate::sync_simplifier_with_cas_config(core.simplifier_mut(), config),
    )
}
