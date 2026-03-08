use crate::{CasConfig, ReplCore};

use super::sync::sync_config_to_core;

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
        sync_config_to_core,
    )
}
