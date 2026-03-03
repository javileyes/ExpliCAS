//! Runtime adapter for `config` command over REPL state.

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
    let applied = crate::evaluate_and_apply_config_command(line, config);
    if applied.sync_simplifier {
        crate::sync_simplifier_with_cas_config(core.simplifier_mut(), config);
    }
    applied.message
}
