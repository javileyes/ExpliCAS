use crate::{CasConfig, ReplCore};

fn build_repl_simplifier_from_config(config: &CasConfig) -> cas_solver::Simplifier {
    crate::build_simplifier_with_rule_config(crate::solver_rule_config_from_cas_config(config))
}

/// Build a `ReplCore` preconfigured from persisted CLI config.
pub fn build_repl_core_with_config(config: &CasConfig) -> ReplCore {
    cas_solver::build_runtime_with_config(
        config,
        build_repl_simplifier_from_config,
        ReplCore::with_simplifier,
        crate::sync_simplifier_with_cas_config,
    )
}

/// Rebuild simplifier from persisted config and reset runtime/session state.
pub fn reset_repl_core_with_config(core: &mut ReplCore, config: &CasConfig) {
    cas_solver::reset_runtime_with_config(
        core,
        config,
        build_repl_simplifier_from_config,
        crate::sync_simplifier_with_cas_config,
    );
}

/// Full reset: state reset + profile cache clear.
pub fn reset_repl_core_full_with_config(core: &mut ReplCore, config: &CasConfig) {
    cas_solver::reset_runtime_full_with_config(
        core,
        config,
        build_repl_simplifier_from_config,
        crate::sync_simplifier_with_cas_config,
    );
}

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
