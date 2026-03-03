use crate::{CasConfig, ReplCore};

/// Reset non-config runtime state while preserving the current simplifier/profile setup.
pub fn reset_repl_runtime_state(core: &mut ReplCore) {
    core.clear_state();
    core.set_debug_mode(false);
    core.set_last_stats(None);
    core.set_health_enabled(false);
    core.set_last_health_report(None);
}

/// Build a `ReplCore` preconfigured from persisted CLI config.
pub fn build_repl_core_with_config(config: &CasConfig) -> ReplCore {
    let simplifier =
        crate::build_simplifier_with_rule_config(crate::solver_rule_config_from_cas_config(config));
    let mut core = ReplCore::with_simplifier(simplifier);
    crate::sync_simplifier_with_cas_config(core.simplifier_mut(), config);
    core
}

/// Build REPL prompt text from current REPL core state.
pub fn build_repl_prompt(core: &ReplCore) -> String {
    crate::build_prompt_from_eval_options(core.eval_options())
}

/// Clone current eval options from REPL core state.
pub fn eval_options_from_repl_core(core: &ReplCore) -> cas_solver::EvalOptions {
    core.eval_options().clone()
}

/// Clear engine profile cache for the active REPL core.
pub fn clear_repl_profile_cache(core: &mut ReplCore) {
    core.clear_profile_cache();
}

/// Rebuild simplifier from persisted config and reset runtime/session state.
pub fn reset_repl_core_with_config(core: &mut ReplCore, config: &CasConfig) {
    core.set_simplifier(crate::build_simplifier_with_rule_config(
        crate::solver_rule_config_from_cas_config(config),
    ));
    crate::sync_simplifier_with_cas_config(core.simplifier_mut(), config);
    reset_repl_runtime_state(core);
}

/// Full reset: state reset + profile cache clear.
pub fn reset_repl_core_full_with_config(core: &mut ReplCore, config: &CasConfig) {
    reset_repl_core_with_config(core, config);
    clear_repl_profile_cache(core);
}

/// Apply `profile` command line to REPL core simplifier profiler.
pub fn apply_profile_command_on_repl_core(core: &mut ReplCore, line: &str) -> String {
    crate::apply_profile_command(core.simplifier_mut(), line)
}

/// Evaluate `profile` command and return user-facing message.
pub fn evaluate_profile_command_message_on_repl_core(core: &mut ReplCore, line: &str) -> String {
    apply_profile_command_on_repl_core(core, line)
}
