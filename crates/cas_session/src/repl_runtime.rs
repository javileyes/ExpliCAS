//! Runtime helpers for REPL-core lifecycle operations.

use crate::{CasConfig, ReplCore};

/// Reset non-config runtime state while preserving the current simplifier/profile setup.
pub fn reset_repl_runtime_state(core: &mut ReplCore) {
    core.state.clear();
    core.debug_mode = false;
    core.last_stats = None;
    core.health_enabled = false;
    core.last_health_report = None;
}

/// Build a `ReplCore` preconfigured from persisted CLI config.
pub fn build_repl_core_with_config(config: &CasConfig) -> ReplCore {
    let simplifier =
        crate::build_simplifier_with_rule_config(crate::solver_rule_config_from_cas_config(config));
    let mut core = ReplCore::with_simplifier(simplifier);
    crate::sync_simplifier_with_cas_config(&mut core.engine.simplifier, config);
    core
}

/// Build REPL prompt text from current REPL core state.
pub fn build_repl_prompt(core: &ReplCore) -> String {
    crate::build_prompt_from_eval_options(core.state.options())
}

/// Clone current eval options from REPL core state.
pub fn eval_options_from_repl_core(core: &ReplCore) -> cas_solver::EvalOptions {
    core.state.options().clone()
}

/// Clear engine profile cache for the active REPL core.
pub fn clear_repl_profile_cache(core: &mut ReplCore) {
    core.engine.clear_profile_cache();
}

/// Rebuild simplifier from persisted config and reset runtime/session state.
pub fn reset_repl_core_with_config(core: &mut ReplCore, config: &CasConfig) {
    core.engine.simplifier =
        crate::build_simplifier_with_rule_config(crate::solver_rule_config_from_cas_config(config));
    crate::sync_simplifier_with_cas_config(&mut core.engine.simplifier, config);
    reset_repl_runtime_state(core);
}

/// Full reset: state reset + profile cache clear.
pub fn reset_repl_core_full_with_config(core: &mut ReplCore, config: &CasConfig) {
    reset_repl_core_with_config(core, config);
    clear_repl_profile_cache(core);
}

/// Apply `profile` command line to REPL core simplifier profiler.
pub fn apply_profile_command_on_repl_core(core: &mut ReplCore, line: &str) -> String {
    crate::apply_profile_command(&mut core.engine.simplifier, line)
}

#[cfg(test)]
mod tests {
    use super::{
        apply_profile_command_on_repl_core, build_repl_core_with_config, build_repl_prompt,
        clear_repl_profile_cache, eval_options_from_repl_core, reset_repl_core_full_with_config,
        reset_repl_core_with_config, reset_repl_runtime_state,
    };

    #[test]
    fn reset_repl_runtime_state_clears_session_and_runtime_flags() {
        let mut core = crate::ReplCore::new();
        core.debug_mode = true;
        core.health_enabled = true;
        core.last_stats = Some(crate::PipelineStats::default());
        core.last_health_report = Some("cached".to_string());

        let expr = cas_parser::parse("x+1", &mut core.engine.simplifier.context).expect("parse");
        core.state.history_push(crate::EntryKind::Expr(expr), "x+1");
        assert_eq!(core.state.history_len(), 1);

        reset_repl_runtime_state(&mut core);

        assert!(!core.debug_mode);
        assert!(!core.health_enabled);
        assert!(core.last_stats.is_none());
        assert!(core.last_health_report.is_none());
        assert_eq!(core.state.history_len(), 0);
    }

    #[test]
    fn clear_repl_profile_cache_empties_cache() {
        let mut core = crate::ReplCore::new();
        if let Err(err) =
            crate::evaluate_eval_command_output(&mut core.engine, &mut core.state, "x + x", false)
        {
            panic!("eval failed: {err:?}");
        }
        assert_eq!(core.engine.profile_cache_len(), 1);

        clear_repl_profile_cache(&mut core);
        assert_eq!(core.engine.profile_cache_len(), 0);
    }

    #[test]
    fn apply_profile_command_on_repl_core_accepts_show() {
        let mut core = crate::ReplCore::new();
        let out = apply_profile_command_on_repl_core(&mut core, "profile");
        assert!(!out.trim().is_empty());
    }

    #[test]
    fn reset_repl_core_with_config_resets_runtime_state() {
        let mut core = crate::ReplCore::new();
        core.debug_mode = true;
        core.health_enabled = true;
        let config = crate::CasConfig::default();
        reset_repl_core_with_config(&mut core, &config);
        assert!(!core.debug_mode);
        assert!(!core.health_enabled);
    }

    #[test]
    fn reset_repl_core_full_with_config_clears_profile_cache() {
        let mut core = crate::ReplCore::new();
        if let Err(err) =
            crate::evaluate_eval_command_output(&mut core.engine, &mut core.state, "x + x", false)
        {
            panic!("eval failed: {err:?}");
        }
        assert_eq!(core.engine.profile_cache_len(), 1);
        let config = crate::CasConfig::default();
        reset_repl_core_full_with_config(&mut core, &config);
        assert_eq!(core.engine.profile_cache_len(), 0);
    }

    #[test]
    fn build_repl_core_with_config_applies_toggle_sync() {
        let config = crate::CasConfig {
            distribute: true,
            ..crate::CasConfig::default()
        };
        let mut core = build_repl_core_with_config(&config);
        let eval = crate::evaluate_eval_command_output(
            &mut core.engine,
            &mut core.state,
            "(x+1)^2",
            false,
        );
        assert!(eval.is_ok());
    }

    #[test]
    fn build_repl_prompt_returns_default_prompt() {
        let core = crate::ReplCore::new();
        let prompt = build_repl_prompt(&core);
        assert_eq!(prompt, "> ");
    }

    #[test]
    fn eval_options_from_repl_core_clones_default_options() {
        let core = crate::ReplCore::new();
        let options = eval_options_from_repl_core(&core);
        assert_eq!(
            options.steps_mode,
            cas_solver::EvalOptions::default().steps_mode
        );
        assert_eq!(
            options.shared.context_mode,
            cas_solver::EvalOptions::default().shared.context_mode
        );
    }
}
