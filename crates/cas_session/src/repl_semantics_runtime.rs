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
    let applied = crate::evaluate_and_apply_context_command(line, core.state.options_mut());
    if applied.rebuild_simplifier {
        core.engine.simplifier = cas_solver::Simplifier::with_profile(core.state.options());
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
        crate::sync_simplifier_with_cas_config(&mut core.engine.simplifier, config);
    }
    applied.message
}

/// Apply `autoexpand` command to REPL core state.
pub fn apply_autoexpand_command_on_repl_core(
    line: &str,
    core: &mut ReplCore,
) -> ReplSemanticsApplyOutput {
    let applied = crate::evaluate_and_apply_autoexpand_command(line, core.state.options_mut());
    if applied.rebuild_simplifier {
        core.engine.simplifier = cas_solver::Simplifier::with_profile(core.state.options());
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
        crate::sync_simplifier_with_cas_config(&mut core.engine.simplifier, config);
    }
    applied.message
}

/// Apply `semantics` command to REPL core state.
pub fn apply_semantics_command_on_repl_core(
    line: &str,
    core: &mut ReplCore,
) -> crate::SemanticsCommandOutput {
    let out = crate::evaluate_semantics_command_line(
        line,
        &mut core.simplify_options,
        core.state.options_mut(),
    );
    if out.sync_simplifier {
        core.engine.simplifier = cas_solver::Simplifier::with_profile(core.state.options());
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
        crate::sync_simplifier_with_cas_config(&mut core.engine.simplifier, config);
    }
    out.lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::{
        apply_autoexpand_command_on_repl_core, apply_context_command_on_repl_core,
        apply_semantics_command_on_repl_core, evaluate_autoexpand_command_on_repl,
        evaluate_context_command_on_repl, evaluate_semantics_command_on_repl,
    };

    #[test]
    fn apply_context_command_on_repl_core_updates_mode() {
        let mut core = crate::ReplCore::new();
        let out = apply_context_command_on_repl_core("context solve", &mut core);
        assert!(out.rebuilt_simplifier);
        assert_eq!(
            core.state.options().shared.context_mode,
            cas_solver::ContextMode::Solve
        );
    }

    #[test]
    fn apply_autoexpand_command_on_repl_core_updates_policy() {
        let mut core = crate::ReplCore::new();
        let out = apply_autoexpand_command_on_repl_core("autoexpand on", &mut core);
        assert_eq!(
            core.state.options().shared.expand_policy,
            cas_solver::ExpandPolicy::Auto
        );
        assert!(out.message.contains("Auto-expand"));
    }

    #[test]
    fn apply_semantics_command_on_repl_core_updates_domain() {
        let mut core = crate::ReplCore::new();
        let out = apply_semantics_command_on_repl_core("semantics set domain assume", &mut core);
        assert!(out.sync_simplifier);
        assert_eq!(
            core.simplify_options.shared.semantics.domain_mode,
            cas_solver::DomainMode::Assume
        );
    }

    #[test]
    fn evaluate_context_command_on_repl_returns_message() {
        let mut core = crate::ReplCore::new();
        let message = evaluate_context_command_on_repl(
            "context solve",
            &mut core,
            &crate::CasConfig::default(),
        );
        assert!(message.contains("Context"));
    }

    #[test]
    fn evaluate_autoexpand_command_on_repl_returns_message() {
        let mut core = crate::ReplCore::new();
        let message = evaluate_autoexpand_command_on_repl(
            "autoexpand on",
            &mut core,
            &crate::CasConfig::default(),
        );
        assert!(message.contains("Auto-expand"));
    }

    #[test]
    fn evaluate_semantics_command_on_repl_returns_summary_lines() {
        let mut core = crate::ReplCore::new();
        let message = evaluate_semantics_command_on_repl(
            "semantics set domain assume",
            &mut core,
            &crate::CasConfig::default(),
        );
        assert!(!message.trim().is_empty());
    }
}
