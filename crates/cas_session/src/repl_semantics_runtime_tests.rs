use crate::{
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
        core.eval_options().shared.context_mode,
        cas_solver::ContextMode::Solve
    );
}

#[test]
fn apply_autoexpand_command_on_repl_core_updates_policy() {
    let mut core = crate::ReplCore::new();
    let out = apply_autoexpand_command_on_repl_core("autoexpand on", &mut core);
    assert_eq!(
        core.eval_options().shared.expand_policy,
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
        core.simplify_options().shared.semantics.domain_mode,
        cas_solver::DomainMode::Assume
    );
}

#[test]
fn evaluate_context_command_on_repl_returns_message() {
    let mut core = crate::ReplCore::new();
    let message =
        evaluate_context_command_on_repl("context solve", &mut core, &crate::CasConfig::default());
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
