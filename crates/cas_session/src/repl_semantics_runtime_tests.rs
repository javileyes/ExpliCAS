use crate::repl_api::{
    evaluate_autoexpand_command_on_repl, evaluate_context_command_on_repl,
    evaluate_semantics_command_on_repl,
};
#[allow(unused_imports)]
use cas_solver::session_api::{assumptions::*, settings::*, simplifier::*};
use cas_solver_core::domain_mode::DomainMode;
use cas_solver_core::eval_option_axes::ContextMode;
use cas_solver_core::expand_policy::ExpandPolicy;

#[test]
fn apply_context_command_on_repl_core_updates_mode() {
    let mut core = crate::repl_core::ReplCore::new();
    let out = apply_context_command_on_repl_core("context solve", &mut core);
    assert!(out.rebuilt_simplifier);
    assert_eq!(core.eval_options().shared.context_mode, ContextMode::Solve);
}

#[test]
fn apply_autoexpand_command_on_repl_core_updates_policy() {
    let mut core = crate::repl_core::ReplCore::new();
    let out = apply_autoexpand_command_on_repl_core("autoexpand on", &mut core);
    assert_eq!(core.eval_options().shared.expand_policy, ExpandPolicy::Auto);
    assert!(out.message.contains("Auto-expand"));
}

#[test]
fn apply_semantics_command_on_repl_core_updates_domain() {
    let mut core = crate::repl_core::ReplCore::new();
    let out = apply_semantics_command_on_repl_core("semantics set domain assume", &mut core);
    assert!(out.sync_simplifier);
    assert_eq!(
        core.simplify_options().shared.semantics.domain_mode,
        DomainMode::Assume
    );
}

#[test]
fn evaluate_context_command_on_repl_returns_message() {
    let mut core = crate::repl_core::ReplCore::new();
    let message = evaluate_context_command_on_repl(
        "context solve",
        &mut core,
        &crate::config::CasConfig::default(),
    );
    assert!(message.contains("Context"));
}

#[test]
fn evaluate_autoexpand_command_on_repl_returns_message() {
    let mut core = crate::repl_core::ReplCore::new();
    let message = evaluate_autoexpand_command_on_repl(
        "autoexpand on",
        &mut core,
        &crate::config::CasConfig::default(),
    );
    assert!(message.contains("Auto-expand"));
}

#[test]
fn evaluate_semantics_command_on_repl_returns_summary_lines() {
    let mut core = crate::repl_core::ReplCore::new();
    let message = evaluate_semantics_command_on_repl(
        "semantics set domain assume",
        &mut core,
        &crate::config::CasConfig::default(),
    );
    assert!(!message.trim().is_empty());
}
