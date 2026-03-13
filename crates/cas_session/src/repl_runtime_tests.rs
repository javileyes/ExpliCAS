use crate::repl::{
    build_repl_core_with_config, reset_repl_core_full_with_config, reset_repl_core_with_config,
};
#[allow(unused_imports)]
use cas_solver::session_api::{assumptions::*, lifecycle::*, profile::*, simplifier::*};
use cas_solver_core::eval_options::EvalOptions;
use cas_solver_core::phase_stats::PipelineStats;

#[test]
fn reset_repl_runtime_state_clears_session_and_runtime_flags() {
    let mut core = crate::repl_core::ReplCore::new();
    core.set_debug_mode(true);
    core.set_health_enabled(true);
    core.set_last_stats(Some(PipelineStats::default()));
    core.set_last_health_report(Some("cached".to_string()));

    let expr = cas_parser::parse("x+1", &mut core.simplifier_mut().context).expect("parse");
    core.state_mut()
        .history_push(cas_session_core::types::EntryKind::Expr(expr), "x+1");
    assert_eq!(core.state().history_len(), 1);

    reset_repl_runtime_state(&mut core);

    assert!(!core.debug_mode());
    assert!(!core.health_enabled());
    assert!(core.last_stats().is_none());
    assert!(core.last_health_report().is_none());
    assert_eq!(core.state().history_len(), 0);
}

#[test]
fn clear_repl_profile_cache_empties_cache() {
    let mut core = crate::repl_core::ReplCore::new();
    if let Err(err) = core.with_engine_and_state(|engine, state| {
        cas_solver::session_api::eval::evaluate_eval_command_output(engine, state, "x + x", false)
    }) {
        panic!("eval failed: {err:?}");
    }
    assert_eq!(core.profile_cache_len(), 1);

    clear_repl_profile_cache(&mut core);
    assert_eq!(core.profile_cache_len(), 0);
}

#[test]
fn apply_profile_command_on_repl_core_accepts_show() {
    let mut core = crate::repl_core::ReplCore::new();
    let out = apply_profile_command_on_repl_core(&mut core, "profile");
    assert!(!out.trim().is_empty());
}

#[test]
fn evaluate_profile_command_message_on_repl_core_accepts_show() {
    let mut core = crate::repl_core::ReplCore::new();
    let out = evaluate_profile_command_message_on_repl_core(&mut core, "profile");
    assert!(!out.trim().is_empty());
}

#[test]
fn reset_repl_core_with_config_resets_runtime_state() {
    let mut core = crate::repl_core::ReplCore::new();
    core.set_debug_mode(true);
    core.set_health_enabled(true);
    let config = crate::config::CasConfig::default();
    reset_repl_core_with_config(&mut core, &config);
    assert!(!core.debug_mode());
    assert!(!core.health_enabled());
}

#[test]
fn reset_repl_core_full_with_config_clears_profile_cache() {
    let mut core = crate::repl_core::ReplCore::new();
    if let Err(err) = core.with_engine_and_state(|engine, state| {
        cas_solver::session_api::eval::evaluate_eval_command_output(engine, state, "x + x", false)
    }) {
        panic!("eval failed: {err:?}");
    }
    assert_eq!(core.profile_cache_len(), 1);
    let config = crate::config::CasConfig::default();
    reset_repl_core_full_with_config(&mut core, &config);
    assert_eq!(core.profile_cache_len(), 0);
}

#[test]
fn build_repl_core_with_config_applies_toggle_sync() {
    let config = crate::config::CasConfig {
        distribute: true,
        ..crate::config::CasConfig::default()
    };
    let mut core = build_repl_core_with_config(&config);
    let eval = core.with_engine_and_state(|engine, state| {
        cas_solver::session_api::eval::evaluate_eval_command_output(engine, state, "(x+1)^2", false)
    });
    assert!(eval.is_ok());
}

#[test]
fn build_repl_prompt_returns_default_prompt() {
    let core = crate::repl_core::ReplCore::new();
    let prompt = build_repl_prompt(&core);
    assert_eq!(prompt, "> ");
}

#[test]
fn eval_options_from_repl_core_clones_default_options() {
    let core = crate::repl_core::ReplCore::new();
    let options = eval_options_from_repl_core(&core);
    assert_eq!(options.steps_mode, EvalOptions::default().steps_mode);
    assert_eq!(
        options.shared.context_mode,
        EvalOptions::default().shared.context_mode
    );
}
