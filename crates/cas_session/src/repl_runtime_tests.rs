use crate::{
    apply_profile_command_on_repl_core, build_repl_core_with_config, build_repl_prompt,
    clear_repl_profile_cache, eval_options_from_repl_core,
    evaluate_profile_command_message_on_repl_core, reset_repl_core_full_with_config,
    reset_repl_core_with_config, reset_repl_runtime_state,
};

#[test]
fn reset_repl_runtime_state_clears_session_and_runtime_flags() {
    let mut core = crate::ReplCore::new();
    core.set_debug_mode(true);
    core.set_health_enabled(true);
    core.set_last_stats(Some(crate::PipelineStats::default()));
    core.set_last_health_report(Some("cached".to_string()));

    let expr = cas_parser::parse("x+1", &mut core.simplifier_mut().context).expect("parse");
    core.state_mut()
        .history_push(crate::EntryKind::Expr(expr), "x+1");
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
    let mut core = crate::ReplCore::new();
    if let Err(err) = core.with_engine_and_state(|engine, state| {
        crate::evaluate_eval_command_output(engine, state, "x + x", false)
    }) {
        panic!("eval failed: {err:?}");
    }
    assert_eq!(core.profile_cache_len(), 1);

    clear_repl_profile_cache(&mut core);
    assert_eq!(core.profile_cache_len(), 0);
}

#[test]
fn apply_profile_command_on_repl_core_accepts_show() {
    let mut core = crate::ReplCore::new();
    let out = apply_profile_command_on_repl_core(&mut core, "profile");
    assert!(!out.trim().is_empty());
}

#[test]
fn evaluate_profile_command_message_on_repl_core_accepts_show() {
    let mut core = crate::ReplCore::new();
    let out = evaluate_profile_command_message_on_repl_core(&mut core, "profile");
    assert!(!out.trim().is_empty());
}

#[test]
fn reset_repl_core_with_config_resets_runtime_state() {
    let mut core = crate::ReplCore::new();
    core.set_debug_mode(true);
    core.set_health_enabled(true);
    let config = crate::CasConfig::default();
    reset_repl_core_with_config(&mut core, &config);
    assert!(!core.debug_mode());
    assert!(!core.health_enabled());
}

#[test]
fn reset_repl_core_full_with_config_clears_profile_cache() {
    let mut core = crate::ReplCore::new();
    if let Err(err) = core.with_engine_and_state(|engine, state| {
        crate::evaluate_eval_command_output(engine, state, "x + x", false)
    }) {
        panic!("eval failed: {err:?}");
    }
    assert_eq!(core.profile_cache_len(), 1);
    let config = crate::CasConfig::default();
    reset_repl_core_full_with_config(&mut core, &config);
    assert_eq!(core.profile_cache_len(), 0);
}

#[test]
fn build_repl_core_with_config_applies_toggle_sync() {
    let config = crate::CasConfig {
        distribute: true,
        ..crate::CasConfig::default()
    };
    let mut core = build_repl_core_with_config(&config);
    let eval = core.with_engine_and_state(|engine, state| {
        crate::evaluate_eval_command_output(engine, state, "(x+1)^2", false)
    });
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
