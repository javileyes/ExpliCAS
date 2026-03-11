use super::super::super::*;

#[test]
fn evaluate_timeline_invocation_cli_actions_with_session_returns_actions() {
    let mut engine = crate::cas_solver::Engine::new();
    let mut session = cas_session::SessionState::new();
    let options = crate::cas_solver::EvalOptions::default();
    let actions = evaluate_timeline_invocation_cli_actions_with_session(
        &mut engine,
        &mut session,
        "timeline x+1",
        &options,
        VerbosityLevel::Normal,
    )
    .expect("timeline eval");
    assert!(!actions.is_empty());
}
