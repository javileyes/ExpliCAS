//! Session-backed eval text command orchestration.

use std::path::Path;

use cas_solver::runtime::Engine;
use cas_solver::session_api::eval::evaluate_eval_text_simplify_with_session;

fn can_skip_persisted_session_state(expr: &str, auto_store: bool) -> bool {
    !auto_store && !expr.contains('#')
}

fn should_use_read_only_persisted_session(expr: &str, auto_store: bool) -> bool {
    !auto_store && expr.contains('#')
}

/// Evaluate plain-text `eval` command with optional persisted session.
///
/// Returns:
/// - evaluation result (rendered string or error message)
/// - optional load warning
/// - optional save warning
pub fn evaluate_eval_text_command_with_session(
    session_path: Option<&Path>,
    domain: &str,
    expr: &str,
    auto_store: bool,
) -> (Result<String, String>, Option<String>, Option<String>) {
    if session_path.is_none() || can_skip_persisted_session_state(expr, auto_store) {
        let mut engine = Engine::new();
        let mut state = crate::state_core::SessionState::new();
        let result =
            evaluate_eval_text_simplify_with_session(&mut engine, &mut state, expr, auto_store);
        return (result, None, None);
    }

    if should_use_read_only_persisted_session(expr, auto_store) {
        return crate::session_io::run_read_only_with_domain_session(
            session_path,
            domain,
            |engine, state| {
                evaluate_eval_text_simplify_with_session(engine, state, expr, auto_store)
            },
        );
    }

    crate::session_io::run_with_domain_session(session_path, domain, |engine, state| {
        evaluate_eval_text_simplify_with_session(engine, state, expr, auto_store)
    })
}
