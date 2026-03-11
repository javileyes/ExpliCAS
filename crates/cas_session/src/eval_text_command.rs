//! Session-backed eval text command orchestration.

use std::path::Path;

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
    crate::run_with_domain_session(session_path, domain, |engine, state| {
        crate::solver_exports::evaluate_eval_text_simplify_with_session(
            engine, state, expr, auto_store,
        )
    })
}
