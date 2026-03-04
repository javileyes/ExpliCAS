//! Canonical stateless JSON bridge for external frontends.
//!
//! Keeps frontend crates (CLI/FFI) depending on application/session layer
//! rather than directly on solver internals.

/// Stateless canonical eval JSON entry point.
pub fn evaluate_eval_json_canonical(expr: &str, opts_json: &str) -> String {
    cas_solver::eval_str_to_json(expr, opts_json)
}

/// Stateless canonical substitute JSON entry point.
pub fn evaluate_substitute_json_canonical(
    expr: &str,
    target: &str,
    replacement: &str,
    opts_json: Option<&str>,
) -> String {
    cas_solver::substitute_str_to_json(expr, target, replacement, opts_json)
}
