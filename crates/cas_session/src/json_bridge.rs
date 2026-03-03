//! Canonical stateless JSON bridge for external frontends.
//!
//! Keeps frontend crates (CLI/FFI) depending on application/session layer
//! rather than directly on solver internals.

/// Stateless canonical eval JSON entry point.
///
/// Kept in this module as a stable facade for lints/frontends while
/// implementation lives in `json_bridge_eval`.
pub fn evaluate_eval_json_canonical(expr: &str, opts_json: &str) -> String {
    crate::json_bridge_eval::evaluate_eval_json_canonical(expr, opts_json)
}

/// Stateless canonical substitute JSON entry point.
///
/// Kept in this module as a stable facade for lints/frontends while
/// implementation lives in `json_bridge_substitute`.
pub fn evaluate_substitute_json_canonical(
    expr: &str,
    target: &str,
    replacement: &str,
    opts_json: Option<&str>,
) -> String {
    crate::json_bridge_substitute::evaluate_substitute_json_canonical(
        expr,
        target,
        replacement,
        opts_json,
    )
}
