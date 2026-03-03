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

#[cfg(test)]
mod tests {
    use super::{evaluate_eval_json_canonical, evaluate_substitute_json_canonical};

    #[test]
    fn eval_json_bridge_returns_valid_contract() {
        let payload = evaluate_eval_json_canonical("x + x", "{}");
        let json: serde_json::Value = serde_json::from_str(&payload).expect("valid json");
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["ok"], true);
    }

    #[test]
    fn substitute_json_bridge_returns_valid_contract() {
        let payload = evaluate_substitute_json_canonical("x^2 + 1", "x", "y", None);
        let json: serde_json::Value = serde_json::from_str(&payload).expect("valid json");
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["ok"], true);
    }

    #[test]
    fn substitute_json_bridge_parse_error_contract() {
        let payload = evaluate_substitute_json_canonical("x^2 + 1", "invalid(((", "y", Some("{}"));
        let json: serde_json::Value = serde_json::from_str(&payload).expect("valid json");
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["ok"], false);
        assert_eq!(json["error"]["kind"], "ParseError");
    }
}
