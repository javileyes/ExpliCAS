#[cfg(test)]
mod tests {
    use crate::{evaluate_eval_json_canonical, evaluate_substitute_json_canonical};

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
