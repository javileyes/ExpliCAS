//! Contract tests for substitute_str_to_json API.
//!
//! These tests verify the JSON contract for substitute operations:
//! 1. Schema version is stable (v1)
//! 2. Request echo is present and accurate
//! 3. Options are reflected correctly
//! 4. Steps follow the schema
//! 5. No __hold leak in result or steps
//! 6. Error path produces correct error structure

use cas_engine::json::substitute_str_to_json;

// ============================================================================
// Schema & Structure Tests
// ============================================================================

#[test]
fn schema_version_is_one() {
    let json = substitute_str_to_json("x^2 + 1", "x^2", "y", None);
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["schema_version"], 1);
}

#[test]
fn ok_true_on_success() {
    let json = substitute_str_to_json("x^2 + 1", "x^2", "y", None);
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["ok"], true);
}

#[test]
fn result_present_on_success() {
    let json = substitute_str_to_json("x^2 + 1", "x^2", "y", None);
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(v["result"].is_string());
    assert!(v["result"].as_str().unwrap().contains("y"));
}

// ============================================================================
// Request Echo Tests
// ============================================================================

#[test]
fn request_echo_present() {
    let json = substitute_str_to_json("x^4 + 1", "x^2", "y", None);
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert!(v["request"].is_object());
    assert_eq!(v["request"]["expr"], "x^4 + 1");
    assert_eq!(v["request"]["target"], "x^2");
    assert_eq!(v["request"]["with"], "y");
}

// ============================================================================
// Options Tests
// ============================================================================

#[test]
fn options_reflect_mode() {
    let json = substitute_str_to_json("x^4 + 1", "x^2", "y", Some(r#"{"mode":"exact"}"#));
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert!(v["options"]["substitute"].is_object());
    assert_eq!(v["options"]["substitute"]["mode"], "exact");
}

#[test]
fn options_default_is_power() {
    let json = substitute_str_to_json("x^4 + 1", "x^2", "y", None);
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(v["options"]["substitute"]["mode"], "power");
}

#[test]
fn options_reflect_steps_flag() {
    let json = substitute_str_to_json("x^2 + 1", "x^2", "y", Some(r#"{"steps":true}"#));
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(v["options"]["substitute"]["steps"], true);
}

// ============================================================================
// Steps Tests
// ============================================================================

#[test]
fn steps_present_when_requested() {
    let json = substitute_str_to_json("x^4 + x^2 + 1", "x^2", "y", Some(r#"{"steps":true}"#));
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert!(v["steps"].is_array());
    let steps = v["steps"].as_array().unwrap();
    assert!(!steps.is_empty(), "Should have at least one step");
}

#[test]
fn steps_have_required_fields() {
    let json = substitute_str_to_json("x^2 + 1", "x^2", "y", Some(r#"{"steps":true}"#));
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();

    let steps = v["steps"].as_array().unwrap();
    for step in steps {
        assert!(step["rule"].is_string(), "step.rule must be string");
        assert!(step["before"].is_string(), "step.before must be string");
        assert!(step["after"].is_string(), "step.after must be string");
    }
}

#[test]
fn steps_rule_names_are_stable() {
    let json = substitute_str_to_json("x^4 + x^2 + 1", "x^2", "y", Some(r#"{"steps":true}"#));
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();

    let steps = v["steps"].as_array().unwrap();
    let valid_rules = [
        "SubstituteExact",
        "SubstitutePowerMultiple",
        "SubstitutePowOfTarget",
    ];

    for step in steps {
        let rule = step["rule"].as_str().unwrap();
        assert!(valid_rules.contains(&rule), "Unknown rule: {}", rule);
    }
}

#[test]
fn steps_before_not_equal_after() {
    let json = substitute_str_to_json("x^4 + x^2 + 1", "x^2", "y", Some(r#"{"steps":true}"#));
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();

    let steps = v["steps"].as_array().unwrap();
    for step in steps {
        let before = step["before"].as_str().unwrap();
        let after = step["after"].as_str().unwrap();
        assert_ne!(before, after, "step.before should not equal step.after");
    }
}

// ============================================================================
// No __hold Leak Tests
// ============================================================================

#[test]
fn no_hold_leak_in_result() {
    let json = substitute_str_to_json("x^4 + x^2 + 1", "x^2", "y", None);
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();

    let result = v["result"].as_str().unwrap();
    assert!(
        !result.contains("__hold"),
        "Result contains __hold: {}",
        result
    );
}

#[test]
fn no_hold_leak_in_steps() {
    let json = substitute_str_to_json("x^4 + x^2 + 1", "x^2", "y", Some(r#"{"steps":true}"#));
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();

    let steps = v["steps"].as_array().unwrap();
    for step in steps {
        let before = step["before"].as_str().unwrap();
        let after = step["after"].as_str().unwrap();
        assert!(
            !before.contains("__hold"),
            "step.before contains __hold: {}",
            before
        );
        assert!(
            !after.contains("__hold"),
            "step.after contains __hold: {}",
            after
        );
    }
}

// ============================================================================
// Error Path Tests
// ============================================================================

#[test]
fn parse_error_has_correct_structure() {
    let json = substitute_str_to_json("x^2 + 1", "invalid(((", "y", None);
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(v["ok"], false);
    assert!(v["error"].is_object());
    assert_eq!(v["error"]["kind"], "ParseError");
    assert_eq!(v["error"]["code"], "E_PARSE");
    assert!(v["error"]["message"].is_string());
}

#[test]
fn error_still_has_request_echo() {
    let json = substitute_str_to_json("x^2 + 1", "invalid(((", "y", None);
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();

    // Even on error, request echo should be present for debugging
    assert!(v["request"].is_object());
    assert_eq!(v["request"]["expr"], "x^2 + 1");
    assert_eq!(v["request"]["target"], "invalid(((");
}

// ============================================================================
// Power Mode Behavior Tests
// ============================================================================

#[test]
fn power_mode_replaces_multiples() {
    let json = substitute_str_to_json("x^4 + x^2 + 1", "x^2", "y", Some(r#"{"mode":"power"}"#));
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();

    let result = v["result"].as_str().unwrap();
    // x^4 should become y^2, x^2 should become y
    assert!(result.contains("y"), "Should contain y");
    assert!(!result.contains("x"), "Should not contain x (all replaced)");
}

#[test]
fn exact_mode_only_replaces_exact() {
    let json = substitute_str_to_json("x^4 + x^2 + 1", "x^2", "y", Some(r#"{"mode":"exact"}"#));
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();

    let result = v["result"].as_str().unwrap();
    // x^4 should remain (not a multiple), x^2 should become y
    assert!(result.contains("y"), "Should contain y (from x^2)");
    assert!(result.contains("x"), "Should still contain x (from x^4)");
}
