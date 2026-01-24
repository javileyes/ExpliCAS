//! Smoke tests for wire model in eval-json output.
//!
//! Verifies that the wire field is present and correctly structured
//! in JSON output from eval commands.

use assert_cmd::cargo;
use serde_json::Value;
use std::process::Command;

/// Helper to run eval-json and parse the wire field
fn eval_json_wire(expr: &str) -> Value {
    let output = Command::new(cargo::cargo_bin!("cas_cli"))
        .arg("eval")
        .arg(expr)
        .arg("--format")
        .arg("json")
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: Value = serde_json::from_str(&stdout).expect("Failed to parse JSON");
    json.get("wire").cloned().unwrap_or(Value::Null)
}

#[test]
fn test_wire_present_on_success() {
    let wire = eval_json_wire("2+2");

    // Wire should exist
    assert!(!wire.is_null(), "wire field should be present");

    // Schema version should be 1
    assert_eq!(
        wire.get("schema_version"),
        Some(&Value::Number(1.into())),
        "schema_version should be 1"
    );

    // Messages should be an array
    let messages = wire.get("messages").expect("messages should exist");
    assert!(messages.is_array(), "messages should be an array");
    assert!(
        !messages.as_array().unwrap().is_empty(),
        "messages should not be empty"
    );
}

#[test]
fn test_wire_output_message_present() {
    let wire = eval_json_wire("3*4");

    let messages = wire.get("messages").expect("messages should exist");
    let msgs = messages.as_array().unwrap();

    // Should have at least one output message
    let output_msgs: Vec<_> = msgs
        .iter()
        .filter(|m| m.get("kind") == Some(&Value::String("output".into())))
        .collect();

    assert!(
        !output_msgs.is_empty(),
        "should have at least one output message"
    );

    // Output message should contain result
    let output_text = output_msgs[0].get("text").unwrap().as_str().unwrap();
    assert!(
        output_text.contains("Result"),
        "output should contain 'Result'"
    );
    assert!(
        output_text.contains("12"),
        "output should contain the result value"
    );
}

#[test]
fn test_wire_steps_summary_when_enabled() {
    let output = Command::new(cargo::cargo_bin!("cas_cli"))
        .arg("eval")
        .arg("x^2 + 2*x + 1")
        .arg("--steps")
        .arg("on")
        .arg("--format")
        .arg("json")
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: Value = serde_json::from_str(&stdout).expect("Failed to parse JSON");
    let wire = json.get("wire").expect("wire should exist");

    let messages = wire.get("messages").expect("messages should exist");
    let msgs = messages.as_array().unwrap();

    // If there are steps, should have a steps summary message
    let steps_count = json
        .get("steps_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    if steps_count > 0 {
        let step_msgs: Vec<_> = msgs
            .iter()
            .filter(|m| m.get("kind") == Some(&Value::String("steps".into())))
            .collect();

        assert!(
            !step_msgs.is_empty(),
            "should have steps message when steps_count > 0"
        );
    }
}

#[test]
fn test_wire_message_order() {
    // Test that messages appear in expected order: warn, info, output, steps
    let wire = eval_json_wire("1/x"); // This might produce requires

    let messages = wire.get("messages").expect("messages should exist");
    let msgs = messages.as_array().unwrap();

    // Find position of output message
    let output_pos = msgs
        .iter()
        .position(|m| m.get("kind") == Some(&Value::String("output".into())));

    assert!(output_pos.is_some(), "should have output message");

    // Warn and info should come before output (if present)
    for (i, msg) in msgs.iter().enumerate() {
        let kind = msg.get("kind").and_then(|v| v.as_str()).unwrap_or("");
        if kind == "warn" || kind == "info" {
            assert!(
                i < output_pos.unwrap(),
                "warn/info messages should come before output"
            );
        }
    }
}

#[test]
fn test_wire_schema_version_stable() {
    // Verify schema_version is exactly 1 (not changing unexpectedly)
    for expr in &["1+1", "x^2", "sin(pi/2)"] {
        let wire = eval_json_wire(expr);
        assert_eq!(
            wire.get("schema_version"),
            Some(&Value::Number(1.into())),
            "schema_version should always be 1 for expr: {}",
            expr
        );
    }
}
