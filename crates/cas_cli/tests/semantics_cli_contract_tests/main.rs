//! CLI contract tests for PR1.1 semantic flags.
//!
//! # Contract: Semantic Flags in Wire Output
//!
//! These tests verify that:
//! 1. New flags are reflected in the wire semantics block
//! 2. Defaults are correct (real/strict/principal)

use serde_json::{json, Value};
use std::process::Command;

/// Run the CLI binary directly (not via `cargo run`) for stable test execution.
/// Uses CARGO_BIN_EXE_cas_cli set automatically by Cargo for integration tests.
fn run_cli(args: &[&str]) -> (String, i32) {
    // Get the binary path from the environment variable set by Cargo
    let bin_path = env!("CARGO_BIN_EXE_cas_cli");

    let output = Command::new(bin_path)
        .args(args)
        .output()
        .expect("Failed to execute binary");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let code = output.status.code().unwrap_or(-1);
    (stdout, code)
}

fn parse_wire(s: &str) -> Value {
    // Trim whitespace and find wire JSON content (in case of extra output)
    let trimmed = s.trim();
    serde_json::from_str(trimmed)
        .unwrap_or_else(|e| panic!("Failed to parse wire JSON: {} (error: {})", trimmed, e))
}

fn assert_optional_empty_domain_blocked_hint(wire: &Value, output: &str) {
    let blocked = &wire["blocked_hints"];
    if blocked.is_null() {
        return;
    }

    let blocked = blocked.as_array().expect("blocked_hints array");
    assert_eq!(blocked.len(), 1, "unexpected blocked hints: {output}");
    assert_eq!(blocked[0]["rule"], "Symbolic Differentiation");
    assert_eq!(
        blocked[0]["tip"],
        "real domain is empty; no real derivative is exposed"
    );
}

fn expected_visible_rule_name(rule_name: &str) -> &str {
    match rule_name {
        "Angle Sum/Diff Identity" => "Aplicar suma/diferencia de ángulos",
        "Exponential Power Identity" => "Reescribir potencia exponencial",
        "Exponential Reciprocal Identity" => "Reescribir recíproco exponencial",
        "Exponential Sum/Difference Identity" => "Reescribir exponenciales",
        "Hyperbolic Angle Sum/Difference Identity" => {
            "Aplicar identidad hiperbólica de suma/diferencia de ángulos"
        }
        "Hyperbolic Double-Angle Identity" => "Aplicar identidad hiperbólica de ángulo doble",
        "Hyperbolic Exponential Identity" => "Aplicar identidad exponencial hiperbólica",
        "Hyperbolic Half-Angle Squares" => "Aplicar identidad hiperbólica de ángulo mitad",
        "Hyperbolic Product-to-Sum Identity" => "Aplicar identidad hiperbólica de producto a suma",
        "Hyperbolic Pythagorean Identity" => "Aplicar identidad pitagórica hiperbólica",
        "Hyperbolic Quotient Identity" => "Aplicar identidad hiperbólica de cociente",
        "Hyperbolic Triple-Angle Identity" => "Aplicar identidad hiperbólica de ángulo triple",
        "Cancel Equal Fractions Difference" => "Cancelar fracciones iguales",
        "Cancel Exact Additive Pairs" => "Cancelar términos opuestos",
        "Quintuple Angle Identity" => "Reescribir ángulo quíntuple",
        "Triple Angle Expansion" | "Triple Angle Identity" => "Reescribir ángulo triple",
        "Tan to Sin/Cos" => "Expandir tangente como seno entre coseno",
        "Secant to Reciprocal Cosine" => "Expandir secante como recíproco de coseno",
        "Cosecant to Reciprocal Sine" => "Expandir cosecante como recíproco de seno",
        "Cotangent to Cosine over Sine" => "Expandir cotangente como coseno entre seno",
        _ => rule_name,
    }
}

fn assert_rule_eq(rule: &Value, expected: &str) {
    let actual = rule.as_str().expect("rule string");
    let expected_visible = expected_visible_rule_name(expected);
    assert_eq!(actual, expected_visible);
}

fn assert_rule_matches_any(rule: &Value, expected: &[&str]) {
    let actual = rule.as_str().expect("rule string");
    let expected_visible: Vec<_> = expected
        .iter()
        .map(|rule_name| expected_visible_rule_name(rule_name))
        .collect();
    assert!(
        expected_visible.contains(&actual),
        "unexpected rule {actual:?}, expected one of {expected_visible:?}"
    );
}

// =============================================================================
// Semantic Flags Tests
// =============================================================================

// =============================================================================
// AssumeScope Tests (PR-SCOPE-1)
// =============================================================================

// =============================================================================
// ConstFold + Complex semantics regression tests
// =============================================================================

fn assert_single_named_derive_step(expr: &str, expected_strategy: &str, expected_rule: &str) {
    let (output, _code) = run_cli(&["eval", expr, "--format", "json", "--steps", "on"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], expected_strategy);
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], expected_rule);
}

mod derive_core;
mod derive_explog;
mod derive_hyperbolic;
mod derive_radicals;
mod derive_rational;
mod derive_trig;
mod eval_core;
mod eval_explog;
mod eval_hyperbolic;
mod eval_radicals;
mod eval_rational;
mod eval_steps;
mod eval_trig;
mod misc_complex;
mod misc_core;
mod misc_domain;
mod misc_explog;
mod misc_numeric;
mod misc_radicals;
mod misc_trig;
