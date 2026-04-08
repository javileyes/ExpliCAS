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

// =============================================================================
// Semantic Flags Tests
// =============================================================================

#[test]
fn semantics_block_present_in_json() {
    let (output, _code) = run_cli(&["eval", "1+1", "--format", "json"]);
    let wire = parse_wire(&output);

    assert!(
        wire.get("semantics").is_some(),
        "Wire output should have 'semantics' field"
    );
}

#[test]
fn semantics_defaults_reflected() {
    let (output, _code) = run_cli(&["eval", "1+1", "--format", "json"]);
    let wire = parse_wire(&output);

    let semantics = &wire["semantics"];
    assert_eq!(semantics["domain_mode"], "generic");
    assert_eq!(semantics["value_domain"], "real");
    assert_eq!(semantics["inv_trig"], "strict");
    assert_eq!(semantics["branch"], "principal");
    assert_eq!(semantics["assume_scope"], "real");
}

#[test]
fn value_domain_complex_reflected() {
    let (output, _code) = run_cli(&[
        "eval",
        "1+1",
        "--format",
        "json",
        "--value-domain",
        "complex",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["semantics"]["value_domain"], "complex");
}

#[test]
fn inv_trig_principal_reflected() {
    let (output, _code) = run_cli(&["eval", "1+1", "--format", "json", "--inv-trig", "principal"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["semantics"]["inv_trig"], "principal");
}

#[test]
fn domain_strict_with_semantics() {
    let (output, _code) = run_cli(&["eval", "1+1", "--format", "json", "--domain", "strict"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["semantics"]["domain_mode"], "strict");
    assert_eq!(wire["domain"]["mode"], "strict");
}

// =============================================================================
// AssumeScope Tests (PR-SCOPE-1)
// =============================================================================

#[test]
fn assume_scope_default_reflected() {
    let (output, _code) = run_cli(&["eval", "1+1", "--format", "json"]);
    let wire = parse_wire(&output);

    assert_eq!(
        wire["semantics"]["assume_scope"], "real",
        "assume_scope default should be 'real'"
    );
}

#[test]
fn assume_scope_wildcard_flag_reflected() {
    let (output, _code) = run_cli(&[
        "eval",
        "1+1",
        "--format",
        "json",
        "--assume-scope",
        "wildcard",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(
        wire["semantics"]["assume_scope"], "wildcard",
        "--assume-scope wildcard should be reflected in wire output"
    );
}

#[test]
fn assume_scope_flag_does_not_change_result() {
    // Infrastructure-only: changing assume_scope should NOT change result
    // (behavior changes come in PR-SCOPE-3)
    let (output1, _) = run_cli(&["eval", "x/x", "--format", "json", "--domain", "generic"]);
    let (output2, _) = run_cli(&[
        "eval",
        "x/x",
        "--format",
        "json",
        "--domain",
        "generic",
        "--assume-scope",
        "wildcard",
    ]);

    let wire1 = parse_wire(&output1);
    let wire2 = parse_wire(&output2);

    assert_eq!(
        wire1["result"], wire2["result"],
        "assume_scope flag should not change result (infra only)"
    );
}

// =============================================================================
// ConstFold + Complex semantics regression tests
// =============================================================================

#[test]
fn const_fold_safe_complex_sqrt_negative_reaches_cli_runtime() {
    let (output, _code) = run_cli(&[
        "eval",
        "sqrt(-1)",
        "--format",
        "json",
        "--value-domain",
        "complex",
        "--complex",
        "on",
        "--const-fold",
        "safe",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["options"]["const_fold"], "safe");
    assert_eq!(wire["semantics"]["value_domain"], "complex");
    assert_eq!(wire["result"], "i");
    assert_eq!(
        wire["warnings"].as_array().map(Vec::len),
        Some(0),
        "complex+safe should not emit imaginary-usage warning"
    );
}

#[test]
fn const_fold_safe_real_sqrt_negative_keeps_warning_contract() {
    let (output, _code) = run_cli(&[
        "eval",
        "sqrt(-1)",
        "--format",
        "json",
        "--value-domain",
        "real",
        "--complex",
        "on",
        "--const-fold",
        "safe",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["options"]["const_fold"], "safe");
    assert_eq!(wire["semantics"]["value_domain"], "real");
    assert_eq!(wire["result"], "undefined");

    let warnings = wire["warnings"].as_array().expect("warnings array");
    assert_eq!(warnings.len(), 1, "real+safe should emit one warning");
    assert_eq!(warnings[0]["rule"], "Imaginary Usage Warning");
}

#[test]
fn const_fold_safe_complex_i_squared_reaches_cli_runtime() {
    let (output, _code) = run_cli(&[
        "eval",
        "i^2",
        "--format",
        "json",
        "--value-domain",
        "complex",
        "--complex",
        "on",
        "--const-fold",
        "safe",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["options"]["const_fold"], "safe");
    assert_eq!(wire["semantics"]["value_domain"], "complex");
    assert_eq!(wire["result"], "-1");
}

#[test]
fn subtraction_self_cancel_shortcut_handles_abs_sub_mirror_runtime_shape() {
    let (output, _code) = run_cli(&[
        "eval",
        "abs((2*u)/(u^2 - 1) - 1) - abs(1 - 2*u/(u^2 - 1))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps[0]["rule"], "Restar dos expresiones iguales");
}

#[test]
fn standalone_trig_square_cube_quotient_reaches_cli_eval_path() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sin(u)^2)^3 - 1)/((sin(u)^2) - 1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "1 + sin(u)^2 + sin(u)^2^2");

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        required
            .iter()
            .filter_map(|item| item.as_str())
            .any(|item| item == "sin(u) - 1 ≠ 0"),
        "expected atomic factor guard in required_display"
    );
    assert!(
        required
            .iter()
            .filter_map(|item| item.as_str())
            .any(|item| item == "sin(u) + 1 ≠ 0"),
        "expected complementary atomic factor guard in required_display"
    );
}

#[test]
fn same_root_family_power_quotient_reaches_cli_eval_path_exactly() {
    let (output, _code) = run_cli(&[
        "eval",
        "(sqrt(x^2 + 1)^5)/(sqrt(x^2 + 1)^3)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "x^2 + 1");
    assert_eq!(wire["required_display"], json!([]));
}

#[test]
fn trig_square_cube_substitution_difference_reaches_cli_eval_path() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sin(u)^2)^3 - 1)/((sin(u)^2) - 1) - ((sin(u)^2)^2 + (sin(u)^2) + 1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        required
            .iter()
            .filter_map(|item| item.as_str())
            .any(|item| item == "sin(u) - 1 ≠ 0"),
        "expected atomic factor guard in required_display"
    );
    assert!(
        required
            .iter()
            .filter_map(|item| item.as_str())
            .any(|item| item == "sin(u) + 1 ≠ 0"),
        "expected complementary atomic factor guard in required_display"
    );
}

#[test]
fn root_ctx_exact_quotient_survives_runtime_before_conjugate_rationalization() {
    let (output, _code) = run_cli(&[
        "eval",
        "((1/sqrt(u))^2 + 2*(1/sqrt(u)) + 1)/((1/sqrt(u)) + 1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "u^(-1/2) + 1");

    let steps = wire["steps"].as_array().expect("steps array");
    let exact_step = steps
        .iter()
        .find(|step| step["rule"] == "Racionalizar el denominador")
        .expect("exact quotient step");
    assert_eq!(exact_step["after"], "1/sqrt(u) + 1");

    let required = wire["required_display"]
        .as_array()
        .expect("required display");
    assert_eq!(
        required
            .iter()
            .filter_map(|item| item.as_str())
            .collect::<Vec<_>>(),
        vec!["u > 0"],
        "runtime should compress reciprocal-sqrt guards to a single positivity guard"
    );
}

#[test]
fn abs_sqrt_denominator_collapses_to_positive_base_guard() {
    let (output, _code) = run_cli(&[
        "eval",
        "1/abs(sqrt(u))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "u^(-1/2)");

    let required = wire["required_display"]
        .as_array()
        .expect("required display");
    assert_eq!(
        required
            .iter()
            .filter_map(|item| item.as_str())
            .collect::<Vec<_>>(),
        vec!["u > 0"],
        "abs(sqrt(u)) denominator should compress to a single positivity guard"
    );
}

#[test]
fn derive_log_abs_sqrt_collapses_to_positive_base_guard() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive ln(abs(sqrt(u))), (1/2)*ln(u)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "(1·ln(u))/2");

    let required = wire["required_display"]
        .as_array()
        .expect("required display");
    assert_eq!(
        required
            .iter()
            .filter_map(|item| item.as_str())
            .collect::<Vec<_>>(),
        vec!["u > 0"],
        "ln(abs(sqrt(u))) should not keep a redundant sqrt(u) != 0 guard"
    );
}

#[test]
fn eval_log_abs_sqrt_of_intrinsically_positive_argument_emits_no_warning() {
    let (output, _code) = run_cli(&["eval", "ln(abs(sqrt(x^2+1)))", "--format", "json"]);
    let wire = parse_wire(&output);

    let warnings = wire["warnings"].as_array().expect("warnings array");
    assert!(
        warnings.is_empty(),
        "intrinsically positive log argument should not emit a warning"
    );

    let required = wire["required_display"]
        .as_array()
        .expect("required display");
    assert!(
        required.is_empty(),
        "intrinsically positive log argument should not surface display guards"
    );
}

#[test]
fn eval_general_base_log_abs_sqrt_of_intrinsically_positive_argument_keeps_only_base_requires() {
    let (output, _code) = run_cli(&["eval", "log(b, abs(sqrt(x^2+1)))", "--format", "json"]);
    let wire = parse_wire(&output);

    let warnings = wire["warnings"].as_array().expect("warnings array");
    assert!(
        warnings.is_empty(),
        "intrinsically positive general-base log argument should not emit a warning"
    );

    let required = wire["required_display"]
        .as_array()
        .expect("required display");
    assert_eq!(
        required
            .iter()
            .filter_map(|item| item.as_str())
            .collect::<Vec<_>>(),
        vec!["b - 1 ≠ 0", "b > 0"]
    );
}

#[test]
fn eval_general_base_log_sqrt_keeps_nontrivial_positive_argument_warning() {
    let (output, _code) = run_cli(&["eval", "log(b, sqrt(u))", "--format", "json"]);
    let wire = parse_wire(&output);

    let warnings = wire["warnings"].as_array().expect("warnings array");
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0]["rule"], "Evaluate Logarithms");
    assert_eq!(warnings[0]["assumption"], "u > 0");

    let required = wire["required_display"]
        .as_array()
        .expect("required display");
    assert_eq!(
        required
            .iter()
            .filter_map(|item| item.as_str())
            .collect::<Vec<_>>(),
        vec!["b - 1 ≠ 0", "b > 0", "u > 0"]
    );
}

#[test]
fn derive_root_nesting_keeps_wire_steps_when_steps_count_is_positive() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sqrt(2)+sqrt(3), sqrt(5+2*sqrt(6))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "sqrt(2·sqrt(6) + 5)");
    assert_eq!(wire["steps_count"], 2);

    let steps = wire["steps"].as_array().expect("steps array");
    assert!(
        !steps.is_empty(),
        "expected wire steps for derive root nesting when steps_count is positive"
    );
}

#[test]
fn derive_json_exposes_strategy_metadata() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(2*x)/cos(x+x), tan(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");

    let messages = wire["wire"]["messages"].as_array().expect("wire messages");
    assert!(
        messages
            .iter()
            .any(|message| message["text"] == "Strategy: contract trig"),
        "expected nested wire envelope to include derive strategy metadata"
    );
    assert!(
        messages
            .iter()
            .any(|message| message["text"] == "1 step(s) via contract trig"),
        "expected nested wire envelope to describe derive steps using the active strategy"
    );
}

#[test]
fn derive_double_angle_after_arg_simplify_uses_direct_expand_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x+x), 2*sin(x)*cos(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir ángulo doble");
}

#[test]
fn derive_mixed_root_and_symbolic_power_uses_single_combine_powers_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sqrt(x)*x^a, x^(a+1/2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "combine powers");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Sumar exponentes de la misma base");
}

#[test]
fn derive_cos_diff_over_sin_diff_quotient_uses_direct_contract_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (cos(x)-cos(3*x))/(sin(3*x)-sin(x)), tan(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Convertir un cociente trigonométrico en tangente"
    );
}

#[test]
fn derive_phase_shift_sum_to_shifted_sine_uses_single_contract_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)+cos(x), sqrt(2)*sin(x+pi/4)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de desfase");
}

#[test]
fn derive_phase_shift_shifted_sine_to_sum_uses_single_expand_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sqrt(2)*sin(x+pi/4), sin(x)+cos(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de desfase");
}

#[test]
fn derive_phase_shift_shifted_cosine_to_sum_uses_single_expand_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sqrt(2)*cos(x-pi/4), sin(x)+cos(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de desfase");
}

#[test]
fn derive_scaled_phase_shift_sum_to_shifted_sine_uses_single_contract_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sin(x)+2*cos(x), 2*sqrt(2)*sin(x+pi/4)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de desfase");
}

#[test]
fn derive_scaled_phase_shift_shifted_sine_to_sum_uses_single_expand_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sqrt(2)*sin(x+pi/4), 2*sin(x)+2*cos(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de desfase");
}

#[test]
fn derive_exact_third_phase_shift_sum_to_shifted_sine_uses_single_contract_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sin(x)+2*sqrt(3)*cos(x), 4*sin(x+pi/3)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de desfase");
}

#[test]
fn derive_exact_third_phase_shift_shifted_sine_to_sum_uses_single_expand_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 4*sin(x+pi/3), 2*sin(x)+2*sqrt(3)*cos(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de desfase");
}

#[test]
fn derive_exact_sixth_phase_shift_sum_to_shifted_sine_uses_single_contract_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sqrt(3)*sin(x)+cos(x), 2*sin(x+pi/6)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de desfase");
}

#[test]
fn derive_exact_sixth_phase_shift_shifted_sine_to_sum_uses_single_expand_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sin(x+pi/6), sqrt(3)*sin(x)+cos(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de desfase");
}

#[test]
fn derive_general_phase_shift_sum_to_shifted_sine_uses_single_contract_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 3*sin(x)+4*cos(x), 5*sin(x+arctan(4/3))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de desfase");
}

#[test]
fn derive_general_phase_shift_shifted_sine_to_sum_uses_single_expand_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 5*sin(x+arctan(4/3)), 3*sin(x)+4*cos(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de desfase");
}

#[test]
fn derive_phase_shift_shifted_sine_to_shifted_cosine_uses_single_contract_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sqrt(2)*sin(x+pi/4), sqrt(2)*cos(x-pi/4)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de desfase");
}

#[test]
fn derive_phase_shift_shifted_terms_with_passthrough_uses_single_contract_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sqrt(2)*sin(x+pi/4)+a, sqrt(2)*cos(x-pi/4)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de desfase");
}

#[test]
fn derive_general_phase_shift_shifted_sine_to_shifted_cosine_uses_single_contract_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 5*sin(x+arctan(4/3)), 5*cos(x-arctan(3/4))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de desfase");
}

#[test]
fn derive_general_phase_shift_shifted_terms_with_passthrough_uses_single_contract_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 5*sin(x+arctan(4/3))+a, 5*cos(x-arctan(3/4))+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de desfase");
}

#[test]
fn derive_phase_shift_sum_with_passthrough_uses_single_contract_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)+cos(x)+a, sqrt(2)*sin(x+pi/4)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de desfase");
}

#[test]
fn derive_scaled_phase_shift_shifted_sine_with_passthrough_to_sum_uses_single_expand_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sqrt(2)*sin(x+pi/4)+a, 2*sin(x)+2*cos(x)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de desfase");
}

#[test]
fn derive_general_phase_shift_sum_with_passthrough_uses_single_contract_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 3*sin(x)+4*cos(x)+a, 5*sin(x+arctan(4/3))+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de desfase");
}

#[test]
fn derive_general_phase_shift_shifted_sine_with_passthrough_to_sum_uses_single_expand_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 5*sin(x+arctan(4/3))+a, 3*sin(x)+4*cos(x)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de desfase");
}

#[test]
fn derive_repeated_phase_shift_sum_uses_two_direct_expand_phase_shift_steps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)+cos(x)+sin(y)+cos(y), sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 2);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert!(steps
        .iter()
        .all(|step| step["rule"] == "Aplicar identidad de desfase"));
}

#[test]
fn derive_repeated_phase_shift_sum_expansion_uses_two_direct_expand_phase_shift_steps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4), sin(x)+cos(x)+sin(y)+cos(y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 2);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert!(steps
        .iter()
        .all(|step| step["rule"] == "Aplicar identidad de desfase"));
}

#[test]
fn derive_hyperbolic_double_angle_avoids_generic_canonicalize_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sinh(x)*cosh(x), sinh(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_ne!(
        steps[0]["rule"],
        "Canonicalize Multiplication",
        "hyperbolic double-angle contraction should not fall back to a generic canonicalization step"
    );
}

#[test]
fn derive_hyperbolic_sum_to_product_prunes_canonicalize_multiplication_tail() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sinh(x)+sinh(y), 2*sinh((x+y)/2)*cosh((x-y)/2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Product-to-Sum Identity");
}

#[test]
fn derive_sophie_germain_expansion_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (x^2 - 2*x*y + 2*y^2)*(x^2 + 2*x*y + 2*y^2), x^4 + 4*y^4",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir la expresión");
}

#[test]
fn derive_hyperbolic_half_angle_backward_avoids_generic_canonicalize_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (cosh(x)-1)/2, sinh(x/2)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Half-Angle Squares");
}

#[test]
fn derive_negative_hyperbolic_cosh_half_angle_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(cosh(x)+1)/2, -cosh(x/2)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Half-Angle Squares");
}

#[test]
fn derive_negative_hyperbolic_sinh_half_angle_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(cosh(x)-1)/2, -sinh(x/2)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Half-Angle Squares");
}

#[test]
fn derive_negative_hyperbolic_cosh_half_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cosh(x/2)^2, -(cosh(x)+1)/2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Half-Angle Squares");
}

#[test]
fn derive_negative_hyperbolic_sinh_half_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -sinh(x/2)^2, -(cosh(x)-1)/2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Half-Angle Squares");
}

#[test]
fn derive_hyperbolic_cosh_double_angle_variant_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cosh(2*x), 2*cosh(x)^2-1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["strategy"], "rewrite hyperbolics");

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_hyperbolic_expansion_from_exp_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x), sinh(x)+cosh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Exponential Identity");
}

#[test]
fn derive_hyperbolic_expansion_from_negative_exp_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(-x), cosh(x)-sinh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Exponential Reciprocal Identity");
}

#[test]
fn derive_hyperbolic_expansion_from_negated_negative_exp_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -exp(-x), sinh(x)-cosh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Exponential Reciprocal Identity");
}

#[test]
fn derive_hyperbolic_tanh_pythagorean_backward_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1/cosh(x)^2, 1-tanh(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Pythagorean Identity");
}

#[test]
fn derive_negative_hyperbolic_tanh_pythagorean_forward_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive tanh(x)^2-1, -1/cosh(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Pythagorean Identity");
}

#[test]
fn derive_negative_hyperbolic_tanh_pythagorean_backward_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -1/cosh(x)^2, tanh(x)^2-1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Pythagorean Identity");
}

#[test]
fn derive_hyperbolic_sinh_definition_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sinh(x), (e^x-e^(-x))/2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Exponential Identity");
}

#[test]
fn derive_negative_hyperbolic_cosh_definition_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(e^x+e^(-x))/2, -cosh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Exponential Identity");
}

#[test]
fn derive_negative_hyperbolic_cosh_definition_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cosh(x), -(e^x+e^(-x))/2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Exponential Identity");
}

#[test]
fn derive_hyperbolic_tanh_definition_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive tanh(x), (e^x-e^(-x))/(e^x+e^(-x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Exponential Identity");
}

#[test]
fn derive_negative_hyperbolic_tanh_quotient_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -sinh(x)/cosh(x), -tanh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Quotient Identity");
}

#[test]
fn derive_negative_hyperbolic_tanh_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -tanh(x), -sinh(x)/cosh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Quotient Identity");
}

#[test]
fn derive_negative_hyperbolic_sinh_double_angle_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -2*sinh(x)*cosh(x), -sinh(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_negative_hyperbolic_sinh_double_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -sinh(2*x), -2*sinh(x)*cosh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_product_to_sum_expansion_does_not_emit_depth_overflow_warning_to_stderr() {
    let bin_path = env!("CARGO_BIN_EXE_cas_cli");
    let output = Command::new(bin_path)
        .args([
            "eval",
            "derive 2*sin(2*x)*cos(x), sin(3*x)+sin(x)",
            "--format",
            "json",
            "--steps",
            "on",
        ])
        .output()
        .expect("Failed to execute binary");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let wire = parse_wire(&stdout);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar producto a suma");
    assert!(
        !stderr.contains("depth_overflow"),
        "expected direct product-to-sum expansion to stay quiet on stderr, got: {stderr}"
    );
}

#[test]
fn derive_cosine_product_to_sum_expansion_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*cos(2*x)*cos(x), cos(3*x)+cos(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar producto a suma");
}

#[test]
fn derive_sine_difference_product_to_sum_expansion_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sin(2*x)*sin(x), cos(x)-cos(3*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar producto a suma");
}

#[test]
fn derive_general_sine_sum_to_product_expansion_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)+sin(y), 2*sin((x+y)/2)*cos((x-y)/2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar suma a producto");
}

#[test]
fn derive_general_cosine_sum_to_product_expansion_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(x)+cos(y), 2*cos((x+y)/2)*cos((x-y)/2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar suma a producto");
}

#[test]
fn derive_general_cosine_difference_sum_to_product_expansion_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(x)-cos(y), -2*sin((x+y)/2)*sin((x-y)/2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar suma a producto");
}

#[test]
fn derive_sine_fourth_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)^4, (3-4*cos(2*x)+cos(4*x))/8",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_cosine_fourth_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(x)^4, (3+4*cos(2*x)+cos(4*x))/8",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_sine_cosine_square_product_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)^2*cos(x)^2, (1-cos(4*x))/8",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_sine_plus_cosine_square_identity_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (sin(x)+cos(x))^2, 1+sin(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Aplicar identidad del cuadrado trigonométrico"
    );
}

#[test]
fn derive_sine_sixth_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)^6, (10-15*cos(2*x)+6*cos(4*x)-cos(6*x))/32",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_cosine_sixth_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(x)^6, (10+15*cos(2*x)+6*cos(4*x)+cos(6*x))/32",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_sine_eighth_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)^8, (35-56*cos(2*x)+28*cos(4*x)-8*cos(6*x)+cos(8*x))/128",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_cosine_eighth_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(x)^8, (35+56*cos(2*x)+28*cos(4*x)+8*cos(6*x)+cos(8*x))/128",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_sine_tenth_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)^10, (126-210*cos(2*x)+120*cos(4*x)-45*cos(6*x)+10*cos(8*x)-cos(10*x))/512",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_cosine_tenth_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(x)^10, (126+210*cos(2*x)+120*cos(4*x)+45*cos(6*x)+10*cos(8*x)+cos(10*x))/512",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_sine_twelfth_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)^12, (462-792*cos(2*x)+495*cos(4*x)-220*cos(6*x)+66*cos(8*x)-12*cos(10*x)+cos(12*x))/2048",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_cosine_twelfth_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(x)^12, (462+792*cos(2*x)+495*cos(4*x)+220*cos(6*x)+66*cos(8*x)+12*cos(10*x)+cos(12*x))/2048",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_sine_fourteenth_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)^14, (1716-3003*cos(2*x)+2002*cos(4*x)-1001*cos(6*x)+364*cos(8*x)-91*cos(10*x)+14*cos(12*x)-cos(14*x))/8192",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_cosine_fourteenth_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(x)^14, (1716+3003*cos(2*x)+2002*cos(4*x)+1001*cos(6*x)+364*cos(8*x)+91*cos(10*x)+14*cos(12*x)+cos(14*x))/8192",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_sine_sixteenth_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)^16, (6435-11440*cos(2*x)+8008*cos(4*x)-4368*cos(6*x)+1820*cos(8*x)-560*cos(10*x)+120*cos(12*x)-16*cos(14*x)+cos(16*x))/32768",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_cosine_sixteenth_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(x)^16, (6435+11440*cos(2*x)+8008*cos(4*x)+4368*cos(6*x)+1820*cos(8*x)+560*cos(10*x)+120*cos(12*x)+16*cos(14*x)+cos(16*x))/32768",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_sine_eighteenth_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)^18, (24310-43758*cos(2*x)+31824*cos(4*x)-18564*cos(6*x)+8568*cos(8*x)-3060*cos(10*x)+816*cos(12*x)-153*cos(14*x)+18*cos(16*x)-cos(18*x))/131072",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_cosine_eighteenth_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(x)^18, (24310+43758*cos(2*x)+31824*cos(4*x)+18564*cos(6*x)+8568*cos(8*x)+3060*cos(10*x)+816*cos(12*x)+153*cos(14*x)+18*cos(16*x)+cos(18*x))/131072",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_sine_twentieth_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)^20, (92378-167960*cos(2*x)+125970*cos(4*x)-77520*cos(6*x)+38760*cos(8*x)-15504*cos(10*x)+4845*cos(12*x)-1140*cos(14*x)+190*cos(16*x)-20*cos(18*x)+cos(20*x))/524288",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_cosine_twentieth_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(x)^20, (92378+167960*cos(2*x)+125970*cos(4*x)+77520*cos(6*x)+38760*cos(8*x)+15504*cos(10*x)+4845*cos(12*x)+1140*cos(14*x)+190*cos(16*x)+20*cos(18*x)+cos(20*x))/524288",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_sine_twenty_second_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)^22, (352716-646646*cos(2*x)+497420*cos(4*x)-319770*cos(6*x)+170544*cos(8*x)-74613*cos(10*x)+26334*cos(12*x)-7315*cos(14*x)+1540*cos(16*x)-231*cos(18*x)+22*cos(20*x)-cos(22*x))/2097152",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_cosine_twenty_second_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(x)^22, (352716+646646*cos(2*x)+497420*cos(4*x)+319770*cos(6*x)+170544*cos(8*x)+74613*cos(10*x)+26334*cos(12*x)+7315*cos(14*x)+1540*cos(16*x)+231*cos(18*x)+22*cos(20*x)+cos(22*x))/2097152",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_sine_twenty_fourth_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)^24, (1352078-2496144*cos(2*x)+1961256*cos(4*x)-1307504*cos(6*x)+735471*cos(8*x)-346104*cos(10*x)+134596*cos(12*x)-42504*cos(14*x)+10626*cos(16*x)-2024*cos(18*x)+276*cos(20*x)-24*cos(22*x)+cos(24*x))/8388608",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_cosine_twenty_fourth_power_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(x)^24, (1352078+2496144*cos(2*x)+1961256*cos(4*x)+1307504*cos(6*x)+735471*cos(8*x)+346104*cos(10*x)+134596*cos(12*x)+42504*cos(14*x)+10626*cos(16*x)+2024*cos(18*x)+276*cos(20*x)+24*cos(22*x)+cos(24*x))/8388608",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar reducción de potencias");
}

#[test]
fn derive_sine_minus_cosine_square_identity_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (sin(x)-cos(x))^2, 1-sin(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Aplicar identidad del cuadrado trigonométrico"
    );
}

#[test]
fn derive_trig_expand_steps_count_matches_visible_steps_for_trig_polynomial_target() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sin(2*x)*sin(x), 4*cos(x)-4*cos(x)^3",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(wire["steps_count"].as_u64(), Some(steps.len() as u64));
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["rule"], "Aplicar producto a suma");
    assert_eq!(steps[1]["rule"], "Triple Angle Expansion");
}

#[test]
fn derive_hyperbolic_cosh_double_angle_expansion_to_sinh_mixed_polynomial_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*cosh(2*x)*sinh(x), 4*cosh(x)^2*sinh(x)-2*sinh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_hyperbolic_cosh_double_angle_expansion_to_cosh_mixed_polynomial_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*cosh(2*x)*cosh(x), 2*cosh(x)+4*sinh(x)^2*cosh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_hyperbolic_cosh_double_angle_contraction_from_sinh_mixed_polynomial_uses_single_named_step(
) {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 4*cosh(x)^2*sinh(x)-2*sinh(x), 2*cosh(2*x)*sinh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_hyperbolic_cosh_double_angle_contraction_from_cosh_mixed_polynomial_uses_single_named_step(
) {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*cosh(x)+4*sinh(x)^2*cosh(x), 2*cosh(2*x)*cosh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_hyperbolic_product_to_sum_expansion_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sinh(2*x)*cosh(x), sinh(3*x)+sinh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Product-to-Sum Identity");
}

#[test]
fn derive_hyperbolic_sum_to_product_contraction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sinh(3*x)-sinh(x), 2*cosh(2*x)*sinh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Product-to-Sum Identity");
}

#[test]
fn derive_binomial_expansion_with_cancellation_uses_expand_strategy() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (a+b)^2 - a^2 - 2*a*b, b^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand");
    let steps = wire["steps"].as_array().expect("steps array");
    assert!(!steps.is_empty());
    assert_eq!(steps[0]["rule"], "Expandir binomio");
}

#[test]
fn derive_hyperbolic_product_to_sum_with_passthrough_uses_two_expand_steps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sinh(2*x)*sinh(x)+a, 4*cosh(x)^3-4*cosh(x)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand");
    assert_eq!(wire["steps_count"], 2);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["rule"], "Hyperbolic Product-to-Sum Identity");
    assert_eq!(steps[1]["rule"], "Hyperbolic Triple-Angle Identity");
}

#[test]
fn derive_hyperbolic_product_to_sum_polynomial_uses_two_expand_steps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sinh(2*x)*sinh(x), 4*cosh(x)^3-4*cosh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand");
    assert_eq!(wire["steps_count"], 2);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["rule"], "Hyperbolic Product-to-Sum Identity");
    assert_eq!(steps[1]["rule"], "Hyperbolic Triple-Angle Identity");
}

#[test]
fn derive_mixed_trig_double_angle_product_uses_named_step_without_depth_warning() {
    let bin_path = env!("CARGO_BIN_EXE_cas_cli");
    let output = Command::new(bin_path)
        .args([
            "eval",
            "derive 4*sin(x)^2*cos(x), 2*sin(2*x)*sin(x)",
            "--format",
            "json",
            "--steps",
            "on",
        ])
        .output()
        .expect("Failed to execute binary");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let wire = parse_wire(&stdout);

    assert_eq!(wire["strategy"], "contract trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir ángulo doble");
    assert!(
        !stderr.contains("depth_overflow"),
        "expected target-aware double-angle path to stay quiet on stderr, got: {stderr}"
    );
}

#[test]
fn derive_mixed_hyperbolic_double_angle_product_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 4*sinh(x)^2*cosh(x), 2*sinh(2*x)*sinh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_trig_product_to_sum_mixed_cos_square_polynomial_uses_expand_trig() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*cos(2*x)*sin(x), 4*cos(x)^2*sin(x)-2*sin(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 2);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["rule"], "Aplicar producto a suma");
    assert_eq!(steps[1]["rule"], "Triple Angle Expansion");
}

#[test]
fn derive_trig_product_to_sum_cosine_difference_polynomial_with_passthrough_uses_expand_trig() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sin(2*x)*sin(x)+a, 4*cos(x)-4*cos(x)^3+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 2);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["rule"], "Aplicar producto a suma");
    assert_eq!(steps[1]["rule"], "Triple Angle Expansion");
}

#[test]
fn derive_trig_product_to_sum_mixed_cos_square_polynomial_with_passthrough_uses_expand_trig() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*cos(2*x)*sin(x)+a, 4*cos(x)^2*sin(x)-2*sin(x)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 2);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["rule"], "Aplicar producto a suma");
    assert_eq!(steps[1]["rule"], "Triple Angle Expansion");
}

#[test]
fn derive_mixed_trig_double_angle_expansion_to_sin_square_polynomial_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sin(2*x)*sin(x), 4*sin(x)^2*cos(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir ángulo doble");
}

#[test]
fn derive_mixed_trig_double_angle_expansion_to_cos_square_polynomial_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sin(2*x)*cos(x), 4*cos(x)^2*sin(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir ángulo doble");
}

#[test]
fn derive_negative_hyperbolic_tanh_double_angle_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -2*tanh(x)/(1+tanh(x)^2), -tanh(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_negative_hyperbolic_tanh_double_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -tanh(2*x), -2*tanh(x)/(1+tanh(x)^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_negative_hyperbolic_sinh_triple_angle_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(3*sinh(x)+4*sinh(x)^3), -sinh(3*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Triple-Angle Identity");
}

#[test]
fn derive_negative_hyperbolic_sinh_triple_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -sinh(3*x), -(3*sinh(x)+4*sinh(x)^3)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Triple-Angle Identity");
}

#[test]
fn derive_negative_hyperbolic_cosh_triple_angle_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(4*cosh(x)^3-3*cosh(x)), -cosh(3*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Triple-Angle Identity");
}

#[test]
fn derive_negative_hyperbolic_cosh_triple_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cosh(3*x), -(4*cosh(x)^3-3*cosh(x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Triple-Angle Identity");
}

#[test]
fn derive_hyperbolic_shifted_pythagorean_forward_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cosh(x)^2-1, sinh(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Pythagorean Identity");
}

#[test]
fn derive_hyperbolic_shifted_pythagorean_add_backward_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cosh(x)^2, 1+sinh(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Pythagorean Identity");
}

#[test]
fn derive_hyperbolic_shifted_double_angle_minus_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cosh(2*x)-1, 2*sinh(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_hyperbolic_negative_shifted_double_angle_minus_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1-cosh(2*x), -2*sinh(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_hyperbolic_negative_shifted_double_angle_minus_backward_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -2*sinh(x)^2, 1-cosh(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_hyperbolic_negative_shifted_double_angle_plus_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -1-cosh(2*x), -2*cosh(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_hyperbolic_negative_shifted_double_angle_plus_backward_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -2*cosh(x)^2, -1-cosh(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_hyperbolic_negative_double_angle_cosh_sq_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1-2*cosh(x)^2, -cosh(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_hyperbolic_negative_double_angle_cosh_sq_backward_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cosh(2*x), 1-2*cosh(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_hyperbolic_shifted_double_angle_plus_backward_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*cosh(x)^2, cosh(2*x)+1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_hyperbolic_double_angle_two_cosh_sq_minus_one_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*cosh(x)^2-1, cosh(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_hyperbolic_double_angle_two_sinh_sq_plus_one_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sinh(x)^2+1, cosh(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_negative_sine_squared_uses_single_pythagorean_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -sin(x)^2, cos(x)^2-1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad pitagórica");
}

#[test]
fn derive_negative_cos_squared_uses_single_pythagorean_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cos(x)^2, sin(x)^2-1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad pitagórica");
}

#[test]
fn eval_fraction_sum_to_sec_squared_keeps_faithful_pythagorean_intermediate() {
    let (output, _code) = run_cli(&[
        "eval",
        "1/(1 + sin(x)) + 1/(1 - sin(x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[1]["rule"], "Aplicar identidad pitagórica");
    assert_eq!(steps[1]["before"], "2/(1 - sin(x)^2)");
    assert!(
        steps[1]["before_latex"]
            .as_str()
            .expect("before latex")
            .contains("^{2}"),
        "expected squared sine to survive in before_latex: {:?}",
        steps[1]["before_latex"]
    );
}

#[test]
fn derive_fraction_sum_to_sec_squared_keeps_faithful_pythagorean_intermediate() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1/(1 + sin(x)) + 1/(1 - sin(x)), 2*sec(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[1]["rule"], "Aplicar identidad pitagórica");
    assert_eq!(steps[1]["before"], "2/(1 - sin(x)^2)");
    assert!(
        steps[1]["before_latex"]
            .as_str()
            .expect("before latex")
            .contains("^{2}"),
        "expected squared sine to survive in before_latex: {:?}",
        steps[1]["before_latex"]
    );
}

#[test]
fn derive_negative_tan_quotient_uses_single_trig_quotient_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -sin(x)/cos(x), -tan(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Convertir un cociente trigonométrico en tangente"
    );
}

#[test]
fn derive_negative_tan_expansion_uses_single_trig_expansion_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -tan(x), -sin(x)/cos(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir una identidad trigonométrica");
}

#[test]
fn derive_combine_like_terms_uses_named_strategy() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive x + x, 2*x",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "combine like terms");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Agrupar términos semejantes");
}

#[test]
fn derive_nested_fraction_one_over_sum_uses_named_strategy() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1/(1/a + 1/b), (a*b)/(a+b)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "nested fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Cancelar factores en una fracción");
}

#[test]
fn derive_nested_fraction_one_over_sum_uses_common_denominator_substep() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1/(1/x + 1/y), (x*y)/(x+y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "nested fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 1);
    assert_eq!(
        substeps[0]["title"],
        "Primero simplificar la suma del denominador"
    );
    let after = substeps[0]["after_latex"].as_str().expect("after_latex");
    assert!(
        after.contains("x\\cdot y")
            || after.contains("y\\cdot x")
            || after.contains("x \\cdot y")
            || after.contains("y \\cdot x"),
        "expected common denominator product in substep, got: {after}"
    );
    assert!(
        !after.contains("\\frac{1}{y} \\cdot x") && !after.contains("\\frac{1}{x} \\cdot y"),
        "expected to avoid partially simplified reciprocal product in substep, got: {after}"
    );
}

#[test]
fn root_nesting_drops_intrinsically_nonnegative_radicand_require() {
    let (output, _code) = run_cli(&[
        "eval",
        "sqrt(x^2 + 2*sqrt(x^2 + 1) + 2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "(x^2 + 1)^(1/2) + 1");
    assert_eq!(wire["required_display"], json!([]));
}

#[test]
fn derive_nested_fraction_one_over_sum_drops_redundant_reciprocal_sum_guard() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1/(1/a + 1/b), (a*b)/(a+b)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    let required: Vec<&str> = required.iter().filter_map(|item| item.as_str()).collect();

    assert!(
        !required.contains(&"1 / a + 1 / b ≠ 0"),
        "expected reciprocal-sum guard to be dominated by atomic guards: {required:?}"
    );
    assert!(required.contains(&"a + b ≠ 0"));
    assert!(required.contains(&"a ≠ 0"));
    assert!(required.contains(&"b ≠ 0"));
}

#[test]
fn derive_nested_fraction_structural_uses_named_strategy() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive a/(b + c/d), a*d/(b*d+c)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "nested fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Cancelar factores en una fracción");
}

#[test]
fn derive_nested_fraction_reciprocal_sum_difference_shows_common_denominator_substeps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (1/x + 1/y)/(1/x - 1/y), (x+y)/(y-x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "nested fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Cancelar factores en una fracción");

    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 2);
    assert_eq!(
        substeps[0]["title"],
        "Llevar el numerador y el denominador a común denominador"
    );
    assert_eq!(
        substeps[1]["title"],
        "Cancelar el denominador común de numerador y denominador"
    );
}

#[test]
fn derive_consecutive_factorial_ratio_uses_named_factorial_rewrite_and_keeps_guard() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (n+1)!/n!, n+1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite factorials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Cancelar factoriales consecutivos");

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        required.iter().any(|item| item == "n! ≠ 0"),
        "expected factorial nonzero guard in required_display: {required:?}"
    );
}

#[test]
fn derive_consecutive_factorial_ratio_with_passthrough_uses_named_factorial_rewrite_and_keeps_guard(
) {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (n+1)!/n!+a, n+1+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite factorials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Cancelar factoriales consecutivos");

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        required.iter().any(|item| item == "n! ≠ 0"),
        "expected factorial nonzero guard in required_display: {required:?}"
    );
}

#[test]
fn derive_gap_two_factorial_ratio_uses_named_factorial_rewrite_with_didactic_substeps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (n+1)!/(n-1)!, n*(n+1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite factorials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Cancelar factoriales consecutivos");

    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 2);
    assert_eq!(
        substeps[0]["title"],
        "Expandir el factorial superior hasta llegar al factorial inferior"
    );
    assert_eq!(substeps[1]["title"], "Cancelar el factorial común");

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        required.iter().any(|item| item == "(n - 1)! ≠ 0"),
        "expected factorial nonzero guard in required_display: {required:?}"
    );
}

#[test]
fn derive_inverse_tan_reciprocal_identity_uses_named_inverse_trig_rewrite_and_keeps_guard() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive arctan(a)+arctan(1/a), pi/2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite inverse trigs");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de arctangentes");

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        required.iter().any(|item| item == "a ≠ 0"),
        "expected inverse-tan nonzero guard in required_display: {required:?}"
    );
}

#[test]
fn derive_difference_of_squares_fraction_uses_named_fraction_cancel_and_keeps_guard() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (a^2-b^2)/(a-b), a+b",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "cancel fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Factorizar una diferencia de cuadrados y cancelar"
    );
    assert_eq!(steps[0]["before"], "(a^2 - b^2)/(a - b)");

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        required.iter().any(|item| item == "a - b ≠ 0"),
        "expected denominator nonzero guard in required_display: {required:?}"
    );
}

#[test]
fn derive_difference_of_squares_fraction_with_passthrough_uses_named_fraction_cancel_and_keeps_guard(
) {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (a^2-b^2)/(a-b)+c, a+b+c",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "cancel fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Factorizar una diferencia de cuadrados y cancelar"
    );
    assert_eq!(steps[0]["before"], "(a^2 - b^2)/(a - b) + c");

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        required.iter().any(|item| item == "a - b ≠ 0"),
        "expected denominator nonzero guard in required_display: {required:?}"
    );
}

#[test]
fn derive_difference_of_cubes_fraction_uses_named_fraction_cancel_and_keeps_guard() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (a^3-b^3)/(a-b), a^2+a*b+b^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "cancel fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Factorizar cubos y cancelar");

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        required.iter().any(|item| item == "a - b ≠ 0"),
        "expected denominator nonzero guard in required_display: {required:?}"
    );
}

#[test]
fn derive_difference_of_cubes_fraction_with_passthrough_uses_named_fraction_cancel_and_keeps_guard()
{
    let (output, _code) = run_cli(&[
        "eval",
        "derive (a^3-b^3)/(a-b)+c, a^2+a*b+b^2+c",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "cancel fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Factorizar cubos y cancelar");

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        required.iter().any(|item| item == "a - b ≠ 0"),
        "expected denominator nonzero guard in required_display: {required:?}"
    );
}

#[test]
fn derive_common_factor_fraction_uses_named_fraction_cancel_and_keeps_guards() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (a*x^2)/(b*x), (a*x)/b",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "cancel fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Cancelar un factor común");

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        required.iter().any(|item| item == "x ≠ 0"),
        "expected cancelled-factor guard in required_display: {required:?}"
    );
}

#[test]
fn derive_perfect_square_root_to_abs_uses_named_radical_rewrite_and_keeps_guard() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sqrt(a^2 + 2*a*b + b^2), abs(a+b)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite radicals");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Reconocer un cuadrado perfecto bajo la raíz"
    );

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        required.is_empty(),
        "expected no required_display guards: {required:?}"
    );
}

#[test]
fn derive_perfect_square_root_to_abs_with_passthrough_uses_named_radical_rewrite() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sqrt(a^2 + 2*a*b + b^2)+c, abs(a+b)+c",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite radicals");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Reconocer un cuadrado perfecto bajo la raíz"
    );
    let after = steps[0]["after"].as_str().expect("after");
    assert!(
        after == "|a + b| + c" || after == "abs(a+b)+c",
        "unexpected after: {after}"
    );
}

#[test]
fn derive_odd_half_power_with_passthrough_uses_named_expand_odd_half_power_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sqrt(x^3)+a, abs(x)*sqrt(x)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand odd half power");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Extraer potencia par de la raíz");
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 2);
    assert_eq!(wire["required_display"], json!(["x ≥ 0"]));
}

#[test]
fn derive_sqrt_odd_power_extracts_even_power_from_root_with_didactic_substeps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sqrt(x^5), x^2*sqrt(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand odd half power");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Extraer potencia par de la raíz");
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 2);
    assert_eq!(
        substeps[0]["title"],
        "Separar el radicando en una potencia par y un factor"
    );
    let second_title = substeps[1]["title"].as_str().expect("second title");
    assert!(
        second_title.contains("Como x ≥ 0"),
        "unexpected second title: {second_title}"
    );
}

#[test]
fn derive_higher_odd_half_power_with_passthrough_uses_named_expand_odd_half_power_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sqrt(x^7)+a, x^3*sqrt(x)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand odd half power");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Extraer potencia par de la raíz");
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 2);
    assert_eq!(
        substeps[0]["title"],
        "Separar el radicando en una potencia par y un factor"
    );
    let second_title = substeps[1]["title"].as_str().expect("second title");
    assert!(
        second_title.contains("Como x ≥ 0"),
        "unexpected second title: {second_title}"
    );
    assert_eq!(wire["required_display"], json!(["x ≥ 0"]));
}

#[test]
fn derive_hyperbolic_double_angle_with_passthrough_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sinh(x)*cosh(x)+a, sinh(2*x)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Double-Angle Identity");
}

#[test]
fn derive_hyperbolic_pythagorean_with_passthrough_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cosh(x)^2-sinh(x)^2+a, 1+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Pythagorean Identity");
}

#[test]
fn derive_pythagorean_identity_uses_named_trig_rewrite() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)^2 + cos(x)^2, 1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite trigs");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar la identidad pitagórica");
}

#[test]
fn derive_reciprocal_trig_product_with_passthrough_uses_named_trig_rewrite() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive tan(x)*cot(x)+a, 1+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite trigs");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Cancelar funciones trigonométricas recíprocas"
    );
}

#[test]
fn derive_negative_sec_reciprocal_uses_single_reciprocal_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -1/cos(x), -sec(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Aplicar identidad trigonométrica recíproca"
    );
}

#[test]
fn derive_shifted_sec_squared_uses_single_reciprocal_pythagorean_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sec(x)^2-1, tan(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad pitagórica recíproca");
}

#[test]
fn derive_shifted_cot_squared_uses_single_reciprocal_pythagorean_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cot(x)^2, csc(x)^2-1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad pitagórica recíproca");
}

#[test]
fn derive_one_minus_sec_squared_uses_single_reciprocal_pythagorean_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1-sec(x)^2, -tan(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad pitagórica recíproca");
}

#[test]
fn derive_negative_cot_squared_uses_single_reciprocal_pythagorean_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cot(x)^2, 1-csc(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad pitagórica recíproca");
}

#[test]
fn derive_shifted_double_angle_plus_uses_single_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(2*x)+1, 2*cos(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir ángulo doble");
}

#[test]
fn derive_shifted_double_angle_minus_backward_uses_single_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sin(x)^2, 1-cos(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir ángulo doble");
}

#[test]
fn derive_shifted_double_angle_negative_forward_uses_single_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(2*x)-1, -2*sin(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir ángulo doble");
}

#[test]
fn derive_shifted_double_angle_negative_backward_uses_single_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -2*sin(x)^2, cos(2*x)-1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir ángulo doble");
}

#[test]
fn derive_negative_cosine_double_angle_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)^2-cos(x)^2, -cos(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir ángulo doble");
}

#[test]
fn derive_negative_cosine_double_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cos(2*x), sin(x)^2-cos(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir ángulo doble");
}

#[test]
fn derive_negative_sine_double_angle_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -2*sin(x)*cos(x), -sin(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir ángulo doble");
}

#[test]
fn derive_negative_sine_double_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -sin(2*x), -2*sin(x)*cos(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir ángulo doble");
}

#[test]
fn derive_negative_half_angle_square_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(1-cos(2*x))/2, -sin(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de ángulo mitad");
}

#[test]
fn derive_negative_half_angle_square_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cos(x)^2, -(1+cos(2*x))/2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Aplicar identidad de ángulo mitad");
}

#[test]
fn derive_negative_half_angle_tangent_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(1-cos(2*x))/sin(2*x), -tan(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Aplicar identidad de tangente de ángulo mitad"
    );
}

#[test]
fn derive_negative_half_angle_tangent_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -tan(x), -(1-cos(2*x))/sin(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Aplicar identidad de tangente de ángulo mitad"
    );
}

#[test]
fn derive_negative_trig_sine_triple_angle_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(3*sin(x)-4*sin(x)^3), -sin(3*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Triple Angle Expansion");
}

#[test]
fn derive_negative_trig_sine_triple_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -sin(3*x), -(3*sin(x)-4*sin(x)^3)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Triple Angle Expansion");
}

#[test]
fn derive_negative_trig_cosine_triple_angle_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(4*cos(x)^3-3*cos(x)), -cos(3*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Triple Angle Expansion");
}

#[test]
fn derive_negative_trig_cosine_triple_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cos(3*x), -(4*cos(x)^3-3*cos(x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Triple Angle Expansion");
}

#[test]
fn derive_trig_tangent_triple_angle_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (3*tan(x)-tan(x)^3)/(1-3*tan(x)^2), tan(3*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Triple Angle Expansion");
}

#[test]
fn derive_trig_tangent_triple_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive tan(3*x), (3*tan(x)-tan(x)^3)/(1-3*tan(x)^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Triple Angle Expansion");
}

#[test]
fn derive_negative_trig_tangent_triple_angle_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(3*tan(x)-tan(x)^3)/(1-3*tan(x)^2), -tan(3*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Triple Angle Expansion");
}

#[test]
fn derive_negative_trig_tangent_triple_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -tan(3*x), -(3*tan(x)-tan(x)^3)/(1-3*tan(x)^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Triple Angle Expansion");
}

#[test]
fn derive_trig_quintuple_sine_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(5*x), 5*sin(x)-20*sin(x)^3+16*sin(x)^5",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Quintuple Angle Identity");
}

#[test]
fn derive_negative_trig_quintuple_sine_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -sin(5*x), -(5*sin(x)-20*sin(x)^3+16*sin(x)^5)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Quintuple Angle Identity");
}

#[test]
fn derive_trig_quintuple_cosine_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(5*x), 16*cos(x)^5-20*cos(x)^3+5*cos(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Quintuple Angle Identity");
}

#[test]
fn derive_negative_trig_quintuple_cosine_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cos(5*x), -(16*cos(x)^5-20*cos(x)^3+5*cos(x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Quintuple Angle Identity");
}

#[test]
fn derive_trig_quintuple_sine_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 5*sin(x)-20*sin(x)^3+16*sin(x)^5, sin(5*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Quintuple Angle Identity");
}

#[test]
fn derive_negative_trig_quintuple_sine_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(5*sin(x)-20*sin(x)^3+16*sin(x)^5), -sin(5*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Quintuple Angle Identity");
}

#[test]
fn derive_trig_quintuple_cosine_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 16*cos(x)^5-20*cos(x)^3+5*cos(x), cos(5*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Quintuple Angle Identity");
}

#[test]
fn derive_negative_trig_quintuple_cosine_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(16*cos(x)^5-20*cos(x)^3+5*cos(x)), -cos(5*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Quintuple Angle Identity");
}

#[test]
fn derive_trig_angle_sum_sine_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x+y), sin(x)*cos(y)+cos(x)*sin(y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Angle Sum/Diff Identity");
}

#[test]
fn derive_trig_angle_sum_sine_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)*cos(y)+cos(x)*sin(y), sin(x+y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Angle Sum/Diff Identity");
}

#[test]
fn derive_trig_recursive_six_x_sine_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(6*x), sin(5*x)*cos(x)+cos(5*x)*sin(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Angle Sum/Diff Identity");
}

#[test]
fn derive_trig_recursive_six_x_sine_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(5*x)*cos(x)+cos(5*x)*sin(x), sin(6*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Angle Sum/Diff Identity");
}

#[test]
fn derive_trig_angle_diff_sine_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)*cos(y)-cos(x)*sin(y), sin(x-y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Angle Sum/Diff Identity");
}

#[test]
fn derive_negative_trig_angle_diff_sine_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(sin(x)*cos(y)-cos(x)*sin(y)), -sin(x-y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Angle Sum/Diff Identity");
}

#[test]
fn derive_trig_angle_diff_cosine_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(x-y), cos(x)*cos(y)+sin(x)*sin(y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Angle Sum/Diff Identity");
}

#[test]
fn derive_trig_recursive_six_x_cosine_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(5*x)*cos(x)-sin(5*x)*sin(x), cos(6*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Angle Sum/Diff Identity");
}

#[test]
fn derive_negative_trig_recursive_six_x_cosine_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(cos(5*x)*cos(x)-sin(5*x)*sin(x)), -cos(6*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Angle Sum/Diff Identity");
}

#[test]
fn derive_hyperbolic_angle_sum_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sinh(x)*cosh(y)+cosh(x)*sinh(y), sinh(x+y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Angle Sum/Difference Identity");
}

#[test]
fn derive_hyperbolic_angle_diff_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cosh(x)*cosh(y)-sinh(x)*sinh(y), cosh(x-y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Angle Sum/Difference Identity");
}

#[test]
fn derive_negative_hyperbolic_angle_diff_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(cosh(x)*cosh(y)-sinh(x)*sinh(y)), -cosh(x-y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Angle Sum/Difference Identity");
}

#[test]
fn derive_recursive_hyperbolic_sinh_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sinh(6*x), sinh(5*x)*cosh(x)+cosh(5*x)*sinh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Angle Sum/Difference Identity");
}

#[test]
fn derive_hyperbolic_sinh_angle_sum_expansion_uses_expand_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sinh(x+y), sinh(x)*cosh(y)+cosh(x)*sinh(y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Angle Sum/Difference Identity");
}

#[test]
fn derive_recursive_hyperbolic_cosh_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cosh(6*x), cosh(5*x)*cosh(x)+sinh(5*x)*sinh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Angle Sum/Difference Identity");
}

#[test]
fn derive_hyperbolic_tanh_angle_sum_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive tanh(x+y), (tanh(x)+tanh(y))/(1+tanh(x)*tanh(y))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Angle Sum/Difference Identity");
}

#[test]
fn derive_hyperbolic_tanh_angle_sum_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (tanh(x)+tanh(y))/(1+tanh(x)*tanh(y)), tanh(x+y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Angle Sum/Difference Identity");
}

#[test]
fn derive_hyperbolic_tanh_triple_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive tanh(3*x), (3*tanh(x)+tanh(x)^3)/(1+3*tanh(x)^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Triple-Angle Identity");
}

#[test]
fn derive_hyperbolic_tanh_triple_angle_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (3*tanh(x)+tanh(x)^3)/(1+3*tanh(x)^2), tanh(3*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Triple-Angle Identity");
}

#[test]
fn derive_exponential_sum_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x+y), exp(x)*exp(y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Exponential Sum/Difference Identity");
}

#[test]
fn derive_exponential_sum_expansion_with_passthrough_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x+y)+a, exp(x)*exp(y)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Exponential Sum/Difference Identity");
}

#[test]
fn derive_exponential_difference_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x-y), exp(x)/exp(y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Exponential Sum/Difference Identity");
}

#[test]
fn derive_exponential_product_contraction_with_passthrough_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x)*exp(y)+a, exp(x+y)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Exponential Sum/Difference Identity");
}

#[test]
fn derive_exponential_product_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x)*exp(y), exp(x+y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Exponential Sum/Difference Identity");
}

#[test]
fn derive_exponential_product_quotient_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x)*exp(y)/exp(z), exp(x+y-z)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Exponential Sum/Difference Identity");
}

#[test]
fn derive_exponential_quotient_with_power_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x)/exp(y)^2, exp(x-2*y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Exponential Sum/Difference Identity");
}

#[test]
fn derive_exponential_power_quotient_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x)^2/exp(y)^3, exp(2*x-3*y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Exponential Sum/Difference Identity");
}

#[test]
fn derive_exponential_reciprocal_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1/exp(x), exp(-x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Exponential Reciprocal Identity");
}

#[test]
fn derive_exponential_reciprocal_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(-x), 1/exp(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Exponential Reciprocal Identity");
}

#[test]
fn derive_negative_exponential_reciprocal_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -1/exp(x), -exp(-x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Exponential Reciprocal Identity");
}

#[test]
fn derive_exponential_power_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x)^3, exp(3*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Exponential Power Identity");
}

#[test]
fn derive_exponential_reciprocal_power_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1/exp(x)^2, exp(-2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Exponential Reciprocal Identity");
}

#[test]
fn derive_exponential_scaled_argument_to_scaled_cosh_uses_direct_identity() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(2*x)+exp(-2*x), 2*cosh(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Exponential Identity");
}

#[test]
fn derive_exponential_to_scaled_cosh_uses_direct_identity() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x)+exp(-x), 2*cosh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Exponential Identity");
}

#[test]
fn derive_scaled_cosh_to_exponential_sum_uses_direct_identity() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*cosh(2*x), exp(2*x)+exp(-2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Exponential Identity");
}

#[test]
fn derive_scaled_sinh_to_exponential_difference_uses_direct_identity() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sinh(x), exp(x)-exp(-x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Exponential Identity");
}

#[test]
fn derive_scaled_exponential_ratio_to_tanh_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (e^(2*x)-e^(-2*x))/(e^(2*x)+e^(-2*x)), tanh(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Exponential Identity");
}

#[test]
fn derive_tanh_exponential_definition_drops_redundant_nonzero_requires() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive tanh(x), (e^x - e^(-x))/(e^x + e^(-x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["required_display"], json!([]));
}

#[test]
fn derive_tanh_reciprocal_exponential_definition_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive tanh(x), (exp(x)-1/exp(x))/(exp(x)+1/exp(x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Exponential Identity");
}

#[test]
fn derive_negative_tanh_reciprocal_exponential_definition_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -tanh(x), (1/exp(x)-exp(x))/(exp(x)+1/exp(x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Exponential Identity");
}

#[test]
fn derive_scaled_cosh_reciprocal_exponential_definition_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*cosh(x), exp(x)+1/exp(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Exponential Identity");
}

#[test]
fn derive_scaled_sinh_reciprocal_exponential_definition_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sinh(x), exp(x)-1/exp(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Exponential Identity");
}

#[test]
fn derive_half_cosh_reciprocal_exponential_recognition_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (exp(x)+1/exp(x))/2, cosh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Exponential Identity");
}

#[test]
fn derive_half_sinh_reciprocal_exponential_recognition_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (exp(x)-1/exp(x))/2, sinh(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Exponential Identity");
}

#[test]
fn derive_negative_scaled_cosh_reciprocal_exponential_definition_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -2*cosh(x), -(exp(x)+1/exp(x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Exponential Identity");
}

#[test]
fn derive_negative_scaled_sinh_reciprocal_exponential_definition_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -2*sinh(x), 1/exp(x)-exp(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Exponential Identity");
}

#[test]
fn derive_negative_half_cosh_reciprocal_exponential_definition_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cosh(x), -(exp(x)+1/exp(x))/2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Exponential Identity");
}

#[test]
fn derive_negative_half_sinh_reciprocal_exponential_definition_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -sinh(x), -(exp(x)-1/exp(x))/2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite hyperbolics");
    assert_eq!(wire["steps_count"], 1);
    assert_eq!(wire["required_display"], json!([]));
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Hyperbolic Exponential Identity");
}

#[test]
fn derive_log_higher_even_power_expansion_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive ln(x^4), 4*ln(abs(x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Sacar un exponente fuera del logaritmo");
}

#[test]
fn derive_log_general_base_power_expansion_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive log(b, x^3), 3*log(b, x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Sacar un exponente fuera del logaritmo");
}

#[test]
fn derive_log_expansion_to_zero_keeps_single_step_but_closes_on_final_result() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive ln(x^3) + ln(y^2) - ln(x^3*y^2), 0",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir logaritmos");
    assert_eq!(steps[0]["after"], "0");
}

#[test]
fn derive_grouped_even_log_product_expansion_uses_direct_expand_log() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive ln((x*y)^2), ln(x^2)+ln(y^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir logaritmos");
}

#[test]
fn derive_grouped_abs_log_product_expansion_uses_direct_expand_log() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*ln(abs(x*y)), 2*ln(abs(x))+2*ln(abs(y))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir logaritmos");
}

#[test]
fn derive_grouped_general_base_log_product_expansion_uses_direct_expand_log() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive log(b,(x*y)^2), 2*log(b,x)+2*log(b,y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir logaritmos");
}

#[test]
fn derive_grouped_even_log_product_expansion_with_passthrough_uses_direct_expand_log() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive ln((x*y)^2)+a, ln(x^2)+ln(y^2)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir logaritmos");
}

#[test]
fn derive_grouped_abs_log_product_expansion_with_passthrough_uses_direct_expand_log() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*ln(abs(x*y))+a, 2*ln(abs(x))+2*ln(abs(y))+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir logaritmos");
}

#[test]
fn derive_grouped_general_base_log_product_expansion_with_passthrough_uses_direct_expand_log() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive log(b,(x*y)^2)+a, 2*log(b,x)+2*log(b,y)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Expandir logaritmos");
}

#[test]
fn derive_factored_log_difference_squares_with_passthrough_uses_two_named_steps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive log(x^2-y^2)+a, log(x-y)+log(x+y)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 2);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["rule"], "Factorizar");
    assert_eq!(steps[1]["rule"], "Expandir logaritmos");
}

#[test]
fn derive_factored_log_difference_squares_uses_two_named_steps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive log(x^2-y^2), log(x-y)+log(x+y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 2);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["rule"], "Factorizar");
    assert_eq!(steps[1]["rule"], "Expandir logaritmos");
    let required = wire["required_display"]
        .as_array()
        .expect("required_display array");
    assert_eq!(required.len(), 2);
    assert!(required.iter().any(|item| item == "x + y > 0"));
    assert!(required.iter().any(|item| item == "x - y > 0"));
}

#[test]
fn derive_factored_log_quotient_difference_squares_uses_two_named_steps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive log((x^2-y^2)/(u*v)), log(x-y)+log(x+y)-log(u)-log(v)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 2);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["rule"], "Factorizar");
    assert_eq!(steps[1]["rule"], "Expandir logaritmos");
}

#[test]
fn derive_difference_of_cubes_fraction_keeps_requested_target_text_in_final_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (x^3-1)/(x-1), x^2+x+1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(wire["strategy"], "cancel fraction");
    assert_eq!(steps[0]["rule"], "Factorizar cubos y cancelar");
    assert_eq!(steps[0]["after"], "x^2 + x + 1");
}

#[test]
fn derive_geometric_difference_fraction_uses_direct_cancel_fraction() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (x^4-1)/(x-1), x^3+x^2+x+1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "cancel fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["before"], "(x^4 - 1)/(x - 1)");
    assert_eq!(steps[0]["after"], "x^3 + x^2 + x + 1");
}

#[test]
fn derive_geometric_difference_even_quotient_uses_direct_cancel_fraction() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (x^4-1)/(x^2-1), x^2+1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "cancel fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["before"], "(x^4 - 1)/(x^2 - 1)");
    assert_eq!(steps[0]["after"], "x^2 + 1");
}

#[test]
fn derive_grouped_even_log_product_uses_direct_log_contraction() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive ln(x^2)+ln(y^2), ln((x*y)^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract logs");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Contraer logaritmos");
}

#[test]
fn derive_grouped_even_log_product_with_passthrough_uses_direct_log_contraction() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive ln(x^2)+ln(y^2)+a, ln((x*y)^2)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract logs");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Contraer logaritmos");
}

#[test]
fn derive_scaled_abs_log_product_uses_direct_log_contraction() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*ln(abs(x))+2*ln(abs(y)), 2*ln(abs(x*y))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract logs");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Contraer logaritmos");
}

#[test]
fn derive_scaled_abs_log_product_with_passthrough_uses_direct_log_contraction() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*ln(abs(x))+2*ln(abs(y))+a, 2*ln(abs(x*y))+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract logs");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Contraer logaritmos");
}

#[test]
fn derive_grouped_general_base_log_product_uses_direct_log_contraction() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*log(b,x)+2*log(b,y), log(b,(x*y)^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract logs");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Contraer logaritmos");
}

#[test]
fn derive_grouped_general_base_log_product_with_passthrough_uses_direct_log_contraction() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*log(b,x)+2*log(b,y)+a, log(b,(x*y)^2)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract logs");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Contraer logaritmos");
}

#[test]
fn derive_rationalize_then_cancel_to_zero_uses_rationalize_strategy() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1 / (sqrt(x) - 1) - (sqrt(x) + 1) / (x - 1), 0",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rationalize");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["rule"], "Racionalizar el denominador");
    assert_eq!(steps[1]["rule"], "Restar dos expresiones iguales");
}

#[test]
fn derive_radical_notable_quotient_uses_single_rationalize_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (x^(3/2)-1)/(sqrt(x)-1), sqrt(x)+x+1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rationalize");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Reconocer un cociente notable");
}

#[test]
fn derive_log_higher_even_power_drops_redundant_nonzero_requires() {
    let (output, _code) = run_cli(&["eval", "derive ln(x^4), 4*ln(abs(x))", "--format", "json"]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert_eq!(required.len(), 1);
    assert_eq!(required[0], "x ≠ 0");
}

#[test]
fn eval_abs_denominator_displays_inner_nonzero_guard() {
    let (output, _code) = run_cli(&["eval", "1/abs(x)", "--format", "json"]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert_eq!(required.len(), 1);
    assert_eq!(required[0], "x ≠ 0");
}

#[test]
fn eval_abs_product_denominator_expands_to_atomic_nonzero_guards() {
    let (output, _code) = run_cli(&["eval", "1/abs(x*y)", "--format", "json"]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert_eq!(
        required
            .iter()
            .filter_map(|item| item.as_str())
            .collect::<Vec<_>>(),
        vec!["x ≠ 0", "y ≠ 0"]
    );
}

#[test]
fn eval_abs_factored_product_denominator_expands_to_atomic_nonzero_guards() {
    let (output, _code) = run_cli(&["eval", "1/abs((x-1)*(x+1))", "--format", "json"]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert_eq!(
        required
            .iter()
            .filter_map(|item| item.as_str())
            .collect::<Vec<_>>(),
        vec!["x - 1 ≠ 0", "x + 1 ≠ 0"]
    );
}

#[test]
fn eval_log_abs_factored_product_expands_to_atomic_nonzero_guards() {
    let (output, _code) = run_cli(&["eval", "ln(abs((x-1)*(x+1)))", "--format", "json"]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert_eq!(
        required
            .iter()
            .filter_map(|item| item.as_str())
            .collect::<Vec<_>>(),
        vec!["x - 1 ≠ 0", "x + 1 ≠ 0"]
    );
}

#[test]
fn eval_abs_quotient_expands_to_atomic_numerator_and_denominator_guards() {
    let (output, _code) = run_cli(&["eval", "1/abs(x/(x+1))", "--format", "json"]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    let required: Vec<_> = required.iter().filter_map(|item| item.as_str()).collect();
    assert_eq!(required.len(), 2);
    assert!(required.contains(&"x ≠ 0"));
    assert!(required.contains(&"x + 1 ≠ 0"));
}

#[test]
fn eval_log_abs_quotient_expands_to_atomic_numerator_and_denominator_guards() {
    let (output, _code) = run_cli(&["eval", "ln(abs((x-1)/(x+1)))", "--format", "json"]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert_eq!(
        required
            .iter()
            .filter_map(|item| item.as_str())
            .collect::<Vec<_>>(),
        vec!["x - 1 ≠ 0", "x + 1 ≠ 0"]
    );
}

#[test]
fn derive_log_multifactor_expansion_drops_redundant_composite_positive_require() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive ln((x^2*y)/(z*t)), 2*ln(abs(x)) + ln(y) - ln(z) - ln(t)",
        "--format",
        "json",
    ]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert_eq!(required.len(), 4);
    assert!(required.iter().all(|item| item != "y·x^2 / (t·z) > 0"));
}

#[test]
fn eval_general_base_log_surfaces_positive_base_and_argument_requires() {
    let (output, _code) = run_cli(&["eval", "log(b, x)", "--format", "json"]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert_eq!(required.len(), 3);
    assert!(required.iter().any(|item| item == "b - 1 ≠ 0"));
    assert!(required.iter().any(|item| item == "b > 0"));
    assert!(required.iter().any(|item| item == "x > 0"));
}

#[test]
fn derive_general_base_log_expansion_surfaces_positive_factor_requires() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive log(b, (x^2*y^3)/(z^2*t)), 2*log(b, x) + 3*log(b, y) - 2*log(b, z) - log(b, t)",
        "--format",
        "json",
    ]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert_eq!(required.len(), 6);
    assert!(required.iter().any(|item| item == "b - 1 ≠ 0"));
    assert!(required.iter().any(|item| item == "b > 0"));
    assert!(required.iter().any(|item| item == "x > 0"));
    assert!(required.iter().any(|item| item == "y > 0"));
    assert!(required.iter().any(|item| item == "z > 0"));
    assert!(required.iter().any(|item| item == "t > 0"));
}

#[test]
fn derive_exponential_power_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(3*x), exp(x)^3",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Exponential Power Identity");
}

#[test]
fn exponential_quotient_drops_redundant_nonzero_requires() {
    let (output, _code) = run_cli(&["eval", "exp(x)/exp(y)", "--format", "json"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "e^(x - y)");
    assert_eq!(
        wire["required_display"].as_array().map(Vec::len),
        Some(0),
        "exp quotient should not require e^y ≠ 0"
    );
}

#[test]
fn derive_exponential_sum_difference_drops_redundant_nonzero_requires() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x+y-z), exp(x)*exp(y)/exp(z)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "e^x·e^y / e^z");
    assert_eq!(
        wire["required_display"].as_array().map(Vec::len),
        Some(0),
        "derive exponential expansion should not require e^z ≠ 0"
    );
}

#[test]
fn derive_consecutive_telescoping_fraction_split_omits_trivial_substitution_substep() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1/(u*(u+1)), 1/u - 1/(u+1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Descomponer en fracciones telescópicas");
    assert!(
        steps[0].get("substeps").is_none(),
        "literal telescoping split should not emit the tautological 'Aquí u = u' substep"
    );
}

#[test]
fn derive_consecutive_telescoping_fraction_combine_omits_trivial_substitution_substep() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1/u - 1/(u+1), 1/(u*(u+1))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "combine fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Recomponer fracción telescópica");
    assert!(
        steps[0].get("substeps").is_none(),
        "literal telescoping combine should not emit the tautological 'Aquí u = u' substep"
    );
}

#[test]
fn derive_fraction_difference_to_single_fraction_shows_common_denominator_substeps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1/(x - 1) - 1/(x + 1), 2/(x^2 - 1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "combine fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Restar fracciones");
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 2);
    assert_eq!(substeps[0]["title"], "Llevar a denominador común");
    assert_eq!(
        substeps[1]["title"],
        "Simplificar el numerador y el denominador"
    );
}

#[test]
fn derive_tan_plus_cot_keeps_single_common_denominator_substep() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive tan(x) + cot(x), sec(x)*csc(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "simplify");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 4);
    assert_eq!(steps[2]["rule"], "Sumar fracciones");
    let substeps = steps[2]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 1);
    assert_eq!(substeps[0]["title"], "Llevar a denominador común");
}

#[test]
fn complex_principal_inv_trig_warning_surfaces_in_cli_wire() {
    let (output, _code) = run_cli(&[
        "eval",
        "arcsin(sin(x))",
        "--format",
        "json",
        "--value-domain",
        "complex",
        "--inv-trig",
        "principal",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["semantics"]["value_domain"], "complex");
    assert_eq!(wire["semantics"]["inv_trig"], "principal");
    assert_eq!(wire["result"], "x");

    let warnings = wire["warnings"].as_array().expect("warnings array");
    assert_eq!(
        warnings.len(),
        1,
        "principal inverse-trig should emit one warning"
    );
    assert_eq!(warnings[0]["rule"], "Principal Branch Inverse Trig");
}
