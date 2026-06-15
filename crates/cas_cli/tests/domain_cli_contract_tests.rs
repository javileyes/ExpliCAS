//! CLI domain mode contract tests.
//!
//! # Contract: --domain CLI Flag
//!
//! The CLI should support `--domain strict|generic|assume` to control
//! simplification behavior:
//!
//! - `--domain strict`: Only cancel provably nonzero factors
//! - `--domain generic` (default): Legacy behavior (`x/x → 1`)
//! - `--domain assume`: Simplify with warnings/assumptions
//!
//! Wire output should reflect the domain mode in a stable field.

use serde_json::Value;
use std::process::Command;

fn run_cli(args: &[&str]) -> (String, i32) {
    let output = Command::new(env!("CARGO_BIN_EXE_cas_cli"))
        .args(args)
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let code = output.status.code().unwrap_or(-1);
    (stdout, code)
}

fn parse_wire(s: &str) -> Value {
    serde_json::from_str(s).unwrap_or_else(|_| panic!("Failed to parse wire payload: {}", s))
}

// =============================================================================
// CLI --domain flag tests
// =============================================================================

#[test]
fn cli_domain_generic_x_div_x_simplifies_to_1() {
    // Generic (default) => x/x -> 1
    let (output, _code) = run_cli(&["eval", "x/x", "--format", "json"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "1", "Generic mode should simplify x/x to 1");
}

#[test]
fn cli_domain_strict_x_div_x_stays_unchanged() {
    // Strict => x/x stays as x/x
    let (output, _code) = run_cli(&["eval", "x/x", "--format", "json", "--domain", "strict"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    let result = wire["result"].as_str().unwrap_or("");
    assert!(
        result == "x/x" || result == "x / x",
        "Strict mode should NOT simplify x/x, got: {}",
        result
    );
}

#[test]
fn cli_domain_strict_partial_cancel_contract() {
    // Strict auto-eval preserves a residual fraction shape instead of collapsing to 2.
    // Depending on the active simplification path, this may appear as 4/2, 2*x/x,
    // or the equivalent reordered form (x*2)/x.
    let (output, _code) = run_cli(&[
        "eval",
        "4*x/(2*x)",
        "--format",
        "json",
        "--domain",
        "strict",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    let result = wire["result"].as_str().unwrap_or("");
    assert!(
        result == "4 / 2"
            || result == "4/2"
            || result == "(x·2)/x"
            || result == "(x * 2)/x"
            || result == "2·x/x"
            || result == "2 * x/x"
            || result == "(2·x)/x"
            || result == "(2 * x)/x",
        "Expected strict residual fraction preserving x-domain, got: {}",
        result
    );
    let required = wire["required_display"].as_array().unwrap();
    assert_eq!(
        required.len(),
        1,
        "Strict should preserve one domain condition"
    );
    assert_eq!(required[0], "x ≠ 0");
}

#[test]
fn cli_domain_generic_pow_zero_emits_required_nonzero() {
    // x^0 -> 1 must surface the definability condition on the wire
    // (0^0 is undefined), mirroring the x/x cancellation contract.
    let (output, _code) = run_cli(&["eval", "x^0", "--format", "json", "--domain", "generic"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "1");
    let required = wire["required_display"].as_array().unwrap();
    assert_eq!(
        required.len(),
        1,
        "expected exactly the x ≠ 0 condition, got: {required:?}"
    );
    assert_eq!(required[0], "x ≠ 0");
}

#[test]
fn cli_domain_assume_pow_zero_emits_required_nonzero() {
    let (output, _code) = run_cli(&["eval", "x^0", "--format", "json", "--domain", "assume"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "1");
    let required = wire["required_display"].as_array().unwrap();
    assert_eq!(
        required.len(),
        1,
        "expected exactly the x ≠ 0 condition, got: {required:?}"
    );
    assert_eq!(required[0], "x ≠ 0");
}

#[test]
fn cli_domain_generic_pow_zero_proven_base_unconditional() {
    let (output, _code) = run_cli(&[
        "eval",
        "(x^2+1)^0",
        "--format",
        "json",
        "--domain",
        "generic",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "1");
    let required = wire["required_display"].as_array().unwrap();
    assert!(
        required.is_empty(),
        "provably nonzero base needs no condition, got: {required:?}"
    );
}

#[test]
fn cli_branch_flag_is_alias_of_inv_trig() {
    // --branch is a deprecated alias of --inv-trig: principal must actually
    // apply the principal inverse-trig policy (it was a silent no-op before).
    let (output, _code) = run_cli(&[
        "eval",
        "arctan(tan(2))",
        "--format",
        "json",
        "--branch",
        "principal",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "2");

    // Default stays conservative.
    let (default_output, _code) = run_cli(&["eval", "arctan(tan(2))", "--format", "json"]);
    let default_wire = parse_wire(&default_output);
    assert_eq!(default_wire["result"], "arctan(tan(2))");
}

#[test]
fn cli_domain_assume_emits_warning() {
    // Assume => x/x -> 1 WITH warning
    let (output, _code) = run_cli(&["eval", "x/x", "--format", "json", "--domain", "assume"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "1", "Assume mode should simplify x/x to 1");

    // Contract: warning present (or warnings[] if using list)
    let has_warning = wire.get("warning").is_some()
        || wire.get("warnings").is_some()
        || wire.get("assumptions").is_some();
    assert!(
        has_warning,
        "Assume mode must emit warning/assumptions field. Wire: {}",
        wire
    );
}

#[test]
fn cli_domain_strict_numeric_preserves_fraction_in_auto_eval() {
    // Strict auto-eval keeps the fraction shape; solver-specific solve simplification is separate.
    let (output, _code) = run_cli(&["eval", "2/2", "--format", "json", "--domain", "strict"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    let result = wire["result"].as_str().unwrap_or("");
    assert!(
        result == "2 / 2" || result == "2/2",
        "Strict auto-eval should preserve 2/2, got: {}",
        result
    );
    assert!(
        wire["required_conditions"].as_array().unwrap().is_empty(),
        "Pure numeric fraction should not emit domain conditions"
    );
}

// =============================================================================
// Wire schema: domain field
// =============================================================================

#[test]
fn cli_wire_includes_domain_mode() {
    let (output, _code) = run_cli(&["eval", "x+x", "--format", "json", "--domain", "strict"]);
    let wire = parse_wire(&output);

    // Contract: wire output should include domain.mode field
    assert!(
        wire.get("domain").is_some(),
        "wire output should have 'domain' field"
    );
    assert_eq!(
        wire["domain"]["mode"], "strict",
        "domain.mode should reflect --domain flag"
    );
}

#[test]
fn cli_domain_default_is_generic() {
    // Without --domain flag, should use generic
    let (output, _code) = run_cli(&["eval", "x/x", "--format", "json"]);
    let wire = parse_wire(&output);

    // In generic mode, x/x simplifies to 1
    assert_eq!(wire["ok"], true);
    assert_eq!(
        wire["result"], "1",
        "Default (generic) should simplify x/x to 1"
    );
}

#[test]
fn cli_domain_generic_factor_cancellation_renders_atomic_requires_without_composite_duplicate() {
    let (output, _code) = run_cli(&[
        "eval",
        "(x^5 + x^4 - 2*x^2 - 2*x) / (x^3 - x)",
        "--format",
        "json",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "(x^3 - 2) / (x - 1)");

    let required = wire["required_display"]
        .as_array()
        .expect("required_display array");
    let required: Vec<_> = required.iter().filter_map(|item| item.as_str()).collect();

    assert!(
        !required.contains(&"x^3 - x ≠ 0"),
        "composite denominator guard should be expanded for display: {:?}",
        required
    );
    assert!(required.contains(&"x ≠ 0"));
    assert!(required.contains(&"x ≠ 1"));
    assert!(required.contains(&"x ≠ -1"));
}

// =============================================================================
// Power-of-a-power must keep the absolute value over the reals
// =============================================================================
//
// `(x^m)^n` with even inner exponent `m` equals `|x|^(m*n)`, NOT `x^(m*n)`,
// whenever the product exponent has an odd numerator. Flattening to the bare
// power drops a sign (odd integer exponent) or a domain restriction (even
// denominator), which then leaks into a wrong derivative. Regression guard for
// the `(x^2)^(3/2) -> x^3` / `diff = 3x^2` soundness bug.

fn eval_result(expr: &str) -> String {
    let (output, _code) = run_cli(&["eval", expr, "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["ok"], true, "eval failed for {expr}: {output}");
    wire["result"].as_str().unwrap_or("").to_string()
}

#[test]
fn power_power_even_inner_keeps_abs_over_reals() {
    // The reported case and its siblings: the absolute value must survive.
    assert_eq!(eval_result("(x^2)^(3/2)"), "|x|·x^2"); // = |x|^3
    assert_eq!(eval_result("(x^2)^(5/2)"), "|x|·x^4");
    assert_eq!(eval_result("(x^6)^(1/2)"), "|x|·x^2");
    assert_eq!(eval_result("(x^2)^(1/4)"), "|x|^(1/2)"); // domain: real for all x
    assert_eq!(eval_result("((x+1)^2)^(3/2)"), "|x + 1|·(x + 1)^2");

    // Sign-safe cases (even numerator product) must STAY the bare power.
    assert_eq!(eval_result("(x^4)^(1/2)"), "x^2");
    assert_eq!(eval_result("(x^4)^(3/2)"), "x^6");
    assert_eq!(eval_result("(x^2)^(1/3)"), "x^(2/3)");
}

#[test]
fn power_power_even_inner_derivative_is_sound() {
    // The headline bug: diff((x^2)^(3/2), x) was 3*x^2 (wrong sign for x<0).
    // The correct derivative is 3*x*|x| = 3*x*sqrt(x^2); the engine returns the
    // (correct, unreduced) product-rule form. It must NOT be the bare 3*x^2.
    let d = eval_result("diff((x^2)^(3/2), x)");
    assert_ne!(d, "3·x^2", "derivative must not drop the absolute value");
    assert!(
        d.contains("|x|"),
        "derivative of (x^2)^(3/2) must carry |x|, got: {d}"
    );
}

// =============================================================================
// d/dx of |x|-equivalent forms presents as the textbook sign(x), with x != 0
// =============================================================================
//
// d/dx |h| = sign(h) for an affine h, defined for h != 0. The compact-form
// presentation previously regressed d/dx sqrt((-x)^2) to x/sqrt((-x)^2) (the
// radicand left un-reduced); it now unifies the family to sign(h). The x != 0
// domain condition must survive the rewrite (sign is the textbook form but the
// derivative still does not exist at the corner).

fn eval_result_and_required(expr: &str) -> (String, Vec<String>) {
    let (output, _code) = run_cli(&["eval", expr, "--format", "json"]);
    let wire = parse_wire(&output);
    assert_eq!(wire["ok"], true, "eval failed for {expr}: {output}");
    let result = wire["result"].as_str().unwrap_or("").to_string();
    let required = wire["required_display"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|c| c.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    (result, required)
}

#[test]
fn diff_abs_family_presents_as_sign_with_nonzero_condition() {
    for (expr, expected_result, expected_condition) in [
        ("diff(abs(x), x)", "sign(x)", "x ≠ 0"),
        ("diff(sqrt(x^2), x)", "sign(x)", "x ≠ 0"),
        ("diff(sqrt((-x)^2), x)", "sign(x)", "x ≠ 0"),
        ("diff(abs(x+1), x)", "sign(x + 1)", "x ≠ -1"),
    ] {
        let (result, required) = eval_result_and_required(expr);
        assert_eq!(result, expected_result, "result for {expr}");
        assert!(
            required.iter().any(|c| c == expected_condition),
            "for {expr}: expected condition {expected_condition}, got {required:?}"
        );
    }
}

#[test]
fn diff_abs_family_does_not_spuriously_introduce_sign() {
    // Non-affine / non-polynomial abs arguments must NOT become sign(...): they
    // are smooth or the abs is incidental (e.g. sqrt(sec(x)) = |cos(x)^(-1/2)|,
    // which is >= 0). A spurious sign breaks downstream cancellation.
    let (smooth_abs, _) = eval_result_and_required("diff(abs(x^2+1), x)");
    assert_eq!(smooth_abs, "2·x", "|x^2+1| is smooth: no sign");
    // The residual must still verify to 0 (regression guard for the sec case).
    assert_eq!(
        eval_result("diff(sqrt(sec(x)), x) - sqrt(sec(x))*tan(x)/2"),
        "0",
        "sqrt(sec(x)) derivative residual must cancel (no spurious sign)"
    );
}

#[test]
fn sign_evaluates_on_numeric_constants() {
    assert_eq!(eval_result("sign(-3)"), "-1");
    assert_eq!(eval_result("sign(3)"), "1");
    assert_eq!(eval_result("sign(0)"), "0");
    assert_eq!(eval_result("sign(-1/2)"), "-1");
    // Symbolic sign stays symbolic.
    assert_eq!(eval_result("sign(x)"), "sign(x)");
}
