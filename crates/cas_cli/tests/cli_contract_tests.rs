//! CLI contract tests for the unified budget system and eval command.
//!
//! These tests validate the CLI behavior including:
//! - Help output shows correct commands
//! - wire output parsing and schema version
//! - Budget presets and strict mode

use assert_cmd::cargo;
use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;

/// Get the CLI command
fn cli() -> Command {
    Command::new(cargo::cargo_bin!("cas_cli"))
}

/// Test that help output shows the supported public commands.
#[test]
fn test_help_shows_correct_commands() {
    cli()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("eval"))
        .stdout(predicate::str::contains("envelope"))
        .stdout(predicate::str::contains("repl"))
        .stdout(predicate::str::contains("help"));
}

/// Test that eval command help shows budget options.
#[test]
fn test_eval_help_shows_budget_options() {
    cli()
        .args(["eval", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--budget"))
        .stdout(predicate::str::contains("--strict"))
        .stdout(predicate::str::contains("--format"))
        .stdout(predicate::str::contains("small"))
        .stdout(predicate::str::contains("standard"))
        .stdout(predicate::str::contains("unlimited"));
}

/// Test that eval with --format json produces valid wire output with schema_version.
#[test]
fn test_eval_wire_output_has_schema_version() {
    let output = cli()
        .args(["eval", "x+1", "--format", "json"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");

    assert_eq!(wire["schema_version"], 1);
    assert_eq!(wire["ok"], true);
    assert!(wire["budget"].is_object());
    assert_eq!(wire["budget"]["preset"], "standard");
    assert_eq!(wire["budget"]["mode"], "best-effort");
}

/// Test that eval can emit the stable wire output directly.
#[test]
fn test_eval_emits_wire_output() {
    let output = cli()
        .args(["eval", "2+2", "--format", "json"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "4");
}

#[test]
fn test_eval_calculus_residuals_emit_conservative_steps_json() {
    let cases = [(
        "integrate(exp(x^2), x)",
        "integrate(e^(x^2), x)",
        "Conservar integral residual",
    )];

    for (input, expected_result, expected_rule) in cases {
        let output = cli()
            .args(["eval", input, "--format", "json", "--steps", "on"])
            .output()
            .expect("Failed to run CLI");

        assert!(
            output.status.success(),
            "input: {input}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout = String::from_utf8(output.stdout).unwrap();
        let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");
        let steps = wire["steps"].as_array().expect("steps array");

        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(wire["steps_count"], 1, "input: {input}");
        assert_eq!(steps.len(), 1, "input: {input}");
        assert_eq!(steps[0]["rule"], expected_rule, "input: {input}");
    }
}

#[test]
fn test_eval_trig_log_integral_residual_compacts_cleanup_noise_json() {
    let output = cli()
        .args([
            "eval",
            "integrate(1/((tan(x)-2)*ln(tan(x))), x)",
            "--format",
            "json",
            "--steps",
            "on",
        ])
        .output()
        .expect("Failed to run CLI");

    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");
    let steps = wire["steps"].as_array().expect("steps array");
    let rules: Vec<&str> = steps
        .iter()
        .map(|step| step["rule"].as_str().expect("step rule"))
        .collect();

    assert_eq!(wire["ok"], true);
    assert_eq!(
        wire["result"],
        "integrate(cos(x) / (ln(sin(x) / cos(x))·(sin(x) - 2·cos(x))), x)"
    );
    assert_eq!(
        wire["required_display"],
        serde_json::json!([
            "cos(x) ≠ 0",
            "tan(x) - 1 ≠ 0",
            "tan(x) - 2 ≠ 0",
            "tan(x) > 0"
        ])
    );
    assert_eq!(
        rules,
        vec![
            "Expandir tangente como seno entre coseno",
            "Conservar integral residual"
        ]
    );
}

#[test]
fn test_eval_diff_sign_polynomial_returns_zero_with_nonzero_domain_json() {
    let output = cli()
        .args([
            "eval",
            "diff(sign(x), x)",
            "--format",
            "json",
            "--steps",
            "on",
        ])
        .output()
        .expect("Failed to run CLI");

    assert!(
        output.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");
    let steps = wire["steps"].as_array().expect("steps array");
    let first_substeps = steps[0]["substeps"].as_array().expect("substeps array");

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "0");
    assert_eq!(wire["required_display"], serde_json::json!(["x ≠ 0"]));
    assert_eq!(steps[0]["rule"], "Calcular la derivada");
    assert!(
        first_substeps
            .iter()
            .any(|substep| substep["title"] == "Usar derivada de sign(u) fuera de u = 0"),
        "missing sign derivative substep: {first_substeps:?}"
    );
}

#[test]
fn test_eval_inverse_trig_canonical_reciprocal_root_exact_values_json() {
    let cases = [
        ("arcsin(sqrt(2)/2)", "1/4\u{00b7}pi"),
        ("arccos(sqrt(2)/2)", "1/4\u{00b7}pi"),
        ("arcsin(sqrt(3)/2)", "1/3\u{00b7}pi"),
        ("arccos(sqrt(3)/2)", "1/6\u{00b7}pi"),
        ("arctan(3^(-1/2))", "1/6\u{00b7}pi"),
    ];

    for (input, expected) in cases {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");

        assert!(output.status.success(), "input: {input}");

        let stdout = String::from_utf8(output.stdout).unwrap();
        let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");

        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(wire["result"], expected, "input: {input}");
        assert_eq!(wire["warnings"], serde_json::json!([]), "input: {input}");
        assert_eq!(
            wire["required_display"],
            serde_json::json!([]),
            "input: {input}"
        );
    }
}

#[test]
fn test_eval_wire_returns_undefined_for_empty_diff_domain() {
    let output = cli()
        .args(["eval", "diff(atanh(sqrt(x^2+2)), x)", "--format", "json"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");
    assert_eq!(wire["result"], "undefined");
    assert_eq!(
        wire["blocked_hints"],
        serde_json::Value::Null,
        "undefined domain is now explained by the derivative step, not a blocked residual"
    );
}

#[test]
fn test_eval_scaled_arctan_surd_diff_stays_off_rationalize_overflow_route() {
    let output = cli()
        .args([
            "eval",
            "diff(7*arctan((2*x+1)/sqrt(3))/sqrt(3), x)",
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "7 / (2\u{00b7}(x^2 + x + 1))");
    assert!(
        !stderr.contains("depth_overflow") && !stderr.contains("WARN"),
        "scaled arctan-surd diff should use the compact route, got stderr:\n{stderr}"
    );
}

#[test]
fn test_eval_atanh_exact_square_symbolic_denominator_diff_stays_compact_without_overflow() {
    let input = "diff(atanh(sqrt(4*x+4)/a), x)";
    let output = cli()
        .args(["eval", input, "--format", "json", "--steps", "on"])
        .output()
        .expect("Failed to run CLI");

    assert!(
        output.status.success(),
        "CLI failed for {input}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");

    assert_eq!(wire["ok"], true);
    assert_eq!(
        wire["result"],
        "a / (sqrt(x + 1)\u{00b7}(a^2 - 4\u{00b7}x - 4))"
    );
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["a \u{2260} 0", "a^2 - 4\u{00b7}x - 4 > 0", "x > -1"])
    );
    assert!(
        !stderr.contains("depth_overflow") && !stderr.contains("WARN"),
        "atanh exact-square symbolic denominator diff should stay off the overflow route, got stderr:\n{stderr}"
    );

    let steps = wire["steps"].as_array().expect("steps should be present");
    for expected_rule in [
        "Reconocer un cuadrado perfecto bajo la raíz",
        "Sacar constante de una fracción",
        "Calcular la derivada",
    ] {
        assert!(
            steps.iter().any(|step| step["rule"] == expected_rule),
            "expected rule {expected_rule} in public trace for {input}, got {steps:?}"
        );
    }
    let derivative_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la derivada")
        .expect("expected derivative step");
    let substeps = derivative_step["substeps"]
        .as_array()
        .expect("derivative step should expose substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar regla de la cadena"),
        "expected chain-rule substep for {input}, got {substeps:?}"
    );
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Identificar u y du"),
        "expected u/du substep for {input}, got {substeps:?}"
    );
}

#[test]
fn test_eval_diff_periodic_required_display_preserves_argument_scale() {
    let input = "diff(sec((3*x+2)/2), x)";
    let output = cli()
        .args(["eval", input, "--format", "json", "--steps", "on"])
        .output()
        .expect("Failed to run CLI");

    assert!(
        output.status.success(),
        "CLI failed for {input}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");

    assert_eq!(wire["ok"], true);
    assert_eq!(
        wire["result"],
        "3/2\u{00b7}sec((3\u{00b7}x + 2) / 2)\u{00b7}tan((3\u{00b7}x + 2) / 2)"
    );
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["cos((3\u{00b7}x + 2) / 2) \u{2260} 0"])
    );
    let required_text = wire["required_display"]
        .as_array()
        .expect("required_display array")
        .iter()
        .map(|value| value.as_str().expect("required_display string"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !required_text.contains("cos(3\u{00b7}x + 2) \u{2260} 0"),
        "periodic zero-set display must not scale-normalize the argument for {input}: {required_text}"
    );
    assert!(
        !stderr.contains("depth_overflow") && !stderr.contains("WARN"),
        "periodic required-display diff should stay off fragile routes for {input}, got stderr:\n{stderr}"
    );

    let steps = wire["steps"].as_array().expect("steps should be present");
    assert!(
        steps
            .iter()
            .any(|step| step["rule"] == "Calcular la derivada"),
        "expected derivative step in public trace for {input}, got {steps:?}"
    );
}

#[test]
fn test_eval_plain_reciprocal_trig_log_diff_stays_off_depth_overflow_route() {
    let cases = [
        (
            "diff(ln(sec(sqrt(x))+tan(sqrt(x))), x)",
            "1 / (2\u{00b7}sqrt(x)\u{00b7}cos(sqrt(x)))",
            vec![
                "cos(sqrt(x)) \u{2260} 0",
                "tan(sqrt(x)) + sec(sqrt(x)) > 0",
                "x > 0",
            ],
        ),
        (
            "diff(ln(csc(sqrt(x))-cot(sqrt(x))), x)",
            "1 / (2\u{00b7}sqrt(x)\u{00b7}sin(sqrt(x)))",
            vec![
                "sin(sqrt(x)) \u{2260} 0",
                "csc(sqrt(x)) - cot(sqrt(x)) > 0",
                "x > 0",
            ],
        ),
        (
            "diff(ln(sec(sqrt(3*x+1))+tan(sqrt(3*x+1))), x)",
            "3 / (2\u{00b7}sqrt(3\u{00b7}x + 1)\u{00b7}cos(sqrt(3\u{00b7}x + 1)))",
            vec![
                "cos(sqrt(3\u{00b7}x + 1)) \u{2260} 0",
                "tan(sqrt(3\u{00b7}x + 1)) + sec(sqrt(3\u{00b7}x + 1)) > 0",
                "x > -1/3",
            ],
        ),
        (
            "diff(ln(csc(sqrt(3*x+1))-cot(sqrt(3*x+1))), x)",
            "3 / (2\u{00b7}sqrt(3\u{00b7}x + 1)\u{00b7}sin(sqrt(3\u{00b7}x + 1)))",
            vec![
                "sin(sqrt(3\u{00b7}x + 1)) \u{2260} 0",
                "csc(sqrt(3\u{00b7}x + 1)) - cot(sqrt(3\u{00b7}x + 1)) > 0",
                "x > -1/3",
            ],
        ),
    ];

    for (input, expected_result, expected_required_display) in cases {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .unwrap_or_else(|err| panic!("Failed to run CLI for {input}: {err}"));

        assert!(output.status.success(), "CLI failed for {input}");

        let stdout = String::from_utf8(output.stdout).unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");

        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(wire["result"], expected_result, "input: {input}");
        let result_latex = wire["result_latex"]
            .as_str()
            .expect("result_latex should be a string");
        assert!(
            result_latex.contains("\\sqrt") && !result_latex.contains("}^{-"),
            "post-calculus LaTeX should mirror the compact reciprocal-root display for {input}, got: {result_latex}"
        );
        let mut actual_required = wire["required_display"]
            .as_array()
            .expect("required_display array")
            .iter()
            .map(|condition| condition.as_str().expect("required condition").to_owned())
            .collect::<Vec<_>>();
        let mut expected_required = expected_required_display
            .iter()
            .map(|condition| (*condition).to_string())
            .collect::<Vec<_>>();
        actual_required.sort();
        expected_required.sort();
        assert_eq!(actual_required, expected_required, "input: {input}");
        assert!(
            wire.get("blocked_hints").is_none(),
            "successful reciprocal-trig log diff should not surface non-actionable cycle hints for {input}: {:?}",
            wire["blocked_hints"]
        );
        assert!(
            !stderr.contains("depth_overflow") && !stderr.contains("WARN"),
            "plain reciprocal-trig log diff should use the direct route for {input}, got stderr:\n{stderr}"
        );
    }

    let output = cli()
        .args([
            "eval",
            "diff(ln(sec(sqrt(x))+tan(sqrt(x))), x)",
            "--format",
            "json",
            "--steps",
            "on",
        ])
        .output()
        .expect("Failed to run CLI with steps enabled");
    assert!(output.status.success());
    let wire: Value =
        serde_json::from_slice(&output.stdout).expect("Invalid wire output with steps enabled");
    let final_step = wire["steps"]
        .as_array()
        .expect("steps array")
        .last()
        .expect("final step");
    let final_after_latex = final_step["after_latex"]
        .as_str()
        .expect("final after_latex string");
    assert!(
        final_after_latex.contains("\\sqrt{x}") && !final_after_latex.contains("{x}^{-"),
        "post-calculus step LaTeX should mirror the compact reciprocal-root display, got: {final_after_latex}"
    );
    for step in wire["steps"].as_array().expect("steps array") {
        for field in ["rule_latex", "before_latex", "after_latex"] {
            let Some(latex) = step[field].as_str() else {
                continue;
            };
            assert!(
                !latex.contains("{x}^{-"),
                "calculus step {field} should not leak reciprocal-root power notation, got: {latex}"
            );
        }
    }
}

#[test]
fn test_eval_plain_log_tan_sqrt_diff_uses_sqrt_in_scaled_trig_argument() {
    let output = cli()
        .args(["eval", "diff(ln(tan(sqrt(x))), x)", "--format", "json"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());
    let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");

    assert_eq!(wire["ok"], true);
    assert_eq!(wire["result"], "1 / (sin(2\u{00b7}sqrt(x))\u{00b7}sqrt(x))");
    assert!(
        !wire["result"]
            .as_str()
            .expect("result string")
            .contains("x^(1/2)"),
        "post-calculus display should not leak half-power notation in scaled trig arguments"
    );
    let mut required_display = wire["required_display"]
        .as_array()
        .expect("required_display array")
        .iter()
        .map(|value| value.as_str().expect("required_display string"))
        .collect::<Vec<_>>();
    required_display.sort_unstable();
    // `cos(sqrt(x)) ≠ 0` is now listed explicitly (tan(sqrt(x)) requires it), matching the sibling
    // `diff(ln(tan(sqrt(x))+1))` case below — the derivative is only valid where the original
    // tan-containing function is defined, even though `cos≠0` is implied by `tan(sqrt(x)) > 0`.
    assert_eq!(
        required_display,
        ["cos(sqrt(x)) \u{2260} 0", "tan(sqrt(x)) > 0", "x > 0"]
    );
}

#[test]
fn test_diff_cancelling_reciprocal_trig_product_keeps_domain_condition() {
    // A reciprocal-trig factor (tan/sec → cos≠0, cot/csc → sin≠0) that CANCELS away in a product
    // must still impose its domain condition on the derivative: the original function is undefined
    // where the cancelled factor blew up, so the derivative does not exist there either. Before the
    // fix these returned the derivative with NO condition (e.g. diff(tan(x)*cos(x)) → cos(x) on all
    // of ℝ, though tan(x)·cos(x) is undefined at cos(x)=0).
    for (input, expected) in [
        ("diff(sec(x)*cos(x), x)", "cos(x) \u{2260} 0"),
        ("diff(tan(x)*cos(x), x)", "cos(x) \u{2260} 0"),
        ("diff(cot(x)*sin(x), x)", "sin(x) \u{2260} 0"),
        ("diff(sin(x)*cot(x), x)", "sin(x) \u{2260} 0"),
        ("diff(csc(x)*sin(x), x)", "sin(x) \u{2260} 0"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        let displays = wire["required_display"]
            .as_array()
            .expect("required_display array");
        assert!(
            displays.iter().any(|v| v.as_str() == Some(expected)),
            "{input}: expected required condition `{expected}`, got {displays:?}"
        );
    }

    // Already-conditioned single function must NOT gain a duplicate: diff(tan(x)) carries exactly
    // one cos(x) ≠ 0 (from the 1/cos² result), and the differand re-attachment dedupes against it.
    let output = cli()
        .args(["eval", "diff(tan(x), x)", "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
    let cos_conditions = wire["required_display"]
        .as_array()
        .expect("required_display array")
        .iter()
        .filter(|v| v.as_str() == Some("cos(x) \u{2260} 0"))
        .count();
    assert_eq!(
        cos_conditions, 1,
        "diff(tan(x)) must not duplicate cos(x) ≠ 0"
    );
}

#[test]
fn test_eval_matrix_commutator_does_not_collapse_to_zero() {
    // Matrix multiplication is non-commutative, so the commutator A·B − B·A is
    // generally nonzero. The engine's exact-zero / equivalent-pair root shortcuts
    // and the additive cancellation matchers compare products as commutative
    // factor multisets, which previously collapsed A·B − B·A to 0 (a wrong
    // answer). The bug only surfaced in the steps-off fast path (the steps-on
    // path evaluates the products first), so BOTH modes are checked here.
    let cases = [
        (
            "[[1,2],[3,4]]*[[5,6],[7,8]] - [[5,6],[7,8]]*[[1,2],[3,4]]",
            "[[-4, -12], [12, 4]]",
        ),
        (
            // Nilpotent generators: [E12, E21] = E11 − E22.
            "[[0,1],[0,0]]*[[0,0],[1,0]] - [[0,0],[1,0]]*[[0,1],[0,0]]",
            "[[1, 0], [0, -1]]",
        ),
        (
            "[[5,6],[7,8]]*[[1,2],[3,4]] - [[1,2],[3,4]]*[[5,6],[7,8]]",
            "[[4, 12], [-12, -4]]",
        ),
    ];
    for (input, expected) in cases {
        for mode in ["off", "on"] {
            let output = cli()
                .args(["eval", input, "--format", "json", "--steps", mode])
                .output()
                .expect("Failed to run CLI");
            assert!(output.status.success(), "{input} (steps={mode})");
            let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
            assert_eq!(
                wire["result"].as_str(),
                Some(expected),
                "{input} (steps={mode}): matrix commutator must not collapse to 0"
            );
        }
    }

    // A genuinely identical product difference A·B − A·B is still the zero matrix
    // / 0 (order-preserving structural equality is sound), and the scalar
    // commutator x·y − y·x stays 0 (scalar multiplication IS commutative).
    for (input, expected) in [
        (
            "[[1,2],[3,4]]*[[5,6],[7,8]] - [[1,2],[3,4]]*[[5,6],[7,8]]",
            "0",
        ),
        ("x*y - y*x", "0"),
    ] {
        for mode in ["off", "on"] {
            let output = cli()
                .args(["eval", input, "--format", "json", "--steps", mode])
                .output()
                .expect("Failed to run CLI");
            let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
            assert_eq!(
                wire["result"].as_str(),
                Some(expected),
                "{input} (steps={mode}): genuine cancellation must still hold"
            );
        }
    }
}

#[test]
fn test_eval_cosh_cube_difference_does_not_collapse_to_zero() {
    // cosh³(x) − cosh(x) = cosh(x)·(cosh²(x) − 1) = cosh(x)·sinh²(x), which is
    // NOT identically 0. The "Hyperbolic Pythagorean Identity Cancellation
    // Bridge" rule recognised the FactorThenRewrite pattern and, at the root,
    // unconditionally collapsed it to 0 (a wrong-answer, e.g. cosh(3x)−cosh(x)
    // → 0). The fix declines that standalone case, leaving the correct expanded
    // form (just as a plain polynomial y³−y is left unfactored). (The sin/cos
    // analogues already worked.)
    for (input, expected) in [
        ("cosh(x)^3 - cosh(x)", "cosh(x)^3 - cosh(x)"),
        ("4*cosh(x)^3 - 4*cosh(x)", "4·cosh(x)^3 - 4·cosh(x)"),
        // cosh(3x) expands (triple angle) to 4cosh³−3cosh; the difference is
        // 4cosh³−4cosh = 4cosh·sinh², never 0.
        ("cosh(3*x) - cosh(x)", "4·cosh(x)^3 - 4·cosh(x)"),
    ] {
        for mode in ["off", "on"] {
            let output = cli()
                .args(["eval", input, "--format", "json", "--steps", mode])
                .output()
                .expect("Failed to run CLI");
            assert!(output.status.success(), "{input} (steps={mode})");
            let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
            let result = wire["result"].as_str();
            assert_ne!(
                result,
                Some("0"),
                "{input} (steps={mode}): hyperbolic cube difference must not collapse to 0"
            );
            assert_eq!(result, Some(expected), "{input} (steps={mode})");
        }
    }

    // The genuinely-zero hyperbolic Pythagorean identities must still collapse:
    // 4cosh·sinh² + 4cosh − 4cosh³ = 4cosh(sinh² + 1 − cosh²) = 0.
    for input in [
        "4*cosh(x)*sinh(x)^2 + 4*cosh(x) - 4*cosh(x)^3",
        "sinh(2*x+1)*(cosh(2*x+1)^2 - 1) - sinh(2*x+1)^3",
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(
            wire["result"].as_str(),
            Some("0"),
            "{input}: genuine hyperbolic Pythagorean zero identity must still collapse"
        );
    }
}

#[test]
fn test_eval_sum_of_absolute_values_inequality_solves_piecewise() {
    // A SUM of absolute values is piecewise-linear, so the inequality has a real
    // interval solution. The old "isolate one abs and split cases" strategy lost
    // the other terms and wrongly returned "No solution" (or a malformed
    // residual for `>`/`>=`). The piecewise/breakpoint solver returns the exact
    // union of intervals. Ground truth cross-checked against sympy.
    for (input, expected) in [
        ("abs(x) + abs(x-1) < 5", "(-2, 3)"),
        ("abs(x) + abs(x-1) <= 3", "[-1, 2]"),
        // |x|+|x-1| = 1 on all of [0,1], so `<= 1` is the whole closed interval,
        // not just its endpoints (a discrete-vs-interval merge bug guard).
        ("abs(x) + abs(x-1) <= 1", "[0, 1]"),
        ("abs(x-2) + abs(x+2) < 6", "(-3, 3)"),
        ("abs(x) + abs(x+1) < 4", "(-5/2, 3/2)"),
        ("2*abs(x) + abs(x-3) < 8", "(-5/3, 11/3)"),
        // Rational breakpoints (slope 2): bps at -1/2, 1/2.
        ("abs(2*x-1) + abs(2*x+1) < 4", "(-1, 1)"),
        // `>` was previously malformed; now a union of two open rays.
        ("abs(x) + abs(x-1) > 5", "(-infinity, -2) U (3, infinity)"),
        ("abs(x) + abs(x-1) >= 1", "All real numbers"),
        // Three terms.
        ("abs(x) + abs(x-1) + abs(x-2) < 4", "(-1/3, 7/3)"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }

    // Genuinely-empty sums (min of |x|+|x-1| is 1) must still report No solution,
    // and single-abs / non-abs inequalities must be unchanged by the new path.
    for (input, expected) in [
        ("abs(x) + abs(x-1) < 1", "No solution"),
        ("abs(x-2) + abs(x+2) < 3", "No solution"),
        ("abs(x) < 5", "(-5, 5)"),
        ("abs(x-3) >= 2", "(-infinity, 1] U [5, infinity)"),
        ("x^2 - 4 < 0", "(-2, 2)"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
}

#[test]
fn test_eval_sum_of_absolute_values_equation_solves_piecewise() {
    // The same piecewise/breakpoint solver handles EQUATIONS. The old
    // isolate-one-abs strategy leaked a malformed nested-`solve` residual (and
    // for the flat-minimum case wrongly returned a half-line). On each segment a
    // strictly-linear piece contributes its single crossing; a constant piece
    // equal to the target contributes the whole segment. Cross-checked against an
    // independent exact (fractions) oracle over 400 random sums (0 mismatches).
    for (input, expected) in [
        // Above the minimum (1): two isolated crossings.
        ("abs(x) + abs(x-1) = 3", "{ -1, 2 }"),
        ("abs(x) + abs(x-1) = 2", "{ -1/2, 3/2 }"),
        ("abs(x-2) + abs(x+2) = 8", "{ -4, 4 }"),
        ("abs(x) + abs(x-1) + abs(x-2) = 4", "{ -1/3, 7/3 }"),
        ("2*abs(x) + abs(x-3) = 6", "{ -1, 3 }"),
        // At the flat minimum: the whole closed segment is the solution set.
        ("abs(x) + abs(x-1) = 1", "[0, 1]"),
        ("abs(x+1) + abs(x-1) = 2", "[-1, 1]"),
        // Below the minimum: empty.
        ("abs(x) + abs(x-1) = 1/2", "No solution"),
        // Non-convex signed coefficients: a flat piece yields a ray, and a single
        // crossing yields a point (convexity is NOT assumed by the solver).
        ("abs(x) - abs(x-1) = 0", "{ 1/2 }"),
        ("abs(x) - abs(x-1) = -1", "(-infinity, 0]"),
        ("abs(x) - abs(x-1) = 1", "[1, infinity)"),
        // Affine remainder term folded into the per-segment line.
        ("abs(x) + abs(x-1) + x = 3", "{ -2, 4/3 }"),
        // Single-abs equations are untouched (still the existing path).
        ("abs(x) = 3", "{ 3, -3 }"),
        ("abs(2*x-1) = 5", "{ 3, -2 }"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
}

#[test]
fn test_eval_rational_power_polynomial_equation_solves_by_substitution() {
    // Equations that are a polynomial of degree >= 2 in x^(1/q) (a
    // quadratic-in-disguise) used to leak a malformed internal `Solve: solve(...)`
    // residual under ok=true and drop every root. They are now solved by the
    // u = x^(1/q) substitution, with the correct real-root domain on
    // back-substitution: even q drops negative u-roots, odd q keeps them.
    // Cross-checked against an independent exact (fractions) oracle over 300
    // random cases (0 mismatches).
    for (input, expected) in [
        // Quadratic in sqrt(x): even root, both u-roots non-negative.
        ("x - 3*sqrt(x) + 2 = 0", "{ 1, 4 }"),
        ("x - 5*sqrt(x) + 6 = 0", "{ 4, 9 }"),
        // A negative u-root is dropped by the even-root domain (sqrt(x) = -3 has no
        // real solution), leaving only the valid root.
        ("x + sqrt(x) - 6 = 0", "{ 4 }"),
        // Quadratic in x^(1/3): the ODD root keeps the negative u-root (x^(1/3) = -1).
        ("x^(2/3) - x^(1/3) - 2 = 0", "{ -1, 8 }"),
        ("x^(2/3) + x^(1/3) - 6 = 0", "{ -27, 8 }"),
        // sqrt(x)^2 normalizes to x; still a quadratic in sqrt(x).
        ("sqrt(x)^2 - 3*sqrt(x) + 2 = 0", "{ 1, 4 }"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }

    // Must NOT disturb the existing paths: plain polynomials, exponential
    // substitution, single-power equations, and surd-root quadratics are unchanged.
    for (input, expected) in [
        ("x^4 - 5*x^2 + 4 = 0", "{ -2, -1, 1, 2 }"),
        ("e^(2*x) - 3*e^x + 2 = 0", "{ ln(2), 0 }"),
        ("sqrt(x) = 2", "{ 4 }"),
        ("x^2 - 5*x + 6 = 0", "{ 2, 3 }"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
}

#[test]
fn test_eval_log_polynomial_equation_solves_by_substitution() {
    // Equations that are a polynomial of degree >= 2 in ln(x) used to leak a
    // malformed `Solve: solve(x - e^(...))` residual and drop every root. They are
    // now solved by the u = ln(x) substitution, back-substituting ln(x) = u_root
    // (= e^(u_root), the existing path with the ln domain). Cross-checked against
    // an independent oracle over 250 random ln-polynomials (0 mismatches).
    for (input, expected) in [
        ("ln(x)^2 - ln(x) - 2 = 0", "{ 1 / e, e^2 }"),
        ("ln(x)^2 - 3*ln(x) + 2 = 0", "{ e, e^2 }"),
        ("ln(x)^2 = ln(x)", "{ 1, e }"),
        ("ln(x)^2 - 1 = 0", "{ 1 / e, e }"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }

    // Single-log equations are unchanged (degree-1, handled directly).
    for (input, expected) in [("ln(x) = 2", "{ e^2 }"), ("ln(x) - 1 = 0", "{ e }")] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
}

#[test]
fn test_eval_sum_of_two_radicals_equation_solves_and_verifies() {
    // `√f + √g = c` used to leak `Solve: solve(x - (c - √g)^(1/(1/2)) = 0, x) = 0`
    // and drop the root. It is now reduced by squaring to the single radical
    // `√(f·g) = (c² - f - g)/2`, solved, and each candidate VERIFIED exactly
    // against the original (both radicands perfect rational squares summing to c),
    // dropping extraneous roots. Cross-checked against an independent oracle over
    // 300 random cases (0 mismatches).
    for (input, expected) in [
        ("sqrt(x+3) + sqrt(x) = 3", "{ 1 }"),
        ("sqrt(x+1) + sqrt(x) = 1", "{ 0 }"),
        ("sqrt(x+5) + sqrt(x) = 5", "{ 4 }"),
        ("sqrt(x-1) + sqrt(x+4) = 5", "{ 5 }"),
        // Symmetric radicands (difference of squares under the reduction).
        ("sqrt(x+1) + sqrt(x-1) = 3", "{ 85/36 }"),
        ("sqrt(x-2) + sqrt(x+2) = 5", "{ 641/100 }"),
        // No real solution: the single candidate is extraneous (or the minimum of
        // the LHS exceeds c) — verification drops it.
        ("sqrt(x) + sqrt(x+8) = 2", "No solution"),
        ("sqrt(x+1) + sqrt(x) = 0", "No solution"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
}

#[test]
fn test_eval_radical_inequality_keeps_argument_domain() {
    // `sqrt(g(x)) {<,<=} c` requires g(x) >= 0, but for a COMPOUND argument the
    // engine dropped that domain, returning e.g. `sqrt(x-1) < 3 → (-inf, 10)`
    // (which wrongly includes points where the radicand is negative) instead of
    // `[1, 10)`. The fix intersects with the solved argument domain `g(x) >= 0`
    // (`g(x) > 0` for ln). Ground truth cross-checked against sympy.
    for (input, expected) in [
        ("sqrt(x-1) < 3", "[1, 10)"),
        ("sqrt(2*x-1) <= 3", "[1/2, 5]"),
        // Bare-variable argument unchanged.
        ("sqrt(x) < 2", "[0, 4)"),
        ("sqrt(x) >= 2", "[4, infinity)"),
        // `>` / `>=` already implied the domain via the bound; still correct.
        ("sqrt(x-1) > 2", "(5, infinity)"),
        ("sqrt(x+2) > 1", "(-1, infinity)"),
        // Range correction (sqrt ≥ 0): a negative upper threshold is impossible.
        ("sqrt(x-1) < -1", "No solution"),
        // sqrt(g) <= 0 forces g = 0: a single point in the domain.
        ("sqrt(x+3) <= 0", "{ -3 }"),
        // ln argument domain is g(x) > 0 (open).
        ("ln(x-1) < 0", "(1, 2)"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }

    // Quadratic radicand: the domain `x²-4 >= 0` splits the solution into two
    // intervals (the lone interval before the fix dropped the |x|>=2 domain).
    // The `-√13` lower bound renders via the existing surd-bound style; assert
    // the structural domain split rather than the exact surd spelling.
    let output = cli()
        .args(["eval", "sqrt(x^2-4) < 3", "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
    let result = wire["result"].as_str().expect("result string");
    assert!(
        result.contains("-2]") && result.contains("[2,") && result.contains(" U "),
        "sqrt(x^2-4) < 3 must split on the x²-4>=0 domain, got {result:?}"
    );
}

#[test]
fn test_eval_radical_inequality_case_splits_on_rhs_sign() {
    // A radical inequality `√f {op} g` with a NON-constant RHS must case-split on
    // the sign of g, not square blindly. Squaring loses the RHS-sign branches:
    // `√x < x-2` is `(4, ∞)` (the `[0,1)` the naive square keeps fails `g > 0`),
    // and `√(x-2) > 4-x` is `(3, ∞)` (`4-x < 0` already satisfies `>` for x > 4).
    // For a LINEAR radicand the domain `f >= 0` is rational-bounded, so the
    // case-split intersections compare rational-vs-surd endpoints exactly. Verified
    // against an independent membership oracle over 350 random cases (0 mismatches).
    for (input, expected) in [
        ("sqrt(x) < x-2", "(4, infinity)"),
        ("sqrt(x) < x+1", "[0, infinity)"),
        ("sqrt(x+1) > x-1", "[-1, 3)"),
        ("sqrt(x-2) > 4-x", "(3, infinity)"),
        // Non-strict touch point `√f = g = 0` is an isolated solution the squared
        // intersection drops as a degenerate overlap; recovered via `solve(√f = g)`.
        ("sqrt(x+3) <= -x-3", "{ -3 }"),
        // Detached point unioned with an interval: `√0 = 0 = -2+2` AND [0, ∞).
        ("sqrt(2*x+4) <= x+2", "[-2, -2] U [0, infinity)"),
        // Boundary at the open endpoint of a non-empty branch stays closed.
        ("sqrt(2*x+4) <= 2*x-2", "[5/2, infinity)"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
}

#[test]
fn test_eval_radical_inequality_fractional_constant_and_degenerate() {
    // Hardening cases an adversarial workflow surfaced (all previously wrong):
    for (input, expected) in [
        // FRACTIONAL RHS slope: g² must be built EXPANDED, not Pow(g,2) (the factored
        // form dropped the squared outer rational factor → wrong "No solution").
        ("sqrt(x) < x/2 - 3", "(2·(sqrt(7) + 4), infinity)"),
        ("sqrt(4*x+2) >= (1/2)*x - 6", "[-1/2, 2·(sqrt(66) + 10)]"),
        // Fractional RHS in a NON-STRICT branch: the boundary `√f = g` is now solved as
        // the polynomial `f = g² ∧ g >= 0` (the radical-equation solver leaks on
        // fractional g). The `[2, ...]` endpoint stays CLOSED.
        (
            "sqrt(x^2-4) <= (1/2)*x+5",
            "[2/3·(5 - 4·sqrt(7)), -2] U [2, 2/3·(4·sqrt(7) + 5)]",
        ),
        // Boundary touch with fractional g: `√(9-x²) = (1/3)x-1` at x=3 (`√0=0`).
        ("sqrt(-x^2+9) <= (1/3)*x-1", "{ 3 }"),
        // CONSTANT g: `solve(const, x)` errors, so the sign is taken from the constant.
        ("sqrt(4-x^2) < 5", "[-2, 2]"),
        ("sqrt(x-2) >= 0*x - 4", "[2, infinity)"),
        // DEGENERATE radicand: `-x²` has domain {0}; the single-point `f >= 0` must
        // survive the case-split intersections (a bare Discrete operand collapsed to ∅).
        ("sqrt(-x^2) < x+1", "{ 0 }"),
        ("sqrt(-(x-1)^2) < x", "{ 1 }"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
}

#[test]
fn test_eval_ln_of_even_numerator_power_uses_abs() {
    // `ln(x^(p/q))` with q ODD and p EVEN is real for EVERY x != 0 (under the
    // engine's real power semantics `(-8)^(2/3) = 4`), so it expands to
    // `(p/q)·ln|x|` over the domain x != 0. The engine used to emit `(p/q)·ln(x)`,
    // which wrongly NARROWS the domain to x > 0 (dropping the x < 0 branch).
    for (input, expected) in [
        ("ln(x^(2/3))", "2/3·ln(|x|)"),
        ("ln(x^(4/3))", "4/3·ln(|x|)"),
        ("ln(x^(2/5))", "2/5·ln(|x|)"),
        ("ln(x^(-2/3))", "-2/3·ln(|x|)"),
        ("ln(x^(6/3))", "2·ln(|x|)"), // reduces to the even integer 2
        // Even INTEGER already used |x|; unchanged.
        ("ln(x^2)", "2·ln(|x|)"),
        // ODD numerator keeps the sign of x -> domain x > 0, bare ln(x).
        ("ln(x^(1/3))", "1/3·ln(x)"),
        ("ln(x^(5/3))", "5/3·ln(x)"),
        ("ln(x^3)", "3·ln(x)"),
        // q EVEN forces x >= 0 already -> bare ln(x).
        ("ln(x^(1/2))", "1/2·ln(x)"),
        ("ln(x^(3/2))", "3/2·ln(x)"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
}

#[test]
fn test_eval_even_integrand_negative_interval_reflects() {
    // An EVEN integrand over a strictly-negative interval reflects to the positive
    // branch (`∫_{-3}^{-2} f = ∫_2^3 f`). `√(x²-1)`'s antiderivative uses `acosh` (real
    // only for arg >= 1), so the negative branch used to decline; it now evaluates to
    // the SAME closed form as the reflected positive interval.
    for (neg, pos) in [
        (
            "integrate(sqrt(x^2-1), x, -3, -2)",
            "integrate(sqrt(x^2-1), x, 2, 3)",
        ),
        (
            "integrate(sqrt(x^2-4), x, -5, -3)",
            "integrate(sqrt(x^2-4), x, 3, 5)",
        ),
        (
            "integrate(1/sqrt(x^2-1), x, -3, -2)",
            "integrate(1/sqrt(x^2-1), x, 2, 3)",
        ),
    ] {
        let run = |e: &str| -> String {
            let out = cli()
                .args(["eval", e, "--format", "json"])
                .output()
                .expect("run");
            let w: Value = serde_json::from_slice(&out.stdout).expect("json");
            w["result"].as_str().unwrap_or("").to_string()
        };
        let r = run(neg);
        assert_eq!(r, run(pos), "{neg}");
        assert!(!r.contains("integrate("), "{neg} should evaluate, got {r}");
    }
}

#[test]
fn test_eval_sqrt_of_perfect_square_inequality_is_abs() {
    // `√(perfect square) {op} affine` is `|·| {op} affine`: `√(x²-6x+9) = |x-3|`. The
    // solve path used to keep the raw radical and emit a wrong conditional
    // (`√(x²-6x+9) > x-3 → "All real numbers if x-3 >= 0"`). Simplifying the sides before
    // the abs hook collapses `√(square) → |·|` so the exact segment method applies.
    for (input, expected) in [
        ("sqrt(x^2-6*x+9) > x-3", "(-infinity, 3)"),
        ("sqrt(x^2-6*x+9) <= x-3", "[3, infinity)"),
        ("sqrt((x-3)^2) > x-3", "(-infinity, 3)"),
        ("sqrt(x^2) > x", "(-infinity, 0)"),
        ("sqrt(x^2) >= x", "All real numbers"),
        ("sqrt(x^2) < x", "No solution"),
        ("sqrt((x-1)^2) <= x", "[1/2, infinity)"),
        ("sqrt((x-2)^2) < x", "(1, infinity)"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
}

#[test]
fn test_eval_single_abs_inequality_uses_segment_method() {
    // A SINGLE `|f| {op} g` with an affine (non-constant) RHS used to fall to the
    // isolate-one-abs path, which solves the boundary EQUATION and returns the root
    // (`|x| > x+1 → {-1/2}`) or "No solution" instead of the interval. Route single-abs
    // INEQUALITIES through the exact piecewise/segment method (single-abs equations and
    // sum-of-abs are unchanged). Verified by a membership oracle over 300 cases.
    for (input, expected) in [
        ("abs(x) > x", "(-infinity, 0)"),
        ("abs(x) >= x", "All real numbers"),
        ("abs(x) < x", "No solution"),
        ("abs(x-3) > x-3", "(-infinity, 3)"),
        ("abs(x-1) <= x-1", "[1, infinity)"),
        ("abs(x) > x+1", "(-infinity, -1/2)"),
        ("abs(x) < x+1", "(-1/2, infinity)"),
        ("abs(2*x) <= x+3", "[-1, 3]"),
        ("abs(x-2) > 2*x", "(-infinity, 2/3)"),
        // Unchanged: abs vs constant, sum-of-abs, single-abs equation.
        ("abs(x) > 2", "(-infinity, -2) U (2, infinity)"),
        ("abs(x)+abs(x-1) < 3", "(-1, 2)"),
        ("abs(x) = x+1", "{ -1/2 }"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
}

#[test]
fn test_eval_infinity_over_infinity_is_undefined() {
    // `∞/∞` is indeterminate; the generic `a/a -> 1` / `(a·X)/(b·X) -> a/b` cancellation used to
    // treat `∞` as a cancellable factor and fabricate a finite value. A dedicated rule now folds it
    // to `undefined` (including finite-scaled, symbolic-scaled, and multi-factor forms and signs).
    for (input, expected) in [
        ("inf/inf", "undefined"),
        ("(2*inf)/inf", "undefined"),
        ("(-inf)/inf", "undefined"),
        ("inf/(2*inf)", "undefined"),
        ("(3*inf)/(-inf)", "undefined"),
        // Finite-scaled `(c·∞)/(d·∞)` must NOT cancel `∞` to `c/d`.
        ("(2*inf)/(5*inf)", "undefined"),
        ("(2*inf)/(2*inf)", "undefined"),
        ("(10*inf)/(4*inf)", "undefined"),
        ("(-2*inf)/(-3*inf)", "undefined"),
        // Symbolic-scaled `(x·∞)/(k·x·∞)` and identical `(x·∞)/(x·∞)` are still `∞/∞`, not `1`.
        ("(x*inf)/(2*x*inf)", "undefined"),
        ("(x*inf)/(x*inf)", "undefined"),
        // Multi-factor products: the shared finite cofactor does not make it finite.
        ("(2*inf*sin(x))/(5*inf*sin(x))", "undefined"),
        ("(inf*sin(x))/(inf*cos(x))", "undefined"),
        // `∞^p` with a positive literal exponent is `∞`: `∞^2/∞^2` is NOT `1`, `∞^3/∞^2` is NOT `∞`.
        ("inf^2/inf^2", "undefined"),
        ("(inf^3)/(inf^2)", "undefined"),
        ("(inf^2)/(inf^3)", "undefined"),
        ("(2*inf^2)/(inf^2)", "undefined"),
        ("(inf^2*x)/(inf^2*y)", "undefined"),
        ("sqrt(inf)/sqrt(inf)", "undefined"),
        // Finite divisions are unaffected.
        ("1/inf", "0"),
        ("2/inf", "0"),
        ("inf/2", "infinity"),
        ("inf/0", "undefined"),
        ("inf-inf", "undefined"),
        ("0*inf", "undefined"),
        ("x/x", "1"),
        // Non-positive / symbolic exponents stay finite or unevaluated (NOT folded).
        ("inf^0", "1"),
        ("inf^(-1)", "0"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
}

#[test]
fn test_eval_infinity_quotient_plain_matches_steps() {
    // CONSISTENCY: an `∞/∞` quotient must evaluate identically whether or not steps are requested.
    // Several cancellation primitives (plain-mode root shortcuts AND per-node Core rules) used to
    // race the `∞/∞ -> undefined` fold; in the default (no-step-listener) path a cancellation won,
    // so `(2·∞)/(5·∞)` returned `2/5` plain but `undefined` with `--steps`. The fold now runs up
    // front in both modes.
    for input in [
        "(2*inf)/(5*inf)",
        "(2*inf)/(2*inf)",
        "(x*inf)/(2*x*inf)",
        "(2*inf*sin(x))/(5*inf*sin(x))",
        "(inf*sin(x))/(inf*cos(x))",
        "inf/inf",
        // Nested `∞/∞`: the fold is recursive, so an enclosing power/root/log/product/sum cannot
        // let the inner quotient escape via a cancellation that runs before the indeterminate fold.
        "((2*inf)/(5*inf))^2",
        "sqrt((2*inf)/(5*inf))",
        "ln((2*inf)/(5*inf))",
        "(2*inf)/(5*inf)*5",
        "2*((2*inf)/(5*inf))",
        "1+(2*inf)/(3*inf)",
        "abs((2*inf)/(5*inf))",
    ] {
        let plain = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(plain.status.success(), "plain {input}");
        let plain_wire: Value = serde_json::from_slice(&plain.stdout).expect("Invalid wire output");

        let steps = cli()
            .args(["eval", input, "--format", "json", "--steps", "on"])
            .output()
            .expect("Failed to run CLI");
        assert!(steps.status.success(), "steps {input}");
        let steps_wire: Value = serde_json::from_slice(&steps.stdout).expect("Invalid wire output");

        assert_eq!(
            plain_wire["result"].as_str(),
            Some("undefined"),
            "plain {input}"
        );
        assert_eq!(
            plain_wire["result"].as_str(),
            steps_wire["result"].as_str(),
            "plain vs --steps divergence for {input}"
        );
    }
}

#[test]
fn test_eval_matrix_power_zero_is_identity_not_scalar_one() {
    // `M^0` is the n×n IDENTITY matrix (the multiplicative identity of the matrix ring), NOT the
    // scalar `1`; a non-square matrix has no `M^0`. The scalar `x^0 -> 1` rule used to collapse a
    // matrix base to `1`, fabricating nonsense (`M^0 + 5 -> 6`, `trace(M^0) -> trace(1)`).
    for (input, expected) in [
        ("[[1,2],[3,4]]^0", "[[1, 0], [0, 1]]"),
        ("[[1,2],[3,4]]^0 + [[1,2],[3,4]]", "[[2, 2], [3, 5]]"),
        ("3*[[1,2],[3,4]]^0", "[[3, 0], [0, 3]]"),
        ("trace([[1,2],[3,4]]^0)", "2"),
        ("[[a,b],[c,d]]^0", "[[1, 0], [0, 1]]"),
        ("[[0,0],[0,0]]^0", "[[1, 0], [0, 1]]"), // ring identity even for the zero matrix
        ("[[1,2,3],[4,5,6]]^0", "undefined"),    // non-square has no M^0
        // Scalar `x^0` is unaffected.
        ("5^0", "1"),
        ("(x+1)^0", "1"),
        ("0^0", "undefined"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
}

#[test]
fn test_eval_common_scale_zero_collapse_requires_exact_zero() {
    // A "collapse to 0" shortcut mistook `1/(x²−1) − 1/(x−1)` (= −x/(x²−1)) for a common-scale
    // cancellation and folded it to 0 — a wrong CONSTANT that then poisoned `solve` into a false
    // "All real numbers". A root shortcut may now only collapse to 0 when the expression exactly
    // vanishes at a generic rational point.
    for (input, expected) in [
        ("1/(x^2-1) - 1/(x-1)", "-x / (x^2 - 1)"),
        ("1/(x-1) - 1/(x^2-1)", "x / (x^2 - 1)"),
        ("1/(2^2-1) - 1/(2-1)", "-2/3"),
        ("solve(1/(x^2-1)=1/(x-1), x)", "{ 0 }"),
        ("solve(1/(x^2-1)-1/(x-1)=0, x)", "{ 0 }"),
        ("solve((x+2)/(x^2-1)=(x+2)/(x-1), x)", "{ -2, 0 }"),
        // Genuine zero differences must STILL collapse (the guard only vetoes non-zero witnesses).
        ("2*x/(x-1) - 2*x/(x-1)", "0"),
        ("(x-1)*(x+1) - (x^2-1)", "0"),
        ("csc(x)^2 - cot(x)^2", "1"),
        ("(a+b)^2 - a^2 - 2*a*b - b^2", "0"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
}

#[test]
fn test_eval_numeric_quotient_plain_matches_steps() {
    // A top-level pure-real `Number/Number` quotient must evaluate IDENTICALLY whether or not
    // steps are requested. A RealOnly "complex noop" root shortcut (`is_real_domain_complex_noop_root`)
    // accepted any bare `Number` as a Gaussian component, so a real `Number/Number` matched and was
    // returned UNEVALUATED in the plain (no-step-listener) path while `--steps` ran the full pipeline.
    // The result then depended on whether steps were asked for — a consistency/soundness defect. Worst
    // case: `1/0` reported `"1 / 0"` with `ok:true` (a division by zero accepted as a valid value)
    // in plain mode but `undefined` with `--steps`. The shortcut now requires an actual imaginary unit
    // `i`, so real quotients fold through the pipeline in both modes.
    for (input, expected) in [
        // Division by zero is undefined — never a valid finite value, in either mode.
        ("1/0", "undefined"),
        ("0/0", "undefined"),
        ("2/0", "undefined"),
        ("100/0", "undefined"),
        // Exact integer quotients fold.
        ("6/3", "2"),
        ("8/4", "2"),
        ("144/12", "12"),
        ("5/1", "5"),
        ("0/7", "0"),
        // Reducible/irreducible rationals fold to lowest terms.
        ("10/4", "5/2"),
        ("9/6", "3/2"),
        ("7/2", "7/2"),
    ] {
        let plain = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(plain.status.success(), "plain {input}");
        let plain_wire: Value = serde_json::from_slice(&plain.stdout).expect("Invalid wire output");

        let steps = cli()
            .args(["eval", input, "--format", "json", "--steps", "on"])
            .output()
            .expect("Failed to run CLI");
        assert!(steps.status.success(), "steps {input}");
        let steps_wire: Value = serde_json::from_slice(&steps.stdout).expect("Invalid wire output");

        assert_eq!(
            plain_wire["result"].as_str(),
            Some(expected),
            "plain {input}"
        );
        assert_eq!(
            plain_wire["result"].as_str(),
            steps_wire["result"].as_str(),
            "plain vs --steps divergence for {input}"
        );
    }
}

#[test]
fn test_eval_rational_constant_inequality_sign_split() {
    // `N/D {op} c` with a polynomial denominator. With `P = N − c·D`, solve `P {op} 0`
    // where `D > 0` and `P {flip op} 0` where `D < 0` (poles excluded), then verify the
    // candidate numerically before returning. The general division path otherwise
    // reciprocates without flipping (`1/(x²+1) < 1/2 → (-1,1)`, `1/x³ < 8 → (-∞,1/2)`,
    // both wrong). Only verified candidates are emitted; an unorderable higher-surd
    // answer (`1/x⁴ > 1/4 → ±4^(1/4)`) declines to its prior behaviour rather than risk
    // a fresh wrong answer (next step: surd-aware interval ordering).
    for (input, expected) in [
        // Positive-definite quadratic denominators (D > 0 everywhere).
        ("1/(x^2+1) < 1/2", "(-infinity, -1) U (1, infinity)"),
        ("2/(x^2+1) < 1", "(-infinity, -1) U (1, infinity)"),
        ("1/(x^2+1) > 2", "No solution"),
        ("5/(x^2+4) <= 1", "(-infinity, -1] U [1, infinity)"),
        ("1/(x^2+1) < 0", "No solution"), // constant target, never holds
        ("1/(x^2+1) >= 0", "All real numbers"),
        // Even-power poles at 0 (D ≥ 0, vanishing at 0): the pole is excluded.
        ("1/x^2 < 4", "(-infinity, -1/2) U (1/2, infinity)"),
        ("1/x^2 > 4", "(-1/2, 0) U (0, 1/2)"),
        ("1/x^2 > 0", "(-infinity, 0) U (0, infinity)"),
        ("1/(x-1)^2 < 4", "(-infinity, 1/2) U (3/2, infinity)"),
        // Sign-varying denominators (linear, odd powers): the sign split flips on D < 0.
        ("1/(x+3) < 1/2", "(-infinity, -3) U (-1, infinity)"),
        ("1/x < 4", "(-infinity, 0) U (1/4, infinity)"),
        ("1/x^3 < 8", "(-infinity, 0) U (1/2, infinity)"),
        ("1/x^4 < 16", "(-infinity, -1/2) U (1/2, infinity)"),
        ("1/x^3 >= -1", "(-infinity, -1] U (0, infinity)"),
        ("2/x^4 >= 2", "[-1, 0) U (0, 1]"),
        // Quadratic-surd / golden-ratio endpoints, compared exactly during verification.
        ("5/x^2 > 1/4", "(-10·5^(-1/2), 0) U (0, 10·5^(-1/2))"),
        (
            "(1+x)/x^2 <= 1",
            "(-infinity, 1/2·(1 - sqrt(5))] U [phi, infinity)",
        ),
        // Numerator and denominator share a factor: the removable pole at 0 stays
        // excluded (NOT cancelled — `x/(x³−x) ≤ 0` is `(-1,0)∪(0,1)`, not `(-1,1)`).
        ("x/(x^3-x) <= 0", "(-1, 0) U (0, 1)"),
        // Reciprocal-power form `x^(-n) {op} c`: the splitter folds it to `c/x^n`, so it
        // routes through the same verified path (was a flipped/inverted wrong answer).
        ("x^(-2) > 4", "(-1/2, 0) U (0, 1/2)"),
        ("x^(-2) < 4", "(-infinity, -1/2) U (1/2, infinity)"),
        ("x^(-3) < 8", "(-infinity, 0) U (1/2, infinity)"),
        ("x^(-4) < 16", "(-infinity, -1/2) U (1/2, infinity)"),
        ("2*x^(-3) >= 2", "(0, 1]"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
}

#[test]
fn test_eval_matrix_inverse_routes_and_no_scalar_broadcast() {
    // `M^(-1)` / `c/M` used to fall to scalar arithmetic and fabricate `1/[[…]]`
    // (a non-square matrix has NO inverse; a symbolic one is not elementwise 1/entry).
    // They now route to the matrix inverse, and `ScalarMatrixRule` no longer broadcasts
    // a matrix-valued operand (e.g. `inverse(M)`) as if it were a scalar.
    for (input, expected) in [
        // Numeric square: the actual inverse.
        ("[[1,2],[3,4]]^(-1)", "[[-2, 1], [3/2, -1/2]]"),
        ("1/[[1,2],[3,4]]", "[[-2, 1], [3/2, -1/2]]"),
        ("2/[[1,2],[3,4]]", "[[-4, 2], [3, -1]]"),
        // Round-trip M·M^(-1) = I.
        ("[[1,2],[3,4]] * [[1,2],[3,4]]^(-1)", "[[1, 0], [0, 1]]"),
        // Symbolic / non-square: honest residual (NOT `1/[[…]]`).
        ("[[a,b],[c,d]]^(-1)", "inverse([[a, b], [c, d]])"),
        ("[[1,2,3],[4,5,6]]^(-1)", "inverse([[1, 2, 3], [4, 5, 6]])"),
        // Singular: undefined (no inverse exists).
        ("[[1,2],[2,4]]^(-1)", "undefined"),
        // Facet 2: a symbolic inverse times a matrix stays a residual, not a broadcast.
        (
            "[[a,b],[c,d]]^(-1) * [[1,0],[0,1]]",
            "inverse([[a, b], [c, d]])·[[1, 0], [0, 1]]",
        ),
        // Ordinary scalar·matrix and matrix·matrix are unaffected.
        ("3 * [[1,2],[3,4]]", "[[3, 6], [9, 12]]"),
        ("[[1,2],[3,4]] * [[5,6],[7,8]]", "[[19, 22], [43, 50]]"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
}

#[test]
fn test_eval_nested_power_text_is_parenthesized_and_round_trips() {
    // `^` is right-associative, so a surviving nested power must be parenthesized
    // in the TEXT output. `(4*x^2)^(1/2)` simplifies to `2·(x^2)^(1/2)` but was
    // rendered `2·x^2^(1/2)`, which re-parses as `2·x^(2^(1/2)) = 2·x^√2` — a
    // different, wrong expression. The fix wraps the power base in parentheses so
    // the text round-trips to the same value.
    let output = cli()
        .args(["eval", "(4*x^2)^(1/2)", "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
    let result = wire["result"].as_str().expect("result string");
    assert!(
        result.contains("(x^2)"),
        "nested power base must be parenthesized, got {result:?}"
    );

    // Round-trip: feed the rendered text back in; it must evaluate to the true
    // value `2·|x|`, not the mis-parsed `2·x^√2`.
    let reparse = cli()
        .args(["eval", &result.replace('·', "*"), "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire2: Value = serde_json::from_slice(&reparse.stdout).expect("Invalid wire output");
    assert_eq!(
        wire2["result"].as_str(),
        Some("2·|x|"),
        "rendered nested-power text must round-trip to 2·|x|, got {:?}",
        wire2["result"]
    );

    // Other clean power renderings are unchanged.
    for (input, expected) in [
        ("x^2", "x^2"),
        ("(x+1)^2", "(x + 1)^2"),
        ("x^2*y^3", "x^2·y^3"),
        ("(x^2)^(1/2)", "|x|"),
        ("x^2^3", "x^8"),
    ] {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let w: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        assert_eq!(w["result"].as_str(), Some(expected), "{input}");
    }
}

#[test]
fn test_eval_complementary_inverse_trig_respects_domain() {
    // `arcsin(x) + arccos(x) = π/2` (and the `arcsec + arccsc` form that reduces
    // to it) holds only where both terms are real, i.e. on `[-1, 1]` for
    // arcsin/arccos. For a concrete argument provably OUTSIDE that interval both
    // terms are undefined, so the identity must NOT collapse the sum to π/2.
    // Previously `arccos(2) + arcsin(2)` and `arcsec(1/2) + arccsc(1/2)` returned
    // π/2 — a wrong answer.
    for input in [
        "arccos(2) + arcsin(2)",
        "arcsin(2) + arccos(2)",
        "arccos(3) + arcsin(3)",
        "arcsec(1/2) + arccsc(1/2)",
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_ne!(
            wire["result"].as_str(),
            Some("1/2·pi"),
            "{input}: out-of-domain inverse-trig sum must not collapse to π/2"
        );
    }

    // Valid arguments still apply the identity: symbolic (with the domain
    // condition), in-interval constants, and the `|x| >= 1` arcsec/arccsc form.
    for input in [
        "arccos(x) + arcsin(x)",
        "arccos(1/2) + arcsin(1/2)",
        "arccos(1) + arcsin(1)",
        "arccos(-1) + arcsin(-1)",
        "arcsec(2) + arccsc(2)",
        "arcsec(x) + arccsc(x)",
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(
            wire["result"].as_str(),
            Some("1/2·pi"),
            "{input}: valid complementary inverse-trig sum must give π/2"
        );
    }

    // The symbolic arcsin/arccos form carries its `-1 ≤ x ≤ 1` domain condition.
    let output = cli()
        .args(["eval", "arccos(x) + arcsin(x)", "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
    let displays = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        displays.iter().any(|v| v.as_str() == Some("-1 ≤ x ≤ 1")),
        "arccos(x)+arcsin(x) must carry the -1 ≤ x ≤ 1 condition, got {displays:?}"
    );

    // The symbolic arcsec/arccsc form collapses to π/2 too, but `arcsec`/`arccsc`
    // are real only for `|arg| ≥ 1`, so the sum MUST carry the exterior-interval
    // condition (it is `x ≤ -1 or x ≥ 1` for the bare variable, and scales with an
    // affine argument). Previously the condition was dropped: the collapse to π/2
    // removed the `arccos(1/x)` witness before the per-function domain was attached.
    for (input, expected_condition) in [
        ("arcsec(x) + arccsc(x)", "x ≤ -1 or x ≥ 1"),
        ("arccsc(x) + arcsec(x)", "x ≤ -1 or x ≥ 1"),
        ("arcsec(2*x) + arccsc(2*x)", "x ≤ -1/2 or x ≥ 1/2"),
        ("arcsec(x + 1) + arccsc(x + 1)", "x ≤ -2 or x ≥ 0"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(
            wire["result"].as_str(),
            Some("1/2·pi"),
            "{input}: in-domain arcsec/arccsc sum must give π/2"
        );
        let displays = wire["required_display"]
            .as_array()
            .expect("required_display");
        assert!(
            displays
                .iter()
                .any(|v| v.as_str() == Some(expected_condition)),
            "{input} must carry the {expected_condition:?} domain condition, got {displays:?}"
        );
    }
}

#[test]
fn test_eval_single_abs_affine_equation_recovers_instead_of_leaking() {
    // A single-abs equation that reorients to `var = α·|arg| + β` (the variable
    // ends up on the abs side, e.g. an effective negative slope) used to leak a
    // malformed nested-`solve` residual. It is piecewise-linear with one
    // breakpoint, so the shared exact segment core solves it. The decompose step
    // distributes a constant factor over the sum, so divided/scaled forms
    // (`2x + |x-1| = 3`, `(|x|+|x-1|)/2 = 1`) are handled too. Cross-checked
    // against an independent exact (fractions) oracle (0 mismatches).
    for (input, expected) in [
        // Reoriented `var = c - |arg|` (the previously-leaking shape).
        ("x + abs(x-1) = 3", "{ 2 }"),
        ("abs(x-1) = 3 - x", "{ 2 }"),
        ("x + abs(x-1) = 5", "{ 3 }"),
        ("abs(x-2) = 4 - x", "{ 3 }"),
        // Nested absolute value: the outer split feeds the single-abs solver.
        ("abs(x + abs(x-1)) = 3", "{ 2 }"),
        // Coefficient ≠ 1 on the variable / on the abs: the reorientation divides
        // by the leading coefficient, which the decompose step now distributes.
        ("2*x - abs(x) = 1", "{ 1 }"),
        ("2*x + 2*abs(x-2) + 1 = 6", "{ 9/4 }"),
        // Divided sum (top-level, ≥2 abs terms under a constant denominator).
        ("(abs(x) + abs(x-1))/2 = 1", "{ -1/2, 3/2 }"),
        // Degenerate-slope branch yields a ray, not a point.
        ("x = abs(x)", "[0, infinity)"),
        ("x - abs(x-2) = 0", "{ 1 }"),
        // Working single-abs cases are unchanged (positive-slope RHS path).
        ("abs(x-1) = x + 1", "{ 0 }"),
        ("abs(2*x-1) = x", "{ 1, 1/3 }"),
        ("abs(x) = 2*x - 3", "{ 3 }"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
}

#[test]
fn test_eval_shifted_log_tan_sqrt_diff_finishes_without_depth_overflow() {
    let cases = [
        (
            "diff(ln(tan(sqrt(x))+1), x)",
            "1 / (2\u{00b7}sqrt(x)\u{00b7}cos(sqrt(x))^2\u{00b7}(tan(sqrt(x)) + 1))",
            vec!["x > 0", "cos(sqrt(x)) \u{2260} 0", "tan(sqrt(x)) + 1 > 0"],
        ),
        (
            "diff(ln(1-tan(sqrt(x))), x)",
            "-1 / (2\u{00b7}sqrt(x)\u{00b7}cos(sqrt(x))^2\u{00b7}(1 - tan(sqrt(x))))",
            vec!["x > 0", "cos(sqrt(x)) \u{2260} 0", "1 - tan(sqrt(x)) > 0"],
        ),
        (
            "diff(ln(tan(sqrt(x))-1), x)",
            "1 / (2\u{00b7}sqrt(x)\u{00b7}cos(sqrt(x))^2\u{00b7}(tan(sqrt(x)) - 1))",
            vec!["x > 0", "cos(sqrt(x)) \u{2260} 0", "tan(sqrt(x)) - 1 > 0"],
        ),
        (
            "diff(ln(2+tan(sqrt(x))), x)",
            "1 / (2\u{00b7}sqrt(x)\u{00b7}cos(sqrt(x))^2\u{00b7}(tan(sqrt(x)) + 2))",
            vec!["x > 0", "cos(sqrt(x)) \u{2260} 0", "tan(sqrt(x)) + 2 > 0"],
        ),
        (
            "diff(ln(1+tan(sqrt(2*x+3))), x)",
            "1 / (sqrt(2\u{00b7}x + 3)\u{00b7}cos(sqrt(2\u{00b7}x + 3))^2\u{00b7}(tan(sqrt(2\u{00b7}x + 3)) + 1))",
            vec![
                "x > -3/2",
                "cos(sqrt(2\u{00b7}x + 3)) \u{2260} 0",
                "tan(sqrt(2\u{00b7}x + 3)) + 1 > 0",
            ],
        ),
    ];

    for (input, expected_result, expected_required) in cases {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");

        assert!(output.status.success(), "input: {input}");
        let stderr = String::from_utf8(output.stderr).expect("stderr utf8");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");

        assert_eq!(wire["ok"], true, "input: {input}");
        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert!(
            !stderr.contains("depth_overflow"),
            "shifted tan sqrt diff should stay off the fragile simplification route for {input}, got stderr:\n{stderr}"
        );
        let mut actual_required = wire["required_display"]
            .as_array()
            .expect("required_display array")
            .iter()
            .map(|value| value.as_str().expect("required string").to_owned())
            .collect::<Vec<_>>();
        let mut expected_required = expected_required
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        actual_required.sort();
        expected_required.sort();
        assert_eq!(actual_required, expected_required, "input: {input}");
    }
}

#[test]
fn test_eval_diff_requires_explicit_variable_diagnostic() {
    let output = cli()
        .args(["eval", "diff(sin(e^(x^2)))", "--format", "json"])
        .output()
        .expect("Failed to run CLI");

    assert!(
        output.status.success(),
        "eval should surface the diagnostic as JSON, not a process failure"
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");

    assert!(stderr.is_empty(), "unexpected stderr: {stderr}");
    assert_eq!(wire["ok"], false);
    assert_eq!(wire["kind"], "InternalError");
    assert_eq!(wire["code"], "E_INTERNAL");
    assert_eq!(
        wire["error"],
        "diff requiere variable explícita: diff(expr, x)"
    );
}

#[test]
fn test_eval_text_returns_undefined_for_empty_diff_domain_without_blocked_stderr() {
    cli()
        .args(["eval", "diff(atanh(sqrt(x^2+2)), x)"])
        .assert()
        .success()
        .stdout(predicate::str::contains("undefined"))
        .stderr(predicate::str::is_empty());
}

#[test]
fn test_eval_text_groups_repeated_strict_blocked_hints_on_stderr() {
    let output = cli()
        .args(["eval", "x/x", "--domain", "strict"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();

    assert!(
        stdout.contains("x / x"),
        "expected strict result to remain residual, got stdout: {stdout}"
    );
    assert_eq!(
        stderr.matches("Blocked: requires x").count(),
        1,
        "repeated blocked hints should be grouped in stderr: {stderr}"
    );
    assert!(stderr.contains("Cancel Identical Numerator/Denominator"));
    assert!(stderr.contains("Simplify Nested Fraction"));
    assert!(stderr.contains("Cancel Common Factors"));
    assert_eq!(
        stderr
            .matches("use `domain generic` to allow definability assumptions")
            .count(),
        1,
        "repeated tips should be shown once in stderr: {stderr}"
    );
}

/// Test that budget presets can be selected.
#[test]
fn test_eval_with_budget_preset() {
    let output = cli()
        .args(["eval", "x+1", "--format", "json", "--budget", "small"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");

    assert_eq!(wire["budget"]["preset"], "small");
}

/// Test that the legacy "cli" budget alias is no longer accepted.
#[test]
fn test_eval_rejects_legacy_cli_budget_alias() {
    cli()
        .args(["eval", "x+1", "--budget", "cli"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid value 'cli'"));
}

/// Test that --strict flag sets mode to strict.
#[test]
fn test_eval_strict_mode() {
    let output = cli()
        .args(["eval", "x+1", "--format", "json", "--strict"])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");

    assert_eq!(wire["budget"]["mode"], "strict");
}

/// Test that text format output works.
#[test]
fn test_eval_text_format() {
    cli()
        .args(["eval", "2+2"])
        .assert()
        .success()
        .stdout(predicate::str::contains("4"));
}

/// Test that budget exceeded with --strict causes failure or returns unexpanded.
/// expand((a+b)^200) with small budget should either fail or return unexpanded.
#[test]
fn test_eval_budget_exceeded_strict() {
    let output = cli()
        .args([
            "eval",
            "expand((a+b)^200)",
            "--format",
            "json",
            "--budget",
            "small",
            "--strict",
        ])
        .output()
        .expect("Failed to run CLI");

    let stdout = String::from_utf8(output.stdout).unwrap();
    let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");

    // Either ok=false (error) OR result is unexpanded (didn't actually expand)
    let ok = wire["ok"].as_bool().unwrap_or(true);
    let result = wire["result"].as_str().unwrap_or("");
    let exceeded = wire["budget"]["exceeded"].is_object();

    // Budget enforcement: either fails (ok=false), reports exceeded, or returns unexpanded
    let unexpanded = result.contains("^200") || result.contains("(a + b)^200");

    assert!(
        !ok || exceeded || unexpanded,
        "Expected ok=false, budget.exceeded, or unexpanded result, got: {}",
        stdout
    );

    // Verify mode is strict in response
    assert_eq!(wire["budget"]["mode"], "strict");
}

/// Test that budget exceeded with best-effort returns partial result.
#[test]
fn test_eval_budget_exceeded_best_effort() {
    let output = cli()
        .args([
            "eval",
            "expand((a+b)^200)",
            "--format",
            "json",
            "--budget",
            "small",
            // No --strict, so best-effort mode
        ])
        .output()
        .expect("Failed to run CLI");

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).unwrap();
    let wire: Value = serde_json::from_str(&stdout).expect("Invalid wire output");

    // Should succeed (ok=true) even if budget was reached
    assert_eq!(wire["ok"], true);
    // Result should be non-empty (partial or unexpanded)
    assert!(wire["result"]
        .as_str()
        .map(|s| !s.is_empty())
        .unwrap_or(false));
}
