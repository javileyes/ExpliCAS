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
fn test_eval_abs_of_quadratic_equals_variable_splits_and_verifies() {
    // `|f(x)| = g(x)` with a degree-≥2 polynomial `f` and a variable RHS leaked an `arcsin`/`sqrt`
    // residual (the isolation path). Split into `f = ±g` and verify each root against the ORIGINAL
    // `|f(r)| = g(r)` (which enforces `g(r) ≥ 0`). Linear `|f|` (piecewise handler) and constant-RHS
    // (isolation, keeps surds) forms are untouched.
    for (input, expected) in [
        // `|x²−1| = √2` splits into `x² = 1±√2`; the `1−√2 < 0` branch is now DISPROVEN (an even root of
        // a provably-negative surd is non-real) and dropped, instead of leaking `±√(1−√2)` (imaginary).
        (
            "abs(x^2 - 1) = sqrt(2)",
            "{ (sqrt(2) + 1)^(1/2), -((sqrt(2) + 1)^(1/2)) }",
        ),
        // `|E| = 0 ⟺ E = 0`: the FULL zero-set of a factored product (the abs isolation dropped all but
        // the first factor, `|x·(x−2)| = 0 → {0}`).
        ("abs(x*(x-2)) = 0", "{ 0, 2 }"),
        ("abs((x-1)*(x-3)*(x+2)) = 0", "{ -2, 1, 3 }"),
        ("abs(sin(x)) = 0", "{ k·pi : k ∈ ℤ }"),
        // `|x²−1| = x+1`: f=g ⟹ {2,−1}; f=−g ⟹ {0,−1}; all have g ≥ 0.
        ("abs(x^2 - 1) = x + 1", "{ -1, 2, 0 }"),
        ("abs(x^2 - 4) = x + 2", "{ -2, 3, 1 }"),
        // `|f| = |h|` needs no sign condition (both branches kept).
        ("abs(x^2 - 1) = abs(x + 1)", "{ -1, 2, 0 }"),
        // Verification DROPS roots where the RHS is negative.
        ("abs(x^2 - 2) = x", "{ 2, 1 }"),
        ("abs(x^2 - 1) = -x - 5", "No solution"),
        // Controls: linear `|f|` and constant-RHS quadratic keep their existing handlers.
        ("abs(x - 3) = 2*x", "{ 1 }"),
        ("abs(x^2 - 4) = 3", "{ sqrt(7), -(sqrt(7)), 1, -1 }"),
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
fn test_eval_symbolic_quadratic_with_negative_constant_discriminant_is_empty() {
    // `x² = c` with a PROVABLY-NEGATIVE constant `c` (surd OR transcendental) has no real root, but the
    // symbolic-coefficient quadratic path emitted `±√(negative)/(2a)` as if real (a mixed surd /
    // transcendental radicand doesn't syntactically expose its sign). The discriminant now gates on
    // `provable_const_sign` — a proven-negative constant delta ⇒ No solution.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(x^2 = 1-sqrt(2), x)"), "No solution");
    assert_eq!(r("solve(abs(x^2+2) = sqrt(2), x)"), "No solution");
    assert_eq!(r("solve(x^2 = -pi, x)"), "No solution");
    assert_eq!(r("solve(x^2 = e-3, x)"), "No solution");
    // Controls: a POSITIVE constant discriminant keeps both real roots; a free-variable (symbolic)
    // quadratic is untouched (sign undecidable ⇒ kept).
    assert_eq!(
        r("solve(x^2 = sqrt(2)-1, x)"),
        "{ -((sqrt(2) - 1)^(1/2)), (sqrt(2) - 1)^(1/2) }"
    );
    assert_eq!(
        r("solve(a*x^2+b*x+c=0, x)"),
        "{ (-(b^2 - 4·a·c)^(1/2) - b) / (2·a), ((b^2 - 4·a·c)^(1/2) - b) / (2·a) }"
    );
}

#[test]
fn test_eval_solver_function_aliases_solve_via_canonical_forms() {
    // `log2`/`log10`/`cbrt` used to error `función [...] no definida` in solve():
    // they now rewrite to their canonical invertible forms (`log(2,·)`, `log(10,·)`,
    // `u^(1/3)`) at the solve entry. The reciprocal trig aliases (`csc`/`sec`/`cot`)
    // are handled at the EQUATION level (a subtree `1/sin` rewrite gets re-folded to
    // `csc` by the simplifier): `csc ⟺ sin = 1/c`, `sec ⟺ cos = 1/c`,
    // `cot(g) = c ⟺ cos − c·sin = 0` — the cos/sin form keeps `cot = 0 → π/2 + kπ`,
    // which a `1/tan` rewrite would lose.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(log2(x)=3, x)"), "{ 8 }");
    assert_eq!(r("solve(log10(x)=2, x)"), "{ 100 }");
    assert_eq!(r("solve(abs(log2(x))<3, x)"), "(1/8, 8)");
    assert_eq!(r("solve(cbrt(x)=-2, x)"), "{ -8 }");
    assert_eq!(
        r("solve(csc(x)=2, x)"),
        "{ 1/6\u{b7}pi + k\u{b7}2\u{b7}pi, 5/6\u{b7}pi + k\u{b7}2\u{b7}pi : k \u{2208} \u{2124} }"
    );
    assert_eq!(
        r("solve(sec(x)=2, x)"),
        "{ 1/3\u{b7}pi + k\u{b7}2\u{b7}pi, 5/3\u{b7}pi + k\u{b7}2\u{b7}pi : k \u{2208} \u{2124} }"
    );
    assert_eq!(
        r("solve(cot(x)=0, x)"),
        "{ 1/2\u{b7}pi + k\u{b7}pi : k \u{2208} \u{2124} }"
    );
    // Range honesty comes free from the owning solver: |1/c| > 1 has no solution.
    assert_eq!(r("solve(csc(x)=1/2, x)"), "No solution");
    assert_eq!(r("solve(csc(x)=0, x)"), "No solution");
}

#[test]
fn test_eval_const_over_trig_equation_reduces_to_full_family() {
    // SOUNDNESS: `c/trig(x) = k` (`2/sin(x)=4`) isolated to the boundary and
    // returned only the PRINCIPAL value `{π/6}`, dropping the second branch and
    // all periodicity; the coefficient-1 form (`1/sin(x)=2`) folded `1/sin → csc`
    // mid-isolation and leaked `solve(csc(x)=2)`. Reduce `c/trig(g)=k` to
    // `trig(g)=c/k` and route to the bare-trig solver for the full periodic family.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    let sin_half =
        "{ 1/6\u{b7}pi + k\u{b7}2\u{b7}pi, 5/6\u{b7}pi + k\u{b7}2\u{b7}pi : k \u{2208} \u{2124} }";
    let cos_half =
        "{ 1/3\u{b7}pi + k\u{b7}2\u{b7}pi, 5/3\u{b7}pi + k\u{b7}2\u{b7}pi : k \u{2208} \u{2124} }";
    let tan_one = "{ 1/4\u{b7}pi + k\u{b7}pi : k \u{2208} \u{2124} }";
    // Numerator ≠ 1: was principal-value-only.
    assert_eq!(r("solve(2/sin(x)=4, x)"), sin_half);
    assert_eq!(r("solve(5/cos(x)=10, x)"), cos_half);
    // Numerator = 1: was the `solve(csc(x)=...)` leak.
    assert_eq!(r("solve(1/sin(x)=2, x)"), sin_half);
    assert_eq!(r("solve(1/cos(x)=2, x)"), cos_half);
    // Tangent (reduces to `tan(g)=c/k`, not the cot homogeneous path).
    assert_eq!(r("solve(3/tan(x)=3, x)"), tan_one);
    assert_eq!(r("solve(1/tan(x)=1, x)"), tan_one);
    // The reduced target is scale-invariant: `4/sin=8` and `2/sin=4` both give sin=1/2.
    assert_eq!(r("solve(4/sin(x)=8, x)"), sin_half);
    // Negative numerator flips the sign: sin(x) = -1/2.
    assert_eq!(
        r("solve(-2/sin(x)=4, x)"),
        "{ -1/6\u{b7}pi + k\u{b7}2\u{b7}pi, 7/6\u{b7}pi + k\u{b7}2\u{b7}pi : k \u{2208} \u{2124} }"
    );
    // Shifted/scaled argument routes through the full-family solver too.
    assert_eq!(
        r("solve(2/sin(2*x)=4, x)"),
        "{ 1/12\u{b7}pi + k\u{b7}pi, 5/12\u{b7}pi + k\u{b7}pi : k \u{2208} \u{2124} }"
    );
    // Range honesty: `|c/k| > 1` for sin/cos has no solution.
    assert_eq!(r("solve(1/sin(x)=1/2, x)"), "No solution");
    assert_eq!(r("solve(2/cos(x)=1, x)"), "No solution");

    // NO REGRESSION: bare csc/sec/cot and `trig(x)/c` (constant DENOMINATOR) keep
    // their own handling.
    assert_eq!(r("solve(csc(x)=2, x)"), sin_half);
    assert_eq!(r("solve(sin(x)/2=1, x)"), "No solution");
}

#[test]
fn test_eval_second_derivative_of_sin_tan_is_numerically_equivalent() {
    // P0-G: `diff(sin(x)·tan(x), x, 2)` returned a NON-equivalent tree (wrong at
    // every sample point). Root cause: `collect_mul_factors_int_pow` returned a
    // repeated base (`2·sin·sin·cos`, a legal mid-pipeline non-canonical tree from
    // the double-angle expansion) as TWO entries, and the factor-from-Add
    // subtraction removed the common exponent from each — over-cancelling a
    // factor. The collector now aggregates duplicates. Same root cause fixed the
    // C5 family `diff((x+tan(x))^n, x)` for n = 3, 4 (dropped a cos / hung).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // (8 - (2·sin·cos)²)/(4·cos³) = (2 - sin²cos²·... ) — numerically f'' of sin·tan.
    assert_eq!(
        r("diff(sin(x)*tan(x), x, 2)"),
        "(8 - (2\u{b7}sin(x)\u{b7}cos(x))^2) / (4\u{b7}cos(x)^3)"
    );
    // The minimal over-cancel repro keeps its sin factor.
    assert_eq!(
        r("simplify((-sin(x)^3*cos(x) + 2*sin(x)^2*cos(x)) / (cos(x)^2*sin(x)))"),
        "(2\u{b7}sin(x) - sin(x)^2) / cos(x)"
    );
    // C5 siblings: n = 3 and n = 4 produce the correct 3(x+tan)²·(1+sec²) shape.
    assert_eq!(
        r("diff((x+tan(x))^3, x)"),
        "3\u{b7}(sin(x) / cos(x) + x)^2\u{b7}(2\u{b7}cos(x)^2 - 1 + 3) / (2\u{b7}cos(x)^2)"
    );
}

#[test]
fn test_eval_abs_of_log_threshold_inequality_solves_both_branches() {
    // `|ln(x)| {op} c`: the two-sided reduction was ALREADY correct, but the interval
    // algebra downstream could not ORDER the transcendental endpoints (`e²` vs `1/e²`),
    // so the intersection collapsed (`< 2` → "No solution", `≤ 2` → `[e², e²]`) and the
    // union filled the gap (`> 2` → `(0, ∞)`). `compare_values` now decides constant
    // transcendental endpoints by the exact value-bounds oracle.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(abs(ln(x)) < 2, x)"), "(1 / e^2, e^2)");
    assert_eq!(r("solve(abs(ln(x)) <= 2, x)"), "[1 / e^2, e^2]");
    assert_eq!(
        r("solve(abs(ln(x)) > 2, x)"),
        "(0, 1 / e^2) U (e^2, infinity)"
    );
    assert_eq!(
        r("solve(abs(ln(x)) >= 2, x)"),
        "(0, 1 / e^2] U [e^2, infinity)"
    );
    // The equation sibling stays as it was (already correct).
    assert_eq!(r("solve(abs(ln(x)) = 2, x)"), "{ e^2, 1 / e^2 }");
    // An exponential inside the abs: one side is vacuous (e^x − 1 > −2 always).
    assert_eq!(r("solve(abs(e^x - 1) < 2, x)"), "(-infinity, ln(3))");
    // Polynomial controls keep their surd-endpoint rendering.
    assert_eq!(r("solve(abs(x) < 2, x)"), "(-2, 2)");
}

#[test]
fn test_eval_irrational_fractional_base_exponential_inequality_flips() {
    // `base^x {op} c` where the CONSTANT base is provably in (0, 1) but IRRATIONAL
    // (`sin(1)`, `cos(1)`): the `log(base, ·)` isolation must flip the direction
    // (decreasing exponential), decided by the exact value-bounds oracle. It used to
    // flip only for exact rationals, returning the reversed ray. Bases > 1 and the
    // equation form stay unflipped.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // (The `if sin(1) > 0` guard is PRUNED — the sign is provable — and the isolation
    // renders the equivalent `ln` ratio; the direction is what this contract fixes.)
    assert_eq!(
        r("solve(sin(1)^x > 2, x)"),
        "(-infinity, ln(2) / ln(sin(1)))"
    );
    assert_eq!(
        r("solve(cos(1)^x >= 3, x)"),
        "(-infinity, ln(3) / ln(cos(1))]"
    );
    assert_eq!(
        r("solve(sin(1)^x < 2, x)"),
        "(ln(2) / ln(sin(1)), infinity)"
    );
    // Controls: base > 1 keeps direction; equations never flip; symbolic base untouched.
    assert_eq!(r("solve(pi^x > 5, x)"), "(ln(5) / ln(pi), infinity)");
    assert_eq!(r("solve(sin(1)^x = 2, x)"), "{ ln(2) / ln(sin(1)) }");
    assert_eq!(r("solve(a^x > 2, x)"), "(log(a, 2), infinity) if a > 0");
}

#[test]
fn test_eval_const_over_surd_affine_denominator_keeps_true_pole_only() {
    // `c/(a·x + b) {op} 0` with a NON-RATIONAL constant intercept `b`: the simplifier
    // rationalizes the denominator through its conjugate (`1/(x+√2) → (√2−x)/(2−x²)`),
    // fabricating a spurious REMOVABLE pole at the conjugate that the rational-inequality
    // path punched out of the answer (`(−√2,√2)∪(√2,∞)`), collapsed odd-root denominators
    // to a false "No solution", and returned the conjugate as a root of `c/g = 0`. The
    // raw-tree reduction `c/g {op} 0 ⟺ g {op'} 0` keeps only the TRUE pole.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(1/(x+sqrt(2))>0, x)"), "(-(sqrt(2)), infinity)");
    assert_eq!(r("solve(-1/(x-sqrt(2))>0, x)"), "(-infinity, sqrt(2))");
    assert_eq!(r("solve(2/(x+sqrt(3))>=0, x)"), "(-(sqrt(3)), infinity)");
    assert_eq!(r("solve(1/(2*x+sqrt(2))>0, x)"), "(-(2^(-1/2)), infinity)");
    assert_eq!(r("solve(1/(x+2^(1/3))>0, x)"), "(-(2^(1/3)), infinity)");
    assert_eq!(r("solve(1/(1+sqrt(2)-x)>0, x)"), "(-infinity, sqrt(2) + 1)");
    // Non-strict + negative constant: the `≥` split's equation branch must NOT
    // resurrect the conjugate as a boundary singleton.
    assert_eq!(
        r("solve(-2/3/(2*x+sqrt(2))>=0, x)"),
        "(-infinity, -(2^(-1/2)))"
    );
    // A nonzero constant over ANYTHING is never zero (raw check, before the
    // rationalizer plants a conjugate numerator root).
    assert_eq!(r("solve(-2/3/(2*x+sqrt(2))=0, x)"), "No solution");
    // Nonzero thresholds solve in `u = g(x)` space (all-rational breakpoints) and map
    // back through the affine — these previously returned a false "No solution" or a
    // malformed residual.
    assert_eq!(
        r("solve(1/(x+sqrt(2))>1, x)"),
        "(-2\u{b7}2^(-1/2), 1 - sqrt(2))"
    );
    assert_eq!(
        r("solve(1/(x+sqrt(2))<1, x)"),
        "(-infinity, -2\u{b7}2^(-1/2)) U (1 - sqrt(2), infinity)"
    );
    assert_eq!(r("solve(1/(x+sqrt(2))=1, x)"), "{ 1 - sqrt(2) }");
    assert_eq!(r("solve(2/(x-sqrt(3))<=-1, x)"), "[sqrt(3) - 2, sqrt(3))");
    // Orientation flips for a negative slope.
    assert_eq!(r("solve(1/(-x+sqrt(2))>2, x)"), "(sqrt(2) - 1/2, sqrt(2))");
    // Controls: rational pole, symbolic intercept, bare 1/x, and the equation
    // forms with a variable numerator keep their owners.
    assert_eq!(r("solve(1/(x-2)>0, x)"), "(2, infinity)");
    assert_eq!(r("solve(1/(x+a)>0, x)"), "(-a, infinity)");
    assert_eq!(r("solve(1/x>0, x)"), "(0, infinity)");
    assert_eq!(r("solve(x/(x-2)=0, x)"), "{ 0 }");
    assert_eq!(r("solve(1/(x-2)=3, x)"), "{ 7/3 }");
    assert_eq!(r("solve(1/(x-2)>1, x)"), "(2, 3)");
}

#[test]
fn test_eval_rational_exponent_constants_are_sign_decidable() {
    // A constant `base^(p/q)` (`e^(1/3)`, `2^(1/3)`, ...) is now sign-decidable via
    // exact n-th-root value bounds (`const_sign::interval_pow`), closing the P0-F-log
    // family (an out-of-domain negative root `e^(1/3)/(1-e^(1/3))` was kept) and its
    // guard siblings (even-root threshold, abs-split, quadratic discriminant) that
    // previously only decided rationals/linear surds.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Log-equation domain filter: the candidate root is provably negative (x > 0
    // required) and the conditional's constant conditions are provably false, so the
    // whole thing prunes to a clean "No solution".
    assert_eq!(r("solve(ln(x)-ln(x+1)=1/3, x)"), "No solution");
    // Control: a positive in-domain root is KEPT (e^(-1/3)/(1-e^(-1/3)) ~ 2.53 > 0),
    // unconditionally (the coefficient 1 - e^(-1/3) is provably nonzero).
    assert_eq!(
        r("solve(ln(x)-ln(x+1)=-1/3, x)"),
        "{ e^(-1/3) / (1 - e^(-1/3)) }"
    );
    // Even-root RANGE correction with a transcendental-power threshold (`√ >= 0`).
    assert_eq!(r("solve(sqrt(x) < -e^(1/3), x)"), "No solution");
    assert_eq!(r("solve(sqrt(x) >= -2^(1/3), x)"), "[0, infinity)");
    // abs-split: the `x² = 1 - e^(1/3)` branch radicand is provably negative — the
    // spurious complex pair is dropped, keeping only the real pair.
    assert_eq!(
        r("solve(abs(x^2-1) = e^(1/3), x)"),
        "{ (e^(1/3) + 1)^(1/2), -((e^(1/3) + 1)^(1/2)) }"
    );
    // Quadratic with a provably-negative transcendental-power constant.
    assert_eq!(r("solve(x^2 = 1-e^(1/3), x)"), "No solution");
    // Control: positive threshold still squares (the sound branch).
    assert_eq!(r("solve(sqrt(x) > e^(1/3), x)"), "(e^(2/3), infinity)");
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
        // DIFFERENCE of two radicals `√f − √g = c`: the reduction flips the RHS sign and the
        // verification checks `√f − √g == c`, so the sign carries through.
        ("sqrt(x+5) - sqrt(x) = 1", "{ 4 }"),
        ("sqrt(3*x+1) - sqrt(x+4) = 1", "{ 5 }"),
        ("sqrt(x) - sqrt(x-3) = 1", "{ 4 }"),
        // A difference exceeding its bound, and a negatively-signed one, are dropped by verification.
        ("sqrt(x+5) - sqrt(x) = 10", "No solution"),
        ("sqrt(x) - sqrt(x+5) = 1", "No solution"),
        // MONOMIAL reduced RHS (`√(fg) = c·x`, no constant term): these returned a
        // wrong "No solution" (or dropped a root) because the single-radical solver
        // mishandles `√(quad) = c·x`. The reduction now squares to the POLYNOMIAL
        // `fg − reduced_rhs² = 0` and verifies, bypassing that solver. Cross-checked
        // vs sympy solveset.
        ("sqrt(5*x-1) - sqrt(x+2) = 1", "{ 2 }"),
        ("sqrt(4*x+1) - sqrt(x) = 1", "{ 0, 4/9 }"),
        ("sqrt(3*x+1) - sqrt(x) = 1", "{ 0, 1 }"),
        ("sqrt(2*x+7) - sqrt(x+3) = 1", "{ -3, 1 }"),
        ("sqrt(3*x+4) - sqrt(x) = 2", "{ 0, 4 }"),
        // EQUAL radicands with `c = 0` (the both-sides equality `√A = √B`): the candidate makes
        // both radicands equal but IRRATIONAL (√7 at x=2), so the verification must accept the
        // canceling surds rather than demanding each radicand be a perfect square.
        ("sqrt(2*x+3) = sqrt(x+5)", "{ 2 }"),
        ("sqrt(x+1) = sqrt(2*x-3)", "{ 4 }"),
        ("sqrt(2*x+8) - sqrt(x+5) = 0", "{ -3 }"),
        // Equal-slope radicands never meet: genuine no-solution stays no-solution.
        ("sqrt(x+3) - sqrt(x+5) = 0", "No solution"),
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
fn test_eval_single_radical_equals_polynomial_squares_and_verifies() {
    // `√(quadratic) = polynomial` (`√(5x²+9x−2) = 3x`): the isolation core
    // mis-filtered after squaring — a wrong "No solution" (true `{1/4, 2}`) or a
    // dropped root (`√(5x²+9x) = 3x → {0}`, missing `9/4`). Square exactly to
    // `f − g² = 0`, solve, and keep roots with `g(r) ≥ 0`. Cross-checked vs sympy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // The confirmed wrong-answers, now fixed.
    assert_eq!(r("solve(sqrt(5*x^2+9*x-2) = 3*x, x)"), "{ 1/4, 2 }");
    assert_eq!(r("solve(sqrt(5*x^2+9*x) = 3*x, x)"), "{ 0, 9/4 }");
    assert_eq!(r("solve(sqrt(x^2-4) = x-1, x)"), "{ 5/2 }");
    // `g(r) < 0` extraneous roots dropped: `√(6x²+x−1) = 2x` has candidates {1/2, −1};
    // only 1/2 has `2x ≥ 0`.
    assert_eq!(r("solve(sqrt(6*x^2+x-1) = 2*x, x)"), "{ 1/2 }");
    // Squared quadratic with complex reduced roots, and a constant `f − g²`, stay
    // "No solution".
    assert_eq!(r("solve(sqrt(9*x^2+1) = 3*x, x)"), "No solution");
    assert_eq!(r("solve(sqrt(x^2+5*x+6) = x+1, x)"), "No solution");

    // NO REGRESSION: previously-correct degree-2 radicands (rational and surd
    // roots), degree-1 radicands (isolation path), and the perfect-square identity
    // are unchanged.
    assert_eq!(r("solve(sqrt(2*x^2+x) = 2*x, x)"), "{ 0, 1/2 }");
    assert_eq!(r("solve(sqrt(x^2+7*x) = x+3, x)"), "{ 9 }");
    assert_eq!(
        r("solve(sqrt(3*x^2+5*x-2) = 2*x, x)"),
        "{ 1/2·(5 - sqrt(17)), 1/2·(sqrt(17) + 5) }"
    );
    assert_eq!(r("solve(sqrt(x+1) = 2, x)"), "{ 3 }");
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
        // A negative SURD threshold: decided exactly (`√x < −√2` impossible; `√x > −√2` holds on the
        // whole domain) — it used to fall through to the unsound squaring branch (`√x < −√2 → [0,2)`).
        ("sqrt(x) < -sqrt(2)", "No solution"),
        ("sqrt(x) > -sqrt(2)", "[0, infinity)"),
        ("sqrt(x-1) >= -sqrt(3)", "[1, infinity)"),
        // sqrt(g) <= 0 forces g = 0: a single point in the domain (a degenerate interval `[p, p]`).
        ("sqrt(x+3) <= 0", "[-3, -3]"),
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
        // intersection drops as a degenerate overlap; recovered via `solve(√f = g)` (rendered `[p, p]`).
        ("sqrt(x+3) <= -x-3", "[-3, -3]"),
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
        // Boundary touch with fractional g: `√(9-x²) = (1/3)x-1` at x=3 (`√0=0`). The
        // single-radical equation solver now resolves this boundary (it previously
        // leaked, so the non-strict root re-union was skipped), so x=3 re-unions as the
        // degenerate `[3, 3]` — the engine's standard form for a point-only non-strict
        // solution (`x² <= 0 → [0, 0]`), not the bug-dependent `{ 3 }`.
        ("sqrt(-x^2+9) <= (1/3)*x-1", "[3, 3]"),
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
fn test_eval_radical_equation_drops_extraneous_root_via_rhs_sign() {
    // A single-radical equation `√f = g` carries the range constraint `g ≥ 0` (√ is nonnegative);
    // squaring loses it, so the solver returned BOTH quadratic roots. `√(x+1) = -x` yielded
    // `{φ, ½(1-√5)}`, but `φ > 0` makes the RHS `-x < 0` — extraneous. Recording `NonNegative(g)` lets
    // the EXACT surd-sign prover drop it. The golden-ratio root is the named constant `phi`, whose sign
    // the surd parser cannot read; the `const_value_bounds` fallback (arbitrary-precision interval
    // arithmetic) decides `-phi < 0` exactly. A valid root has `g = √f ≥ 0`, so this never overdrops.
    for (input, expected) in [
        ("sqrt(x+1) = -x", "{ 1/2·(1 - sqrt(5)) }"),
        ("sqrt(x+1) = -1*x", "{ 1/2·(1 - sqrt(5)) }"),
        // φ is VALID here (`√(φ+1) = φ`), so it must be KEPT — the condition `x ≥ 0` holds at φ.
        ("sqrt(x+1) = x", "{ phi }"),
        // RATIONAL squared-roots already filtered, but the condition is consistent.
        ("sqrt(x+6) = -x", "{ -2 }"),
        ("sqrt(x) = x - 2", "{ 4 }"),
        ("sqrt(x-1) = x - 3", "{ 5 }"),
        ("sqrt(x+1) = x - 1", "{ 3 }"),
        // Surd squared-roots with a non-unit RHS slope stay correct.
        ("sqrt(x+1) = -2*x", "{ 1/8·(1 - sqrt(17)) }"),
        // No real root survives the RHS-sign constraint.
        ("sqrt(x-1) = -x", "No solution"),
        // Pure isolation (constant RHS) unaffected.
        ("sqrt(x) = 2", "{ 4 }"),
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
fn test_eval_abs_threshold_and_ln_square_inequalities() {
    // `|g(x)| {op} c` (constant `c`) and `ln(x)^2 {op} c` are NON-MONOTONIC: the isolation/
    // split path dropped the operator and returned the boundary equation (`|x^2-2x| < 1` ->
    // "No solution"; `ln(x)^2 > 1` -> "All real numbers if x>0"). Both now reduce to the
    // two sub-inequalities the engine already solves exactly, intersected (`<`) or unioned
    // (`>`). The non-strict cases additionally exercise the closed-endpoint root filter, which
    // keeps the `e^{±√t}` band intact (`compare_values` cannot order `E`-bearing bounds).
    for (input, expected) in [
        // abs of a quadratic WITH a linear term (the symmetric `|x^2-k|` already worked).
        ("abs(x^2-2x) < 1", "(1 - sqrt(2), 1) U (1, sqrt(2) + 1)"),
        (
            "abs(x^2-2x) > 1",
            "(-infinity, 1 - sqrt(2)) U (sqrt(2) + 1, infinity)",
        ),
        ("abs(x^2-2x) <= 1", "[1 - sqrt(2), sqrt(2) + 1]"),
        ("abs(x^2-5x+6) < 2", "(1, 4)"),
        ("abs(x^2-5x+6) <= 2", "[1, 4]"),
        // c <= 0 edges.
        ("abs(x^2-2x) > 0", "(-infinity, 0) U (0, 2) U (2, infinity)"),
        // ln(x)^2 {op} c: strict and non-strict, integer and surd thresholds.
        ("ln(x)^2 > 1", "(0, 1 / e) U (e, infinity)"),
        ("ln(x)^2 < 1", "(1 / e, e)"),
        ("ln(x)^2 >= 1", "(0, 1 / e] U [e, infinity)"),
        ("ln(x)^2 <= 1", "[1 / e, e]"),
        ("ln(x)^2 <= 4", "[1 / e^2, e^2]"),
        ("ln(x)^2 < 4", "(1 / e^2, e^2)"),
        // ln(x)^2 c-edge cases: domain-aware, never a fabricated "All reals".
        ("ln(x)^2 > 0", "(0, 1) U (1, infinity)"),
        ("ln(x)^2 <= 0", "[1, 1]"),
        ("ln(x)^2 < -1", "No solution"),
        // Regression: genuinely-dropped isolated roots of non-strict inequalities survive.
        ("(x-2)^2*(x+1) <= 0", "(-infinity, -1] U [2, 2]"),
        ("x+1/x <= 2", "(-infinity, 0) U [1, 1]"),
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
        // Additive: `∞ + finite = ∞`, so `(∞+1)/(∞+1)` is `∞/∞`, NOT `1`. `∞ − ∞` stays indeterminate.
        ("(inf+1)/(inf+1)", "undefined"),
        ("(inf+inf)/(inf+inf)", "undefined"),
        ("(2*inf+2*inf)/(inf+inf)", "undefined"),
        ("(inf+x)/(inf+x)", "undefined"),
        ("((-inf)+5)/((-inf)+5)", "undefined"),
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
        // Additive ∞ in the quotient (was `1`/`2` plain vs `undefined` steps).
        "(inf+inf)/(inf+inf)",
        "(2*inf+2*inf)/(inf+inf)",
        "(inf+1)/(inf+1)",
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
fn test_eval_irreducible_cubic_single_real_root_by_cardano() {
    // An irreducible cubic (no rational root) with a SINGLE real root (Cardano Δ > 0) is solved
    // exactly by radicals instead of leaking a residual. The root is `∛(-q/2+√Δ) + ∛(-q/2-√Δ) - B/3`
    // (real cube roots). These are numerically verified to satisfy the cubic in the dev probes
    // (e.g. `x³+x²+3` → −1.8637, `x³-x-1` → the plastic number 1.3247).
    for input in [
        "solve(x^3+x^2+3=0, x)",
        "solve(x^3-2*x^2-4*x-2=0, x)",
        "solve(x^3+x-1=0, x)",
        "solve(x^3-x-1=0, x)",
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        let result = wire["result"].as_str().unwrap_or("");
        // A single discrete real root expressed by radicals: `{ … }`, not a residual `Solve:`/`if`.
        assert!(
            result.starts_with("{ ")
                && result.contains("^(1/3)")
                && !result.contains("Solve")
                && !result.contains(" if "),
            "{input} -> {result}"
        );
    }
    // FACTOR case: a higher-degree polynomial `(rational factors)·(irreducible Δ>0 cubic)` peels its
    // rational roots, then solves the leftover cubic by Cardano and unions — previously the cubic
    // factor's real root was silently dropped (`x⁴+x³+3x → {0}` lost the root of `x³+x²+3`). The
    // rational roots are reported as a DISTINCT set (the `x²` factor's double `0` collapses to one).
    for input in [
        "solve(x^4+x^3+3*x=0, x)",         // x·(x³+x²+3)
        "solve(x^4-2*x^3-4*x^2-2*x=0, x)", // x·(x³-2x²-4x-2)
        "solve(x^5+x^4+3*x^2=0, x)",       // x²·(x³+x²+3), double 0 deduped
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        let result = wire["result"].as_str().unwrap_or("");
        // `{ 0, <cubic radical root> }`: rational root 0 plus the cubic's single real root, no residual,
        // no duplicate `0`.
        assert!(
            result.starts_with("{ 0, ")
                && result.contains("^(1/3)")
                && !result.contains("0, 0")
                && !result.contains("Solve")
                && !result.contains(" if "),
            "{input} -> {result}"
        );
    }
    // Rational-root and clean cubics are unaffected (NOT routed to Cardano).
    for (input, expected) in [
        ("solve(x^3-1=0, x)", "{ 1 }"),
        ("solve(x^3-6*x^2+11*x-6=0, x)", "{ 1, 2, 3 }"),
        ("solve(x^3-2=0, x)", "{ 2^(1/3) }"),
        ("solve(x^3+3*x^2+3*x+1=0, x)", "{ -1 }"),
        ("solve(x^3-3*x+2=0, x)", "{ -2, 1 }"),
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
fn test_eval_casus_irreducibilis_cubic_three_real_roots() {
    // The casus irreducibilis: an irreducible cubic with Δ < 0 has THREE distinct real roots that
    // cannot be written with real radicals, so they are emitted in trigonometric form
    // `2√(-p/3)·cos(φ/3 - 2πk/3) - B/3` (the engine collapses special arccos values to sin/cos
    // ratios). Each root is numerically verified to satisfy its cubic in the dev probes
    // (e.g. `x³-3x+1` → {1.532, 0.347, -1.879}, `x³-7x+7` → {1.692, 1.357, -3.049}).
    let three_root_cases = [
        "solve(x^3-3*x+1=0, x)",
        "solve(x^3-7*x+7=0, x)",
        "solve(x^3-3*x^2+1=0, x)",
    ];
    for input in three_root_cases {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        let result = wire["result"].as_str().unwrap_or("");
        // Three real roots in a trig closed form: `{ a, b, c }`, no residual.
        assert!(
            result.starts_with("{ ")
                && (result.contains("cos(") || result.contains("sin("))
                && result.matches(", ").count() == 2
                && !result.contains("Solve")
                && !result.contains(" if "),
            "{input} -> {result}"
        );
    }
    // As a FACTOR of a higher-degree polynomial, the casus-irreducibilis cubic is now also solved:
    // `x⁴-3x²+x = x·(x³-3x+1)` yields the rational root 0 plus the three trig roots (4 total).
    let factor = cli()
        .args(["eval", "solve(x^4-3*x^2+x=0, x)", "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    assert!(factor.status.success());
    let fwire: Value = serde_json::from_slice(&factor.stdout).expect("Invalid wire output");
    let fresult = fwire["result"].as_str().unwrap_or("");
    assert!(
        fresult.starts_with("{ 0, ")
            && (fresult.contains("cos(") || fresult.contains("sin("))
            && fresult.matches(", ").count() == 3,
        "factor casus -> {fresult}"
    );
}

#[test]
fn test_eval_fraction_base_power_is_parenthesized() {
    // A non-integer rational base under a power must keep its parentheses: `(3/2)^(1/3)`, NOT
    // `3/2^(1/3)` — the latter re-parses (under standard precedence, `^` binds tighter than `/`) as
    // `3/(2^(1/3))`, a DIFFERENT, wrong value. This is most visible in Cardano radicals like
    // `solve(10x³-4x²+18x-27=0)` whose real root is `1/15·((17161/2)^(1/3) + 2 - 262^(1/3))`.
    for input in ["(3/2)^(1/3)", "(17161/2)^(1/3)", "(7/3)^(1/5)", "(2/3)^x"] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        let result = wire["result"].as_str().unwrap_or("");
        // The result must be IDEMPOTENT: re-evaluating the printed form yields the same string. A
        // dropped-paren form would re-parse differently, so this catches the precedence bug directly.
        let reparse = cli()
            .args(["eval", result, "--format", "json"])
            .output()
            .expect("Failed to re-run CLI");
        let rwire: Value = serde_json::from_slice(&reparse.stdout).expect("Invalid wire output");
        assert_eq!(
            rwire["result"].as_str(),
            Some(result),
            "{input} -> {result} did not round-trip"
        );
        // And it must literally carry the parenthesized base (not the bare `n/m^...`).
        assert!(
            result.contains(")^") && !result.contains("/2^(") && !result.contains("/3^("),
            "{input} -> {result}"
        );
    }
    // Integer bases are NOT over-parenthesized.
    for (input, expected) in [("2^(1/3)", "2^(1/3)"), ("262^(1/3)", "262^(1/3)")] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
    // The Cardano radical that exposed the bug now renders unambiguously.
    let cardano = cli()
        .args([
            "eval",
            "solve(10*x^3-4*x^2+18*x-27=0, x)",
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to run CLI");
    let cwire: Value = serde_json::from_slice(&cardano.stdout).expect("Invalid wire output");
    let cresult = cwire["result"].as_str().unwrap_or("");
    assert!(
        cresult.contains("(17161/2)^(1/3)"),
        "cardano fraction radicand -> {cresult}"
    );
}

#[test]
fn test_eval_matrix_shape_mismatch_is_undefined() {
    // A shape-incompatible matrix operation has no value, so it must return the `undefined` sentinel
    // (like `1/0` or a singular inverse), never echo the malformed operation back as a valid result.
    // Previously these returned the operation unchanged with `ok:true`.
    for input in [
        "[[1,2],[3,4]] + [[1,2,3],[4,5,6]]", // add, different dims
        "[[1,2],[3,4]] - [[1,2,3]]",         // sub, different dims
        "[[1,2],[3,4]] * [[1,2,3]]",         // mul, inner dims 2 != 1
        "[[1,2,3],[4,5,6]]^2",               // non-square power
        "[[1,2,3]] + 5",                     // matrix + scalar (no broadcast)
        "([[1,2],[3,4]] + [[1,2,3],[4,5,6]]) * [[1,2],[3,4]]", // distributed mismatch propagates
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some("undefined"), "{input}");
    }
    // Compatible matrix operations are unaffected (must still compute exactly).
    for (input, expected) in [
        ("[[1,2],[3,4]] + [[5,6],[7,8]]", "[[6, 8], [10, 12]]"),
        (
            "[[1,2],[3,4]] * [[1,2,3],[4,5,6]]",
            "[[9, 12, 15], [19, 26, 33]]",
        ),
        ("3 * [[1,2],[3,4]]", "[[3, 6], [9, 12]]"),
        ("det([[1,2],[3,4]])", "-2"),
        ("[[1,2,3]] + [[4,5,6]]", "[5, 7, 9]"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
    // A matrix plus a SYMBOLIC operand is left untouched (the symbol may later bind to a matrix),
    // never prematurely declared `undefined`.
    let out = cli()
        .args(["eval", "[[1,2,3]] + y", "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
    assert_ne!(
        wire["result"].as_str(),
        Some("undefined"),
        "matrix + symbolic must not be undefined"
    );
}

#[test]
fn test_eval_apart_partial_fractions() {
    // `apart(p/q)` (alias `partfrac`) gives the partial-fraction decomposition, exact over Q. The
    // result is `Hold`-protected so the fraction-combining rules don't pull it back over a common
    // denominator. Single-variable is inferred; `apart(p/q, x)` names it. An IMPROPER fraction
    // (deg p ≥ deg q) is polynomial-divided first: `p/q = quotient + remainder/q`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("apart(1/(x^2-1))"), "1/2 / (x - 1) - 1/2 / (x + 1)");
    // Improper fractions: the polynomial quotient is prepended to the proper decomposition.
    assert_eq!(r("apart(x^3/(x^2-1))"), "1/2 / (x - 1) + 1/2 / (x + 1) + x");
    assert_eq!(r("apart(x^2/(x^2-1))"), "1/2 / (x - 1) + 1 - 1/2 / (x + 1)");
    assert_eq!(
        r("apart(x^4/(x^2-1))"),
        "1/2 / (x - 1) + x^2 + 1 - 1/2 / (x + 1)"
    );
    assert_eq!(
        r("apart(1/(x^3-x))"),
        "1/2 / (x - 1) + 1/2 / (x + 1) - 1 / x"
    );
    assert_eq!(
        r("apart(1/((x-1)*(x-2)*(x-3)))"),
        "1/2 / (x - 3) + 1/2 / (x - 1) - 1 / (x - 2)"
    );
    assert_eq!(r("apart((x+3)/(x^2-x-2))"), "5/3 / (x - 2) - 2/3 / (x + 1)");
    assert_eq!(r("apart(1/(x^2+x))"), "1 / x - 1 / (x + 1)");
    // Not a rational fraction, or an irreducible high-degree denominator ⇒ honest residual.
    assert_eq!(r("apart(x^2+1)"), "apart(x^2 + 1)");
    assert_eq!(r("apart(1/(x^3-x-1))"), "apart(1 / (x^3 - x - 1))");
    // Repeated roots get the full multiplicity ladder A_k/(x-r)^k, NOT the
    // Ostrogradsky/Hermite integral form (which dropped the 1/(2(x-1)^2) term and
    // returned a non-equivalent answer). Soundness regression guard for B2.
    assert_eq!(
        r("apart(1/((x-1)^2*(x+1)))"),
        "1/4 / (x + 1) + 1/2 / (x - 1)^2 - 1/4 / (x - 1)"
    );
    assert_eq!(r("apart(1/((x-1)^2))"), "1 / (x - 1)^2");
    assert_eq!(r("apart((x+1)/((x-1)^2))"), "1 / (x - 1) + 2 / (x - 1)^2");
    assert_eq!(r("apart(1/((x-1)^3))"), "1 / (x - 1)^3");
    assert_eq!(
        r("apart(1/(x*(x-1)^2))"),
        "1 / x + 1 / (x - 1)^2 - 1 / (x - 1)"
    );
    // A degree-1 denominator IS already a partial fraction. `apart` used to
    // decline it (min denominator degree 2) and echo an unevaluated
    // `apart(1/(x-2))` residual; it now returns the fraction. An IMPROPER
    // degree-1 fraction gets the polynomial part split off.
    assert_eq!(r("apart(1/(x-2))"), "1 / (x - 2)");
    assert_eq!(r("apart(5/(x-3))"), "5 / (x - 3)");
    assert_eq!(r("apart(1/(2*x-4))"), "1/2 / (x - 2)");
    assert_eq!(r("apart((x+1)/(x-2))"), "3 / (x - 2) + 1");
    assert_eq!(r("apart((3*x+1)/(x+2))"), "3 - 5 / (x + 2)");
    // A shared numerator/denominator factor cancels to a degree-1 pole, which
    // now decomposes (returns to itself) instead of echoing the residual.
    assert_eq!(r("apart((x+1)/(x^2-x-2))"), "1 / (x - 2)");
    // A SCALED monomial numerator `c*x^k` (c != 1) simplifies to `Mul(c, Div(x^k,
    // D))` — the constant pulls OUT of the division — so the old `Expr::Div`-only
    // match echoed an unevaluated `apart(2x/…)` residual while the unit `x/…`
    // decomposed fine. Any fraction-like shape (nested Div / reciprocal factor)
    // now normalizes to `num/den` first. Cross-checked vs sympy.
    assert_eq!(
        r("apart((2*x)/((x-1)^2*(x+1)))"),
        "1/2 / (x - 1) + 1 / (x - 1)^2 - 1/2 / (x + 1)"
    );
    assert_eq!(
        r("apart((3*x)/((x-1)*(x+1)))"),
        "3/2 / (x - 1) + 3/2 / (x + 1)"
    );
    assert_eq!(
        r("apart((2*x^2)/((x-1)^2*(x+1)))"),
        "1/2 / (x + 1) + 1 / (x - 1)^2 + 3/2 / (x - 1)"
    );
    assert_eq!(r("apart((5*x)/((x-2)*(x+3)))"), "2 / (x - 2) + 3 / (x + 3)");
    // Scaled improper fraction: the polynomial quotient is split off.
    assert_eq!(
        r("apart((2*x^3)/((x-1)*(x+2)))"),
        "2/3 / (x - 1) + 16/3 / (x + 2) + 2·x - 2"
    );
}

#[test]
fn test_eval_binary_matrix_ops_dot_cross_linsolve() {
    // The 2-argument matrix/vector operations: dot product, cross product, and linear-system
    // solving. dot/cross fold numerically and stay exact symbolically; linsolve returns the UNIQUE
    // solution by exact rational RREF of [A|b], declining (residual) on a singular or inconsistent
    // system. Cross-checked against numpy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // dot
    assert_eq!(r("dot([1,2,3],[4,5,6])"), "32");
    assert_eq!(r("dot([a,b],[c,d])"), "a·c + b·d");
    assert_eq!(r("dot([1,2],[3,4,5])"), "dot([[1], [2]], [[3], [4], [5]])"); // length mismatch
                                                                             // cross
    assert_eq!(r("cross([1,0,0],[0,1,0])"), "[[0], [0], [1]]");
    assert_eq!(r("cross([2,3,4],[5,6,7])"), "[[-3], [6], [-3]]");
    // linsolve
    assert_eq!(r("linsolve([[1,1],[1,-1]], [3,1])"), "[[2], [1]]");
    assert_eq!(
        r("linsolve([[1,2,3],[0,1,4],[5,6,0]], [6,5,11])"),
        "[[1], [1], [1]]"
    );
    // Singular and inconsistent systems decline to honest residuals.
    assert_eq!(
        r("linsolve([[1,2],[2,4]], [3,6])"),
        "linsolve([[1, 2], [2, 4]], [[3], [6]])"
    );
    assert_eq!(
        r("linsolve([[1,1],[1,1]], [1,2])"),
        "linsolve([[1, 1], [1, 1]], [[1], [2]])"
    );
}

#[test]
fn test_eval_vector_norm() {
    // `norm(v)` is the Euclidean / Frobenius norm √(Σ |entryᵢ|²) and it is VALUE-DEPENDENT
    // (`|entry|² = entry²` only for real entries), so it is domain-threaded (Fase 2 V0):
    // - real mode (default): `i` is an ordinary symbol — the same contract as the gated
    //   Gaussian rules (`abs(3+4i)` stays residual) — so every entry squares RAW and an
    //   `i`-carrying vector stays an honest unevaluated radical.
    // - complex mode: a Gaussian entry folds its MAGNITUDE (`|a+bi|² = a²+b²`) exactly and
    //   a symbolic entry squares its modulus (`|x|²`) — `x²` would be wrong for ℂ-valued x
    //   (x:=i would make `sqrt(x²+1)` collapse to 0 while the norm is sqrt(2)).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    let rc = |input: &str| -> String {
        let out = cli()
            .args([
                "eval",
                input,
                "--value-domain",
                "complex",
                "--format",
                "json",
            ])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Real entries: identical in both domains.
    assert_eq!(r("norm([3,4])"), "5");
    assert_eq!(r("norm([1,2,2])"), "3");
    assert_eq!(r("norm([1,1])"), "sqrt(2)");
    assert_eq!(r("norm([3,-4])"), "5");
    assert_eq!(r("norm([[3,4],[0,12]])"), "13"); // Frobenius norm of a matrix
    assert_eq!(r("norm([a,b])"), "(a^2 + b^2)^(1/2)"); // symbolic, real-valued symbols
    assert_eq!(rc("norm([3,4])"), "5");
    assert_eq!(rc("norm([[3,4],[0,12]])"), "13");
    // Real mode + `i`: honest symbolic radical (i is a plain symbol here; the Imaginary
    // Usage Warning nudges to complex mode). Folding these to 5 / sqrt(2) was the V0
    // incoherence: the metric layer treated `i` as imaginary while every gated Gaussian
    // rule (and `abs(3+4i)`) kept it symbolic.
    assert_eq!(r("norm([3,4i])"), "(9 + 16·i^2)^(1/2)");
    assert_eq!(r("norm([1,i])"), "(1 + i^2)^(1/2)");
    assert_eq!(r("norm([1+i,1])"), "(2 + i^2 + 2·i)^(1/2)");
    assert_eq!(r("norm([2i])"), "2·|i|"); // sqrt((2i)²) = |2i| with i symbolic
    assert_eq!(r("norm([3i,4i])"), "5·|i|");
    // Complex mode: the magnitude fold lives HERE (its correct domain).
    assert_eq!(rc("norm([3,4i])"), "5"); // NOT sqrt(9+(4i)^2) = i·sqrt(7)
    assert_eq!(rc("norm([1,i])"), "sqrt(2)"); // NOT sqrt(1+i^2) = 0
    assert_eq!(rc("norm([1+i,1])"), "sqrt(3)"); // |1+i|^2 + 1 = 3
    assert_eq!(rc("norm([2i])"), "2");
    assert_eq!(rc("norm([3i,4i])"), "5");
    // Complex mode + symbols: Hermitian form — the V0 P0 fix. `(x^2+y^2)^(1/2)` here was a
    // latent wrong answer (x:=i, y:=1 → 0 instead of sqrt(2)).
    assert_eq!(rc("norm([x,y])"), "(|x|^2 + |y|^2)^(1/2)");
    // Conscious contract: `dot` stays BILINEAR (no conjugation — SymPy's default), so over
    // ℂ `norm(v) ≠ sqrt(dot(v,v))` by design: dot([i,1],[i,1]) = i²+1 = 0 while the norm
    // of [i,1] is sqrt(2) (pinned above).
    assert_eq!(rc("dot([i,1],[i,1])"), "0");
}

#[test]
fn test_eval_gaussian_reciprocal_clean_form() {
    // Fase 2 · residual C1 del frente complejo, cerrado: `(z)^(-1)` llegaba mangled
    // (`(1/2·2 - i)/(2)`) SOLO por la ruta Pow(z,-1) — AddFractions construía el
    // numerador cruzado con `Mul(Number, Number)` crudo (mul2_raw) y el pipeline
    // abandonaba el ciclo combina↔separa en el lado sin plegar (los gemelos reales
    // se limpiaban vía factor-out, que declina con `i`). El builder ahora pliega
    // Number×Number exacto en la emisión.
    let rc = |input: &str| -> String {
        let out = cli()
            .args([
                "eval",
                input,
                "--value-domain",
                "complex",
                "--format",
                "json",
            ])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(rc("(1+i)^(-1)"), "1/2 - 1/2·i");
    assert_eq!(rc("(2+i)^(-1)"), "2/5 - 1/5·i");
    assert_eq!(rc("(1-i)^(-1)"), "1/2 + 1/2·i");
    assert_eq!(rc("(3+4i)^(-1)"), "3/25 - 4/25·i");
    // Verificación de valor: z·z^(-1) = 1 exacto.
    assert_eq!(rc("(1+i)*(1+i)^(-1)"), "1");
    assert_eq!(rc("(2+i)*(2+i)^(-1)"), "1");
}

#[test]
fn test_eval_diff_multivar_input_latex() {
    // Tanda-2 ciclo 5 arregló el DROP de variables; el cierre vectorial C aplica la
    // decisión del usuario (pregunta abierta #3): ∂ GLOBAL cuando la derivación
    // involucra más de una variable (mixtas O target multivariable); el univariable
    // conserva `d` BYTE-idéntico. Denominador derecha-a-izquierda (convención de
    // parciales mixtas).
    let latex_of = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["input_latex"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        latex_of("diff(x^2*y^3, x, y)"),
        "\\frac{\\partial^{2}}{\\partial y \\, \\partial x}({x}^{2}\\cdot {y}^{3})"
    );
    assert_eq!(
        latex_of("diff(x^2*y, x)"),
        "\\frac{\\partial}{\\partial x}(y\\cdot {x}^{2})"
    );
    assert_eq!(
        latex_of("diff(x^5, x, 2)"),
        "\\frac{d^{2}}{dx^{2}}({x}^{5})"
    );
    assert_eq!(
        latex_of("diff(x^2*y^2, x, 2, y)"),
        "\\frac{\\partial^{3}}{\\partial y \\, \\partial x^{2}}({x}^{2}\\cdot {y}^{2})"
    );
    // El 2-args queda BYTE-IDÉNTICO (ambos renderers).
    assert_eq!(latex_of("diff(x^2, x)"), "\\frac{d}{dx}({x}^{2})");
    assert_eq!(
        latex_of("sqrt(diff(x^2,x))"),
        "\\sqrt{\\frac{d}{dx}({x}^{2})}"
    );
}

#[test]
fn test_eval_complex_rule_names_localized() {
    // Fase 2 · C2: los nombres de regla del frente complejo llegan al wire
    // LOCALIZADOS (es fuente, en vía tabla) — la "barra baja" del frente elevada al
    // patrón de los verbos vectoriales. Solo AÑADIR claves, jamás editar existentes.
    let rules_of = |input: &str, lang: Option<&str>| -> Vec<String> {
        let mut args = vec![
            "eval",
            input,
            "--value-domain",
            "complex",
            "--steps",
            "on",
            "--format",
            "json",
        ];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["steps"]
            .as_array()
            .map(|steps| {
                steps
                    .iter()
                    .map(|s| s["rule"].as_str().unwrap_or("").to_string())
                    .collect()
            })
            .unwrap_or_default()
    };
    assert_eq!(
        rules_of("(3+4i)/(1-2i)", None),
        vec!["Dividir números complejos"]
    );
    assert_eq!(
        rules_of("(3+4i)/(1-2i)", Some("en")),
        vec!["Divide complex numbers"]
    );
    assert_eq!(
        rules_of("sin(i)", None),
        vec!["Aplicar trigonometría de argumento imaginario"]
    );
    let euler_unimodular = rules_of("abs(e^(2*i))", Some("en"));
    assert_eq!(
        euler_unimodular,
        vec![
            "Apply Euler's formula",
            "Apply the unimodular absolute value"
        ]
    );
}

#[test]
fn test_eval_limit_oscillation_dne() {
    // Tanda-3 ciclo 3 (item del frontier-audit estrechado a exactamente esto por el
    // meta-audit): sin/cos/tan(g) con lateral de g probadamente ±∞ → el límite NO
    // EXISTE por oscilación, con motivo educativo — como ya hacían los laterales
    // discrepantes. Conservador: sin divergencia probada del argumento, residual.
    let eval_full = |input: &str| -> (String, String) {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        let warnings = wire["warnings"]
            .as_array()
            .map(|w| {
                w.iter()
                    .filter_map(|x| x["assumption"].as_str())
                    .collect::<Vec<_>>()
                    .join(" | ")
            })
            .unwrap_or_default();
        (wire["result"].as_str().unwrap_or("").to_string(), warnings)
    };
    for probe in [
        "limit(sin(1/x), x, 0)",
        "limit(cos(1/x), x, 0)",
        "limit(tan(1/x), x, 0)",
        "limit(sin(1/(x-2)), x, 2)",
        "limit(cos(1/x^2), x, 0)",
    ] {
        let (result, warnings) = eval_full(probe);
        assert_eq!(result, "undefined", "{probe}");
        assert!(
            warnings.contains("OSCILLATES"),
            "{probe} debe llevar el motivo de oscilación, got: {warnings}"
        );
    }
    // Pins: el sandwich, el notable, los laterales discrepantes (SU motivo propio),
    // el infinito y el continuo quedan intactos.
    assert_eq!(eval_full("limit(x*sin(1/x), x, 0)").0, "0");
    assert_eq!(eval_full("limit(sin(x)/x, x, 0)").0, "1");
    let (r, w) = eval_full("limit(abs(x)/x, x, 0)");
    assert_eq!(r, "undefined");
    assert!(w.contains("one-sided limits disagree"));
    assert_eq!(eval_full("limit(sin(1/x), x, infinity)").0, "0");
    assert_eq!(eval_full("limit(sin(x), x, 0)").0, "0");
}

#[test]
fn test_eval_limit_complex_domain_kill_switch() {
    // Fase 3 · F0 (P0): el motor de límites razona con el ORDEN REAL — bajo
    // `--value-domain complex` fabricaba valores (`e^(-1/z²)→0` y `z·sin(1/z)→0`
    // cuando en ℂ NINGUNO existe: singularidad esencial). El kill-switch de
    // entrada declina TODO límite complejo a residual honesto; F10/F11 re-otorgan
    // selectivamente con justificación analítica. Los 7 WRONG del scoping:
    let eval_complex = |input: &str| -> (String, String) {
        let out = cli()
            .args([
                "eval",
                input,
                "--value-domain",
                "complex",
                "--format",
                "json",
            ])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        let warnings = wire["warnings"]
            .as_array()
            .map(|w| {
                w.iter()
                    .filter_map(|x| x["assumption"].as_str())
                    .collect::<Vec<_>>()
                    .join(" | ")
            })
            .unwrap_or_default();
        (wire["result"].as_str().unwrap_or("").to_string(), warnings)
    };
    for probe in [
        "limit(e^(-1/z^2), z, 0)",
        "limit(z*sin(1/z), z, 0)",
        "limit(tanh(z), z, i*pi/2)",
        "limit(atan(z), z, 2*i)",
        "limit(1/(z^2+1), z, i*1)",
        "limit(1/z^2, z, 0)",
        "limit(e^z, z, infinity)",
    ] {
        let (result, warnings) = eval_complex(probe);
        assert!(
            result.starts_with("limit("),
            "{probe} debe declinar a residual bajo complex, got: {result}"
        );
        assert!(
            warnings.contains("complex value domain"),
            "{probe} debe llevar el motivo del kill-switch, got: {warnings}"
        );
    }
    // Never-fabricate (cuerpo-con-I): la protección era coincidental (declinaban
    // por Polynomial-sobre-ℚ); estos pins la vuelven CONTRATO — F11 los hereda.
    for probe in ["limit(e^(i*x), x, infinity)", "limit(i*sin(x)/x, x, 0)"] {
        let (result, _) = eval_complex(probe);
        assert!(
            result.starts_with("limit("),
            "{probe} jamás fabrica bajo complex, got: {result}"
        );
    }
}

#[test]
fn test_eval_limit_imaginary_point_real_domain_residual() {
    // Fase 3 · F0, mitad real: el gate léxico del wire es un colador (`i` desnudo
    // rechaza pero `2*i`/`i*pi`/`i*1` PASAN) y el motor sustituía el punto — en el
    // polo de tanh en iπ/2 emitía el VALOR `tanh(pi·i/2)`. Punto-con-I en dominio
    // real → residual honesto + el Imaginary Usage Warning estándar (el mismo
    // escape-hatch que ofrece simplify).
    let eval_full = |input: &str| -> (String, String) {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        let warnings = wire["warnings"]
            .as_array()
            .map(|w| {
                w.iter()
                    .filter_map(|x| x["assumption"].as_str())
                    .collect::<Vec<_>>()
                    .join(" | ")
            })
            .unwrap_or_default();
        (wire["result"].as_str().unwrap_or("").to_string(), warnings)
    };
    for probe in [
        "limit(tanh(z), z, i*pi/2)",
        "limit(1/(z^2+1), z, i*1)",
        "limit(atan(z), z, 2*i)",
        // F0b: evasión del barrido adversarial — punto imaginario deletreado como
        // raíz par de constante negativa NO-racional (sqrt(-pi^2) = i·pi alcanzaba
        // el polo de tanh); el detector decide con provable_const_sign, exacto.
        "limit(tanh(x), x, sqrt(-pi^2)/2)",
    ] {
        let (result, warnings) = eval_full(probe);
        assert!(
            result.starts_with("limit("),
            "{probe} debe declinar a residual con punto imaginario, got: {result}"
        );
        assert!(
            warnings.contains("imaginary unit"),
            "{probe} debe llevar el motivo del punto imaginario, got: {warnings}"
        );
        assert!(
            warnings.contains("semantics set value complex"),
            "{probe} debe ofrecer el escape-hatch estándar, got: {warnings}"
        );
    }
    // Never-fabricate en real (cuerpo-con-I, punto real): siguen residual.
    for probe in ["limit(e^(i*x), x, infinity)", "limit(i*sin(x)/x, x, 0)"] {
        let (result, _) = eval_full(probe);
        assert!(
            result.starts_with("limit("),
            "{probe} jamás fabrica en real, got: {result}"
        );
    }
    // Pins real byte-idénticos: el guard es invisible sin `i` en el punto.
    assert_eq!(eval_full("limit(sin(x)/x, x, 0)").0, "1");
    assert_eq!(eval_full("limit(e^z, z, 2)").0, "exp(2)");
    let (r, w) = eval_full("limit(1/x, x, 0)");
    assert_eq!(r, "undefined");
    assert!(w.contains("one-sided limits disagree"));
}

#[test]
fn test_eval_solve_critical_points() {
    // Cierre vectorial · V7d (decisión del usuario): los diff inline se pre-evalúan
    // en el path de solve_system (con fold numérico de los artefactos x^(2-1)), así
    // que el flujo de puntos críticos con gradiente LINEAL resuelve one-shot; el
    // no-lineal sigue declinando honesto (scope-out: Gröbner = mate-nueva).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("solve([diff(x^2+y^2-2*x-4*y, x)=0, diff(x^2+y^2-2*x-4*y, y)=0], [x,y])"),
        "{ x = 1, y = 2 }"
    );
    // Sistema acoplado (parciales cruzadas): ∇(x²+xy+y²−3x) = 0 → (2, −1).
    assert_eq!(
        r("solve([diff(x^2+x*y+y^2-3*x, x)=0, diff(x^2+x*y+y^2-3*x, y)=0], [x,y])"),
        "{ x = 2, y = -1 }"
    );
    // El flujo curricular COMPLETO compone: clasificación en el crítico hallado.
    assert_eq!(
        r("subs(subs(det(hessian(x^2+y^2-2*x-4*y,[x,y])), x, 1), y, 2)"),
        "4"
    );
    // Pins: lineal puro intacto; gradiente NO-lineal declina honesto.
    assert_eq!(r("solve([x+y=3, x-y=1], [x,y])"), "{ x = 2, y = 1 }");
    let out = cli()
        .args([
            "eval",
            "solve([diff(x^3+y^3-3*x*y, x)=0, diff(x^3+y^3-3*x*y, y)=0], [x,y])",
        ])
        .output()
        .expect("Failed to run CLI");
    let text =
        String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    assert!(
        text.contains("non-linear") || text.contains("degree > 1"),
        "el gradiente no-lineal debe declinar honesto, got: {text}"
    );
}

#[test]
fn test_eval_subs_inline() {
    // Cierre vectorial · pregunta abierta #2 (decisión del usuario): subs(expr, x, v)
    // — evaluación-en-punto inline, ORDER-SAFE por construcción (declina mientras el
    // target contenga cálculo sin evaluar: la cascada deriva ANTES de ligar, la trampa
    // de orden del flujo let no puede ocurrir). Multi-variable por anidamiento.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("subs(x^2, x, 3)"), "9");
    assert_eq!(r("subs(x^2+y, x, 2)"), "y + 4");
    assert_eq!(r("subs(subs(x^2*y, x, 1), y, 2)"), "2");
    assert_eq!(r("subs(x^2, x, x+1)"), "(x + 1)^2");
    // LA trampa de orden, resuelta: deriva primero, liga después — siempre.
    assert_eq!(r("subs(diff(x^2*y, x), x, 1)"), "2·y");
    assert_eq!(r("subs(subs(diff(x^2*y, x), x, 1), y, 3)"), "6");
    // Plano tangente one-shot (f=x²y en (1,2)): f=2, fx=4, fy=1.
    assert_eq!(r("subs(subs(x^2*y, x, 1), y, 2)"), "2");
    assert_eq!(r("subs(subs(diff(x^2*y,x), x, 1), y, 2)"), "4");
    assert_eq!(r("subs(subs(diff(x^2*y,y), x, 1), y, 2)"), "1");
    // Clasificación de críticos one-shot: det(H) de x³+y³−3xy en (1,1) → 27 > 0.
    assert_eq!(
        r("subs(subs(det(hessian(x^3+y^3-3*x*y,[x,y])), x, 1), y, 1)"),
        "27"
    );
    // Declines honestos: cálculo residual dentro (no-elemental) y var no-Variable.
    assert_eq!(
        r("subs(integrate(1/ln(x), x), x, 2)"),
        "subs(integrate(1 / ln(x), x), x, 2)"
    );
    assert_eq!(r("subs(x^2, 3, 1)"), "subs(x^2, 3, 1)");
}

#[test]
fn test_eval_reciprocal_cis() {
    // Tanda-3 ciclo 4: n/(cos u ± i·sin u) → n·(cos u ∓ i·sin u) — identidad ENTERA
    // (cis·cis̄ = cos²+sin² = 1 en todo ℂ), sin guard de realidad. Cierra el residual
    // B2: la canonicalización de exponente negativo convertía e^(-ix) en 1/e^(ix)
    // ANTES de Euler, y Euler expandía solo el denominador.
    let rc = |input: &str| -> String {
        let out = cli()
            .args([
                "eval",
                input,
                "--value-domain",
                "complex",
                "--format",
                "json",
            ])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(rc("e^(-i*x)"), "cos(x) - i·sin(x)");
    assert_eq!(rc("exp(-i*x)"), "cos(x) - i·sin(x)");
    assert_eq!(rc("1/(cos(x)+i*sin(x))"), "cos(x) - i·sin(x)");
    assert_eq!(rc("2/(cos(x)+i*sin(x))"), "2·cos(x) - 2·i·sin(x)");
    assert_eq!(rc("1/(cos(x)-i*sin(x))"), "cos(x) + i·sin(x)");
    // Pins: Euler directo intacto; la unimodularidad sigue viva sobre el matcher
    // compartido refactorizado; denominador no-cis intacto; real mode gated.
    assert_eq!(rc("e^(i*x)"), "cos(x) + i·sin(x)");
    assert_eq!(rc("abs(e^(2*i))"), "1");
    assert_eq!(rc("1/(cos(x)+sin(x))"), "1 / (sin(x) + cos(x))");
    assert_eq!(r("e^(-i*x)"), "1 / e^(i·x)");
}

#[test]
fn test_eval_gaussian_surd_modulus() {
    // Tanda-3 ciclo 2: |a+b·i| con componentes reales DECIDIBLES (provable_const_sign
    // — surds, e/π; la disciplina V0 hace declinar los símbolos). Cierra la familia
    // π-racional que la unimodularidad dejó nombrada (el trig pliega a surds ANTES
    // del abs). El caso ambos-racionales conserva su dueño exacto (GaussianRational).
    let rc = |input: &str| -> String {
        let out = cli()
            .args([
                "eval",
                input,
                "--value-domain",
                "complex",
                "--format",
                "json",
            ])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // La familia π-racional cierra: unimodulares por la vía surd.
    assert_eq!(rc("abs(1/2 + i*sqrt(3)/2)"), "1");
    assert_eq!(rc("abs(e^(i*pi/3))"), "1");
    assert_eq!(rc("abs(e^(i*pi/4))"), "1");
    // Módulos surd generales, forma factorizada incluida.
    assert_eq!(rc("abs(1 + i*sqrt(3))"), "2");
    assert_eq!(rc("abs(sqrt(2) + i*sqrt(2))"), "2");
    // Puro-imaginario decidible: el signo ya está decidido — emite ±b directo.
    assert_eq!(rc("abs(i*sqrt(3))"), "sqrt(3)");
    assert_eq!(rc("abs(-i*sqrt(3))"), "sqrt(3)");
    assert_eq!(rc("abs(i*pi)"), "pi");
    // Transcendentales: forma exacta sin plegar, sound.
    assert_eq!(rc("abs(e + i*pi)"), "(pi^2 + e^2)^(1/2)");
    // Ownership: racionales al dueño exacto; símbolos declinan (disciplina V0).
    assert_eq!(rc("abs(3+4*i)"), "5");
    assert_eq!(rc("abs(x + i*sqrt(3))"), "|x + i·sqrt(3)|");
}

#[test]
fn test_eval_complex_angle_sum() {
    // Tanda-3 ciclo 1: argumento complejo MIXTO re+iθ — la suma de ángulos entera
    // (válida ∀ re,θ ∈ ℂ, sin guard, como el puente). Puro-imaginario sigue siendo
    // del puente; real mode intacto. ONE-DIRECTION (la contracción trig casa
    // cos/sin, jamás cosh/sinh — no existe lado de ping-pong).
    let rc = |input: &str| -> String {
        let out = cli()
            .args([
                "eval",
                input,
                "--value-domain",
                "complex",
                "--format",
                "json",
            ])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(rc("sin(1+i)"), "sin(1)·cosh(1) + i·cos(1)·sinh(1)");
    assert_eq!(rc("cos(1+i)"), "cos(1)·cosh(1) - i·sin(1)·sinh(1)");
    assert_eq!(rc("sin(x+i)"), "sin(x)·cosh(1) + i·cos(x)·sinh(1)");
    assert_eq!(rc("sin(2+3*i)"), "sin(2)·cosh(3) + i·cos(2)·sinh(3)");
    assert_eq!(rc("sinh(1+i)"), "cos(1)·sinh(1) + i·sin(1)·cosh(1)");
    assert_eq!(rc("cosh(1+i)"), "cos(1)·cosh(1) + i·sin(1)·sinh(1)");
    // tan compone vía Tan→Sin/Cos + esta regla: cociente honesto expandido.
    assert_eq!(
        rc("tan(1+i)"),
        "(sin(1)·cosh(1) + i·cos(1)·sinh(1)) / (cos(1)·cosh(1) - i·sin(1)·sinh(1))"
    );
    // Ownership intacto: puro-imaginario del puente, real sin i, real mode gated.
    assert_eq!(rc("sin(i)"), "i·sinh(1)");
    assert_eq!(rc("sin(2)"), "sin(2)");
    assert_eq!(r("sin(1+i)"), "sin(1 + i)");
    // Verificación cruzada con el walker (independiente de la regla).
    assert_eq!(rc("approx(sin(1+i))"), "1.29845758142 + 0.634963914785·i");
}

#[test]
fn test_eval_trig_of_imaginary_bridge() {
    // Fase 2 · trig-de-i (residual B4b): el puente trig↔hiperbólico de argumento
    // puro-imaginario — identidades ENTERAS (válidas para y complejo arbitrario, sin
    // guard de realidad, a diferencia de la unimodularidad). ONE-DIRECTION.
    let rc = |input: &str| -> String {
        let out = cli()
            .args([
                "eval",
                input,
                "--value-domain",
                "complex",
                "--format",
                "json",
            ])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Los 6 brazos, literal y simbólico.
    assert_eq!(rc("sin(i)"), "i·sinh(1)");
    assert_eq!(rc("cos(i)"), "cosh(1)");
    assert_eq!(rc("tan(i)"), "i·tanh(1)");
    assert_eq!(rc("sin(i*x)"), "i·sinh(x)");
    assert_eq!(rc("sinh(i)"), "i·sin(1)");
    assert_eq!(rc("cosh(i*x)"), "cos(x)");
    assert_eq!(rc("tanh(3*i)"), "i·tan(3)");
    // Composición exacta a través del puente: cosh(iπ) = cos(π) = -1.
    assert_eq!(rc("cosh(i*pi)"), "-1");
    // El decline de argumento mixto GRADUÓ en tanda-3 ciclo 1 (ComplexAngleSumRule);
    // el pin migra a la forma expandida — la propiedad de ESTE test (puro-imaginario
    // es del puente) se conserva en los asserts de arriba.
    assert_eq!(rc("sin(1+i)"), "sin(1)·cosh(1) + i·cos(1)·sinh(1)");
    assert_eq!(r("sin(i)"), "sin(i)");
    // La red B1 con los brazos hiperbólicos nuevos del walker: refuta la identidad
    // FALSA (jamás confirma la verdadera desde probe — el wire equiv es Bool
    // exacto-solo, "false" = no-probado; residual nombrado).
    assert_eq!(rc("equiv(sin(i), i*sinh(2))"), "false");
    // approx compone con el puente (el walker evalúa sinh complejo).
    assert_eq!(rc("approx(sin(i))"), "1.17520119364·i");
}

#[test]
fn test_eval_unimodular_abs() {
    // Fase 2 · residual B2 cerrado con disciplina V0: `|cos θ ± i·sin θ| = 1` SOLO con
    // θ constante real DECIDIBLE (provable_const_sign). Una variable DEBE declinar:
    // bajo ComplexEnabled puede tomar valor complejo y la unimodularidad es falsa
    // (x:=i ⇒ |e^(i·i)| = 1/e ≠ 1) — el mismo sticky-fold que V0 mató en norm.
    let rc = |input: &str| -> String {
        let out = cli()
            .args([
                "eval",
                input,
                "--value-domain",
                "complex",
                "--format",
                "json",
            ])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Graduates: literal, surd (const_sign prueba sqrt(2)), conjugado, vía Euler.
    assert_eq!(rc("abs(e^(2*i))"), "1");
    assert_eq!(rc("abs(cos(2)+i*sin(2))"), "1");
    assert_eq!(rc("abs(e^(i*sqrt(2)))"), "1");
    assert_eq!(rc("abs(cos(2)-i*sin(2))"), "1");
    assert_eq!(rc("abs(e^(-2*i))"), "1");
    // Declines V0-discipline: símbolo (puede ser complejo), θ distinto, θ=i, real mode.
    assert_eq!(rc("abs(cos(x)+i*sin(x))"), "|cos(x) + i·sin(x)|");
    // El mismatch θ≠θ' dejó de ser residual en tanda-3 ciclo 2: el módulo
    // Gaussiano-surd lo computa por la vía general (const_sign acota trig de
    // racionales) → √(sin(3)²+cos(2)²), correcto y más informativo. La propiedad
    // de ESTE test (unimodularidad solo con θ IGUAL y decidible) sigue fijada
    // por los asserts de arriba.
    assert_eq!(rc("abs(cos(2)+i*sin(3))"), "(sin(3)^2 + cos(2)^2)^(1/2)");
    // θ=i: la unimodularidad sigue DECLINANDO aquí, pero el puente trig-de-i
    // (ciclo 3) compone |cosh(1) − sinh(1)| = 1/e — exactamente el contraejemplo
    // |e^(i·i)| = 1/e que motiva el guard de realidad. El valor confirma la
    // disciplina: NO es 1.
    assert_eq!(rc("abs(cos(i)+i*sin(i))"), "1 / e");
    assert_eq!(r("abs(cos(2)+i*sin(2))"), "|cos(2) + i·sin(2)|");
}

#[test]
fn test_eval_componentwise_diff_over_matrix() {
    // Fase 2 V1: `diff` distributes componentwise over a `Matrix` target, ALL-OR-NOTHING
    // (a non-differentiable component keeps the whole call an honest residual), and the
    // higher-order desugar composes for free (`diff(M, x, 2)` = nested componentwise diffs).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("diff([x^2,x^3],x)"), "[[2·x], [3·x^2]]");
    assert_eq!(r("diff([x^2*y, sin(x)],x)"), "[[2·x·y], [cos(x)]]");
    // Higher-order rides the target-agnostic desugar: diff(M,x,2) → diff(diff(M,x),x).
    assert_eq!(r("diff([x^2,x^3],x,2)"), "[[2], [6·x]]");
    // All-or-nothing: sign(x) has no derivative here, so NOTHING is derived (never a
    // half-differentiated matrix).
    assert_eq!(r("diff([x, sign(x)],x)"), "diff([[x], [sign(x)]], x)");
    // Var-list stays a decline (the list-of-vars arity belongs to the vectorial verbs, V3+).
    assert_eq!(r("diff(x^2+y^2,[x,y])"), "diff(x^2 + y^2, [[x], [y]])");
    // Scalar pins: the componentwise arm must not disturb the scalar cascade.
    assert_eq!(r("diff(x^2*y^3,x,y)"), "6·x·y^2");
    assert_eq!(r("wronskian([x^2,x^3],x)"), "x^4");
    // Narration: the componentwise arm emits a visible step (the diff call is the root, so
    // the Matrix-as-leaf wire gap does not swallow it).
    let out = cli()
        .args([
            "eval",
            "diff([x^2,x^3],x)",
            "--steps",
            "on",
            "--format",
            "json",
        ])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
    let steps = wire["steps"].as_array().expect("steps array");
    assert!(
        !steps.is_empty(),
        "componentwise diff at the root must narrate"
    );
    assert_eq!(
        steps[0]["rule"].as_str().unwrap_or(""),
        "Calcular la derivada"
    );
}

#[test]
fn test_eval_gradient_verb() {
    // Fase 2 V3: `gradient(f, [vars])` / alias `grad` — the first vectorial verb. Output
    // is an n×1 COLUMN (the parser's own [x,y] convention), re-enterable and composable.
    // The list-of-vars arity is EXCLUSIVE to the verbs (arity-2 exact); `diff`'s 3+-arity
    // SymPy convention keeps its own owner.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("gradient(x^2+y^2,[x,y])"), "[[2·x], [2·y]]");
    assert_eq!(r("gradient(x^2*y,[x,y])"), "[[2·x·y], [x^2]]");
    assert_eq!(r("grad(x*y*z,[x,y,z])"), "[[y·z], [x·z], [x·y]]");
    assert_eq!(
        r("gradient(sin(x*y),[x,y])"),
        "[[y·cos(x·y)], [x·cos(x·y)]]"
    );
    // Composition (the result is a live Matrix): norm — pin THIS form, the engine does
    // not extract the square factor from the radical — and the directional derivative.
    assert_eq!(r("norm(gradient(x^2+y^2,[x,y]))"), "(4·x^2 + 4·y^2)^(1/2)");
    assert_eq!(r("dot(gradient(x^2*y,[x,y]),[1,0])"), "2·x·y");
    // Honest declines: non-variable list entry, Matrix field (jacobian territory, V4),
    // over-cap var list (VERB_MAX_VARS=8).
    assert_eq!(r("gradient(x^2,[x,2])"), "gradient(x^2, [[x], [2]])");
    assert_eq!(
        r("gradient([x,y],[x,y])"),
        "gradient([[x], [y]], [[x], [y]])"
    );
    assert_eq!(
        r("gradient(x^2, [a,b,c,d,e,f,g,h,k])"),
        "gradient(x^2, [[a], [b], [c], [d], [e], [f], [g], [h], [k]])"
    );
    // Never-confirm fixture: the OTHER verbs stay unregistered until their cycle lands
    // (detector of the gate-without-rule gotcha, in both directions).
    let err_of = |input: &str| -> String {
        let out = cli()
            .args(["eval", input])
            .output()
            .expect("Failed to run CLI");
        String::from_utf8_lossy(&out.stderr).to_string() + &String::from_utf8_lossy(&out.stdout)
    };
    // The six verbs are all registered now; the detector moves to the Fase-3
    // scope-out names, whose "función no definida" decline is part of the contract.
    for probe in [
        "lineintegral(x^2, [x,y])",
        "surface_integral(x*y, [x,y,z])",
        "potential([2*x*y, x^2], [x,y])",
    ] {
        assert!(
            err_of(probe).contains("no definida"),
            "Fase-3 scope-out name must stay 'función no definida': {probe}"
        );
    }
    // Narration: rule name localizes es/en; one keyed substep per component.
    let steps_json = |input: &str, lang: Option<&str>| -> Value {
        let mut args = vec!["eval", input, "--steps", "on", "--format", "json"];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        serde_json::from_slice(&out.stdout).expect("Invalid wire output")
    };
    let es = steps_json("gradient(x^2+y^2,[x,y])", None);
    assert_eq!(
        es["steps"][0]["rule"].as_str().unwrap(),
        "Calcular el gradiente"
    );
    let subs = es["steps"][0]["substeps"].as_array().expect("substeps");
    assert_eq!(subs.len(), 2, "one substep per component");
    assert!(subs[0]["title"]
        .as_str()
        .unwrap()
        .contains("Derivar respecto de x"));
    let en = steps_json("gradient(x^2+y^2,[x,y])", Some("en"));
    assert_eq!(
        en["steps"][0]["rule"].as_str().unwrap(),
        "Compute the gradient"
    );
    assert!(en["steps"][0]["substeps"][0]["title"]
        .as_str()
        .unwrap()
        .contains("Differentiate with respect to x"));
}

#[test]
fn test_eval_jacobian_hessian_verbs() {
    // Fase 2 V4: jacobian (ROWS = functions, COLUMNS = variables — the orientation pin)
    // and hessian (n×n symmetric, computed as the jacobian of the internal gradient),
    // plus the bracket-aware `equiv` micro-cable that powers the metamorphic fixture.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("jacobian([x^2*y, x+y],[x,y])"), "[[2·x·y, x^2], [1, 1]]");
    assert_eq!(
        r("jacobian([x*y, x+y, sin(x)],[x,y])"),
        "[[y, x], [1, 1], [cos(x), 0]]"
    );
    assert_eq!(r("hessian(x^2*y,[x,y])"), "[[2·y, 2·x], [2·x, 0]]");
    assert_eq!(r("hessian(x^2+y^2,[x,y])"), "[[2, 0], [0, 2]]");
    // det(hessian) composes — the second-derivative-test discriminant.
    assert_eq!(r("det(hessian(x^2*y,[x,y]))"), "-4·x^2");
    // Metamorphic fixture via bracket-aware equiv: hessian ≡ jacobian ∘ gradient.
    assert_eq!(
        r("equiv(jacobian(gradient(x^3*y^2,[x,y]),[x,y]), hessian(x^3*y^2,[x,y]))"),
        "true"
    );
    assert_eq!(r("equiv([x,y],[y,x])"), "false");
    // Scalar equiv pins intact (the shared splitter now tracks brackets).
    assert_eq!(r("equiv(diff(x^2,x), 2*x)"), "true");
    assert_eq!(r("equiv(x+1, x+2)"), "false");
    // Honest declines: scalar target for jacobian (gradient owns scalars), matrix field
    // for hessian, general matrix target for jacobian.
    assert_eq!(r("jacobian(x^2,[x,y])"), "jacobian(x^2, [[x], [y]])");
    assert_eq!(r("hessian([x,y],[x,y])"), "hessian([[x], [y]], [[x], [y]])");
    assert_eq!(
        r("jacobian([[1,2],[3,4]],[x,y])"),
        "jacobian([[1, 2], [3, 4]], [[x], [y]])"
    );
    // Narration: localized rule names + one keyed substep per row.
    let steps_json = |input: &str, lang: Option<&str>| -> Value {
        let mut args = vec!["eval", input, "--steps", "on", "--format", "json"];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        serde_json::from_slice(&out.stdout).expect("Invalid wire output")
    };
    let es = steps_json("jacobian([x^2*y, x+y],[x,y])", None);
    assert_eq!(
        es["steps"][0]["rule"].as_str().unwrap(),
        "Calcular el jacobiano"
    );
    let subs = es["steps"][0]["substeps"].as_array().expect("substeps");
    assert_eq!(subs.len(), 2, "one substep per row");
    assert!(subs[0]["title"].as_str().unwrap().contains("Fila 1"));
    let en = steps_json("hessian(x^2*y,[x,y])", Some("en"));
    assert_eq!(
        en["steps"][0]["rule"].as_str().unwrap(),
        "Compute the Hessian"
    );
    assert!(en["steps"][0]["substeps"][0]["title"]
        .as_str()
        .unwrap()
        .contains("Row 1"));
}

#[test]
fn test_eval_divergence_laplacian_verbs() {
    // Fase 2 V5: the scalar-output verbs. divergence REQUIRES #components == #vars
    // (mismatch → honest residual, never undefined); laplacian = div∘grad computed
    // internally; vector-laplacian stays a named scope-out. Both carry the bounded
    // budget exemption — without it a raw sum of quotient derivatives was a FALSE
    // residual (laplacian(ln(x²+y²)) hit the anti-worsen budget).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("divergence([x^2,y^2],[x,y])"), "2·x + 2·y");
    assert_eq!(r("divergence([x*y, y*z, z*x],[x,y,z])"), "x + y + z");
    assert_eq!(r("laplacian(x^2+y^2,[x,y])"), "4");
    assert_eq!(r("laplacian(x^2+y^2+z^2,[x,y,z])"), "6");
    assert_eq!(r("laplacian(sin(x)*cos(y),[x,y])"), "-2·sin(x)·cos(y)");
    // The classic HARMONIC check: Δ ln(x²+y²) = 0 exactly (this was a false residual
    // before the bounded exemption — the budget-rejection class, chokepoint-D).
    assert_eq!(r("laplacian(ln(x^2+y^2),[x,y])"), "0");
    // Honest declines: component/var mismatch, scalar target for divergence,
    // vector-laplacian scope-out.
    assert_eq!(
        r("divergence([x^2,y^2],[x,y,z])"),
        "divergence([[x^2], [y^2]], [[x], [y], [z]])"
    );
    assert_eq!(r("divergence(x^2,[x,y])"), "divergence(x^2, [[x], [y]])");
    assert_eq!(
        r("laplacian([x,y],[x,y])"),
        "laplacian([[x], [y]], [[x], [y]])"
    );
    // Narration: localized rule names + the defining-formula keyed substep.
    let steps_json = |input: &str, lang: Option<&str>| -> Value {
        let mut args = vec!["eval", input, "--steps", "on", "--format", "json"];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        serde_json::from_slice(&out.stdout).expect("Invalid wire output")
    };
    let es = steps_json("divergence([x^2,y^2],[x,y])", None);
    assert_eq!(
        es["steps"][0]["rule"].as_str().unwrap(),
        "Calcular la divergencia"
    );
    assert!(es["steps"][0]["substeps"][0]["title"]
        .as_str()
        .unwrap()
        .contains("∇·F"));
    let en = steps_json("laplacian(x^2+y^2,[x,y])", Some("en"));
    assert_eq!(
        en["steps"][0]["rule"].as_str().unwrap(),
        "Compute the Laplacian"
    );
    assert!(en["steps"][0]["substeps"][0]["title"]
        .as_str()
        .unwrap()
        .contains("second derivatives"));
}

#[test]
fn test_eval_curl_verb() {
    // Fase 2 V6: curl 3D (3×1 column, standard sign convention) and 2D (SCALAR
    // ∂Q/∂x − ∂P/∂y — never zero-padded), alias rot, and the conservativity
    // metamorphics that tie the verbs together.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("curl([y,-x,0],[x,y,z])"), "[[0], [0], [-2]]");
    assert_eq!(r("curl([y,-x],[x,y])"), "-2"); // 2D = SCALAR (pinned convention)
    assert_eq!(r("rot([x*y, y*z, z*x],[x,y,z])"), "[[-y], [-z], [-x]]");
    // Conservativity test (the elemental half of the potential-field item): a gradient
    // field is irrotational — and div∘curl vanishes identically.
    assert_eq!(
        r("curl(gradient(x*y*z,[x,y,z]),[x,y,z])"),
        "[[0], [0], [0]]"
    );
    assert_eq!(r("curl([y,x],[x,y])"), "0");
    assert_eq!(
        r("equiv(divergence(curl([x*y, y*z, z*x],[x,y,z]),[x,y,z]), 0)"),
        "true"
    );
    // Honest declines: 2 components with 3 vars, 4D, scalar target.
    assert_eq!(
        r("curl([x,y],[x,y,z])"),
        "curl([[x], [y]], [[x], [y], [z]])"
    );
    assert_eq!(r("curl(x^2,[x,y])"), "curl(x^2, [[x], [y]])");
    // Narration: localized rule name + shape-aware formula substep (3D vs 2D).
    let steps_json = |input: &str, lang: Option<&str>| -> Value {
        let mut args = vec!["eval", input, "--steps", "on", "--format", "json"];
        if let Some(l) = lang {
            args.extend(["--lang", l]);
        }
        let out = cli().args(&args).output().expect("Failed to run CLI");
        serde_json::from_slice(&out.stdout).expect("Invalid wire output")
    };
    let es = steps_json("curl([y,-x,0],[x,y,z])", None);
    assert_eq!(
        es["steps"][0]["rule"].as_str().unwrap(),
        "Calcular el rotacional"
    );
    assert!(es["steps"][0]["substeps"][0]["title"]
        .as_str()
        .unwrap()
        .contains("∇×F"));
    let en = steps_json("curl([y,-x],[x,y])", Some("en"));
    assert_eq!(en["steps"][0]["rule"].as_str().unwrap(), "Compute the curl");
    assert!(en["steps"][0]["substeps"][0]["title"]
        .as_str()
        .unwrap()
        .contains("2D curl (scalar)"));
}

#[test]
fn test_eval_abs_vector_and_componentwise_integrate() {
    // Fase 2 V7: (a) |v| of a VECTOR is its Euclidean norm, inheriting V0's domain
    // decision wholesale (never re-deciding it); a general matrix stays residual.
    // (b) integrate distributes componentwise, ALL-OR-NOTHING and conditions-
    // conservative: a non-elementary component (or one whose antiderivative carries
    // required conditions) declines the WHOLE call — the north star's protected
    // residuals never end up half-integrated inside a matrix.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    let rc = |input: &str| -> String {
        let out = cli()
            .args([
                "eval",
                input,
                "--value-domain",
                "complex",
                "--format",
                "json",
            ])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // V7a — abs of a vector is the norm, in BOTH domains (V0 inheritance).
    assert_eq!(r("abs([3,4])"), "5");
    assert_eq!(r("abs([x,y])"), "(x^2 + y^2)^(1/2)");
    assert_eq!(rc("abs([x,y])"), "(|x|^2 + |y|^2)^(1/2)");
    assert_eq!(rc("abs([1,i])"), "sqrt(2)");
    // General matrix: honest residual (matrix modulus ≠ Frobenius norm); scalar abs
    // untouched (the abs family is 4-historic-P0 territory).
    assert_eq!(r("abs([[1,2],[3,4]])"), "|[[1, 2], [3, 4]]|");
    assert_eq!(r("abs(-3)"), "3");
    // V7b — componentwise antiderivatives.
    assert_eq!(r("integrate([x, x^2], x)"), "[[1/2·x^2], [1/3·x^3]]");
    assert_eq!(r("integrate([cos(x), e^x], x)"), "[[sin(x)], [e^x]]");
    assert_eq!(r("integrate([1/x, x], x)"), "[[ln(|x|)], [1/2·x^2]]");
    // ALL-OR-NOTHING: e^(-x²) has no elementary antiderivative — the whole call echoes.
    assert_eq!(
        r("integrate([x, e^(-x^2)], x)"),
        "integrate([[x], [1 / e^(x^2)]], x)"
    );
    // Definite integrals over a Matrix stay an honest residual (indefinite-only scope).
    assert_eq!(
        r("integrate([x,x^2], x, 0, 1)"),
        "integrate([[x], [x^2]], x, 0, 1)"
    );
}

#[test]
fn test_eval_steps_under_matrix_literal() {
    // Fase 2 V2 (P0-wire): steps that fire UNDER a `Matrix` node used to be silently
    // discarded — `rewrite_at_expr_path_with` treated Matrix as a leaf, so the step's
    // global snapshot came back unchanged and the didactic pipeline dropped it (correct
    // values, EMPTY narration). Matrix now descends like Function (flat cell index).
    let steps_of = |input: &str| -> Vec<Value> {
        let out = cli()
            .args(["eval", input, "--steps", "on", "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["steps"].as_array().cloned().unwrap_or_default()
    };
    // The fixed case: two derivatives inside a matrix literal narrate one step each,
    // with GLOBAL (matrix-shaped) snapshots.
    let steps = steps_of("[diff(x^2,x), diff(x^3,x)]");
    assert!(
        steps.len() >= 2,
        "steps under a Matrix literal must narrate (got {})",
        steps.len()
    );
    let first_before = steps[0]["before"].as_str().unwrap_or("");
    assert!(
        first_before.contains("[["),
        "snapshots must be global (matrix-shaped), got: {first_before}"
    );
    // Differential controls (the two shapes that always worked): unchanged emission.
    assert!(
        steps_of("diff(x^2,x) + diff(x^3,x)").len() >= 2,
        "steps under Add"
    );
    assert!(
        !steps_of("sqrt(diff(x^4,x))").is_empty(),
        "steps under Function arg"
    );
}

#[test]
fn test_eval_matmul_function() {
    // `matmul` sat in the eval gate with NO dispatch arm — the live gate-without-rule
    // gotcha (silent residual while `A*B` evaluated). Now it shares the `*` math; a
    // dimension mismatch declines to an honest residual (not undefined).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("matmul([[1,2],[3,4]],[[5,6],[7,8]])"),
        "[[19, 22], [43, 50]]"
    );
    assert_eq!(r("matmul([[1,2]],[[3],[4]])"), "[11]");
    assert_eq!(r("matmul([[1,2]],[[3,4]])"), "matmul([1, 2], [3, 4])");
}

#[test]
fn test_eval_matrix_nullspace() {
    // `nullspace(A)` (aliases `null`/`kernel`) returns a basis of {x : A·x = 0} by exact rational
    // RREF, rows = basis vectors. Verified elsewhere by A·v = 0. A trivial kernel is the zero vector;
    // symbolic entries decline.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("nullspace([[1,2],[2,4]])"), "[-2, 1]");
    assert_eq!(r("nullspace([[1,2,3],[4,5,6],[7,8,9]])"), "[1, -2, 1]");
    assert_eq!(r("nullspace([[1,0],[0,1]])"), "[0, 0]"); // trivial kernel
    assert_eq!(r("nullspace([[1,1,1]])"), "[[-1, 1, 0], [-1, 0, 1]]"); // 2-D kernel
    assert_eq!(r("nullspace([[a,b],[c,d]])"), "nullspace([[a, b], [c, d]])");
}

#[test]
fn test_eval_number_theory_divisors_and_crt() {
    // `divisors(n)` lists the positive divisors (sorted), and `crt` solves a system of congruences
    // (Chinese Remainder Theorem), declining on an inconsistent non-coprime system. sympy-checked.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("divisors(12)"), "[1, 2, 3, 4, 6, 12]");
    assert_eq!(r("divisors(7)"), "[1, 7]");
    assert_eq!(r("divisors(36)"), "[1, 2, 3, 4, 6, 9, 12, 18, 36]");
    assert_eq!(r("crt([2,3],[3,5])"), "8"); // x≡2 (mod 3), x≡3 (mod 5)
    assert_eq!(r("crt([1,2,3],[2,3,5])"), "23");
    // Inconsistent congruences with non-coprime moduli ⇒ honest residual.
    assert_eq!(r("crt([2,4],[3,6])"), "crt([[2], [4]], [[3], [6]])");
}

#[test]
fn test_eval_number_theory_gcdext() {
    // `gcdext(a,b)` (aliases `bezout`/`xgcd`) returns [g, x, y] with a·x + b·y = g = gcd(a,b) — the
    // Bézout coefficients from extended Euclid.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("gcdext(12,18)"), "[6, -1, 1]"); // 12·(-1) + 18·1 = 6
    assert_eq!(r("gcdext(3,7)"), "[1, -2, 1]"); // 3·(-2) + 7·1 = 1
    assert_eq!(r("gcdext(48,36)"), "[12, 1, -1]"); // 48·1 + 36·(-1) = 12
    assert_eq!(r("gcdext(17,5)"), "[1, -2, 7]");
}

#[test]
fn test_eval_number_theory_modular() {
    // Modular arithmetic: modinv (modular inverse via extended Euclid, residual when gcd≠1) and the
    // Jacobi symbol (−1/0/1). Cross-checked against sympy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("modinv(3,7)"), "5"); // 3·5 = 15 ≡ 1 (mod 7)
    assert_eq!(r("modinv(10,17)"), "12");
    assert_eq!(r("modinv(2,4)"), "modinv(2, 4)"); // gcd(2,4)=2 ⇒ no inverse
    assert_eq!(r("jacobi(2,7)"), "1"); // 2 is a QR mod 7
    assert_eq!(r("jacobi(3,7)"), "-1");
    assert_eq!(r("jacobi(2,15)"), "1");
    assert_eq!(r("jacobi(6,9)"), "0"); // gcd(6,9) ≠ 1
}

#[test]
fn test_eval_number_theory_divisor_functions() {
    // Divisor functions: τ/numdivisors (count), σ/sigma (sum), and iscomposite (1/0). All exact via
    // integer factorization. σ(6) = 12 = 2·6 confirms the perfect number.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("tau(12)"), "6");
    assert_eq!(r("numdivisors(12)"), "6");
    assert_eq!(r("tau(7)"), "2"); // prime ⇒ 2 divisors
    assert_eq!(r("sigma(12)"), "28");
    assert_eq!(r("sigma(6)"), "12"); // perfect number: σ(n) = 2n
    assert_eq!(r("sigma(7)"), "8"); // prime p ⇒ σ = p + 1
    assert_eq!(r("iscomposite(12)"), "1");
    assert_eq!(r("iscomposite(7)"), "0"); // prime
    assert_eq!(r("iscomposite(1)"), "0"); // neither prime nor composite
}

#[test]
fn test_eval_number_theory_primes_and_totient() {
    // New number-theory functions: isprime (1/0, the engine has no boolean), nextprime, prevprime,
    // and Euler's totient. All exact (BigInt trial division / factorization).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("isprime(7)"), "1");
    assert_eq!(r("isprime(12)"), "0");
    assert_eq!(r("isprime(1)"), "0");
    assert_eq!(r("isprime(2)"), "1");
    assert_eq!(r("nextprime(10)"), "11");
    assert_eq!(r("nextprime(13)"), "17");
    assert_eq!(r("prevprime(10)"), "7");
    assert_eq!(r("prevprime(3)"), "2");
    // No prime below 2 ⇒ honest residual.
    assert_eq!(r("prevprime(2)"), "prevprime(2)");
    assert_eq!(r("totient(12)"), "4");
    assert_eq!(r("totient(7)"), "6"); // prime: φ(p) = p−1
    assert_eq!(r("totient(36)"), "12");
    // Controls: existing number-theory calls unchanged.
    assert_eq!(r("gcd(48,36)"), "12");
    assert_eq!(r("prime_factors(12)"), "2^2·3");
}

#[test]
fn test_eval_combinatorial_sequences() {
    // Combinatorial integer sequences: Fibonacci (F₀=0, F₁=1), Lucas (L₀=2, L₁=1), and Catalan
    // (Cₙ = (2n)!/((n+1)!·n!)), all computed by exact BigInt iteration. Negative indices decline
    // to honest residuals (the closed forms here are defined for n ≥ 0).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("fibonacci(10)"), "55");
    assert_eq!(r("fib(20)"), "6765");
    assert_eq!(r("fibonacci(0)"), "0");
    assert_eq!(r("fibonacci(1)"), "1");
    assert_eq!(r("lucas(10)"), "123");
    assert_eq!(r("lucas(0)"), "2");
    assert_eq!(r("catalan(5)"), "42");
    assert_eq!(r("catalan(0)"), "1");
    assert_eq!(r("catalan(10)"), "16796");
    // Negative index ⇒ honest residual.
    assert_eq!(r("fibonacci(-1)"), "fibonacci(-1)");
    assert_eq!(r("catalan(-2)"), "catalan(-2)");
}

#[test]
fn test_eval_bernoulli_and_stirling_numbers() {
    // Bernoulli numbers Bₙ (rational, B₁=−1/2 convention) and Stirling numbers of the second
    // (set partitions into k blocks) and first (unsigned: permutations with k cycles) kind. All
    // exact (BigInt/BigRational recurrences). Negative / k>n cases give honest residuals or 0.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("bernoulli(0)"), "1");
    assert_eq!(r("bernoulli(1)"), "-1/2");
    assert_eq!(r("bernoulli(2)"), "1/6");
    assert_eq!(r("bernoulli(3)"), "0"); // odd n>1 ⇒ 0
    assert_eq!(r("bernoulli(4)"), "-1/30");
    assert_eq!(r("bernoulli(6)"), "1/42");
    assert_eq!(r("bernoulli(-1)"), "bernoulli(-1)"); // honest residual
    assert_eq!(r("stirling2(4,2)"), "7");
    assert_eq!(r("stirling2(5,3)"), "25");
    assert_eq!(r("stirling2(0,0)"), "1");
    assert_eq!(r("stirling2(2,5)"), "0"); // k>n ⇒ 0
    assert_eq!(r("stirling1(4,2)"), "11"); // unsigned: permutations of 4 with 2 cycles
    assert_eq!(r("stirling1(5,2)"), "50");
    assert_eq!(r("stirling1(3,3)"), "1");
}

#[test]
fn test_eval_vector_projection_and_angle() {
    // `proj(u,v)` = (⟨u,v⟩/⟨v,v⟩)·v (vector projection of u onto v, in v's shape) and
    // `angle(u,v)` = arccos(⟨u,v⟩/(‖u‖‖v‖)). Both require numeric vectors; the engine folds
    // arccos at the standard cosines, so nice vectors give clean closed forms. A zero direction
    // / zero vector, or symbolic / irrational-entry operands, decline to honest residuals.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Projection (returned as a column in v's shape).
    assert_eq!(r("proj([3,4],[1,0])"), "[[3], [0]]");
    assert_eq!(r("proj([3,4],[1,1])"), "[[7/2], [7/2]]");
    assert_eq!(r("proj([2,3,6],[1,2,2])"), "[[20/9], [40/9], [40/9]]");
    // Zero direction ⇒ honest residual (projection undefined).
    assert_eq!(r("proj([3,4],[0,0])"), "proj([[3], [4]], [[0], [0]])");
    // Angle: standard cosines fold to exact multiples of π.
    assert_eq!(r("angle([1,0],[0,1])"), "1/2·pi"); // perpendicular
    assert_eq!(r("angle([1,0],[1,0])"), "0"); // parallel
    assert_eq!(r("angle([1,0],[-1,0])"), "pi"); // antiparallel
    assert_eq!(r("angle([1,0],[1,1])"), "1/4·pi");
    assert_eq!(r("angle([3,4],[4,3])"), "arccos(24/25)"); // generic ⇒ exact arccos
                                                          // Zero vector ⇒ honest residual.
    assert_eq!(r("angle([0,0],[1,1])"), "angle([[0], [0]], [[1], [1]])");
}

#[test]
fn test_eval_convergent_p_series_even_zeta() {
    // `sum(c/k^p, k, 1, inf)` with EVEN p has Euler's closed form c·ζ(2m) = c·(rational)·π^(2m).
    // Odd p (ζ(3), ζ(5), …, no known closed form in π), the divergent harmonic series (p=1), and
    // any lower bound ≠ 1 MUST stay honest residuals — solving them would be unsound.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("sum(1/k^2, k, 1, inf)"), "1/6·pi^2"); // ζ(2) = π²/6
    assert_eq!(r("sum(1/k^4, k, 1, inf)"), "1/90·pi^4"); // ζ(4) = π⁴/90
    assert_eq!(r("sum(1/k^6, k, 1, inf)"), "1/945·pi^6"); // ζ(6) = π⁶/945
    assert_eq!(r("sum(1/k^8, k, 1, inf)"), "1/9450·pi^8"); // ζ(8) = π⁸/9450
    assert_eq!(r("sum(2/k^2, k, 1, inf)"), "1/3·pi^2"); // 2·ζ(2) = π²/3
    assert_eq!(r("sum(k^(-2), k, 1, inf)"), "1/6·pi^2"); // negative-power form
                                                         // Honest residuals: no elementary closed form, or out of scope.
    assert_eq!(r("sum(1/k^3, k, 1, inf)"), "sum(1 / k^3, k, 1, infinity)"); // Apéry, odd
    assert_eq!(r("sum(1/k^5, k, 1, inf)"), "sum(1 / k^5, k, 1, infinity)"); // odd
    assert_eq!(r("sum(1/k^2, k, 2, inf)"), "sum(1 / k^2, k, 2, infinity)"); // lower bound ≠ 1
}

#[test]
fn test_eval_chain_substitution_rational_scale() {
    // The transcendental chain-substitution gate accepts `F(g)` when `d/dx F(g) == k·integrand` for a
    // NONZERO RATIONAL `k` (was limited to `k = ±1`), returning `F(g)/k`. So `∫ x·e^(x²)` no longer
    // residualizes: `d/dx e^(x²) = 2x·e^(x²)`, scale ½. The answer stays soundness-gated by exact
    // differentiation, so every result differentiates back to the integrand.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("integrate(x*e^(x^2), x)"), "1/2·e^(x^2)");
    assert_eq!(r("integrate(x*cos(x^2), x)"), "1/2·sin(x^2)");
    assert_eq!(r("integrate(x*e^(2*x^2), x)"), "1/4·e^(2·x^2)");
    assert_eq!(r("integrate(5*cos(x)*e^(sin(x)), x)"), "5·e^sin(x)");
    // Unchanged `k = ±1` cases stay correct.
    assert_eq!(r("integrate(2*x*e^(x^2), x)"), "e^(x^2)");
    assert_eq!(r("integrate(cos(x)*e^(sin(x)), x)"), "e^sin(x)");
    assert_eq!(r("integrate(sin(x)*e^(cos(x)), x)"), "-(e^cos(x))");
}

#[test]
fn test_eval_improper_rational_integral_real_root_quadratic_denominator() {
    // An improper `∫_a^∞ p/q` with a `½·ln|p/q|` antiderivative and a quadratic denominator with
    // REAL roots OUTSIDE [a, ∞) used to decline: `nonzero_on_unbounded_interval` returned `Unknown`
    // for a degree-2 factor with non-negative discriminant (it only certified the no-real-root case).
    // Now it decides EXACTLY from the vertex `−b/2a` and the sign of `q` at the bound (no surds). The
    // boundary limit (already supported) supplies the value; tail divergence shows up as `±∞`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Convergent: roots ±1 / 0,1 / ±2 / −1,−2 all lie below the lower bound -> value computed.
    assert_eq!(r("integrate(1/(x^2-1), x, 2, oo)"), "-1/2·ln(1/3)"); // = ½ln3
    assert_eq!(r("integrate(2/(x^2-1), x, 2, oo)"), "-ln(1/3)"); // = ln3
    assert_eq!(r("integrate(1/(x^2-x), x, 2, oo)"), "-ln(1/2)"); // = ln2
    assert_eq!(r("integrate(1/(x^2+3*x+2), x, 1, oo)"), "-ln(2/3)"); // = ln(3/2)
    assert_eq!(r("integrate(1/(x^2-9), x, 4, oo)"), "-1/6·ln(1/7)"); // = (1/6)ln7
                                                                     // SOUNDNESS: tail-divergent and pole-in-range must NOT fabricate a finite value.
    assert_eq!(r("integrate(x/(x^2-1), x, 2, oo)"), "infinity"); // diverges (~1/x tail)
    assert_eq!(r("integrate(1/(x^2-1), x, 0, oo)"), "undefined"); // pole at x=1 ∈ [0,∞)
    assert_eq!(r("integrate(1/(x^2-1), x, 1/2, oo)"), "undefined"); // pole at x=1 ∈ [1/2,∞)
                                                                    // Unchanged: finite definite (already worked) and the no-real-root quadratic.
    assert_eq!(r("integrate(1/(x^2-1), x, 2, 5)"), "1/2·ln(2)");
    assert_eq!(r("integrate(1/(x^2+1), x, 0, oo)"), "1/2·pi");
}

#[test]
fn test_eval_improper_rational_integral_degree_n_denominator_divergence() {
    // The engine EXPANDS a denominator like `(x^2-1)(x^2-4)` into a single degree-4 polynomial, so the
    // factor-by-factor `Mul` walk never sees the quadratics. `nonzero_on_unbounded_interval` now splits
    // a degree-≥3 polynomial via its RATIONAL roots (`factor_rational_roots`) and certifies each factor,
    // so a pole strictly inside `[a, ∞)` is detected (`undefined`) and a `~1/x` tail diverges
    // (`infinity`) instead of a conservative residual. Removable singularities are pre-simplified by the
    // engine, so the cert never fabricates a divergence for a hole.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // SOUNDNESS: a pole at a rational root strictly inside the (unbounded) range -> divergent.
    assert_eq!(r("integrate(1/(x^3-x), x, 1/2, oo)"), "undefined"); // pole at x=1
    assert_eq!(r("integrate(1/(x^4-1), x, 0, oo)"), "undefined"); // pole at x=1
    assert_eq!(r("integrate(1/((x^2-1)*(x^2-4)), x, 0, oo)"), "undefined"); // poles at x=1,2
                                                                            // SOUNDNESS: a `~1/x` tail of a degree-n integrand diverges to +∞ (no fabricated finite value).
    assert_eq!(r("integrate(x^2/(x^3-x), x, 2, oo)"), "infinity");
    // A removable singularity is simplified away first, so its hole is NOT read as a pole.
    assert_eq!(r("integrate((x-1)/(x^3-x), x, 1/2, oo)"), "-ln(1/3)"); // = ln3, integrand 1/(x²+x)
}

#[test]
fn test_eval_log_sum_limit_at_infinity_and_convergent_degree_n_improper_integral() {
    // `lim_{x→∞} Σ cᵢ·ln(pᵢ(x))` with `Σ cᵢ·deg pᵢ = 0` is the finite `Σ cᵢ·ln(lead pᵢ)`, not the
    // `+∞−∞` residual the limit engine left for N≥3 terms (it only combined a two-term `ln p − ln q`).
    // `log_sum_limit_at_infinity` decides it from polynomial growth, which lets a partial-fraction log
    // antiderivative of an `∫_a^∞ p/q` with a degree-n denominator that splits into LINEAR factors over
    // ℚ resolve at the boundary. A leftover irreducible quadratic factor (an arctan term) still
    // declines — the next peldaño.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // The N-term sum-of-logs limit (degree sum 0 -> finite; nonzero -> ±∞).
    assert_eq!(r("limit(ln(x-1)+ln(x+1)-2*ln(x), x, infinity)"), "0");
    assert_eq!(r("limit(1/2*ln(x-1)+1/2*ln(x+1)-ln(x), x, infinity)"), "0");
    assert_eq!(r("limit(ln(2*x^2+1)-ln(x^2), x, infinity)"), "ln(2)");
    assert_eq!(r("limit(3*ln(x)-ln(x^3), x, infinity)"), "0");
    // Convergent improper integrals with a degree-n denominator factoring over ℚ into linears.
    // ∫_2^∞ 1/(x³−x) = ln2 − ½ln3; ∫_3^∞ 1/(x³−4x) = ⅛ln(9/5).
    assert_eq!(r("integrate(1/(x^3-x), x, 2, oo)"), "1/2·(2·ln(2) - ln(3))");
    assert_eq!(
        r("integrate(1/(x^3-4*x), x, 3, oo)"),
        "1/8·(2·ln(3) - ln(5))"
    );
    // Soundness preserved: the single bare log still diverges, and the `−∞` side stays residual.
    assert_eq!(r("limit(ln(x^2-1), x, infinity)"), "infinity");
}

#[test]
fn test_eval_arctan_plus_log_boundary_limit_and_irreducible_quadratic_improper_integral() {
    // A rational partial-fraction antiderivative with an irreducible quadratic factor mixes an
    // `arctan` term with the logs. When the `arctan` sits BETWEEN the logs in the Add tree the
    // additive fallback splits the logs individually into `+∞ − ∞` and stalls. `log_sum_limit_at_infinity`
    // now absorbs the arctan terms (`arctan(q) → sign(lead q)·π/2`) alongside the log block, so the
    // boundary limit resolves regardless of order and `∫_a^∞ 1/(x⁴−1)` computes.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // The mixed arctan+log boundary limit (arctan interleaved between the two logs) -> finite.
    assert_eq!(
        r("limit(1/4*ln(x-1)-1/2*arctan(x)-1/4*ln(x+1), x, infinity)"),
        "-1/4·pi"
    );
    // ∫_2^∞ 1/(x⁴−1): denominator (x²−1)(x²+1) -> two linear logs + an arctan. Numerically ≈ 0.04283.
    assert_eq!(
        r("integrate(1/(x^4-1), x, 2, oo)"),
        "1/4·(ln(3) + 2·arctan(2)) - 1/4·pi"
    );
    // Soundness preserved for the irreducible-quadratic family: pole in range -> divergent, ~1/x tail -> +∞.
    assert_eq!(r("integrate(1/(x^4-1), x, 0, oo)"), "undefined"); // pole at x=1
    assert_eq!(r("integrate(x^3/(x^4-1), x, 2, oo)"), "infinity"); // ~1/x tail
                                                                   // A PRE-FACTORED denominator with an irreducible quadratic factor: the antiderivative is
                                                                   // `Add(__hold(−½·arctan x − ¼·ln(x²+1)), ½·ln|x−1|)`; the surviving inner `__hold` used to block
                                                                   // the boundary limit. Stripping ALL holds first lets it fold. Numerically ≈ 0.170535673.
    assert_eq!(
        r("integrate(1/((x-1)*(x^2+1)), x, 2, oo)"),
        "1/4·(ln(5) + 2·arctan(2)) - 1/4·pi"
    );
    // The expanded-equivalent denominator computes to the SAME value.
    assert_eq!(
        r("integrate(1/(x^3-x^2+x-1), x, 2, oo)"),
        "1/4·(ln(5) + 2·arctan(2)) - 1/4·pi"
    );
    // Soundness preserved here too: pole at x=1 in range -> undefined; ~1/x tail -> +∞.
    assert_eq!(r("integrate(1/((x-1)*(x^2+1)), x, 0, oo)"), "undefined");
    assert_eq!(r("integrate(x^2/((x-1)*(x^2+1)), x, 2, oo)"), "infinity");
    // Edge: a lone arctan and a pure arctan pair are left to the unary/additive rules (unchanged).
    assert_eq!(r("limit(arctan(x), x, infinity)"), "pi / 2");
}

#[test]
fn test_eval_divergent_p_series_is_infinity() {
    // A divergent p-series `Σ c/n^p` with `0 < p ≤ 1` (the harmonic series and slower) now reports its
    // divergence as `±infinity` instead of a residual: every term eventually shares the sign of `c`.
    // The ζ-convergent `p > 1` cases, alternating series, and a sum that includes the `n = 0` pole are
    // unchanged.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Harmonic and slower (p ≤ 1) -> diverges with the sign of the coefficient.
    assert_eq!(r("sum(1/n, n, 1, oo)"), "infinity");
    assert_eq!(r("sum(2/n, n, 1, oo)"), "infinity");
    assert_eq!(r("sum(1/(2*n), n, 1, oo)"), "infinity");
    assert_eq!(r("sum(1/sqrt(n), n, 1, oo)"), "infinity"); // p = 1/2
    assert_eq!(r("sum(-1/n, n, 1, oo)"), "-infinity");
    assert_eq!(r("sum(1/n, n, 5, oo)"), "infinity"); // tail from any start ≥ 1 still diverges
                                                     // MUST NOT regress: p > 1 converges (ζ), alternating is conditionally convergent, n = 0 is a pole.
    assert_eq!(r("sum(1/n^2, n, 1, oo)"), "1/6·pi^2");
    assert_eq!(
        r("sum(1/n^(3/2), n, 1, oo)"),
        "sum(1 / n^(3/2), n, 1, infinity)"
    ); // ζ(3/2), no closed form
    assert_eq!(r("sum(1/n^3, n, 1, oo)"), "sum(1 / n^3, n, 1, infinity)"); // ζ(3), deliberate residual
    assert_eq!(
        r("sum((-1)^n/n, n, 1, oo)"),
        "sum((-1)^n / n, n, 1, infinity)"
    ); // alternating
    assert_eq!(r("sum(1/n, n, 0, oo)"), "undefined"); // n = 0 pole in range
}

#[test]
fn test_eval_summation_pole_in_range_is_undefined() {
    // A finite or infinite sum whose summand has a POLE (a `1/0` term) at an integer in the range is
    // UNDEFINED — the telescoping/closed-form builders otherwise compute THROUGH it. The pole
    // detector folds `n^k` exactly (`as_rational_const` declines `Pow`, so a quadratic denominator's
    // root went undetected); ALL roots are checked, incl. the NEGATIVE one and the start itself.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Poles inside the range -> undefined (finite and infinite upper bound).
    assert_eq!(r("sum(1/(n^2-1), n, -2, 5)"), "undefined"); // poles at n=±1
    assert_eq!(r("sum(1/(n^2-4), n, -2, 5)"), "undefined"); // poles at n=±2 (n=-2 is the start)
    assert_eq!(r("sum(1/(n^2-1), n, -1, 5)"), "undefined"); // n=-1 start is itself a pole
    assert_eq!(r("sum(1/(n^2-1), n, 1, 5)"), "undefined");
    assert_eq!(r("sum(1/(n^2-1), n, -2, inf)"), "undefined"); // telescoped to -5/12 before
    assert_eq!(r("sum(1/(n^2-4), n, -2, inf)"), "undefined");
    // No pole in the range -> the exact value is unchanged.
    assert_eq!(r("sum(1/(n^2-1), n, 2, 5)"), "17/30");
    assert_eq!(r("sum(1/(n^2-1), n, -3, -2)"), "11/24");
    assert_eq!(r("sum(1/(n^2-1), n, 2, inf)"), "3/4");
    assert_eq!(r("sum(1/2^k, k, 0, inf)"), "2");
    assert_eq!(r("sum(k, k, -2, 5)"), "12");
}

#[test]
fn test_eval_arclength_curve() {
    // `arclength(f, x, a, b)` = ∫ₐᵇ √(1 + (df/dx)²) dx, rewritten to the definite integral and
    // evaluated by the integration engine: a clean closed form when the integrand is elementary,
    // an honest residual integral otherwise (catenary, elliptic, x³, …).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("arclength(2*x+1, x, 0, 3)"), "3·sqrt(5)"); // straight line
    assert_eq!(r("arclength(x, x, 0, 5)"), "5·sqrt(2)"); // diagonal
    assert_eq!(r("arclength(3, x, 0, 4)"), "4"); // flat line, length = b − a
    assert_eq!(r("arclength(x^2, x, 0, 1)"), "1/4·asinh(2) + 1/2·sqrt(5)"); // parabola
    assert_eq!(r("arclength(x^(3/2), x, 0, 1)"), "13/27·sqrt(13) - 8/27"); // power curve
    assert_eq!(r("arc_length(x^2, x, 0, 1)"), "1/4·asinh(2) + 1/2·sqrt(5)"); // alias
                                                                             // Honest residual integrals when the integrand is not elementary.
    assert_eq!(
        r("arclength(x^3, x, 0, 1)"),
        "integrate((9·x^4 + 1)^(1/2), x, 0, 1)"
    );
}

#[test]
fn test_eval_reciprocal_positive_function_inequality_flips() {
    // SOUNDNESS: `c/f(x) OP k` with a provably-positive function denominator (abs, …) and k > 0 must
    // FLIP when isolating the denominator: `c/f > k ⟺ f < c/k`. Previously the engine kept the
    // direction, returning the COMPLEMENT (`1/abs(x)>2 → (-∞,-1/2)∪(1/2,∞)`). The denominator pole is
    // conveyed via the `x ≠ ...` required condition (so the interval ∩ condition is the true set).
    let run = |input: &str| -> (String, Vec<String>) {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        let conds = wire["required_display"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        (wire["result"].as_str().unwrap_or("").to_string(), conds)
    };
    // 1/abs(x) > 2 ⟺ 0 < abs(x) < 1/2 — the SET itself now punctures the pole
    // (scout family C: "(-1/2, 1/2)" asserted x=0 in the set and relied on the
    // reader combining it with the side condition).
    assert_eq!(
        run("solve(1/abs(x)>2, x)"),
        ("(-1/2, 0) U (0, 1/2)".into(), vec!["x ≠ 0".into()])
    );
    assert_eq!(
        run("solve(2/abs(x)>1, x)"),
        ("(-2, 0) U (0, 2)".into(), vec!["x ≠ 0".into()])
    );
    assert_eq!(
        run("solve(1/abs(x-1)>2, x)"),
        ("(1/2, 1) U (1, 3/2)".into(), vec!["x ≠ 1".into()])
    );
    // The `<` direction is unchanged (it was already the larger side): abs(x) > 1/2.
    assert_eq!(
        run("solve(1/abs(x)<2, x)").0,
        "(-infinity, -1/2) U (1/2, infinity)"
    );
    // Controls: bare-variable reciprocal (sign-split path) and equality are unchanged.
    assert_eq!(run("solve(1/x>2, x)").0, "(0, 1/2)");
    assert_eq!(run("solve(1/x<2, x)").0, "(-infinity, 0) U (1/2, infinity)");
}

#[test]
fn test_eval_const_over_abs_denominator_vs_zero_reduces_to_sign() {
    // SOUNDNESS: `c/g {op} 0` with an abs INSIDE the denominator (`1/(|x|-1) < 0`)
    // fell to the generic rational path, which cannot find g's zeros through the
    // abs and returned garbage (`< 0 → ℝ`; `> 0 → (-∞,-∞)∪(∞,∞)`). Since `c/g` is
    // never 0 and shares g's sign, `c/g {op} 0 ⟺ g {op'} 0` with a STRICT op' (the
    // pole g=0 is undefined, not 0, so `≤/≥` collapse to `</>`). Delegate to the
    // abs solver.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `< 0 ⟺ |x|-1 < 0 ⟺ |x| < 1`.
    assert_eq!(r("solve(1/(abs(x) - 1) < 0, x)"), "(-1, 1)");
    // `> 0 ⟺ |x|-1 > 0 ⟺ |x| > 1`.
    assert_eq!(
        r("solve(1/(abs(x) - 1) > 0, x)"),
        "(-infinity, -1) U (1, infinity)"
    );
    assert_eq!(
        r("solve(1/(abs(x) - 2) > 0, x)"),
        "(-infinity, -2) U (2, infinity)"
    );
    // Always-positive denominator: the reduction gives an always-true `|x|+1 > 0`.
    assert_eq!(r("solve(1/(abs(x) + 1) > 0, x)"), "All real numbers");
    // Shifted abs argument and a non-unit numerator constant.
    assert_eq!(
        r("solve(5/(abs(x-3) - 1) > 0, x)"),
        "(-infinity, 2) U (4, infinity)"
    );
    // Coefficiented abs argument.
    assert_eq!(
        r("solve(1/(abs(2*x) - 1) > 0, x)"),
        "(-infinity, -1/2) U (1/2, infinity)"
    );
    // Non-strict operators keep the pole OPEN (the value is undefined at g=0, not 0).
    assert_eq!(r("solve(1/(abs(x) - 1) <= 0, x)"), "(-1, 1)");
    assert_eq!(
        r("solve(1/(abs(x) - 1) >= 0, x)"),
        "(-infinity, -1) U (1, infinity)"
    );
    // Negative numerator flips the reduced sign: `-1/(|x|-1) < 0 ⟺ |x|-1 > 0`.
    assert_eq!(
        r("solve(-1/(abs(x) - 1) < 0, x)"),
        "(-infinity, -1) U (1, infinity)"
    );
    assert_eq!(r("solve(3/(abs(x) - 2) < 0, x)"), "(-2, 2)");

    // The reduction needs only the numerator's SIGN, decided exactly via the shared
    // const-sign chokepoint — so a surd (`√2`) or transcendental (`e−3`, `π`)
    // numerator works too, not just a rational.
    assert_eq!(
        r("solve(sqrt(2)/(abs(x) - 1) > 0, x)"),
        "(-infinity, -1) U (1, infinity)"
    );
    assert_eq!(r("solve(-sqrt(2)/(abs(x) - 1) > 0, x)"), "(-1, 1)");
    // `e − 3 < 0` flips the reduced sign; `π > 0` keeps it.
    assert_eq!(r("solve((e-3)/(abs(x) - 1) > 0, x)"), "(-1, 1)");
    assert_eq!(r("solve(pi/(abs(x) - 2) < 0, x)"), "(-2, 2)");

    // NO REGRESSION: non-abs reciprocal denominators, the bare `A/|g| {op} c` forms
    // (c ≠ 0), and equations keep their existing owners.
    assert_eq!(r("solve(1/(x - 1) < 0, x)"), "(-infinity, 1)");
    assert_eq!(r("solve(5/(x - 3) > 0, x)"), "(3, infinity)");
    assert_eq!(r("solve(1/abs(x) > 2, x)"), "(-1/2, 0) U (0, 1/2)");
}

#[test]
fn test_eval_periodic_trig_inequality_declines() {
    // SOUNDNESS: a periodic `sin`/`cos`/`tan` inequality has an infinite periodic-union solution
    // that the monotonic inversion used to emit as a single wrong ray. Since cycle P2 the sin/cos
    // interior cases SOLVE exactly via PeriodicIntervalUnion; tan still declines honestly (P3).
    // The bare out-of-range cases (ℝ/∅) and equations are unaffected.
    let run = |input: &str| -> (bool, String) {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        (
            wire["ok"].as_bool().unwrap_or(false),
            wire["result"].as_str().unwrap_or("").to_string(),
        )
    };
    for (input, expected) in [
        ("sin(x)>0", "{ (k·2·pi, pi + k·2·pi) : k ∈ ℤ }"),
        ("cos(x)<0", "{ (1/2·pi + k·2·pi, 3/2·pi + k·2·pi) : k ∈ ℤ }"),
        (
            "sin(x)>1/2",
            "{ (1/6·pi + k·2·pi, 5/6·pi + k·2·pi) : k ∈ ℤ }",
        ),
        ("tan(x)>1", "{ (1/4·pi + k·pi, 1/2·pi + k·pi) : k ∈ ℤ }"),
        ("sin(2*x)>0", "{ (k·pi, 1/2·pi + k·pi) : k ∈ ℤ }"),
        (
            "cos(x)>=1/2",
            "{ [-1/3·pi + k·2·pi, 1/3·pi + k·2·pi] : k ∈ ℤ }",
        ),
    ] {
        let (ok, result) = run(input);
        assert!(ok, "{input} should be ok=true, got {result:?}");
        assert_eq!(result, expected, "{input}");
    }
    let plain = |input: &str| run(input).1;
    // Out-of-range bare sin/cos are still answered exactly (not pre-empted by the residual decline).
    assert_eq!(plain("cos(x)<=1"), "All real numbers");
    assert_eq!(plain("sin(x)>2"), "No solution");
    // Equations and constant-trig (variable is linear) still solve (two-family periodic set).
    assert_eq!(
        plain("sin(x)=1/2"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(plain("sin(2)*x>0"), "(0, infinity)");
}

#[test]
fn test_eval_periodic_trig_equation_emits_family() {
    // A bare `sin/cos/tan(x)=c` equation has an INFINITE periodic family; the unary-inverse path
    // returned only the principal root (`solve(tan(x)=1)→{π/4}`, dropping `+kπ`). Emit the whole
    // family via the `Periodic` SolutionSet. tan is period π for every c; sin/cos collapse to a
    // single family only for c ∈ {0,±1} (other c are two families → decline, unchanged).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Period π families.
    assert_eq!(r("solve(sin(x)=0, x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(tan(x)=0, x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x)=0, x)"), "{ 1/2·pi + k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(tan(x)=1, x)"), "{ 1/4·pi + k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(tan(x)=sqrt(3), x)"), "{ 1/3·pi + k·pi : k ∈ ℤ }");
    // tan is complete even for a symbolic threshold.
    assert_eq!(r("solve(tan(x)=2, x)"), "{ arctan(2) + k·pi : k ∈ ℤ }");
    // Scaled argument `trig(a·x)=c`: divide base and period by `a` (a>1 shrinks the period).
    assert_eq!(r("solve(cos(2*x)=1, x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(2*x)=0, x)"), "{ k·1/2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(tan(2*x)=1, x)"), "{ 1/8·pi + k·1/2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(x/2)=0, x)"), "{ k·2·pi : k ∈ ℤ }");
    // Squared trig via the double-angle reduction `sin(arg)^2=c <=> cos(2·arg)=1-2c`.
    assert_eq!(r("solve(sin(x)^2=1, x)"), "{ 1/2·pi + k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x)^2=1, x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(x)^2=1/2, x)"), "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(2*x)^2=1, x)"), "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }");
    // sin(x)^2=1/4 -> cos(2x)=1/2 -> the TWO families {π/6+kπ, 5π/6+kπ}.
    assert_eq!(
        r("solve(sin(x)^2=1/4, x)"),
        "{ 1/6·pi + k·pi, 5/6·pi + k·pi : k ∈ ℤ }"
    );
    // Period 2π families (c = ±1, the two roots coincide -> one family).
    assert_eq!(r("solve(sin(x)=1, x)"), "{ 1/2·pi + k·2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x)=1, x)"), "{ k·2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x)=-1, x)"), "{ pi + k·2·pi : k ∈ ℤ }");
    // Two-family `sin/cos=c` (0 < |c| < 1): BOTH principal roots, shared period 2π.
    assert_eq!(
        r("solve(sin(x)=1/2, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(cos(x)=1/2, x)"),
        "{ 1/3·pi + k·2·pi, 5/3·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(x)=-1/2, x)"),
        "{ -1/6·pi + k·2·pi, 7/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(x)=1/3, x)"),
        "{ arcsin(1/3) + k·2·pi, pi - arcsin(1/3) + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(sin(x)=2, x)"), "No solution"); // |c|>1
                                                        // A SURD constant in the `= 0` form (`2·cos(x) − √3 = 0`) used to collapse to the principal root
                                                        // `{π/6}`, dropping the periodic family AND the second base root — the `A·trig + B` normalization
                                                        // required a RATIONAL offset `B`, so a surd `B` fell through to the principal-inverse isolation. The
                                                        // offset is now kept symbolically, so the `= 0` form matches the trusted direct-RHS form.
    assert_eq!(
        r("solve(2*cos(x)-sqrt(3)=0, x)"),
        "{ 1/6·pi + k·2·pi, 11/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(x)-sqrt(3)/2=0, x)"),
        "{ 1/3·pi + k·2·pi, 2/3·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(tan(x)-sqrt(3)=0, x)"), "{ 1/3·pi + k·pi : k ∈ ℤ }");
    assert_eq!(
        r("solve(cos(x)+sqrt(2)/2=0, x)"),
        "{ 3/4·pi + k·2·pi, 5/4·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(2*x)-sqrt(3)/2=0, x)"),
        "{ 1/6·pi + k·pi, 1/3·pi + k·pi : k ∈ ℤ }"
    );
}

#[test]
fn test_eval_pi_shifted_argument_trig_keeps_periodic_family() {
    // `trig(a·x + b) = c` with `b` a π-multiple additive shift: the simplifier expands the
    // angle-addition (`sin(x + π/4) → (√2/2)·(sin x + cos x)`), and the isolation then returned only the
    // PRINCIPAL root (`sin(x + π/4) = 1/2 → {−π/12}`, dropping both the `+2kπ` family and the second
    // branch). Now `trig(u) = c` is solved for `u = a·x + b` and mapped back through `x = (u − b)/a`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Both branches, full 2π period.
    assert_eq!(
        r("solve(sin(x + pi/4) = 1/2, x)"),
        "{ -1/12·pi + k·2·pi, 7/12·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(cos(x + pi/3) = 1/2, x)"),
        "{ k·2·pi, 4/3·pi + k·2·pi : k ∈ ℤ }"
    );
    // A coefficient AND a π-shift: base and period both scale by `1/a`.
    assert_eq!(
        r("solve(sin(2*x + pi/4) = 1/2, x)"),
        "{ -1/24·pi + k·pi, 7/24·pi + k·pi : k ∈ ℤ }"
    );
    // Single-family `c ∈ {0, ±1}` cases, and tan (period π).
    assert_eq!(
        r("solve(cos(x - pi/6) = 0, x)"),
        "{ 2/3·pi + k·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(tan(x + pi/4) = 1, x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(x - pi/2) = 1, x)"), "{ pi + k·2·pi : k ∈ ℤ }");
    // Out of range stays unsatisfiable.
    assert_eq!(r("solve(sin(x + pi/4) = 2, x)"), "No solution");
    // A SYMBOLIC (non-π) shift — `arctan`, surd — is mishandled the same way and now also keeps the
    // full family (the auxiliary-angle dispatch target `sin(x + arctan(b/a)) = c` relies on this).
    assert_eq!(
        r("solve(sin(x + arctan(4/3)) = 1, x)"),
        "{ 1/2·pi - arctan(4/3) + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(x + sqrt(2)) = 1/2, x)"),
        "{ 1/6·pi - sqrt(2) + k·2·pi, 5/6·pi - sqrt(2) + k·2·pi : k ∈ ℤ }"
    );
    // Controls: a PLAIN-rational additive shift and the bare/coefficient forms are handled by the
    // existing periodic path and must be UNCHANGED (this handler declines — it gates on a symbolic shift).
    assert_eq!(
        r("solve(sin(x + 1) = 1/2, x)"),
        "{ 1/6·(pi - 6) + k·2·pi, 5/6·pi - 1 + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(2*x) = 1/2, x)"),
        "{ 1/12·pi + k·pi, 5/12·pi + k·pi : k ∈ ℤ }"
    );
}

#[test]
fn test_eval_periodic_trig_product_equation_unions_families() {
    // A PRODUCT of periodic trig factors (or a `cos(a)±cos(b)` / `sin(a)±sin(b)` that reduces to one
    // via sum-to-product) used to drop periodicity: each factor was solved for its PRINCIPAL root
    // only and the roots unioned into a wrong finite set (`solve(cos(2x)-cos(x))→{0}`). Now every
    // factor yields its full `Periodic` family and the families are unioned over a common period.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Explicit products, equal period: the sin·cos product now reduces through
    // the double angle (`sin·cos = 0 ⇔ sin(2x) = 0`), yielding the SAME set in
    // its compact single-family form: {kπ} ∪ {π/2+kπ} ≡ {kπ/2} (k even ↦ kπ,
    // k odd ↦ π/2+(k−1)π/2·2... i.e. exact set equality).
    assert_eq!(r("solve(sin(x)*cos(x)=0, x)"), "{ k·1/2·pi : k ∈ ℤ }");
    assert_eq!(
        r("solve((2*cos(x)+1)*(cos(x)-1)=0, x)"),
        "{ 2/3·pi + k·2·pi, 4/3·pi + k·2·pi, k·2·pi : k ∈ ℤ }"
    );
    // Mixed periods (π and 2π): expand to the common period 2π, then union.
    assert_eq!(
        r("solve(sin(x)*(2*cos(x)-1)=0, x)"),
        "{ k·2·pi, pi + k·2·pi, 1/3·pi + k·2·pi, 5/3·pi + k·2·pi : k ∈ ℤ }"
    );
    // `cos(2x) − cos(x)` simplifies (in the solve context) to the single-atom polynomial
    // `2·cos(x)² − cos(x) − 1`, so the double-angle poly-in-`cos` path solves it (`cos ∈ {1, −1/2}`);
    // the family order reflects that path (the same complete set as sum-to-product).
    assert_eq!(
        r("solve(cos(2*x)-cos(x), x)"),
        "{ 2/3·pi + k·2·pi, 4/3·pi + k·2·pi, k·2·pi : k ∈ ℤ }"
    );
    // `sin(2x) − sin(x) = sin(x)·(2·cos(x) − 1)` stays on the sum-to-product / product path.
    assert_eq!(
        r("solve(sin(2*x)-sin(x), x)"),
        // Factor-wise family union (post-2026-07-13): same set, `k·2π` base last.
        "{ 1/3·pi + k·2·pi, pi + k·2·pi, 5/3·pi + k·2·pi, k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(cos(2*x)+cos(x), x)"),
        "{ pi + k·2·pi, 1/3·pi + k·2·pi, 5/3·pi + k·2·pi : k ∈ ℤ }"
    );
    // SOUNDNESS: a product mixing a trig factor with a non-periodic factor cannot be one periodic
    // set; it must stay an honest residual rather than emit a half-solved/wrong set.
    assert_eq!(r("solve((x-1)*sin(x)=0, x)"), "Solve: sin(x)·(x - 1) = 0");
    // Non-trig products are unaffected.
    assert_eq!(r("solve((x-1)*(x-2)=0, x)"), "{ 1, 2 }");
}

#[test]
fn test_eval_quadratic_in_trig_equation_unions_periodic_roots() {
    // A polynomial of degree ≥ 2 in a single trig atom (`2·sin(x)² − 3·sin(x) + 1 = 0`, NOT a perfect
    // square, so the squared-trig reduction misses it) leaked an `arcsin(… − cos(2x) …)` residual once
    // the double-angle identity fired. Substitute `u = sin(x)`, solve `P(u) = 0`, back-substitute each
    // root through the periodic solver (range guard drops `|u| > 1`), and union the families over a
    // common period — `union_solution_sets` drops a `Periodic ∪ Periodic`, so the handler combines them.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `(2sin-1)(sin-1)=0`: BOTH families kept (`sin = 1/2` and `sin = 1`).
    assert_eq!(
        r("solve(2*sin(x)^2 - 3*sin(x) + 1 = 0, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi, 1/2·pi + k·2·pi : k ∈ ℤ }"
    );
    // `(2cos-1)(cos+1)=0`.
    assert_eq!(
        r("solve(2*cos(x)^2 + cos(x) - 1 = 0, x)"),
        "{ pi + k·2·pi, 1/3·pi + k·2·pi, 5/3·pi + k·2·pi : k ∈ ℤ }"
    );
    // Mixed periods: `sin = 0` (period π) and `sin = 1` (period 2π) combine over 2π.
    assert_eq!(
        r("solve(sin(x)^2 - sin(x) = 0, x)"),
        "{ k·2·pi, pi + k·2·pi, 1/2·pi + k·2·pi : k ∈ ℤ }"
    );
    // SOUNDNESS: a root outside `[-1, 1]` is dropped (`cos = 2` has no angle).
    assert_eq!(
        r("solve(cos(x)^2 - cos(x) - 2 = 0, x)"),
        "{ pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(2*sin(x)^2 + 5*sin(x) + 2 = 0, x)"),
        "{ -1/6·pi + k·2·pi, 7/6·pi + k·2·pi : k ∈ ℤ }"
    );
    // Controls: a pure square stays with the squared-trig reduction (compact form); a single trig and a
    // Pythagorean mix (two distinct atoms) are unchanged.
    assert_eq!(
        r("solve(2*sin(x)^2 - 1 = 0, x)"),
        "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(x) = 1/2, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    // A MIXED `sin²(x) + cos(x)` now reduces via the Pythagorean identity (`sin² = 1 − cos²`) to a
    // polynomial in `cos(x)` and solves (`cos(x)·(1 − cos(x)) = 0 ⟹ cos(x) ∈ {0, 1}`).
    assert_eq!(
        r("solve(sin(x)^2 + cos(x) = 1, x)"),
        "{ 1/2·pi + k·2·pi, 3/2·pi + k·2·pi, k·2·pi : k ∈ ℤ }"
    );
}

#[test]
fn test_eval_double_angle_and_mixed_trig_reduce_to_single_atom() {
    // A double-angle `cos(2x)` folds (via the simplifier) to `2·cos(x)² − 1`; when the rest is a
    // polynomial in `cos(x)` the equation becomes a single-atom quadratic. When it mixes `sin` and
    // `cos` (e.g. `cos(2x) − sin(x) → 2·cos(x)² − sin(x) − 1`) the Pythagorean identity eliminates the
    // all-even atom. Both were `arccos(…)` / `arcsin(…)` residuals before.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `cos(2x) + 3cos(x) + 2 = 0 ⟹ 2cos² + 3cos + 1 = 0 ⟹ cos ∈ {−1, −1/2}`.
    assert_eq!(
        r("solve(cos(2*x) + 3*cos(x) + 2 = 0, x)"),
        "{ pi + k·2·pi, 2/3·pi + k·2·pi, 4/3·pi + k·2·pi : k ∈ ℤ }"
    );
    // Mixed via Pythagorean: `cos(2x) = sin(x) ⟹ 2cos² − sin − 1 ⟹ −2sin² − sin + 1 = 0 ⟹
    // sin ∈ {1/2, −1}`. The `A = B` form and the pre-expanded form agree.
    assert_eq!(
        r("solve(cos(2*x) = sin(x), x)"),
        "{ -1/2·pi + k·2·pi, 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(2*cos(x)^2 - sin(x) - 1 = 0, x)"),
        "{ -1/2·pi + k·2·pi, 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    // `2sin²(x) + 3cos(x) − 3 = 0 ⟹ 2cos² − 3cos + 1 = 0 ⟹ cos ∈ {1, 1/2}`.
    assert_eq!(
        r("solve(2*sin(x)^2 + 3*cos(x) - 3 = 0, x)"),
        "{ 1/3·pi + k·2·pi, 5/3·pi + k·2·pi, k·2·pi : k ∈ ℤ }"
    );
    // Controls: a pure single-atom quadratic and a two-term `cos(2x) + cos(x)` (solved as a PRODUCT via
    // sum-to-product) are unchanged.
    assert_eq!(
        r("solve(2*sin(x)^2 - 3*sin(x) + 1 = 0, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi, 1/2·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(cos(2*x) + cos(x) = 0, x)"),
        "{ pi + k·2·pi, 1/3·pi + k·2·pi, 5/3·pi + k·2·pi : k ∈ ℤ }"
    );
}

#[test]
fn test_eval_homogeneous_linear_trig_equation_reduces_to_tangent() {
    // A HOMOGENEOUS linear trig equation `a·sin(g) + b·cos(g) = 0` (same argument `g`, `a ≠ 0`) reduces
    // to `tan(g) = −b/a` — dividing by `cos(g)` loses nothing since `cos(g) = 0` is never a solution when
    // `a ≠ 0`. The isolation path otherwise leaks an `arcsin(cos(x)·…)` residual.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `sin = cos ⟹ tan = 1`, `sin + cos = 0 ⟹ tan = −1` (period π, one family).
    assert_eq!(r("solve(sin(x) = cos(x), x)"), "{ 1/4·pi + k·pi : k ∈ ℤ }");
    assert_eq!(
        r("solve(sin(x) + cos(x) = 0, x)"),
        "{ -1/4·pi + k·pi : k ∈ ℤ }"
    );
    // Irrational coefficient: `√3·sin − cos = 0 ⟹ tan = 1/√3 ⟹ π/6`.
    assert_eq!(
        r("solve(sqrt(3)*sin(x) - cos(x) = 0, x)"),
        "{ 1/6·pi + k·pi : k ∈ ℤ }"
    );
    // A non-notable ratio keeps the exact `arctan`.
    assert_eq!(
        r("solve(2*sin(x) - 3*cos(x) = 0, x)"),
        "{ arctan(3/2) + k·pi : k ∈ ℤ }"
    );
    // Affine argument: `sin(2x) = cos(2x) ⟹ tan(2x) = 1 ⟹ π/8 + kπ/2`.
    assert_eq!(
        r("solve(sin(2*x) = cos(2*x), x)"),
        "{ 1/8·pi + k·1/2·pi : k ∈ ℤ }"
    );
    // Controls: bare `sin/cos = 0` (owned by the periodic handler) and a product (not a sum) are
    // unchanged. (The inhomogeneous `… = c` is now solved by the auxiliary-angle handler — see below.)
    assert_eq!(r("solve(sin(x) = 0, x)"), "{ k·pi : k ∈ ℤ }");
    // (Same compact-form recontract as the product-union test above:
    // sin·cos = 0 reduces via sin(2x) = 0 to the equivalent {kπ/2}.)
    assert_eq!(r("solve(sin(x)*cos(x) = 0, x)"), "{ k·1/2·pi : k ∈ ℤ }");
}

#[test]
fn test_eval_inhomogeneous_linear_trig_uses_auxiliary_angle() {
    // `a·sin(g) + b·cos(g) = c` (`c ≠ 0`) reduces by the auxiliary angle to
    // `sin(g + arctan(b/a)) = c/√(a²+b²)` (normalizing `a > 0`), dispatched to the shifted-argument
    // solver. It was an `arcsin(… − cos(x) …)` residual before.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `c/R = 1` (tangent): a single family. `3·sin + 4·cos = 5 ⟹ sin(x + arctan(4/3)) = 1`.
    assert_eq!(
        r("solve(3*sin(x) + 4*cos(x) = 5, x)"),
        "{ 1/2·pi - arctan(4/3) + k·2·pi : k ∈ ℤ }"
    );
    // `c/R < 1` (notable): `sin + cos = 1 ⟹ sin(x + π/4) = 1/√2 ⟹ {2kπ, π/2 + 2kπ}`.
    assert_eq!(
        r("solve(sin(x) + cos(x) = 1, x)"),
        "{ k·2·pi, 1/2·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(x) - cos(x) = 1, x)"),
        "{ 1/2·pi + k·2·pi, pi + k·2·pi : k ∈ ℤ }"
    );
    // SOUNDNESS: `|c| > R ⟹ |c/R| > 1 ⟹` No solution (the surd range guard).
    assert_eq!(r("solve(3*sin(x) + 4*cos(x) = 6, x)"), "No solution");
    assert_eq!(r("solve(3*sin(x) + 4*cos(x) = 10, x)"), "No solution");
    // Irrational (provable-sign surd) coefficients: `sin + √3·cos = 1 ⟹ R = 2, φ = arctan(√3) = π/3`.
    assert_eq!(
        r("solve(sin(x) + sqrt(3)*cos(x) = 1, x)"),
        "{ -1/6·pi + k·2·pi, 1/2·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sqrt(3)*sin(x) + cos(x) = 1, x)"),
        "{ k·2·pi, 2/3·pi + k·2·pi : k ∈ ℤ }"
    );
    // A COMPOUND coefficient `2·√2` (rational × surd): `classify_linear_trig_leaf` now multiplies the
    // outer factor by the inner coefficient (it used to discard the `√2`). `R = √(1+8) = 3`.
    assert_eq!(
        r("solve(sin(x) + 2*sqrt(2)*cos(x) = 3, x)"),
        "{ 1/2·pi - arctan(4·2^(-1/2)) + k·2·pi : k ∈ ℤ }"
    );
    // Controls: the homogeneous `c = 0` is the tangent reduction (and its compound-coefficient case is
    // now correct too, thanks to the same `classify_linear_trig_leaf` fix).
    assert_eq!(r("solve(sin(x) = cos(x), x)"), "{ 1/4·pi + k·pi : k ∈ ℤ }");
    assert_eq!(
        r("solve(2*sqrt(2)*sin(x) - cos(x) = 0, x)"),
        "{ arctan(2^(-1/2) / 2) + k·pi : k ∈ ℤ }"
    );
}

#[test]
fn test_eval_abs_of_trig_equation_keeps_periodicity() {
    // `|A| = c` with a trig-bearing argument was solved to PRINCIPAL roots by the generic abs isolation
    // (`|2·sin(x)−1| = 1 → {π/2, 0}`). It now splits into `A = c ∨ A = −c`, solving each branch fully so
    // trig stays periodic, then unions the families (over a common period when they differ).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `sin = 1` (period 2π) ∪ `sin = 0` (period π) — combined over 2π.
    assert_eq!(
        r("solve(abs(2*sin(x) - 1) = 1, x)"),
        "{ 1/2·pi + k·2·pi, k·2·pi, pi + k·2·pi : k ∈ ℤ }"
    );
    // `tan = ±1`, period π.
    assert_eq!(
        r("solve(abs(tan(x)) = 1, x)"),
        "{ 1/4·pi + k·pi, -1/4·pi + k·pi : k ∈ ℤ }"
    );
    // Both branches' non-principal `π − arcsin` roots are kept.
    assert_eq!(
        r("solve(abs(sin(x) - 1/2) = 1/4, x)"),
        "{ arcsin(3/4) + k·2·pi, pi - arcsin(3/4) + k·2·pi, arcsin(1/4) + k·2·pi, pi - arcsin(1/4) + k·2·pi : k ∈ ℤ }"
    );
    // One branch is out of range (`cos = 2`) and contributes nothing.
    assert_eq!(
        r("solve(abs(cos(x) - 1) = 1, x)"),
        "{ 1/2·pi + k·pi : k ∈ ℤ }"
    );
    // `c = 0` is a single branch.
    assert_eq!(
        r("solve(abs(2*sin(x) - 1) = 0, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    // Both branches out of range ⇒ empty.
    assert_eq!(r("solve(abs(2*sin(x) - 1) = 5, x)"), "No solution");
    // Controls: bare `|trig| = c` keeps the periodic-trig reduction's form; non-trig `|A|` and a
    // negative RHS are unchanged.
    assert_eq!(
        r("solve(abs(sin(x)) = 1/2, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi, -1/6·pi + k·2·pi, 7/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(abs(x - 1) = 2, x)"), "{ 3, -1 }");
    assert_eq!(r("solve(abs(sin(x)) = -1, x)"), "No solution");
}

#[test]
fn test_eval_trig_power_equation_keeps_periodicity() {
    // A trig EXPRESSION that simplifies to a perfect square / odd power of a single trig
    // (`cos(x)^2-1 -> -sin(x)^2`, `sin(x)*tan(x) -> sin^2/cos`, `(cos+1)(cos-1)sin -> -sin^3`)
    // collapsed to a single (often duplicated) finite root because the squared-trig reduction only
    // saw a bare `trig^2 = c` with the constant on the OTHER side and `n = 2`. Peeling a leading
    // coefficient/`Neg` and reducing `trig(arg)^n = 0` to `trig(arg) = 0` (with a complementary-
    // denominator guard for the quotient form) restores the full periodic family.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Squared / Neg / conjugate-factor forms.
    assert_eq!(r("solve(cos(x)^2-1, x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(x)^2-1, x)"), "{ 1/2·pi + k·pi : k ∈ ℤ }");
    assert_eq!(r("solve((cos(x)-1)*(cos(x)+1), x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(-cos(x)^2, x)"), "{ 1/2·pi + k·pi : k ∈ ℤ }");
    // Odd-power forms (sin^3 used to decline; -sin^3 collapsed to {0}).
    assert_eq!(r("solve(sin(x)^3, x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x)^3, x)"), "{ 1/2·pi + k·pi : k ∈ ℤ }");
    assert_eq!(
        r("solve((cos(x)+1)*(cos(x)-1)*sin(x), x)"),
        "{ k·pi : k ∈ ℤ }"
    );
    // Quotient form with a complementary denominator (sin*tan = sin^2/cos).
    assert_eq!(r("solve(sin(x)*tan(x), x)"), "{ k·pi : k ∈ ℤ }");
    // Controls: the `= c` squared forms and non-trig equations are unchanged.
    assert_eq!(r("solve(sin(x)^2 = 1, x)"), "{ 1/2·pi + k·pi : k ∈ ℤ }");
    assert_eq!(
        r("solve(4*cos(x)^2 = 1, x)"),
        "{ 1/3·pi + k·pi, 2/3·pi + k·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(x^2 - 1, x)"), "{ -1, 1 }");
}

#[test]
fn test_eval_trig_equation_with_surd_rhs_keeps_full_periodic_family() {
    // `sin(x) = √2/2` (and the other special-angle SURD right-hand sides) returned only the principal
    // value `{π/4}`: the periodic solver classified the RHS magnitude with `as_rational_const`, which
    // bails on an irrational, so the whole periodic path declined and the generic inverse leaked one
    // root. The classification is now exact over a quadratic surd (`linear_surd_sign`), so the full
    // two-branch periodic family is emitted — and `arcsin(√2/2)` simplifies to `π/4`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("solve(sin(x) = sqrt(2)/2, x)"),
        "{ 1/4·pi + k·2·pi, 3/4·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(x) = sqrt(3)/2, x)"),
        "{ 1/3·pi + k·2·pi, 2/3·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(cos(x) = sqrt(3)/2, x)"),
        "{ 1/6·pi + k·2·pi, 11/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(cos(x) = -sqrt(2)/2, x)"),
        "{ 3/4·pi + k·2·pi, 5/4·pi + k·2·pi : k ∈ ℤ }"
    );
    // Controls: rational RHS (special angle and general), the ±1 / 0 boundaries, and out-of-range
    // are all unchanged.
    assert_eq!(
        r("solve(sin(x) = 1/2, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(x) = 1/3, x)"),
        "{ arcsin(1/3) + k·2·pi, pi - arcsin(1/3) + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(sin(x) = 1, x)"), "{ 1/2·pi + k·2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(x) = 0, x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(x) = 2, x)"), "No solution");
}

#[test]
fn test_eval_trig_equation_affine_argument_and_odd_power_keep_family() {
    // Two more periodic-family-drop forms. (b) an AFFINE argument `sin(x - 1) = 0` returned only the
    // principal `{1}` — the arg detector handled `a·x` but not `a·x + b`; it now peels the offset and
    // shifts the family (`x = (u - b)/a`). (c) an ODD power `cos(x)^3 = 1` returned `{0}` — it now
    // reduces `trig^n = c` (n odd) to `trig = c^(1/n)` (a bijection on ℝ) and recurses.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // (b) affine argument: shifted, scaled+shifted, and the c=1 single-family form.
    assert_eq!(r("solve(sin(x-1)=0, x)"), "{ 1 + k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x+1)=0, x)"), "{ 1/2·(pi - 2) + k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(3*x-1)=0, x)"), "{ 1/3 + k·1/3·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x-1)=1, x)"), "{ 1 + k·2·pi : k ∈ ℤ }");
    // (c) odd power = constant: real n-th root, then the full family.
    assert_eq!(r("solve(cos(x)^3=1, x)"), "{ k·2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(x)^3=1, x)"), "{ 1/2·pi + k·2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x)^3=-1, x)"), "{ pi + k·2·pi : k ∈ ℤ }");
    // sin(x)^5 = 1/32 -> sin(x) = 1/2 -> both branches.
    assert_eq!(
        r("solve(sin(x)^5=1/32, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    // SOUNDNESS: sin(x)^n ∈ [-1, 1], so an out-of-range RHS has NO real solution (must not leak the
    // spurious non-real arcsin(2^(1/3)) the cube-root reduction would otherwise produce).
    assert_eq!(r("solve(sin(x)^3=2, x)"), "No solution");
    assert_eq!(r("solve(cos(x)^3=8, x)"), "No solution");
    // Controls: the n=2 square reduction and the bare/scaled forms are unchanged.
    assert_eq!(r("solve(cos(x)^2=1, x)"), "{ k·pi : k ∈ ℤ }");
    assert_eq!(
        r("solve(sin(2*x)=1/2, x)"),
        "{ 1/12·pi + k·pi, 5/12·pi + k·pi : k ∈ ℤ }"
    );
}

#[test]
fn test_eval_even_power_and_abs_trig_equation_keeps_family() {
    // `trig(x)^n = c` for EVEN n >= 4 (and `|trig(x)| = c`) collapsed the infinite periodic root set
    // to a finite pair, or leaked a spurious arcsin(>1) for an out-of-range RHS. Now reduced to
    // `trig = ±c^(1/n)` (resp. `trig = ±c`) with a range guard.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Even power: the full two-branch periodic family (sin=+/-1, +/-c^(1/n), ...).
    assert_eq!(
        r("solve(sin(x)^4=1, x)"),
        "{ 1/2·pi + k·2·pi, -1/2·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(cos(x)^4=1, x)"), "{ k·2·pi, pi + k·2·pi : k ∈ ℤ }");
    assert_eq!(
        r("solve(sin(x)^4=1/16, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi, -1/6·pi + k·2·pi, 7/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(sin(x)^4=0, x)"), "{ k·pi : k ∈ ℤ }");
    // An n-th-root RHS (not a quadratic surd) now also emits the full family.
    assert_eq!(
        r("solve(sin(x)^4=1/4, x)"),
        "{ arcsin((1/4)^(1/4)) + k·2·pi, pi - arcsin((1/4)^(1/4)) + k·2·pi, -arcsin((1/4)^(1/4)) + k·2·pi, arcsin((1/4)^(1/4)) + pi + k·2·pi : k ∈ ℤ }"
    );
    // SOUNDNESS: an out-of-range RHS has NO real solution (no spurious arcsin(>1)).
    assert_eq!(r("solve(sin(x)^4=4, x)"), "No solution");
    assert_eq!(r("solve(sin(x)^6=2, x)"), "No solution");
    assert_eq!(r("solve(sin(x)^4=-1, x)"), "No solution");
    // |trig(x)| = c reduces the same way.
    assert_eq!(
        r("solve(abs(sin(x))=1, x)"),
        "{ 1/2·pi + k·2·pi, -1/2·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(abs(cos(x))=0, x)"), "{ 1/2·pi + k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(abs(sin(x))=2, x)"), "No solution");
    // Controls: n=2, odd power, and the bare form are unchanged.
    assert_eq!(r("solve(sin(x)^2=1, x)"), "{ 1/2·pi + k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x)^3=1, x)"), "{ k·2·pi : k ∈ ℤ }");
    assert_eq!(
        r("solve(sin(x)=1/2, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
}

#[test]
fn test_eval_boundary_trig_inequality_is_periodic_point_set_or_residual() {
    // A bare sin/cos inequality at the EXACT range boundary +-1 returned a wrong ray
    // (`sin(x) >= 1 -> [pi/2, infinity)`). The TOUCH side holds only where the trig equals the extreme,
    // so it is the periodic point set; the COMPLEMENT side is R minus those points (not representable)
    // and declines to an honest residual instead of the wrong ray.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Touch side -> periodic point set (Periodic variant).
    assert_eq!(r("solve(sin(x) >= 1, x)"), "{ 1/2·pi + k·2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(sin(x) <= -1, x)"), "{ -1/2·pi + k·2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x) >= 1, x)"), "{ k·2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(cos(x) <= -1, x)"), "{ pi + k·2·pi : k ∈ ℤ }");
    // Complement side -> honest residual (no more wrong ray).
    assert_eq!(
        r("solve(cos(x) < 1, x)"),
        "{ (k·2·pi, 2·pi + k·2·pi) : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(sin(x) > -1, x)"),
        "{ (-1/2·pi + k·2·pi, 3/2·pi + k·2·pi) : k ∈ ℤ }"
    );
    // Range-guard combinations stay exact R / empty.
    assert_eq!(r("solve(sin(x) <= 1, x)"), "All real numbers");
    assert_eq!(r("solve(sin(x) > 1, x)"), "No solution");
    assert_eq!(r("solve(sin(x) >= -1, x)"), "All real numbers");
    assert_eq!(r("solve(cos(x) < -1, x)"), "No solution");
}

#[test]
fn test_eval_reciprocal_power_inequality_keeps_pole_sign_split() {
    // `c/xⁿ {op} k` with an ODD `n ≥ 3` (or a surd-border even `n`) used to drop the sign-flip across
    // the x=0 pole, returning a complement / phantom ray / a union with the pole filled in. The
    // sign-split candidate is now ordered exactly (cube/4th/5th-root bounds) and verified, so each
    // case is the correct punctured union. Verified numerically against the ground-truth predicate.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("solve(2/x^3 > -1, x)"),
        "(-infinity, -(2^(1/3))) U (0, infinity)"
    );
    assert_eq!(r("solve(1/x^3 > 2, x)"), "(0, (1/2)^(1/3))");
    assert_eq!(
        r("solve(1/x^3 < 2, x)"),
        "(-infinity, 0) U ((1/2)^(1/3), infinity)"
    );
    assert_eq!(r("solve(1/x^5 > 2, x)"), "(0, (1/2)^(1/5))");
    assert_eq!(r("solve(3/x^3 > 1, x)"), "(0, 3^(1/3))");
    assert_eq!(r("solve(2/x^3 < -1, x)"), "(-(2^(1/3)), 0)");
    // Surd-border even power: the pole at 0 must be EXCLUDED (punctured union, not a single interval).
    assert_eq!(r("solve(1/x^4 > 1/4, x)"), "(-(4^(1/4)), 0) U (0, 4^(1/4))");
    // Controls that must stay correct: rational-border even power and linear denominator.
    assert_eq!(r("solve(1/x^2 > 1, x)"), "(-1, 0) U (0, 1)");
    assert_eq!(r("solve(1/x > 2, x)"), "(0, 1/2)");
}

#[test]
fn test_eval_two_sided_rational_inequality_moves_to_one_side() {
    // `A(x) {op} B(x)` with the variable on BOTH sides and a rational difference (`1/(x-1) > 1/(x+1)`)
    // reached a path that emitted a garbage `inf^(1/2)` bound when the difference numerator is a nonzero
    // constant — or `{2}` / "No solution" for other shapes — even though the explicit-difference form
    // solved correctly. It is now moved to one side (`(A - B) {op} 0`) and routed through the verified
    // `N/D {op} 0` path.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Constant-numerator difference (the `inf^(1/2)` garbage case).
    assert_eq!(
        r("solve(1/(x-1) > 1/(x+1), x)"),
        "(-infinity, -1) U (1, infinity)"
    );
    assert_eq!(r("solve(1/(x+2) > 1/(x-2), x)"), "(-2, 2)");
    assert_eq!(
        r("solve(3/(x-1) > 3/(x+1), x)"),
        "(-infinity, -1) U (1, infinity)"
    );
    assert_eq!(r("solve(1/(x-1) < 1/(x+1), x)"), "(-1, 1)");
    // A linear-numerator difference (was returning the boundary point `{2}`).
    assert_eq!(r("solve(1/(x-1) > 3/(x+1), x)"), "(-infinity, -1) U (1, 2)");
    // Fraction vs a polynomial side (was "No solution"); irrational golden-ratio bounds.
    assert_eq!(
        r("solve(1/(x-1) > x, x)"),
        "(-infinity, 1/2·(1 - sqrt(5))) U (1, phi)"
    );
    // Non-strict keeps the numerator zero as a CLOSED endpoint, poles excluded.
    assert_eq!(
        r("solve(2/(x-1) >= 3/(x-2), x)"),
        "(-infinity, -1] U (1, 2)"
    );
    // Controls: an already-correct two-sided form, a radical two-sided (NOT preempted), and a
    // polynomial two-sided (declines the rational path, solved by its own).
    assert_eq!(r("solve(1/(x-1) > 2/(x+1), x)"), "(-infinity, -1) U (1, 3)");
    assert_eq!(r("solve(sqrt(x) > x - 2, x)"), "[0, 4)");
    assert_eq!(r("solve(x^2 > x, x)"), "(-infinity, 0) U (1, infinity)");

    // A nonzero constant numerator over a SINGLE-POLE linear-SURD/π denominator reduces exactly to the
    // boundary `g {op'} 0` (`Polynomial::from_expr` declines the irrational intercept `x − √2`, which
    // used to leave a garbage `(√2+∞, ∞)` interval on the legacy path).
    assert_eq!(r("solve(1/(x-sqrt(2)) > 0, x)"), "(sqrt(2), infinity)");
    assert_eq!(r("solve(1/(x-pi) > 0, x)"), "(pi, infinity)");
    assert_eq!(r("solve(1/(x-pi) < 0, x)"), "(-infinity, pi)");
    assert_eq!(r("solve(2/(x-sqrt(3)) < 0, x)"), "(-infinity, sqrt(3))");
}

#[test]
fn test_eval_sign_via_abs_excludes_pole() {
    // `g/|g| {op} c` is `sign(g) {op} c`, sign in {-1, +1} and undefined at g=0. The generic path
    // returned a CLOSED ray including the 0/0 point (`x/|x| = 1 -> [0, infinity)`) or "No solution" for
    // the inequality forms. It now reduces to a strict sign condition on g, with OPEN pole exclusion.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(x/abs(x) = 1, x)"), "(0, infinity)");
    assert_eq!(r("solve(x/abs(x) = -1, x)"), "(-infinity, 0)");
    assert_eq!(r("solve(abs(x)/x = 1, x)"), "(0, infinity)");
    assert_eq!(r("solve((x-2)/abs(x-2) = 1, x)"), "(2, infinity)");
    assert_eq!(r("solve((x-2)/abs(x-2) >= 1, x)"), "(2, infinity)");
    assert_eq!(
        r("solve((x-2)/abs(x-2) <= 1, x)"),
        "(-infinity, 2) U (2, infinity)"
    );
    assert_eq!(r("solve((x-2)/abs(x-2) < 1, x)"), "(-infinity, 2)");
    // sign(g) is never 0 or out of {-1,+1}.
    assert_eq!(r("solve(x/abs(x) = 2, x)"), "No solution");
    assert_eq!(r("solve(x/abs(x) = 0, x)"), "No solution");
    // Controls: genuine abs equations/inequalities (denominator is not |numerator|) are unchanged.
    assert_eq!(r("solve(abs(x) = 3, x)"), "{ 3, -3 }");
    assert_eq!(r("solve(abs(x)/2 = 1, x)"), "{ -2, 2 }");
    assert_eq!(
        r("solve(abs(x-1) > 2, x)"),
        "(-infinity, -1) U (3, infinity)"
    );
}

#[test]
fn test_eval_sign_via_abs_with_coefficient_excludes_pole() {
    // The sign form carries a COEFFICIENT: `c·g/|g| = c·sign(g)`. The bare detector required the
    // numerator to equal the abs-argument exactly, so any coefficient (`-x/|x|`, `3x/|x|`) fell to the
    // generic path that returned a CLOSED ray including the `0/0` pole — or "No solution" for the
    // inequalities. Peeling `c` reduces to `sign(g) {op} k/c` (flipping a strict op when `c < 0`), with
    // OPEN pole exclusion.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Negative unit coefficient (`-sign(g)`): the ray flips and the pole is OPEN.
    assert_eq!(r("solve(-x/abs(x) = 1, x)"), "(-infinity, 0)");
    assert_eq!(r("solve(-x/abs(x) = -1, x)"), "(0, infinity)");
    assert_eq!(r("solve(-(x-2)/abs(x-2) = 1, x)"), "(-infinity, 2)");
    assert_eq!(r("solve(-3*x/abs(3*x) = 1, x)"), "(-infinity, 0)");
    // Negated inequalities were "No solution"; now the correct half-line.
    assert_eq!(r("solve(-x/abs(x) >= 1, x)"), "(-infinity, 0)");
    assert_eq!(r("solve(-x/abs(x) < 1, x)"), "(0, infinity)");
    // Positive coefficient ≠ 1 (the abs-argument is bare `x`, not `c·x`): also excludes the pole now.
    assert_eq!(r("solve(3*x/abs(x) = 3, x)"), "(0, infinity)");
    assert_eq!(r("solve(2*x/abs(x) = 2, x)"), "(0, infinity)");
    // `|g|/g` with a negated denominator: `|x|/(-x) = -sign(x)` (was a garbage conditional).
    assert_eq!(r("solve(abs(x)/(-x) = 1, x)"), "(-infinity, 0)");
    // ABS in the NUMERATOR with a coefficient/negation: `c·|g|/g = c·sign(g)`. `2*abs(x)/x` and
    // `-abs(x)/x` simplify to `Div(Mul(c, |x|), x)`, whose raw numerator is not a bare abs, so the
    // coefficient sibling of `|x|/x` returned a garbage `All real numbers if [linear] >= 0`
    // conditional (a wrong answer). Peeling `c` from BOTH sides of the division fixes it.
    assert_eq!(r("solve(-abs(x)/x = 1, x)"), "(-infinity, 0)");
    assert_eq!(r("solve(-abs(x)/x = -1, x)"), "(0, infinity)");
    assert_eq!(r("solve(2*abs(x)/x = 2, x)"), "(0, infinity)");
    assert_eq!(r("solve(-2*abs(x)/x = 2, x)"), "(-infinity, 0)");
    assert_eq!(r("solve(abs(x)/(2*x) = 1/2, x)"), "(0, infinity)");
    // Controls: the bare and matched-coefficient forms are unchanged; a rescaled RHS that no sign
    // value can hit is empty.
    assert_eq!(r("solve(x/abs(x) = 1, x)"), "(0, infinity)");
    assert_eq!(r("solve(2*x/abs(2*x) = 1, x)"), "(0, infinity)");
    assert_eq!(r("solve(-x/abs(x) = 2, x)"), "No solution");
    assert_eq!(r("solve(3*x/abs(x) = 2, x)"), "No solution");
}

#[test]
fn test_eval_sign_via_abs_with_additive_constant_excludes_pole() {
    // An ADDITIVE constant on the sign form (`sign(g) + d {op} k`) was not peeled, so the detector
    // declined and the generic path returned "No solution" (or a closed ray with the `0/0` pole). The
    // constant now folds into the reduced RHS: `coeff·sign(g) + offset {op} k ⟺ sign(g) {op} (k-offset)/coeff`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `sign(x) + 1 > 0` ⟺ `sign(x) > -1` ⟺ `x > 0` (pole open).
    assert_eq!(r("solve(x/abs(x) + 1 > 0, x)"), "(0, infinity)");
    assert_eq!(r("solve(x/abs(x) + 1 = 2, x)"), "(0, infinity)");
    assert_eq!(r("solve(x/abs(x) - 1 < 0, x)"), "(-infinity, 0)");
    assert_eq!(r("solve(x/abs(x) - 1 = 0, x)"), "(0, infinity)");
    // `sign(x) > -2` holds for both sign values, so the whole domain minus the pole.
    assert_eq!(
        r("solve(2 + x/abs(x) > 0, x)"),
        "(-infinity, 0) U (0, infinity)"
    );
    assert_eq!(
        r("solve(x/abs(x) + 2 > 0, x)"),
        "(-infinity, 0) U (0, infinity)"
    );
    // Negated sign with an offset: `-sign(x) + 1 > 0` ⟺ `sign(x) < 1` ⟺ `x < 0`. The `3 - sign(x)`
    // constant exceeds the sign range, so again everything but the pole.
    assert_eq!(r("solve(-x/abs(x) + 1 > 0, x)"), "(-infinity, 0)");
    assert_eq!(
        r("solve(3 - x/abs(x) > 0, x)"),
        "(-infinity, 0) U (0, infinity)"
    );
    // Control: no offset (Family 3) and an unreachable reduced RHS stay correct.
    assert_eq!(r("solve(x/abs(x) = 1, x)"), "(0, infinity)");
    assert_eq!(r("solve(x/abs(x) + 1 > 3, x)"), "No solution");
}

#[test]
fn test_eval_sign_form_equals_variable_rhs_splits_on_sign() {
    // `coeff·sign(g) + offset = h(x)` with a VARIABLE RHS (`x/|x| = x`) leaked a
    // malformed residual (the isolation cleared the denominator to `x = x·|x|`).
    // The sign form is a step function, so it splits on `sign(g) = ±1`: solve
    // `h = coeff+offset` on `g > 0` and `h = -coeff+offset` on `g < 0`, unioning
    // (the pole `g = 0` excluded by the STRICT branch). Verified by substitution.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(x/abs(x) = x, x)"), "{ 1, -1 }");
    assert_eq!(r("solve(abs(x)/x = x, x)"), "{ 1, -1 }");
    // Coefficiented sign form: `2·sign(x) = x`.
    assert_eq!(r("solve(2*x/abs(x) = x, x)"), "{ 2, -2 }");
    assert_eq!(r("solve(x/abs(x) = 2*x, x)"), "{ 1/2, -1/2 }");
    // A branch's root can fall OUTSIDE its sign-domain and be dropped: `sign(x) = x²`
    // keeps only x=1 (x=-1 has sign -1 ≠ 1).
    assert_eq!(r("solve(x/abs(x) = x^2, x)"), "{ 1 }");
    assert_eq!(r("solve(x/abs(x) = x - 2, x)"), "{ 3 }");
    // `sign(x) = -x` and `-sign(x) = x` have NO solution (neither ±1 lands in its
    // own half-line) — the audit's stated "{-1,1}" was itself wrong.
    assert_eq!(r("solve(x/abs(x) = -x, x)"), "No solution");
    assert_eq!(r("solve(-x/abs(x) = x, x)"), "No solution");

    // NO REGRESSION: constant-RHS equations and inequalities keep their handler.
    assert_eq!(r("solve(x/abs(x) = 1, x)"), "(0, infinity)");
    assert_eq!(r("solve(-abs(x)/x = 1, x)"), "(-infinity, 0)");
    assert_eq!(r("solve(x/abs(x) > 0, x)"), "(0, infinity)");
}

#[test]
fn test_eval_sign_form_sum_partitions_at_poles() {
    // A SUM of ≥2 sign forms `Σ cᵢ·sign(gᵢ) {op} k` is a step function (the simplifier combines it over a
    // common denominator and the isolation path then returns "No solution" / a garbage residual). It now
    // partitions ℝ at the `gᵢ = 0` poles, evaluates the constant sum on each open region, and keeps the
    // satisfying ones — the poles excluded.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `sign(x+1) + sign(x-1) > 0` is +2 only on `(1, ∞)`.
    assert_eq!(
        r("solve((x+1)/abs(x+1) + (x-1)/abs(x-1) > 0, x)"),
        "(1, infinity)"
    );
    assert_eq!(
        r("solve(x/abs(x) + (x-2)/abs(x-2) > 0, x)"),
        "(2, infinity)"
    );
    // A difference of signs (`sign(x) − sign(x-2)`) is +2 only between the poles.
    assert_eq!(r("solve(x/abs(x) - (x-2)/abs(x-2) > 0, x)"), "(0, 2)");
    // `= 0` keeps the middle region where the signs cancel.
    assert_eq!(
        r("solve((x+1)/abs(x+1) + (x-1)/abs(x-1) = 0, x)"),
        "(-1, 1)"
    );
    // Three terms, and a constant RHS on the sum.
    assert_eq!(
        r("solve(x/abs(x) + (x-1)/abs(x-1) + (x-2)/abs(x-2) > 1, x)"),
        "(2, infinity)"
    );
    assert_eq!(
        r("solve((x+1)/abs(x+1) + (x-1)/abs(x-1) = 2, x)"),
        "(1, infinity)"
    );
    // A weighted sum; `>= 0` holds on two regions, the pole between them excluded.
    assert_eq!(
        r("solve(2*x/abs(x) + (x-1)/abs(x-1) >= 0, x)"),
        "(0, 1) U (1, infinity)"
    );
    assert_eq!(
        r("solve(x/abs(x) + (x-2)/abs(x-2) >= 0, x)"),
        "(0, 2) U (2, infinity)"
    );
    assert_eq!(
        r("solve(x/abs(x) + (x-2)/abs(x-2) < 0, x)"),
        "(-infinity, 0)"
    );
    // Controls: a SINGLE sign form (n = 1) stays with the dedicated handler.
    assert_eq!(r("solve(x/abs(x) = 1, x)"), "(0, infinity)");
    assert_eq!(r("solve(x/abs(x) + 1 > 0, x)"), "(0, infinity)");
}

#[test]
fn test_eval_polynomial_in_log_inequality_back_substitutes_through_exp() {
    // `P(ln(x)) {op} 0` (degree ≥ 2 in `ln(x)`) used to collapse to "No solution": the polynomial-in-u
    // path solved the EQUATION but the inequality dropped the band. It now solves for `u = ln(x)` and
    // maps each u-interval directly through the increasing `x = e^u`: `a < ln(x) < b  ⟺  e^a < x < e^b`,
    // with `-∞ → 0` (the `x > 0` domain edge, OPEN) and `+∞ → +∞`. Building `e^bound` directly avoids the
    // bound comparator (which could not order `1/e²` against `e²` and previously emptied the band).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Distinct rational roots: the band between the two exponentials.
    assert_eq!(r("solve(ln(x)^2 - 3*ln(x) + 2 < 0, x)"), "(e, e^2)");
    assert_eq!(r("solve(ln(x)^2 - 3*ln(x) + 2 <= 0, x)"), "[e, e^2]");
    // Complement: the two outer rays, `(0, e)` keeping the `x > 0` domain edge.
    assert_eq!(
        r("solve(ln(x)^2 - 3*ln(x) + 2 > 0, x)"),
        "(0, e) U (e^2, infinity)"
    );
    // A root at 0 (`ln(x)(ln(x) - 2)`) maps to `x = 1`.
    assert_eq!(r("solve(ln(x)^2 - 2*ln(x) < 0, x)"), "(1, e^2)");
    assert_eq!(r("solve(ln(x)^2 - 5*ln(x) + 6 < 0, x)"), "(e^2, e^3)");
    // Symmetric `ln² - 4`: the band is `(e^-2, e^2)`, rendered with the reciprocal lower bound.
    assert_eq!(r("solve(ln(x)^2 - 4 < 0, x)"), "(1 / e^2, e^2)");
    assert_eq!(
        r("solve(ln(x)^2 - 4 >= 0, x)"),
        "(0, 1 / e^2] U [e^2, infinity)"
    );
    // Controls: a single `ln` (degree 1) stays the ordinary monotonic isolation, and the equation form
    // is unchanged.
    assert_eq!(r("solve(ln(x) > 1, x)"), "(e, infinity)");
    assert_eq!(r("solve(ln(x)^2 - 3*ln(x) + 2 = 0, x)"), "{ e, e^2 }");
}

#[test]
fn test_eval_affine_argument_polynomial_in_log_inequality() {
    // A polynomial-in-`ln(g)` inequality with an AFFINE argument `g = a·x + b` (`ln(2x)`, `ln(x-1)`)
    // used to return "No solution" (the handler was restricted to the bare `ln(x)`). The u-band now maps
    // back through the affine inverse `x = (e^u − b)/a`: `u ∈ (p, q) ⟺ x ∈ ((e^p − b)/a, (e^q − b)/a)`,
    // with the bounds swapping when a < 0 and the `−∞` end giving the domain edge `−b/a`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Scaled argument `ln(2x)`: band `e^-2 < 2x < e^2`.
    assert_eq!(r("solve(ln(2*x)^2 - 4 < 0, x)"), "(1 / (2·e^2), 1/2·e^2)");
    // Shifted argument `ln(x-1)`: band `e < x-1 < e^2`.
    assert_eq!(
        r("solve(ln(x-1)^2 - 3*ln(x-1) + 2 < 0, x)"),
        "(1 + e, e^2 + 1)"
    );
    // Complement with the domain edge `x > 1/3` kept open.
    assert_eq!(
        r("solve(ln(3*x-1)^2 - 4 >= 0, x)"),
        "(1/3, (e^2 + 1) / (3·e^2)] U [1/3·(e^2 + 1), infinity)"
    );
    // Negative slope `ln(1-x)` (a = -1): the bounds swap, giving `1 - e^2 < x < 1 - e^-2`.
    assert_eq!(
        r("solve(ln(1-x)^2 - 4 < 0, x)"),
        "(1 - e^2, -(1 - e^2) / e^2)"
    );
    // AllReals in u ⇒ the affine DOMAIN `g > 0` (`2x > 0 ⟺ x > 0`), NOT a blanket `x > 0` coincidence.
    assert_eq!(r("solve(ln(2*x)^2 + 1 > 0, x)"), "(0, infinity)");
    assert_eq!(r("solve(ln(x-1)^2 + 1 < 0, x)"), "No solution");
    // Controls: the bare `ln(x)` case and the single-`ln` affine isolation are unchanged.
    assert_eq!(r("solve(ln(x)^2 - 4 < 0, x)"), "(1 / e^2, e^2)");
    assert_eq!(r("solve(ln(2*x) > 1, x)"), "(1/2·e, infinity)");
}

#[test]
fn test_eval_rational_power_polynomial_inequality() {
    // A polynomial-in-`x^(1/q)` inequality (`x − 3√x + 2 < 0`, a quadratic in `√x`) used to emit an
    // honest-but-incomplete residual. It now substitutes `u = x^(1/q)`, solves the u-inequality, and
    // maps the u-band back through `x = u^q`, honouring the `u ≥ 0` (and `x ≥ 0`) domain for even q.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Quadratic in `√x` (q = 2 even): `1 < √x < 2 ⟺ 1 < x < 4`.
    assert_eq!(r("solve(x - 3*sqrt(x) + 2 < 0, x)"), "(1, 4)");
    assert_eq!(r("solve(x - 3*sqrt(x) + 2 <= 0, x)"), "[1, 4]");
    // Complement keeps the domain edge `x = 0` (`√x < 1 ⟺ 0 ≤ x < 1`).
    assert_eq!(
        r("solve(x - 3*sqrt(x) + 2 > 0, x)"),
        "[0, 1) U (4, infinity)"
    );
    assert_eq!(
        r("solve(x - 3*sqrt(x) + 2 >= 0, x)"),
        "[0, 1] U [4, infinity)"
    );
    assert_eq!(r("solve(x - 5*sqrt(x) + 6 < 0, x)"), "(4, 9)");
    // Quadratic in `x^(1/3)` (q = 3 odd): the whole real line is the u-domain, so the band is signed.
    assert_eq!(r("solve(x^(2/3) - x^(1/3) - 2 < 0, x)"), "(-1, 8)");
    assert_eq!(
        r("solve(x^(2/3) - x^(1/3) - 2 > 0, x)"),
        "(-infinity, -1) U (8, infinity)"
    );
    // No constant term (`u² - 3u = u(u-3)`): `0 < √x < 3 ⟺ 0 < x < 9`, the pole at u=0 open.
    assert_eq!(r("solve(x - 3*sqrt(x) < 0, x)"), "(0, 9)");
    // Controls: a degree-1 `√x` stays the ordinary monotonic isolation, a plain polynomial is unchanged,
    // and the equation form is untouched.
    assert_eq!(r("solve(sqrt(x) - 2 < 0, x)"), "[0, 4)");
    assert_eq!(r("solve(x^2 - 3*x + 2 < 0, x)"), "(1, 2)");
    assert_eq!(r("solve(x - 5*sqrt(x) + 6 = 0, x)"), "{ 4, 9 }");
}

#[test]
fn test_eval_high_degree_polynomial_inequality_with_rational_root() {
    // `xⁿ - c > 0` for odd n with a RATIONAL root (`x⁵-1 = (x-1)(x⁴+x³+x²+x+1)`) used to return
    // "No solution": the inequality path declined because it could not certify the positive-definite
    // residual quartic, while the EQUATION path finds the real root {1}. Running the sign analysis
    // over the equation's roots (its alternation + end-behaviour guards keep it sound) recovers the
    // interval. This also unblocks the reciprocal form `1/xⁿ > c` for n up to 12.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Direct polynomial inequalities (were "No solution").
    assert_eq!(r("solve(x^5 > 1, x)"), "(1, infinity)");
    assert_eq!(r("solve(x^9 > 1, x)"), "(1, infinity)");
    assert_eq!(r("solve(x^7 < 1, x)"), "(-infinity, 1)");
    assert_eq!(r("solve(x^5 > 32, x)"), "(2, infinity)");
    // Reciprocal forms for odd n >= 5 with a rational boundary (were inventing the negative ray).
    assert_eq!(r("solve(1/x^5 > 1, x)"), "(0, 1)");
    assert_eq!(r("solve(1/x^7 > 1, x)"), "(0, 1)");
    assert_eq!(r("solve(1/x^9 > 1, x)"), "(0, 1)");
    assert_eq!(r("solve(1/x^7 < 1, x)"), "(-infinity, 0) U (1, infinity)");
    // Surd-boundary and lower-degree controls remain correct.
    assert_eq!(r("solve(x^5 > 2, x)"), "(2^(1/5), infinity)");
    assert_eq!(r("solve(1/x^3 > 2, x)"), "(0, (1/2)^(1/3))");
    assert_eq!(r("solve(x^3 - x > 0, x)"), "(-1, 0) U (1, infinity)");
}

#[test]
fn test_eval_squared_irrational_quadratic_factor_keeps_its_roots() {
    // A polynomial with a SQUARED (or equal-constant) irreducible quadratic factor dropped that
    // factor's irrational roots: `(x²-3)²·(x-1) = 0` returned `{1}`, losing ±√3. The quartic-factor
    // solver factors the deflated monic quartic into `(x²+px+q)(x²+rx+s)`, but when the two factors
    // share a constant term (`q = s`, the perfect-square case) the `p = (d-qb)/(s-q)` formula divided
    // by zero, so that case was skipped — the roots of the repeated quadratic vanished. The `q = s`
    // branch now solves `p,r` from `t²-bt+(c-2q)=0` directly.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // The in-core biquadratic residual solver now owns these (2026-07-14): same set,
    // cleaner `-(sqrt(3))` render (was the quartic-factor owner's `-3·3^(-1/2)`).
    assert_eq!(
        r("solve((x^2-3)^2*(x-1) = 0, x)"),
        "{ 1, -(sqrt(3)), sqrt(3) }"
    );
    assert_eq!(
        r("solve((x^2-7)^2*(x-3) = 0, x)"),
        "{ 3, -(sqrt(7)), sqrt(7) }"
    );
    // A general (non-symmetric) irreducible quadratic, squared: roots (3±√5)/2.
    assert_eq!(
        r("solve((x^2-3*x+1)^2*(x-1) = 0, x)"),
        "{ 1, 1/2·(sqrt(5) + 3), 1/2·(3 - sqrt(5)) }"
    );
    // The bug survives full expansion (same quintic, factored back internally).
    assert_eq!(
        r("solve(x^5 - x^4 - 6*x^3 + 6*x^2 + 9*x - 9 = 0, x)"),
        "{ 1, -(sqrt(3)), sqrt(3) }"
    );
    // Degree-6 with two rational cofactor roots; the squared factor still contributes ±√3.
    assert_eq!(
        r("solve((x^2-3)^2*(x^2-4) = 0, x)"),
        "{ -2, 2, -(sqrt(3)), sqrt(3) }"
    );
    // Controls: the DISTINCT-quadratic-factor case and a plain quadratic are unchanged.
    assert_eq!(
        r("solve(x^5-5*x^3+x^2-5 = 0, x)"),
        "{ -1, sqrt(5), -5·5^(-1/2) }"
    );
    assert_eq!(r("solve(x^2-5*x+6 = 0, x)"), "{ 2, 3 }");
}

#[test]
fn test_eval_content_scaled_squared_quadratic_factor_keeps_roots() {
    // A CONTENT / scalar-multiple wrapper on the squared-quadratic case dropped the irrational roots:
    // `2·(x²-3)²·(x-1) = 0` returned `{1}`. After peeling the rational root, the deflated quotient is
    // `2·(x²-3)²` — a NON-monic quartic, which the factorizer rejected. Normalizing the quotient to
    // monic (dividing by the leading coefficient preserves the roots) recovers ±√3. The remaining
    // higher-multiplicity cases (`(x²-3)³`, two distinct irrational-root factors) deflate past degree 4
    // and stay residual — they need general ℚ-factorization.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Outer scalar content.
    assert_eq!(
        r("solve(2*(x^2-3)^2*(x-1) = 0, x)"),
        "{ 1, -(sqrt(3)), sqrt(3) }"
    );
    // Content folded INTO the squared factor (`(2x²-6)² = 4·(x²-3)²`).
    assert_eq!(
        r("solve((2*x^2-6)^2*(x-1) = 0, x)"),
        "{ 1, -(sqrt(3)), sqrt(3) }"
    );
    // A different scalar and root.
    assert_eq!(
        r("solve(3*(x^2-5)^2*(x-2) = 0, x)"),
        "{ 2, -(sqrt(5)), sqrt(5) }"
    );
    // NEGATIVE content (leading coefficient < 0) normalizes the same way.
    assert_eq!(
        r("solve(-2*(x^2-3)^2*(x-1) = 0, x)"),
        "{ 1, -(sqrt(3)), sqrt(3) }"
    );
    // Content on a non-repeated quartic (distinct factors) stays correct.
    assert_eq!(
        r("solve(2*x^4 - 10*x^2 + 12 = 0, x)"),
        "{ -(sqrt(2)), -(sqrt(3)), sqrt(2), sqrt(3) }"
    );
    // Control: the monic case is unchanged.
    assert_eq!(
        r("solve((x^2-3)^2*(x-1) = 0, x)"),
        "{ 1, -(sqrt(3)), sqrt(3) }"
    );
}

#[test]
fn test_eval_unsound_power_monomial_inequality_declines_to_residual() {
    // A power-monomial inequality `c·x^e {op} k` is solved by the engine's MONOTONIC isolation, which
    // emits a single ray — correct ONLY when `x^e` is strictly monotonic (`e > 0`, odd numerator).
    // An even-numerator VALLEY (`x^(2/3) = |x|^(2/3)`) is now SOLVED exactly by the `|x| {op} k^(q/p)`
    // reduction (its truth is two rays / a bounded interval). A NEGATIVE non-integer exponent
    // (`1/x^(1/3)`, `1/√x`) — a reciprocal fractional power with a pole — is still declined to an honest
    // residual (correct solving of the reciprocals is the next rung).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Even-numerator valleys are now SOLVED exactly (two rays for `>`, a bounded interval for `<`).
    assert_eq!(
        r("solve(x^(2/3) > 2, x)"),
        "(-infinity, -(2^(3/2))) U (2^(3/2), infinity)"
    );
    assert_eq!(r("solve(x^(2/3) < 2, x)"), "(-(2^(3/2)), 2^(3/2))");
    assert_eq!(
        r("solve(x^(2/5) > 2, x)"),
        "(-infinity, -(2^(5/2))) U (2^(5/2), infinity)"
    );
    // Negative non-integer exponents / reciprocal fractional powers (were complement / pole) — declined.
    assert_eq!(r("solve(1/x^(1/3) > 2, x)"), "(0, 1/8)");
    assert_eq!(r("solve(1/x^(1/2) > 2, x)"), "(0, 1/4)");
    assert_eq!(r("solve(x^(-1/3) > 2, x)"), "solve(x^(-1 / 3) > 2, x)");
    // KEEP: strictly-monotonic powers (e > 0, odd numerator) stay solved EXACTLY.
    assert_eq!(r("solve(x^(1/3) > 2, x)"), "(8, infinity)");
    assert_eq!(r("solve(x^(1/2) < 2, x)"), "[0, 4)");
    assert_eq!(r("solve(x^(3/2) > 2, x)"), "(2^(2/3), infinity)");
    assert_eq!(r("solve(x^(5/3) > 2, x)"), "(2^(3/5), infinity)");
    // KEEP: integer-exponent reciprocals are owned by the rational-constant path (Class B).
    assert_eq!(r("solve(1/x^3 > 2, x)"), "(0, (1/2)^(1/3))");
    assert_eq!(r("solve(1/x > 2, x)"), "(0, 1/2)");
    // KEEP: the EQUATION form is untouched (op gate) — both valley roots are found.
    assert_eq!(r("solve(x^(2/3) = 8, x)"), "{ -64·2^(-3/2), 64·2^(-3/2) }");
}

#[test]
fn test_eval_wrapped_non_monotonic_power_inequality_declines_to_residual() {
    // An even-numerator VALLEY through its WRAPPERS — a shifted/scaled affine base `(x-1)^(2/3)`, an
    // additive constant `x^(2/3) + 1` — is now SOLVED exactly by the `|a·x+b| {op} k^(q/p)` reduction.
    // The `sqrt` FUNCTION reciprocal `1/sqrt(x)` is SOLVED since U2 (w-space).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Shifted / scaled affine base (even-numerator valley) — SOLVED.
    assert_eq!(
        r("solve((x-1)^(2/3) > 4, x)"),
        "(-infinity, -7) U (9, infinity)"
    );
    assert_eq!(
        r("solve((2*x-3)^(2/3) > 4, x)"),
        "(-infinity, -5/2) U (11/2, infinity)"
    );
    // Additive constant on the power — SOLVED.
    assert_eq!(
        r("solve(x^(2/3) + 1 > 5, x)"),
        "(-infinity, -8) U (8, infinity)"
    );
    assert_eq!(r("solve(5 - x^(2/3) > 1, x)"), "(-8, 8)");
    // sqrt FUNCTION reciprocal — SOLVED since U2 via the w-space substitution.
    assert_eq!(r("solve(1/sqrt(x) > 2, x)"), "(0, 1/4)");
    assert_eq!(r("solve(1/sqrt(x-1) > 2, x)"), "(1, 5/4)");
    // KEEP: a shifted/scaled STRICTLY-MONOTONIC power (e > 0, odd numerator) stays solved exactly.
    assert_eq!(r("solve((x-1)^(1/3) > 2, x)"), "(9, infinity)");
    assert_eq!(r("solve(sqrt(x-1) > 2, x)"), "(5, infinity)");
    assert_eq!(r("solve(sqrt(x) < 2, x)"), "[0, 4)");
    // KEEP: an integer power of an affine base is a polynomial inequality, solved exactly.
    assert_eq!(
        r("solve((x-1)^2 > 4, x)"),
        "(-infinity, -1) U (3, infinity)"
    );
    assert_eq!(r("solve(1/(x-1) > 2, x)"), "(1, 3/2)");
}

#[test]
fn test_eval_uncombined_like_power_terms_valley_inequality() {
    // The solve path extracts the power term from the RAW LHS, where `x^(2/3) + x^(2/3)` (the variable
    // on BOTH sides of the `Add`) hit the `(_, _) => None` arm and bypassed the valley reduction — the
    // monotonic fall-through then dropped the `x < 0` ray (`> 8` gave `(8, ∞)`) or emitted garbage
    // (`>= 8` gave `[-8, -8] ∪ [8, ∞)`). The extractor now COMBINES like power terms (same affine base
    // and exponent), matching the standalone simplifier's `→ 2·x^(2/3)` fold.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("solve(x^(2/3) + x^(2/3) > 8, x)"),
        "(-infinity, -8) U (8, infinity)"
    );
    assert_eq!(
        r("solve(x^(2/3) + x^(2/3) >= 8, x)"),
        "(-infinity, -8] U [8, infinity)"
    );
    assert_eq!(r("solve(x^(2/3) + x^(2/3) < 8, x)"), "(-8, 8)");
    // Mixed coefficients combine to 3·x^(2/3); three terms also fold.
    assert_eq!(
        r("solve(x^(2/3) + 2*x^(2/3) > 9, x)"),
        "(-infinity, -(3^(3/2))) U (3^(3/2), infinity)"
    );
    assert_eq!(
        r("solve(x^(2/3) + x^(2/3) + x^(2/3) > 12, x)"),
        "(-infinity, -8) U (8, infinity)"
    );
    // Shifted base combines too.
    assert_eq!(
        r("solve((x-1)^(2/3) + (x-1)^(2/3) > 8, x)"),
        "(-infinity, -7) U (9, infinity)"
    );
    // UNLIKE exponents sharing the SAME base (`x^(2/3) + x^(4/3)`, a quartic in `x^(1/3)`) are not a
    // single valley, but the rational-power-polynomial handler now solves them (`u⁴ + u² - 8 > 0`).
    assert_eq!(
        r("solve(x^(2/3) + x^(4/3) > 8, x)"),
        "(-infinity, -((1/2·(sqrt(33) - 1))^(3/2))) U ((1/2·(sqrt(33) - 1))^(3/2), infinity)"
    );
    // A DIFFERENT base (`(x-1)^(2/3)`) is not an `x`-power polynomial, so it stays residual.
    assert_eq!(
        r("solve(x^(2/3) + (x-1)^(2/3) > 8, x)"),
        "solve(x - (8 - (x - 1)^(2/3))^(1 / 2/3) = 0, x)"
    );
    // Exact cancellation is empty; the odd-power and integer-power forms stay correct.
    assert_eq!(r("solve(x^(2/3) - x^(2/3) > 0, x)"), "No solution");
    assert_eq!(r("solve(x^(1/3) + x^(1/3) > 8, x)"), "(64, infinity)");
    assert_eq!(
        r("solve(x^2 + x^2 > 8, x)"),
        "(-infinity, -2) U (2, infinity)"
    );
}

#[test]
fn test_eval_definite_integral_removable_pole_is_not_undefined() {
    // A rationalization step turns `1/(√x·(1+x))` into `(√x³−√x)/(x³−x)`, inventing a SPURIOUS
    // denominator root at x=1 where the numerator also vanishes (removable). The FTC pole scan used
    // to reject it as an in-interval pole and return a false `undefined` on a convergent / regular
    // proper integral. The (continuous) antiderivative `2·arctan(√x)` is finite at x=1, certifying
    // the singularity removable, so the integral evaluates.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Regular proper interval [1/2, 4] (NO singularity in it): 2·(arctan(2) − arctan(√½)) ≈ 0.9833.
    assert_eq!(
        r("integrate(1/(sqrt(x)*(1+x)),x,1/2,4)"),
        "2·(arctan(2) - arctan(sqrt(1/2)))"
    );
    // Convergent improper integral [1, ∞) = π/2.
    assert_eq!(r("integrate(1/(sqrt(x)*(1+x)),x,1,inf)"), "1/2·pi");
    // The interval clear of the spurious root is unaffected.
    assert_eq!(
        r("integrate(1/(sqrt(x)*(1+x)),x,4,9)"),
        "2·(arctan(3) - arctan(2))"
    );
    // Pure-rational removable singularities also evaluate: (x−1)/(x²−1) = 1/(x+1) on [0,3] = ln(4).
    assert_eq!(r("integrate((x-1)/(x^2-1),x,0,3)"), "ln(4)");
    // SOUNDNESS: a GENUINE interior pole (numerator nonzero) still diverges → undefined.
    assert_eq!(r("integrate(1/(x-1),x,0,2)"), "undefined");
    assert_eq!(r("integrate(1/((x-1)*(x-3)),x,0,4)"), "undefined");
    assert_eq!(r("integrate(1/(x-2)^2,x,1,3)"), "undefined");
}

#[test]
fn test_eval_definite_integral_provably_positive_transcendental_denominator() {
    // `∫ 1/(e^x+1)` computes the antiderivative `ln(e^x/(e^x+1))`, but the DEFINITE
    // form leaked: the pole certificate could not `Polynomial::from_expr` the
    // transcendental denominator `e^x+1`, returned Unknown, and declined. Since
    // `e^x+1 > 0` everywhere (the real-domain sign prover decides `e^x > 0`), it has
    // no pole, so the FTC evaluation is safe. Cross-checked vs sympy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `∫₀¹ 1/(e^x+1) = ln(2) + 1 - ln(1+e)` = `ln(2e/(1+e))` ≈ 0.37989.
    assert_eq!(r("integrate(1/(e^x+1), x, 0, 1)"), "ln((e·2)/(1 + e))");
    assert_eq!(r("integrate(1/(e^x+3), x, 0, 1)"), "1/3·ln((e·4)/(3 + e))");
    // Numerator = e^x (antiderivative ln(e^x+1)).
    assert_eq!(r("integrate(e^x/(e^x+1), x, 0, 1)"), "ln(1/2·(1 + e))");
    // SOUNDNESS: a denominator with a REAL root is NOT provably positive everywhere,
    // so it is NOT falsely certified — `e^x-1` vanishes at x=0 (a genuine pole inside
    // [-1,1]) and stays an honest residual, and polynomial poles are still caught.
    assert_eq!(
        r("integrate(1/(e^x-1), x, -1, 1)"),
        "integrate(1 / (e^x - 1), x, -1, 1)"
    );
    assert_eq!(r("integrate(1/(x-1), x, 0, 2)"), "undefined");
    // No regression on the already-working rational and log cases.
    assert_eq!(r("integrate(1/(x^2+1), x, 0, 1)"), "1/4·pi");
    assert_eq!(r("integrate(1/x, x, 1, e)"), "1");
}

#[test]
fn test_eval_nth_root_reciprocal_integral_uses_correct_conjugate() {
    // `1/x^(1/n)` rationalized its denominator by multiplying by the BARE root `x^(1/n)`, which only
    // clears a SQUARE root: `x^(1/4)·x^(1/4) = x^(1/2) ≠ x`. So `1/x^(1/4)` became `x^(1/4)/x = x^(-3/4)`
    // and integrated to a WRONG `4·x^(1/4)` (whose derivative `x^(-3/4)` ≠ the integrand `x^(-1/4)`).
    // The conjugate `x^((n-1)/n)` now clears it correctly: `1/x^(1/4) → x^(3/4)/x → x^(-1/4)`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Indefinite even-root antiderivatives (true `(n/(n-1))·x^((n-1)/n)`), verified by diff-back.
    assert_eq!(r("integrate(1/x^(1/4),x)"), "4/3·x^(3/4)");
    assert_eq!(r("integrate(1/x^(1/6),x)"), "6/5·x^(5/6)");
    assert_eq!(r("diff(4/3*x^(3/4),x)"), "x^(-1/4)");
    // Definite even-root integrals on [0,1] (true `n/(n-1)`).
    assert_eq!(r("integrate(1/x^(1/4),x,0,1)"), "4/3");
    assert_eq!(r("integrate(1/x^(1/6),x,0,1)"), "6/5");
    assert_eq!(r("integrate(1/x^(1/8),x,0,1)"), "8/7");
    // Square-root rationalization (n=2) is unchanged.
    assert_eq!(r("integrate(1/sqrt(x),x)"), "2·sqrt(x)");
    assert_eq!(r("integrate(1/(x*sqrt(x)),x)"), "-2 / sqrt(x)");
    // ODD-root and general fractional reciprocal powers: the simplifier rationalizes
    // `1/x^(1/3)` to `x^(2/3)/x` (and leaves `1/x^(2/5)` as-is), which the power-rule
    // matcher missed — only the even-root `1/x^(1/(2k))` cases above worked. Folding
    // `(c·)x^a/x^b → c·x^(a-b)` for a FRACTIONAL exponent recovers the power rule, in
    // both the indefinite and definite paths. Verified by diff-back and sympy.
    assert_eq!(r("integrate(1/x^(1/3),x)"), "3/2·x^(2/3)");
    assert_eq!(r("integrate(1/x^(1/3),x,1,8)"), "9/2");
    assert_eq!(r("integrate(1/x^(2/5),x)"), "5/3·x^(3/5)");
    assert_eq!(r("integrate(1/x^(2/3),x)"), "3·x^(1/3)");
    assert_eq!(r("integrate(3/x^(1/2),x)"), "6·sqrt(x)");
    assert_eq!(r("diff(3/2*x^(2/3),x)"), "x^(-1/3)");
    // Integer-exponent quotients keep their existing (unfolded) path.
    assert_eq!(r("integrate(1/x,x)"), "ln(|x|)");
    assert_eq!(r("integrate(x^3/x,x)"), "1/3·x^3");
}

#[test]
fn test_eval_fractional_binomial_taylor_at_zero() {
    // `taylor((1+x)^α, x, 0, n)` for a fractional α declined at center 0 (the analytic Maclaurin
    // engine has no binomial-series case), although the SAME expansion works at a nonzero center.
    // Falling back to the definition-by-differentiation method at 0 now produces the binomial series.
    // The coefficients are the exact generalized binomials C(α, k).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // sqrt(1+x) = 1 + x/2 - x^2/8 + x^3/16 - 5x^4/128.
    assert_eq!(
        r("taylor(sqrt(1+x),x,0,4)"),
        "1/128·(8·x^3 + 64·x + 128 - 5·x^4 - 16·x^2)"
    );
    // 1/sqrt(1+x) = 1 - x/2 + 3x^2/8 - 5x^3/16.
    assert_eq!(
        r("taylor(1/sqrt(1+x),x,0,3)"),
        "1/2·(3/4·x^2 + 2 - 5/8·x^3 - x)"
    );
    // (1+x)^(1/3) = 1 + x/3 - x^2/9 + 5x^3/81.
    assert_eq!(
        r("taylor((1+x)^(1/3),x,0,3)"),
        "1/9·(5/9·x^3 + 3·x + 9 - x^2)"
    );
    // The analytic-engine cases keep their canonical Maclaurin forms (tried first).
    assert_eq!(
        r("taylor(exp(x),x,0,4)"),
        "1/24·(x^4 + 4·x^3 + 12·x^2 + 24·x + 24)"
    );
    assert_eq!(
        r("taylor(log(1+x),x,0,4)"),
        "1/12·(4·x^3 + 12·x - 3·x^4 - 6·x^2)"
    );
    // The 2-argument form `taylor(f, x)` / `series(f, x)` defaults to a Maclaurin expansion of
    // the default order (6) — the most natural invocation, previously an "undefined" arity error.
    assert_eq!(
        r("taylor(exp(x),x)"),
        "1/720·(x^6 + 6·x^5 + 30·x^4 + 120·x^3 + 360·x^2 + 720·x + 720)"
    );
    assert_eq!(r("taylor(exp(x),x)"), r("taylor(exp(x),x,6)"));
    assert_eq!(
        r("series(1/(1-x),x)"),
        "x^6 + x^5 + x^4 + x^3 + x^2 + x + 1"
    );
    // Textbook / SymPy / Mathematica command aliases parse to the canonical command.
    assert_eq!(r("Taylor(exp(x),x,4)"), r("taylor(exp(x),x,4)"));
    assert_eq!(r("Series(sin(x),x)"), r("series(sin(x),x)"));
    assert_eq!(r("Sum(k,k,1,n)"), r("sum(k,k,1,n)"));
    assert_eq!(r("summation(k^2,k,1,n)"), r("sum(k^2,k,1,n)"));
    assert_eq!(r("prod(k,k,1,5)"), r("product(k,k,1,5)"));
}

#[test]
fn test_eval_periodic_trig_equation_with_outside_coefficient_emits_full_family() {
    // SOUNDNESS: an OUTSIDE coefficient/offset (`2·sin x = 1`, `2·cos x + 1 = 0`) left the trig side a
    // `Mul`/`Add` that the bare-trig detector could not see, so the equation fell through to the
    // unary-inverse path and returned only the PRINCIPAL value (`{π/6}`) — an incomplete solution set
    // presented as complete, with ok=true and no warning. Normalising `A·trig(a·x)+B=C` to
    // `trig(a·x)=(C−B)/A` before detection now routes it through the full `Periodic` generator.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Outside coefficient -> the SAME family the bare `trig=c'` form yields.
    assert_eq!(
        r("solve(2*sin(x)=1, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(2*cos(x)+1=0, x)"),
        "{ 2/3·pi + k·2·pi, 4/3·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(3*tan(x)=3, x)"), "{ 1/4·pi + k·pi : k ∈ ℤ }");
    assert_eq!(r("solve(5*sin(x)=5, x)"), "{ 1/2·pi + k·2·pi : k ∈ ℤ }"); // c=1 single family
    assert_eq!(
        r("solve(3*sin(x)=1, x)"),
        "{ arcsin(1/3) + k·2·pi, pi - arcsin(1/3) + k·2·pi : k ∈ ℤ }"
    );
    // Negative coefficient (sign folds into c), additive offset, and scaled argument all work.
    assert_eq!(
        r("solve(-2*sin(x)=1, x)"),
        "{ -1/6·pi + k·2·pi, 7/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(2*sin(x)+1=2, x)"),
        "{ 1/6·pi + k·2·pi, 5/6·pi + k·2·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(2*sin(2*x)=1, x)"),
        "{ 1/12·pi + k·pi, 5/12·pi + k·pi : k ∈ ℤ }"
    );
    // SOUNDNESS edges: out-of-range stays empty; c=±1 single family.
    assert_eq!(r("solve(2*sin(x)=3, x)"), "No solution");
    assert_eq!(r("solve(2*sin(x)=2, x)"), "{ 1/2·pi + k·2·pi : k ∈ ℤ }");
    // SQUARED trig with an outside coefficient: `A·trig(arg)^2=c` folds to `trig(arg)^2=c/A` so the
    // double-angle reduction runs (previously `4·cos²x=1` dropped the `+kπ` and returned `{π/3, 2π/3}`).
    assert_eq!(
        r("solve(4*cos(x)^2=1, x)"),
        "{ 1/3·pi + k·pi, 2/3·pi + k·pi : k ∈ ℤ }"
    );
    assert_eq!(
        r("solve(4*sin(x)^2=1, x)"),
        "{ 1/6·pi + k·pi, 5/6·pi + k·pi : k ∈ ℤ }"
    );
    assert_eq!(r("solve(2*cos(x)^2=1, x)"), "{ 1/4·pi + k·1/2·pi : k ∈ ℤ }");
    assert_eq!(r("solve(3*sin(x)^2=3, x)"), "{ 1/2·pi + k·pi : k ∈ ℤ }"); // sin²=1 single family
    assert_eq!(r("solve(4*cos(x)^2=5, x)"), "No solution"); // cos²=5/4 > 1
}

#[test]
fn test_eval_variable_base_log_inequality_declines() {
    // SOUNDNESS: `log(x, c) ≷ k` reads x as the BASE, so logₓ(c)=ln(c)/ln(x) is NON-monotonic
    // (decreasing on x>1, sign change at x=1). The engine's monotonic isolation emitted a wrong ray
    // (and an `undefined` endpoint for k=0). With no exact split representation it now declines to an
    // honest residual (ok=true). Constant-base log and equations are unaffected.
    let run = |input: &str| -> (bool, String) {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        (
            wire["ok"].as_bool().unwrap_or(false),
            wire["result"].as_str().unwrap_or("").to_string(),
        )
    };
    // Variable-base log inequalities decline to a residual (ok=true), never a wrong ray / "undefined".
    for input in [
        "log(x,2)>3",
        "log(x,4)>2",
        "log(x,3)<1",
        "log(x,1/2)>1",
        "log(x,2)>0",
    ] {
        let (ok, result) = run(input);
        assert!(
            ok,
            "{input} should be ok=true (honest residual), got {result:?}"
        );
        assert!(
            result.contains("solve(") && !result.contains("undefined"),
            "{input} should be a clean residual, got {result:?}"
        );
    }
    let plain = |input: &str| run(input).1;
    // EQ-safety: equations still solve.
    assert_eq!(plain("log(x,2)=3"), "{ 2^(1/3) }");
    // Constant-base log (monotonic) is unaffected.
    assert_eq!(plain("log(2,x)>3"), "(8, infinity)");
    assert_eq!(plain("log(2,x)<3"), "(0, 8)");
    assert_eq!(plain("ln(x)<0"), "(0, 1)");
}

#[test]
fn test_eval_trig_inequality_out_of_range() {
    // SOUNDNESS: `sin(x)`/`cos(x)` ≷ c with c PROVABLY outside [-1, 1] is ℝ or ∅, not the finite ray
    // (sometimes with a non-real `arcsin(c)` endpoint) the generic monotonic inversion produced.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("cos(x)<=1"), "All real numbers"); // cos ≤ 1 always
    assert_eq!(r("cos(x)>1"), "No solution"); // cos > 1 never
    assert_eq!(r("sin(x)>2"), "No solution"); // out of range, no non-real arcsin(2) endpoint
    assert_eq!(r("cos(x)<-1"), "No solution");
    assert_eq!(r("sin(x)<2"), "All real numbers");
    assert_eq!(r("cos(x)>=-1"), "All real numbers"); // cos ≥ -1 always
    assert_eq!(r("sin(x)>=2"), "No solution");
    // Controls: an in-range threshold now SOLVES exactly (cycle P2 PeriodicIntervalUnion; the old
    // `(1/6·pi, infinity)` ray was unsound: sin(x)>1/2 is false at x=pi, which lies in that ray).
    // Equations are unchanged.
    assert_eq!(
        r("sin(x)>1/2"),
        "{ (1/6·pi + k·2·pi, 5/6·pi + k·2·pi) : k ∈ ℤ }"
    );
    assert_eq!(r("cos(x)=2"), "No solution");
    assert_eq!(
        r("sin(x)=1/3"),
        "{ arcsin(1/3) + k·2·pi, pi - arcsin(1/3) + k·2·pi : k ∈ ℤ }"
    );
}

#[test]
fn test_eval_exponential_positivity_inequality() {
    // SOUNDNESS: `b^x {>,>=} c` with a positive base and c <= 0 is identically TRUE (b^x > 0 always),
    // so the solution is ℝ — not the empty set the op-agnostic EmptySet classification produced. The
    // product/sum cascade self-heals via AllReals ∩ s = s.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("e^x>0"), "All real numbers");
    assert_eq!(r("e^x>=0"), "All real numbers");
    assert_eq!(r("e^x>-1"), "All real numbers");
    assert_eq!(r("2^x>0"), "All real numbers");
    assert_eq!(r("(1/2)^x>0"), "All real numbers");
    assert_eq!(r("x*e^x>0"), "(0, infinity)"); // sign(x·e^x) = sign(x)
    assert_eq!(r("x^2*e^x>0"), "(-infinity, 0) U (0, infinity)"); // ℝ∖{0}
                                                                  // Controls: `<`/`<=`/`=` against c <= 0 stay empty (b^x is never <0, ≤0, or =0); rhs > 0 solves normally.
    assert_eq!(r("e^x<0"), "No solution");
    assert_eq!(r("e^x<=0"), "No solution");
    assert_eq!(r("e^x=0"), "No solution");
    assert_eq!(r("e^x>5"), "(ln(5), infinity)");
    assert_eq!(r("e^x-1>0"), "(0, infinity)");
    // SOUNDNESS: a bare additive single-exponential `a*base^x + c {op} k` was declined by the
    // substitution gate (only `base^x` with no higher power) and fell to the boundary equation,
    // returning "No solution" (or a malformed conditional) when the isolated threshold is negative —
    // truth is all reals since base^x > 0. Now it substitutes u=base^x and the u>0 clamp answers it.
    assert_eq!(r("e^x+1>0"), "All real numbers");
    assert_eq!(r("e^x+5>2"), "All real numbers");
    assert_eq!(r("3^x+1>0"), "All real numbers");
    assert_eq!(r("e^x+1>=0"), "All real numbers");
    assert_eq!(r("2*e^x+3>0"), "All real numbers");
    assert_eq!(r("e^x+1<0"), "No solution"); // base^x = -1 has no solution, so < never holds
                                             // Equation narration is unchanged (the bare gate is inequality-only).
    assert_eq!(r("e^x=2"), "{ ln(2) }");
    assert_eq!(r("e^x+1=0"), "No solution");
}

#[test]
fn test_eval_mixed_base_exponential_normalizes_to_common_prime() {
    // Terms with DIFFERENT integer bases that share a common prime (`4^x` and `2^x`, `9^x` and `3^x`)
    // used to error with "Cannot isolate: variable on both sides". They are now rewritten to the common
    // prime base (`4^x → 2^(2x)`), making the relation a polynomial in the single atom `p^x`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `u = 2^x`: `u² - 3u + 2 = 0 ⟹ u ∈ {1, 2} ⟹ x ∈ {0, 1}`.
    assert_eq!(r("solve(4^x - 3*2^x + 2 = 0, x)"), "{ 0, 1 }");
    assert_eq!(r("solve(9^x - 4*3^x + 3 = 0, x)"), "{ 0, 1 }");
    assert_eq!(r("solve(4^x - 5*2^x + 4 = 0, x)"), "{ 0, 2 }");
    // A branch out of range (`2^x = -1`) is dropped.
    assert_eq!(r("solve(4^x - 2^x - 2 = 0, x)"), "{ 1 }");
    // The inequality form normalizes too.
    assert_eq!(r("solve(4^x - 3*2^x + 2 < 0, x)"), "(0, 1)");
    // Three bases sharing the prime 2 (`8=2³, 4=2², 2=2¹`), a cubic in `2^x`.
    assert_eq!(r("solve(8^x - 6*4^x + 8*2^x = 0, x)"), "{ 1, 2 }");
    // Controls: a single base (already handled), base e (non-integer), and INCOMPATIBLE primes decline.
    assert_eq!(r("solve(2^(2*x) - 3*2^x + 2 = 0, x)"), "{ 0, 1 }");
    assert_eq!(r("solve(e^(2*x) - 3*e^x + 2 = 0, x)"), "{ ln(2), 0 }");
    assert_eq!(r("solve(2^x = 8, x)"), "{ 3 }");
}

#[test]
fn test_eval_two_different_base_exponential_divides_to_a_log() {
    // Two exponentials with DIFFERENT (incompatible-prime) bases: `A·M^x + B·N^x = 0 ⟺ (M/N)^x = −B/A`,
    // i.e. `x = ln(−B/A)/ln(M/N)`. The A=B forms happened to isolate, but the one-sided
    // (`4^x − 9^x = 0`) and both-coefficiented (`5·2^x = 3^x`) forms errored with "Cannot isolate 'x'".
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `(4/9)^x = 1 ⟹ x = 0` (the ratio is 1, so `ln(1) = 0`).
    assert_eq!(r("solve(4^x - 9^x = 0, x)"), "{ 0 }");
    assert_eq!(r("solve(2^x - 5^x = 0, x)"), "{ 0 }");
    // Non-unit coefficients ⟹ a genuine log. `ln(3/2)/ln(4/9)` folds to the
    // exact rational −1/2 (`4/9 = (2/3)²`, `3/2 = (2/3)⁻¹`); a truly irrational
    // ratio (`ln(1/5)/ln(2/3)`, distinct primes) stays symbolic.
    assert_eq!(r("solve(2*4^x = 3*9^x, x)"), "{ -1/2 }");
    assert_eq!(r("solve(5*2^x = 3^x, x)"), "{ ln(1/5) / ln(2/3) }");
    // SOUNDNESS: `(M/N)^x > 0`, so a non-positive ratio has no real solution.
    assert_eq!(r("solve(4^x + 9^x = 0, x)"), "No solution");
    assert_eq!(r("solve(2^x = -3^x, x)"), "No solution");
    // Controls: same-base polynomial forms and a nonzero-constant RHS are NOT this shape.
    assert_eq!(r("solve(4^x - 3*2^x + 2 = 0, x)"), "{ 0, 1 }");
    assert_eq!(r("solve(2^x = 8, x)"), "{ 3 }");
}

#[test]
fn test_eval_exponential_reciprocal_polynomial_clears_the_reciprocal() {
    // Equations that mix an exponential with its RECIPROCAL (`e^x + e^(−x)`, the hyperbolic form) used
    // to bail — `función [cosh] no definida` for base `e`, `Cannot isolate 'x'` for general bases —
    // because `simplify` folds `e^x + e^(−x)` into `2·cosh(x)`. The Laurent map `u = base^x` (built on
    // the raw tree, so `simplify` never runs) clears the `1/u` and solves the polynomial in `u`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `u = base^x`: `u² − 2u + 1 = 0 ⟹ u = 1 ⟹ x = 0`.
    assert_eq!(r("solve(e^x + e^(-x) = 2, x)"), "{ 0 }");
    assert_eq!(r("solve(3^x + 3^(-x) = 2, x)"), "{ 0 }");
    // `u² − 1 = 0 ⟹ u = 1` (the `u = −1` root is dropped: `base^x > 0`).
    assert_eq!(r("solve(e^x - e^(-x) = 0, x)"), "{ 0 }");
    // Distinct positive roots: `u² − 3u + 2 = 0 ⟹ u ∈ {1, 2}`.
    assert_eq!(r("solve(e^x + 2*e^(-x) = 3, x)"), "{ 0, ln(2) }");
    // An affine exponent (`2^(1−x) = 2·2^(−x)`) folds the `2` into the coefficient.
    assert_eq!(r("solve(2^x - 3 + 2^(1-x) = 0, x)"), "{ 0, 1 }");
    // `2^x + 2^(−x) = 5/2 ⟹ u ∈ {1/2, 2} ⟹ x ∈ {−1, 1}` (`ln(1/2)/ln(2)` folds to −1).
    assert_eq!(r("solve(2^x + 2^(-x) = 5/2, x)"), "{ -1, 1 }");
    // `cosh(x) ≥ 1` always, so `= 1/2·2 = 1` (i.e. sum `= 1`) has NO real solution.
    assert_eq!(r("solve(e^x + e^(-x) = 1, x)"), "No solution");
    // SURD-discriminant roots: BOTH `u = 2 ± √3` are provably positive, so both back-substitute. The
    // exact-surd-sign upgrade to the positivity prover keeps the second root (it used to drop it behind
    // a spurious `2 − √3 > 0` guard).
    assert_eq!(
        r("solve(e^x + e^(-x) = 4, x)"),
        "{ ln(2 - sqrt(3)), ln(sqrt(3) + 2) }"
    );
    // `u² − 2u − 1 = 0 ⟹ u = 1 ± √2`; the negative `1 − √2` is now DISPROVEN positive ⟹ dropped.
    assert_eq!(r("solve(e^x - e^(-x) = 2, x)"), "{ ln(sqrt(2) + 1) }");
    assert_eq!(
        r("solve(e^x + e^(-x) = 3, x)"),
        "{ ln(1/2·(3 - sqrt(5))), ln(1/2·(sqrt(5) + 3)) }"
    );
    // Controls: the pure positive-power forms are owned by the existing path and must be UNCHANGED.
    assert_eq!(r("solve(e^(2*x) - 3*e^x + 2 = 0, x)"), "{ ln(2), 0 }");
    assert_eq!(r("solve(4^x - 3*2^x + 2 = 0, x)"), "{ 0, 1 }");
}

#[test]
fn test_eval_fractional_base_exponential_inequality_direction() {
    // SOUNDNESS: `a^x ≷ k` with 0 < a < 1 (decreasing) must FLIP the inequality direction when
    // isolating x through the logarithm. Previously the engine kept the direction, returning the
    // reversed (wrong) ray. The bound is exact; only the direction was wrong. Truth vs sympy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // (1/2)^x > 4 ⟺ x < -2. The change-of-base boundary `ln(c)/ln(b)` folds to
    // the exact rational for these fractional-base/argument pairs.
    assert_eq!(r("(1/2)^x>4"), "(-infinity, -2)");
    assert_eq!(r("(1/2)^x<1/4"), "(2, infinity)"); // x > 2
    assert_eq!(r("(1/2)^x>=2"), "(-infinity, -1]"); // x <= -1
    assert_eq!(r("0.3^x<0.09"), "(2, infinity)"); // x > 2 (9/100 = (3/10)²)
    assert_eq!(r("(1/3)^x>1/9"), "(-infinity, 2)"); // x < 2
                                                    // Controls: base > 1 keeps direction; equations are never flipped.
    assert_eq!(r("2^x>4"), "(2, infinity)");
    assert_eq!(r("2^x<4"), "(-infinity, 2)");
    assert_eq!(r("2^x>=8"), "[3, infinity)");
    assert_eq!(r("(1/2)^x=4"), "{ -2 }");
    assert_eq!(r("2^x=4"), "{ 2 }");
    // SOUNDNESS: an ADDITIVE/scaled single exponential `a*base^x + c {op} k` is isolated to the pure
    // `base^x {op'} (k-c)/a` and solved by the terminal for EVERY base — including a fractional base
    // with a positive threshold (`(1/2)^x - 4 > 0 -> (1/2)^x > 4`) or a negative threshold
    // (`(1/2)^x + 1 > 0 -> (1/2)^x > -1 -> all reals`). The substitution path would decline a
    // fractional base to a residual, so the isolation runs first.
    assert_eq!(r("(1/2)^x-4>0"), "(-infinity, -2)"); // x < -2
    assert_eq!(r("(1/2)^x-1>0"), "(-infinity, 0)");
    assert_eq!(r("(1/2)^x+1>0"), "All real numbers");
    assert_eq!(r("(1/2)^x+1<0"), "No solution");
    assert_eq!(r("(1/3)^x-1>0"), "(-infinity, 0)");
}

#[test]
fn test_eval_exponential_polynomial_inequality_back_substitution() {
    // SOUNDNESS (B3): a polynomial-in-`u = e^x` INEQUALITY was solved in u-space and the interval was
    // returned WITHOUT back-substituting `x = ln(u)` (the equation path back-substituted, the
    // inequality path forgot). `e^(2x)-3e^x+2<0` leaked the u-interval `(1, 2)` instead of `(0, ln 2)`.
    // The fix clamps the u-solution to `u > 0` (range of e^x) and maps each endpoint through ln.
    // Truth verified vs sympy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // u = e^x in (1, 2) ⟺ x in (0, ln 2). All four operators, base e.
    assert_eq!(r("e^(2*x)-3*e^x+2<0"), "(0, ln(2))");
    assert_eq!(r("e^(2*x)-3*e^x+2>0"), "(-infinity, 0) U (ln(2), infinity)");
    assert_eq!(r("e^(2*x)-3*e^x+2<=0"), "[0, ln(2)]");
    assert_eq!(
        r("e^(2*x)-3*e^x+2>=0"),
        "(-infinity, 0] U [ln(2), infinity)"
    );
    // A base > 1 other than e maps through log_base: 2^x in [1, 2] ⟺ x in [0, 1].
    assert_eq!(r("2^(2*x)-3*2^x+2<=0"), "[0, 1]");
    // u must be > 0: a root <= 0 is clamped away. u in (-2, 1) ⟺ (0, 1) ⟺ x < 0; u in (-2, -1) ⟺ empty.
    assert_eq!(r("e^(2*x)+e^x-2<0"), "(-infinity, 0)");
    assert_eq!(r("e^(2*x)+3*e^x+2<0"), "No solution");
    // U3: the IRRATIONAL roots (e^x = (1±√5)/2) now back-substitute exactly —
    // the surd sign oracles classify the endpoints ((1−√5)/2 clamps away as
    // provably negative; (1+√5)/2 = φ maps through the boundary equation to
    // ln(φ)). Previously an honest decline.
    assert_eq!(r("e^(2*x)-e^x-1<0"), "(-infinity, ln(phi))");
    assert_eq!(r("e^(2*x)-e^x-1>0"), "(ln(phi), infinity)");
    // A FRACTIONAL base (0 < a < 1) likewise declines to the residual (decreasing inverse + ln-ratio
    // bounds the downstream interval comparison cannot order) rather than leak the u-interval.
    assert_eq!(
        r("(1/2)^(2*x)-3*(1/2)^x+2<0"),
        "solve((1/2)^(2·x) + 2 - 3·(1/2)^x < 0, x)"
    );
    // Controls: the equation path still back-substitutes; e^(2x) = -5 has no real solution.
    assert_eq!(r("e^(2*x)-3*e^x+2=0"), "{ ln(2), 0 }");
    assert_eq!(r("e^(2*x)=-5"), "No solution");
}

#[test]
fn test_eval_exponential_coefficient_equals_base_inequality() {
    // SOUNDNESS: when the linear coefficient equals the base, the simplifier merges
    // `c·base^x = base^(x+1)`, and the exponential substitution could not match the `Add`-in-exponent
    // `base^(x+1)`. The strategy declined and the fallback returned the EQUATION root, dropping the
    // operator: `2^(2x)-2·2^x<0` -> `{1}` instead of `(-inf, 1)`. Now `substitute_expr_pattern` maps the
    // affine exponent `base^(x+1) -> base^1·u` (numeric base, integer constant), so the inequality solves
    // and back-substitutes correctly. Truth vs sympy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // u = base^x in (0, base) <=> base^x < base <=> x < 1. All four operators, bases 2/3/10.
    assert_eq!(r("2^(2*x)-2*2^x<0"), "(-infinity, 1)");
    assert_eq!(r("2^(2*x)-2*2^x>0"), "(1, infinity)");
    assert_eq!(r("2^(2*x)-2*2^x<=0"), "(-infinity, 1]");
    assert_eq!(r("2^(2*x)-2*2^x>=0"), "[1, infinity)");
    assert_eq!(r("3^(2*x)-3*3^x<0"), "(-infinity, 1)");
    assert_eq!(r("10^(2*x)-10*10^x<0"), "(-infinity, 1)");
    // The affine-exponent substitution also drives the equation form: 2^(x+1) = 8 <=> 2·u = 8 <=> x = 2.
    assert_eq!(r("2^(2*x)-2*2^x=0"), "{ 1 }");
    assert_eq!(r("2^(x+1)=8"), "{ 2 }");
    // Controls: a coefficient that is NOT the base does not merge, so the inner base^x substitutes
    // as before (2^(2x)-4·2^x stays a clean u^2-4u): boundary x=2, not 1.
    assert_eq!(r("2^(2*x)-4*2^x<0"), "(-infinity, 2)");
}

#[test]
fn test_eval_factorable_exponential_inequality() {
    // SOUNDNESS (peldaño 1): a degree-2 exponential inequality collapsed to one side with NO constant
    // term, `A·base^(2x) + B·base^x {op} 0`, factors out base^x > 0 to the single exponential
    // `base^x {op} -B/A`. For base e the coefficient merges (`e·e^x = e^(x+1)`) so the substitution
    // was blocked and the fallback leaked the equation root `{1}`; for a SYMBOLIC coefficient the
    // polynomial-in-u inequality solver errored (ok=false). Both now reduce and solve. Truth vs sympy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // base e, coefficient == base: e^x in (0, e) <=> x < 1.
    assert_eq!(r("e^(2*x)-e*e^x<0"), "(-infinity, 1)");
    assert_eq!(r("e^(2*x)-e*e^x>0"), "(1, infinity)");
    assert_eq!(r("e^(2*x)-e*e^x<=0"), "(-infinity, 1]");
    assert_eq!(r("e^(2*x)-e*e^x>=0"), "[1, infinity)");
    // SYMBOLIC coefficient pi (was a loud ok=false "symbolic coefficient" error): e^x < pi <=> x < ln(pi).
    assert_eq!(r("e^(2*x)-pi*e^x<0"), "(-infinity, ln(pi))");
    assert_eq!(r("e^(2*x)-pi*e^x>0"), "(ln(pi), infinity)");
    assert_eq!(r("e^(2*x)-2*pi*e^x<0"), "(-infinity, ln(2·pi))");
    // Controls: a constant term keeps the substitution path (B3), and the equation is unchanged.
    assert_eq!(r("e^(2*x)-3*e^x+2<0"), "(0, ln(2))");
    assert_eq!(r("e^(2*x)-e*e^x=0"), "{ 1 }");
}

#[test]
fn test_eval_nonunit_exponent_exponential_inequality() {
    // SOUNDNESS: a single exponential with a NON-UNIT integer exponent, `base^(k*x) {op} c`, could not
    // be isolated by the unit-exponent terminal (`e^(2x)<2` -> residual, `e^(2x)<e` -> ok=false). Since
    // `base^(k*x)` (base>1) is strictly increasing, recover the ray from the boundary EQUATION
    // `base^(k*x)=c` (which solves) + monotonicity. This also closes the degree-3+ inequality: the
    // factor-out cofactor `e^(2x)-e` of `e^(3x)-e*e^x<0` resolves here. Truth vs sympy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Non-unit single exponential, base e: e^(2x) {op} c <=> 2x {op} ln(c) <=> x {op} ln(c)/2.
    assert_eq!(r("e^(2*x)<e"), "(-infinity, 1/2)");
    assert_eq!(r("e^(2*x)>e"), "(1/2, infinity)");
    assert_eq!(r("e^(2*x)<=e"), "(-infinity, 1/2]");
    assert_eq!(r("e^(2*x)<2"), "(-infinity, 1/2·ln(2))");
    // Non-positive threshold resolves by sign (base^(k*x) > 0 always); positivity stays correct.
    assert_eq!(r("e^(2*x)<0"), "No solution");
    assert_eq!(r("e^(2*x)>0"), "All real numbers");
    assert_eq!(r("e^(2*x)>1"), "(0, infinity)");
    // Degree-3 (and degree-4) collapsed: factor out e^x>0 to the non-unit cofactor.
    assert_eq!(r("e^(3*x)-e*e^x<0"), "(-infinity, 1/2)"); // was the WRONG point {1/2}
    assert_eq!(r("e^(3*x)-e*e^x>0"), "(1/2, infinity)");
    assert_eq!(r("e^(3*x)-e*e^x<=0"), "(-infinity, 1/2]");
    assert_eq!(r("e^(3*x)-pi*e^x<0"), "(-infinity, 1/2·ln(pi))");
    assert_eq!(r("e^(4*x)-e*e^x<0"), "(-infinity, 1/3)");
    // SYMBOLIC-CONSTANT thresholds/coefficients beyond bare e/pi: the threshold sign is delegated to
    // the boundary equation (Discrete root -> ray, Empty -> sign), so e^2, sqrt(2), 2*e all solve; a
    // provably non-positive threshold (-e) resolves by sign.
    assert_eq!(r("e^(2*x)<e^2"), "(-infinity, 1)");
    assert_eq!(r("e^(2*x)>e^2"), "(1, infinity)");
    assert_eq!(r("e^(2*x)<e^3"), "(-infinity, 3/2)");
    assert_eq!(r("e^(3*x)-e^2*e^x>0"), "(1, infinity)"); // was the WRONG point {1}
    assert_eq!(r("e^(3*x)-e^2*e^x<0"), "(-infinity, 1)");
    assert_eq!(r("e^(3*x)+e*e^x<0"), "No solution"); // e^x(e^(2x)+e) > 0 always
    assert_eq!(r("e^(3*x)+e*e^x>0"), "All real numbers");
    assert_eq!(r("e^(3*x)+pi*e^x<0"), "No solution");
    // Controls: a degree-3 with RATIONAL roots stays on the substitution path; equations unchanged.
    assert_eq!(r("e^(3*x)-e^x<0"), "(-infinity, 0)");
    assert_eq!(r("e^(3*x)-e*e^x=0"), "{ 1/2 }");
}

#[test]
fn test_eval_rational_sum_inequality_routing() {
    // SOUNDNESS regression: a rational-SUM inequality `x + c/x {op} k` (LHS an Add containing a
    // rational term) used to skip the reliable rational path and have its operator dropped, returning
    // the empty set (strict) or a degenerate point (non-strict). Now the LHS is combined into a single
    // fraction N/D and routed through the verified rational path. Truth cross-checked vs sympy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(x + 1/x > 2, x)"), "(0, 1) U (1, infinity)");
    assert_eq!(r("solve(x + 1/x < 2, x)"), "(-infinity, 0)");
    assert_eq!(r("solve(x + 1/x >= 2, x)"), "(0, infinity)");
    assert_eq!(r("solve(x + 2/x > 3, x)"), "(0, 1) U (2, infinity)");
    assert_eq!(r("solve(2*x + 1/x > 1, x)"), "(0, infinity)");
    assert_eq!(r("solve(x + 4/x >= 4, x)"), "(0, infinity)");
    assert_eq!(r("solve(x - 2 + 1/x > 0, x)"), "(0, 1) U (1, infinity)");
    assert_eq!(r("solve(x + 3 + 2/x > 0, x)"), "(-2, -1) U (0, infinity)");
    assert_eq!(r("solve(3*x + 12/x > 12, x)"), "(0, 2) U (2, infinity)");
    assert_eq!(r("solve(x + 9/x > 6, x)"), "(0, 3) U (3, infinity)");
    assert_eq!(r("solve(x + 1/(x-1) > 2, x)"), "(1, infinity)");
    assert_eq!(r("solve(2*x + 3/(x-1) > 5, x)"), "(1, infinity)");
    // Surd bounds: x + 1/x >= 3  ⟹  (0, (3-√5)/2] ∪ [(3+√5)/2, ∞).
    assert_eq!(
        r("solve(x + 1/x >= 3, x)"),
        "(0, 1/2·(3 - sqrt(5))] U [1/2·(sqrt(5) + 3), infinity)"
    );
    // Non-strict touch-point cases: the solution is a half-line PLUS the isolated touch point, which
    // requires unioning a Discrete point with a Continuous interval (previously the interval was
    // silently dropped, collapsing the answer to the lone point `[p, p]`).
    assert_eq!(r("solve(x + 1/x <= 2, x)"), "(-infinity, 0) U [1, 1]");
    assert_eq!(r("solve(x + 4/x <= 4, x)"), "(-infinity, 0) U [2, 2]");
    assert_eq!(r("solve(x + 9/x <= 6, x)"), "(-infinity, 0) U [3, 3]");
    // Controls: the single-fraction form and ordinary inequalities are unchanged.
    assert_eq!(r("solve((x^2+1)/x > 2, x)"), "(0, 1) U (1, infinity)");
    assert_eq!(r("solve(1/x < 1, x)"), "(-infinity, 0) U (1, infinity)");
    assert_eq!(r("solve(x^2 > 4, x)"), "(-infinity, -2) U (2, infinity)");
}

#[test]
fn test_eval_derangement_isperfect_harmonic() {
    // derangement(n)/subfactorial (permutations with no fixed point), isperfect(n) (σ(n)=2n, 1/0 —
    // the engine has no boolean), and harmonic(n) = Σ_{k=1}^n 1/k (exact rational). All BigInt/
    // BigRational exact; isperfect reuses the same divisor-sum core as sigma.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("derangement(0)"), "1");
    assert_eq!(r("derangement(1)"), "0");
    assert_eq!(r("derangement(4)"), "9");
    assert_eq!(r("derangement(5)"), "44");
    assert_eq!(r("subfactorial(4)"), "9"); // alias
    assert_eq!(r("derangement(-1)"), "derangement(-1)"); // honest residual
    assert_eq!(r("isperfect(6)"), "1");
    assert_eq!(r("isperfect(28)"), "1");
    assert_eq!(r("isperfect(496)"), "1");
    assert_eq!(r("isperfect(12)"), "0");
    assert_eq!(r("isperfect(1)"), "0"); // 1 is not perfect (σ(1)=1)
    assert_eq!(r("harmonic(1)"), "1");
    assert_eq!(r("harmonic(4)"), "25/12");
    assert_eq!(r("harmonic(5)"), "137/60");
    // Control: sigma (which now shares the divisor-sum core) is unchanged.
    assert_eq!(r("sigma(28)"), "56");
}

#[test]
fn test_eval_limit_abs_finite_tail_at_infinity() {
    // `lim_{x→∞} |u(x)| = |L|` when the rational argument has a finite tail L — previously only the
    // divergent case (`abs → +∞`) was handled, so `|(x-1)/(x+1)|` stayed an unevaluated residual.
    // Composing through `ln` (`lim ln(|u|) = ln(|L|)`) is what an improper rational integral with a
    // log antiderivative needs at its infinite bound.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("limit(abs((x-1)/(x+1)), x, inf)"), "1");
    assert_eq!(r("limit(abs((2*x+1)/(x-3)), x, inf)"), "2");
    assert_eq!(r("limit(ln(abs((x-1)/(x+1))), x, inf)"), "0");
    assert_eq!(r("limit(ln(abs((3*x+1)/(x+1))), x, inf)"), "ln(3)");
    // Improper integral unlocked by the composition: ∫₁^∞ 1/(x(x+1)) = ln 2.
    assert_eq!(r("integrate(1/(x*(x+1)), x, 1, inf)"), "-ln(1/2)");
    // Controls: a divergent abs still → ∞, a finite-point abs is unchanged, plain ln/sqrt unaffected.
    assert_eq!(r("limit(abs(x^2-x), x, inf)"), "infinity");
    assert_eq!(r("limit(abs(x-3), x, 5)"), "2");
    assert_eq!(r("limit(ln(x^2+1), x, inf)"), "infinity");
    assert_eq!(r("limit(sqrt((x^2+1)/x^2), x, inf)"), "1");
}

#[test]
fn test_eval_matrix_eigenvectors_rational() {
    // `eigenvectors(A)` (capstone of the linear-algebra core) returns, for each distinct RATIONAL
    // eigenvalue, the null-space basis of A−λI by exact rational RREF — rows are the eigenvectors.
    // Verified elsewhere by A·v = λ·v. A defective matrix yields fewer vectors (geometric
    // multiplicity); surd / complex / symbolic spectra decline to honest residuals.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("eigenvectors([[2,1],[1,2]])"), "[[1, 1], [-1, 1]]");
    assert_eq!(r("eigenvectors([[2,0],[0,3]])"), "[[0, 1], [1, 0]]");
    // Defective matrix (Jordan block): repeated eigenvalue 1 with a SINGLE eigenvector.
    assert_eq!(r("eigenvectors([[1,1],[0,1]])"), "[1, 0]");
    // Repeated eigenvalue with a full 2-D eigenspace plus a simple one.
    assert_eq!(
        r("eigenvectors([[5,4,2],[4,5,2],[2,2,2]])"),
        "[[-1, 1, 0], [-1/2, 0, 1], [2, 2, 1]]"
    );
    // Surd / complex / symbolic spectra → honest residual.
    assert_eq!(
        r("eigenvectors([[2,-1,0],[-1,2,-1],[0,-1,2]])"),
        "eigenvectors([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])"
    );
    assert_eq!(
        r("eigenvectors([[0,-1],[1,0]])"),
        "eigenvectors([[0, -1], [1, 0]])"
    );
    assert_eq!(
        r("eigenvectors([[a,b],[c,d]])"),
        "eigenvectors([[a, b], [c, d]])"
    );
}

#[test]
fn test_eval_matrix_rref() {
    // Reduced row echelon form was unimplemented. It now computes the exact RREF by Gauss-Jordan
    // over BigRational, with an honest residual for symbolic entries.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("rref([[1,2],[3,4]])"), "[[1, 0], [0, 1]]"); // full rank → identity
    assert_eq!(r("rref([[1,2],[2,4]])"), "[[1, 2], [0, 0]]"); // rank 1
    assert_eq!(
        r("rref([[1,2,3],[4,5,6],[7,8,9]])"),
        "[[1, 0, -1], [0, 1, 2], [0, 0, 0]]"
    );
    assert_eq!(r("rref([[2,4,6],[1,2,3]])"), "[[1, 2, 3], [0, 0, 0]]"); // pivot normalized
    assert_eq!(r("rref([[0,1],[1,0]])"), "[[1, 0], [0, 1]]"); // pivot swap
    assert_eq!(r("rref([[a,b],[c,d]])"), "rref([[a, b], [c, d]])"); // symbolic residual
}

#[test]
fn test_eval_matrix_eigenvalues_real() {
    // `eigenvalues(A)` was unimplemented. It now returns the REAL spectrum as the roots of the
    // characteristic polynomial: rational roots peeled exactly, a deflated quadratic closed by the
    // quadratic formula. A complex-conjugate pair (negative discriminant) declines to an honest
    // residual — this is a real-domain engine. Cross-checked against numpy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Rational spectra.
    assert_eq!(r("eigenvalues([[2,1],[1,2]])"), "[3, 1]");
    assert_eq!(r("eigenvalues([[1,1],[0,1]])"), "[1, 1]"); // repeated eigenvalue
    assert_eq!(r("eigenvalues([[5,4,2],[4,5,2],[2,2,2]])"), "[1, 10, 1]");
    // Rational root peeled, then a surd quadratic factor: 2, 2 ± √2.
    assert_eq!(
        r("eigenvalues([[2,-1,0],[-1,2,-1],[0,-1,2]])"),
        "[2, sqrt(2) + 2, 2 - sqrt(2)]"
    );
    // Complex spectrum (rotation) → honest residual in the real domain.
    assert_eq!(
        r("eigenvalues([[0,-1],[1,0]])"),
        "eigenvalues([[0, -1], [1, 0]])"
    );
    // Symbolic / non-square → honest residual.
    assert_eq!(
        r("eigenvalues([[a,b],[c,d]])"),
        "eigenvalues([[a, b], [c, d]])"
    );
}

#[test]
fn test_eval_matrix_charpoly() {
    // `charpoly(A) = det(λI − A)` was unimplemented. It now returns the monic characteristic
    // polynomial in `lambda`, for numeric and symbolic matrices, 2×2 and 3×3. (A bounded
    // budget exemption lets the cofactor expansion of a small numeric matrix commit instead of
    // being rejected by the anti-worsen node budget.)
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // [[2,1],[1,2]]: λ² − 4λ + 3 (eigenvalues 1, 3).
    assert_eq!(r("charpoly([[2,1],[1,2]])"), "lambda^2 + 3 - 4·lambda");
    // Diagonal 3×3 factors directly to (λ−1)(λ−2)(λ−3).
    assert_eq!(
        r("charpoly([[1,0,0],[0,2,0],[0,0,3]])"),
        "(lambda - 3)·(lambda - 2)·(lambda - 1)"
    );
    // Tridiagonal: λ³ − 6λ² + 10λ − 4 (trace 6, det 4).
    assert_eq!(
        r("charpoly([[2,-1,0],[-1,2,-1],[0,-1,2]])"),
        "lambda^3 + 10·lambda - 6·lambda^2 - 4"
    );
    // Symbolic 2×2: λ² − (a+d)λ + (ad − bc), kept in det form.
    assert_eq!(
        r("charpoly([[a,b],[c,d]])"),
        "(lambda - a)·(lambda - d) - b·c"
    );
    // Non-square stays an honest residual.
    assert_eq!(r("charpoly([[1,2,3]])"), "charpoly([1, 2, 3])");
}

#[test]
fn test_eval_wronskian() {
    // `wronskian([f₁,…,fₙ], x)` = det of the matrix of 0th…(n−1)th derivatives — the linear-
    // independence test. Reuses symbolic differentiation + determinant. A bounded budget exemption
    // lets the cofactor expansion commit.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("wronskian([sin(x),cos(x)], x)"), "-1");
    assert_eq!(r("wronskian([1,x,x^2], x)"), "2");
    assert_eq!(r("wronskian([1,x,x^2,x^3], x)"), "12"); // 0!·1!·2!·3!
    assert_eq!(r("wronskian([e^x,e^(2*x)], x)"), "e^(3·x)");
    assert_eq!(r("wronskian([x,x^2], x)"), "x^2");
    // Linearly DEPENDENT functions ⇒ Wronskian 0 (the key application).
    assert_eq!(r("wronskian([x,2*x], x)"), "0");
    assert_eq!(r("wronskian([sin(x),2*sin(x)], x)"), "0");
}

#[test]
fn test_eval_matrix_adjugate() {
    // `adjugate(A)` (alias `adj`) is the transpose of the cofactor matrix — a polynomial in the
    // entries, ALWAYS defined (no det≠0 condition), so it works symbolically too. Satisfies
    // A·adj(A) = det(A)·I (verified separately).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("adjugate([[1,2],[3,4]])"), "[[4, -2], [-3, 1]]");
    assert_eq!(r("adjugate([[a,b],[c,d]])"), "[[d, -b], [-c, a]]"); // symbolic
    assert_eq!(
        r("adjugate([[1,2,3],[0,1,4],[5,6,0]])"),
        "[[-24, 18, 5], [20, -15, -4], [-5, 4, 1]]"
    );
    // A·adj(A) = det(A)·I.
    assert_eq!(
        r("[[1,2],[3,4]] * adjugate([[1,2],[3,4]])"),
        "[[-2, 0], [0, -2]]"
    );
}

#[test]
fn test_eval_matrix_integer_power() {
    // `M^n` for an integer exponent: `n=0 → I`, `n=1 → M`, `|n|≥2` for an all-numeric square matrix
    // is repeated multiplication (negative ⇒ inverse powered), folding exactly. A bounded budget
    // exemption lets the unfolded products commit. Cross-checked against numpy.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("[[1,1],[0,1]]^3"), "[[1, 3], [0, 1]]");
    assert_eq!(r("[[1,1],[1,0]]^5"), "[[8, 5], [5, 3]]"); // Fibonacci
    assert_eq!(r("[[2,0],[0,3]]^2"), "[[4, 0], [0, 9]]");
    assert_eq!(
        r("[[1,2,0],[0,1,1],[0,0,1]]^4"),
        "[[1, 8, 12], [0, 1, 4], [0, 0, 1]]"
    );
    assert_eq!(r("[[1,2],[3,4]]^0"), "[[1, 0], [0, 1]]"); // M^0 = I
    assert_eq!(r("[[2,0],[0,2]]^(-2)"), "[[1/4, 0], [0, 1/4]]"); // negative power via inverse
                                                                 // Controls: a singular base to a negative power is undefined; symbolic power / inverse stay
                                                                 // honest residuals; a non-square base is undefined.
    assert_eq!(r("[[1,2],[2,4]]^(-1)"), "undefined");
    assert_eq!(r("[[a,b],[c,d]]^2"), "[[a, b], [c, d]]^2");
    assert_eq!(r("[[a,b],[c,d]]^(-1)"), "inverse([[a, b], [c, d]])");
    assert_eq!(r("[[1,2,3],[4,5,6]]^2"), "undefined");
}

#[test]
fn test_eval_matrix_rank_exact() {
    // Matrix rank was recognized-but-unimplemented (returned an error). It now computes the
    // exact rank by Gaussian elimination over BigRational, for any shape, with an honest
    // residual for symbolic entries.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("rank([[1,2],[2,4]])"), "1");
    assert_eq!(r("rank([[1,2],[3,4]])"), "2");
    assert_eq!(r("rank([[1,2,3],[4,5,6],[7,8,9]])"), "2");
    assert_eq!(r("rank([[1,0,0],[0,1,0],[0,0,1]])"), "3");
    assert_eq!(r("rank([[0,0],[0,0]])"), "0");
    assert_eq!(r("rank([[1,2,3],[2,4,6]])"), "1"); // 2x3, rank 1
                                                   // Symbolic entry stays an honest residual (no fabricated number).
    assert_eq!(r("rank([[a,2],[3,4]])"), "rank([[a, 2], [3, 4]])");
    // Controls: the sibling matrix functions are unchanged.
    assert_eq!(r("det([[1,2],[3,4]])"), "-2");
    assert_eq!(r("trace([[1,2],[3,4]])"), "5");
}

#[test]
fn test_eval_diff_cancelling_bounded_inverse_keeps_domain_condition() {
    // `diff(2·arcsin(x)+2·arccos(x)) → 0` silently dropped the `-1<x<1` differentiability
    // interval when the derivative cancelled (the condition vanished with the √(1-x²) radical).
    // The differand is now walked for bounded-inverse subterms, re-emitting each one's OPEN
    // derivative-domain condition even on cancellation.
    let cond = |input: &str| -> Vec<String> {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["required_display"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    };
    assert!(cond("diff(2*arcsin(x)+2*arccos(x), x)").contains(&"-1 < x < 1".to_string()));
    assert!(cond("diff(arccosh(x)-arccosh(x), x)").contains(&"x > 1".to_string()));
    assert!(cond("diff(arctanh(x)-arctanh(x), x)").contains(&"-1 < x < 1".to_string()));
    // Non-cancelling stays exactly one condition (no duplicate from the new walker).
    assert_eq!(cond("diff(arcsin(x), x)"), vec!["-1 < x < 1".to_string()]);
    // All-real derivative domains gain no spurious condition; plain (non-diff) is untouched.
    assert!(cond("diff(arctan(x)-arctan(x), x)").is_empty());
    assert_eq!(cond("arcsin(x)-arcsin(x)"), vec!["-1 ≤ x ≤ 1".to_string()]);
}

#[test]
fn test_eval_symmetric_surd_even_quartic_integral_verifies() {
    // `c / (x^4 + p·x^2 + r)` whose even quartic factors over ℝ into the symmetric SURD pair
    // `(x²+a·x+s)(x²−a·x+s)` with `s=√r ∈ ℚ` but `a=√(2s−p)` irrational was an unevaluated residual
    // (the rational-coefficient factor path could not carry the √). It now integrates to a verified
    // arctan+log closed form. Numerically checked: F'(x) = integrand (err ~1e-11).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Phi_12 = x^4-x^2+1 (factor √3): no longer a bare `integrate(...)` residual.
    let phi12 = r("integrate(1/(x^4-x^2+1), x)");
    assert!(
        !phi12.starts_with("integrate("),
        "x^4-x^2+1 must integrate to a closed form, got residual: {phi12}"
    );
    assert!(
        phi12.contains("arctan") && phi12.contains("ln") && phi12.contains("sqrt(3)"),
        "expected arctan+log closed form over √3, got: {phi12}"
    );
    // x^4-3x^2+4 uses √7; the scaled numerator stays a closed form too.
    assert!(!r("integrate(1/(x^4-3*x^2+4), x)").starts_with("integrate("));
    assert!(!r("integrate(2/(x^4-x^2+1), x)").starts_with("integrate("));
    // Controls: routes owned elsewhere stay byte-identical.
    assert_eq!(r("integrate(1/(x^2+1), x)"), "arctan(x)");
    // `1/(x^6+1)` = 1/((x^2+1)(x^4-x^2+1)): the even quartic now integrates as a
    // FACTOR via G1 Cap. B (previously out of the symmetric-surd cycle's scope).
    assert!(!r("integrate(1/(x^6+1), x)").starts_with("integrate("));
    // `x^4+3x^2+1` factors into two irreducible quadratics with IRRATIONAL
    // constants (u-roots (-3±√5)/2), not the symmetric-surd form. It graduated
    // via G1 R2 (`EvenQuarticRealResolvent`, 2026-07-15): the conjugate split
    // over ℚ(√5) renders the arctan pair (both u-roots negative, no real
    // poles). Numerically confirmed against sympy at 30 digits.
    let real_resolvent = r("integrate(1/(x^4+3*x^2+1), x)");
    assert!(
        !real_resolvent.starts_with("integrate("),
        "x^4+3x^2+1 must integrate via the real-resolvent split: {real_resolvent}"
    );
    assert!(
        real_resolvent.contains("arctan") && real_resolvent.contains("sqrt(5)"),
        "expected the arctan pair over Q(sqrt(5)): {real_resolvent}"
    );
}

#[test]
fn test_eval_parametric_linear_degenerate_branch() {
    // A parametric linear equation whose coefficient cancels (`a·x = a`) dropped the `a ≠ 0` guard
    // and the `a = 0 ⇒ ℝ` branch, returning a bare `{1}`. It now emits the full conditional, matching
    // the structurally identical compound `(a-1)·x = a-1`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("solve(a*x=a, x)"),
        "{ 1 } if a != 0; All real numbers if a = 0"
    );
    assert_eq!(
        r("solve(a*x=2*a, x)"),
        "{ 2 } if a != 0; All real numbers if a = 0"
    );
    assert_eq!(
        r("solve(b*x=b, x)"),
        "{ 1 } if b != 0; All real numbers if b = 0"
    );
    assert_eq!(
        r("solve(a*(x-1)=0, x)"),
        "{ 1 } if a != 0; All real numbers if a = 0"
    );
    // Controls: a numeric-coefficient equation, a non-degenerate parametric solve (root still
    // contains the parameter), the compound form, and a non-linear equation are all UNCHANGED.
    assert_eq!(r("solve(2*x=4, x)"), "{ 2 }");
    assert_eq!(r("solve(a*x=b, x)"), "{ b / a }");
    assert_eq!(
        r("solve((a-1)*x=a-1, x)"),
        "{ 1 } if a - 1 != 0; All real numbers if a - 1 = 0"
    );
    assert_eq!(r("solve(x^2=4, x)"), "{ -2, 2 }");
}

#[test]
fn test_eval_reducible_quartic_factor_roots() {
    // A polynomial whose deflated quartic factor splits into two rational quadratics dropped the
    // quadratic factor's roots: `x⁵-5x³+x²-5 = (x+1)(x²-5)(x²-x+1)` returned only `{-1}`, losing the
    // `±√5` roots of `x²-5`. The quartic is now factored into quadratics and each is solved.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // The quintic recovers -1 plus the ±√5 surd roots (3 real roots, no residual).
    let quintic = r("solve(x^5-5*x^3+x^2-5=0, x)");
    assert!(
        quintic.contains("-1")
            && quintic.contains("sqrt(5)")
            && quintic.matches(", ").count() == 2
            && !quintic.contains("Solve"),
        "x^5-5x^3+x^2-5 -> {quintic}"
    );
    // Standalone reducible quartics with only-rational or mixed real roots.
    assert_eq!(r("solve(x^4+x^3-x-1=0, x)"), "{ -1, 1 }"); // (x²-1)(x²+x+1)
    assert_eq!(r("solve(x^4-3*x^2-4=0, x)"), "{ -2, 2 }"); // (x²-4)(x²+1)
                                                           // An IRREDUCIBLE quartic correctly declines (Ferrari deferred) — stays an honest residual.
    assert!(r("solve(x^4-x-1=0, x)").contains("solve("));
    // The reducible-quartic INEQUALITY now works through the sign-analysis chain.
    assert_eq!(r("x^4-3*x^2-4>0"), "(-infinity, -2) U (2, infinity)");
    // Controls: biquadratics and lower-degree solves are unchanged.
    assert_eq!(r("solve(x^4-5*x^2+4=0, x)"), "{ -2, -1, 1, 2 }");
    assert_eq!(r("solve(x^3-2=0, x)"), "{ 2^(1/3) }");
}

#[test]
fn test_eval_complex_negative_base_odd_root_principal_branch() {
    // In complex mode, a negative base under a rational `p/q` with ODD denominator is the PRINCIPAL
    // value `r^(p/q)·(cos(πp/q) + i·sin(πp/q))`, not the real odd root: `(-1)^(1/3) = 1/2 + (√3/2)i`,
    // not `-1`. The real-odd-root literal value was leaking into complex mode (Round-5 audit, P0).
    let cx = |input: &str| -> String {
        let out = cli()
            .args([
                "eval",
                input,
                "--value-domain",
                "complex",
                "--format",
                "json",
            ])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    let re = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Complex principal branch: a non-real value with the correct real part, NOT the real odd root.
    let r13 = cx("(-1)^(1/3)");
    assert!(
        r13.contains('i') && r13.contains("1/2") && r13 != "-1",
        "(-1)^(1/3) complex -> {r13}"
    );
    assert!(cx("(-8)^(1/3)").contains('i') && cx("(-8)^(1/3)") != "-2");
    assert!(cx("(-1)^(2/3)").contains('i') && cx("(-1)^(2/3)") != "1");
    // Even-root complex (sqrt(-n) → i·sqrt(n)) and positive bases are unaffected.
    assert_eq!(cx("(-4)^(1/2)"), "2·i");
    assert_eq!(cx("8^(1/3)"), "2");
    // REAL mode keeps the engine's real-odd-root convention.
    assert_eq!(re("(-8)^(1/3)"), "-2");
    assert_eq!(re("(-1)^(1/3)"), "-1");
}

#[test]
fn test_eval_abs_equation_quadratic_arg_split() {
    // `|arg(x)| = c` (constant `c ≥ 0`) with a quadratic argument carrying a linear term leaked a
    // circular residual `solve(x − (2x+3)^(1/2)=0)` from the recursive isolation, even though
    // `solve(x²-2x = 3)` returns `{-1, 3}`. The `|arg|=c → arg=±c` split now solves it.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(abs(x^2-2*x)=3, x)"), "{ -1, 3 }");
    assert_eq!(r("solve(abs(x^2-x)=2, x)"), "{ -1, 2 }");
    assert_eq!(r("solve(abs(x^2-2*x)=0, x)"), "{ 0, 2 }"); // c = 0: single branch, no duplicate
    assert_eq!(r("solve(abs(x^2-2*x)=-1, x)"), "No solution"); // c < 0
                                                               // Both branches contribute: |x²-5x|=6 has four roots {-1, 2, 3, 6}.
    let four = r("solve(abs(x^2-5*x)=6, x)");
    assert!(
        four.contains("-1")
            && four.contains("6")
            && four.matches(", ").count() == 3
            && !four.contains("Solve"),
        "|x^2-5x|=6 -> {four}"
    );
    // Cases the normal path already solved are unchanged.
    assert_eq!(r("solve(abs(x^2+x)=2, x)"), "{ -2, 1 }");
    assert_eq!(r("solve(abs(x-3)=2, x)"), "{ 5, 1 }");
}

#[test]
fn test_eval_biquadratic_surd_roots() {
    // A biquadratic `a·x⁴ + b·x² + c` whose x-roots are surds leaked a circular residual
    // (`solve(x − (8x²−15)^(1/4)=0)`); the `z = x²` substitution now solves it. Roots verified
    // numerically in the dev probes (|p(root)| < 1e-13).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Four surd roots {±√3, ±√5}.
    let four = r("x^4-8*x^2+15=0");
    assert!(
        four.contains("sqrt(5)")
            && four.contains("sqrt(3)")
            && four.matches(", ").count() == 3
            && !four.contains("Solve"),
        "x^4-8x^2+15=0 -> {four}"
    );
    // Only the non-negative z root survives: {±√3} (z = -1 dropped).
    let two = r("x^4-2*x^2-3=0");
    assert!(
        two.contains("sqrt(3)") && two.matches(", ").count() == 1 && !two.contains("Solve"),
        "x^4-2x^2-3=0 -> {two}"
    );
    // No real roots when both z roots are negative or complex.
    assert_eq!(r("x^4+x^2+1=0"), "No solution");
    assert_eq!(r("x^4+3*x^2+2=0"), "No solution");
    // Rational-root biquadratics and general (non-biquadratic) quartics are unchanged.
    assert_eq!(r("x^4-5*x^2+4=0"), "{ -2, -1, 1, 2 }");
    assert!(r("x^4-x-1=0").contains("solve(")); // general quartic stays a residual (Ferrari deferred)
                                                // The biquadratic INEQUALITY is now operator-sensitive (biquad solver → Discrete → sign analysis).
    let gt = r("x^4-8*x^2+15>0");
    let lt = r("x^4-8*x^2+15<0");
    assert_ne!(gt, lt, "operator must matter");
    assert!(
        gt.contains(" U ") && !gt.contains("Solve") && !gt.contains('{'),
        "x^4-8x^2+15>0 -> {gt}"
    );
}

#[test]
fn test_eval_irreducible_polynomial_inequality_sign_analysis() {
    // An irreducible polynomial inequality was rewritten to `Equal(p, 0)`, dropping the operator and
    // returning the equation's root SET — so `> 0` and `< 0` gave byte-identical output. Sign analysis
    // over the (closed-form) real roots now yields the correct interval union, respecting the operator
    // and using open endpoints for strict ops, closed for non-strict.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Δ>0 cubic (one real root): half-line, operator-sensitive, no longer a root set.
    let gt = r("x^3+x+1>0");
    let lt = r("x^3+x+1<0");
    assert_ne!(gt, lt, "operator must matter (was the P0 defect)");
    assert!(
        gt.contains("infinity") && !gt.contains('{') && !gt.contains("Solve"),
        "x^3+x+1>0 -> {gt}"
    );
    assert!(gt.starts_with('(') && gt.ends_with("infinity)"), "{gt}"); // (r, infinity)
    assert!(lt.starts_with("(-infinity"), "{lt}"); // (-infinity, r)
                                                   // Non-strict closes the endpoint at the root.
    let geq = r("x^3+x+1>=0");
    assert!(geq.starts_with('[') && geq.ends_with("infinity)"), "{geq}");
    // Casus irreducibilis (three real roots): a two-piece interval union.
    let casus = r("x^3-3*x+1>0");
    assert!(
        casus.contains(" U ") && casus.contains("infinity") && !casus.contains('{'),
        "x^3-3x+1>0 -> {casus}"
    );
    assert_ne!(casus, r("x^3-3*x+1<0"), "operator must matter");
    // Controls: factorable inequalities and the underlying equation are unchanged.
    assert_eq!(r("x^2-1>0"), "(-infinity, -1) U (1, infinity)");
    assert_eq!(r("x^2-1<0"), "(-1, 1)");
    assert_eq!(r("x^3-2=0"), "{ 2^(1/3) }");
}

#[test]
fn test_eval_symbolic_power_of_power_guards_base_sign() {
    // `(x^a)^b = x^(a·b)` holds for ALL real x only when both exponents are integers; with a
    // non-integer exponent it needs `x ≥ 0` (for x<0, `x^a` is not real and the fold drops the sign,
    // so `((-2)^a)^b ≠ (-2)^(a·b)`). The old unconditional fold was a wrong value. Now: integer and
    // provably-non-negative bases still fold; a non-provably-non-negative or negative base declines
    // in the default (generic) domain (honest unevaluated form), and `--domain assume` opts in.
    for (input, expected) in [
        ("(x^2)^3", "x^6"), // integer exponents: unconditional, valid for all x
        ("(x^3)^2", "x^6"),
        ("((-2)^3)^2", "64"), // integer exponents over a negative base: still exact
        ("(2^a)^b", "2^(a·b)"), // provably-positive base: unconditional fold
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
    // Symbolic exponents over an unknown- or negative-sign base no longer fold to a wrong value in
    // the default domain — they stay an honest unevaluated form.
    for input in ["(x^a)^b", "((-2)^a)^b"] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(
            wire["result"].as_str(),
            Some(input),
            "{input} should stay unevaluated"
        );
    }
    // `--domain assume` opts into the analytic fold (the user accepts x ≥ 0).
    let assumed = cli()
        .args(["eval", "(x^a)^b", "--domain", "assume", "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&assumed.stdout).expect("Invalid wire output");
    assert_eq!(wire["result"].as_str(), Some("x^(a·b)"));
}

#[test]
fn test_eval_solve_all_reals_inlines_domain_condition() {
    // An identity equation whose solution is all reals RESTRICTED by a domain condition must show
    // that condition in the default text surface (`All real numbers if x > 0`), matching the in-set
    // conditional convention (`1/x=1/x → "… if x != 0"`), not a dishonest bare `All real numbers`.
    for (input, expected) in [
        ("solve(ln(x^2)=2*ln(x), x)", "All real numbers if x > 0"),
        ("solve(2*ln(x)=ln(x^2), x)", "All real numbers if x > 0"),
        ("solve(e^(ln(x))=x, x)", "All real numbers if x > 0"),
        ("solve(sqrt(x)^2=x, x)", "All real numbers if x ≥ 0"),
        (
            "solve(ln(x^2)=2*ln(abs(x)), x)",
            "All real numbers if x ≠ 0",
        ),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
        // The LaTeX surface uses the matching `\begin{cases}` conditional form.
        assert!(
            wire["result_latex"]
                .as_str()
                .unwrap_or("")
                .contains("\\begin{cases}"),
            "{input} latex"
        );
    }
    // Controls: an in-set conditional is NOT double-rendered; an unconditional identity stays bare;
    // a `simplify` result whose `required_display` is intentionally JSON-only keeps its bare text.
    for (input, expected) in [
        ("solve(1/x=1/x, x)", "All real numbers if x != 0"),
        ("solve(0*x=0, x)", "All real numbers"),
        ("sqrt(x)*sqrt(x)", "x"),
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
fn test_eval_multi_factor_cancellation_fully_reduces() {
    // `(2·x·y)/(5·x·y)` shares TWO common factors. The plain-mode one-factor shortcut cancelled only
    // `y`, returning the partially-reduced `2·x / (5·x)` and diverging from `--steps` (which cancels
    // all common factors to `2/5`). When a residual common factor remains the shortcut now declines,
    // so the full pipeline reduces it completely.
    for (input, expected) in [
        ("(2*x*y)/(5*x*y)", "2/5"),
        ("(x*y*z)/(u*y*z)", "x / u"),
        ("(6*x*y)/(4*x*y)", "3/2"),
        ("(a*b*c)/(d*b*c)", "a / d"),
        // Single common factor is unaffected (still cancels in the shortcut).
        ("(x*y)/(u*y)", "x / u"),
        ("(a*b)/b", "a"),
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
fn test_eval_finite_plus_infinity_absorbs_in_both_modes() {
    // `finite + ∞ = ∞` (absorption). In plain mode `∞` was treated as a symbolic atom, so the
    // "symbolic atom + literal" shortcut returned `∞ + 1` UNEVALUATED — diverging from `--steps`,
    // which absorbs it. `∞`/`undefined` are no longer symbolic atoms; finite constants (`π`,`e`,`i`)
    // still are.
    for (input, expected) in [
        ("inf+1", "infinity"),
        ("1+inf", "infinity"),
        ("inf+5", "infinity"),
        ("inf-1", "infinity"),
        ("2+inf+3", "infinity"),
        ("(-inf)+3", "-infinity"),
        // Finite atoms stay symbolic; undefined still propagates.
        ("pi+1", "1 + pi"),
        ("e+2", "2 + e"),
        ("undefined+1", "undefined"),
        ("inf+x", "x + infinity"),
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
fn test_eval_non_real_solution_rejected_in_real_domain() {
    // In the RealOnly domain, a provably NON-REAL solution (the imaginary unit `i`, `√(negative)`,
    // an even root of a negative `(-1)^(1/2)`, or anything carrying them) has no real solution. The
    // `ln`/`exp` inversion did not re-check reality, so `solve(ln(x)=√(-1)) → {e^((-1)^(1/2))}` (= e^i)
    // and `solve(x=i) → {i}` slipped through. An ODD root of a negative (`(-8)^(1/3) = -2`) is REAL.
    for input in [
        "solve(ln(x)=sqrt(-1), x)",
        "solve(x=sqrt(-1), x)",
        "solve(x=e^(sqrt(-1)), x)",
        "solve(ln(x)=sqrt(-4), x)",
        "solve(x=i, x)",
        "solve(x=2*i, x)",
        "solve(x=1+i, x)",
        "solve(x^2=e^(sqrt(-1)), x)",
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some("No solution"), "{input}");
    }
    // REAL solutions (incl. odd roots of negatives) must survive.
    for (input, expected) in [
        ("solve(x=5, x)", "{ 5 }"),
        ("solve(x^2=4, x)", "{ -2, 2 }"),
        ("solve(x=(-8)^(1/3), x)", "{ -2 }"),
        ("solve(x^3=-8, x)", "{ -2 }"),
        ("solve(ln(x)=2, x)", "{ e^2 }"),
        ("solve(x=sqrt(2), x)", "{ sqrt(2) }"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
    // Complex domain keeps an imaginary solution.
    let output = cli()
        .args([
            "eval",
            "solve(x=i, x)",
            "--format",
            "json",
            "--value-domain",
            "complex",
        ])
        .output()
        .expect("Failed to run CLI");
    assert!(output.status.success());
    let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
    assert_eq!(wire["result"].as_str(), Some("{ i }"));
}

#[test]
fn test_eval_inequality_intersects_factor_function_domain() {
    // A domain-restricted function (`ln`, `√`) appearing as a FACTOR (not the bare LHS) must still
    // exclude its undefined region: `ln(x)·(x−2)² ≤ 0` is `(0,1]∪{2}`, NOT `(−∞,1]∪{2}` (`ln` is
    // undefined for x ≤ 0). The inequality result is now intersected with the LHS's implicit domain.
    for (input, expected) in [
        ("solve(ln(x)*(x-2)^2<=0, x)", "(0, 1] U [2, 2]"),
        ("solve(ln(x)*(x-2)>=0, x)", "(0, 1] U [2, infinity)"),
        ("solve(ln(x)*(x-3)<=0, x)", "[1, 3]"),
        // Bare-function controls (already correct) stay correct.
        ("solve(ln(x)<=0, x)", "(0, 1]"),
        ("solve(ln(x)>=0, x)", "[1, infinity)"),
        ("solve(sqrt(x)>=2, x)", "[4, infinity)"),
        // No domain restriction -> unchanged.
        ("solve(x^2-1>0, x)", "(-infinity, -1) U (1, infinity)"),
        ("solve((x-1)*(x-3)<=0, x)", "[1, 3]"),
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
fn test_eval_even_power_exponential_keeps_positive_root() {
    // `a^(2x) = k` is solved as `(a^x)^2 = k -> a^x = ±√k`. The POSITIVE root gives the real solution
    // `x = log_a(√k)`; the NEGATIVE root `a^x = -√k` is unsatisfiable (a^x > 0). The back-substitution
    // aggregator used to let the negative root's guarded (false) conditional OVERWRITE the real
    // solution, returning the empty `{…} if -√k > 0`. Discrete solutions now survive a sibling
    // conditional branch.
    // Cases with a clean closed form:
    for (input, expected) in [
        ("solve(2^(2*x)=2, x)", "{ 1/2 }"),
        ("solve(e^(2*x)=5, x)", "{ 1/2·ln(5) }"),
        // Unchanged controls (clean even powers / direct log / negative or zero RHS):
        ("solve(3^(2*x)=9, x)", "{ 1 }"),
        ("solve(3^(2*x)=81, x)", "{ 2 }"),
        ("solve(3^(2*x)=16, x)", "{ ln(4) / ln(3) }"),
        ("solve(e^(2*x)=-5, x)", "No solution"),
        ("solve(3^(2*x)=0, x)", "No solution"),
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some(expected), "{input}");
    }
    // Cases whose value is correct but not fully simplified (e.g. `3^(2x)=27` is `3/2`): assert they
    // return a single real solution rather than the old empty `{…} if -√k > 0` conditional.
    for input in [
        "solve(3^(2*x)=27, x)",
        "solve(2^(2*x)=8, x)",
        "solve(5^(2*x)=125, x)",
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        let result = wire["result"].as_str().unwrap_or("");
        assert!(
            result.starts_with("{ ") && !result.contains("if") && !result.contains("No solution"),
            "{input} -> {result}"
        );
    }
}

#[test]
fn test_eval_equation_with_undefined_side_has_no_solution() {
    // A relation with an `undefined` side has NO real solution — nothing equals/compares to
    // `undefined`. In RealOnly, `ln(-2)` / `ln(-1)` / `1/0` simplify to `undefined`, so these are
    // unsatisfiable. The isolation path used to emit a degenerate `All real numbers if undefined = 0`
    // conditional (its guard is never true).
    for input in [
        "solve(ln(x)=ln(-2), x)",
        "solve(x=ln(-1), x)",
        "solve(x=ln(-2), x)",
        "solve(ln(x)=undefined, x)",
        "solve(x+1=undefined, x)",
        "solve(x=1/0, x)",
        // Matrix-equation members: the scalar-broadcast of `A*X` (2x2) minus the
        // 2x1 column RHS folds to `undefined` AFTER the subtraction, so the
        // var-eliminated residual — not a bare side — is non-finite. Under the
        // engine's scalar-X semantics a 2x2 can never equal a 2x1 column, so the
        // sound answer is "No solution", not "All real numbers if undefined = 0".
        "solve([[1,2],[3,4]]*X=[[5],[6]], X)",
        "solve([[1,0],[0,1]]*X=[[2],[3]], X)",
        "solve(X*[[1,2],[3,4]]=[[5],[6]], X)",
    ] {
        let output = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        assert!(output.status.success(), "{input}");
        let wire: Value = serde_json::from_slice(&output.stdout).expect("Invalid wire output");
        assert_eq!(wire["result"].as_str(), Some("No solution"), "{input}");
    }
    // Controls: a defined (possibly non-real-rejected) RHS is unaffected.
    for (input, expected) in [
        ("solve(ln(x)=ln(2), x)", "{ 2 }"),
        ("solve(x=sqrt(-4), x)", "No solution"),
        ("solve(ln(x)=2, x)", "{ e^2 }"),
        ("solve(x^2=4, x)", "{ -2, 2 }"),
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
fn test_eval_nonstrict_inequality_includes_isolated_roots() {
    // For a NON-STRICT inequality `f ≤ 0` / `f ≥ 0`, every real in-domain root of `f` is a solution
    // (the value `0` satisfies the relation), but the interval sign-analysis emits only the
    // sign-CHANGE regions and drops the isolated roots of even-multiplicity factors that fall
    // outside them. Those roots are now unioned back in (as degenerate `[p, p]` intervals or a
    // `{p}` discrete set), with poles excluded by construction.
    for (input, expected) in [
        ("solve((x-2)^2*(x+1)<=0, x)", "(-infinity, -1] U [2, 2]"),
        ("solve((x+1)^2*(x-3)^3>=0, x)", "[-1, -1] U [3, infinity)"),
        ("solve(x^2/(x-1)>=0, x)", "[0, 0] U (1, infinity)"),
        (
            "solve(x^2*(x^2-4)>=0, x)",
            "(-infinity, -2] U [0, 0] U [2, infinity)",
        ),
        ("solve(x^3*(x-2)^2<=0, x)", "(-infinity, 0] U [2, 2]"),
        (
            "solve((x-1)*(x-2)^2*(x-3)>=0, x)",
            "(-infinity, 1] U [2, 2] U [3, infinity)",
        ),
        ("solve((x-1)^4*(x+1)<=0, x)", "(-infinity, -1] U [1, 1]"),
        ("solve(x^2/((x-1)*(x-2))<=0, x)", "[0, 0] U (1, 2)"),
        ("solve((x-3)^2/(x-1)<=0, x)", "(-infinity, 1) U [3, 3]"),
        ("solve((x+3)^2*(x-1)*(x-5)<=0, x)", "[-3, -3] U [1, 5]"),
        // Pure touch point -> the single solution, rendered as a degenerate interval `[p, p]` (the
        // root flows through the interval-union machinery once Discrete∪interval unions keep both sides).
        ("solve((x-2)^2<=0, x)", "[2, 2]"),
        ("solve(-(x-2)^2>=0, x)", "[2, 2]"),
        // STRICT controls: `0` does NOT satisfy `<`/`>`, so NO isolated root is added.
        ("solve((x-2)^2*(x+1)<0, x)", "(-infinity, -1)"),
        ("solve(x^2/(x-1)>0, x)", "(1, infinity)"),
        // Squares are everywhere-nonnegative; a pole is never a solution.
        ("solve((x-2)^4>=0, x)", "All real numbers"),
        ("solve(1/(x-2)^2<=0, x)", "No solution"),
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

#[test]
fn test_eval_polynomial_in_absolute_value_substitutes_and_splits_branches() {
    // `|x|² − 3|x| + 2 = 0` reaches solve as `x² − 3|x| + 2 = 0` (the simplifier
    // folds `|x|² → x²`). Because `x² = |x|²` it is a quadratic in `u = |x|`:
    // `u² − 3u + 2 = 0 ⟹ u ∈ {1,2} ⟹ x ∈ {±1, ±2}`. It used to leak a malformed
    // `solve(x − √(3|x| − 2) = 0, …)` residual, dropping the negative branch and
    // every root.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("solve(abs(x)^2 - 3*abs(x) + 2 = 0, x)"),
        "{ 1, -1, 2, -2 }"
    );
    assert_eq!(r("solve(x^2 - 3*abs(x) + 2 = 0, x)"), "{ 1, -1, 2, -2 }");
    assert_eq!(
        r("solve(2*abs(x)^2 - 3*abs(x) + 1 = 0, x)"),
        "{ 1/2, -1/2, 1, -1 }"
    );
    // c = 0 border: `u(u−3) = 0 ⟹ u ∈ {0,3}`, and `u = 0 ⟹ x = 0` is a single root.
    assert_eq!(r("solve(abs(x)^2 - 3*abs(x) = 0, x)"), "{ 0, 3, -3 }");
    assert_eq!(r("solve(abs(x)^3 - abs(x) = 0, x)"), "{ 0, 1, -1 }");
    // Higher even degree: `u⁴ − 5u² + 4 = 0 ⟹ u ∈ {1,2}`.
    assert_eq!(r("solve(x^4 - 5*abs(x)^2 + 4 = 0, x)"), "{ -2, -1, 1, 2 }");
    // A negative `u`-root has no real pre-image and is dropped: `u = (1±√5)/2`,
    // keep only `φ`.
    assert_eq!(r("solve(abs(x)^2 - abs(x) - 1 = 0, x)"), "{ phi, -phi }");
    // Every `u`-root negative ⇒ no real solution.
    assert_eq!(r("solve(abs(x)^2 + 3*abs(x) + 2 = 0, x)"), "No solution");

    // GATES: a term that breaks evenness in x (`x + |x|`) is not a polynomial in
    // |x| — it declines here and the piecewise handler solves it.
    assert_eq!(r("solve(x + abs(x) - 4 = 0, x)"), "{ 2 }");
    assert_eq!(r("solve(x^2 + abs(x) - x = 0, x)"), "{ 0 }");
    // Plain polynomials and the degree-1 `|x|` isolation are untouched.
    assert_eq!(r("solve(x^2 - 3*x + 2 = 0, x)"), "{ 1, 2 }");
    assert_eq!(r("solve(abs(x) = 2, x)"), "{ 2, -2 }");
}

#[test]
fn test_eval_single_abs_equals_polynomial_solves_both_branches_with_domain() {
    // A single `|f|` term with a non-constant degree-≥2 remainder is `|f| = g(x)`.
    // Isolating the abs and recursing is unsound: the generic path solved only the
    // `f = g` branch and skipped `g ≥ 0`, so `x² + |x−1| − 3 = 0` returned the
    // spurious `{−2.56, 1.56}` (missing the real `−1`), and `x² − 3|x−1| + 2 = 0`
    // leaked a malformed residual. Both branches are now solved and each root kept
    // only when `g(r) ≥ 0` (decided exactly, so surd roots verify).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // Was `{ -2.56, 1.56 }` (spurious root, missing `-1`).
    assert_eq!(
        r("solve(x^2 + abs(x-1) - 3 = 0, x)"),
        "{ 1/2·(sqrt(17) - 1), -1 }"
    );
    // Orientation-independent (abs on the RHS).
    assert_eq!(
        r("solve(3 - x^2 = abs(x-1), x)"),
        "{ 1/2·(sqrt(17) - 1), -1 }"
    );
    // Was a malformed `solve(x − √(3|x−1| − 2))` residual.
    assert_eq!(
        r("solve(x^2 - 3*abs(x-1) + 2 = 0, x)"),
        "{ 1/2·(-sqrt(13) - 3), 1/2·(sqrt(13) - 3) }"
    );
    // A coefficient on the abs term.
    assert_eq!(
        r("solve(x^2 + 2*abs(x-1) - 5 = 0, x)"),
        "{ 2·sqrt(2) - 1, -1 }"
    );
    // `g(r) ≥ 0` verification keeps the on-domain root and drops the off-domain
    // one: `|x−1| = x²−1` keeps `{1, −2}` (both have `x²−1 ≥ 0`).
    assert_eq!(r("solve(abs(x-1) = x^2 - 1, x)"), "{ 1, -2 }");
    // Every candidate has `g < 0` ⇒ no real solution.
    assert_eq!(r("solve(x^2 + abs(x-1) + 3 = 0, x)"), "No solution");
    assert_eq!(r("solve(abs(x-5) = -x^2 - 1, x)"), "No solution");

    // NO REGRESSION: linear `g` stays with the isolation path, constant `g` and
    // bare `|x|` polynomials with their own handlers, multi-abs with the
    // piecewise handler.
    assert_eq!(r("solve(abs(x-2) = x, x)"), "{ 1 }");
    assert_eq!(r("solve(abs(2*x-1) = x + 1, x)"), "{ 2, 0 }");
    assert_eq!(r("solve(abs(x^2-2*x) = 3, x)"), "{ -1, 3 }");
    assert_eq!(r("solve(x^2 - 3*abs(x) + 2 = 0, x)"), "{ 1, -1, 2, -2 }");
    assert_eq!(r("solve(abs(x-1) + abs(x+1) = 4, x)"), "{ -2, 2 }");
}

#[test]
fn test_eval_single_abs_polynomial_inequality_sign_splits_at_the_abs_zero() {
    // A polynomial inequality with a single `|f|` term was solved by an opaque
    // path that returned a WRONG "No solution" (`x² − 3|x| + 2 < 0` is
    // `(−2,−1) ∪ (1,2)`, not ∅). It now splits at `f = 0` into the `|f| = ±f`
    // branches, solves each, intersects with the branch domain, and unions.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("solve(abs(x)^2 - 3*abs(x) + 2 < 0, x)"),
        "(-2, -1) U (1, 2)"
    );
    assert_eq!(r("solve(x^2 - 3*abs(x) + 2 < 0, x)"), "(-2, -1) U (1, 2)");
    // The `>=` complement, with closed boundaries.
    assert_eq!(
        r("solve(x^2 - 3*abs(x) + 2 >= 0, x)"),
        "(-infinity, -2] U [-1, 1] U [2, infinity)"
    );
    // `<=` includes the boundary points.
    assert_eq!(r("solve(x^2 - 3*abs(x) + 2 <= 0, x)"), "[-2, -1] U [1, 2]");
    assert_eq!(
        r("solve(x^2 - abs(x) - 2 > 0, x)"),
        "(-infinity, -2) U (2, infinity)"
    );
    // Shifted abs argument (the split is at x = 1, not symmetric).
    assert_eq!(
        r("solve(x^2 - 3*abs(x-1) + 2 < 0, x)"),
        "(1/2·(-sqrt(13) - 3), 1/2·(sqrt(13) - 3))"
    );
    assert_eq!(
        r("solve(2*abs(x-1) + x^2 - 5 < 0, x)"),
        "(-1, 2·sqrt(2) - 1)"
    );
    // Always-signed remainders: empty / full without a spurious split.
    assert_eq!(r("solve(x^2 + abs(x) + 1 < 0, x)"), "No solution");
    assert_eq!(r("solve(x^2 + abs(x) + 1 > 0, x)"), "All real numbers");

    // NO REGRESSION: bare `|f| {op} c`, two-abs, sign-form, and top-level
    // `|quadratic| {op} c` keep their own handlers.
    assert_eq!(r("solve(abs(x-1) <= 2, x)"), "[-1, 3]");
    assert_eq!(r("solve(abs(x-1) < abs(x-3), x)"), "(-infinity, 2)");
    assert_eq!(r("solve(x/abs(x) = 1, x)"), "(0, infinity)");
}

#[test]
fn test_eval_single_abs_polynomial_equation_sign_splits_at_the_abs_zero() {
    // An EQUATION with a single `|f|` term entangled MULTIPLICATIVELY with a
    // polynomial (`x·|x| = 4`) is not `|f| = g` (isolated) nor a pure
    // polynomial-in-|x| (the odd `x` factor is not a function of `|x|`). The
    // isolation path reoriented to `x = 4/|x|` and leaked a malformed
    // `solve(x − 4/|x| = 0)` residual. The sign split at `f = 0` (same handler as
    // the inequality form) now solves each `|f| = ±f` polynomial branch and keeps
    // the roots on that branch's half-line.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    // `x·|x|` is a strictly increasing bijection: exactly one real root, its sign
    // matching the RHS. Positive RHS lands in the `x ≥ 0` branch (`x² = c`).
    assert_eq!(r("solve(x*abs(x) = 4, x)"), "{ 2 }");
    assert_eq!(r("solve(x*abs(x) = 2, x)"), "{ sqrt(2) }");
    // Negative RHS lands in the `x < 0` branch (`−x² = c`).
    assert_eq!(r("solve(x*abs(x) = -4, x)"), "{ -2 }");
    assert_eq!(r("solve(x*abs(x) + 1 = 0, x)"), "{ -1 }");
    // Rational leading coefficient is cleared before the split.
    assert_eq!(r("solve(2*x*abs(x) = 8, x)"), "{ 2 }");
    // A quadratic branch keeps ALL in-domain roots and drops the out-of-domain
    // one: `x·|x| − x = 0` is `x(|x|−1) = 0` → `{−1, 0, 1}`.
    assert_eq!(r("solve(x*abs(x) - x = 0, x)"), "{ 0, 1, -1 }");
    assert_eq!(r("solve(x*abs(x) + 2*x = 3, x)"), "{ 1 }");
    // Shifted abs argument: the split is at x = 1, and `u·|u|` (u = x−1) is a
    // bijection, so a single root.
    assert_eq!(r("solve((x-1)*abs(x-1) = 4, x)"), "{ 3 }");
    assert_eq!(r("solve(x*abs(x-1) = 6, x)"), "{ 3 }");

    // NO REGRESSION: isolated-abs (`|f| = g`), poly-in-|x|, bare `|f| = c`, and
    // the sign form keep their own, already-correct equation handlers.
    assert_eq!(r("solve(abs(x) = 4, x)"), "{ 4, -4 }");
    assert_eq!(r("solve(x^2 - 3*abs(x) + 2 = 0, x)"), "{ 1, -1, 2, -2 }");
    assert_eq!(
        r("solve(abs(x-1) = 3 - x^2, x)"),
        "{ 1/2·(sqrt(17) - 1), -1 }"
    );
    assert_eq!(r("solve(x/abs(x) = 1, x)"), "(0, infinity)");
}

#[test]
fn test_eval_abs_as_a_factor_inequality_sign_splits() {
    // When the abs is a FACTOR rather than an added term (`|x|³ − |x| = |x|(x²−1)`),
    // removing it leaves a constant remainder, so the earlier "non-constant
    // remainder" gate wrongly declined and the generic path returned "No
    // solution". The gate now also fires on a degree-≥2 branch, so the sign
    // split still applies. `|x|³ − |x| < 0` is `(−1,0) ∪ (0,1)` — 0 excluded
    // (the value there is exactly 0).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(abs(x)^3 - abs(x) < 0, x)"), "(-1, 0) U (0, 1)");
    assert_eq!(
        r("solve(abs(x)^3 - 4*abs(x) > 0, x)"),
        "(-infinity, -2) U (2, infinity)"
    );
    // `>=` includes the isolated zero at x = 0.
    assert_eq!(
        r("solve(abs(x)^3 - abs(x) >= 0, x)"),
        "(-infinity, -1] U [0, 0] U [1, infinity)"
    );
    // No regression on the added-term forms or bare `|f| {op} c`.
    assert_eq!(r("solve(x^2 - 3*abs(x) + 2 < 0, x)"), "(-2, -1) U (1, 2)");
    assert_eq!(r("solve(abs(x-1) <= 2, x)"), "[-1, 3]");
}

#[test]
fn test_eval_multi_abs_polynomial_relation_partitions_at_breakpoints() {
    // Two-or-more affine `|f|` terms PLUS a degree-≥2 remainder — the linear
    // sum-of-abs handler carries only a linear remainder, so `x² + |x−1| + |x+1|
    // < 5` used to return a wrong "No solution" (the true set is (1−√6, √6−1)).
    // Partition at the breakpoints and solve the polynomial per segment.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("solve(x^2 + abs(x-1) + abs(x+1) < 5, x)"),
        "(1 - sqrt(6), sqrt(6) - 1)"
    );
    // The equation form gives the two boundary points.
    assert_eq!(
        r("solve(x^2 + abs(x-1) + abs(x+1) = 5, x)"),
        "{ 1 - sqrt(6), sqrt(6) - 1 }"
    );
    // Minimum is 2 at x=0, so `<= 2` is the isolated point {0}.
    assert_eq!(r("solve(x^2 + abs(x-1) + abs(x+1) <= 2, x)"), "[0, 0]");
    assert_eq!(
        r("solve(x^2 - abs(x-1) - abs(x+1) > 0, x)"),
        "(-infinity, -2) U (2, infinity)"
    );
    // Three abs terms.
    assert_eq!(
        r("solve(x^2 + abs(x) + abs(x-1) + abs(x+1) < 6, x)"),
        "(1/2·(3 - sqrt(33)), 1/2·(sqrt(33) - 3))"
    );
    // Shifted breakpoints (min 4 at 0, so `< 5` is (-1, 1)).
    assert_eq!(r("solve(x^2 + abs(x-2) + abs(x+2) < 5, x)"), "(-1, 1)");
    // Empty result stays empty.
    assert_eq!(r("solve(x^2 - abs(x-1) - abs(x+1) < -3, x)"), "No solution");

    // NO REGRESSION: a LINEAR remainder keeps the existing sum-of-abs handler.
    assert_eq!(r("solve(abs(x-1) + abs(x+1) < 3, x)"), "(-3/2, 3/2)");
    assert_eq!(
        r("solve(abs(x) + abs(x-2) >= 4, x)"),
        "(-infinity, -1] U [3, infinity)"
    );
    // Single abs stays with the sign-split handler.
    assert_eq!(r("solve(x^2 - 3*abs(x) + 2 < 0, x)"), "(-2, -1) U (1, 2)");
}

#[test]
fn test_eval_finite_geometric_sum_with_symbolic_ratio() {
    // A finite geometric sum with a SYMBOLIC ratio used to decline and echo
    // `sum(r^k, k, 0, n)`; the numeric-ratio builders only handle a rational
    // base. It now emits the closed form `(r^(n+1) - r^a)/(r - 1)` (removable
    // singularity at r=1, matching how the engine simplifies through such holes).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("sum(r^k, k, 0, n)"), "(r^(n + 1) - 1) / (r - 1)");
    assert_eq!(r("sum(r^k, k, 1, n)"), "(r^(n + 1) - r) / (r - 1)");
    assert_eq!(r("sum(x^k, k, 0, n)"), "(x^(n + 1) - 1) / (x - 1)");
    assert_eq!(r("sum((x+1)^k, k, 0, n)"), "((x + 1)^(n + 1) - 1) / x");
    // A numeric ratio keeps the cleaner integer-base form; a numeric upper bound
    // still expands directly. (The arithmetic-geometric `k·r^k` is closed by its
    // own sibling builder — see the arithmetic-geometric contract test.)
    assert_eq!(r("sum(2^k, k, 0, n)"), "2^(n + 1) - 1");
    assert_eq!(r("sum(x^k, k, 0, 3)"), "x^3 + x^2 + x + 1");
    // A leading coefficient (numeric or symbolic, index-free) is carried
    // through: `sum(c·r^k) = c·(r^(n+1) - r^a)/(r - 1)`.
    assert_eq!(r("sum(3*r^k, k, 0, n)"), "(3·r^(n + 1) - 3) / (r - 1)");
    assert_eq!(r("sum(5*r^k, k, 1, n)"), "(5·r^(n + 1) - 5·r) / (r - 1)");
    assert_eq!(r("sum(c*x^k, k, 0, n)"), "(c·x^(n + 1) - c) / (x - 1)");
    // The bare index `k` is NOT a coefficient — `k·r^k` stays with the
    // arithmetic-geometric builder, not hijacked into `k·(...)`.
    assert_eq!(
        r("sum(k*r^k, k, 1, n)"),
        "r·(n·r^(n + 1) + r^n·(-n - 1) + 1) / (1 - r)^2"
    );
}

#[test]
fn test_eval_finite_arithmetic_geometric_sum_with_symbolic_ratio() {
    // A finite arithmetic-geometric sum `sum(k·r^k)` with a SYMBOLIC ratio used
    // to decline: the numeric builder decomposes the ratio as a rational. It now
    // emits the closed form `r(1 - (n+1)r^n + n·r^(n+1))/(1-r)^2` (verified
    // numerically: at r=2, n=3 the value is 34).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("sum(k*r^k, k, 1, n)"),
        "r·(n·r^(n + 1) + r^n·(-n - 1) + 1) / (1 - r)^2"
    );
    // Lower bound 0 shares the formula (the k=0 term is zero).
    assert_eq!(
        r("sum(k*r^k, k, 0, n)"),
        "r·(n·r^(n + 1) + r^n·(-n - 1) + 1) / (1 - r)^2"
    );
    assert_eq!(
        r("sum(k*x^k, k, 1, n)"),
        "x·(n·x^(n + 1) + x^n·(-n - 1) + 1) / (1 - x)^2"
    );
    // The degree-2 cofactor `k^2*r^k` is now closed by its own sibling builder
    // (see the quadratic-geometric contract test); a lower bound >= 2 still
    // stays a residual (needs a symbolic head correction).
    assert_eq!(r("sum(k*r^k, k, 2, n)"), "sum(k·r^k, k, 2, n)");
    // The pure geometric sum (cycle sibling) is unaffected.
    assert_eq!(r("sum(r^k, k, 0, n)"), "(r^(n + 1) - 1) / (r - 1)");
}

#[test]
fn test_eval_ln_quotient_change_of_base_folds_fractional_bases() {
    // `ln(c)/ln(b) = log_b(c)` now folds for reciprocal/fractional rationals
    // (a negative rational), not just integer-power pairs. It used to leak
    // `ln(8)/ln(1/2)` into a solve boundary as `(ln(8)/ln(1/2), inf)` instead of
    // the folded `(-3, inf)`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("ln(8)/ln(1/2)"), "-3");
    assert_eq!(r("ln(1/8)/ln(2)"), "-3");
    assert_eq!(r("ln(1/4)/ln(2)"), "-2");
    assert_eq!(r("ln(16)/ln(1/2)"), "-4");
    // Integer-power pairs still fold; irrational ratios still decline.
    assert_eq!(r("ln(8)/ln(2)"), "3");
    assert_eq!(r("ln(8)/ln(4)"), "3/2");
    assert_eq!(r("ln(7)/ln(2)"), "ln(7) / ln(2)");
    // The exponential-inequality boundary now folds to the exact rational.
    assert_eq!(r("solve((1/2)^x < 8, x)"), "(-3, infinity)");
    assert_eq!(r("solve((1/3)^x > 9, x)"), "(-infinity, -2)");
}

#[test]
fn test_eval_finite_quadratic_geometric_sum_with_symbolic_ratio() {
    // `sum(k^2*r^k)` with a symbolic ratio: the numeric arithmetic-geometric
    // builder handles a rational ratio, but the symbolic case declined. It now
    // emits the `(1-r)^3` closed form (verified numerically: at r=2, n=3 -> 90).
    // The formula is large, so the SumRule must be budget-exempt or it is dropped.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(
        r("sum(k^2*r^k, k, 1, n)"),
        "r·(r^(n + 1)·(2·n^2 + 2·n - 1) + r + 1 - n^2·r^(n + 2) - r^n·(n + 1)^2) / (1 - r)^3"
    );
    assert_eq!(
        r("sum(k^2*x^k, k, 0, n)"),
        "x·(x^(n + 1)·(2·n^2 + 2·n - 1) + x + 1 - n^2·x^(n + 2) - x^n·(n + 1)^2) / (1 - x)^3"
    );
    // Siblings unchanged: degree-1 arith-geo, pure geometric, numeric ratio,
    // Faulhaber, and numeric-bound sums.
    assert_eq!(
        r("sum(k*r^k, k, 1, n)"),
        "r·(n·r^(n + 1) + r^n·(-n - 1) + 1) / (1 - r)^2"
    );
    assert_eq!(r("sum(r^k, k, 0, n)"), "(r^(n + 1) - 1) / (r - 1)");
    assert_eq!(r("sum(k^2, k, 1, n)"), "1/6·n·(n + 1)·(2·n + 1)");
    assert_eq!(r("sum(k, k, 1, 10)"), "55");
    // A degree-3 cofactor stays a residual.
    assert_eq!(r("sum(k^3*r^k, k, 1, n)"), "sum(k^3·r^k, k, 1, n)");
}

#[test]
fn test_eval_reciprocal_root_laurent_equation_solves() {
    // A Laurent polynomial in `sqrt(x)` — a root mixed with its reciprocal — used
    // to leak a malformed `solve(x - (x^(-1/2)+1)^(1/(1/2)))` residual. It now
    // substitutes `u = x^(1/q)`, clears the `1/u^k`, and solves. `√x - 1/√x = 1`
    // is `u^2 - u - 1 = 0`, so `u = φ` (the negative surd root is dropped since
    // √x >= 0), giving `x = φ^2 = 1 + φ`.
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(sqrt(x) - 1/sqrt(x) = 1, x)"), "{ 1 + phi }");
    assert_eq!(r("solve(sqrt(x) + 1/sqrt(x) = 5/2, x)"), "{ 1/4, 4 }");
    assert_eq!(r("solve(2*sqrt(x) - 3/sqrt(x) = 1, x)"), "{ 9/4 }");
    assert_eq!(r("solve(sqrt(x) - 2/sqrt(x) = 0, x)"), "{ 2 }");
    // A double root and a genuinely empty case.
    assert_eq!(r("solve(sqrt(x) + 4/sqrt(x) = 4, x)"), "{ 4 }");
    assert_eq!(r("solve(sqrt(x) + 1/sqrt(x) = 1, x)"), "No solution");
    // No regression: pure-positive-power forms keep the sibling handler, plain
    // polynomials and Laurent-in-x are untouched.
    assert_eq!(r("solve(x - 3*sqrt(x) + 2 = 0, x)"), "{ 1, 4 }");
    assert_eq!(r("solve(x^(2/3) - x^(1/3) - 2 = 0, x)"), "{ -1, 8 }");
    assert_eq!(r("solve(1/x + x = 5/2, x)"), "{ 1/2, 2 }");
}

#[test]
fn test_eval_reciprocal_root_laurent_combined_fraction_and_higher_roots() {
    // `simplify` combines the reciprocal-root Laurent over a common denominator
    // (`x^(1/3) − 1/x^(1/3) → (x^(4/3) − x^(2/3))/x`) or renders a term as
    // `x^(2/3)/x`. Handling the top-level `Div(N, x^m)` and the term-level
    // `x^a/x^b` closes the cube/fourth-root reciprocal family (odd roots keep the
    // negative solution).
    let r = |input: &str| -> String {
        let out = cli()
            .args(["eval", input, "--format", "json"])
            .output()
            .expect("Failed to run CLI");
        let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
        wire["result"].as_str().unwrap_or("").to_string()
    };
    assert_eq!(r("solve(x^(1/3) - 1/x^(1/3) = 0, x)"), "{ -1, 1 }");
    assert_eq!(r("solve(x^(1/3) - 2/x^(1/3) = 1, x)"), "{ -1, 8 }");
    assert_eq!(r("solve(x^(1/3) + 1/x^(1/3) = 5/2, x)"), "{ 1/8, 8 }");
    // An even root drops the negative branch (x^(1/4) >= 0).
    assert_eq!(r("solve(x^(1/4) - 1/x^(1/4) = 0, x)"), "{ 1 }");
    // The cycle-sibling sqrt Pow-sum forms remain correct.
    assert_eq!(r("solve(sqrt(x) + 1/sqrt(x) = 5/2, x)"), "{ 1/4, 4 }");
    // No regression: an ordinary rational `(x^2-1)/x = 0` is untouched.
    assert_eq!(r("solve((x^2-1)/x = 0, x)"), "{ -1, 1 }");
}
