use super::*;

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
