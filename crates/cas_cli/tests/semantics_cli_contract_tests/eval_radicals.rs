use super::*;

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
        vec!["b > 0", "b ≠ 1"]
    );
}
#[test]
fn eval_general_base_log_sqrt_keeps_nontrivial_positive_argument_warning() {
    let (output, _code) = run_cli(&["eval", "log(b, sqrt(u))", "--format", "json"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "1/2·log(b, u)");
    let warnings = wire["warnings"].as_array().expect("warnings array");
    assert_eq!(warnings.len(), 0);

    let required = wire["required_display"]
        .as_array()
        .expect("required display");
    assert_eq!(
        required
            .iter()
            .filter_map(|item| item.as_str())
            .collect::<Vec<_>>(),
        vec!["b > 0", "b ≠ 1", "u > 0"]
    );
}
#[test]
fn eval_diff_rejects_static_and_empty_real_log_root_domains() {
    for expr in [
        "diff(ln(0), x)",
        "diff(log2(0), x)",
        "diff(log(2, 0), x)",
        "diff(log(1, 2), x)",
        "diff(log(1, x), x)",
        "diff(log(-2, x), x)",
        "diff(sqrt(-1), x)",
        "diff(sqrt(-x^2-1), x)",
    ] {
        let (output, code) = run_cli(&["eval", expr, "--format", "json", "--steps", "on"]);
        assert_eq!(code, 0, "unexpected command failure for {expr}: {output}");
        let wire = parse_wire(&output);

        assert_eq!(wire["result"], "undefined", "unexpected result for {expr}");
        assert_eq!(wire["required_display"], json!([]));
        assert!(
            !output.contains("Usar regla de sqrt(u)")
                && !output.contains("Usar regla de ln(u)")
                && !output.contains("1/(x · ln(1))")
                && !output.contains("Identificar u y du"),
            "empty real domain should not expose a calculus chain rule for {expr}: {output}"
        );
    }

    for expr in [
        "diff(sqrt(0), x)",
        "diff(sqrt(4), x)",
        "diff(ln(1), x)",
        "diff(log2(1), x)",
        "diff(log(2, 1), x)",
        "diff(log(2, x), x)",
    ] {
        let (output, code) = run_cli(&["eval", expr, "--format", "json", "--steps", "on"]);
        assert_eq!(code, 0, "unexpected command failure for {expr}: {output}");
        let wire = parse_wire(&output);

        if expr == "diff(log(2, x), x)" {
            assert_eq!(
                wire["result"], "1 / (x·ln(2))",
                "unexpected result for {expr}"
            );
            assert_eq!(wire["required_display"], json!(["x > 0"]));
        } else {
            assert_eq!(wire["result"], "0", "unexpected result for {expr}");
            assert_eq!(wire["required_display"], json!([]));
        }
    }
}
#[test]
fn eval_solve_surd_inequality_steps_drop_reorder_negation_cleanup_noise() {
    // Solving a constant-surd inequality used to flatten the solver's internal
    // scratch (reorder/negation cleanups, candidate arithmetic) into the wire
    // `steps` via the ENGINE-EVENT path. Solve now narrates exclusively through
    // its structured `solve_steps` channel and the engine listener stays parked
    // for the whole solve span, so NO engine rewrite step may leak — the
    // original complaint (the `Quitar paréntesis…` noise chain) is subsumed.
    let (output, code) = run_cli(&[
        "eval",
        "solve(sqrt(19)-sqrt(17)+x > sqrt(21)-sqrt(19)+x, x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0, "output: {output}");
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "All real numbers");
    // An all-clean wire omits the `steps` key entirely (empty vecs are skipped).
    let steps = wire["steps"].as_array().cloned().unwrap_or_default();
    assert!(
        steps.is_empty(),
        "solve must not leak engine rewrite steps into the wire steps channel: {steps:?}"
    );
}
#[test]
fn eval_collapsed_shifted_root_square_difference_collapses_to_zero() {
    let (output, code) = run_cli(&[
        "eval",
        "sqrt(x + 2*sqrt(x-1)) - sqrt(x-1) - 1",
        "--format",
        "json",
    ]);
    assert_eq!(
        code, 0,
        "expected successful CLI exit, got {code} with output: {output}"
    );

    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "0");
}
#[test]
fn eval_phi_quadratic_surd_identities_decide_exactly() {
    // φ = 1/2 + (1/2)·√5 lives in the exact surd layer (as_linear_surd), so
    // φ-vs-√5 identities decide exactly instead of lingering symbolically.
    for (input, expected) in [
        ("sqrt(5) - (2*phi - 1)", "0"),
        ("phi + 1/phi", "sqrt(5)"),
        ("2*phi - 1 - sqrt(5)", "0"),
        ("4*phi - 2", "2·sqrt(5)"),
        ("2*phi - 1 - sqrt(20)/2", "0"),
        // Guards: the φ display canonical survives where the surd form is no
        // simpler, and the lone constant is untouched.
        ("1 + phi", "1 + phi"),
        ("phi", "phi"),
        ("(-phi)", "-phi"),
        // The old minimal-polynomial identities keep working.
        ("phi^2 - phi - 1", "0"),
        ("phi - (1+sqrt(5))/2", "0"),
    ] {
        let (output, code) = run_cli(&["eval", input, "--format", "json"]);
        assert_eq!(code, 0, "{input}: {output}");
        let wire = parse_wire(&output);
        assert_eq!(wire["result"], expected, "{input}");
    }

    // Equivalence is now sound in BOTH directions: the true identity holds
    // and near-misses stay refuted (no overreach).
    for (input, expected) in [
        ("equiv(2*phi - 1, sqrt(5))", "true"),
        ("equiv(phi, (1+sqrt(5))/2)", "true"),
        ("equiv(4*phi - 2, sqrt(20))", "true"),
        ("equiv(2*phi, sqrt(5))", "false"),
        ("equiv(phi, sqrt(5))", "false"),
    ] {
        let (output, code) = run_cli(&["eval", input, "--format", "json"]);
        assert_eq!(code, 0, "{input}: {output}");
        let wire = parse_wire(&output);
        assert_eq!(wire["result"], expected, "{input}");
    }

    // Decimal display treats φ like π/e (D5 gate).
    let (output, code) = run_cli(&[
        "eval",
        "phi",
        "--numeric-display",
        "decimal",
        "--format",
        "json",
    ]);
    assert_eq!(code, 0, "output: {output}");
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "1.61803398875");
}
