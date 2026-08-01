use super::*;

#[test]
fn integrate_contract_linear_elementary_by_parts_narrates_u_dv_du_v() {
    // The polynomial(linear) * {exp,sin,cos,sinh} family now narrates the full
    // by-parts choice (u = polynomial, dv = elementary factor), mirroring the
    // log narrator with the opposite u/dv assignment.
    for (input, dv_latex, v_latex) in [
        ("integrate(x*cos(x), x)", "\\cos(x)", "\\sin(x)"),
        ("integrate(x*sin(x), x)", "\\sin(x)", "-\\cos(x)"),
        ("integrate(x*exp(x), x)", "{e}^{x}", "{e}^{x}"),
        ("integrate(x*sinh(x), x)", "\\sinh(x)", "\\cosh(x)"),
    ] {
        let substeps = integration_substeps(input);
        assert_eq!(
            substep_after_latex(&substeps, "Elegir u y dv"),
            Some(format!("u = x,\\; dv = {dv_latex}\\,dx").as_str()),
            "u/dv narration mismatch for {input}, got {substeps:?}"
        );
        assert_eq!(
            substep_after_latex(&substeps, "Calcular du y v"),
            Some(format!("du = 1\\,dx,\\; v = {v_latex}").as_str()),
            "du/v narration mismatch for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|s| s["title"] == "Aplicar la fórmula de integración por partes"),
            "missing apply-formula substep for {input}, got {substeps:?}"
        );
    }

    // Affine inner argument: v carries the 1/2 chain factor; non-unit du shows.
    let affine = integration_substeps("integrate((2*x+3)*exp(x), x)");
    assert_eq!(
        substep_after_latex(&affine, "Elegir u y dv"),
        Some("u = 2\\cdot x + 3,\\; dv = {e}^{x}\\,dx"),
        "u/dv for (2x+3)exp(x), got {affine:?}"
    );
    assert_eq!(
        substep_after_latex(&affine, "Calcular du y v"),
        Some("du = 2\\,dx,\\; v = {e}^{x}"),
        "du/v for (2x+3)exp(x), got {affine:?}"
    );

    // Regression: the ln family keeps ITS narration (u = ln), not duplicated by
    // the new poly-elementary narrator.
    let log = integration_substeps("integrate(x*ln(x), x)");
    assert_eq!(
        substep_after_latex(&log, "Elegir u y dv"),
        Some("u = \\ln(x),\\; dv = x\\,dx"),
        "ln by-parts narration regressed, got {log:?}"
    );

    // The repeated degree>=2 case now unrolls each by-parts application: the
    // master title stays "repetida" and the FIRST "Elegir u y dv" chooses the
    // full polynomial (u = x^2), with v = e^x and du = 2x.
    let repeated = integration_substeps("integrate(x^2*exp(x), x)");
    assert!(
        repeated
            .iter()
            .any(|s| s["title"] == "Usar integración por partes repetida"),
        "expected repeated title for x^2*exp(x), got {repeated:?}"
    );
    assert_eq!(
        substep_after_latex(&repeated, "Elegir u y dv"),
        Some("u = {x}^{2},\\; dv = {e}^{x}\\,dx"),
        "repeated case must now narrate the first u/dv, got {repeated:?}"
    );
    assert_eq!(
        substep_after_latex(&repeated, "Calcular du y v"),
        Some("du = 2\\cdot x\\,dx,\\; v = {e}^{x}"),
        "repeated case must narrate the first du/v, got {repeated:?}"
    );

    // No polynomial factor: the new narrator must not fire (different route).
    let product = integration_substeps("integrate(cos(x)*exp(x), x)");
    assert!(
        substep_after_latex(&product, "Elegir u y dv").is_none(),
        "cos(x)*exp(x) must not get poly-elementary narration, got {product:?}"
    );

    // Results are untouched (presentation-only change).
    assert_antiderivative_verifies("integrate(x*cos(x), x)");
    assert_antiderivative_verifies("integrate(x*exp(x), x)");
    assert_antiderivative_verifies("integrate((2*x+3)*exp(x), x)");
}
#[test]
fn integrate_contract_single_inverse_by_parts_narrates_u_dv_du_v() {
    // A bare inverse function integrates by parts with u = f(x), dv = dx, so
    // v = x and du = f'(x) dx. The narration now exposes that choice for the
    // arc-trig and inverse-hyperbolic family.
    for (input, u_latex, du_latex) in [
        (
            "integrate(arctan(x), x)",
            "\\arctan(x)",
            "\\frac{1}{{x}^{2} + 1}",
        ),
        (
            "integrate(arcsin(x), x)",
            "\\arcsin(x)",
            "{(1 - {x}^{2})}^{-\\frac{1}{2}}",
        ),
        (
            "integrate(arccos(x), x)",
            "\\arccos(x)",
            "-{(1 - {x}^{2})}^{-\\frac{1}{2}}",
        ),
        (
            "integrate(asinh(x), x)",
            "\\operatorname{asinh}(x)",
            "{({x}^{2} + 1)}^{-\\frac{1}{2}}",
        ),
    ] {
        let substeps = integration_substeps(input);
        assert_eq!(
            substep_after_latex(&substeps, "Elegir u y dv"),
            Some(format!("u = {u_latex},\\; dv = dx").as_str()),
            "u/dv narration mismatch for {input}, got {substeps:?}"
        );
        assert_eq!(
            substep_after_latex(&substeps, "Calcular du y v"),
            Some(format!("du = {du_latex}\\,dx,\\; v = x").as_str()),
            "du/v narration mismatch for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|s| s["title"] == "Aplicar la fórmula de integración por partes"),
            "missing apply-formula substep for {input}, got {substeps:?}"
        );
    }

    // Affine argument: u carries the inner, v stays x, du folds the chain factor.
    let affine = integration_substeps("integrate(atan(2*x+1), x)");
    assert_eq!(
        substep_after_latex(&affine, "Elegir u y dv"),
        Some("u = \\arctan(2\\cdot x + 1),\\; dv = dx"),
        "u/dv for atan(2x+1), got {affine:?}"
    );

    // The new narrator must not fire on a product (owned by the other narrators)
    // and must leave plain trig/exp (which integrate directly) un-narrated.
    let product = integration_substeps("integrate(x*cos(x), x)");
    assert_eq!(
        substep_after_latex(&product, "Elegir u y dv"),
        Some("u = x,\\; dv = \\cos(x)\\,dx"),
        "x*cos(x) must keep its poly-elementary narration, got {product:?}"
    );
    assert!(
        integration_substeps("integrate(cos(x), x)")
            .iter()
            .all(|s| s["title"] != "Elegir u y dv"),
        "plain cos(x) integrates directly and must not be narrated by parts"
    );

    // Results are untouched (presentation-only change).
    assert_antiderivative_verifies("integrate(arctan(x), x)");
    assert_antiderivative_verifies("integrate(arcsin(x), x)");
}
#[test]
fn integrate_contract_bare_logarithm_by_parts_narrates_u_dv_du_v() {
    // A bare ln(affine) -- previously emitting NO by-parts substeps -- now
    // narrates u = ln, dv = dx, v = x, du = (ln arg)' dx via the single-inverse
    // narrator (the bare-ln by-parts gate makes the title fire).
    for (input, u_latex, du_latex) in [
        ("integrate(ln(x), x)", "\\ln(x)", "\\frac{1}{x}"),
        (
            "integrate(ln(2*x+1), x)",
            "\\ln(2\\cdot x + 1)",
            "\\frac{2}{2\\cdot x + 1}",
        ),
    ] {
        let substeps = integration_substeps(input);
        assert_eq!(
            substep_after_latex(&substeps, "Elegir u y dv"),
            Some(format!("u = {u_latex},\\; dv = dx").as_str()),
            "u/dv narration mismatch for {input}, got {substeps:?}"
        );
        assert_eq!(
            substep_after_latex(&substeps, "Calcular du y v"),
            Some(format!("du = {du_latex}\\,dx,\\; v = x").as_str()),
            "du/v narration mismatch for {input}, got {substeps:?}"
        );
    }

    // Regression: x*ln(x) keeps its polynomial*ln narration (u = ln, dv = x dx,
    // v = x^2/2), NOT the bare-ln dv = dx, proving the new branch did not steal it.
    let product = integration_substeps("integrate(x*ln(x), x)");
    assert_eq!(
        substep_after_latex(&product, "Elegir u y dv"),
        Some("u = \\ln(x),\\; dv = x\\,dx"),
        "x*ln(x) must keep its poly*ln narration, got {product:?}"
    );

    // Results untouched.
    assert_eq!(
        evaluated_expr_with_required_conditions("integrate(ln(x), x)").0,
        "x * ln(x) - x"
    );
}
#[test]
fn integrate_contract_quadratic_exp_by_parts_exposes_didactic_substep() {
    let input = "integrate(x^2*exp(x), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(wire["result"], "e^x·(x^2 + 2 - 2·x)");
    assert!(
        !stderr.contains("depth_overflow"),
        "quadratic exp by-parts didactic trace should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar integración por partes repetida"),
        "expected repeated integration-by-parts substep, got {substeps:?}"
    );
    // x^2 e^x reduces in two by-parts applications, so the repeated narration
    // unrolls two "Elegir u y dv"/"Aplicar la fórmula" blocks and closes with
    // TWO honest sub-steps (C1.6; the audit's P0 witness published
    // `∫−2·sin(x)dx ⟹ 2x·sin(x) + (2−x²)cos(x)` here): the remaining-term
    // closer integrates ITS OWN integrand, and a separate recomposition
    // assembles the boundary pieces into the engine's answer, gated on a
    // PROVED equality.
    let count_title = |title: &str| {
        substeps
            .iter()
            .filter(|substep| substep["title"] == title)
            .count()
    };
    assert_eq!(count_title("Elegir u y dv"), 2, "got {substeps:?}");
    assert_eq!(
        count_title("Aplicar la fórmula de integración por partes"),
        2,
        "got {substeps:?}"
    );
    let first_choice = substeps
        .iter()
        .find(|substep| substep["title"] == "Elegir u y dv")
        .expect("first u/dv choice");
    assert_eq!(
        first_choice["after_latex"], "u = {x}^{2},\\; dv = {e}^{x}\\,dx",
        "got {substeps:?}"
    );
    let remaining = substeps
        .iter()
        .find(|substep| substep["title"] == "Integrar el término restante")
        .expect("remaining-term closer");
    assert_eq!(
        remaining["after_latex"], "2\\cdot {e}^{x}",
        "the closer must integrate its OWN integrand (∫2·e^x = 2·e^x), never \
         quote the whole answer, got {substeps:?}"
    );
    let closer = substeps
        .last()
        .expect("repeated narration should not be empty");
    assert_eq!(closer["title"], "Recomponer las piezas de por partes");
    assert_eq!(
        closer["before_latex"], "{x}^{2}\\cdot {e}^{x} - 2\\cdot x\\cdot {e}^{x} + 2\\cdot {e}^{x}",
        "the recomposition assembles the alternating boundary pieces, got {substeps:?}"
    );
    assert_eq!(
        closer["after_latex"], "e^{x}\\cdot ({x}^{2} + 2 - 2\\cdot x)",
        "the recomposition (not the remaining-term closer) lands on the final \
         antiderivative, got {substeps:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_repeated_trig_by_parts_exposes_didactic_substep() {
    for (input, expected_result) in [
        ("integrate(x^2*sin(x), x)", "2·x·sin(x) + (2 - x^2)·cos(x)"),
        ("integrate(x^2*cos(x), x)", "2·x·cos(x) + (x^2 - 2)·sin(x)"),
        (
            "integrate(x^3*sin(x), x)",
            "(6·x - x^3)·cos(x) + (3·x^2 - 6)·sin(x)",
        ),
        (
            "integrate(x^3*cos(x), x)",
            "(x^3 - 6·x)·sin(x) + (3·x^2 - 6)·cos(x)",
        ),
        (
            "integrate(x^4*sin(2*x+1), x)",
            "(x^3 - 3/2·x)·sin(2·x + 1) + (-1/2·x^4 + 3/2·x^2 - 3/4)·cos(2·x + 1)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result);
        assert!(
            !stderr.contains("depth_overflow"),
            "repeated trig by-parts didactic trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Usar integración por partes repetida"),
            "expected repeated integration-by-parts substep for {input}, got {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_linear_by_parts_exposes_didactic_substep() {
    for (input, expected_result) in [
        ("integrate(x*exp(x), x)", "(x - 1)·e^x"),
        ("integrate(x*sin(x), x)", "sin(x) - x·cos(x)"),
        ("integrate(x*cos(x), x)", "cos(x) + x·sin(x)"),
        ("integrate(x*sinh(x), x)", "x·cosh(x) - sinh(x)"),
        ("integrate(x*cosh(x), x)", "x·sinh(x) - cosh(x)"),
        ("integrate((2*x+3)*exp(2*x+1), x)", "(x + 1)·e^(2·x + 1)"),
        (
            "integrate((2*x+3)*sin(2*x+1), x)",
            "1/2·sin(2·x + 1) - (cos(2·x + 1)·(2·x + 3))/2",
        ),
        (
            "integrate((2*x+3)*cos(2*x+1), x)",
            "1/2·cos(2·x + 1) + (sin(2·x + 1)·(2·x + 3))/2",
        ),
        (
            "integrate((2*x+3)*sinh(2*x+1), x)",
            "(cosh(2·x + 1)·(2·x + 3))/2 - 1/2·sinh(2·x + 1)",
        ),
        (
            "integrate((2*x+3)*cosh(2*x+1), x)",
            "(sinh(2·x + 1)·(2·x + 3))/2 - 1/2·cosh(2·x + 1)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result);
        assert!(
            !stderr.contains("depth_overflow"),
            "linear by-parts didactic trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        if input.contains("2*x+3")
            && ["exp", "sin", "cos", "sinh", "cosh"]
                .iter()
                .any(|kernel| input.contains(kernel))
        {
            assert_eq!(
                steps.first().and_then(|step| step["rule"].as_str()),
                Some("Calcular la integral"),
                "affine by-parts should not expand before integration for {input}"
            );
            assert!(
                !steps
                    .iter()
                    .any(|step| step["rule"] == "Expandir la expresión"),
                "affine by-parts should preserve compact presentation for {input}, got {steps:?}"
            );
        }
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Usar integración por partes"),
            "expected integration-by-parts substep for {input}, got {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_inverse_trig_by_parts_exposes_didactic_substep() {
    for (input, expected_result, expected_required_display) in [
        (
            "integrate(arcsin(x), x)",
            "sqrt(1 - x^2) + x·arcsin(x)",
            serde_json::json!(["-1 < x < 1"]),
        ),
        (
            "integrate(arccos(x), x)",
            "x·arccos(x) - sqrt(1 - x^2)",
            serde_json::json!(["-1 < x < 1"]),
        ),
        (
            "integrate(arctan(x), x)",
            "-1/2·ln(x^2 + 1) + x·arctan(x)",
            serde_json::json!([]),
        ),
        (
            "integrate(arctan(1/(2*x+1)), x)",
            "1/4·ln((2·x + 1)^2 + 1) + 1/2·(2·x + 1)·arctan(1 / (2·x + 1))",
            serde_json::json!(["x ≠ -1/2"]),
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "inverse-trig by-parts trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        assert_eq!(
            steps.len(),
            1,
            "expected compact direct integration trace for {input}, got {steps:?}"
        );
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Usar integración por partes"),
            "expected integration-by-parts substep for {input}, got {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_inverse_hyperbolic_affine_by_parts_exposes_didactic_substep() {
    for (input, expected_result, expected_required_display) in [
        (
            "integrate(asinh(2*x+1), x)",
            "1/2·(2·x + 1)·asinh(2·x + 1) - 1/2·sqrt((2·x + 1)^2 + 1)",
            serde_json::json!([]),
        ),
        (
            "integrate(atanh(2*x+1), x)",
            "1/4·ln(1 - (2·x + 1)^2) + 1/2·(2·x + 1)·atanh(2·x + 1)",
            serde_json::json!(["-1 < x < 0"]),
        ),
        (
            "integrate(acosh(2*x+1), x)",
            "1/2·(2·x + 1)·acosh(2·x + 1) - 1/2·sqrt(2·x)·sqrt(2·x + 2)",
            serde_json::json!(["x > 0"]),
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "inverse-hyperbolic by-parts trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        assert_eq!(
            steps.len(),
            1,
            "expected compact direct integration trace for {input}, got {steps:?}"
        );
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Usar integración por partes"),
            "expected integration-by-parts substep for {input}, got {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_polynomial_derivative_substitution_exposes_didactic_substep() {
    for (input, expected_result, expected_substep_title) in [
        (
            "integrate(2*x*exp(x^2), x)",
            "e^(x^2)",
            "Usar la regla de exp(u) -> exp(u)",
        ),
        (
            "integrate(2*x*cos(x^2), x)",
            "sin(x^2)",
            "Usar la regla de cos(u) -> sin(u)",
        ),
        (
            "integrate(2*x*sin(x^2), x)",
            "-cos(x^2)",
            "Usar la regla de sin(u) -> -cos(u)",
        ),
        (
            "integrate(2*x*sinh(x^2), x)",
            "cosh(x^2)",
            "Usar la regla de sinh(u) -> cosh(u)",
        ),
        (
            "integrate(2*x*cosh(x^2), x)",
            "sinh(x^2)",
            "Usar la regla de cosh(u) -> sinh(u)",
        ),
        (
            "integrate(2*x*tanh(x^2), x)",
            "ln(cosh(x^2))",
            "Usar la regla de tanh(u) -> ln(cosh(u))",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert!(
            wire["required_display"]
                .as_array()
                .expect("required_display should be an array")
                .is_empty(),
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "polynomial derivative substitution trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_substep_title),
            "expected {expected_substep_title} substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected concrete u/du substep for {input}, got {substeps:?}"
        );
        let u_du_substep = substeps
            .iter()
            .find(|substep| substep["title"] == "Identificar u y du")
            .expect("expected concrete u/du substep");
        let before_latex = u_du_substep["before_latex"]
            .as_str()
            .expect("u/du substep should expose before_latex");
        let after_latex = u_du_substep["after_latex"]
            .as_str()
            .expect("u/du substep should expose after_latex");
        assert!(
            before_latex.contains("u =") && after_latex.contains("du ="),
            "u/du substep should label substitution evidence for {input}, got {u_du_substep:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "polynomial derivative substitution should not use only the generic substitution substep for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_log_power_product_substitution_exposes_didactic_substep() {
    for (input, expected_result, expected_substep_title) in [
        (
            "integrate(2*x*ln(x^2+1), x)",
            "(ln(x^2 + 1) - 1)·(x^2 + 1)",
            "Usar la regla de u'·ln(u) -> u·(ln(u)-1)",
        ),
        (
            "integrate(2*x*ln(x^2+1)^2, x)",
            "(x^2 + 1)·(ln(x^2 + 1)^2 - 2·ln(x^2 + 1) + 2)",
            "Usar la regla de u'·ln(u)^n por partes",
        ),
        (
            "integrate(2*x*ln(x^2+1)^3, x)",
            "(ln(x^2 + 1)^3 - 3·ln(x^2 + 1)^2 + 6·ln(x^2 + 1) - 6)·(x^2 + 1)",
            "Usar la regla de u'·ln(u)^n por partes",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert!(
            wire["required_display"]
                .as_array()
                .expect("required_display should be an array")
                .is_empty(),
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "log-power product substitution trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        assert_eq!(
            steps.len(),
            1,
            "expected compact direct substitution trace for {input}, got {steps:?}"
        );
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_substep_title),
            "expected log-power product table substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected concrete u/du substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "log-power product table case should not use generic substitution substep for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_constant_base_log_power_product_substitution_exposes_didactic_substep() {
    for (input, expected_result, expected_required) in [
        (
            "integrate(2*x*log(2,x^2+1)^2, x)",
            "(x^2 + 1)·(log(2, x^2 + 1)^2 + 2 / ln(2)^2 - 2·log(2, x^2 + 1) / ln(2))",
            vec![],
        ),
        (
            "integrate(2*x*log2(x^2+1)^2, x)",
            "(x^2 + 1)·(log2(x^2 + 1)^2 + 2 / ln(2)^2 - 2·log2(x^2 + 1) / ln(2))",
            vec![],
        ),
        (
            "integrate(2*x*log(2,x^2-1)^2, x)",
            "(x^2 - 1)·(log(2, x^2 - 1)^2 + 2 / ln(2)^2 - 2·log(2, x^2 - 1) / ln(2))",
            vec!["x < -1 or x > 1"],
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"],
            serde_json::json!(expected_required),
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "constant-base log-power substitution should not emit depth_overflow for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Usar la regla de u'·log_b(u)^n por partes"),
            "expected constant-base log-power substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected concrete u/du substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "constant-base log-power table case should not use generic substitution for {input}: {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }

    let (invalid_base_result, invalid_base_required) =
        evaluated_integral_with_required_conditions("integrate(2*x*log(1,x^2+1)^2, x)");
    assert_eq!(
        invalid_base_result, "undefined",
        "invalid log base should produce an undefined integrand, not a residual"
    );
    assert!(
        invalid_base_required.is_empty(),
        "undefined invalid-base log integrand should not add assumptions: {invalid_base_required:?}"
    );

    let (symbolic_base_result, _required) =
        evaluated_integral_with_required_conditions("integrate(2*x*log(y,x^2+1)^2, x)");
    assert!(
        symbolic_base_result.starts_with("integrate("),
        "symbolic log base should remain residual, got {symbolic_base_result}"
    );
}
#[test]
fn integrate_contract_hyperbolic_quotient_substitution_exposes_didactic_substep() {
    for (input, expected_result, expected_required_display, expected_substep_title) in [
        (
            "integrate(2*x*cosh(x^2)/sinh(x^2), x)",
            "ln(|sinh(x^2)|)",
            serde_json::json!(["sinh(x^2) ≠ 0"]),
            "Usar la regla de cosh(u)/sinh(u) -> ln|sinh(u)|",
        ),
        (
            "integrate(2*x/tanh(x^2), x)",
            "ln(|sinh(x^2)|)",
            serde_json::json!(["sinh(x^2) ≠ 0"]),
            "Usar la regla de 1/tanh(u) -> ln|sinh(u)|",
        ),
        (
            "integrate(2*x/cosh(x^2)^2, x)",
            "tanh(x^2)",
            serde_json::json!([]),
            "Usar la regla de 1/cosh(u)^2 -> tanh(u)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "hyperbolic quotient substitution trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_substep_title),
            "expected hyperbolic quotient table substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected concrete u/du substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "expected specific hyperbolic quotient trace without generic substitution for {input}, got {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_trig_quotient_substitution_exposes_didactic_substep() {
    for (input, expected_result, expected_required_display, expected_substep_title) in [
        (
            "integrate(2*x*tan(x^2), x)",
            "-ln(|cos(x^2)|)",
            serde_json::json!(["cos(x^2) ≠ 0"]),
            "Usar la regla de tan(u) -> -ln|cos(u)|",
        ),
        (
            "integrate(3*x^2*cot(x^3), x)",
            "ln(|sin(x^3)|)",
            serde_json::json!(["sin(x^3) ≠ 0"]),
            "Usar la regla de cot(u) -> ln|sin(u)|",
        ),
        (
            "integrate(2*x/cos(x^2)^2, x)",
            "tan(x^2)",
            serde_json::json!(["cos(x^2) ≠ 0"]),
            "Usar la regla de 1/cos(u)^2 -> tan(u)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "trig quotient substitution trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_substep_title),
            "expected trig quotient table substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected u/du identification substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "expected specific trig quotient trace without generic substitution for {input}, got {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }

    let (wire, _stderr) =
        cli_eval_json_with_stderr_args("integrate(tan(x^2), x)", &["--steps", "on"]);
    assert_eq!(wire["result"], "integrate(tan(x^2), x)");
    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    assert!(
        !steps
            .iter()
            .any(|step| step["rule"] == "Calcular la integral"),
        "unsupported missing-cofactor tan(x^2) should not emit a fake integration step: {steps:?}"
    );
}
#[test]
fn integrate_contract_direct_sec_csc_derivative_quotients_expose_didactic_substep() {
    for (input, expected_result, expected_required_display) in [
        (
            "integrate(2*x*sin(x^2)/cos(x^2)^2, x)",
            "sec(x^2)",
            serde_json::json!(["cos(x^2) ≠ 0"]),
        ),
        (
            "integrate(3*x^2*cos(x^3)/sin(x^3)^2, x)",
            "-csc(x^3)",
            serde_json::json!(["sin(x^3) ≠ 0"]),
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "direct sec/csc derivative quotient trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        assert_eq!(
            steps.len(),
            1,
            "expected direct public integration step for {input}, got {steps:?}"
        );
        let substeps = steps[0]["substeps"]
            .as_array()
            .expect("direct integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Usar sustitución"),
            "expected substitution substep for {input}, got {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }

    let (wire, _stderr) =
        cli_eval_json_with_stderr_args("integrate(sin(x^2)/cos(x^2)^2, x)", &["--steps", "on"]);
    assert_eq!(wire["result"], "integrate(sin(x^2) / cos(x^2)^2, x)");
    if let Some(steps) = wire["steps"].as_array() {
        assert!(
            !steps.iter().any(|step| step["substeps"].is_array()),
            "unsupported missing-cofactor sec derivative quotient should not emit a fake substep: {steps:?}"
        );
    }
}
#[test]
fn integrate_contract_direct_trig_log_substitution_exposes_didactic_substep() {
    for (input, expected_result, expected_required_display, expected_substep_title) in [
        (
            "integrate(tan(2*x+1), x)",
            "-1/2·ln(|cos(2·x + 1)|)",
            serde_json::json!(["cos(2·x + 1) ≠ 0"]),
            "Usar la regla de tan(u) -> -ln|cos(u)|",
        ),
        (
            "integrate(cot(2*x+1), x)",
            "1/2·ln(|sin(2·x + 1)|)",
            serde_json::json!(["sin(2·x + 1) ≠ 0"]),
            "Usar la regla de cot(u) -> ln|sin(u)|",
        ),
        (
            "integrate(sec(2*x+1), x)",
            "1/2·ln(|tan(2·x + 1) + sec(2·x + 1)|)",
            serde_json::json!(["cos(2·x + 1) ≠ 0"]),
            "Usar la regla de sec(u) -> ln|sec(u)+tan(u)|",
        ),
        (
            "integrate(csc(2*x+1), x)",
            "1/2·ln(|csc(2·x + 1) - cot(2·x + 1)|)",
            serde_json::json!(["sin(2·x + 1) ≠ 0"]),
            "Usar la regla de csc(u) -> ln|csc(u)-cot(u)|",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "trig log substitution trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_substep_title),
            "expected {expected_substep_title} substep for {input}, got {substeps:?}"
        );
        if expected_substep_title != "Usar sustitución" {
            assert!(
                substeps
                    .iter()
                    .all(|substep| substep["title"] != "Usar sustitución"),
                "direct trig log table case should not use the generic substitution substep for {input}: {substeps:?}"
            );
            assert!(
                substeps
                    .iter()
                    .any(|substep| substep["title"] == "Identificar el argumento afín"),
                "expected affine argument substep for {input}, got {substeps:?}"
            );
        }
        assert!(
            steps.iter().filter(|step| step["rule"] != "Calcular la integral").all(|step| {
                step["substeps"]
                    .as_array()
                    .is_none_or(|substeps| {
                        !substeps
                            .iter()
                            .any(|substep| substep["title"] == "Usar sustitución")
                    })
            }),
            "non-integration prep steps should not get substitution substeps for {input}: {steps:?}"
        );
    }

    let (wire, _stderr) =
        cli_eval_json_with_stderr_args("integrate(tan(x^2), x)", &["--steps", "on"]);
    assert_eq!(wire["result"], "integrate(tan(x^2), x)");
    if let Some(steps) = wire["steps"].as_array() {
        assert!(
            steps.iter().all(|step| {
                step["substeps"].as_array().is_none_or(|substeps| {
                    substeps.iter().all(|substep| {
                        substep["title"] != "Usar sustitución"
                            && substep["title"] != "Identificar u y du"
                            && substep["title"] != "Usar la regla de tan(u) -> -ln|cos(u)|"
                    })
                })
            }),
            "unsupported nonlinear tan(x^2) should not emit a fake substitution substep: {steps:?}"
        );
        assert!(
            steps.iter().any(|step| {
                step["rule"] == "Conservar integral residual"
                    && step["substeps"].as_array().is_some_and(|substeps| {
                        substeps.iter().any(|substep| {
                            substep["title"] == "Registrar polo del integrando"
                                && substep["after_latex"] == "\\cos({x}^{2}) \\ne 0"
                        })
                    })
            }),
            "unsupported nonlinear tan(x^2) should expose only the residual domain policy: {steps:?}"
        );
    }
}
#[test]
fn integrate_contract_polynomial_base_substitution_exposes_didactic_substep() {
    for (input, expected_result, expected_required_display, expected_rule_title) in [
        (
            "integrate((2*x+1)/(x^2+x-1), x)",
            "ln(|x^2 + x - 1|)",
            serde_json::json!(["x^2 + x - 1 ≠ 0"]),
            "Usar la regla de u'/u -> ln|u|",
        ),
        (
            "integrate((2*x+1)/(x^2+x-1)^3, x)",
            "-1 / (2·(x^2 + x - 1)^2)",
            serde_json::json!(["x^2 + x - 1 ≠ 0"]),
            "Usar la regla de u'/u^n -> u^(1-n)/(1-n)",
        ),
        (
            "integrate(x/sqrt(x^2+1), x)",
            "sqrt(x^2 + 1)",
            serde_json::json!([]),
            "Usar la regla de u'/sqrt(u) -> 2*sqrt(u)",
        ),
        (
            "integrate(2*x/sqrt(x^2-1), x)",
            "2·sqrt(x^2 - 1)",
            serde_json::json!(["x < -1 or x > 1"]),
            "Usar la regla de u'/sqrt(u) -> 2*sqrt(u)",
        ),
        (
            "integrate(2*x*(x^2-1)^(3/2), x)",
            "2/5·(x^2 - 1)^(5/2)",
            serde_json::json!(["x ≤ -1 or x ≥ 1"]),
            "Usar la regla de u'·u^p -> u^(p+1)/(p+1)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "polynomial-base substitution trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_rule_title),
            "expected polynomial-base table substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected concrete u/du substep for {input}, got {substeps:?}"
        );
        assert_u_du_substep_labels(substeps, input);
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "polynomial-base table case should not use generic substitution substep for {input}: {substeps:?}"
        );
        assert!(
            steps.iter().filter(|step| step["rule"] != "Calcular la integral").all(|step| {
                step["substeps"]
                    .as_array()
                    .is_none_or(|substeps| {
                        !substeps
                            .iter()
                            .any(|substep| substep["title"] == "Usar sustitución")
                    })
            }),
            "non-integration prep steps should not get substitution substeps for {input}: {steps:?}"
        );
    }

    for input in ["integrate(1/(x^2+1), x)", "integrate(1/sqrt(x^2+1), x)"] {
        let (wire, _stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        assert!(
            steps.iter().all(|step| {
                step["substeps"]
                    .as_array()
                    .is_none_or(|substeps| {
                        !substeps
                            .iter()
                            .any(|substep| substep["title"] == "Usar sustitución")
                    })
            }),
            "table-form integral without polynomial cofactor should not emit a fake substitution substep for {input}: {steps:?}"
        );
    }
}
#[test]
fn integrate_contract_nested_inverse_polynomial_substitution_exposes_didactic_substep() {
    for (input, expected_result, expected_required_display, expected_rule_title) in [
        (
            "integrate(2*x/sqrt(4-x^4), x)",
            "arcsin(x^2 / 2)",
            serde_json::json!(["4 - x^4 > 0"]),
            "Usar la regla de u'/sqrt(1-u^2) -> arcsin(u)",
        ),
        (
            "integrate(2*x/sqrt(1+x^4), x)",
            "asinh(x^2)",
            serde_json::json!([]),
            "Usar la regla de u'/sqrt(1+u^2) -> asinh(u)",
        ),
        (
            "integrate(2*x/(1+x^4), x)",
            "arctan(x^2)",
            serde_json::json!([]),
            "Usar la regla de u'/(1+u^2) -> arctan(u)",
        ),
        (
            "integrate(2*x/(4-x^4), x)",
            "1/2·atanh(x^2 / 2)",
            serde_json::json!(["4 - x^4 > 0"]),
            "Usar la regla de u'/(1-u^2) -> atanh(u)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "nested inverse polynomial substitution trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_rule_title),
            "expected nested inverse polynomial table substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Identificar u y du"),
            "expected concrete u/du substep for {input}, got {substeps:?}"
        );
        assert!(
            substeps
                .iter()
                .all(|substep| substep["title"] != "Usar sustitución"),
            "nested inverse polynomial table case should not use generic substitution substep for {input}: {substeps:?}"
        );
        assert!(
            steps.iter().filter(|step| step["rule"] != "Calcular la integral").all(|step| {
                step["substeps"]
                    .as_array()
                    .is_none_or(|substeps| {
                        !substeps
                            .iter()
                            .any(|substep| substep["title"] == "Usar sustitución")
                    })
            }),
            "non-integration prep steps should not get substitution substeps for {input}: {steps:?}"
        );
    }

    for input in [
        "integrate(1/(x^2+1), x)",
        "integrate(1/(1-x^2), x)",
        "integrate(1/(4-x^2), x)",
        "integrate(1/sqrt(x^2+1), x)",
        "integrate(1/sqrt(1-x^2), x)",
        "integrate(1/sqrt(4-(x+1)^2), x)",
        "integrate(1/sqrt(4+(x+1)^2), x)",
        "integrate(2/(1+(2*x+1)^2), x)",
    ] {
        let (wire, _stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        assert!(
            steps.iter().all(|step| {
                step["substeps"]
                    .as_array()
                    .is_none_or(|substeps| {
                        !substeps
                            .iter()
                            .any(|substep| substep["title"] == "Usar sustitución")
                    })
            }),
            "table or linear inverse primitive should not emit a fake substitution substep for {input}: {steps:?}"
        );
    }
}
#[test]
fn integrate_contract_log_by_parts_exposes_didactic_substep_and_keeps_compact_trace() {
    for (input, expected_result, expected_required) in [
        ("integrate(x*ln(x), x)", "1/4·x^2·(2·ln(x) - 1)", "x > 0"),
        (
            "integrate((2*x+1)*ln(2*x+1), x)",
            "1/4·((2·x + 1)^2·ln(2·x + 1) - 2·x^2 - 2·x)",
            "x > -1/2",
        ),
        (
            "integrate((x^2+x+1)*ln(2*x+1), x)",
            "(1/3·x^3 + 1/2·x^2 + x)·ln(2·x + 1) - 1/9·x^3 - 1/6·x^2 - 5/6·x + 5/12·ln(2·x + 1)",
            "x > -1/2",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
        assert!(
            stderr.is_empty(),
            "log by-parts presentation should stay quiet for {input}\nstderr:\n{stderr}"
        );
        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"],
            serde_json::json!([expected_required]),
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        assert_eq!(
            steps.len(),
            1,
            "expected compact direct integration trace for {input}, got {steps:?}"
        );
        assert!(
            steps
                .iter()
                .all(|step| step["rule"] != "Expandir la expresión"),
            "log by-parts trace should not expand before integrating for {input}, got {steps:?}"
        );
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == "Usar integración por partes"),
            "expected integration-by-parts substep for {input}, got {substeps:?}"
        );
        for title in [
            "Elegir u y dv",
            "Calcular du y v",
            "Aplicar la fórmula de integración por partes",
        ] {
            assert!(
                substeps.iter().any(|substep| substep["title"] == title),
                "expected concrete by-parts substep {title:?} for {input}, got {substeps:?}"
            );
        }

        let choice_substep = substeps
            .iter()
            .find(|substep| substep["title"] == "Elegir u y dv")
            .expect("expected u/dv choice substep");
        let choice_latex = choice_substep["after_latex"]
            .as_str()
            .expect("choice substep should expose after_latex");
        assert!(
            choice_latex.contains("\\ln") && choice_latex.contains("dv"),
            "choice substep should show concrete u and dv, got {choice_latex:?}"
        );

        let derivative_substep = substeps
            .iter()
            .find(|substep| substep["title"] == "Calcular du y v")
            .expect("expected du/v calculation substep");
        let derivative_latex = derivative_substep["after_latex"]
            .as_str()
            .expect("du/v substep should expose after_latex");
        assert!(
            derivative_latex.contains("du =") && derivative_latex.contains("v ="),
            "du/v substep should show concrete du and v, got {derivative_latex:?}"
        );
        if input == "integrate(x*ln(x), x)" {
            assert!(
                derivative_latex.contains("\\frac{1}{x}") && derivative_latex.contains("{x}^{2}"),
                "du/v substep should show concrete du and v, got {derivative_latex:?}"
            );
        } else {
            assert!(
                derivative_latex.contains("\\frac{2}{2\\cdot x + 1}"),
                "affine log by-parts substep should show concrete du, got {derivative_latex:?}"
            );
        }
    }
}
#[test]
fn integrate_contract_linear_partial_fraction_log_result_exposes_didactic_substep() {
    let input = "integrate(2/(1-(2*x+1)^2), x)";
    let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

    assert_eq!(wire["result"], "1/2·ln(|(x + 1) / x|)");
    assert_eq!(
        wire["required_display"],
        serde_json::json!(["x ≠ -1", "x ≠ 0"])
    );
    assert!(
        !stderr.contains("depth_overflow"),
        "partial fraction trace should not emit depth_overflow warning\nstderr:\n{stderr}"
    );

    let steps = wire["steps"]
        .as_array()
        .expect("steps should be present with --steps on");
    let integration_step = steps
        .iter()
        .find(|step| step["rule"] == "Calcular la integral")
        .expect("expected public symbolic integration step");
    let substeps = integration_step["substeps"]
        .as_array()
        .expect("integration step should expose didactic substeps");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Descomponer en fracciones parciales"),
        "expected partial-fraction didactic substep, got {substeps:?}"
    );
    let decomposition_substep = substeps
        .iter()
        .find(|substep| substep["title"] == "Descomponer en fracciones parciales")
        .expect("partial-fraction substep should exist");
    let decomposition_latex = decomposition_substep["after_latex"]
        .as_str()
        .expect("partial-fraction substep should expose concrete after_latex");
    assert!(
        decomposition_latex == "\\frac{1}{2\\cdot (x + 1)} - \\frac{1}{2\\cdot x}",
        "partial-fraction substep should show the decomposed simple fractions, got {decomposition_latex}"
    );
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Integrar los términos simples"),
        "expected simple-term integration substep, got {substeps:?}"
    );
    assert!(
        substeps
            .iter()
            .all(|substep| substep["title"] != "Usar sustitución"),
        "partial fractions should not be mislabeled as generic substitution: {substeps:?}"
    );
    assert_antiderivative_verifies(input);
}
#[test]
fn integrate_contract_sqrt_chain_substitutions_expose_didactic_substep() {
    for (input, expected_result, expected_required_display, expected_substep_title) in [
        (
            "integrate(sin(sqrt(x))/(sqrt(x)*cos(sqrt(x))^2), x)",
            "2·sec(sqrt(x))",
            serde_json::json!(["cos(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de sec(u)·tan(u) -> sec(u)",
        ),
        (
            "integrate(tan(sqrt(x))/sqrt(x), x)",
            "-2·ln(|cos(sqrt(x))|)",
            serde_json::json!(["cos(sqrt(x)) ≠ 0", "x > 0"]),
            "Usar la regla de tan(u) -> -ln|cos(u)|",
        ),
        (
            "integrate(tanh(sqrt(x))/sqrt(x), x)",
            "2·ln(cosh(sqrt(x)))",
            serde_json::json!(["x > 0"]),
            "Usar sustitución",
        ),
        (
            "integrate(1/(sqrt(x)*cosh(sqrt(x))^2), x)",
            "2·tanh(sqrt(x))",
            serde_json::json!(["x > 0"]),
            "Usar la regla de 1/cosh(u)^2 -> tanh(u)",
        ),
        (
            "integrate(sinh(sqrt(x))/(sqrt(x)*cosh(sqrt(x))^2), x)",
            "-2 / cosh(sqrt(x))",
            serde_json::json!(["x > 0"]),
            "Usar la regla de sinh(u)/cosh(u)^2 -> -1/cosh(u)",
        ),
    ] {
        let (wire, stderr) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);

        assert_eq!(wire["result"], expected_result, "input: {input}");
        assert_eq!(
            wire["required_display"], expected_required_display,
            "unexpected required_display for {input}: {:?}",
            wire["required_display"]
        );
        assert!(
            !stderr.contains("depth_overflow"),
            "sqrt-chain substitution trace should not emit depth_overflow warning for {input}\nstderr:\n{stderr}"
        );

        let steps = wire["steps"]
            .as_array()
            .expect("steps should be present with --steps on");
        assert_eq!(
            steps.len(),
            1,
            "expected compact direct substitution trace for {input}, got {steps:?}"
        );
        let integration_step = steps
            .iter()
            .find(|step| step["rule"] == "Calcular la integral")
            .expect("expected public symbolic integration step");
        let substeps = integration_step["substeps"]
            .as_array()
            .expect("integration step should expose didactic substeps");
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == expected_substep_title),
            "expected {expected_substep_title} substep for {input}, got {substeps:?}"
        );
        assert_antiderivative_verifies(input);
    }
}
#[test]
fn integrate_contract_abs_linear_definite_narrates_root_split() {
    // |linear| has no single elementary antiderivative, so the FTC narration
    // produces nothing; the root-split route must be narrated instead.
    // Root strictly inside the interval -> "Partir el intervalo en la raíz".
    let (wire, stderr) =
        cli_eval_json_with_stderr_args("integrate(abs(2*x-1), x, 0, 1)", &["--steps", "on"]);
    assert!(stderr.is_empty(), "no stderr expected: {stderr}");
    assert_eq!(wire["result"].as_str(), Some("1/2"));
    let step_text = wire["steps"].to_string();
    assert!(
        step_text.contains("Localizar la raíz del valor absoluto"),
        "abs-linear trace should locate the root: {step_text}"
    );
    assert!(
        step_text.contains("Partir el intervalo en la raíz"),
        "root inside the interval should narrate the split: {step_text}"
    );

    // Root outside the interval -> constant-sign narration, no split.
    let (wire_outside, _) =
        cli_eval_json_with_stderr_args("integrate(abs(x-1), x, 2, 5)", &["--steps", "on"]);
    assert_eq!(wire_outside["result"].as_str(), Some("15/2"));
    let outside_text = wire_outside["steps"].to_string();
    assert!(
        outside_text.contains("El interior mantiene signo constante"),
        "root outside the interval should narrate constant sign: {outside_text}"
    );
    assert!(
        !outside_text.contains("Partir el intervalo en la raíz"),
        "root outside the interval should not narrate a split: {outside_text}"
    );

    // A plain FTC definite integral keeps its antiderivative narration.
    let (wire_ftc, _) =
        cli_eval_json_with_stderr_args("integrate(x^2, x, 0, 1)", &["--steps", "on"]);
    let ftc_text = wire_ftc["steps"].to_string();
    assert!(
        ftc_text.contains("Hallar la antiderivada"),
        "ordinary definite integrals keep the FTC narration: {ftc_text}"
    );
}
/// `F(b) - F(a)` is an EXPRESSION, not two strings glued with a minus sign.
///
/// The regression this pins: the substep used to be built with
/// `format!("{} - {}", upper, lower)`, so when `F(a)` had two or more terms the
/// minus only reached the first one and the line published a FALSE identity —
/// `integrate(cos(t)^2, t, pi/6, pi/3)` displayed a value of `pi/4` while the
/// step itself answered `pi/12`. Building the `Sub` node lets the renderer place
/// the parentheses it already knows how to place (see
/// `cas_formatter::latex::test_latex_sub_with_add_rhs`).
#[test]
fn integrate_contract_definite_bounds_substep_subtracts_all_of_f_of_a() {
    // Finite bounds, multi-term F(a): the whole subtrahend must be grouped.
    assert_eq!(
        definite_bounds_substep_after_latex("integrate(cos(t)^2, t, pi/6, pi/3)"),
        "\\frac{\\sin(\\frac{2}{3}\\cdot \\pi)}{4} + \\frac{\\frac{\\pi}{3}}{2} \
         - (\\frac{\\sin(\\frac{2}{6}\\cdot \\pi)}{4} + \\frac{\\frac{\\pi}{6}}{2})"
    );

    // Single-term F(a) needs no grouping: the fix must not add noise.
    let single_term = definite_bounds_substep_after_latex("integrate(2*x, x, 1, 3)");
    assert!(
        !single_term.contains("- ("),
        "a one-term F(a) should not be parenthesized, got {single_term}"
    );

    // Improper bound: `lim` has no node to subtract from, so the operand is
    // delimited by hand — otherwise the limit appears to apply to the first
    // summand only (and each summand diverges on its own).
    let improper = definite_bounds_substep_after_latex("integrate(1/(x^4-1), x, 2, oo)");
    assert!(
        improper.starts_with("\\lim_{x \\to \\infty} \\left("),
        "the limit operand must be delimited, got {improper}"
    );
    assert!(
        improper.contains("\\right) - \\left("),
        "the subtrahend must be delimited when the limit branch is taken, got {improper}"
    );
}
/// A substep must not announce a manoeuvre it does not perform, and the thing
/// being decomposed is the RATIONAL FUNCTION, not its denominator.
///
/// Three regressions pinned at once (audit rows 031/032/033):
///  - over a denominator irreducible in Q (`x^3 - 2`), "factor the denominator"
///    returned the denominator itself and "decompose" returned the integrand;
///  - the decompose substep used the factored DENOMINATOR as its `before`, so
///    on `x^5 - 1` it asserted that a polynomial equals a sum of fractions;
///  - `∫dx/x` announced a partial-fraction decomposition of `1/x` into `1/x`.
#[test]
fn integrate_contract_partial_fraction_substeps_never_claim_a_manoeuvre_they_skip() {
    let titles = |input: &str| -> Vec<(String, String, String)> {
        let (wire, _) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
        wire["steps"]
            .as_array()
            .expect("steps with --steps on")
            .iter()
            .filter_map(|step| step["substeps"].as_array())
            .flatten()
            .map(|s| {
                (
                    s["title"].as_str().unwrap_or_default().to_string(),
                    s["before"].as_str().unwrap_or_default().to_string(),
                    s["after"].as_str().unwrap_or_default().to_string(),
                )
            })
            .collect()
    };

    // Irreducible in Q: no factorization claim, no identity "decomposition".
    let irreducible = titles("integrate(1/(x^3-2), x)");
    assert!(
        !irreducible
            .iter()
            .any(|(t, _, _)| t.contains("Factorizar") || t.contains("Descomponer")),
        "x^3 - 2 is irreducible in Q; neither claim may be published: {irreducible:?}"
    );

    // Real decomposition: the `before` is the integrand, not the denominator.
    let quintic = titles("integrate(1/(x^5-1), x)");
    let decompose = quintic
        .iter()
        .find(|(t, _, _)| t.contains("Descomponer"))
        .expect("x^5 - 1 does decompose");
    assert_eq!(decompose.1, "1 / (x^5 - 1)");

    // Already decomposed: only the honest table statement survives.
    let simple = titles("integrate(1/x, x)");
    assert!(
        !simple.iter().any(|(t, _, _)| t.contains("Descomponer")),
        "1/x is already decomposed: {simple:?}"
    );
    assert!(
        simple
            .iter()
            .any(|(_, b, a)| b == "1 / x" && a == "ln(|x|)"),
        "the table statement must survive: {simple:?}"
    );

    // A genuine decomposition is untouched.
    let genuine = titles("integrate(1/(x^2+x), x)");
    assert!(
        genuine
            .iter()
            .any(|(t, b, a)| t.contains("Descomponer") && b == "1 / (x^2 + x)" && a != b),
        "a real decomposition must still narrate: {genuine:?}"
    );
}
/// Row `k` of the Hessian differentiates `∂f/∂x_k`, not `f` — which is what the
/// substep's own title says. Using `f` as the `before` made the line a false
/// statement (from `y·x²` "comes" `[2y, 2x]`) and hid the gradient component,
/// the one intermediate that makes the jump followable. The jacobian arm next
/// door was already correct.
#[test]
fn hessian_row_substeps_start_from_the_first_derivative() {
    let (wire, _) = cli_eval_json_with_stderr_args("hessian(x^2*y, [x,y])", &["--steps", "on"]);
    let subs: Vec<(String, String, String)> = wire["steps"]
        .as_array()
        .expect("steps with --steps on")
        .iter()
        .filter_map(|step| step["substeps"].as_array())
        .flatten()
        .map(|s| {
            (
                s["title"].as_str().unwrap_or_default().to_string(),
                s["before"].as_str().unwrap_or_default().to_string(),
                s["after"].as_str().unwrap_or_default().to_string(),
            )
        })
        .collect();
    assert_eq!(subs.len(), 2, "one substep per row: {subs:?}");
    // ∂f/∂x = 2xy and ∂f/∂y = x², NOT f = y·x² twice.
    assert_eq!(subs[0].1, "2·x·y");
    assert_eq!(subs[1].1, "x^2");
    // And both sides fold: the machinery's `x^(2 - 1 - 1)` never reaches the student.
    for (title, before, after) in &subs {
        assert!(
            !before.contains("^(") && !after.contains("^("),
            "raw exponent arithmetic leaked into «{title}»: {before} -> {after}"
        );
    }
    assert_eq!(subs[0].2, "[2·y, 2·x]");
    assert_eq!(subs[1].2, "[2·x, 0]");
}
/// The reverse-nested-fraction narrator may only claim `A = c·B` when that
/// identity actually holds. It used to fire on pattern match alone and publish
/// `A = (1-x)²·A` inside `diff(arctan((1+x)/(1-x)), x)`.
#[test]
fn nested_fraction_common_factor_substep_requires_the_identity_to_hold() {
    let (wire, _) =
        cli_eval_json_with_stderr_args("diff(arctan((1+x)/(1-x)), x)", &["--steps", "on"]);
    let titles: Vec<String> = wire["steps"]
        .as_array()
        .expect("steps with --steps on")
        .iter()
        .filter_map(|step| step["substeps"].as_array())
        .flatten()
        .filter_map(|s| s["title"].as_str().map(str::to_string))
        .collect();
    assert!(
        !titles.iter().any(|t| t.contains("sacando factor común")),
        "the identity does not hold here, so the substep must decline: {titles:?}"
    );
    // The rest of the trace is untouched.
    assert!(
        titles.iter().any(|t| t.contains("Invertir la fracción")),
        "the genuine manoeuvres must survive: {titles:?}"
    );
}
/// Linearity over a sum integrand, verified term by term.
///
/// The audit's second witness: `integrate(2*x/sqrt(4+x^4)+1, x)` published one
/// magic step while the same integrand WITHOUT the `+1` narrated fine — the
/// chain's only additive decomposition sat behind a gate demanding the whole
/// integrand be a polynomial.
#[test]
fn integrate_contract_additive_integrand_narrates_linearity_then_each_term() {
    let subs = |input: &str| -> Vec<(String, String, String)> {
        let (wire, _) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
        wire["steps"]
            .as_array()
            .expect("steps with --steps on")
            .iter()
            .filter_map(|step| step["substeps"].as_array())
            .flatten()
            .map(|s| {
                (
                    s["title"].as_str().unwrap_or_default().to_string(),
                    s["before"].as_str().unwrap_or_default().to_string(),
                    s["after"].as_str().unwrap_or_default().to_string(),
                )
            })
            .collect()
    };

    let witness = subs("integrate(2*x/sqrt(4+x^4)+1, x)");
    assert!(
        witness.iter().any(|(t, _, _)| t.contains("linealidad")),
        "the witness must open with linearity: {witness:?}"
    );
    assert!(
        witness.iter().any(|(t, _, _)| t.contains("asinh")),
        "and each term keeps its own method: {witness:?}"
    );

    // A sum must never be labelled with a single term's method.
    for input in ["integrate(ln(x)+x, x)", "integrate(x*e^x+sin(2*x), x)"] {
        let s = subs(input);
        assert_eq!(
            s.first().map(|(t, _, _)| t.as_str()),
            Some("Usar linealidad de la integral"),
            "{input} must open with linearity, not with one term's method: {s:?}"
        );
    }

    // The polynomial arm keeps ownership (its pins fix a 2-substep narration).
    let poly = subs("integrate(x^2+3*x+1, x)");
    assert_eq!(
        poly.len(),
        2,
        "polynomial arm must still own this: {poly:?}"
    );

    // A PRODUCT integrand is not linearity and must be untouched.
    let product = subs("integrate(e^x*sin(x), x)");
    assert!(
        !product.iter().any(|(t, _, _)| t.contains("linealidad")),
        "a product is not a sum: {product:?}"
    );
}
/// A vector `integrate`/`diff` narrates component by component. The engine
/// already worked that way and said so in its rule description, but the didactic
/// chain did not recognise the `Expr::Matrix` shape and returned empty.
#[test]
fn integrate_contract_vector_calculus_narrates_each_component() {
    let subs = |input: &str| -> Vec<(String, String, String)> {
        let (wire, _) = cli_eval_json_with_stderr_args(input, &["--steps", "on"]);
        wire["steps"]
            .as_array()
            .expect("steps with --steps on")
            .iter()
            .filter_map(|step| step["substeps"].as_array())
            .flatten()
            .map(|s| {
                (
                    s["title"].as_str().unwrap_or_default().to_string(),
                    s["before"].as_str().unwrap_or_default().to_string(),
                    s["after"].as_str().unwrap_or_default().to_string(),
                )
            })
            .collect()
    };

    for (input, header) in [
        (
            "integrate([cos(x), e^x], x)",
            "Integrar cada componente del vector",
        ),
        (
            "diff([x^2, sin(x)], x)",
            "Derivar cada componente del vector",
        ),
        // Definite: the split rides the bounds along.
        (
            "integrate([cos(t), sin(t)], t, 0, pi)",
            "Integrar cada componente del vector",
        ),
    ] {
        let s = subs(input);
        assert_eq!(
            s.first().map(|(t, _, _)| t.as_str()),
            Some(header),
            "{input} must open with the component split: {s:?}"
        );
        // The header SHOWS the split (pending per-component operations); a
        // header that merely restates the parent is pruned by policy.
        assert_ne!(s[0].1, s[0].2, "{input} header restates its parent: {s:?}");
        // At least one component narrates. NOT "every component": `∫e^x dx = e^x`
        // is a fixed point and has nothing to narrate — demanding a substep per
        // component is what would push a narrator into publishing `e^x -> e^x`.
        assert!(
            s.len() >= 2,
            "{input} must narrate at least one component after the split: {s:?}"
        );
    }

    // Both components non-trivial: one substep each on top of the header.
    let both = subs("diff([x^2, sin(x)], x)");
    assert!(
        both.len() >= 3,
        "both components of this one do narrate: {both:?}"
    );

    // And the fixed point publishes NOTHING rather than an identity substep.
    let fixed_point = subs("integrate(e^x, x)");
    assert!(
        fixed_point.is_empty(),
        "∫e^x dx is a fixed point; «use substitution: e^x -> e^x» teaches nothing: {fixed_point:?}"
    );
}
/// A step's red/green is an ASSERTION: "replace this piece with that piece and
/// you get the next state". When the span lands on the wrong subtree the
/// assertion is FALSE, and the audit found it published as an identity under a
/// rule name. The witness the user reported: `taylor(sin(x), x, 0, 5)` step 3
/// highlighted `x → x` — the one summand that does NOT change — while the
/// fraction that actually reduced went unmarked.
#[test]
fn highlight_spans_are_verified_before_being_published() {
    let (wire, _) = cli_eval_json_with_stderr_args("taylor(sin(x), x, 0, 5)", &["--steps", "on"]);
    let steps = wire["steps"].as_array().expect("steps with --steps on");
    for step in steps {
        let rule_latex = step["rule_latex"].as_str().unwrap_or_default();
        // A rule line whose two sides are identical asserts nothing.
        if let Some((lhs, rhs)) = rule_latex.split_once("\\rightarrow") {
            assert_ne!(
                lhs.replace("{\\color{red}{", "")
                    .replace(['{', '}', ' '], ""),
                rhs.replace("{\\color{green}{", "")
                    .replace(['{', '}', ' '], ""),
                "a rule line must not assert `A → A`: {rule_latex}"
            );
        }
    }
    // The witness step now names the fraction it reduces, not the untouched `x`.
    let witness = steps
        .iter()
        .find(|s| s["before"].as_str().unwrap_or_default().contains("720"))
        .expect("the step that reduces /720 must exist");
    let rule_latex = witness["rule_latex"].as_str().unwrap_or_default();
    assert!(
        rule_latex.contains("720") && rule_latex.contains("120"),
        "the rule line must show the fraction it reduces: {rule_latex}"
    );
}
/// The truthfulness predicate is UNIFORM in the number of spans: collapse each
/// contiguous run of coloured spans into a hole and require the untouched
/// remainder to be identical on both sides. That lifts the multi-span exception
/// C1.3 had to declare, and it subsumes the one-to-one substitution check.
#[test]
fn highlight_guard_accepts_many_to_one_spans_and_still_rejects_the_witness() {
    // Many-to-one: two adjacent terms become one. TRUE, must publish.
    let (wire, _) = cli_eval_json_with_stderr_args(
        "1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)",
        &["--steps", "on"],
    );
    let steps = wire["steps"].as_array().expect("steps");
    let many_to_one = steps.iter().any(|s| {
        let b = s["before_latex"].as_str().unwrap_or_default();
        let a = s["after_latex"].as_str().unwrap_or_default();
        b.matches("\\color{red}").count() > a.matches("\\color{green}").count()
            && a.contains("\\color{green}")
    });
    assert!(
        many_to_one,
        "a many-to-one span is a truthful claim and must survive the guard"
    );

    // And the witness never publishes the FALSE span. (C1.3 made it decline to
    // the whole state; C2.1 recovers a TRUE partial span from the structural
    // diff. What must hold in both is the same: the untouched `x` is never what
    // the step claims it changed.)
    let (witness, _) =
        cli_eval_json_with_stderr_args("taylor(sin(x), x, 0, 5)", &["--steps", "on"]);
    let reduce_step = witness["steps"]
        .as_array()
        .expect("steps")
        .iter()
        .find(|s| s["before"].as_str().unwrap_or_default().contains("720"))
        .expect("the /720 step");
    let before_latex = reduce_step["before_latex"].as_str().unwrap_or_default();
    assert!(
        !before_latex.contains("{\\color{red}{x}}"),
        "the bare `x` does not change and must never be the span: {before_latex}"
    );
}
/// When the recorded focus lies (PATH DRIFT), the span is recomputed from the
/// two states themselves: the minimal structural diff. That is true by
/// construction — everything outside it is identical on both sides — so the
/// guard passes and the step keeps its PRECISION instead of falling back to
/// colouring everything.
///
/// This closes the user's reported witness end to end: the rule name (C2.4),
/// the honest decline (C1.3) and now the precise span.
#[test]
fn declined_spans_recover_precision_from_the_structural_diff() {
    let (wire, _) = cli_eval_json_with_stderr_args("taylor(sin(x), x, 0, 5)", &["--steps", "on"]);
    let step = wire["steps"]
        .as_array()
        .expect("steps")
        .iter()
        .find(|s| s["before"].as_str().unwrap_or_default().contains("720"))
        .expect("the /720 step");
    let before = step["before_latex"].as_str().unwrap_or_default();
    let after = step["after_latex"].as_str().unwrap_or_default();
    // PARTIAL span (the fraction), not the whole state, and not the `x`.
    assert!(
        before.starts_with("x + {\\color{red}{"),
        "the untouched `x` must stay outside the span: {before}"
    );
    assert!(
        before.contains("720") && after.contains("120"),
        "the span must be the fraction that reduces: {before} / {after}"
    );
}
#[test]
fn integrate_contract_shifted_trig_power_narrates_u_du() {
    // La vía u-du simbólica (L16 pista (a)) debe narrar como el libro de
    // texto: identificar u y du con la base afín, y aplicar la regla de
    // potencia — no el "Usar sustitución" genérico sin evidencia.
    let input = "integrate(cos(x)*(sin(x)+1)^2, x)";
    let substeps = integration_substeps(input);
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Identificar u y du"),
        "expected concrete u/du substep for {input}, got {substeps:?}"
    );
    assert_u_du_substep_labels(&substeps, input);
    let u_du = substeps
        .iter()
        .find(|substep| substep["title"] == "Identificar u y du")
        .expect("u/du substep");
    assert!(
        u_du["before"]
            .as_str()
            .unwrap_or_default()
            .contains("sin(x) + 1"),
        "u debe ser la base afín sin(x)+1, got {u_du:?}"
    );
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar regla de potencia para integrales"),
        "expected power-rule application substep for {input}, got {substeps:?}"
    );
}
#[test]
fn integrate_contract_div_u_du_narrates_reciprocal_and_log_rules() {
    // La extensión Div de la vía u-du narra con evidencia: identificar u/du y
    // la regla aplicada (potencia para m>1, ln|u| para m=1).
    let substeps = integration_substeps("integrate(cos(x)/(sin(x)+2)^2, x)");
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Identificar u y du"),
        "expected u/du substep, got {substeps:?}"
    );
    assert!(
        substeps
            .iter()
            .any(|substep| substep["title"] == "Usar regla de potencia para integrales"),
        "expected power-rule substep, got {substeps:?}"
    );

    let log_substeps = integration_substeps("integrate(cosh(x)/(sinh(x)+3), x)");
    assert_u_du_substep_labels(&log_substeps, "integrate(cosh(x)/(sinh(x)+3), x)");
    assert!(
        log_substeps
            .iter()
            .any(|substep| substep["title"] == "Usar la regla de ln|u| con derivada interna"),
        "expected ln|u| substep, got {log_substeps:?}"
    );
}
#[test]
fn integrate_contract_symbolic_table_narrates_u_du_and_rule() {
    for (input, rule_title) in [
        (
            "integrate(cos(x)*cos(sin(x)), x)",
            "Usar la regla de cos(u) -> sin(u)",
        ),
        (
            "integrate(cos(x)*exp(sin(x)), x)",
            "Usar la regla de exp(u) -> exp(u)",
        ),
    ] {
        let substeps = integration_substeps(input);
        assert_u_du_substep_labels(&substeps, input);
        assert!(
            substeps
                .iter()
                .any(|substep| substep["title"] == rule_title),
            "expected '{rule_title}' for {input}, got {substeps:?}"
        );
    }
}
