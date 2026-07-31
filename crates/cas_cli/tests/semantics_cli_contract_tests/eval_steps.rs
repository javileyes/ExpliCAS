use super::*;

#[test]
fn eval_symbolic_integration_step_keeps_integral_latex_in_before_wire() {
    let (output, _code) = run_cli(&[
        "eval",
        "integrate(x^2, x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);

    let step = &steps[0];
    let rule_latex = step["rule_latex"].as_str().expect("rule_latex");
    let before_latex = step["before_latex"].as_str().expect("before_latex");

    assert!(
        rule_latex.contains("\\int"),
        "expected rule_latex to show an integral, got: {rule_latex}"
    );
    assert!(
        before_latex.contains("\\int"),
        "expected before_latex to show an integral, got: {before_latex}"
    );
    assert!(
        !before_latex.contains("\\text{integrate}"),
        "expected before_latex to avoid function-style integrate(), got: {before_latex}"
    );
}
#[test]
fn eval_symbolic_differentiation_step_keeps_derivative_latex_in_before_wire() {
    let (output, _code) = run_cli(&["eval", "diff(x^2, x)", "--format", "json", "--steps", "on"]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);

    let step = &steps[0];
    let rule_latex = step["rule_latex"].as_str().expect("rule_latex");
    let before_latex = step["before_latex"].as_str().expect("before_latex");

    assert!(
        rule_latex.contains("\\frac{d}{dx}"),
        "expected rule_latex to show a derivative, got: {rule_latex}"
    );
    assert!(
        before_latex.contains("\\frac{d}{dx}"),
        "expected before_latex to show a derivative, got: {before_latex}"
    );
    assert!(
        !before_latex.contains("\\text{diff}"),
        "expected before_latex to avoid function-style diff(), got: {before_latex}"
    );
}
#[test]
fn eval_partitioned_zero_chunks_keep_step_highlights_localized_in_mixed_sum() {
    let (output, _code) = run_cli(&[
        "eval",
        "((cos(x))^3 - (3*cos(x) + cos(3*x))/4) + (tan(x) + 1/tan(x) - 2/sin(2*x)) + (atanh((x^2 - 1)/(x^2 + 1)) - log(x)) + (sqrt(x + sqrt(x^2 - y^2)) - (sqrt(x+y) + sqrt(x-y))/sqrt(2))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 4);

    let step2_before = steps[1]["before_latex"]
        .as_str()
        .expect("step2 before_latex");
    assert!(
        step2_before.contains("{\\color{red}{\\tan(x)}}")
            && step2_before.contains("{\\color{red}{\\frac{1}{\\tan(x)}}}")
            && step2_before.contains("{\\color{red}{\\frac{2}{\\sin(2\\cdot x)}}}"),
        "expected step 2 before_latex to highlight exactly the trig reciprocal chunk, got: {step2_before}"
    );
    assert!(
        !step2_before.contains("{\\color{red}{\\tan(x) + \\operatorname{atanh}")
            && !step2_before.contains("{\\color{red}{\\sqrt"),
        "expected step 2 before_latex to avoid swallowing unrelated terms, got: {step2_before}"
    );

    let step3_before = steps[2]["before_latex"]
        .as_str()
        .expect("step3 before_latex");
    assert!(
        step3_before.contains("{\\color{red}{\\operatorname{atanh}")
            && step3_before.contains("{\\color{red}{\\ln(x)}}"),
        "expected step 3 before_latex to highlight the full atanh-log chunk, got: {step3_before}"
    );
    assert!(
        !step3_before.contains("{\\color{red}{\\sqrt"),
        "expected step 3 before_latex to keep the root-denesting chunk outside the highlight, got: {step3_before}"
    );
}
#[test]
fn eval_pythagorean_identity_difference_collapses_to_zero_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "sin(x)^2 + cos(x)^2 - 1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Cancelar la subexpresión idénticamente nula"
    );
}
#[test]
fn eval_pythagorean_identity_scaled_difference_collapses_to_zero_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "k*(sin(x)^2 + cos(x)^2) - k*(1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Collapse Common-Scale Equivalent Difference"
    );
}
#[test]
fn eval_exact_phase_shift_pair_passthrough_difference_collapses_in_one_phase_shift_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((sin(x)+cos(x)+sin(y)+cos(y)) + m) - ((sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4)) + m)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Cancelar la subexpresión idénticamente nula"
    );
}
#[test]
fn eval_exact_third_phase_shift_passthrough_difference_collapses_in_one_phase_shift_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "((4*sin(x+pi/3)) + m) - ((2*sin(x)+2*sqrt(3)*cos(x)) + m)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Cancelar la subexpresión idénticamente nula"
    );
}
#[test]
fn eval_polynomial_identity_common_factor_zero_uses_factor_common_substeps() {
    let (output, _code) = run_cli(&[
        "eval",
        "x*y + x*z - x*(y+z)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 1);
    assert_eq!(substeps[0]["title"], "Usar el factor común");
}
#[test]
fn eval_polynomial_identity_binomial_square_zero_uses_square_formula_substeps() {
    let (output, _code) = run_cli(&[
        "eval",
        "x^2 + 2*x + 1 - (x+1)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Cancelar la subexpresión idénticamente nula"
    );
}
#[test]
fn eval_polynomial_identity_geometric_difference_zero_uses_factorization_substeps() {
    let (output, _code) = run_cli(&[
        "eval",
        "x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 1);
    assert_eq!(
        substeps[0]["title"],
        "Usar a^n - 1 = (a - 1) · (a^(n-1) + a^(n-2) + ... + a + 1)"
    );
}
#[test]
fn eval_polynomial_identity_sophie_germain_zero_uses_named_substeps() {
    let (output, _code) = run_cli(&[
        "eval",
        "x^4 + 4*y^4 - (x^2-2*x*y+2*y^2)*(x^2+2*x*y+2*y^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 1);
    assert_eq!(
        substeps[0]["title"],
        "Usar a^4 + 4b^4 = (a^2 - 2ab + 2b^2) · (a^2 + 2ab + 2b^2)"
    );
}
#[test]
fn eval_finite_telescoping_product_difference_collapses_in_one_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "product(1 - 1/(k+a)^2, k, m, n) - (((m+a-1)*(n+a+1))/((m+a)*(n+a)))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Cancelar la subexpresión idénticamente nula"
    );
}
#[test]
fn eval_solve_polynomial_in_atom_narrates_substitution_and_back_substitution() {
    // `u = x^(1/2)` quadratic-in-disguise: the substitution line shows the
    // clean display `u` (never the collision-safe internal `__…_u` name) with
    // identity-zero noise stripped, then one back-substitution line per root
    // followed by that branch's own narration.
    let (output, code) = run_cli(&[
        "eval",
        "solve(x-3*sqrt(x)+2=0,x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0, "output: {output}");
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "{ 1, 4 }");
    let steps = wire["solve_steps"].as_array().cloned().unwrap_or_default();
    let descs: Vec<&str> = steps
        .iter()
        .filter_map(|s| s["description"].as_str())
        .collect();
    assert_eq!(
        descs.first().copied(),
        Some("Sustitución detectada: u = x^(1/2)")
    );
    assert_eq!(steps[0]["equation"], "u^2 + 2 - 3·u = 0");
    assert!(
        descs.contains(&"Sustitución inversa: x^(1/2) = 1")
            && descs.contains(&"Sustitución inversa: x^(1/2) = 2"),
        "expected one back-substitution line per root, got {steps:?}"
    );

    // The trig-atom variant chains into the periodic narration and keeps the
    // out-of-range branch honest (sin(x) = 2 narrated, no family emitted).
    let (output, code) = run_cli(&[
        "eval",
        "solve(sin(x)^2-3*sin(x)+2=0,x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0, "output: {output}");
    let wire = parse_wire(&output);
    let steps = wire["solve_steps"].as_array().cloned().unwrap_or_default();
    let descs: Vec<&str> = steps
        .iter()
        .filter_map(|s| s["description"].as_str())
        .collect();
    assert_eq!(steps[0]["equation"], "u^2 - 3·u + 2 = 0");
    assert!(
        descs.contains(&"Sustitución inversa: sin(x) = 2"),
        "the out-of-range branch must still narrate its back-substitution, got {steps:?}"
    );
    assert!(
        descs.contains(&"Familia periódica de soluciones (k entero cualquiera)"),
        "the in-range branch must chain into the periodic narration, got {steps:?}"
    );
}
#[test]
fn eval_solve_abs_equations_narrate_argument_zero_and_case_splits() {
    // `|E| = 0 ⟺ E = 0`: the equivalence line, then the argument's own solve.
    let (output, code) = run_cli(&[
        "eval",
        "solve(abs(x*(x-2))=0,x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0, "output: {output}");
    let wire = parse_wire(&output);
    let steps = wire["solve_steps"].as_array().cloned().unwrap_or_default();
    let descs: Vec<&str> = steps
        .iter()
        .filter_map(|s| s["description"].as_str())
        .collect();
    assert_eq!(
        descs.first().copied(),
        Some("El valor absoluto es cero exactamente cuando su argumento es cero"),
        "expected the |E|=0 equivalence line, got {steps:?}"
    );
    assert!(
        descs.iter().any(|d| d.starts_with("Resuelve el factor")),
        "expected the argument's factored solve to chain in, got {steps:?}"
    );

    // `|f| = |h|` splits into the two signed cases.
    let (output, code) = run_cli(&[
        "eval",
        "solve(abs(x^2-4)=abs(x-2),x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0, "output: {output}");
    let wire = parse_wire(&output);
    let steps = wire["solve_steps"].as_array().cloned().unwrap_or_default();
    let descs: Vec<&str> = steps
        .iter()
        .filter_map(|s| s["description"].as_str())
        .collect();
    assert_eq!(
        descs,
        vec![
            "Descompón el valor absoluto (Case 1): x^2 - 4 = x - 2",
            "Descompón el valor absoluto (Case 2): x^2 - 4 = -(x - 2)",
        ],
        "expected the two split-case lines, got {steps:?}"
    );

    // Sign-split (single |f| entangled with polynomial-in-x structure, linear
    // g included): the two case lines carry the SUBSTITUTED relation.
    let (output, code) = run_cli(&[
        "eval",
        "solve(abs(x^2-1)=x+1,x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0, "output: {output}");
    let wire = parse_wire(&output);
    let steps = wire["solve_steps"].as_array().cloned().unwrap_or_default();
    let descs: Vec<&str> = steps
        .iter()
        .filter_map(|s| s["description"].as_str())
        .collect();
    assert_eq!(
        descs,
        vec![
            "Valor absoluto por signo: caso argumento ≥ 0 (|f| = f)",
            "Valor absoluto por signo: caso argumento < 0 (|f| = −f)",
        ],
        "expected the sign-split case lines, got {steps:?}"
    );
    assert_eq!(steps[0]["equation"], "x^2 - x - 2 = 0");

    // The inequality form is owned by the same handler and narrates the same way.
    let (output, code) = run_cli(&[
        "eval",
        "solve(x^2-3*abs(x)+2<0,x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(code, 0, "output: {output}");
    let wire = parse_wire(&output);
    let steps = wire["solve_steps"].as_array().cloned().unwrap_or_default();
    assert_eq!(
        steps.len(),
        2,
        "expected two sign-split lines, got {steps:?}"
    );
    assert_eq!(steps[0]["equation"], "x^2 + 2 - 3·x < 0");
}
/// A quadratic that factors over ℚ is SOLVED by factoring, and the trace says
/// so — the roots may not appear from nowhere.
///
/// Before 2026-07-28 the zero-product strategy only fired when the input was
/// ALREADY a product, so `u² − 3u + 2 = 0` fell through to the formula and its
/// entire narration was one line: «Se detectó una ecuación cuadrática.
/// Aplicando la fórmula cuadrática.» The reader was then shown `u = 1` and
/// `u = 2` with nothing in between. Reported by the user against the
/// substitution wrapper, where the gap is starkest.
#[test]
fn eval_factorable_quadratic_narrates_the_factorization_and_the_zero_product() {
    let (output, _code) = run_cli(&[
        "eval",
        "solve(u^2-3*u+2=0,u)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "{ 1, 2 }");
    let descriptions: Vec<String> = wire["solve_steps"]
        .as_array()
        .expect("solve_steps array")
        .iter()
        .map(|s| s["description"].as_str().unwrap_or_default().to_string())
        .collect();
    let joined = descriptions.join(" | ");
    assert!(
        joined.contains("Ecuación factorizada: (u - 1)·(u - 2) = 0"),
        "{joined}"
    );
    assert!(joined.contains("Resuelve el factor: u - 1 = 0"), "{joined}");
    assert!(joined.contains("Resuelve el factor: u - 2 = 0"), "{joined}");
    assert!(
        !joined.contains("fórmula cuadrática"),
        "a factorable quadratic must not announce the formula: {joined}"
    );

    // The substitution wrapper inherits it: the atom's quadratic is where the
    // roots come from, and that is exactly what was missing from the trace.
    let (output, _code) = run_cli(&[
        "eval",
        "solve(e^(2*x)-3*e^x+2=0,x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);
    assert_eq!(wire["result"], "{ ln(2), 0 }");
    let joined: String = wire["solve_steps"]
        .as_array()
        .expect("solve_steps array")
        .iter()
        .map(|s| s["description"].as_str().unwrap_or_default())
        .collect::<Vec<_>>()
        .join(" | ");
    assert!(joined.contains("Ecuación sustituida"), "{joined}");
    assert!(joined.contains("Ecuación factorizada"), "{joined}");
    assert!(joined.contains("Sustitución inversa"), "{joined}");

    // The gate DECLINES what does not factor over ℚ: an irrational-root
    // quadratic keeps the formula, and its result is untouched.
    let (output, _code) = run_cli(&[
        "eval",
        "solve(x^2-2=0,x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);
    let joined: String = wire["solve_steps"]
        .as_array()
        .expect("solve_steps array")
        .iter()
        .map(|s| s["description"].as_str().unwrap_or_default())
        .collect::<Vec<_>>()
        .join(" | ");
    assert!(joined.contains("fórmula cuadrática"), "{joined}");
    assert!(!joined.contains("factorizada"), "{joined}");
}
/// The atom substitution paths must show HOW the u-roots were obtained.
///
/// `solve_polynomial_in_atom` solved the u-polynomial and dropped its
/// narration on the floor (`let (u_solution, _) = solve(...)`), justified by
/// «the exp route shows neither». Once the exp route started narrating its
/// factorization, that left `ln`/`sin`/radical atoms showing `u = 1` and
/// `u = 2` arriving from nowhere — the user's original complaint, one path
/// over.
///
/// The republished lines must speak of `u`: the real substitution variable is
/// a collision-safe synthetic (`__trig_u`, `__rps_u`, …) the reader never
/// typed, and a step naming it would be narrating an internal form.
#[test]
fn eval_atom_substitution_narrates_the_u_polynomial_without_leaking_the_synthetic_var() {
    for (input, expected_result) in [
        ("solve(ln(x)^2-3*ln(x)+2=0,x)", "{ e, e^2 }"),
        (
            "solve(sin(x)^2-3*sin(x)+2=0,x)",
            "{ 1/2·pi + k·2·pi : k ∈ ℤ }",
        ),
    ] {
        let (output, _code) = run_cli(&["eval", input, "--format", "json", "--steps", "on"]);
        let wire = parse_wire(&output);
        assert_eq!(wire["result"], expected_result, "{input}");
        let steps = wire["solve_steps"].as_array().expect("solve_steps array");
        let joined: String = steps
            .iter()
            .map(|s| {
                format!(
                    "{} :: {}",
                    s["description"].as_str().unwrap_or_default(),
                    s["equation"].as_str().unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        // The derivation of the u-roots is present…
        assert!(
            joined.contains("Ecuación factorizada: (u - 1)·(u - 2) = 0"),
            "{input}: {joined}"
        );
        assert!(joined.contains("Resuelve el factor: u - 1 = 0"), "{joined}");
        // …and it lands on the roots the back-substitution then consumes.
        assert!(joined.contains("u = 1"), "{joined}");
        assert!(joined.contains("u = 2"), "{joined}");
        // No internal symbol may reach the reader.
        assert!(
            !joined.contains("__"),
            "a synthetic substitution symbol leaked into the trace: {joined}"
        );
    }
}
