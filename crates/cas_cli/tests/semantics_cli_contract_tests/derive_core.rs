use super::*;

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
    assert_rule_eq(&steps[0]["rule"], "Agrupar términos semejantes");
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
    assert_rule_eq(&steps[0]["rule"], "Cancelar factoriales consecutivos");

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
fn derive_finite_telescoping_product_uses_concrete_endpoint_substeps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive product((k+1)/k, k, 1, n), n+1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "finite sums/products");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Evaluar producto telescópico finito");
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 3);
    assert_eq!(
        substeps[0]["title"],
        "Escribir los primeros y últimos factores del producto"
    );
    assert_eq!(
        substeps[0]["before_latex"],
        json!("\\prod_{k=1}^{n} \\frac{k + 1}{k}")
    );
    assert_eq!(
        substeps[1]["title"],
        "Los factores intermedios se cancelan por parejas"
    );
    assert_eq!(substeps[1]["after_latex"], json!("\\frac{n + 1}{1}"));
    assert_eq!(
        substeps[2]["title"],
        "Solo quedan el último numerador y el primer denominador"
    );
    assert_eq!(substeps[2]["after_latex"], json!("n + 1"));
}
#[test]
fn derive_inverse_tan_reciprocal_identity_uses_named_inverse_trig_rewrite_and_keeps_guard() {
    // The sound identity is arctan(a)+arctan(1/a) = (π/2)·sign(a) (it is -π/2 for
    // a<0), so the derivation target carries the sign factor; deriving the bare
    // π/2 no longer holds unconditionally (only for a>0).
    let (output, _code) = run_cli(&[
        "eval",
        "derive arctan(a)+arctan(1/a), (pi/2)*sign(a)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite inverse trigs");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar identidad de arctangentes");

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        required.iter().any(|item| item == "a ≠ 0"),
        "expected inverse-tan nonzero guard in required_display: {required:?}"
    );
}
