use super::*;

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

    let steps = wire["steps"].as_array().expect("steps array");
    assert!(
        !steps.is_empty(),
        "expected wire steps for derive root nesting when steps_count is positive"
    );
    assert_eq!(wire["steps_count"], steps.len());
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
    assert_rule_eq(&steps[0]["rule"], "Extraer potencia par de la raíz");
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
    assert_eq!(substeps[0]["before_latex"], json!("{x}^{5}"));
    assert_eq!(substeps[0]["after_latex"], json!("{x}^{4}\\cdot x"));
    assert_eq!(
        substeps[1]["before_latex"],
        json!("\\sqrt{{x}^{4}\\cdot x}")
    );
    assert_eq!(substeps[1]["after_latex"], json!("\\sqrt{x}\\cdot {x}^{2}"));
}
#[test]
fn derive_raw_odd_half_power_uses_concrete_root_split_substeps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive x^(3/2), abs(x)*sqrt(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand odd half power");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Extraer potencia par de la raíz");
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 2);
    assert_eq!(
        substeps[0]["title"],
        "Separar el radicando en una potencia par y un factor"
    );
    assert_eq!(substeps[0]["before_latex"], json!("{x}^{3}"));
    assert_eq!(substeps[0]["after_latex"], json!("{x}^{2}\\cdot x"));
    let second_title = substeps[1]["title"].as_str().expect("second title");
    assert!(
        second_title.contains("Como x ≥ 0"),
        "unexpected second title: {second_title}"
    );
}
