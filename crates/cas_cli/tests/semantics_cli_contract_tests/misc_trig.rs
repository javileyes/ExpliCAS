use super::*;

#[test]
fn root_product_equiv_carries_intersection_domain_requires() {
    let (output, code) = run_cli(&[
        "eval",
        "equiv(sqrt(x*y),sqrt(x)*sqrt(y))",
        "--format",
        "json",
    ]);
    assert_eq!(code, 0);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "true");
    assert_eq!(wire["required_display"], json!(["x ≥ 0", "y ≥ 0"]));
}
#[test]
fn inv_trig_principal_reflected() {
    let (output, _code) = run_cli(&["eval", "1+1", "--format", "json", "--inv-trig", "principal"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["semantics"]["inv_trig"], "principal");
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

    assert_eq!(wire["result"], "1 + sin(u)^2 + sin(u)^4");

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
fn inverse_reciprocal_trig_domains_hide_redundant_public_nonzero_guard() {
    for (expr, result, builtin) in [
        ("arcsec(x)", "arccos(1 / x)", "NonNegative"),
        ("arccsc(x)", "arcsin(1 / x)", "NonNegative"),
    ] {
        let (output, code) = run_cli(&["eval", expr, "--format", "json"]);
        assert_eq!(
            code, 0,
            "expected successful CLI exit for {expr}, got {code}: {output}"
        );
        let wire = parse_wire(&output);

        assert_eq!(wire["result"], result);
        assert_eq!(wire["required_display"], json!(["x ≤ -1 or x ≥ 1"]));

        let required_conditions = wire["required_conditions"]
            .as_array()
            .expect("required_conditions");
        assert!(
            required_conditions
                .iter()
                .any(|condition| condition["kind"] == "NonZero"
                    && condition["expr_canonical"] == "x"),
            "structured nonzero guard should remain for {expr}: {required_conditions:?}"
        );
        assert!(
            required_conditions
                .iter()
                .any(|condition| condition["kind"] == builtin
                    && condition["expr_canonical"] == "1 - (1 / x)^2"),
            "structured bounded-domain guard should remain for {expr}: {required_conditions:?}"
        );
    }
}
#[test]
fn inverse_reciprocal_trig_affine_domains_display_solved_exterior_interval() {
    for (expr, result, display, nonzero, bounded) in [
        (
            "arcsec(x + 1)",
            "arccos(1 / (x + 1))",
            "x ≤ -2 or x ≥ 0",
            "x + 1",
            "1 - (1 / (x + 1))^2",
        ),
        (
            "arccsc(2*x - 1)",
            "arcsin(1 / (2·x - 1))",
            "x ≤ 0 or x ≥ 1",
            "2·x - 1",
            "1 - (1 / (2·x - 1))^2",
        ),
    ] {
        let (output, code) = run_cli(&["eval", expr, "--format", "json"]);
        assert_eq!(
            code, 0,
            "expected successful CLI exit for {expr}, got {code}: {output}"
        );
        let wire = parse_wire(&output);

        assert_eq!(wire["result"], result);
        assert_eq!(wire["required_display"], json!([display]));

        let required_conditions = wire["required_conditions"]
            .as_array()
            .expect("required_conditions");
        assert!(
            required_conditions
                .iter()
                .any(|condition| condition["kind"] == "NonZero"
                    && condition["expr_canonical"] == nonzero),
            "structured nonzero guard should remain for {expr}: {required_conditions:?}"
        );
        assert!(
            required_conditions
                .iter()
                .any(|condition| condition["kind"] == "NonNegative"
                    && condition["expr_canonical"] == bounded),
            "structured bounded-domain guard should remain for {expr}: {required_conditions:?}"
        );
    }
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
    assert!(
        warnings.is_empty() || warnings.len() == 1,
        "principal inverse-trig warning contract changed unexpectedly: {warnings:?}"
    );
    if let Some(first_warning) = warnings.first() {
        assert_eq!(first_warning["rule"], "Principal Branch Inverse Trig");
    }
}
