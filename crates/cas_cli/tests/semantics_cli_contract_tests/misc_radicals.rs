use super::*;

#[test]
fn root_product_merge_in_generic_carries_nonnegative_requires() {
    let (output, code) = run_cli(&["eval", "sqrt(x)*sqrt(y)", "--format", "json"]);
    assert_eq!(code, 0);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "sqrt(x·y)");
    assert_eq!(wire["domain"]["mode"], "generic");
    assert_eq!(wire["required_display"], json!(["x ≥ 0", "y ≥ 0"]));
}
#[test]
fn root_quotient_merge_in_generic_carries_positive_denominator_requires() {
    let (output, code) = run_cli(&["eval", "sqrt(x)/sqrt(y)", "--format", "json"]);
    assert_eq!(code, 0);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "sqrt(x / y)");
    assert_eq!(wire["domain"]["mode"], "generic");
    assert_eq!(wire["required_display"], json!(["y > 0", "x ≥ 0"]));
}
#[test]
fn root_product_merge_in_assume_preserves_intrinsic_requires_not_assumes() {
    let (output, code) = run_cli(&[
        "eval",
        "sqrt(x)*sqrt(y)",
        "--domain",
        "assume",
        "--format",
        "json",
    ]);
    assert_eq!(code, 0);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "sqrt(x·y)");
    assert_eq!(wire["domain"]["mode"], "assume");
    assert_eq!(wire["required_display"], json!(["x ≥ 0", "y ≥ 0"]));
    assert!(
        wire["assumptions_used"].is_null()
            || wire["assumptions_used"]
                .as_array()
                .is_some_and(Vec::is_empty),
        "expected no assumptions_used, got {:?}",
        wire["assumptions_used"]
    );

    let wire_messages = wire["wire"]["messages"].as_array().expect("wire messages");
    assert!(wire_messages
        .iter()
        .any(|message| message["text"] == "ℹ️ Requires:"));
    assert!(!wire_messages
        .iter()
        .any(|message| message["text"] == "ℹ️ Assume:"));
}
#[test]
fn root_quotient_merge_in_assume_preserves_intrinsic_requires_not_assumes() {
    let (output, code) = run_cli(&[
        "eval",
        "sqrt(x)/sqrt(y)",
        "--domain",
        "assume",
        "--format",
        "json",
    ]);
    assert_eq!(code, 0);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "sqrt(x / y)");
    assert_eq!(wire["domain"]["mode"], "assume");
    assert_eq!(wire["required_display"], json!(["y > 0", "x ≥ 0"]));
    assert!(
        wire["assumptions_used"].is_null()
            || wire["assumptions_used"]
                .as_array()
                .is_some_and(Vec::is_empty),
        "expected no assumptions_used, got {:?}",
        wire["assumptions_used"]
    );

    let wire_messages = wire["wire"]["messages"].as_array().expect("wire messages");
    assert!(wire_messages
        .iter()
        .any(|message| message["text"] == "ℹ️ Requires:"));
    assert!(!wire_messages
        .iter()
        .any(|message| message["text"] == "ℹ️ Assume:"));
}
#[test]
fn root_product_merge_is_disabled_in_strict_but_definedness_requires_remain() {
    let (output, code) = run_cli(&[
        "eval",
        "sqrt(x)*sqrt(y)",
        "--domain",
        "strict",
        "--format",
        "json",
    ]);
    assert_eq!(code, 0);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "sqrt(x)·sqrt(y)");
    assert_eq!(wire["domain"]["mode"], "strict");
    assert_eq!(wire["required_display"], json!(["x ≥ 0", "y ≥ 0"]));
}
#[test]
fn const_fold_safe_real_sqrt_negative_keeps_warning_contract() {
    let (output, _code) = run_cli(&[
        "eval",
        "sqrt(-1)",
        "--format",
        "json",
        "--value-domain",
        "real",
        "--complex",
        "on",
        "--const-fold",
        "safe",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["options"]["const_fold"], "safe");
    assert_eq!(wire["semantics"]["value_domain"], "real");
    assert_eq!(wire["result"], "undefined");

    let warnings = wire["warnings"].as_array().expect("warnings array");
    assert_eq!(warnings.len(), 1, "real+safe should emit one warning");
    assert_eq!(warnings[0]["rule"], "Imaginary Usage Warning");
}
#[test]
fn same_root_family_power_quotient_reaches_cli_eval_path_exactly() {
    let (output, _code) = run_cli(&[
        "eval",
        "(sqrt(x^2 + 1)^5)/(sqrt(x^2 + 1)^3)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "x^2 + 1");
    assert_eq!(wire["required_display"], json!([]));
}
#[test]
fn abs_sqrt_denominator_collapses_to_positive_base_guard() {
    let (output, _code) = run_cli(&[
        "eval",
        "1/abs(sqrt(u))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "1 / sqrt(u)");

    let required = wire["required_display"]
        .as_array()
        .expect("required display");
    assert_eq!(
        required
            .iter()
            .filter_map(|item| item.as_str())
            .collect::<Vec<_>>(),
        vec!["u > 0"],
        "abs(sqrt(u)) denominator should compress to a single positivity guard"
    );
}
#[test]
fn root_nesting_drops_intrinsically_nonnegative_radicand_require() {
    let (output, _code) = run_cli(&[
        "eval",
        "sqrt(x^2 + 2*sqrt(x^2 + 1) + 2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "sqrt(x^2 + 1) + 1");
    assert_eq!(wire["required_display"], json!([]));
}
