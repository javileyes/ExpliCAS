use super::*;

#[test]
fn expand_log_function_blocks_symbolic_product_expansion_in_generic() {
    let (output, code) = run_cli(&["eval", "expand_log(ln(x*y))", "--format", "json"]);
    assert_eq!(code, 0);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "ln(x·y)");
    assert_eq!(wire["domain"]["mode"], "generic");
    assert_eq!(wire["required_display"], json!(["x·y > 0"]));
}
#[test]
fn expand_log_function_allows_symbolic_product_expansion_in_assume() {
    let (output, code) = run_cli(&[
        "eval",
        "expand_log(ln(x*y))",
        "--domain",
        "assume",
        "--format",
        "json",
    ]);
    assert_eq!(code, 0);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "ln(x) + ln(y)");
    assert_eq!(wire["domain"]["mode"], "assume");
    assert_eq!(wire["required_display"], json!([]));
    assert_eq!(
        wire["assumptions_used"],
        json!([
            {
                "kind": "positive",
                "display": "x > 0",
                "expr_canonical": "x",
                "rule": "expand_log"
            },
            {
                "kind": "positive",
                "display": "y > 0",
                "expr_canonical": "y",
                "rule": "expand_log"
            }
        ])
    );
    let wire_messages = wire["wire"]["messages"].as_array().expect("wire messages");
    assert!(wire_messages
        .iter()
        .any(|message| message["text"] == "ℹ️ Assume:"));
    assert!(!wire_messages
        .iter()
        .any(|message| message["text"] == "ℹ️ Requires:"));
}
#[test]
fn expand_log_function_surfaces_assumptions_in_assume_envelope() {
    let (output, code) = run_cli(&["envelope", "expand_log(ln(x*y))", "--domain", "assume"]);
    assert_eq!(code, 0);
    let wire = parse_wire(&output);

    let assumptions = wire["transparency"]["assumptions_used"]
        .as_array()
        .expect("assumptions_used array");
    assert!(assumptions
        .iter()
        .any(|item| item["display"] == "x > 0" && item["rule"] == "expand_log"));
    assert!(assumptions
        .iter()
        .any(|item| item["display"] == "y > 0" && item["rule"] == "expand_log"));
}
#[test]
fn equiv_log_product_requires_both_sides_domain() {
    let (output, code) = run_cli(&["eval", "equiv(ln(x*y),ln(x)+ln(y))", "--format", "json"]);
    assert_eq!(code, 0);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "true");
    assert_eq!(wire["required_display"], json!(["x > 0", "y > 0"]));
}
#[test]
fn log_even_power_minus_negative_log_uses_inferred_negative_domain_for_abs() {
    let (output, code) = run_cli(&["eval", "ln(x^2) - 2*ln(-x)", "--format", "json"]);
    assert_eq!(code, 0);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "0");
    assert_eq!(wire["domain"]["mode"], "generic");
    assert_eq!(wire["required_display"], json!(["x < 0"]));
}
#[test]
fn exponential_quotient_drops_redundant_nonzero_requires() {
    let (output, _code) = run_cli(&["eval", "exp(x)/exp(y)", "--format", "json"]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "e^(x - y)");
    assert_eq!(
        wire["required_display"].as_array().map(Vec::len),
        Some(0),
        "exp quotient should not require e^y ≠ 0"
    );
}
/// Los cuelgues F13 que interceptaban el argumento antes de llegar a approx
/// quedan cerrados: potencias enteras gigantes ya no se materializan ni en el
/// extractor de factores de raíz (exponente entero = nada que extraer) ni en
/// el guard exacto de división por cero (cap = el techo de plegado).
#[test]
fn big_integer_powers_no_longer_hang_the_simplifier() {
    for (input, expected) in [
        ("5^(-123456789)", "1 / 5^123456789"),
        ("10^86297568", "10^86297568"),
        ("5^100000", "5^100000"),
    ] {
        let (output, _code) = run_cli(&["eval", input, "--format", "json", "--no-pretty"]);
        let wire = parse_wire(&output);
        assert_eq!(wire["result"], expected, "in: {output}");
    }
}
