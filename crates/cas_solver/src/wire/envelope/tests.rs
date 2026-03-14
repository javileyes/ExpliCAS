use super::evaluate_envelope_wire_command;
use cas_api_models::{EvalDomainMode, EvalValueDomain};
use serde_json::Value;

fn parse_envelope(payload: &str) -> Value {
    serde_json::from_str(payload).expect("valid envelope json")
}

fn assert_display_is_one_of(actual: &Value, expected: &[&str]) {
    let display = actual.as_str().expect("display string");
    assert!(
        expected.contains(&display),
        "unexpected display {:?}, expected one of {:?}",
        display,
        expected
    );
}

#[test]
fn evaluate_envelope_wire_command_returns_wire_contract() {
    let payload =
        evaluate_envelope_wire_command("x + x", EvalDomainMode::Generic, EvalValueDomain::Real);
    assert!(payload.contains("\"schema_version\": 1"));
    assert!(payload.contains("\"kind\": \"eval_result\""));
}

#[test]
fn evaluate_envelope_wire_command_complex_value_domain_preserves_sqrt_negative_without_warning() {
    let payload = evaluate_envelope_wire_command(
        "sqrt(-1)",
        EvalDomainMode::Generic,
        EvalValueDomain::Complex,
    );
    let wire = parse_envelope(&payload);

    assert_eq!(wire["result"]["kind"], "eval_result");
    assert_eq!(wire["result"]["value"]["display"], "(-1)^(1/2)");
    assert_eq!(
        wire["transparency"]["required_conditions"]
            .as_array()
            .expect("required_conditions array")
            .len(),
        0
    );
    assert_eq!(
        wire["transparency"]["assumptions_used"]
            .as_array()
            .expect("assumptions_used array")
            .len(),
        0
    );
}

#[test]
fn evaluate_envelope_wire_command_real_value_domain_surfaces_imaginary_warning() {
    let payload =
        evaluate_envelope_wire_command("sqrt(-1)", EvalDomainMode::Generic, EvalValueDomain::Real);
    let wire = parse_envelope(&payload);

    assert_eq!(wire["result"]["kind"], "eval_result");
    assert_eq!(wire["result"]["value"]["display"], "(-1)^(1/2)");
    let assumptions_used = wire["transparency"]["assumptions_used"]
        .as_array()
        .expect("assumptions_used array");
    assert_eq!(assumptions_used.len(), 1);
    assert_eq!(assumptions_used[0]["kind"], "domain_warning");
    assert_eq!(assumptions_used[0]["rule"], "Imaginary Usage Warning");
}

#[test]
fn evaluate_envelope_wire_command_complex_value_domain_collapses_explicit_i_forms() {
    let i_squared =
        evaluate_envelope_wire_command("i^2", EvalDomainMode::Generic, EvalValueDomain::Complex);
    let i_squared_wire = parse_envelope(&i_squared);
    assert_eq!(i_squared_wire["result"]["value"]["display"], "-1");
    assert!(i_squared_wire["transparency"]["assumptions_used"]
        .as_array()
        .expect("assumptions_used array")
        .is_empty());

    let reciprocal =
        evaluate_envelope_wire_command("1/i", EvalDomainMode::Generic, EvalValueDomain::Complex);
    let reciprocal_wire = parse_envelope(&reciprocal);
    assert_eq!(reciprocal_wire["result"]["value"]["display"], "-i");
    assert!(reciprocal_wire["transparency"]["assumptions_used"]
        .as_array()
        .expect("assumptions_used array")
        .is_empty());

    let gaussian_fraction = evaluate_envelope_wire_command(
        "(1+i)/(1-i)",
        EvalDomainMode::Generic,
        EvalValueDomain::Complex,
    );
    let gaussian_fraction_wire = parse_envelope(&gaussian_fraction);
    assert_eq!(gaussian_fraction_wire["result"]["value"]["display"], "i");
    assert!(gaussian_fraction_wire["transparency"]["assumptions_used"]
        .as_array()
        .expect("assumptions_used array")
        .is_empty());
}

#[test]
fn evaluate_envelope_wire_command_value_domain_splits_log_exp_variable_case_only_when_needed() {
    let real_payload =
        evaluate_envelope_wire_command("ln(exp(x))", EvalDomainMode::Strict, EvalValueDomain::Real);
    let real_wire = parse_envelope(&real_payload);
    assert_eq!(real_wire["result"]["value"]["display"], "x");
    let real_required = real_wire["transparency"]["required_conditions"]
        .as_array()
        .expect("required_conditions array");
    assert_eq!(real_required.len(), 1);
    assert_eq!(real_required[0]["kind"], "Positive");
    assert_eq!(real_required[0]["expr_canonical"], "e^x");

    let complex_payload = evaluate_envelope_wire_command(
        "ln(exp(x))",
        EvalDomainMode::Strict,
        EvalValueDomain::Complex,
    );
    let complex_wire = parse_envelope(&complex_payload);
    assert_eq!(complex_wire["result"]["value"]["display"], "ln(e^x)");
    let complex_required = complex_wire["transparency"]["required_conditions"]
        .as_array()
        .expect("required_conditions array");
    assert_eq!(complex_required.len(), 1);
    assert_eq!(complex_required[0]["kind"], "Positive");
    assert_eq!(complex_required[0]["expr_canonical"], "e^x");
    assert!(complex_wire["transparency"]["assumptions_used"]
        .as_array()
        .expect("assumptions_used array")
        .is_empty());
}

#[test]
fn evaluate_envelope_wire_command_value_domain_keeps_provable_log_exp_constant_stable() {
    let real_payload =
        evaluate_envelope_wire_command("exp(ln(5))", EvalDomainMode::Strict, EvalValueDomain::Real);
    let real_wire = parse_envelope(&real_payload);
    assert_eq!(real_wire["result"]["value"]["display"], "5");
    assert!(real_wire["transparency"]["required_conditions"]
        .as_array()
        .expect("required_conditions array")
        .is_empty());

    let complex_payload = evaluate_envelope_wire_command(
        "exp(ln(5))",
        EvalDomainMode::Strict,
        EvalValueDomain::Complex,
    );
    let complex_wire = parse_envelope(&complex_payload);
    assert_eq!(complex_wire["result"]["value"]["display"], "5");
    assert!(complex_wire["transparency"]["required_conditions"]
        .as_array()
        .expect("required_conditions array")
        .is_empty());
    assert!(complex_wire["transparency"]["assumptions_used"]
        .as_array()
        .expect("assumptions_used array")
        .is_empty());
}

#[test]
fn evaluate_envelope_wire_command_real_value_domain_preserves_explicit_i_forms_with_warning() {
    let i_squared =
        evaluate_envelope_wire_command("i^2", EvalDomainMode::Generic, EvalValueDomain::Real);
    let i_squared_wire = parse_envelope(&i_squared);
    assert_eq!(i_squared_wire["result"]["value"]["display"], "i^2");
    let i_squared_assumed = i_squared_wire["transparency"]["assumptions_used"]
        .as_array()
        .expect("assumptions_used array");
    assert_eq!(i_squared_assumed.len(), 1);
    assert_eq!(i_squared_assumed[0]["rule"], "Imaginary Usage Warning");

    let reciprocal =
        evaluate_envelope_wire_command("1/i", EvalDomainMode::Generic, EvalValueDomain::Real);
    let reciprocal_wire = parse_envelope(&reciprocal);
    assert_eq!(reciprocal_wire["result"]["value"]["display"], "1 / i");
    let reciprocal_assumed = reciprocal_wire["transparency"]["assumptions_used"]
        .as_array()
        .expect("assumptions_used array");
    assert_eq!(reciprocal_assumed.len(), 1);
    assert_eq!(reciprocal_assumed[0]["rule"], "Imaginary Usage Warning");

    let gaussian_fraction = evaluate_envelope_wire_command(
        "(1+i)/(1-i)",
        EvalDomainMode::Generic,
        EvalValueDomain::Real,
    );
    let gaussian_fraction_wire = parse_envelope(&gaussian_fraction);
    assert_eq!(
        gaussian_fraction_wire["result"]["value"]["display"],
        "(1 + i) / (1 - i)"
    );
    let gaussian_assumed = gaussian_fraction_wire["transparency"]["assumptions_used"]
        .as_array()
        .expect("assumptions_used array");
    assert_eq!(gaussian_assumed.len(), 1);
    assert_eq!(gaussian_assumed[0]["rule"], "Imaginary Usage Warning");
}

#[test]
fn evaluate_envelope_wire_command_assume_preserves_required_conditions_for_x_over_x() {
    let payload =
        evaluate_envelope_wire_command("x/x", EvalDomainMode::Assume, EvalValueDomain::Real);
    let wire = parse_envelope(&payload);

    assert_eq!(wire["result"]["value"]["display"], "1");
    let required = wire["transparency"]["required_conditions"]
        .as_array()
        .expect("required_conditions array");
    assert_eq!(required.len(), 1);
    assert_eq!(required[0]["kind"], "NonZero");
    assert_eq!(required[0]["expr_canonical"], "x");
    assert_eq!(
        wire["transparency"]["assumptions_used"]
            .as_array()
            .expect("assumptions_used array")
            .len(),
        0
    );
}

#[test]
fn evaluate_envelope_wire_command_strict_preserves_x_over_x_and_surfaces_blocked_hints() {
    let payload =
        evaluate_envelope_wire_command("x/x", EvalDomainMode::Strict, EvalValueDomain::Real);
    let wire = parse_envelope(&payload);

    assert_eq!(wire["result"]["value"]["display"], "x / x");
    let required = wire["transparency"]["required_conditions"]
        .as_array()
        .expect("required_conditions array");
    assert_eq!(required.len(), 1);
    assert_eq!(required[0]["kind"], "NonZero");

    let blocked = wire["transparency"]["blocked_hints"]
        .as_array()
        .expect("blocked_hints array");
    assert!(
        !blocked.is_empty(),
        "strict envelope should surface blocked hints for x/x"
    );
    assert!(
        blocked.iter().any(|hint| hint["tip"]
            .as_str()
            .is_some_and(|tip| tip.contains("domain generic"))),
        "strict envelope should suggest domain generic in blocked hints"
    );
}

#[test]
fn evaluate_envelope_wire_command_generic_simplifies_x_over_x_without_blocked_hints() {
    let payload =
        evaluate_envelope_wire_command("x/x", EvalDomainMode::Generic, EvalValueDomain::Real);
    let wire = parse_envelope(&payload);

    assert_eq!(wire["result"]["value"]["display"], "1");
    let required = wire["transparency"]["required_conditions"]
        .as_array()
        .expect("required_conditions array");
    assert_eq!(required.len(), 1);
    assert_eq!(required[0]["kind"], "NonZero");
    assert!(
        wire["transparency"]["blocked_hints"].is_null(),
        "generic envelope should omit blocked_hints when empty"
    );
}

#[test]
fn evaluate_envelope_wire_command_strict_preserves_exp_ln_x_with_intrinsic_guard() {
    let payload =
        evaluate_envelope_wire_command("exp(ln(x))", EvalDomainMode::Strict, EvalValueDomain::Real);
    let wire = parse_envelope(&payload);

    assert_eq!(wire["result"]["value"]["display"], "e^ln(x)");
    let required = wire["transparency"]["required_conditions"]
        .as_array()
        .expect("required_conditions array");
    assert_eq!(required.len(), 1);
    assert_eq!(required[0]["kind"], "Positive");
    assert_eq!(required[0]["expr_canonical"], "x");
    assert!(wire["transparency"]["assumptions_used"]
        .as_array()
        .expect("assumptions_used array")
        .is_empty());
}

#[test]
fn evaluate_envelope_wire_command_strict_keeps_abs_radical_family_safe_without_assumptions() {
    let ln_payload =
        evaluate_envelope_wire_command("ln(a^2)", EvalDomainMode::Strict, EvalValueDomain::Real);
    let ln_wire = parse_envelope(&ln_payload);

    assert_display_is_one_of(
        &ln_wire["result"]["value"]["display"],
        &["2·ln(|a|)", "2 * ln(|a|)"],
    );
    let ln_required = ln_wire["transparency"]["required_conditions"]
        .as_array()
        .expect("required_conditions array");
    assert_eq!(ln_required.len(), 2);
    assert_eq!(ln_required[0]["kind"], "Positive");
    assert_eq!(ln_required[0]["expr_canonical"], "a^2");
    assert_eq!(ln_required[1]["expr_canonical"], "|a|");
    assert!(ln_wire["transparency"]["assumptions_used"]
        .as_array()
        .expect("assumptions_used array")
        .is_empty());

    let sqrt_payload =
        evaluate_envelope_wire_command("sqrt(x^2)", EvalDomainMode::Strict, EvalValueDomain::Real);
    let sqrt_wire = parse_envelope(&sqrt_payload);

    assert_eq!(sqrt_wire["result"]["value"]["display"], "sqrt(x^2)");
    assert!(sqrt_wire["transparency"]["required_conditions"]
        .as_array()
        .expect("required_conditions array")
        .is_empty());
    assert!(sqrt_wire["transparency"]["assumptions_used"]
        .as_array()
        .expect("assumptions_used array")
        .is_empty());
}

#[test]
fn evaluate_envelope_wire_command_generic_simplifies_exp_ln_x_from_intrinsic_guard() {
    let payload = evaluate_envelope_wire_command(
        "exp(ln(x))",
        EvalDomainMode::Generic,
        EvalValueDomain::Real,
    );
    let wire = parse_envelope(&payload);

    assert_eq!(wire["result"]["value"]["display"], "x");
    let required = wire["transparency"]["required_conditions"]
        .as_array()
        .expect("required_conditions array");
    assert_eq!(required.len(), 1);
    assert_eq!(required[0]["kind"], "Positive");
    assert_eq!(required[0]["expr_canonical"], "x");
    assert!(wire["transparency"]["assumptions_used"]
        .as_array()
        .expect("assumptions_used array")
        .is_empty());
}

#[test]
fn evaluate_envelope_wire_command_generic_preserves_log_exp_inverse_without_assumptions() {
    let payload = evaluate_envelope_wire_command(
        "log(b,b^x)",
        EvalDomainMode::Generic,
        EvalValueDomain::Real,
    );
    let wire = parse_envelope(&payload);

    assert_eq!(wire["result"]["value"]["display"], "log(b, b^x)");
    assert_eq!(
        wire["transparency"]["required_conditions"]
            .as_array()
            .expect("required_conditions array")
            .len(),
        0
    );
    assert_eq!(
        wire["transparency"]["assumptions_used"]
            .as_array()
            .expect("assumptions_used array")
            .len(),
        0
    );
    assert!(
        wire["transparency"]["blocked_hints"].is_null(),
        "generic envelope should omit blocked_hints when empty"
    );
}

#[test]
fn evaluate_envelope_wire_command_generic_keeps_safe_forms_for_even_power_and_abs() {
    let ln_payload =
        evaluate_envelope_wire_command("ln(a^2)", EvalDomainMode::Generic, EvalValueDomain::Real);
    let ln_wire = parse_envelope(&ln_payload);

    assert_display_is_one_of(
        &ln_wire["result"]["value"]["display"],
        &["2·ln(|a|)", "2 * ln(|a|)"],
    );
    let ln_required = ln_wire["transparency"]["required_conditions"]
        .as_array()
        .expect("required_conditions array");
    assert_eq!(ln_required.len(), 2);
    assert_eq!(ln_required[0]["kind"], "Positive");
    assert_eq!(ln_required[0]["expr_canonical"], "a^2");
    assert_eq!(ln_required[1]["expr_canonical"], "|a|");
    assert!(ln_wire["transparency"]["assumptions_used"]
        .as_array()
        .expect("assumptions_used array")
        .is_empty());

    let sqrt_payload =
        evaluate_envelope_wire_command("sqrt(x^2)", EvalDomainMode::Generic, EvalValueDomain::Real);
    let sqrt_wire = parse_envelope(&sqrt_payload);

    assert_eq!(sqrt_wire["result"]["value"]["display"], "|x|");
    assert!(sqrt_wire["transparency"]["required_conditions"]
        .as_array()
        .expect("required_conditions array")
        .is_empty());
    assert!(sqrt_wire["transparency"]["assumptions_used"]
        .as_array()
        .expect("assumptions_used array")
        .is_empty());
}

#[test]
fn evaluate_envelope_wire_command_assume_collapses_even_power_and_abs_with_warning() {
    let ln_payload =
        evaluate_envelope_wire_command("ln(a^2)", EvalDomainMode::Assume, EvalValueDomain::Real);
    let ln_wire = parse_envelope(&ln_payload);

    assert_display_is_one_of(
        &ln_wire["result"]["value"]["display"],
        &["2·ln(a)", "2 * ln(a)"],
    );
    let ln_required = ln_wire["transparency"]["required_conditions"]
        .as_array()
        .expect("required_conditions array");
    assert_eq!(ln_required.len(), 2);
    assert_eq!(ln_required[0]["expr_canonical"], "a");
    let ln_assumed = ln_wire["transparency"]["assumptions_used"]
        .as_array()
        .expect("assumptions_used array");
    assert_eq!(ln_assumed.len(), 1);
    assert_eq!(ln_assumed[0]["rule"], "Log Even Power");
    assert_eq!(ln_assumed[0]["display"], "a > 0");

    let sqrt_payload =
        evaluate_envelope_wire_command("sqrt(x^2)", EvalDomainMode::Assume, EvalValueDomain::Real);
    let sqrt_wire = parse_envelope(&sqrt_payload);

    assert_eq!(sqrt_wire["result"]["value"]["display"], "x");
    assert!(sqrt_wire["transparency"]["required_conditions"]
        .as_array()
        .expect("required_conditions array")
        .is_empty());
    let sqrt_assumed = sqrt_wire["transparency"]["assumptions_used"]
        .as_array()
        .expect("assumptions_used array");
    assert_eq!(sqrt_assumed.len(), 1);
    assert_eq!(sqrt_assumed[0]["rule"], "Abs Under Positivity");
    assert_eq!(sqrt_assumed[0]["display"], "x > 0");
}

#[test]
fn evaluate_envelope_wire_command_assume_surfaces_log_exp_warning_and_guard() {
    let payload =
        evaluate_envelope_wire_command("log(b,b^x)", EvalDomainMode::Assume, EvalValueDomain::Real);
    let wire = parse_envelope(&payload);

    assert_eq!(wire["result"]["value"]["display"], "x");

    let required = wire["transparency"]["required_conditions"]
        .as_array()
        .expect("required_conditions array");
    assert_eq!(required.len(), 1);
    assert_eq!(required[0]["kind"], "NonZero");
    assert_eq!(required[0]["expr_canonical"], "b - 1");

    let assumptions_used = wire["transparency"]["assumptions_used"]
        .as_array()
        .expect("assumptions_used array");
    assert_eq!(assumptions_used.len(), 1);
    assert_eq!(assumptions_used[0]["kind"], "domain_warning");
    assert_eq!(assumptions_used[0]["rule"], "Log-Exp Inverse");
    assert_eq!(assumptions_used[0]["display"], "b > 0");
}
