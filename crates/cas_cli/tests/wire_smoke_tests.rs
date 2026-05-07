//! Smoke tests for the eval wire model.
//!
//! Verifies that the wire field is present and correctly structured
//! in JSON output from eval commands.

use assert_cmd::cargo;
use serde_json::Value;
use std::process::Command;

/// Helper to run eval and parse the wire field from the JSON output.
fn eval_wire(expr: &str) -> Value {
    let output = Command::new(cargo::cargo_bin!("cas_cli"))
        .arg("eval")
        .arg(expr)
        .arg("--format")
        .arg("json")
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: Value = serde_json::from_str(&stdout).expect("Failed to parse JSON");
    json.get("wire").cloned().unwrap_or(Value::Null)
}

/// Helper to run eval and parse the full JSON output.
fn eval_json_with_args(expr: &str, args: &[&str]) -> Value {
    let mut command = Command::new(cargo::cargo_bin!("cas_cli"));
    command.arg("eval").arg(expr).arg("--format").arg("json");
    command.args(args);

    let output = command.output().expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).expect("Failed to parse JSON")
}

/// Helper to run eval and parse the full JSON output.
fn eval_json(expr: &str) -> Value {
    eval_json_with_args(expr, &[])
}

#[test]
fn test_wire_present_on_success() {
    let wire = eval_wire("2+2");

    // Wire should exist
    assert!(!wire.is_null(), "wire field should be present");

    // Schema version should be 1
    assert_eq!(
        wire.get("schema_version"),
        Some(&Value::Number(1.into())),
        "schema_version should be 1"
    );

    // Messages should be an array
    let messages = wire.get("messages").expect("messages should exist");
    assert!(messages.is_array(), "messages should be an array");
    assert!(
        !messages.as_array().unwrap().is_empty(),
        "messages should not be empty"
    );
}

#[test]
fn test_eval_negative_power_denominator_preserves_required_conditions() {
    let json = eval_json("ln(y)*(z+1)^(-2)");

    assert_eq!(json["ok"], true);
    assert_eq!(json["result"], "ln(y) / (z + 1)^2");

    let required = json["required_conditions"]
        .as_array()
        .expect("required_conditions should be an array");

    assert!(
        required
            .iter()
            .any(|condition| condition["kind"] == "Positive" && condition["expr_display"] == "y"),
        "expected Positive(y), got: {required:?}"
    );
    assert!(
        required
            .iter()
            .any(|condition| condition["kind"] == "NonZero"
                && condition["expr_display"] == "z + 1"),
        "expected NonZero(z + 1), got: {required:?}"
    );
}

#[test]
fn test_eval_calculus_residual_domain_modes_preserve_required_conditions() {
    let expr = "diff(integrate(ln(y)*(z+1)^(-2), x), x) - ln(y)*(z+1)^(-2)";

    for mode in ["generic", "assume"] {
        let json = eval_json_with_args(expr, &["--domain", mode]);

        assert_eq!(json["ok"], true, "eval should succeed in {mode}");
        assert_eq!(json["result"], "0", "residual should verify in {mode}");
        assert_eq!(json["domain"]["mode"], mode);
        assert_eq!(json["semantics"]["domain_mode"], mode);

        let required = json["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array");
        assert_eq!(
            required.len(),
            2,
            "expected only the intrinsic input-domain guards in {mode}: {required:?}"
        );
        assert!(
            required.iter().any(|condition| {
                condition["kind"] == "Positive" && condition["expr_canonical"] == "y"
            }),
            "expected Positive(y) in {mode}, got: {required:?}"
        );
        assert!(
            required.iter().any(|condition| {
                condition["kind"] == "NonZero" && condition["expr_canonical"] == "z + 1"
            }),
            "expected NonZero(z + 1) in {mode}, got: {required:?}"
        );

        let messages = json["wire"]["messages"]
            .as_array()
            .expect("wire messages should be an array");
        assert!(
            messages.iter().any(|message| message["text"]
                .as_str()
                .is_some_and(|text| text.contains("Requires:"))),
            "intrinsic calculus-domain guards should render as Requires in {mode}: {messages:?}"
        );
        assert!(
            !messages.iter().any(|message| message["text"]
                .as_str()
                .is_some_and(|text| text.contains("Assume:"))),
            "intrinsic calculus-domain guards should not render as Assume in {mode}: {messages:?}"
        );
    }
}

#[test]
fn test_eval_json_steps_preserve_post_calculus_presentation_step() {
    let cases = [
        (
            "diff(integrate(3/(2*sqrt(3*x+1)*cosh(sqrt(3*x+1))^2), x), x)",
            "3 / (2·sqrt(3·x + 1)·cosh(sqrt(3·x + 1))^2)",
            "((3 · x + 1))^(-1/2)",
            ["sqrt(3 · x + 1)", "cosh(sqrt(3 · x + 1))"],
            Some("3·x + 1"),
        ),
        (
            "diff(arctan(sqrt(x)), x)",
            "1 / (2·sqrt(x)·(x + 1))",
            "x^(-1/2)",
            ["sqrt(x)", "x + 1"],
            Some("x"),
        ),
        (
            "diff(arctan(sqrt(3*x)), x)",
            "3 / (2·sqrt(3·x)·(3·x + 1))",
            "((3 · x))^(-1/2)",
            ["sqrt(3 · x)", "3 · x + 1"],
            Some("x"),
        ),
        (
            "diff(arctan(sqrt(x+1)), x)",
            "1 / (2·sqrt(x + 1)·(x + 2))",
            "((x + 1))^(-1/2)",
            ["sqrt(x + 1)", "x + 2"],
            Some("x + 1"),
        ),
        (
            "diff(arctan(sqrt(3-2*x)), x)",
            "-1 / (2·sqrt(3 - 2·x)·(2 - x))",
            "((3 - 2 · x))^(-1/2)",
            ["sqrt(3 - 2 · x)", "2 - x"],
            Some("3 - 2·x"),
        ),
        (
            "diff(arctan(sqrt(x^2+1)), x)",
            "x / (sqrt(x^2 + 1)·(x^2 + 2))",
            "((x^2 + 1))^(-1/2)",
            ["sqrt(x^2 + 1)", "x^2 + 2"],
            None,
        ),
    ];

    for (expr, expected_result, expected_before, expected_after, required_expr) in cases {
        let json = eval_json_with_args(expr, &["--steps", "on"]);

        assert_eq!(json["ok"], true, "expr: {expr}");
        assert_eq!(json["result"], expected_result, "expr: {expr}");

        let steps = json["steps"].as_array().expect("steps should be an array");
        assert!(
            steps.iter().any(|step| {
                step["rule"] == "Presentar resultado de cálculo en forma compacta"
                    && step["before"]
                        .as_str()
                        .is_some_and(|before| before.contains(expected_before))
                    && step["after"].as_str().is_some_and(|after| {
                        expected_after.iter().all(|needle| after.contains(needle))
                    })
            }),
            "expected public JSON steps to include compact post-calculus presentation for {expr}: {steps:?}"
        );

        let required = json["required_conditions"]
            .as_array()
            .expect("required_conditions should be an array");
        if let Some(required_expr) = required_expr {
            assert!(
                required.iter().any(|condition| {
                    condition["kind"] == "Positive" && condition["expr_canonical"] == required_expr
                }),
                "post-calculus presentation should preserve the sqrt-domain guard for {expr}: {required:?}"
            );
        } else {
            assert!(
                required.is_empty(),
                "post-calculus presentation should not add redundant sqrt-domain guards for {expr}: {required:?}"
            );
        }
    }
}

#[test]
fn test_eval_integral_assume_mode_separates_assumptions_from_requires() {
    let generic = eval_json_with_args("integrate(ln(a^2), x)", &["--domain", "generic"]);

    assert_eq!(generic["ok"], true);
    assert_eq!(generic["result"], "2·x·ln(|a|)");
    assert_eq!(generic["domain"]["mode"], "generic");
    assert!(generic["assumptions_used"].is_null());

    let generic_required = generic["required_conditions"]
        .as_array()
        .expect("generic required_conditions should be an array");
    assert_eq!(
        generic_required.len(),
        1,
        "expected only the absolute-log domain guard: {generic_required:?}"
    );
    assert_eq!(generic_required[0]["kind"], "NonZero");
    assert_eq!(generic_required[0]["expr_canonical"], "a");

    let generic_messages = generic["wire"]["messages"]
        .as_array()
        .expect("generic wire messages should be an array");
    assert!(
        generic_messages.iter().any(|message| message["text"]
            .as_str()
            .is_some_and(|text| text.contains("Requires:"))),
        "generic mode should render the log-domain guard as Requires: {generic_messages:?}"
    );
    assert!(
        !generic_messages.iter().any(|message| message["text"]
            .as_str()
            .is_some_and(|text| text.contains("Assume:"))),
        "generic mode should not render an assumption: {generic_messages:?}"
    );

    let assume = eval_json_with_args("integrate(ln(a^2), x)", &["--domain", "assume"]);

    assert_eq!(assume["ok"], true);
    assert_eq!(assume["result"], "2·x·ln(a)");
    assert_eq!(assume["domain"]["mode"], "assume");
    assert!(
        assume["required_conditions"]
            .as_array()
            .expect("assume required_conditions should be an array")
            .is_empty(),
        "assume output should not duplicate a > 0 as a separate required condition"
    );

    let assumptions = assume["assumptions_used"]
        .as_array()
        .expect("assumptions_used should be an array in assume mode");
    assert_eq!(assumptions.len(), 1);
    assert_eq!(assumptions[0]["display"], "a > 0");
    assert_eq!(assumptions[0]["expr_canonical"], "a");
    assert_eq!(assumptions[0]["rule"], "Log Even Power");

    let assume_messages = assume["wire"]["messages"]
        .as_array()
        .expect("assume wire messages should be an array");
    assert!(
        assume_messages.iter().any(|message| message["text"]
            .as_str()
            .is_some_and(|text| text.contains("Assume:"))),
        "assume mode should render the positivity condition as Assume: {assume_messages:?}"
    );
    assert!(
        !assume_messages.iter().any(|message| message["text"]
            .as_str()
            .is_some_and(|text| text.contains("Requires:"))),
        "assume integral output should not also render Requires: {assume_messages:?}"
    );

    let residual_expr = "diff(integrate(ln(a^2), x), x) - ln(a^2)";
    for mode in ["generic", "assume"] {
        let residual = eval_json_with_args(residual_expr, &["--domain", mode]);
        assert_eq!(
            residual["ok"], true,
            "residual eval should succeed in {mode}"
        );
        assert_eq!(
            residual["result"], "0",
            "antiderivative should verify by differentiation in {mode}"
        );
    }
}

#[test]
fn test_eval_integral_radical_abs_boundary_is_conservative_until_assumed() {
    let generic = eval_json_with_args("integrate(sqrt(x^2), x)", &["--domain", "generic"]);

    assert_eq!(generic["ok"], true);
    assert_eq!(generic["result"], "integrate(|x|, x)");
    assert_eq!(generic["domain"]["mode"], "generic");
    assert!(generic["assumptions_used"].is_null());
    assert!(
        generic["required_conditions"]
            .as_array()
            .expect("generic required_conditions should be an array")
            .is_empty(),
        "generic abs integral should not invent domain requirements"
    );

    let generic_residual = eval_json_with_args(
        "diff(integrate(sqrt(x^2), x), x) - sqrt(x^2)",
        &["--domain", "generic"],
    );
    assert_eq!(generic_residual["ok"], true);
    assert_eq!(
        generic_residual["result"], "diff(integrate(|x|, x), x) - |x|",
        "generic mode should leave the unsupported abs antiderivative as a residual"
    );

    let assume = eval_json_with_args("integrate(sqrt(x^2), x)", &["--domain", "assume"]);

    assert_eq!(assume["ok"], true);
    assert_eq!(assume["result"], "1/2·x^2");
    assert_eq!(assume["domain"]["mode"], "assume");
    assert!(
        assume["required_conditions"]
            .as_array()
            .expect("assume required_conditions should be an array")
            .is_empty(),
        "assume output should not duplicate x > 0 as a separate required condition"
    );

    let assumptions = assume["assumptions_used"]
        .as_array()
        .expect("assumptions_used should be an array in assume mode");
    assert_eq!(assumptions.len(), 1);
    assert_eq!(assumptions[0]["display"], "x > 0");
    assert_eq!(assumptions[0]["expr_canonical"], "x");
    assert_eq!(assumptions[0]["rule"], "Abs Under Positivity");

    let assume_messages = assume["wire"]["messages"]
        .as_array()
        .expect("assume wire messages should be an array");
    assert!(
        assume_messages.iter().any(|message| message["text"]
            .as_str()
            .is_some_and(|text| text.contains("Assume:"))),
        "assume mode should render the positivity condition as Assume: {assume_messages:?}"
    );
    assert!(
        !assume_messages.iter().any(|message| message["text"]
            .as_str()
            .is_some_and(|text| text.contains("Requires:"))),
        "assume radical integral output should not also render Requires: {assume_messages:?}"
    );

    let assume_residual = eval_json_with_args(
        "diff(integrate(sqrt(x^2), x), x) - sqrt(x^2)",
        &["--domain", "assume"],
    );
    assert_eq!(assume_residual["ok"], true);
    assert_eq!(
        assume_residual["result"], "0",
        "assumed radical antiderivative should verify by differentiation"
    );
    let residual_assumptions = assume_residual["assumptions_used"]
        .as_array()
        .expect("assumptions_used should be an array for assumed residual");
    assert!(
        residual_assumptions.iter().any(|assumption| {
            assumption["display"] == "x > 0" && assumption["rule"] == "Abs Under Positivity"
        }),
        "assumed residual should surface the positivity assumption: {residual_assumptions:?}"
    );
}

#[test]
fn test_eval_integral_independent_radical_constant_verifies_per_domain_mode() {
    let generic = eval_json_with_args("integrate(sqrt(a^2), x)", &["--domain", "generic"]);

    assert_eq!(generic["ok"], true);
    assert_eq!(generic["result"], "x·|a|");
    assert_eq!(generic["domain"]["mode"], "generic");
    assert!(generic["assumptions_used"].is_null());
    assert!(
        generic["required_conditions"]
            .as_array()
            .expect("generic required_conditions should be an array")
            .is_empty(),
        "generic independent radical constant should not invent domain requirements"
    );

    let generic_displayed_residual =
        eval_json_with_args("diff(x*abs(a), x) - sqrt(a^2)", &["--domain", "generic"]);
    assert_eq!(generic_displayed_residual["ok"], true);
    assert_eq!(
        generic_displayed_residual["result"], "0",
        "generic displayed antiderivative x*|a| should verify without assumptions"
    );

    let generic_assumed_form_residual =
        eval_json_with_args("diff(a*x, x) - sqrt(a^2)", &["--domain", "generic"]);
    assert_eq!(generic_assumed_form_residual["ok"], true);
    assert_eq!(
        generic_assumed_form_residual["result"], "a - |a|",
        "generic mode should not verify a*x without the positivity assumption"
    );

    let assume = eval_json_with_args("integrate(sqrt(a^2), x)", &["--domain", "assume"]);

    assert_eq!(assume["ok"], true);
    assert_eq!(assume["result"], "a·x");
    assert_eq!(assume["domain"]["mode"], "assume");
    assert!(
        assume["required_conditions"]
            .as_array()
            .expect("assume required_conditions should be an array")
            .is_empty(),
        "assume output should not duplicate a > 0 as a separate required condition"
    );

    let assumptions = assume["assumptions_used"]
        .as_array()
        .expect("assumptions_used should be an array in assume mode");
    assert_eq!(assumptions.len(), 1);
    assert_eq!(assumptions[0]["display"], "a > 0");
    assert_eq!(assumptions[0]["expr_canonical"], "a");
    assert_eq!(assumptions[0]["rule"], "Abs Under Positivity");

    let assume_displayed_residual =
        eval_json_with_args("diff(a*x, x) - sqrt(a^2)", &["--domain", "assume"]);
    assert_eq!(assume_displayed_residual["ok"], true);
    assert_eq!(
        assume_displayed_residual["result"], "0",
        "assume displayed antiderivative a*x should verify by differentiation"
    );
    let residual_assumptions = assume_displayed_residual["assumptions_used"]
        .as_array()
        .expect("assumptions_used should be an array for assumed displayed residual");
    assert!(
        residual_assumptions.iter().any(|assumption| {
            assumption["display"] == "a > 0" && assumption["rule"] == "Abs Under Positivity"
        }),
        "assumed displayed residual should surface the positivity assumption: {residual_assumptions:?}"
    );
}

#[test]
fn test_eval_integral_compound_independent_radical_constant_verifies_per_domain_mode() {
    let generic = eval_json_with_args("integrate(sqrt((a+1)^2), x)", &["--domain", "generic"]);

    assert_eq!(generic["ok"], true);
    assert_eq!(generic["result"], "x·|a + 1|");
    assert_eq!(generic["domain"]["mode"], "generic");
    assert!(generic["assumptions_used"].is_null());
    assert!(
        generic["required_conditions"]
            .as_array()
            .expect("generic required_conditions should be an array")
            .is_empty(),
        "generic compound radical constant should not invent domain requirements"
    );

    let generic_displayed_residual = eval_json_with_args(
        "diff(x*abs(a+1), x) - sqrt((a+1)^2)",
        &["--domain", "generic"],
    );
    assert_eq!(generic_displayed_residual["ok"], true);
    assert_eq!(
        generic_displayed_residual["result"], "0",
        "generic displayed antiderivative x*|a + 1| should verify without assumptions"
    );

    let generic_assumed_form_residual =
        eval_json_with_args("diff(x*(a+1), x) - sqrt((a+1)^2)", &["--domain", "generic"]);
    assert_eq!(generic_assumed_form_residual["ok"], true);
    assert_eq!(
        generic_assumed_form_residual["result"], "a + 1 - |a + 1|",
        "generic mode should not verify x*(a+1) without the compound positivity assumption"
    );

    let assume = eval_json_with_args("integrate(sqrt((a+1)^2), x)", &["--domain", "assume"]);

    assert_eq!(assume["ok"], true);
    assert_eq!(assume["result"], "x·(a + 1)");
    assert_eq!(assume["domain"]["mode"], "assume");
    assert!(
        assume["required_conditions"]
            .as_array()
            .expect("assume required_conditions should be an array")
            .is_empty(),
        "assume output should not duplicate a + 1 > 0 as a separate required condition"
    );

    let assumptions = assume["assumptions_used"]
        .as_array()
        .expect("assumptions_used should be an array in assume mode");
    assert_eq!(assumptions.len(), 1);
    assert_eq!(assumptions[0]["display"], "a + 1 > 0");
    assert_eq!(assumptions[0]["expr_canonical"], "a + 1");
    assert_eq!(assumptions[0]["rule"], "Abs Under Positivity");

    let assume_messages = assume["wire"]["messages"]
        .as_array()
        .expect("assume wire messages should be an array");
    assert!(
        assume_messages.iter().any(|message| message["text"]
            .as_str()
            .is_some_and(|text| text.contains("Assume:"))),
        "assume mode should render the compound positivity condition as Assume: {assume_messages:?}"
    );
    assert!(
        !assume_messages.iter().any(|message| message["text"]
            .as_str()
            .is_some_and(|text| text.contains("Requires:"))),
        "assume compound radical integral output should not also render Requires: {assume_messages:?}"
    );

    let assume_displayed_residual =
        eval_json_with_args("diff(x*(a+1), x) - sqrt((a+1)^2)", &["--domain", "assume"]);
    assert_eq!(assume_displayed_residual["ok"], true);
    assert_eq!(
        assume_displayed_residual["result"], "0",
        "assume displayed antiderivative x*(a+1) should verify by differentiation"
    );
    let residual_assumptions = assume_displayed_residual["assumptions_used"]
        .as_array()
        .expect("assumptions_used should be an array for assumed displayed residual");
    assert!(
        residual_assumptions.iter().any(|assumption| {
            assumption["display"] == "a + 1 > 0"
                && assumption["rule"] == "Abs Under Positivity"
        }),
        "assumed displayed residual should surface the compound positivity assumption: {residual_assumptions:?}"
    );
}

#[test]
fn test_eval_integral_reciprocal_compound_radical_constant_requires_abs_until_assumed() {
    let generic = eval_json_with_args("integrate(1/sqrt((a+1)^2), x)", &["--domain", "generic"]);

    assert_eq!(generic["ok"], true);
    assert_eq!(generic["result"], "x / |a + 1|");
    assert_eq!(generic["domain"]["mode"], "generic");
    assert!(generic["assumptions_used"].is_null());

    let generic_required = generic["required_conditions"]
        .as_array()
        .expect("generic required_conditions should be an array");
    assert_eq!(
        generic_required.len(),
        1,
        "expected denominator definability guard: {generic_required:?}"
    );
    assert_eq!(generic_required[0]["kind"], "NonZero");
    assert_eq!(generic_required[0]["expr_canonical"], "a + 1");

    let generic_displayed_residual = eval_json_with_args(
        "diff(x/abs(a+1), x) - 1/sqrt((a+1)^2)",
        &["--domain", "generic"],
    );
    assert_eq!(generic_displayed_residual["ok"], true);
    assert_eq!(
        generic_displayed_residual["result"], "0",
        "generic displayed antiderivative x/|a+1| should verify with nonzero guard"
    );

    let generic_assumed_form_residual = eval_json_with_args(
        "diff(x/(a+1), x) - 1/sqrt((a+1)^2)",
        &["--domain", "generic"],
    );
    assert_eq!(generic_assumed_form_residual["ok"], true);
    assert_eq!(
        generic_assumed_form_residual["result"], "(|a + 1| - a - 1) / (|a + 1|·(a + 1))",
        "generic mode should not verify x/(a+1) without the compound positivity assumption"
    );

    let assume = eval_json_with_args("integrate(1/sqrt((a+1)^2), x)", &["--domain", "assume"]);

    assert_eq!(assume["ok"], true);
    assert_eq!(assume["result"], "x / (a + 1)");
    assert_eq!(assume["domain"]["mode"], "assume");

    let assumptions = assume["assumptions_used"]
        .as_array()
        .expect("assumptions_used should be an array in assume mode");
    assert_eq!(assumptions.len(), 1);
    assert_eq!(assumptions[0]["display"], "a + 1 > 0");
    assert_eq!(assumptions[0]["expr_canonical"], "a + 1");
    assert_eq!(assumptions[0]["rule"], "Abs Under Positivity");
    assert!(
        assume["warnings"]
            .as_array()
            .expect("assume warnings should be an array")
            .is_empty(),
        "accepted assume-mode condition should not also be emitted as a warning"
    );
    assert_eq!(
        assume["required_conditions"]
            .as_array()
            .expect("assume required_conditions should be an array")
            .len(),
        0,
        "positive assumption should cover the redundant nonzero requirement"
    );
    assert_eq!(
        assume["required_display"]
            .as_array()
            .expect("assume required_display should be an array")
            .len(),
        0,
        "assume output should not render a covered Requires block"
    );
    let assume_messages = assume["wire"]["messages"]
        .as_array()
        .expect("assume wire messages should be an array");
    assert!(
        !assume_messages
            .iter()
            .any(|message| message["kind"] == "warn"),
        "accepted assume-mode condition should render only in Assume: {assume_messages:?}"
    );

    let assume_displayed_residual = eval_json_with_args(
        "diff(x/(a+1), x) - 1/sqrt((a+1)^2)",
        &["--domain", "assume"],
    );
    assert_eq!(assume_displayed_residual["ok"], true);
    assert_eq!(
        assume_displayed_residual["result"], "0",
        "assume displayed antiderivative x/(a+1) should verify by differentiation"
    );
    let residual_assumptions = assume_displayed_residual["assumptions_used"]
        .as_array()
        .expect("assumptions_used should be an array for assumed displayed residual");
    assert!(
        residual_assumptions.iter().any(|assumption| {
            assumption["display"] == "a + 1 > 0"
                && assumption["rule"] == "Abs Under Positivity"
        }),
        "assumed displayed residual should surface the compound positivity assumption: {residual_assumptions:?}"
    );
}

#[test]
fn test_eval_assume_mode_dedupes_shared_positive_assumption_across_rules() {
    let generic = eval_json_with_args("sqrt(abs(x))^2 + ln(x^2)", &["--domain", "generic"]);

    assert_eq!(generic["ok"], true);
    assert_eq!(generic["result"], "|x| + 2·ln(|x|)");
    let generic_required = generic["required_conditions"]
        .as_array()
        .expect("generic required_conditions should be an array");
    assert_eq!(
        generic_required.len(),
        1,
        "generic mode should preserve the log-domain nonzero guard: {generic_required:?}"
    );
    assert_eq!(generic_required[0]["kind"], "NonZero");
    assert_eq!(generic_required[0]["expr_canonical"], "x");

    let assume = eval_json_with_args("sqrt(abs(x))^2 + ln(x^2)", &["--domain", "assume"]);

    assert_eq!(assume["ok"], true);
    assert_eq!(assume["result"], "2·ln(x) + x");
    let assumptions = assume["assumptions_used"]
        .as_array()
        .expect("assumptions_used should be an array in assume mode");
    assert_eq!(
        assumptions.len(),
        1,
        "shared positive assumption should render once: {assumptions:?}"
    );
    assert_eq!(assumptions[0]["kind"], "positive");
    assert_eq!(assumptions[0]["display"], "x > 0");
    assert_eq!(assumptions[0]["expr_canonical"], "x");
    assert!(
        assume["required_conditions"]
            .as_array()
            .expect("assume required_conditions should be an array")
            .is_empty(),
        "positive assumption should cover the generic nonzero requirement"
    );

    let assume_messages = assume["wire"]["messages"]
        .as_array()
        .expect("assume wire messages should be an array");
    let assume_bullets = assume_messages
        .iter()
        .filter(|message| message["text"].as_str() == Some("  • x > 0"))
        .count();
    assert_eq!(
        assume_bullets, 1,
        "wire Assume block should not duplicate the same condition: {assume_messages:?}"
    );
}

#[test]
fn test_wire_output_message_present() {
    let wire = eval_wire("3*4");

    let messages = wire.get("messages").expect("messages should exist");
    let msgs = messages.as_array().unwrap();

    // Should have at least one output message
    let output_msgs: Vec<_> = msgs
        .iter()
        .filter(|m| m.get("kind") == Some(&Value::String("output".into())))
        .collect();

    assert!(
        !output_msgs.is_empty(),
        "should have at least one output message"
    );

    // Output message should contain result
    let output_text = output_msgs[0].get("text").unwrap().as_str().unwrap();
    assert!(
        output_text.contains("Result"),
        "output should contain 'Result'"
    );
    assert!(
        output_text.contains("12"),
        "output should contain the result value"
    );
}

#[test]
fn test_wire_steps_summary_when_enabled() {
    let output = Command::new(cargo::cargo_bin!("cas_cli"))
        .arg("eval")
        .arg("x^2 + 2*x + 1")
        .arg("--steps")
        .arg("on")
        .arg("--format")
        .arg("json")
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: Value = serde_json::from_str(&stdout).expect("Failed to parse JSON");
    let wire = json.get("wire").expect("wire should exist");

    let messages = wire.get("messages").expect("messages should exist");
    let msgs = messages.as_array().unwrap();

    // If there are steps, should have a steps summary message
    let steps_count = json
        .get("steps_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    if steps_count > 0 {
        let step_msgs: Vec<_> = msgs
            .iter()
            .filter(|m| m.get("kind") == Some(&Value::String("steps".into())))
            .collect();

        assert!(
            !step_msgs.is_empty(),
            "should have steps message when steps_count > 0"
        );
    }
}

#[test]
fn test_wire_message_order() {
    // Test that messages appear in expected order: warn, info, output, steps
    let wire = eval_wire("1/x"); // This might produce requires

    let messages = wire.get("messages").expect("messages should exist");
    let msgs = messages.as_array().unwrap();

    // Find position of output message
    let output_pos = msgs
        .iter()
        .position(|m| m.get("kind") == Some(&Value::String("output".into())));

    assert!(output_pos.is_some(), "should have output message");

    // Warn and info should come before output (if present)
    for (i, msg) in msgs.iter().enumerate() {
        let kind = msg.get("kind").and_then(|v| v.as_str()).unwrap_or("");
        if kind == "warn" || kind == "info" {
            assert!(
                i < output_pos.unwrap(),
                "warn/info messages should come before output"
            );
        }
    }
}

#[test]
fn test_wire_schema_version_stable() {
    // Verify schema_version is exactly 1 (not changing unexpectedly)
    for expr in &["1+1", "x^2", "sin(pi/2)"] {
        let wire = eval_wire(expr);
        assert_eq!(
            wire.get("schema_version"),
            Some(&Value::Number(1.into())),
            "schema_version should always be 1 for expr: {}",
            expr
        );
    }
}
