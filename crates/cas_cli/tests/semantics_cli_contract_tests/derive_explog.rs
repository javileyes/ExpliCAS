use super::*;

#[test]
fn derive_json_exposes_strategy_metadata() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(2*x)/cos(x+x), tan(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");

    let messages = wire["wire"]["messages"].as_array().expect("wire messages");
    assert!(
        messages
            .iter()
            .any(|message| message["text"] == "Strategy: contract trig"),
        "expected nested wire envelope to include derive strategy metadata"
    );
    assert!(
        messages
            .iter()
            .any(|message| message["text"] == "1 step(s) via contract trig"),
        "expected nested wire envelope to describe derive steps using the active strategy"
    );
}
#[test]
fn derive_repeated_phase_shift_sum_uses_two_direct_expand_phase_shift_steps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)+cos(x)+sin(y)+cos(y), sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    assert_eq!(wire["steps_count"], 2);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert_rule_eq(&steps[0]["rule"], "Aplicar identidad de desfase");
    assert_rule_eq(&steps[1]["rule"], "Aplicar identidad de desfase");
}
#[test]
fn derive_repeated_phase_shift_sum_expansion_uses_two_direct_expand_phase_shift_steps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4), sin(x)+cos(x)+sin(y)+cos(y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 2);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert_rule_eq(&steps[0]["rule"], "Aplicar identidad de desfase");
    assert_rule_eq(&steps[1]["rule"], "Aplicar identidad de desfase");
}
#[test]
fn derive_product_to_sum_expansion_does_not_emit_depth_overflow_warning_to_stderr() {
    let bin_path = env!("CARGO_BIN_EXE_cas_cli");
    let output = Command::new(bin_path)
        .args([
            "eval",
            "derive 2*sin(2*x)*cos(x), sin(3*x)+sin(x)",
            "--format",
            "json",
            "--steps",
            "on",
        ])
        .output()
        .expect("Failed to execute binary");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let wire = parse_wire(&stdout);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar producto a suma");
    assert!(
        !stderr.contains("depth_overflow"),
        "expected direct product-to-sum expansion to stay quiet on stderr, got: {stderr}"
    );
}
#[test]
fn derive_binomial_expansion_with_cancellation_uses_expand_strategy() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (a+b)^2 - a^2 - 2*a*b, b^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand");
    let steps = wire["steps"].as_array().expect("steps array");
    assert!(!steps.is_empty());
    assert_rule_eq(&steps[0]["rule"], "Expandir binomio");
}
#[test]
fn derive_expand_difference_square_delta_json_steps_does_not_panic() {
    let (output, code) = run_cli(&[
        "eval",
        "derive (a+b)^2 - (a-b)^2, 4*a*b",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    assert_eq!(
        code, 0,
        "expected successful CLI exit, got {code} with output: {output}"
    );

    let wire = parse_wire(&output);
    assert_eq!(wire["strategy"], "expand");
    assert_eq!(wire["result_latex"], "4\\cdot a\\cdot b");
    let steps = wire["steps"].as_array().expect("steps array");
    assert!(
        !steps.is_empty(),
        "derive JSON should retain visible steps for expanded delta target"
    );
}
#[test]
fn derive_odd_half_power_with_passthrough_uses_named_expand_odd_half_power_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sqrt(x^3)+a, abs(x)*sqrt(x)+a",
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
    assert_eq!(substeps[0]["before_latex"], json!("{x}^{3}"));
    assert_eq!(substeps[0]["after_latex"], json!("{x}^{2}\\cdot x"));
    assert_eq!(wire["required_display"], json!(["x ≥ 0"]));
}
#[test]
fn derive_higher_odd_half_power_with_passthrough_uses_named_expand_odd_half_power_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sqrt(x^7)+a, x^3*sqrt(x)+a",
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
    assert_eq!(wire["required_display"], json!(["x ≥ 0"]));
}
#[test]
fn derive_exponential_sum_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x+y), exp(x)*exp(y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Exponential Sum/Difference Identity");
}
#[test]
fn derive_exponential_sum_expansion_with_passthrough_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x+y)+a, exp(x)*exp(y)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Exponential Sum/Difference Identity");
}
#[test]
fn derive_exponential_difference_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x-y), exp(x)/exp(y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Exponential Sum/Difference Identity");
}
#[test]
fn derive_exponential_product_contraction_with_passthrough_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x)*exp(y)+a, exp(x+y)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Exponential Sum/Difference Identity");
}
#[test]
fn derive_exponential_product_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x)*exp(y), exp(x+y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Exponential Sum/Difference Identity");
}
#[test]
fn derive_exponential_product_quotient_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x)*exp(y)/exp(z), exp(x+y-z)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Exponential Sum/Difference Identity");
}
#[test]
fn derive_exponential_quotient_with_power_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x)/exp(y)^2, exp(x-2*y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Exponential Sum/Difference Identity");
}
#[test]
fn derive_exponential_power_quotient_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x)^2/exp(y)^3, exp(2*x-3*y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Exponential Sum/Difference Identity");
}
#[test]
fn derive_exponential_reciprocal_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1/exp(x), exp(-x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Exponential Reciprocal Identity");
}
#[test]
fn derive_exponential_reciprocal_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(-x), 1/exp(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Exponential Reciprocal Identity");
}
#[test]
fn derive_negative_exponential_reciprocal_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -1/exp(x), -exp(-x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Exponential Reciprocal Identity");
}
#[test]
fn derive_exponential_power_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x)^3, exp(3*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Exponential Power Identity");
}
#[test]
fn derive_exponential_reciprocal_power_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1/exp(x)^2, exp(-2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Exponential Reciprocal Identity");
}
#[test]
fn derive_grouped_even_log_product_expansion_uses_direct_expand_log() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive ln((x*y)^2), ln(x^2)+ln(y^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Expandir logaritmos");
}
#[test]
fn derive_grouped_abs_log_product_expansion_uses_direct_expand_log() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*ln(abs(x*y)), 2*ln(abs(x))+2*ln(abs(y))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Expandir logaritmos");
}
#[test]
fn derive_grouped_general_base_log_product_expansion_uses_direct_expand_log() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive log(b,(x*y)^2), 2*log(b,x)+2*log(b,y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Expandir logaritmos");
}
#[test]
fn derive_grouped_even_log_product_expansion_with_passthrough_uses_direct_expand_log() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive ln((x*y)^2)+a, ln(x^2)+ln(y^2)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Expandir logaritmos");
}
#[test]
fn derive_grouped_abs_log_product_expansion_with_passthrough_uses_direct_expand_log() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*ln(abs(x*y))+a, 2*ln(abs(x))+2*ln(abs(y))+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Expandir logaritmos");
}
#[test]
fn derive_grouped_general_base_log_product_expansion_with_passthrough_uses_direct_expand_log() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive log(b,(x*y)^2)+a, 2*log(b,x)+2*log(b,y)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Expandir logaritmos");
}
#[test]
fn derive_factored_log_difference_squares_with_passthrough_uses_two_named_steps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive log(x^2-y^2)+a, log(x-y)+log(x+y)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 2);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert_rule_eq(&steps[0]["rule"], "Factorizar");
    assert_rule_eq(&steps[1]["rule"], "Expandir logaritmos");
}
#[test]
fn derive_factored_log_difference_squares_uses_two_named_steps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive log(x^2-y^2), log(x-y)+log(x+y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 2);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert_rule_eq(&steps[0]["rule"], "Factorizar");
    assert_rule_eq(&steps[1]["rule"], "Expandir logaritmos");
    assert_eq!(steps[0]["before"], "ln(x^2 - y^2)");
    assert_eq!(steps[0]["after"], "ln((x + y) · (x - y))");
    let before_latex = steps[0]["before_latex"].as_str().expect("before_latex");
    let after_latex = steps[0]["after_latex"].as_str().expect("after_latex");
    assert!(
        before_latex.contains("\\ln(") && before_latex.contains("{x}^{2} - {y}^{2}"),
        "expected full logarithm in first step before_latex, got: {before_latex}"
    );
    assert!(
        after_latex.contains("\\ln(") && after_latex.contains("(x + y)\\cdot (x - y)"),
        "expected full logarithm in first step after_latex, got: {after_latex}"
    );
    let required = wire["required_display"]
        .as_array()
        .expect("required_display array");
    assert_eq!(required.len(), 2);
    assert!(required.iter().any(|item| item == "x + y > 0"));
    assert!(required.iter().any(|item| item == "x - y > 0"));
}
#[test]
fn derive_factored_log_quotient_difference_squares_uses_two_named_steps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive log((x^2-y^2)/(u*v)), log(x-y)+log(x+y)-log(u)-log(v)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand_log");
    assert_eq!(wire["steps_count"], 2);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert_rule_eq(&steps[0]["rule"], "Factorizar");
    assert_rule_eq(&steps[1]["rule"], "Expandir logaritmos");
}
#[test]
fn derive_grouped_even_log_product_uses_direct_log_contraction() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive ln(x^2)+ln(y^2), ln((x*y)^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract logs");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Contraer logaritmos");
}
#[test]
fn derive_grouped_even_log_product_with_passthrough_uses_direct_log_contraction() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive ln(x^2)+ln(y^2)+a, ln((x*y)^2)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract logs");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Contraer logaritmos");
}
#[test]
fn derive_scaled_abs_log_product_uses_direct_log_contraction() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*ln(abs(x))+2*ln(abs(y)), 2*ln(abs(x*y))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract logs");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Contraer logaritmos");
}
#[test]
fn derive_scaled_abs_log_product_with_passthrough_uses_direct_log_contraction() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*ln(abs(x))+2*ln(abs(y))+a, 2*ln(abs(x*y))+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract logs");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Contraer logaritmos");
}
#[test]
fn derive_grouped_general_base_log_product_uses_direct_log_contraction() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*log(b,x)+2*log(b,y), log(b,(x*y)^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract logs");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Contraer logaritmos");
}
#[test]
fn derive_grouped_general_base_log_product_with_passthrough_uses_direct_log_contraction() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*log(b,x)+2*log(b,y)+a, log(b,(x*y)^2)+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract logs");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Contraer logaritmos");
}
#[test]
fn derive_log_higher_even_power_drops_redundant_nonzero_requires() {
    let (output, _code) = run_cli(&["eval", "derive ln(x^4), 4*ln(abs(x))", "--format", "json"]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert_eq!(required.len(), 1);
    assert_eq!(required[0], "x ≠ 0");
}
#[test]
fn derive_log_multifactor_expansion_drops_redundant_composite_positive_require() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive ln((x^2*y)/(z*t)), 2*ln(abs(x)) + ln(y) - ln(z) - ln(t)",
        "--format",
        "json",
    ]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert_eq!(required.len(), 4);
    assert!(required.iter().all(|item| item != "y·x^2 / (t·z) > 0"));
}
#[test]
fn derive_general_base_log_expansion_surfaces_positive_factor_requires() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive log(b, (x^2*y^3)/(z^2*t)), 2*log(b, x) + 3*log(b, y) - 2*log(b, z) - log(b, t)",
        "--format",
        "json",
    ]);
    let wire = parse_wire(&output);

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert_eq!(required.len(), 6);
    assert!(required.iter().any(|item| item == "b ≠ 1"));
    assert!(required.iter().any(|item| item == "b > 0"));
    assert!(required.iter().any(|item| item == "x > 0"));
    assert!(required.iter().any(|item| item == "y > 0"));
    assert!(required.iter().any(|item| item == "z > 0"));
    assert!(required.iter().any(|item| item == "t > 0"));
}
#[test]
fn derive_exponential_power_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(3*x), exp(x)^3",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite exponentials");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Exponential Power Identity");
}
#[test]
fn derive_exponential_sum_difference_drops_redundant_nonzero_requires() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive exp(x+y-z), exp(x)*exp(y)/exp(z)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["result"], "e^x·e^y / e^z");
    assert_eq!(
        wire["required_display"].as_array().map(Vec::len),
        Some(0),
        "derive exponential expansion should not require e^z ≠ 0"
    );
}
