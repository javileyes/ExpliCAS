use super::*;

#[test]
fn derive_double_angle_after_arg_simplify_uses_direct_expand_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x+x), 2*sin(x)*cos(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Expandir ángulo doble");
}
#[test]
fn derive_mixed_root_and_symbolic_power_uses_single_combine_powers_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sqrt(x)*x^a, x^(a+1/2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "combine powers");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Sumar exponentes de la misma base");
}
#[test]
fn derive_cos_diff_over_sin_diff_quotient_uses_direct_contract_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (cos(x)-cos(3*x))/(sin(3*x)-sin(x)), tan(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Convertir un cociente trigonométrico en tangente"
    );
}
#[test]
fn derive_representative_phase_shift_contract_cases_use_single_named_step() {
    for expr in [
        "derive sin(x)+cos(x), sqrt(2)*sin(x+pi/4)",
        "derive 5*sin(x+arctan(4/3))+a, 5*cos(x-arctan(3/4))+a",
    ] {
        assert_single_named_derive_step(expr, "contract trig", "Aplicar identidad de desfase");
    }
}
#[test]
fn derive_representative_phase_shift_expand_cases_use_single_named_step() {
    let expr = "derive sqrt(2)*sin(x+pi/4), sin(x)+cos(x)";
    assert_single_named_derive_step(expr, "expand trig", "Aplicar identidad de desfase");
}
#[test]
fn derive_representative_exact_sixth_phase_shift_cases_use_single_named_step() {
    let (expr, strategy) = (
        "derive sqrt(3)*sin(x)+cos(x), 2*sin(x+pi/6)",
        "contract trig",
    );
    assert_single_named_derive_step(expr, strategy, "Aplicar identidad de desfase");
}
#[test]
fn derive_sophie_germain_expansion_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (x^2 - 2*x*y + 2*y^2)*(x^2 + 2*x*y + 2*y^2), x^4 + 4*y^4",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Expandir la expresión");
}
#[test]
fn derive_cosine_product_to_sum_expansion_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*cos(2*x)*cos(x), cos(3*x)+cos(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar producto a suma");
}
#[test]
fn derive_sine_difference_product_to_sum_expansion_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sin(2*x)*sin(x), cos(x)-cos(3*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar producto a suma");
}
#[test]
fn derive_general_sine_sum_to_product_expansion_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)+sin(y), 2*sin((x+y)/2)*cos((x-y)/2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar suma a producto");
}
#[test]
fn derive_general_cosine_sum_to_product_expansion_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(x)+cos(y), 2*cos((x+y)/2)*cos((x-y)/2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar suma a producto");
}
#[test]
fn derive_general_cosine_difference_sum_to_product_expansion_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(x)-cos(y), -2*sin((x+y)/2)*sin((x-y)/2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar suma a producto");
}
#[test]
fn derive_representative_trig_power_reduction_cases_use_single_named_step() {
    for expr in [
        "derive sin(x)^4, (3-4*cos(2*x)+cos(4*x))/8",
        "derive cos(x)^4, (3+4*cos(2*x)+cos(4*x))/8",
        "derive sin(x)^12, (462-792*cos(2*x)+495*cos(4*x)-220*cos(6*x)+66*cos(8*x)-12*cos(10*x)+cos(12*x))/2048",
        "derive cos(x)^12, (462+792*cos(2*x)+495*cos(4*x)+220*cos(6*x)+66*cos(8*x)+12*cos(10*x)+cos(12*x))/2048",
        "derive sin(x)^24, (1352078-2496144*cos(2*x)+1961256*cos(4*x)-1307504*cos(6*x)+735471*cos(8*x)-346104*cos(10*x)+134596*cos(12*x)-42504*cos(14*x)+10626*cos(16*x)-2024*cos(18*x)+276*cos(20*x)-24*cos(22*x)+cos(24*x))/8388608",
        "derive cos(x)^24, (1352078+2496144*cos(2*x)+1961256*cos(4*x)+1307504*cos(6*x)+735471*cos(8*x)+346104*cos(10*x)+134596*cos(12*x)+42504*cos(14*x)+10626*cos(16*x)+2024*cos(18*x)+276*cos(20*x)+24*cos(22*x)+cos(24*x))/8388608",
    ] {
        assert_single_named_derive_step(expr, "expand trig", "Aplicar reducción de potencias");
    }
}
#[test]
fn derive_sine_cosine_square_product_reduction_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)^2*cos(x)^2, (1-cos(4*x))/8",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar reducción de potencias");
}
#[test]
fn derive_sine_plus_cosine_square_identity_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (sin(x)+cos(x))^2, 1+sin(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Aplicar identidad del cuadrado trigonométrico"
    );
}
#[test]
fn derive_sine_minus_cosine_square_identity_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (sin(x)-cos(x))^2, 1-sin(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Aplicar identidad del cuadrado trigonométrico"
    );
}
#[test]
fn derive_trig_expand_steps_count_matches_visible_steps_for_trig_polynomial_target() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sin(2*x)*sin(x), 4*cos(x)-4*cos(x)^3",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(wire["steps_count"].as_u64(), Some(steps.len() as u64));
    assert_eq!(steps.len(), 2);
    assert_rule_eq(&steps[0]["rule"], "Aplicar producto a suma");
    assert_rule_eq(&steps[1]["rule"], "Triple Angle Expansion");
}
#[test]
fn derive_mixed_trig_double_angle_product_uses_named_step_without_depth_warning() {
    let bin_path = env!("CARGO_BIN_EXE_cas_cli");
    let output = Command::new(bin_path)
        .args([
            "eval",
            "derive 4*sin(x)^2*cos(x), 2*sin(2*x)*sin(x)",
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

    assert_eq!(wire["strategy"], "contract trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Expandir ángulo doble");
    assert!(
        !stderr.contains("depth_overflow"),
        "expected target-aware double-angle path to stay quiet on stderr, got: {stderr}"
    );
}
#[test]
fn derive_trig_product_to_sum_mixed_cos_square_polynomial_uses_expand_trig() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*cos(2*x)*sin(x), 4*cos(x)^2*sin(x)-2*sin(x)",
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
    assert_rule_eq(&steps[0]["rule"], "Aplicar producto a suma");
    assert_rule_eq(&steps[1]["rule"], "Triple Angle Expansion");
}
#[test]
fn derive_trig_product_to_sum_cosine_difference_polynomial_with_passthrough_uses_expand_trig() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sin(2*x)*sin(x)+a, 4*cos(x)-4*cos(x)^3+a",
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
    assert_rule_eq(&steps[0]["rule"], "Aplicar producto a suma");
    assert_rule_eq(&steps[1]["rule"], "Triple Angle Expansion");
}
#[test]
fn derive_trig_product_to_sum_mixed_cos_square_polynomial_with_passthrough_uses_expand_trig() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*cos(2*x)*sin(x)+a, 4*cos(x)^2*sin(x)-2*sin(x)+a",
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
    assert_rule_eq(&steps[0]["rule"], "Aplicar producto a suma");
    assert_rule_eq(&steps[1]["rule"], "Triple Angle Expansion");
}
#[test]
fn derive_mixed_trig_double_angle_expansion_to_sin_square_polynomial_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sin(2*x)*sin(x), 4*sin(x)^2*cos(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Expandir ángulo doble");
}
#[test]
fn derive_mixed_trig_double_angle_expansion_to_cos_square_polynomial_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sin(2*x)*cos(x), 4*cos(x)^2*sin(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Expandir ángulo doble");
}
#[test]
fn derive_negative_sine_squared_uses_single_pythagorean_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -sin(x)^2, cos(x)^2-1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar identidad pitagórica");
}
#[test]
fn derive_negative_cos_squared_uses_single_pythagorean_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cos(x)^2, sin(x)^2-1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar identidad pitagórica");
}
#[test]
fn derive_fraction_sum_to_sec_squared_keeps_faithful_pythagorean_intermediate() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1/(1 + sin(x)) + 1/(1 - sin(x)), 2*sec(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 4);
    let pythagorean_step = steps.last().expect("pythagorean step");
    assert_rule_eq(&pythagorean_step["rule"], "Aplicar identidad pitagórica");
    assert_eq!(pythagorean_step["before"], "2/(1 - sin(x)^2)");
    assert!(
        pythagorean_step["before_latex"]
            .as_str()
            .expect("before latex")
            .contains("^{2}"),
        "expected squared sine to survive in before_latex: {:?}",
        pythagorean_step["before_latex"]
    );
}
#[test]
fn derive_negative_tan_quotient_uses_single_trig_quotient_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -sin(x)/cos(x), -tan(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Convertir un cociente trigonométrico en tangente"
    );
}
#[test]
fn derive_negative_tan_expansion_uses_single_trig_expansion_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -tan(x), -sin(x)/cos(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(
        &steps[0]["rule"],
        "Expandir tangente como seno entre coseno",
    );
}
#[test]
fn derive_consecutive_factorial_ratio_uses_named_factorial_rewrite_and_keeps_guard() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (n+1)!/n!, n+1",
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

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        required.iter().any(|item| item == "n! ≠ 0"),
        "expected factorial nonzero guard in required_display: {required:?}"
    );
}
#[test]
fn derive_consecutive_factorial_ratio_with_passthrough_uses_named_factorial_rewrite_and_keeps_guard(
) {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (n+1)!/n!+a, n+1+a",
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

    let required = wire["required_display"]
        .as_array()
        .expect("required_display");
    assert!(
        required.iter().any(|item| item == "n! ≠ 0"),
        "expected factorial nonzero guard in required_display: {required:?}"
    );
}
#[test]
fn derive_pythagorean_identity_uses_named_trig_rewrite() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)^2 + cos(x)^2, 1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite trigs");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar la identidad pitagórica");
}
#[test]
fn derive_reciprocal_trig_product_with_passthrough_uses_named_trig_rewrite() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive tan(x)*cot(x)+a, 1+a",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rewrite trigs");
    assert_eq!(wire["steps_count"], 1);
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Reconocer tangente por cotangente como 1");
}
#[test]
fn derive_negative_sec_reciprocal_uses_single_reciprocal_trig_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -1/cos(x), -sec(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(steps[0]["rule"], "Reconocer secante desde un recíproco");
}
#[test]
fn derive_shifted_sec_squared_uses_single_reciprocal_pythagorean_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sec(x)^2-1, tan(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar identidad pitagórica recíproca");
}
#[test]
fn derive_shifted_cot_squared_uses_single_reciprocal_pythagorean_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cot(x)^2, csc(x)^2-1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar identidad pitagórica recíproca");
}
#[test]
fn derive_one_minus_sec_squared_uses_single_reciprocal_pythagorean_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1-sec(x)^2, -tan(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar identidad pitagórica recíproca");
}
#[test]
fn derive_negative_cot_squared_uses_single_reciprocal_pythagorean_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cot(x)^2, 1-csc(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar identidad pitagórica recíproca");
}
#[test]
fn derive_shifted_double_angle_plus_uses_single_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(2*x)+1, 2*cos(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Expandir ángulo doble");
}
#[test]
fn derive_shifted_double_angle_minus_backward_uses_single_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 2*sin(x)^2, 1-cos(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Expandir ángulo doble");
}
#[test]
fn derive_shifted_double_angle_negative_forward_uses_single_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(2*x)-1, -2*sin(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Expandir ángulo doble");
}
#[test]
fn derive_shifted_double_angle_negative_backward_uses_single_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -2*sin(x)^2, cos(2*x)-1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Expandir ángulo doble");
}
#[test]
fn derive_negative_cosine_double_angle_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)^2-cos(x)^2, -cos(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Expandir ángulo doble");
}
#[test]
fn derive_negative_cosine_double_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cos(2*x), sin(x)^2-cos(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Expandir ángulo doble");
}
#[test]
fn derive_negative_sine_double_angle_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -2*sin(x)*cos(x), -sin(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Expandir ángulo doble");
}
#[test]
fn derive_negative_sine_double_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -sin(2*x), -2*sin(x)*cos(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Expandir ángulo doble");
}
#[test]
fn derive_negative_half_angle_square_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(1-cos(2*x))/2, -sin(x)^2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar identidad de ángulo mitad");
}
#[test]
fn derive_negative_half_angle_square_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cos(x)^2, -(1+cos(2*x))/2",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Aplicar identidad de ángulo mitad");
}
#[test]
fn derive_negative_half_angle_tangent_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(1-cos(2*x))/sin(2*x), -tan(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Aplicar identidad de tangente de ángulo mitad"
    );
}
#[test]
fn derive_negative_half_angle_tangent_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -tan(x), -(1-cos(2*x))/sin(2*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0]["rule"],
        "Aplicar identidad de tangente de ángulo mitad"
    );
}
#[test]
fn derive_negative_trig_sine_triple_angle_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(3*sin(x)-4*sin(x)^3), -sin(3*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Triple Angle Expansion");
}
#[test]
fn derive_negative_trig_sine_triple_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -sin(3*x), -(3*sin(x)-4*sin(x)^3)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Triple Angle Expansion");
}
#[test]
fn derive_negative_trig_cosine_triple_angle_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(4*cos(x)^3-3*cos(x)), -cos(3*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Triple Angle Expansion");
}
#[test]
fn derive_negative_trig_cosine_triple_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cos(3*x), -(4*cos(x)^3-3*cos(x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Triple Angle Expansion");
}
#[test]
fn derive_trig_tangent_triple_angle_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (3*tan(x)-tan(x)^3)/(1-3*tan(x)^2), tan(3*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Triple Angle Expansion");
}
#[test]
fn derive_trig_tangent_triple_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive tan(3*x), (3*tan(x)-tan(x)^3)/(1-3*tan(x)^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Triple Angle Expansion");
}
#[test]
fn derive_negative_trig_tangent_triple_angle_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(3*tan(x)-tan(x)^3)/(1-3*tan(x)^2), -tan(3*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Triple Angle Expansion");
}
#[test]
fn derive_negative_trig_tangent_triple_angle_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -tan(3*x), -(3*tan(x)-tan(x)^3)/(1-3*tan(x)^2)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Triple Angle Expansion");
}
#[test]
fn derive_trig_quintuple_sine_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(5*x), 5*sin(x)-20*sin(x)^3+16*sin(x)^5",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Quintuple Angle Identity");
}
#[test]
fn derive_negative_trig_quintuple_sine_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -sin(5*x), -(5*sin(x)-20*sin(x)^3+16*sin(x)^5)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Quintuple Angle Identity");
}
#[test]
fn derive_trig_quintuple_cosine_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(5*x), 16*cos(x)^5-20*cos(x)^3+5*cos(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Quintuple Angle Identity");
}
#[test]
fn derive_negative_trig_quintuple_cosine_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -cos(5*x), -(16*cos(x)^5-20*cos(x)^3+5*cos(x))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Quintuple Angle Identity");
}
#[test]
fn derive_trig_quintuple_sine_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 5*sin(x)-20*sin(x)^3+16*sin(x)^5, sin(5*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Quintuple Angle Identity");
}
#[test]
fn derive_negative_trig_quintuple_sine_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(5*sin(x)-20*sin(x)^3+16*sin(x)^5), -sin(5*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Quintuple Angle Identity");
}
#[test]
fn derive_trig_quintuple_cosine_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 16*cos(x)^5-20*cos(x)^3+5*cos(x), cos(5*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Quintuple Angle Identity");
}
#[test]
fn derive_negative_trig_quintuple_cosine_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(16*cos(x)^5-20*cos(x)^3+5*cos(x)), -cos(5*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Quintuple Angle Identity");
}
#[test]
fn derive_trig_angle_sum_sine_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x+y), sin(x)*cos(y)+cos(x)*sin(y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Angle Sum/Diff Identity");
}
#[test]
fn derive_trig_angle_sum_sine_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)*cos(y)+cos(x)*sin(y), sin(x+y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Angle Sum/Diff Identity");
}
#[test]
fn derive_trig_recursive_six_x_sine_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(6*x), sin(5*x)*cos(x)+cos(5*x)*sin(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Angle Sum/Diff Identity");
}
#[test]
fn derive_trig_recursive_six_x_sine_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(5*x)*cos(x)+cos(5*x)*sin(x), sin(6*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Angle Sum/Diff Identity");
}
#[test]
fn derive_trig_angle_diff_sine_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive sin(x)*cos(y)-cos(x)*sin(y), sin(x-y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Angle Sum/Diff Identity");
}
#[test]
fn derive_negative_trig_angle_diff_sine_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(sin(x)*cos(y)-cos(x)*sin(y)), -sin(x-y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Angle Sum/Diff Identity");
}
#[test]
fn derive_trig_angle_diff_cosine_expansion_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(x-y), cos(x)*cos(y)+sin(x)*sin(y)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Angle Sum/Diff Identity");
}
#[test]
fn derive_trig_recursive_six_x_cosine_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive cos(5*x)*cos(x)-sin(5*x)*sin(x), cos(6*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Angle Sum/Diff Identity");
}
#[test]
fn derive_negative_trig_recursive_six_x_cosine_contraction_uses_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive -(cos(5*x)*cos(x)-sin(5*x)*sin(x)), -cos(6*x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "contract trig");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Angle Sum/Diff Identity");
}
#[test]
fn derive_log_higher_even_power_expansion_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive ln(x^4), 4*ln(abs(x))",
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
    assert_rule_eq(&steps[0]["rule"], "Sacar un exponente fuera del logaritmo");
}
#[test]
fn derive_log_general_base_power_expansion_uses_single_named_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive log(b, x^3), 3*log(b, x)",
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
    assert_rule_eq(&steps[0]["rule"], "Sacar un exponente fuera del logaritmo");
}
#[test]
fn derive_log_expansion_to_zero_keeps_single_step_but_closes_on_final_result() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive ln(x^3) + ln(y^2) - ln(x^3*y^2), 0",
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
    assert_eq!(steps[0]["after"], "0");
}
#[test]
fn derive_radical_notable_quotient_uses_single_rationalize_step() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive (x^(3/2)-1)/(sqrt(x)-1), sqrt(x)+x+1",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "rationalize");
    assert_eq!(wire["steps_count"], 1);

    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Reconocer un cociente notable");
}
#[test]
fn derive_consecutive_telescoping_fraction_split_uses_concrete_nontrivial_substeps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1/(u*(u+1)), 1/u - 1/(u+1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "expand fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Descomponer en fracciones telescópicas");
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 2);
    assert_eq!(substeps[0]["title"], "Introducir el numerador telescópico");
    assert_eq!(substeps[1]["title"], "Separar sobre el denominador común");
    assert!(
        !output.contains("Aquí u = u"),
        "literal telescoping split should not emit the tautological 'Aquí u = u' substep"
    );
}
#[test]
fn derive_consecutive_telescoping_fraction_combine_uses_concrete_nontrivial_substeps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1/u - 1/(u+1), 1/(u*(u+1))",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "combine fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Recomponer fracción telescópica");
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 2);
    assert_eq!(
        substeps[0]["title"],
        "Llevar las fracciones al denominador común"
    );
    assert_eq!(substeps[1]["title"], "Simplificar el numerador telescópico");
    assert!(
        !output.contains("Aquí u = u"),
        "literal telescoping combine should not emit the tautological 'Aquí u = u' substep"
    );
}
#[test]
fn derive_fraction_difference_to_single_fraction_shows_common_denominator_substeps() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive 1/(x - 1) - 1/(x + 1), 2/(x^2 - 1)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "combine fraction");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 1);
    assert_rule_eq(&steps[0]["rule"], "Restar fracciones");
    let substeps = steps[0]["substeps"].as_array().expect("substeps array");
    assert_eq!(substeps.len(), 2);
    assert_eq!(substeps[0]["title"], "Llevar a denominador común");
    assert_eq!(
        substeps[1]["title"],
        "Simplificar el numerador y el denominador"
    );
}
#[test]
fn derive_tan_plus_cot_keeps_single_common_denominator_substep() {
    let (output, _code) = run_cli(&[
        "eval",
        "derive tan(x) + cot(x), sec(x)*csc(x)",
        "--format",
        "json",
        "--steps",
        "on",
    ]);
    let wire = parse_wire(&output);

    assert_eq!(wire["strategy"], "simplify");
    let steps = wire["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 4);
    assert_rule_eq(&steps[2]["rule"], "Sumar fracciones");
    let fraction_substeps = steps[2]["substeps"].as_array().expect("substeps array");
    assert_eq!(fraction_substeps.len(), 1);
    assert_eq!(fraction_substeps[0]["title"], "Llevar a denominador común");

    assert_rule_eq(&steps[3]["rule"], "Aplicar la identidad pitagórica");
    let pythagorean_substeps = steps[3]["substeps"].as_array().expect("substeps array");
    assert_eq!(pythagorean_substeps.len(), 2);
    assert_eq!(pythagorean_substeps[0]["title"], "Usar 1 / cos(u) = sec(u)");
    assert_eq!(pythagorean_substeps[1]["title"], "Usar 1 / sin(u) = csc(u)");
    assert_eq!(
        pythagorean_substeps[0]["before_latex"],
        "\\frac{1}{\\cos(x)}"
    );
    assert_eq!(pythagorean_substeps[0]["after_latex"], "\\sec(x)");
    assert_eq!(
        pythagorean_substeps[1]["before_latex"],
        "\\frac{1}{\\sin(x)}"
    );
    assert_eq!(pythagorean_substeps[1]["after_latex"], "\\csc(x)");
}
