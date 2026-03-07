use super::SimplifierToggleConfig;

pub(super) fn set_simplifier_toggle_rule(
    config: &mut SimplifierToggleConfig,
    rule: &str,
    enabled: bool,
) -> Result<(), String> {
    match rule {
        "distribute" => config.distribute = enabled,
        "expand_binomials" => config.expand_binomials = enabled,
        "distribute_constants" => config.distribute_constants = enabled,
        "factor_difference_squares" => config.factor_difference_squares = enabled,
        "root_denesting" => config.root_denesting = enabled,
        "trig_double_angle" => config.trig_double_angle = enabled,
        "trig_angle_sum" => config.trig_angle_sum = enabled,
        "log_split_exponents" => config.log_split_exponents = enabled,
        "rationalize_denominator" => config.rationalize_denominator = enabled,
        "canonicalize_trig_square" => config.canonicalize_trig_square = enabled,
        "auto_factor" => config.auto_factor = enabled,
        _ => return Err(format!("Unknown rule: {}", rule)),
    }
    Ok(())
}
