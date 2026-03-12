use crate::Simplifier;
use cas_solver_core::simplifier_config::SimplifierToggleConfig;

/// Update one named toggle rule.
pub fn set_simplifier_toggle_rule(
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

/// Apply runtime rule toggles to an existing simplifier instance.
pub fn apply_simplifier_toggle_config(simplifier: &mut Simplifier, config: SimplifierToggleConfig) {
    let mut toggle = |name: &str, enabled: bool| {
        if enabled {
            simplifier.enable_rule(name);
        } else {
            simplifier.disable_rule(name);
        }
    };

    toggle("Distributive Property", config.distribute);
    toggle("Binomial Expansion", config.expand_binomials);
    toggle("Distribute Constant", config.distribute_constants);
    toggle(
        "Factor Difference of Squares",
        config.factor_difference_squares,
    );
    toggle("Root Denesting", config.root_denesting);
    toggle("Double Angle Identity", config.trig_double_angle);
    toggle("Angle Sum/Diff Identity", config.trig_angle_sum);
    toggle("Split Log Exponents", config.log_split_exponents);
    toggle("Rationalize Denominator", config.rationalize_denominator);
    toggle("Canonicalize Trig Square", config.canonicalize_trig_square);

    // If auto_factor is on, keep conservative expansion and disable aggressive expansion
    // to avoid rewrite ping-pong.
    if config.auto_factor {
        simplifier.enable_rule("Automatic Factorization");
        simplifier.enable_rule("Conservative Expand");
        simplifier.disable_rule("Expand Polynomial");
        simplifier.disable_rule("Binomial Expansion");
    } else {
        simplifier.disable_rule("Automatic Factorization");
        simplifier.disable_rule("Conservative Expand");
        simplifier.enable_rule("Expand Polynomial");
        if config.expand_binomials {
            simplifier.enable_rule("Binomial Expansion");
        }
    }
}
