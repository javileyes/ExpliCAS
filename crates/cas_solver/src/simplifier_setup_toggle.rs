use crate::{Simplifier, SimplifierToggleConfig};

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
