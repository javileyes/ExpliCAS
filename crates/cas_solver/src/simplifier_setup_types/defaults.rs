use super::{SimplifierRuleConfig, SimplifierToggleConfig};

impl Default for SimplifierRuleConfig {
    fn default() -> Self {
        Self {
            distribute: false,
            expand_binomials: true,
            factor_difference_squares: false,
            root_denesting: true,
            trig_double_angle: true,
            trig_angle_sum: true,
            log_split_exponents: true,
            rationalize_denominator: true,
            canonicalize_trig_square: false,
            auto_factor: false,
        }
    }
}

impl Default for SimplifierToggleConfig {
    fn default() -> Self {
        Self {
            distribute: false,
            expand_binomials: true,
            distribute_constants: true,
            factor_difference_squares: false,
            root_denesting: true,
            trig_double_angle: true,
            trig_angle_sum: true,
            log_split_exponents: true,
            rationalize_denominator: true,
            canonicalize_trig_square: false,
            auto_factor: false,
        }
    }
}

impl From<SimplifierRuleConfig> for SimplifierToggleConfig {
    fn from(value: SimplifierRuleConfig) -> Self {
        Self {
            distribute: value.distribute,
            expand_binomials: value.expand_binomials,
            distribute_constants: true,
            factor_difference_squares: value.factor_difference_squares,
            root_denesting: value.root_denesting,
            trig_double_angle: value.trig_double_angle,
            trig_angle_sum: value.trig_angle_sum,
            log_split_exponents: value.log_split_exponents,
            rationalize_denominator: value.rationalize_denominator,
            canonicalize_trig_square: value.canonicalize_trig_square,
            auto_factor: value.auto_factor,
        }
    }
}
