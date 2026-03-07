use super::CasConfig;

impl Default for CasConfig {
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
