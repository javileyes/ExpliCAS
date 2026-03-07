pub fn format_simplifier_toggle_config(config: crate::SimplifierToggleConfig) -> String {
    format!(
        "Current Configuration:\n\
           distribute: {}\n\
           expand_binomials: {}\n\
           distribute_constants: {}\n\
           factor_difference_squares: {}\n\
           root_denesting: {}\n\
           trig_double_angle: {}\n\
           trig_angle_sum: {}\n\
           log_split_exponents: {}\n\
           rationalize_denominator: {}\n\
           canonicalize_trig_square: {}\n\
           auto_factor: {}",
        config.distribute,
        config.expand_binomials,
        config.distribute_constants,
        config.factor_difference_squares,
        config.root_denesting,
        config.trig_double_angle,
        config.trig_angle_sum,
        config.log_split_exponents,
        config.rationalize_denominator,
        config.canonicalize_trig_square,
        config.auto_factor
    )
}
