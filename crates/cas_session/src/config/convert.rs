use super::CasConfig;

/// Convert persisted CLI config into simplifier build-time rule config.
pub fn solver_rule_config_from_cas_config(config: &CasConfig) -> crate::SimplifierRuleConfig {
    crate::SimplifierRuleConfig {
        distribute: config.distribute,
        expand_binomials: config.expand_binomials,
        factor_difference_squares: config.factor_difference_squares,
        root_denesting: config.root_denesting,
        trig_double_angle: config.trig_double_angle,
        trig_angle_sum: config.trig_angle_sum,
        log_split_exponents: config.log_split_exponents,
        rationalize_denominator: config.rationalize_denominator,
        canonicalize_trig_square: config.canonicalize_trig_square,
        auto_factor: config.auto_factor,
    }
}

/// Convert persisted CLI config into runtime simplifier toggles.
pub fn solver_toggle_config_from_cas_config(config: &CasConfig) -> crate::SimplifierToggleConfig {
    crate::SimplifierToggleConfig {
        distribute: config.distribute,
        expand_binomials: config.expand_binomials,
        distribute_constants: config.distribute_constants,
        factor_difference_squares: config.factor_difference_squares,
        root_denesting: config.root_denesting,
        trig_double_angle: config.trig_double_angle,
        trig_angle_sum: config.trig_angle_sum,
        log_split_exponents: config.log_split_exponents,
        rationalize_denominator: config.rationalize_denominator,
        canonicalize_trig_square: config.canonicalize_trig_square,
        auto_factor: config.auto_factor,
    }
}

/// Apply runtime simplifier toggles back into persisted CLI config.
pub fn apply_solver_toggle_to_cas_config(
    config: &mut CasConfig,
    toggles: crate::SimplifierToggleConfig,
) {
    config.distribute = toggles.distribute;
    config.expand_binomials = toggles.expand_binomials;
    config.distribute_constants = toggles.distribute_constants;
    config.factor_difference_squares = toggles.factor_difference_squares;
    config.root_denesting = toggles.root_denesting;
    config.trig_double_angle = toggles.trig_double_angle;
    config.trig_angle_sum = toggles.trig_angle_sum;
    config.log_split_exponents = toggles.log_split_exponents;
    config.rationalize_denominator = toggles.rationalize_denominator;
    config.canonicalize_trig_square = toggles.canonicalize_trig_square;
    config.auto_factor = toggles.auto_factor;
}

/// Sync an existing simplifier with toggle values from `CasConfig`.
pub fn sync_simplifier_with_cas_config(
    simplifier: &mut cas_engine::Simplifier,
    config: &CasConfig,
) {
    let toggles = solver_toggle_config_from_cas_config(config);
    cas_solver::apply_simplifier_toggle_config(simplifier, toggles);
}
