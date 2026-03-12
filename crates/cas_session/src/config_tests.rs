#[test]
fn solver_toggle_round_trip_preserves_config_values() {
    let mut config = crate::config::CasConfig {
        distribute: true,
        expand_binomials: false,
        distribute_constants: false,
        factor_difference_squares: true,
        root_denesting: false,
        trig_double_angle: false,
        trig_angle_sum: true,
        log_split_exponents: false,
        rationalize_denominator: false,
        canonicalize_trig_square: true,
        auto_factor: true,
    };
    let toggles = crate::config::solver_toggle_config_from_cas_config(&config);
    config = crate::config::CasConfig::default();
    crate::config::apply_solver_toggle_to_cas_config(&mut config, toggles);
    assert!(config.distribute);
    assert!(!config.expand_binomials);
    assert!(config.auto_factor);
    assert!(config.canonicalize_trig_square);
}
