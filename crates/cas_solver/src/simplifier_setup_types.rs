mod defaults;
mod update;

/// Feature switches for building a configured solver simplifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimplifierRuleConfig {
    pub distribute: bool,
    pub expand_binomials: bool,
    pub factor_difference_squares: bool,
    pub root_denesting: bool,
    pub trig_double_angle: bool,
    pub trig_angle_sum: bool,
    pub log_split_exponents: bool,
    pub rationalize_denominator: bool,
    pub canonicalize_trig_square: bool,
    pub auto_factor: bool,
}

/// Runtime feature switches used to enable/disable simplifier rules.
///
/// Unlike [`SimplifierRuleConfig`], this is applied to an already-built
/// simplifier and supports dynamic CLI toggles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimplifierToggleConfig {
    pub distribute: bool,
    pub expand_binomials: bool,
    pub distribute_constants: bool,
    pub factor_difference_squares: bool,
    pub root_denesting: bool,
    pub trig_double_angle: bool,
    pub trig_angle_sum: bool,
    pub log_split_exponents: bool,
    pub rationalize_denominator: bool,
    pub canonicalize_trig_square: bool,
    pub auto_factor: bool,
}

/// Update one named toggle rule.
pub fn set_simplifier_toggle_rule(
    config: &mut SimplifierToggleConfig,
    rule: &str,
    enabled: bool,
) -> Result<(), String> {
    update::set_simplifier_toggle_rule(config, rule, enabled)
}
