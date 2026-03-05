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
