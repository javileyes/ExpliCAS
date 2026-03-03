use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CasConfig {
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

impl Default for CasConfig {
    fn default() -> Self {
        Self {
            distribute: false,                // Default: Conservative
            expand_binomials: true,           // Needed for many simplifications
            distribute_constants: true,       // Safe distribution of -1, etc.
            factor_difference_squares: false, // Can cause loops if not careful
            root_denesting: true,             // Advanced simplification for nested roots
            trig_double_angle: true,          // sin(2x) -> 2sin(x)cos(x)
            trig_angle_sum: true,             // sin(a+b) -> sin(a)cos(b)...
            log_split_exponents: true,        // ln(x^a) -> a*ln(x)
            rationalize_denominator: true,    // 1/sqrt(2) -> sqrt(2)/2
            canonicalize_trig_square: false,  // cos^2 -> 1-sin^2 (Can prevent simplification)
            auto_factor: false, // Automatically factor polynomials if simpler (disabled by default to avoid loops)
        }
    }
}

impl CasConfig {
    pub fn load() -> Self {
        let path = Path::new("cas_config.toml");
        if path.exists() {
            match fs::read_to_string(path) {
                Ok(content) => match toml::from_str(&content) {
                    Ok(config) => return config,
                    Err(e) => println!("Error parsing config file: {}. Using defaults.", e),
                },
                Err(e) => println!("Error reading config file: {}. Using defaults.", e),
            }
        }
        Self::default()
    }

    pub fn save(&self) -> std::io::Result<()> {
        let content = toml::to_string_pretty(self).map_err(std::io::Error::other)?;
        let mut file = fs::File::create("cas_config.toml")?;
        file.write_all(content.as_bytes())?;
        Ok(())
    }

    pub fn restore() -> Self {
        let config = Self::default();
        let _ = config.save(); // Overwrite file with defaults
        config
    }
}

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
