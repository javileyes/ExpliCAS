use serde::{Deserialize, Serialize};

mod convert;
mod defaults;
mod io;
mod runtime;

pub use convert::{
    apply_solver_toggle_to_cas_config, solver_rule_config_from_cas_config,
    solver_toggle_config_from_cas_config, sync_simplifier_with_cas_config,
};

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
