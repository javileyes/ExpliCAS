use super::input::SetDisplayMode;
use cas_solver_core::eval_options::{AutoExpandBinomials, HeuristicPoly, StepsMode};
use cas_solver_core::rationalize_policy::AutoRationalizeLevel;

/// Normalized mutation plan for applying `set` changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetCommandPlan {
    pub set_transform: Option<bool>,
    pub set_rationalize: Option<AutoRationalizeLevel>,
    pub set_heuristic_poly: Option<HeuristicPoly>,
    pub set_autoexpand_binomials: Option<AutoExpandBinomials>,
    pub set_steps_mode: Option<StepsMode>,
    pub set_display_mode: Option<SetDisplayMode>,
    pub set_max_rewrites: Option<usize>,
    pub set_debug_mode: Option<bool>,
    pub message: String,
}

impl SetCommandPlan {
    pub fn with_message(message: impl Into<String>) -> Self {
        Self {
            set_transform: None,
            set_rationalize: None,
            set_heuristic_poly: None,
            set_autoexpand_binomials: None,
            set_steps_mode: None,
            set_display_mode: None,
            set_max_rewrites: None,
            set_debug_mode: None,
            message: message.into(),
        }
    }
}

/// Side effects produced while applying a `SetCommandPlan`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SetCommandApplyEffects {
    pub set_steps_mode: Option<StepsMode>,
    pub set_display_mode: Option<SetDisplayMode>,
}
