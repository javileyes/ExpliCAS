use super::input::SetDisplayMode;
use crate::eval_option_axes::{AutoExpandBinomials, HeuristicPoly, StepsMode};
use crate::rationalize_policy::AutoRationalizeLevel;

/// Snapshot of current REPL `set` state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetCommandState {
    pub transform: bool,
    pub rationalize: AutoRationalizeLevel,
    pub heuristic_poly: HeuristicPoly,
    pub autoexpand_binomials: AutoExpandBinomials,
    pub steps_mode: StepsMode,
    pub display_mode: SetDisplayMode,
    pub max_rewrites: usize,
    pub debug_mode: bool,
}
