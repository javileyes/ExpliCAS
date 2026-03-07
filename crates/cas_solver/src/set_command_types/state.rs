use super::input::SetDisplayMode;

/// Snapshot of current REPL `set` state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetCommandState {
    pub transform: bool,
    pub rationalize: crate::AutoRationalizeLevel,
    pub heuristic_poly: crate::HeuristicPoly,
    pub autoexpand_binomials: crate::AutoExpandBinomials,
    pub steps_mode: crate::StepsMode,
    pub display_mode: SetDisplayMode,
    pub max_rewrites: usize,
    pub debug_mode: bool,
}
