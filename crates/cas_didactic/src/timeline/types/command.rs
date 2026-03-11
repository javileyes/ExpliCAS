use crate::runtime::{DisplayEvalSteps, DisplaySolveSteps};
use cas_ast::{Equation, ExprId, SolutionSet};

/// Simplify branch payload for CLI `timeline` rendering.
#[derive(Debug, Clone)]
pub struct TimelineSimplifyCommandOutput {
    pub expr_input: String,
    pub use_aggressive: bool,
    pub parsed_expr: ExprId,
    pub simplified_expr: ExprId,
    pub steps: DisplayEvalSteps,
}

/// Solve branch payload for CLI `timeline` rendering.
#[derive(Debug, Clone)]
pub struct TimelineSolveCommandOutput {
    pub equation: Equation,
    pub var: String,
    pub solution_set: SolutionSet,
    pub display_steps: DisplaySolveSteps,
}

/// End-to-end output for CLI `timeline` rendering.
#[derive(Debug)]
pub enum TimelineCommandOutput {
    Solve(TimelineSolveCommandOutput),
    Simplify(TimelineSimplifyCommandOutput),
}
