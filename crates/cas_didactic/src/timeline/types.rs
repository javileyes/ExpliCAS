use cas_ast::{Equation, ExprId, SolutionSet};
use cas_solver::{DisplayEvalSteps, DisplaySolveSteps};

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

/// CLI-facing timeline render artifact.
#[derive(Debug, Clone)]
pub enum TimelineCliRender {
    /// No timeline file should be emitted; return textual lines only.
    NoSteps { lines: Vec<String> },
    /// Timeline file + informational lines.
    Html {
        file_name: &'static str,
        html: String,
        lines: Vec<String>,
    },
}

/// Normalized CLI actions derived from [`TimelineCliRender`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineCliAction {
    Output(String),
    WriteFile { path: String, contents: String },
    OpenFile { path: String },
}
