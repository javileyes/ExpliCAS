#[derive(Debug, Clone)]
pub struct TimelineSolveEvalOutput {
    pub equation: cas_ast::Equation,
    pub var: String,
    pub solution_set: cas_ast::SolutionSet,
    pub display_steps: crate::DisplaySolveSteps,
    pub diagnostics: crate::SolveDiagnostics,
}

#[derive(Debug, Clone)]
pub struct TimelineSimplifyEvalOutput {
    pub parsed_expr: cas_ast::ExprId,
    pub simplified_expr: cas_ast::ExprId,
    pub steps: crate::DisplayEvalSteps,
}
pub use cas_solver_core::solve_command_types::{
    TimelineCommandEvalError, TimelineSimplifyEvalError, TimelineSolveEvalError,
};

#[derive(Debug, Clone)]
pub enum TimelineCommandEvalOutput {
    Solve(TimelineSolveEvalOutput),
    Simplify {
        expr_input: String,
        aggressive: bool,
        output: TimelineSimplifyEvalOutput,
    },
}
