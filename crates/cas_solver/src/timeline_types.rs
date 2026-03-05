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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineSolveEvalError {
    Prepare(crate::SolvePrepareError),
    Solve(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineSimplifyEvalError {
    Parse(String),
    Eval(String),
}

#[derive(Debug, Clone)]
pub enum TimelineCommandEvalOutput {
    Solve(TimelineSolveEvalOutput),
    Simplify {
        expr_input: String,
        aggressive: bool,
        output: TimelineSimplifyEvalOutput,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineCommandEvalError {
    Solve(TimelineSolveEvalError),
    Simplify(TimelineSimplifyEvalError),
}
