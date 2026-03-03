//! Session-local command/eval types for solve and timeline orchestration.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolveCommandInput {
    pub equation: String,
    pub variable: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PreparedSolveEvalRequest {
    pub request: cas_solver::EvalRequest,
    pub var: String,
    pub original_equation: Option<cas_ast::Equation>,
}

#[derive(Debug, Clone)]
pub struct SolveCommandEvalOutput {
    pub var: String,
    pub original_equation: Option<cas_ast::Equation>,
    pub output: cas_solver::EvalOutputView,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolvePrepareError {
    ParseError(String),
    ExpectedEquation,
    NoVariable,
    AmbiguousVariables(Vec<String>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveCommandEvalError {
    Prepare(SolvePrepareError),
    Eval(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineCommandInput {
    Solve(String),
    Simplify { expr: String, aggressive: bool },
}

#[derive(Debug, Clone)]
pub struct TimelineSolveEvalOutput {
    pub equation: cas_ast::Equation,
    pub var: String,
    pub solution_set: cas_ast::SolutionSet,
    pub display_steps: cas_solver::DisplaySolveSteps,
    pub diagnostics: cas_solver::SolveDiagnostics,
}

#[derive(Debug, Clone)]
pub struct TimelineSimplifyEvalOutput {
    pub parsed_expr: cas_ast::ExprId,
    pub simplified_expr: cas_ast::ExprId,
    pub steps: cas_solver::DisplayEvalSteps,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineSolveEvalError {
    Prepare(SolvePrepareError),
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
