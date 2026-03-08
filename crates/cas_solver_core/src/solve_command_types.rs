#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolveCommandInput {
    pub equation: String,
    pub variable: Option<String>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineCommandEvalError {
    Solve(TimelineSolveEvalError),
    Simplify(TimelineSimplifyEvalError),
}
