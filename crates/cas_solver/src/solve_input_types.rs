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
pub enum TimelineCommandInput {
    Solve(String),
    Simplify { expr: String, aggressive: bool },
}
