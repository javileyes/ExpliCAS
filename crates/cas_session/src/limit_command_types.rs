#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LimitCommandInput<'a> {
    pub expr: &'a str,
    pub var: &'a str,
    pub approach: cas_solver::Approach,
    pub presimplify: cas_solver::PreSimplifyMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitCommandEvalError {
    EmptyInput,
    Parse(String),
    Limit(String),
}

#[derive(Debug, Clone)]
pub struct LimitCommandEvalOutput {
    pub var: String,
    pub approach: cas_solver::Approach,
    pub result: String,
    pub warning: Option<String>,
}

/// Output payload for CLI-style `limit` subcommand execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitSubcommandEvalOutput {
    Json(String),
    Text {
        result: String,
        warning: Option<String>,
    },
}

/// Error payload for CLI-style `limit` subcommand execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitSubcommandEvalError {
    Parse(String),
    Limit(String),
}
