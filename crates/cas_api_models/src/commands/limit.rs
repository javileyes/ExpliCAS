/// Limit direction for subcommand-level evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitCommandApproach {
    Infinity,
    NegInfinity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitCommandEvalError {
    EmptyInput,
    Parse(String),
    Limit(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LimitCommandEvalOutput {
    pub var: String,
    pub approach: LimitCommandApproach,
    pub result: String,
    pub warning: Option<String>,
}

/// Pre-simplification policy for subcommand-level limit evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitCommandPreSimplify {
    Off,
    Safe,
}

/// CLI-friendly output contract for `limit` subcommand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitSubcommandOutput {
    Wire(String),
    Text {
        result: String,
        warning: Option<String>,
    },
}

/// Output payload for low-level `limit` subcommand execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitSubcommandEvalOutput {
    Wire(String),
    Text {
        result: String,
        warning: Option<String>,
    },
}

/// Error payload for low-level `limit` subcommand execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitSubcommandEvalError {
    Parse(String),
    Limit(String),
}
