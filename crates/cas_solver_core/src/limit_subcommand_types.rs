/// Limit direction for subcommand-level evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitCommandApproach {
    Infinity,
    NegInfinity,
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
    Json(String),
    Text {
        result: String,
        warning: Option<String>,
    },
}
