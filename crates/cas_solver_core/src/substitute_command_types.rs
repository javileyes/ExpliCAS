/// Render policy for substitute command step output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubstituteRenderMode {
    None,
    Succinct,
    Normal,
    Verbose,
}

/// Substitution mode for subcommand-level evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubstituteCommandMode {
    Exact,
    Power,
}

/// CLI-friendly output contract for `substitute` subcommand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubstituteSubcommandOutput {
    Json(String),
    TextLines(Vec<String>),
}

/// Parse/eval errors for `subst <expr>, <target>, <replacement>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubstituteParseError {
    InvalidArity,
    Expression(String),
    Target(String),
    Replacement(String),
}
