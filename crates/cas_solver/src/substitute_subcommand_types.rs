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
