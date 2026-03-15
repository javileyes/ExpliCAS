use super::plan::SetCommandPlan;

/// Evaluated `set` command result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SetCommandResult {
    ShowHelp { message: String },
    ShowValue { message: String },
    Apply { plan: SetCommandPlan },
    Invalid { message: String },
}
