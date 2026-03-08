/// Parsed input for the `health` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthCommandInput {
    ShowLast,
    SetEnabled { enabled: bool },
    Clear,
    Status(HealthStatusInput),
    Invalid,
}

/// Parsed options for `health status`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthStatusInput {
    pub list_only: bool,
    pub category: Option<String>,
    pub category_missing_arg: bool,
}

/// Evaluated output for a `health ...` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthCommandEvalOutput {
    pub lines: Vec<String>,
    pub set_enabled: Option<bool>,
    pub clear_last_report: bool,
}
