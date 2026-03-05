/// Result of applying a `budget` command against session options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveBudgetCommandResult {
    Current { max_branches: usize },
    Updated { max_branches: usize },
    Invalid { raw_value: String },
}
