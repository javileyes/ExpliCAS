/// Parsed top-level `semantics ...` command shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticsCommandInput {
    Show,
    Help,
    Set { args: Vec<String> },
    Axis { axis: String },
    Preset { args: Vec<String> },
    Unknown { subcommand: String },
}

/// Evaluated output for a full `semantics ...` command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticsCommandOutput {
    pub lines: Vec<String>,
    pub sync_simplifier: bool,
}
