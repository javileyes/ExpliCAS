/// Neutral snapshot of simplifier rule toggles for config command flows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimplifierToggleState {
    pub distribute: bool,
    pub expand_binomials: bool,
    pub distribute_constants: bool,
    pub factor_difference_squares: bool,
    pub root_denesting: bool,
    pub trig_double_angle: bool,
    pub trig_angle_sum: bool,
    pub log_split_exponents: bool,
    pub rationalize_denominator: bool,
    pub canonicalize_trig_square: bool,
    pub auto_factor: bool,
}

/// Parsed input for `config ...` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigCommandInput {
    List,
    Save,
    Restore,
    SetRule { rule: String, enable: bool },
    MissingRuleArg { action: String },
    InvalidUsage,
    UnknownSubcommand { subcommand: String },
}

/// Evaluated result for `config ...` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigCommandResult {
    ShowList {
        message: String,
    },
    SaveRequested,
    RestoreRequested,
    ApplyToggleConfig {
        toggles: SimplifierToggleState,
        message: String,
    },
    Error {
        message: String,
    },
}
