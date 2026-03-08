use crate::expand_policy::ExpandPolicy;

/// View-only budget values used for displaying auto-expand settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoexpandBudgetView {
    pub max_pow_exp: u32,
    pub max_base_terms: u32,
    pub max_generated_terms: u32,
    pub max_vars: u32,
}

/// Runtime state needed to evaluate an `autoexpand` command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoexpandCommandState {
    pub policy: ExpandPolicy,
    pub budget: AutoexpandBudgetView,
}

/// Parsed input for the `autoexpand` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutoexpandCommandInput {
    ShowCurrent,
    SetPolicy(ExpandPolicy),
    UnknownMode(String),
}

/// Normalized result for `autoexpand` command handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutoexpandCommandResult {
    ShowCurrent {
        message: String,
    },
    SetPolicy {
        policy: ExpandPolicy,
        message: String,
    },
    Invalid {
        message: String,
    },
}

/// Result from evaluating + applying an `autoexpand` command to runtime options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoexpandCommandApplyOutput {
    pub message: String,
    pub rebuild_simplifier: bool,
}
