/// Immutable semantics preset definition used by frontends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticsPreset {
    pub name: &'static str,
    pub description: &'static str,
    pub domain: cas_solver::DomainMode,
    pub value: cas_solver::ValueDomain,
    pub branch: cas_solver::BranchPolicy,
    pub inv_trig: cas_solver::InverseTrigPolicy,
    pub const_fold: cas_solver::ConstFoldMode,
}

/// Runtime semantics state affected by preset application.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticsPresetState {
    pub domain: cas_solver::DomainMode,
    pub value: cas_solver::ValueDomain,
    pub branch: cas_solver::BranchPolicy,
    pub inv_trig: cas_solver::InverseTrigPolicy,
    pub const_fold: cas_solver::ConstFoldMode,
}

/// Result of applying a semantics preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticsPresetApplication {
    pub preset: SemanticsPreset,
    pub next: SemanticsPresetState,
}

/// End-to-end evaluation result for `semantics preset` command args.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticsPresetCommandOutput {
    pub lines: Vec<String>,
    pub applied: bool,
}

/// Error returned when preset application cannot proceed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticsPresetApplyError {
    UnknownPreset { name: String },
}
