use crate::branch_policy::BranchPolicy;
use crate::const_fold_types::ConstFoldMode;
use crate::domain_mode::DomainMode;
use crate::inverse_trig_policy::InverseTrigPolicy;
use crate::value_domain::ValueDomain;

/// Immutable semantics preset definition used by frontends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticsPreset {
    pub name: &'static str,
    pub description: &'static str,
    pub domain: DomainMode,
    pub value: ValueDomain,
    pub branch: BranchPolicy,
    pub inv_trig: InverseTrigPolicy,
    pub const_fold: ConstFoldMode,
}

/// Runtime semantics state affected by preset application.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticsPresetState {
    pub domain: DomainMode,
    pub value: ValueDomain,
    pub branch: BranchPolicy,
    pub inv_trig: InverseTrigPolicy,
    pub const_fold: ConstFoldMode,
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
