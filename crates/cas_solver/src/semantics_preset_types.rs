/// Immutable semantics preset definition used by frontends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticsPreset {
    pub name: &'static str,
    pub description: &'static str,
    pub domain: crate::DomainMode,
    pub value: crate::ValueDomain,
    pub branch: crate::BranchPolicy,
    pub inv_trig: crate::InverseTrigPolicy,
    pub const_fold: crate::ConstFoldMode,
}

/// Runtime semantics state affected by preset application.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticsPresetState {
    pub domain: crate::DomainMode,
    pub value: crate::ValueDomain,
    pub branch: crate::BranchPolicy,
    pub inv_trig: crate::InverseTrigPolicy,
    pub const_fold: crate::ConstFoldMode,
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
