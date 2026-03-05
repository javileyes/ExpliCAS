/// Snapshot of semantic settings used for user-facing formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticsViewState {
    pub domain_mode: crate::DomainMode,
    pub value_domain: crate::ValueDomain,
    pub branch: crate::BranchPolicy,
    pub inv_trig: crate::InverseTrigPolicy,
    pub const_fold: crate::ConstFoldMode,
    pub assumption_reporting: crate::AssumptionReporting,
    pub assume_scope: crate::AssumeScope,
    pub hints_enabled: bool,
    pub requires_display: crate::RequiresDisplayLevel,
}

/// Build a semantics view snapshot from simplifier + eval options.
pub fn semantics_view_state_from_options(
    simplify_options: &crate::SimplifyOptions,
    eval_options: &crate::EvalOptions,
) -> SemanticsViewState {
    SemanticsViewState {
        domain_mode: simplify_options.shared.semantics.domain_mode,
        value_domain: simplify_options.shared.semantics.value_domain,
        branch: simplify_options.shared.semantics.branch,
        inv_trig: simplify_options.shared.semantics.inv_trig,
        const_fold: eval_options.const_fold,
        assumption_reporting: eval_options.shared.assumption_reporting,
        assume_scope: simplify_options.shared.semantics.assume_scope,
        hints_enabled: eval_options.hints_enabled,
        requires_display: eval_options.requires_display,
    }
}
