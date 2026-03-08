use crate::assume_scope::AssumeScope;
use crate::assumption_reporting::AssumptionReporting;
use crate::branch_policy::BranchPolicy;
use crate::const_fold_types::ConstFoldMode;
use crate::domain_condition::RequiresDisplayLevel;
use crate::domain_mode::DomainMode;
use crate::eval_options::EvalOptions;
use crate::inverse_trig_policy::InverseTrigPolicy;
use crate::simplify_options::SimplifyOptions;
use crate::value_domain::ValueDomain;

/// Snapshot of semantic settings used for user-facing formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticsViewState {
    pub domain_mode: DomainMode,
    pub value_domain: ValueDomain,
    pub branch: BranchPolicy,
    pub inv_trig: InverseTrigPolicy,
    pub const_fold: ConstFoldMode,
    pub assumption_reporting: AssumptionReporting,
    pub assume_scope: AssumeScope,
    pub hints_enabled: bool,
    pub requires_display: RequiresDisplayLevel,
}

/// Build a semantics view snapshot from simplifier + eval options.
pub fn semantics_view_state_from_options(
    simplify_options: &SimplifyOptions,
    eval_options: &EvalOptions,
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
