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

/// Mutable semantics state for evaluating `semantics set` commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticsSetState {
    pub domain_mode: DomainMode,
    pub value_domain: ValueDomain,
    pub branch: BranchPolicy,
    pub inv_trig: InverseTrigPolicy,
    pub const_fold: ConstFoldMode,
    pub assumption_reporting: AssumptionReporting,
    pub assume_scope: AssumeScope,
    pub hints_enabled: bool,
    pub check_solutions: bool,
    pub requires_display: RequiresDisplayLevel,
}

/// Build a mutable semantics-set snapshot from simplifier + eval options.
pub fn semantics_set_state_from_options(
    simplify_options: &SimplifyOptions,
    eval_options: &EvalOptions,
) -> SemanticsSetState {
    SemanticsSetState {
        domain_mode: simplify_options.shared.semantics.domain_mode,
        value_domain: simplify_options.shared.semantics.value_domain,
        branch: simplify_options.shared.semantics.branch,
        inv_trig: simplify_options.shared.semantics.inv_trig,
        const_fold: eval_options.const_fold,
        assumption_reporting: eval_options.shared.assumption_reporting,
        assume_scope: simplify_options.shared.semantics.assume_scope,
        hints_enabled: eval_options.hints_enabled,
        check_solutions: eval_options.check_solutions,
        requires_display: eval_options.requires_display,
    }
}
