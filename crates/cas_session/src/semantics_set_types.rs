/// Mutable semantics state for evaluating `semantics set` commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticsSetState {
    pub domain_mode: cas_solver::DomainMode,
    pub value_domain: cas_solver::ValueDomain,
    pub branch: cas_solver::BranchPolicy,
    pub inv_trig: cas_solver::InverseTrigPolicy,
    pub const_fold: cas_solver::ConstFoldMode,
    pub assumption_reporting: cas_solver::AssumptionReporting,
    pub assume_scope: cas_solver::AssumeScope,
    pub hints_enabled: bool,
    pub check_solutions: bool,
    pub requires_display: cas_solver::RequiresDisplayLevel,
}

/// Build a mutable semantics-set snapshot from simplifier + eval options.
pub fn semantics_set_state_from_options(
    simplify_options: &cas_solver::SimplifyOptions,
    eval_options: &cas_solver::EvalOptions,
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
