//! Canonical eval output model shared by engine/solver facades.

use cas_ast::{Equation, ExprId};

pub type SolveSubStep =
    crate::solve_aliases::SolveSubStep<Equation, crate::step_types::ImportanceLevel>;
pub type SolveStep =
    crate::solve_aliases::SolveStep<Equation, crate::step_types::ImportanceLevel, SolveSubStep>;

#[derive(Clone, Debug)]
pub struct EvalOutput {
    pub stored_id: Option<u64>,
    pub parsed: ExprId,
    pub resolved: ExprId,
    pub result: crate::eval_models::EvalResult,
    /// Domain warnings with deduplication and rule source.
    pub domain_warnings: Vec<crate::domain_warning::DomainWarning>,
    pub steps: crate::display_steps::DisplaySteps<crate::step_model::Step>,
    pub solve_steps: Vec<SolveStep>,
    /// Assumptions made during solver operations (for Assume mode).
    pub solver_assumptions: Vec<crate::assumption_model::AssumptionRecord>,
    /// Scopes for context-aware display transforms.
    pub output_scopes: Vec<cas_formatter::display_transforms::ScopeTag>,
    /// Required conditions for validity (implicit domain constraints from input).
    pub required_conditions: Vec<crate::domain_condition::ImplicitCondition>,
    /// Blocked hints for transformations unavailable under current policy.
    pub blocked_hints: Vec<crate::blocked_hint::BlockedHint>,
    /// Unified diagnostics with origin tracking.
    pub diagnostics: crate::diagnostics_model::Diagnostics,
}
