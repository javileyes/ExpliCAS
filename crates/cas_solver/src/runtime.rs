//! Explicit public namespace for engine/runtime-facing solver APIs.

pub use crate::display_eval_steps::DisplayEvalSteps;
pub use crate::engine_bridge::{
    Engine, Orchestrator, ParentContext, Rewrite, Rule, RuleProfiler, SimpleRule, Simplifier,
};
pub use crate::eval_output_adapters::required_conditions_from_eval_output;
pub use crate::solver_entrypoints_eval::to_display_steps;
pub use crate::types::{
    DisplaySolveSteps, SolveCtx, SolveDiagnostics, SolveDomainEnv, SolveStep, SolveSubStep,
    SolverEvalSession, SolverEvalStore, SolverOptions, StatelessEvalSession,
};
pub use cas_ast::ordering::compare_expr;
pub use cas_engine::rules;
pub use cas_solver_core::assume_scope::AssumeScope;
pub use cas_solver_core::branch_policy::BranchPolicy;
pub use cas_solver_core::budget_model::{Budget, BudgetExceeded, Metric, Operation, PassStats};
pub use cas_solver_core::domain_mode::DomainMode;
pub use cas_solver_core::error_model::CasError;
pub use cas_solver_core::eval_config::EvalConfig;
pub use cas_solver_core::eval_models::{EvalAction, EvalRequest, EvalResult};
pub use cas_solver_core::eval_option_axes::{
    AutoExpandBinomials, BranchMode, ComplexMode, ContextMode, HeuristicPoly, StepsMode,
};
pub use cas_solver_core::eval_options::EvalOptions;
pub use cas_solver_core::eval_output_model::EvalOutput;
pub use cas_solver_core::expand_policy::{ExpandBudget, ExpandPolicy};
pub use cas_solver_core::inverse_trig_policy::InverseTrigPolicy;
pub use cas_solver_core::phase_budgets::PhaseBudgets;
pub use cas_solver_core::phase_stats::{PhaseStats, PipelineStats};
pub use cas_solver_core::simplify_options::{SharedSemanticConfig, SimplifyOptions};
pub use cas_solver_core::simplify_phase::{PhaseMask, SimplifyPhase};
pub use cas_solver_core::step_model::Step;
pub use cas_solver_core::step_types::{ImportanceLevel, PathStep, StepCategory};
pub use cas_solver_core::value_domain::ValueDomain;
