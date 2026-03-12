//! Explicit public namespace for engine/runtime-facing solver APIs.

pub use crate::budget_runtime_types::{Budget, BudgetExceeded, Metric, Operation, PassStats};
pub use crate::display_eval_steps::DisplayEvalSteps;
pub use crate::engine_runtime_types::{
    AutoExpandBinomials, BranchMode, CasError, ComplexMode, ContextMode, Engine, EvalAction,
    EvalOptions, EvalOutput, EvalRequest, EvalResult, HeuristicPoly, RuleProfiler,
    SharedSemanticConfig, Simplifier, SimplifyOptions, StepsMode,
};
pub use crate::eval_output_adapters::required_conditions_from_eval_output;
pub use crate::phase_runtime_types::{
    ExpandBudget, ExpandPolicy, PhaseBudgets, PhaseMask, PhaseStats, PipelineStats, SimplifyPhase,
};
pub use crate::rule_runtime_types::{Orchestrator, ParentContext, Rewrite, Rule, SimpleRule};
pub use crate::rules_runtime_types::rules;
pub use crate::solver_entrypoints_eval::to_display_steps;
pub use crate::step_runtime_types::{ImportanceLevel, PathStep, Step, StepCategory};
pub use crate::types::{
    DisplaySolveSteps, SolveCtx, SolveDiagnostics, SolveDomainEnv, SolveStep, SolveSubStep,
    SolverEvalSession, SolverEvalStore, SolverOptions, StatelessEvalSession,
};
pub use cas_ast::ordering::compare_expr;
pub use cas_solver_core::assume_scope::AssumeScope;
pub use cas_solver_core::branch_policy::BranchPolicy;
pub use cas_solver_core::domain_mode::DomainMode;
pub use cas_solver_core::eval_config::EvalConfig;
pub use cas_solver_core::inverse_trig_policy::InverseTrigPolicy;
pub use cas_solver_core::value_domain::ValueDomain;
