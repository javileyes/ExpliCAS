//! `cas_engine` compatibility exports.
//!
//! Keeping these in one place reduces the surface area of `lib.rs` while
//! migration is still routing portions of the solver facade through engine APIs.

pub use cas_engine::error;
pub use cas_engine::expand;
pub use cas_engine::rules;
pub use cas_engine::rules::logarithms::LogExpansionRule;
pub use cas_engine::ImportanceLevel;
pub use cas_engine::Orchestrator;
pub use cas_engine::ParentContext;
pub use cas_engine::Rewrite;
pub use cas_engine::Rule;
pub use cas_engine::SharedSemanticConfig;
pub use cas_engine::SimpleRule;
pub use cas_engine::{
    AutoExpandBinomials, BranchMode, Budget, CasError, ComplexMode, ContextMode, DisplayEvalSteps,
    Engine, EvalAction, EvalOptions, EvalOutput, EvalRequest, EvalResult, HeuristicPoly, Metric,
    Operation, PassStats, PathStep, PipelineStats, RuleProfiler, Simplifier, SimplifyOptions, Step,
    StepCategory, StepsMode,
};
pub use cas_engine::{BudgetExceeded, StandardOracle};
pub use cas_engine::{ExpandBudget, PhaseBudgets, PhaseMask, PhaseStats};
pub use cas_engine::{ExpandPolicy, SimplifyPhase};
