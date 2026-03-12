//! Backward-compatible API facade mirroring former `cas_engine::api::*` usage.

pub use crate::command_api::solve::prepare_timeline_solve_equation;
pub use crate::domain_facade::{
    derive_requires_from_equation, domain_delta_check, infer_implicit_domain,
};
pub use crate::solve_safety::{RequirementDescriptor, RuleSolveSafetyExt, SolveSafety};
pub use crate::solver_entrypoints_eval::{expand, expand_with_stats, fold_constants};
pub use crate::solver_entrypoints_proof_verify::{
    cancel_additive_terms_semantic, cancel_common_additive_terms, CancelResult,
};
pub use crate::solver_entrypoints_solve::{solve, solve_with_display_steps};
pub use crate::standard_oracle::{oracle_allows_with_hint, StandardOracle};
pub use crate::telescoping::telescope;
pub use crate::types::{
    DisplaySolveSteps, SolveDiagnostics, SolveStep, SolveSubStep, SolverOptions,
};
pub use cas_ast::{
    BoundType, Case, ConditionPredicate, ConditionSet, Interval, SolutionSet, SolveResult,
};
pub use cas_formatter::{DisplayExpr, LaTeXExpr};
pub use cas_math::evaluator_f64::{
    eval_f64, eval_f64_checked, EvalCheckedError, EvalCheckedOptions,
};
pub use cas_math::expr_predicates::is_zero_expr as is_zero;
pub use cas_solver_core::diagnostics_model::{Diagnostics, RequiredItem};
pub use cas_solver_core::equivalence::EquivalenceResult;
pub use cas_solver_core::verification::{VerifyResult, VerifyStatus, VerifySummary};
pub use cas_solver_core::verify_stats;
pub use cas_solver_core::{
    assume_scope::AssumeScope,
    assumption_model::{AssumptionCollector, AssumptionEvent, AssumptionKey, AssumptionKind},
    assumption_reporting::AssumptionReporting,
    const_fold_types::{ConstFoldMode, ConstFoldResult},
    diagnostics_model::RequireOrigin,
    domain_assumption_classification::classify_assumption,
    domain_condition::{ImplicitCondition, ImplicitDomain, RequiresDisplayLevel},
    domain_context::DomainContext,
    domain_facts_model::{FactStrength, Predicate},
    domain_inference::DomainDelta,
    domain_inference_counter::{get as infer_domain_calls_get, reset as infer_domain_calls_reset},
    domain_normalization::{
        normalize_and_dedupe_conditions, normalize_condition, normalize_condition_expr,
        render_conditions_normalized,
    },
    domain_oracle_model::DomainOracle,
    domain_proof::Proof,
    domain_warning::DomainWarning,
    isolation_utils::contains_var,
    solve_budget::SolveBudget,
    solve_infer::infer_solve_variable,
    solve_safety_policy::{ConditionClass, ProvenanceKind as Provenance},
};

pub use crate::solver_entrypoints_proof_verify::{
    prove_nonzero, prove_positive, verify_solution, verify_solution_set,
};
