//! Solver facade crate.
//!
//! During migration this crate re-exports the solver API from `cas_engine`.

pub mod json;
pub mod substitute;

pub use cas_engine::canonical_forms;
pub use cas_engine::normalize_and_dedupe_conditions;
pub use cas_engine::phase::{ExpandPolicy, SimplifyPhase};
pub use cas_engine::rules;
pub use cas_engine::rules::logarithms::LogExpansionRule;
pub use cas_engine::semantics::{BranchPolicy, InverseTrigPolicy, ValueDomain};
pub use cas_engine::solver::*;
pub use cas_engine::visualizer;
pub use cas_engine::ConstFoldMode;
pub use cas_engine::ParentContext;
pub use cas_engine::Rule;
pub use cas_engine::{error, pattern_marks, phase, semantics};
pub use cas_engine::{limit, Approach, LimitOptions, PreSimplifyMode};
pub use cas_engine::{
    take_blocked_hints, AssumeScope, AssumptionKey, AssumptionReporting, AutoExpandBinomials,
    BlockedHint, BranchMode, Budget, CasError, ComplexMode, ContextMode, DomainMode, Engine,
    EquivalenceResult, EvalAction, EvalOptions, EvalOutput, EvalRequest, EvalResult, HeuristicPoly,
    ImplicitCondition, PipelineStats, RequiresDisplayLevel, Simplifier, SimplifyOptions, StepsMode,
};
pub use cas_engine::{AutoRationalizeLevel, RationalizeOutcome};
pub use cas_engine::{RequirementDescriptor, SimplifyPurpose, SolveSafety};
pub use cas_math::poly_store::{try_get_poly_result_term_count, try_render_poly_result};
pub use json::{
    eval_str_to_json, eval_str_to_output_envelope, substitute_str_to_json, EnvelopeEvalOptions,
};
pub use substitute::{substitute_power_aware, substitute_with_steps, SubstituteOptions};

/// Backward-compatible facade for former `cas_engine::expand::*` imports.
pub mod expand {
    pub use cas_engine::{
        eager_eval_expand_calls, estimate_expand_terms, expand, expand_div, expand_mul, expand_pow,
        expand_with_stats,
    };
}

/// Backward-compatible facade for former `cas_engine::helpers::*` imports.
pub mod helpers {
    pub use cas_engine::{is_zero, prove_nonzero, prove_positive};
}

/// Backward-compatible facade for former `cas_engine::engine::*` imports.
pub mod engine {
    pub use cas_engine::{
        eval_f64, eval_f64_checked, Engine, EquivalenceResult, EvalCheckedError,
        EvalCheckedOptions, LoopConfig, Simplifier,
    };
}

/// Backward-compatible facade for former `cas_engine::rationalize::*` imports.
pub mod rationalize {
    pub use cas_engine::{rationalize_denominator, RationalizeConfig, RationalizeResult};
}

/// Backward-compatible facade for former `cas_engine::telescoping::*` imports.
pub mod telescoping {
    pub use cas_engine::{
        telescope, try_dirichlet_kernel_identity_pub, DirichletKernelResult, TelescopingResult,
        TelescopingStep,
    };
}
