//! Domain guards for solver operations.
//!
//! This module provides decision logic for domain-sensitive solver operations,
//! particularly for logarithmic transformations in exponential equations.
//!
//! # Key Principle
//!
//! In `RealOnly` mode, the solver should **never** materialize invalid expressions
//! like `ln(-5)` or produce `undefined` garbage. Instead:
//! - Proven invalid cases → EmptySet or NeedsComplex
//! - Unknown cases in Assume mode → Ok with assumptions
//! - Disproven cases → Never "assumed away"

use cas_ast::{Context, ExprId};

use crate::domain::{DomainMode, Proof};
use crate::helpers::prove_positive;
use crate::semantics::ValueDomain;
use crate::solver::SolverOptions;
use cas_solver_core::log_domain::{
    DomainModeKind, LogSolveDecision as CoreLogSolveDecision, ProofStatus,
};

/// Decision for whether a logarithmic solve step is valid.
#[derive(Debug, Clone)]
pub enum LogSolveDecision {
    /// Safe to proceed: both base>0 and rhs>0 are proven.
    Ok,

    /// Safe to proceed with recorded assumptions.
    /// Only valid when `domain_mode = Assume`.
    OkWithAssumptions(Vec<SolverAssumption>),

    /// No solutions exist in ℝ (e.g., base>0 but rhs<0).
    EmptySet(String),

    /// Requires complex logarithm (base≤0 or rhs≤0 proven).
    /// In `assume_scope=Real`: error
    /// In `assume_scope=Wildcard`: residual + warning
    NeedsComplex(String),

    /// Cannot justify the step in current mode (strict/generic with unknown).
    /// Carries the missing conditions that would need to be assumed.
    Unsupported(String, Vec<SolverAssumption>),
}

/// Assumptions that the solver may record when proceeding under `Assume` mode.
pub type SolverAssumption = cas_solver_core::log_domain::LogAssumption;

/// Convert to a human-readable assumption string.
pub fn assumption_to_string(
    assumption: SolverAssumption,
    ctx: &Context,
    base: ExprId,
    rhs: ExprId,
) -> String {
    match assumption {
        SolverAssumption::PositiveRhs => {
            format!(
                "positive({})",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: rhs
                }
            )
        }
        SolverAssumption::PositiveBase => {
            format!(
                "positive({})",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: base
                }
            )
        }
    }
}

/// Convert to an AssumptionEvent for the assumptions pipeline.
pub fn assumption_to_assumption_event(
    assumption: SolverAssumption,
    ctx: &Context,
    base: ExprId,
    rhs: ExprId,
) -> crate::assumptions::AssumptionEvent {
    use crate::assumptions::AssumptionEvent;
    match assumption {
        SolverAssumption::PositiveRhs => AssumptionEvent::positive(ctx, rhs),
        SolverAssumption::PositiveBase => AssumptionEvent::positive(ctx, base),
    }
}

/// Convert to a ConditionPredicate for conditional solutions (V2.0).
pub fn assumption_to_condition_predicate(
    assumption: SolverAssumption,
    base: ExprId,
    rhs: ExprId,
) -> cas_ast::ConditionPredicate {
    use cas_ast::ConditionPredicate;
    match assumption {
        SolverAssumption::PositiveRhs => ConditionPredicate::Positive(rhs),
        SolverAssumption::PositiveBase => ConditionPredicate::Positive(base),
    }
}

/// Convert a list of SolverAssumptions to a ConditionSet.
pub fn assumptions_to_condition_set(
    assumptions: &[SolverAssumption],
    base: ExprId,
    rhs: ExprId,
) -> cas_ast::ConditionSet {
    let predicates: Vec<cas_ast::ConditionPredicate> = assumptions
        .iter()
        .map(|a| assumption_to_condition_predicate(*a, base, rhs))
        .collect();
    cas_ast::ConditionSet::from_predicates(predicates)
}

/// Classify whether a logarithmic solve step (for `base^x = rhs`) is valid.
///
/// This function decides whether the solver can proceed with taking logarithms
/// based on the semantic options and provable properties of base and rhs.
///
/// # Arguments
/// - `ctx`: AST context for expression lookup
/// - `base`: The base expression in `base^x`
/// - `rhs`: The right-hand side of the equation
/// - `opts`: Solver semantic options
///
/// # Returns
/// A `LogSolveDecision` indicating how to proceed.
pub(crate) fn classify_log_solve(
    ctx: &Context,
    base: ExprId,
    rhs: ExprId,
    opts: &SolverOptions,
    env: &super::SolveDomainEnv,
) -> LogSolveDecision {
    // Only applies to RealOnly mode
    // In Complex mode, this PR doesn't decide multi-valued branches
    if opts.value_domain != ValueDomain::RealOnly {
        return LogSolveDecision::Ok;
    }

    let mode = opts.domain_mode;
    let vd = opts.value_domain;

    // Check env.required for conditions already proven by domain inference
    let base_in_env = env.has_positive(base);
    let rhs_in_env = env.has_positive(rhs);

    let base_proof = if base_in_env {
        Proof::ProvenImplicit // Already in global requires
    } else {
        prove_positive(ctx, base, vd)
    };

    let rhs_proof = if rhs_in_env {
        Proof::ProvenImplicit // Already in global requires
    } else {
        prove_positive(ctx, rhs, vd)
    };

    let core_decision = cas_solver_core::log_domain::classify_log_solve_by_proofs(
        to_core_mode(mode),
        to_core_proof(base_proof),
        to_core_proof(rhs_proof),
    );
    from_core_decision(core_decision)
}

fn to_core_mode(mode: DomainMode) -> DomainModeKind {
    match mode {
        DomainMode::Strict => DomainModeKind::Strict,
        DomainMode::Generic => DomainModeKind::Generic,
        DomainMode::Assume => DomainModeKind::Assume,
    }
}

fn to_core_proof(proof: Proof) -> ProofStatus {
    match proof {
        Proof::Proven | Proof::ProvenImplicit => ProofStatus::Proven,
        Proof::Unknown => ProofStatus::Unknown,
        Proof::Disproven => ProofStatus::Disproven,
    }
}

fn from_core_decision(d: CoreLogSolveDecision) -> LogSolveDecision {
    match d {
        CoreLogSolveDecision::Ok => LogSolveDecision::Ok,
        CoreLogSolveDecision::OkWithAssumptions(v) => LogSolveDecision::OkWithAssumptions(v),
        CoreLogSolveDecision::EmptySet(msg) => LogSolveDecision::EmptySet(msg.to_string()),
        CoreLogSolveDecision::NeedsComplex(msg) => LogSolveDecision::NeedsComplex(msg.to_string()),
        CoreLogSolveDecision::Unsupported(msg, v) => {
            LogSolveDecision::Unsupported(msg.to_string(), v)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantics::AssumeScope;

    fn make_test_ctx_and_opts(mode: DomainMode, scope: AssumeScope) -> (Context, SolverOptions) {
        let ctx = Context::new();
        let opts = SolverOptions {
            value_domain: ValueDomain::RealOnly,
            domain_mode: mode,
            assume_scope: scope,
            budget: crate::solver::SolveBudget::default(),
            ..Default::default()
        };
        (ctx, opts)
    }

    #[test]
    fn test_both_proven_positive_ok() {
        let (mut ctx, opts) = make_test_ctx_and_opts(DomainMode::Generic, AssumeScope::Real);
        let base = ctx.num(2); // 2 > 0 proven
        let rhs = ctx.num(8); // 8 > 0 proven

        let decision = classify_log_solve(
            &ctx,
            base,
            rhs,
            &opts,
            &super::super::SolveDomainEnv::default(),
        );
        assert!(matches!(decision, LogSolveDecision::Ok));
    }

    #[test]
    fn test_base_positive_rhs_negative_empty() {
        let (mut ctx, opts) = make_test_ctx_and_opts(DomainMode::Generic, AssumeScope::Real);
        let base = ctx.num(2);
        let rhs = ctx.num(-5);

        let decision = classify_log_solve(
            &ctx,
            base,
            rhs,
            &opts,
            &super::super::SolveDomainEnv::default(),
        );
        assert!(matches!(decision, LogSolveDecision::EmptySet(_)));
    }

    #[test]
    fn test_negative_base_needs_complex() {
        let (mut ctx, opts) = make_test_ctx_and_opts(DomainMode::Generic, AssumeScope::Real);
        let base = ctx.num(-2);
        let rhs = ctx.num(5);

        let decision = classify_log_solve(
            &ctx,
            base,
            rhs,
            &opts,
            &super::super::SolveDomainEnv::default(),
        );
        assert!(matches!(decision, LogSolveDecision::NeedsComplex(_)));
    }

    #[test]
    fn test_assume_mode_unknown_rhs_ok_with_assumptions() {
        let (mut ctx, opts) = make_test_ctx_and_opts(DomainMode::Assume, AssumeScope::Real);
        let base = ctx.num(2);
        let rhs = ctx.var("y"); // Unknown

        let decision = classify_log_solve(
            &ctx,
            base,
            rhs,
            &opts,
            &super::super::SolveDomainEnv::default(),
        );
        match decision {
            LogSolveDecision::OkWithAssumptions(assumptions) => {
                assert!(assumptions.contains(&SolverAssumption::PositiveRhs));
            }
            _ => panic!("Expected OkWithAssumptions, got {:?}", decision),
        }
    }

    #[test]
    fn test_generic_mode_unknown_rhs_unsupported() {
        let (mut ctx, opts) = make_test_ctx_and_opts(DomainMode::Generic, AssumeScope::Real);
        let base = ctx.num(2);
        let rhs = ctx.var("y");

        let decision = classify_log_solve(
            &ctx,
            base,
            rhs,
            &opts,
            &super::super::SolveDomainEnv::default(),
        );
        assert!(matches!(decision, LogSolveDecision::Unsupported(_, _)));
    }

    #[test]
    fn test_base_positive_rhs_neg_expr_empty() {
        // Test with Neg(5) which is how parser represents -5
        let (mut ctx, opts) = make_test_ctx_and_opts(DomainMode::Generic, AssumeScope::Real);
        let base = ctx.num(2);
        let five = ctx.num(5);
        let rhs = ctx.add(cas_ast::Expr::Neg(five)); // Neg(5) instead of Number(-5)

        let decision = classify_log_solve(
            &ctx,
            base,
            rhs,
            &opts,
            &super::super::SolveDomainEnv::default(),
        );
        assert!(
            matches!(decision, LogSolveDecision::EmptySet(_)),
            "Expected EmptySet for Neg(5), got {:?}",
            decision
        );
    }
}
