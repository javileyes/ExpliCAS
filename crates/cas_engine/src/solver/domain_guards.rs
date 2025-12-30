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
    Unsupported(String),
}

/// Assumptions that the solver may record when proceeding under `Assume` mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverAssumption {
    /// Assumed: RHS > 0
    PositiveRhs,
    /// Assumed: Base > 0
    PositiveBase,
}

impl SolverAssumption {
    /// Convert to a human-readable assumption string.
    pub fn to_string(&self, ctx: &Context, base: ExprId, rhs: ExprId) -> String {
        match self {
            SolverAssumption::PositiveRhs => {
                format!(
                    "positive({})",
                    cas_ast::DisplayExpr {
                        context: ctx,
                        id: rhs
                    }
                )
            }
            SolverAssumption::PositiveBase => {
                format!(
                    "positive({})",
                    cas_ast::DisplayExpr {
                        context: ctx,
                        id: base
                    }
                )
            }
        }
    }

    /// Convert to an AssumptionEvent for the assumptions pipeline.
    pub fn to_assumption_event(
        &self,
        ctx: &Context,
        base: ExprId,
        rhs: ExprId,
    ) -> crate::assumptions::AssumptionEvent {
        use crate::assumptions::AssumptionEvent;
        match self {
            SolverAssumption::PositiveRhs => AssumptionEvent::positive(ctx, rhs),
            SolverAssumption::PositiveBase => AssumptionEvent::positive(ctx, base),
        }
    }
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
pub fn classify_log_solve(
    ctx: &Context,
    base: ExprId,
    rhs: ExprId,
    opts: &SolverOptions,
) -> LogSolveDecision {
    // Only applies to RealOnly mode
    // In Complex mode, this PR doesn't decide multi-valued branches
    if opts.value_domain != ValueDomain::RealOnly {
        return LogSolveDecision::Ok;
    }

    let mode = opts.domain_mode;
    let base_proof = prove_positive(ctx, base);
    let rhs_proof = prove_positive(ctx, rhs);

    // Case: base>0 proven and rhs<0 proven => EmptySet
    // (a^x > 0 for all real x when a > 0, so no solution exists)
    if base_proof == Proof::Proven && rhs_proof == Proof::Disproven {
        return LogSolveDecision::EmptySet(
            "No real solutions: base^x > 0 for all real x, but RHS ≤ 0".to_string(),
        );
    }

    // If base≤0 proven => needs complex (can't take real log of negative base)
    if base_proof == Proof::Disproven {
        return LogSolveDecision::NeedsComplex(
            "Cannot take real logarithm: base is not positive".to_string(),
        );
    }

    // If rhs≤0 proven (and base proof not yet covered) => needs complex
    if rhs_proof == Proof::Disproven {
        return LogSolveDecision::NeedsComplex(
            "Cannot take real logarithm: RHS is not positive".to_string(),
        );
    }

    // At this point: base is Proven or Unknown, rhs is Proven or Unknown
    match (base_proof, rhs_proof, mode) {
        // Both proven positive: safe to proceed
        (Proof::Proven, Proof::Proven, _) => LogSolveDecision::Ok,

        // Base proven, RHS unknown: only Assume mode allows
        (Proof::Proven, Proof::Unknown, DomainMode::Assume) => {
            LogSolveDecision::OkWithAssumptions(vec![SolverAssumption::PositiveRhs])
        }
        (Proof::Proven, Proof::Unknown, DomainMode::Strict | DomainMode::Generic) => {
            LogSolveDecision::Unsupported(
                "Cannot prove RHS > 0 for logarithm in current domain mode".to_string(),
            )
        }

        // Base unknown, RHS proven: only Assume mode allows
        (Proof::Unknown, Proof::Proven, DomainMode::Assume) => {
            LogSolveDecision::OkWithAssumptions(vec![SolverAssumption::PositiveBase])
        }
        (Proof::Unknown, Proof::Proven, DomainMode::Strict | DomainMode::Generic) => {
            LogSolveDecision::Unsupported(
                "Cannot prove base > 0 for logarithm in current domain mode".to_string(),
            )
        }

        // Both unknown: only Assume mode allows (with both assumptions)
        (Proof::Unknown, Proof::Unknown, DomainMode::Assume) => {
            LogSolveDecision::OkWithAssumptions(vec![
                SolverAssumption::PositiveBase,
                SolverAssumption::PositiveRhs,
            ])
        }
        (Proof::Unknown, Proof::Unknown, DomainMode::Strict | DomainMode::Generic) => {
            LogSolveDecision::Unsupported(
                "Cannot justify logarithm step in RealOnly mode".to_string(),
            )
        }

        // Disproven cases already handled above, but for exhaustiveness:
        (Proof::Disproven, _, _) | (_, Proof::Disproven, _) => {
            LogSolveDecision::NeedsComplex("Requires complex logarithm".to_string())
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
        };
        (ctx, opts)
    }

    #[test]
    fn test_both_proven_positive_ok() {
        let (mut ctx, opts) = make_test_ctx_and_opts(DomainMode::Generic, AssumeScope::Real);
        let base = ctx.num(2); // 2 > 0 proven
        let rhs = ctx.num(8); // 8 > 0 proven

        let decision = classify_log_solve(&ctx, base, rhs, &opts);
        assert!(matches!(decision, LogSolveDecision::Ok));
    }

    #[test]
    fn test_base_positive_rhs_negative_empty() {
        let (mut ctx, opts) = make_test_ctx_and_opts(DomainMode::Generic, AssumeScope::Real);
        let base = ctx.num(2);
        let rhs = ctx.num(-5);

        let decision = classify_log_solve(&ctx, base, rhs, &opts);
        assert!(matches!(decision, LogSolveDecision::EmptySet(_)));
    }

    #[test]
    fn test_negative_base_needs_complex() {
        let (mut ctx, opts) = make_test_ctx_and_opts(DomainMode::Generic, AssumeScope::Real);
        let base = ctx.num(-2);
        let rhs = ctx.num(5);

        let decision = classify_log_solve(&ctx, base, rhs, &opts);
        assert!(matches!(decision, LogSolveDecision::NeedsComplex(_)));
    }

    #[test]
    fn test_assume_mode_unknown_rhs_ok_with_assumptions() {
        let (mut ctx, opts) = make_test_ctx_and_opts(DomainMode::Assume, AssumeScope::Real);
        let base = ctx.num(2);
        let rhs = ctx.var("y"); // Unknown

        let decision = classify_log_solve(&ctx, base, rhs, &opts);
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

        let decision = classify_log_solve(&ctx, base, rhs, &opts);
        assert!(matches!(decision, LogSolveDecision::Unsupported(_)));
    }
}
