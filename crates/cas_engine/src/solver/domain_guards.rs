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

use crate::domain::DomainMode;
use crate::helpers::prove_positive;
use crate::semantics::ValueDomain;
use cas_solver_core::log_domain::{
    classify_log_solve_with_prover_runtime, LogSolveDecision, LogSolveProofRuntime, ProofStatus,
};

struct EngineLogSolveProofRuntime {
    value_domain: ValueDomain,
}

impl LogSolveProofRuntime for EngineLogSolveProofRuntime {
    fn prove_positive_status(&mut self, ctx: &Context, expr: ExprId) -> ProofStatus {
        match prove_positive(ctx, expr, self.value_domain) {
            crate::domain::Proof::Proven | crate::domain::Proof::ProvenImplicit => {
                ProofStatus::Proven
            }
            crate::domain::Proof::Unknown => ProofStatus::Unknown,
            crate::domain::Proof::Disproven => ProofStatus::Disproven,
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
pub(crate) fn classify_log_solve(
    ctx: &Context,
    base: ExprId,
    rhs: ExprId,
    value_domain: ValueDomain,
    domain_mode: DomainMode,
    base_in_env: bool,
    rhs_in_env: bool,
) -> LogSolveDecision {
    let mut runtime = EngineLogSolveProofRuntime { value_domain };
    classify_log_solve_with_prover_runtime(
        ctx,
        base,
        rhs,
        value_domain == ValueDomain::RealOnly,
        match domain_mode {
            DomainMode::Strict => cas_solver_core::log_domain::DomainModeKind::Strict,
            DomainMode::Generic => cas_solver_core::log_domain::DomainModeKind::Generic,
            DomainMode::Assume => cas_solver_core::log_domain::DomainModeKind::Assume,
        },
        base_in_env,
        rhs_in_env,
        &mut runtime,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_solver_core::log_domain::LogAssumption;

    fn make_test_ctx() -> Context {
        Context::new()
    }

    fn classify_for_test(
        ctx: &Context,
        base: ExprId,
        rhs: ExprId,
        mode: DomainMode,
    ) -> LogSolveDecision {
        classify_log_solve(ctx, base, rhs, ValueDomain::RealOnly, mode, false, false)
    }

    #[test]
    fn test_both_proven_positive_ok() {
        let mut ctx = make_test_ctx();
        let base = ctx.num(2); // 2 > 0 proven
        let rhs = ctx.num(8); // 8 > 0 proven

        let decision = classify_for_test(&ctx, base, rhs, DomainMode::Generic);
        assert!(matches!(decision, LogSolveDecision::Ok));
    }

    #[test]
    fn test_base_positive_rhs_negative_empty() {
        let mut ctx = make_test_ctx();
        let base = ctx.num(2);
        let rhs = ctx.num(-5);

        let decision = classify_for_test(&ctx, base, rhs, DomainMode::Generic);
        assert!(matches!(decision, LogSolveDecision::EmptySet(_)));
    }

    #[test]
    fn test_negative_base_needs_complex() {
        let mut ctx = make_test_ctx();
        let base = ctx.num(-2);
        let rhs = ctx.num(5);

        let decision = classify_for_test(&ctx, base, rhs, DomainMode::Generic);
        assert!(matches!(decision, LogSolveDecision::NeedsComplex(_)));
    }

    #[test]
    fn test_assume_mode_unknown_rhs_ok_with_assumptions() {
        let mut ctx = make_test_ctx();
        let base = ctx.num(2);
        let rhs = ctx.var("y"); // Unknown

        let decision = classify_for_test(&ctx, base, rhs, DomainMode::Assume);
        match decision {
            LogSolveDecision::OkWithAssumptions(assumptions) => {
                assert!(assumptions.contains(&LogAssumption::PositiveRhs));
            }
            _ => panic!("Expected OkWithAssumptions, got {:?}", decision),
        }
    }

    #[test]
    fn test_generic_mode_unknown_rhs_unsupported() {
        let mut ctx = make_test_ctx();
        let base = ctx.num(2);
        let rhs = ctx.var("y");

        let decision = classify_for_test(&ctx, base, rhs, DomainMode::Generic);
        assert!(matches!(decision, LogSolveDecision::Unsupported(_, _)));
    }

    #[test]
    fn test_base_positive_rhs_neg_expr_empty() {
        // Test with Neg(5) which is how parser represents -5
        let mut ctx = make_test_ctx();
        let base = ctx.num(2);
        let five = ctx.num(5);
        let rhs = ctx.add(cas_ast::Expr::Neg(five)); // Neg(5) instead of Number(-5)

        let decision = classify_for_test(&ctx, base, rhs, DomainMode::Generic);
        assert!(
            matches!(decision, LogSolveDecision::EmptySet(_)),
            "Expected EmptySet for Neg(5), got {:?}",
            decision
        );
    }
}
