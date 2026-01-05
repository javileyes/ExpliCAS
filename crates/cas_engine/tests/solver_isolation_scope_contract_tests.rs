//! Contract tests for PR-SCOPE-3.3: Classifier in IsolationStrategy.
//!
//! These tests verify that IsolationStrategy uses the same decision policy
//! as classify_log_solve(), ensuring coherent handling across all solve paths.

use cas_ast::{Equation, Expr, RelOp, SolutionSet};
use cas_engine::domain::DomainMode;
use cas_engine::semantics::{AssumeScope, ValueDomain};
use cas_engine::solver::{solve_with_options, SolveAssumptionsGuard, SolverOptions};
use cas_engine::Engine;

fn make_opts(mode: DomainMode, scope: AssumeScope) -> SolverOptions {
    SolverOptions {
        value_domain: ValueDomain::RealOnly,
        domain_mode: mode,
        assume_scope: scope,
    }
}

fn setup_engine() -> Engine {
    Engine::new()
}

// =============================================================================
// Test 1: Strict/Generic rejects unknown RHS (no ln(y) garbage)
// =============================================================================

#[test]
fn strict_mode_returns_conditional_for_unknown_rhs() {
    // V2.0: 2^x = y in Strict mode - should return Conditional (guarded solution)
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let two = ctx.num(2);
    let x = ctx.var("x");
    let y = ctx.var("y");
    let pow = ctx.add(Expr::Pow(two, x));

    let eq = Equation {
        lhs: pow,
        rhs: y,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Strict, AssumeScope::Real);
    let result = solve_with_options(&eq, "x", &mut engine.simplifier, opts);

    // V2.0: Should return Conditional (not crash, not Residual)
    assert!(
        result.is_ok(),
        "Strict mode should return Conditional for unknown RHS, got: {:?}",
        result
    );

    let (solution_set, _steps) = result.unwrap();

    // V2.0: Expect Conditional with guarded solution
    match &solution_set {
        SolutionSet::Conditional(cases) => {
            assert!(!cases.is_empty(), "Should have at least one case");
            // First case should have Positive(y) guard
            let first = &cases[0];
            assert!(
                !first.when.is_empty(),
                "First case should have a guard (Positive(y))"
            );
        }
        _ => panic!(
            "Should be SolutionSet::Conditional, got: {:?}",
            solution_set
        ),
    }
}

#[test]
fn generic_mode_returns_conditional_for_unknown_rhs() {
    // V2.0: 2^x = y in Generic mode - should return Conditional (guarded solution)
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let two = ctx.num(2);
    let x = ctx.var("x");
    let y = ctx.var("y");
    let pow = ctx.add(Expr::Pow(two, x));

    let eq = Equation {
        lhs: pow,
        rhs: y,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Generic, AssumeScope::Real);
    let result = solve_with_options(&eq, "x", &mut engine.simplifier, opts);

    // V2.0: Should return Conditional (not crash, not Residual)
    assert!(
        result.is_ok(),
        "Generic mode should return Conditional for unknown RHS, got: {:?}",
        result
    );

    let (solution_set, _steps) = result.unwrap();

    // V2.0: Expect Conditional with guarded solution
    match &solution_set {
        SolutionSet::Conditional(cases) => {
            assert!(!cases.is_empty(), "Should have at least one case");
            // Verify first case has solution (not just residual)
            let first = &cases[0];
            assert!(
                first.then.has_solutions(),
                "First case should have actual solutions under guard"
            );
        }
        _ => panic!(
            "Should be SolutionSet::Conditional, got: {:?}",
            solution_set
        ),
    }
}

// =============================================================================
// Test 2: Assume + Real mode allows with assumption via isolation path
// =============================================================================

#[test]
fn assume_real_allows_with_assumption_via_isolation() {
    // 2^x = y in Assume+Real mode - should succeed with positive(y) assumption
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let two = ctx.num(2);
    let x = ctx.var("x");
    let y = ctx.var("y");
    let pow = ctx.add(Expr::Pow(two, x));

    let eq = Equation {
        lhs: pow,
        rhs: y,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Assume, AssumeScope::Real);

    // Collect assumptions
    let guard = SolveAssumptionsGuard::new(true);
    let result = solve_with_options(&eq, "x", &mut engine.simplifier, opts);
    let assumptions = guard.finish();

    // Should succeed
    assert!(
        result.is_ok(),
        "Assume mode should succeed with symbolic RHS, got: {:?}",
        result
    );

    // Should have positive(y) assumption
    let has_positive_y = assumptions
        .iter()
        .any(|a| a.kind == "positive" && a.expr == "y");
    assert!(
        has_positive_y,
        "Should have positive(y) assumption, got: {:?}",
        assumptions
    );
}

// =============================================================================
// Test 3: Assume + Wildcard with negative base returns Residual (no garbage)
// =============================================================================

#[test]
fn assume_wildcard_negative_base_returns_residual_isolation() {
    // (-2)^x = 5 in Assume+Wildcard mode - should return Residual, not ln(-2) garbage
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let two = ctx.num(2);
    let neg_two = ctx.add(Expr::Neg(two));
    let x = ctx.var("x");
    let five = ctx.num(5);
    let pow = ctx.add(Expr::Pow(neg_two, x));

    let eq = Equation {
        lhs: pow,
        rhs: five,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Assume, AssumeScope::Wildcard);
    let result = solve_with_options(&eq, "x", &mut engine.simplifier, opts);

    // Should succeed with Residual
    assert!(
        result.is_ok(),
        "Wildcard mode should return Residual, got: {:?}",
        result
    );

    let (solution_set, _steps) = result.unwrap();

    // Should be Residual variant
    assert!(
        matches!(solution_set, cas_ast::SolutionSet::Residual(_)),
        "Should be SolutionSet::Residual, got: {:?}",
        solution_set
    );

    // Verify NO garbage: the residual should not contain ln(-2) or undefined
    if let cas_ast::SolutionSet::Residual(residual_expr) = solution_set {
        let residual_str = cas_ast::DisplayExpr {
            context: &engine.simplifier.context,
            id: residual_expr,
        }
        .to_string();

        assert!(
            !residual_str.contains("ln(-"),
            "Residual should NOT contain ln(-...), got: {}",
            residual_str
        );
        assert!(
            !residual_str.contains("undefined"),
            "Residual should NOT contain 'undefined', got: {}",
            residual_str
        );
    }
}
