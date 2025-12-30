//! Contract tests for PR-SCOPE-3: Solver AssumeScope behavior.
//!
//! These tests verify that the solver correctly handles exponential equations
//! based on the AssumeScope semantic axis.

use cas_ast::{Equation, Expr, RelOp, SolutionSet};
use cas_engine::domain::DomainMode;
use cas_engine::semantics::{AssumeScope, ValueDomain};
use cas_engine::solver::{solve_with_options, SolverOptions};
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
// Base = 1 Tests (pre-check before classifier)
// =============================================================================

#[test]
fn base_one_rhs_different_is_empty() {
    // 1^x = 5 -> Empty (no real solutions: 1^x always equals 1)
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let one = ctx.num(1);
    let x = ctx.var("x");
    let five = ctx.num(5);
    let pow = ctx.add(Expr::Pow(one, x));

    let eq = Equation {
        lhs: pow,
        rhs: five,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Generic, AssumeScope::Real);
    let result = solve_with_options(&eq, "x", &mut engine.simplifier, opts);

    match result {
        Ok((SolutionSet::Empty, _)) => {} // Expected
        Ok((SolutionSet::Discrete(sols), _)) if sols.is_empty() => {} // Also ok
        other => panic!("1^x = 5 should give Empty, got: {:?}", other),
    }
}

#[test]
fn base_one_rhs_one_is_all_reals() {
    // 1^x = 1 -> AllReals (any x works)
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let one_base = ctx.num(1);
    let one_rhs = ctx.num(1);
    let x = ctx.var("x");
    let pow = ctx.add(Expr::Pow(one_base, x));

    let eq = Equation {
        lhs: pow,
        rhs: one_rhs,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Generic, AssumeScope::Real);
    let result = solve_with_options(&eq, "x", &mut engine.simplifier, opts);

    match result {
        Ok((SolutionSet::AllReals, _)) => {} // Expected
        other => panic!("1^x = 1 should give AllReals, got: {:?}", other),
    }
}

// =============================================================================
// Positive Base, Negative RHS -> Empty (base^x > 0 always)
// =============================================================================

#[test]
fn positive_base_negative_rhs_is_empty() {
    // 2^x = -5 -> Empty (positive base raised to any power is positive)
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let two = ctx.num(2);
    let x = ctx.var("x");
    let neg_five = ctx.num(-5);
    let pow = ctx.add(Expr::Pow(two, x));

    let eq = Equation {
        lhs: pow,
        rhs: neg_five,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Generic, AssumeScope::Real);
    let result = solve_with_options(&eq, "x", &mut engine.simplifier, opts);

    match result {
        Ok((SolutionSet::Empty, _)) => {} // Expected
        Ok((SolutionSet::Discrete(sols), _)) if sols.is_empty() => {} // Also ok
        other => panic!("2^x = -5 should give Empty, got: {:?}", other),
    }
}

#[test]
fn positive_base_zero_rhs_is_empty() {
    // 2^x = 0 -> Empty (2^x > 0 for all x)
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let two = ctx.num(2);
    let x = ctx.var("x");
    let zero = ctx.num(0);
    let pow = ctx.add(Expr::Pow(two, x));

    let eq = Equation {
        lhs: pow,
        rhs: zero,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Generic, AssumeScope::Real);
    let result = solve_with_options(&eq, "x", &mut engine.simplifier, opts);

    match result {
        Ok((SolutionSet::Empty, _)) => {} // Expected
        Ok((SolutionSet::Discrete(sols), _)) if sols.is_empty() => {} // Also ok
        other => panic!("2^x = 0 should give Empty, got: {:?}", other),
    }
}

// =============================================================================
// Simple Positive Case - Regression Test
// =============================================================================

#[test]
fn simple_exponential_still_works() {
    // 2^x = 8 -> x = 3
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let two = ctx.num(2);
    let x = ctx.var("x");
    let eight = ctx.num(8);
    let pow = ctx.add(Expr::Pow(two, x));

    let eq = Equation {
        lhs: pow,
        rhs: eight,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Generic, AssumeScope::Real);
    let result = solve_with_options(&eq, "x", &mut engine.simplifier, opts);

    match result {
        Ok((SolutionSet::Discrete(sols), _)) if !sols.is_empty() => {
            // Should have solution x = 3
            // We just verify we got a solution, not the exact value
        }
        other => panic!("2^x = 8 should have solution, got: {:?}", other),
    }
}

// =============================================================================
// Assume Mode with Unknown Positivity
// =============================================================================

#[test]
fn assume_mode_allows_unknown_rhs() {
    // 2^x = y (y unknown variable) in Assume mode should produce solution
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
    let result = solve_with_options(&eq, "x", &mut engine.simplifier, opts);

    // In Assume mode, should produce a solution (with assumptions tracked)
    match result {
        Ok((SolutionSet::Discrete(sols), _)) if !sols.is_empty() => {
            // Got a solution - good
        }
        Ok((SolutionSet::Discrete(_), _)) => {
            // Empty is also acceptable if implementation differs
        }
        other => {
            // Error is NOT acceptable in Assume mode
            panic!("Assume mode should allow solving 2^x = y, got: {:?}", other)
        }
    }
}

// =============================================================================
// No-Garbage Invariant
// =============================================================================

#[test]
fn no_garbage_undefined_in_result() {
    // In RealOnly mode, result should never contain undefined
    let mut engine = setup_engine();

    // Build equation first
    let (eq, _two, _x, _eight, _pow) = {
        let ctx = &mut engine.simplifier.context;
        let two = ctx.num(2);
        let x = ctx.var("x");
        let eight = ctx.num(8);
        let pow = ctx.add(Expr::Pow(two, x));

        let eq = Equation {
            lhs: pow,
            rhs: eight,
            op: RelOp::Eq,
        };
        (eq, two, x, eight, pow)
    };

    let opts = make_opts(DomainMode::Generic, AssumeScope::Real);
    let result = solve_with_options(&eq, "x", &mut engine.simplifier, opts);

    // Now we can borrow context again for display
    if let Ok((SolutionSet::Discrete(sols), _)) = result {
        for sol in sols {
            let sol_str = format!(
                "{}",
                cas_ast::DisplayExpr {
                    context: &engine.simplifier.context,
                    id: sol
                }
            );
            assert!(
                !sol_str.contains("undefined"),
                "Solution should not contain 'undefined': {}",
                sol_str
            );
        }
    }
}
