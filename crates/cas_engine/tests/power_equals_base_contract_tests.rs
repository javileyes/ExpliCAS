//! Contract tests for Power Equals Base Shortcut
//!
//! Tests the solver's ability to solve exponential equations of the form
//! base^x = base without needing logarithms.
//!
//! Key patterns:
//! - a^x = a  ⟹  x = 1 (for any a ≠ 0)
//! - a^x = a^n  ⟹  x = n (equal bases imply equal exponents when a ≠ 0, 1)
//! - 1^x = 1  ⟹  AllReals (already handled by simplification)

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
// Test 1: a^x = a  ⟹  x = 1 (Power Equals Base Shortcut)
// =============================================================================

#[test]
fn power_equals_base_symbolic_strict_mode() {
    // a^x = a in Strict mode - should return {1} without log
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let a = ctx.var("a");
    let x = ctx.var("x");
    let pow = ctx.add(Expr::Pow(a, x));

    let eq = Equation {
        lhs: pow,
        rhs: a,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Strict, AssumeScope::Real);
    let result = solve_with_options(&eq, "x", &mut engine.simplifier, opts);

    assert!(result.is_ok(), "Should solve a^x = a, got: {:?}", result);

    let (solution_set, _steps) = result.unwrap();

    // Should be Discrete with single solution
    match solution_set {
        SolutionSet::Discrete(sols) => {
            assert_eq!(
                sols.len(),
                1,
                "Expected exactly 1 solution, got: {:?}",
                sols
            );
            let sol_str = cas_ast::DisplayExpr {
                context: &engine.simplifier.context,
                id: sols[0],
            }
            .to_string();
            assert_eq!(sol_str, "1", "Expected solution = 1, got: {}", sol_str);
        }
        other => panic!("Expected Discrete solution set, got: {:?}", other),
    }
}

#[test]
fn power_equals_base_symbolic_generic_mode() {
    // a^x = a in Generic mode - should also return {1}
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let a = ctx.var("a");
    let x = ctx.var("x");
    let pow = ctx.add(Expr::Pow(a, x));

    let eq = Equation {
        lhs: pow,
        rhs: a,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Generic, AssumeScope::Real);
    let result = solve_with_options(&eq, "x", &mut engine.simplifier, opts);

    assert!(result.is_ok(), "Should solve a^x = a, got: {:?}", result);

    let (solution_set, _steps) = result.unwrap();

    match solution_set {
        SolutionSet::Discrete(sols) => {
            assert_eq!(sols.len(), 1, "Expected exactly 1 solution");
            let sol_str = cas_ast::DisplayExpr {
                context: &engine.simplifier.context,
                id: sols[0],
            }
            .to_string();
            assert_eq!(sol_str, "1", "Expected solution = 1, got: {}", sol_str);
        }
        other => panic!("Expected Discrete solution set, got: {:?}", other),
    }
}

// =============================================================================
// Test 2: a^x = a^n  ⟹  x = n (Equal Bases Pattern)
// =============================================================================

#[test]
fn power_equals_power_symbolic_gives_exponent() {
    // a^x = a^2 in Strict mode - should return {2}
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let a = ctx.var("a");
    let x = ctx.var("x");
    let two = ctx.num(2);
    let pow_ax = ctx.add(Expr::Pow(a, x));
    let pow_a2 = ctx.add(Expr::Pow(a, two));

    let eq = Equation {
        lhs: pow_ax,
        rhs: pow_a2,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Strict, AssumeScope::Real);
    let result = solve_with_options(&eq, "x", &mut engine.simplifier, opts);

    assert!(result.is_ok(), "Should solve a^x = a^2, got: {:?}", result);

    let (solution_set, _steps) = result.unwrap();

    match solution_set {
        SolutionSet::Discrete(sols) => {
            assert_eq!(sols.len(), 1, "Expected exactly 1 solution");
            let sol_str = cas_ast::DisplayExpr {
                context: &engine.simplifier.context,
                id: sols[0],
            }
            .to_string();
            assert_eq!(sol_str, "2", "Expected solution = 2, got: {}", sol_str);
        }
        other => panic!("Expected Discrete solution set, got: {:?}", other),
    }
}

#[test]
fn power_equals_power_symbolic_gives_exponent_3() {
    // b^x = b^3 - should return {3}
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let b = ctx.var("b");
    let x = ctx.var("x");
    let three = ctx.num(3);
    let pow_bx = ctx.add(Expr::Pow(b, x));
    let pow_b3 = ctx.add(Expr::Pow(b, three));

    let eq = Equation {
        lhs: pow_bx,
        rhs: pow_b3,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Generic, AssumeScope::Real);
    let result = solve_with_options(&eq, "x", &mut engine.simplifier, opts);

    assert!(result.is_ok(), "Should solve b^x = b^3, got: {:?}", result);

    let (solution_set, _steps) = result.unwrap();

    match solution_set {
        SolutionSet::Discrete(sols) => {
            assert_eq!(sols.len(), 1, "Expected exactly 1 solution");
            let sol_str = cas_ast::DisplayExpr {
                context: &engine.simplifier.context,
                id: sols[0],
            }
            .to_string();
            assert_eq!(sol_str, "3", "Expected solution = 3, got: {}", sol_str);
        }
        other => panic!("Expected Discrete solution set, got: {:?}", other),
    }
}

// =============================================================================
// Test 3: 2^x = 2  ⟹  x = 1 (Numeric base, pattern still works via log)
// =============================================================================

#[test]
fn numeric_base_2_to_x_equals_2() {
    // 2^x = 2 - should return {1} (either via shortcut or log)
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let two = ctx.num(2);
    let x = ctx.var("x");
    let pow = ctx.add(Expr::Pow(two, x));

    let eq = Equation {
        lhs: pow,
        rhs: two,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Generic, AssumeScope::Real);
    let result = solve_with_options(&eq, "x", &mut engine.simplifier, opts);

    assert!(result.is_ok(), "Should solve 2^x = 2, got: {:?}", result);

    let (solution_set, _steps) = result.unwrap();

    match solution_set {
        SolutionSet::Discrete(sols) => {
            assert_eq!(sols.len(), 1, "Expected exactly 1 solution");
            let sol_str = cas_ast::DisplayExpr {
                context: &engine.simplifier.context,
                id: sols[0],
            }
            .to_string();
            assert_eq!(sol_str, "1", "Expected solution = 1, got: {}", sol_str);
        }
        other => panic!("Expected Discrete solution set, got: {:?}", other),
    }
}
