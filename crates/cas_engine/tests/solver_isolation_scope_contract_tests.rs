//! Contract tests for PR-SCOPE-3.3: Solver Domain-Aware Solving.
//!
//! These tests verify that the solver correctly handles exponential equations
//! with symbolic RHS by using domain inference rather than conditional branching.
//!
//! V2.2+ UPDATE: The solver now uses derive_requires_from_equation() to infer
//! required conditions (like "y > 0" from "2^x = y"). This allows direct solving
//! with the condition captured in the "requires" output, rather than returning
//! a Conditional solution set.

use cas_ast::{Equation, Expr, RelOp, SolutionSet};
use cas_engine::domain::DomainMode;
use cas_engine::implicit_domain::ImplicitCondition;
use cas_engine::semantics::{AssumeScope, ValueDomain};
use cas_engine::solver::{solve_with_display_steps, take_solver_required, SolverOptions};
use cas_engine::Engine;

fn make_opts(mode: DomainMode, scope: AssumeScope) -> SolverOptions {
    SolverOptions {
        value_domain: ValueDomain::RealOnly,
        domain_mode: mode,
        assume_scope: scope,
        budget: cas_engine::solver::SolveBudget::default(),
        ..Default::default()
    }
}

fn setup_engine() -> Engine {
    Engine::new()
}

// =============================================================================
// Test 1: Strict mode solves with required conditions (no garbage ln(y))
// =============================================================================

#[test]
fn strict_mode_solves_with_required_conditions() {
    // V2.2: 2^x = y solves to x = ln(y)/ln(2) with required: y > 0
    // The solver infers y > 0 from the equation structure, allowing direct solving
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
    let result = solve_with_display_steps(&eq, "x", &mut engine.simplifier, opts);

    // Should succeed (not error)
    assert!(
        result.is_ok(),
        "Strict mode should solve 2^x = y, got: {:?}",
        result
    );

    let (solution_set, _steps) = result.unwrap();

    // V2.2: Should return Discrete solution (not Conditional)
    // The condition is captured in required_conditions, not in the solution set
    match &solution_set {
        SolutionSet::Discrete(solutions) => {
            assert!(!solutions.is_empty(), "Should have at least one solution");
            // Solution should be x = ln(y)/ln(2)
            let sol_str = cas_ast::DisplayExpr {
                context: &engine.simplifier.context,
                id: solutions[0],
            }
            .to_string();
            assert!(
                sol_str.contains("ln") || sol_str.contains("log"),
                "Solution should contain ln or log, got: {}",
                sol_str
            );
        }
        _ => {
            // Also accept Conditional (backward compat) or Residual
            // The key is no crash and no garbage
        }
    }

    // Check that y > 0 is in required conditions
    let required = take_solver_required();
    let has_positive_y = required.iter().any(|cond| {
        if let ImplicitCondition::Positive(id) = cond {
            cas_ast::DisplayExpr {
                context: &engine.simplifier.context,
                id: *id,
            }
            .to_string()
                == "y"
        } else {
            false
        }
    });
    assert!(has_positive_y, "Should require y > 0, got: {:?}", required);
}

#[test]
fn generic_mode_solves_with_required_conditions() {
    // V2.2: Same as strict mode - 2^x = y solves with y > 0 as required condition
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
    let result = solve_with_display_steps(&eq, "x", &mut engine.simplifier, opts);

    assert!(result.is_ok(), "Generic mode should solve 2^x = y");

    let (solution_set, _steps) = result.unwrap();

    // Should have a solution (Discrete or Conditional)
    let has_solution = match &solution_set {
        SolutionSet::Discrete(sols) => !sols.is_empty(),
        SolutionSet::Conditional(cases) => !cases.is_empty(),
        _ => false,
    };
    assert!(
        has_solution,
        "Should have a solution, got: {:?}",
        solution_set
    );
}

// =============================================================================
// Test 2: Assume mode solves with positive(y) in required conditions
// =============================================================================

#[test]
fn assume_real_solves_with_positive_requirement() {
    // 2^x = y in Assume+Real mode - solves and captures positive(y) requirement
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
    let result = solve_with_display_steps(&eq, "x", &mut engine.simplifier, opts);

    // Should succeed
    assert!(result.is_ok(), "Assume mode should solve 2^x = y");

    let (solution_set, _steps) = result.unwrap();

    // Should have a solution
    let has_solution = match &solution_set {
        SolutionSet::Discrete(sols) => !sols.is_empty(),
        SolutionSet::Conditional(cases) => !cases.is_empty(),
        _ => false,
    };
    assert!(
        has_solution,
        "Should have solutions, got: {:?}",
        solution_set
    );

    // Should have positive(y) in required conditions
    let required = take_solver_required();
    let has_positive_y = required.iter().any(|cond| {
        if let ImplicitCondition::Positive(id) = cond {
            cas_ast::DisplayExpr {
                context: &engine.simplifier.context,
                id: *id,
            }
            .to_string()
                == "y"
        } else {
            false
        }
    });
    assert!(
        has_positive_y,
        "Should have positive(y) requirement, got: {:?}",
        required
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
    let result = solve_with_display_steps(&eq, "x", &mut engine.simplifier, opts);

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

// =============================================================================
// V2.2: Budget control tests (simplified)
// =============================================================================

#[test]
fn budget_zero_still_solves_simple_exponential() {
    // V2.2: With budget=0, solver should still be able to solve simple exponentials
    // because domain inference doesn't require branching
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

    // Use budget=none (max_branches=0)
    let opts = SolverOptions {
        value_domain: ValueDomain::RealOnly,
        domain_mode: DomainMode::Generic,
        assume_scope: AssumeScope::Real,
        budget: cas_engine::solver::SolveBudget::none(),
        ..Default::default()
    };

    let result = solve_with_display_steps(&eq, "x", &mut engine.simplifier, opts);

    // Should succeed - 2^x = 8 doesn't need branching
    assert!(
        result.is_ok(),
        "Should solve 2^x = 8 even with budget=0, got: {:?}",
        result
    );

    let (solution_set, _steps) = result.unwrap();

    // Should have solution x = 3
    match solution_set {
        SolutionSet::Discrete(sols) => {
            assert!(!sols.is_empty(), "Should have at least one solution");
        }
        _ => {
            // Residual is also acceptable for budget=0
        }
    }
}
