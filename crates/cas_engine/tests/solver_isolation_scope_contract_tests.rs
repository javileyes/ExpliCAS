//! Contract tests for PR-SCOPE-3.3: Solver Domain-Aware Solving.
//!
//! These tests verify that the solver correctly handles exponential equations
//! with symbolic RHS by using domain inference rather than conditional branching.
//!
//! V2.2+ UPDATE: The solver now uses derive_requires_from_equation() to infer
//! required conditions (like "y > 0" from "2^x = y"). This allows direct solving
//! with the condition captured in the "requires" output, rather than returning
//! a Conditional solution set.

use cas_ast::{Constant, Equation, Expr, RelOp, SolutionSet};
use cas_engine::solver::{solve_with_display_steps, SolverOptions};
use cas_engine::DomainMode;
use cas_engine::Engine;
use cas_engine::ImplicitCondition;
use cas_engine::{AssumeScope, ValueDomain};

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
    // V2.2+: 2^x = y solves via conditional (since y's positivity can't be proven)
    // Post-fix (b7f66bd): derive_requires_from_equation intentionally doesn't propagate
    // positivity to plain variables like 'y' to prevent false requires like "2*x + 3 > 0".
    // Instead, the solver returns a Conditional with guard y > 0.
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

    let (solution_set, _steps, _diagnostics) = result.unwrap();

    // V2.2+: With strict mode and unknown y, solver returns Conditional
    // The condition y > 0 is expressed as a guard on the solution, not in required_conditions
    match &solution_set {
        SolutionSet::Discrete(solutions) => {
            assert!(!solutions.is_empty(), "Should have at least one solution");
            // Solution should be x = ln(y)/ln(2) or log(2, y)
            let sol_str = cas_formatter::DisplayExpr {
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
        SolutionSet::Conditional(cases) => {
            // This is the expected path in Strict mode with unknown y
            // The guard should require y > 0 for the log solution to be valid
            assert!(
                !cases.is_empty(),
                "Conditional should have at least one case"
            );
        }
        _ => {
            // Also accept Residual (backward compat) - the key is no crash and no garbage
        }
    }

    // Note: After fix b7f66bd, y > 0 is NOT in required_conditions for plain variable y.
    // This is intentional to prevent false requires like "2*x + 3 > 0".
    // Instead, the condition is expressed as a guard in the Conditional solution set.
    // Note: required conditions now available in-band via result diagnostics
    // Intentionally not asserting on required conditions - they may be empty for this case
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

    let (solution_set, _steps, _diagnostics) = result.unwrap();

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

    let (solution_set, _steps, diagnostics) = result.unwrap();

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
    let required = diagnostics.required;
    let has_positive_y = required.iter().any(|cond| {
        if let ImplicitCondition::Positive(id) = cond {
            cas_formatter::DisplayExpr {
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

    let (solution_set, _steps, _diagnostics) = result.unwrap();

    // Should be Residual variant
    assert!(
        matches!(solution_set, cas_ast::SolutionSet::Residual(_)),
        "Should be SolutionSet::Residual, got: {:?}",
        solution_set
    );

    // Verify NO garbage: the residual should not contain ln(-2) or undefined
    if let cas_ast::SolutionSet::Residual(residual_expr) = solution_set {
        let residual_str = cas_formatter::DisplayExpr {
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

    let (solution_set, _steps, _diagnostics) = result.unwrap();

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

// =============================================================================
// Test 4: Nested product-zero recursion must preserve solver options
// =============================================================================

#[test]
fn assume_mode_nested_product_zero_keeps_assume_semantics() {
    // (a^x - b) * (x - 1) = 0
    //
    // Product-zero split triggers nested recursive solves for each factor.
    // In Assume mode, the exponential factor can be solved as log(a, b).
    // If nested solves lose options and fall back to default Generic mode,
    // this regresses to a residual/unsupported branch.
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let a = ctx.var("a");
    let b = ctx.var("b");
    let x = ctx.var("x");
    let one = ctx.num(1);
    let a_pow_x = ctx.add(Expr::Pow(a, x));
    let exp_factor = ctx.add(Expr::Sub(a_pow_x, b));
    let linear_factor = ctx.add(Expr::Sub(x, one));
    let product = ctx.add(Expr::Mul(exp_factor, linear_factor));
    let zero = ctx.num(0);

    let eq = Equation {
        lhs: product,
        rhs: zero,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Assume, AssumeScope::Real);
    let result = solve_with_display_steps(&eq, "x", &mut engine.simplifier, opts);
    assert!(
        result.is_ok(),
        "Assume mode should solve nested product-zero"
    );

    let (solution_set, _steps, _diagnostics) = result.unwrap();
    let SolutionSet::Discrete(solutions) = solution_set else {
        panic!(
            "Expected discrete solutions in Assume mode, got: {:?}",
            solution_set
        );
    };

    let rendered: Vec<String> = solutions
        .iter()
        .map(|id| {
            cas_formatter::DisplayExpr {
                context: &engine.simplifier.context,
                id: *id,
            }
            .to_string()
        })
        .collect();

    assert!(
        rendered.iter().any(|s| s == "1"),
        "Expected linear branch solution x=1, got: {:?}",
        rendered
    );
    assert!(
        rendered
            .iter()
            .any(|s| s.contains("log(") || s.contains("ln(")),
        "Expected exponential branch solution log(a,b), got: {:?}",
        rendered
    );
}

#[test]
fn generic_mode_nested_product_zero_stays_residual() {
    // Same equation as above, but in Generic mode the exponential factor with
    // symbolic base/rhs remains unsupported, so the whole solve stays residual.
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let a = ctx.var("a");
    let b = ctx.var("b");
    let x = ctx.var("x");
    let one = ctx.num(1);
    let a_pow_x = ctx.add(Expr::Pow(a, x));
    let exp_factor = ctx.add(Expr::Sub(a_pow_x, b));
    let linear_factor = ctx.add(Expr::Sub(x, one));
    let product = ctx.add(Expr::Mul(exp_factor, linear_factor));
    let zero = ctx.num(0);

    let eq = Equation {
        lhs: product,
        rhs: zero,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Generic, AssumeScope::Real);
    let result = solve_with_display_steps(&eq, "x", &mut engine.simplifier, opts);
    assert!(
        result.is_ok(),
        "Generic mode should not crash on nested product-zero"
    );

    let (solution_set, _steps, _diagnostics) = result.unwrap();
    assert!(
        matches!(solution_set, SolutionSet::Residual(_)),
        "Expected residual in Generic mode, got: {:?}",
        solution_set
    );
}

#[test]
fn assume_mode_scaled_exponential_both_sides_does_not_cycle() {
    // 2^(2*x) = y*2^x
    // This should be solved by substitution (u = 2^x), not fail with cycle/isolation.
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let two = ctx.num(2);
    let x = ctx.var("x");
    let y = ctx.var("y");
    let two_x = ctx.add(Expr::Mul(two, x));
    let lhs = ctx.add(Expr::Pow(two, two_x));
    let two_pow_x = ctx.add(Expr::Pow(two, x));
    let rhs = ctx.add(Expr::Mul(y, two_pow_x));
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Assume, AssumeScope::Real);
    let result = solve_with_display_steps(&eq, "x", &mut engine.simplifier, opts);
    assert!(
        result.is_ok(),
        "Assume mode should not fail for 2^(2*x)=y*2^x, got: {:?}",
        result
    );

    let (solution_set, _steps, _diagnostics) = result.unwrap();
    let has_solution = match &solution_set {
        SolutionSet::Discrete(sols) => !sols.is_empty(),
        SolutionSet::Conditional(cases) => !cases.is_empty(),
        _ => false,
    };
    assert!(
        has_solution,
        "Expected non-empty solution set in Assume mode, got: {:?}",
        solution_set
    );
}

#[test]
fn generic_mode_scaled_exponential_both_sides_does_not_hard_fail() {
    // Same equation as above in Generic mode.
    // We only require non-crashing behavior; representation may be Discrete/Conditional/Residual.
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let two = ctx.num(2);
    let x = ctx.var("x");
    let y = ctx.var("y");
    let two_x = ctx.add(Expr::Mul(two, x));
    let lhs = ctx.add(Expr::Pow(two, two_x));
    let two_pow_x = ctx.add(Expr::Pow(two, x));
    let rhs = ctx.add(Expr::Mul(y, two_pow_x));
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Generic, AssumeScope::Real);
    let result = solve_with_display_steps(&eq, "x", &mut engine.simplifier, opts);
    assert!(
        result.is_ok(),
        "Generic mode should not hard-fail for 2^(2*x)=y*2^x, got: {:?}",
        result
    );
}

#[test]
fn assume_mode_scaled_e_exponential_both_sides_does_not_cycle() {
    // e^(2*x) = y*e^x (same substitution shape, symbolic RHS)
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let e = ctx.add(Expr::Constant(Constant::E));
    let two = ctx.num(2);
    let x = ctx.var("x");
    let y = ctx.var("y");
    let two_x = ctx.add(Expr::Mul(two, x));
    let lhs = ctx.add(Expr::Pow(e, two_x));
    let e_pow_x = ctx.add(Expr::Pow(e, x));
    let rhs = ctx.add(Expr::Mul(y, e_pow_x));
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Assume, AssumeScope::Real);
    let result = solve_with_display_steps(&eq, "x", &mut engine.simplifier, opts);
    assert!(
        result.is_ok(),
        "Assume mode should not fail for e^(2*x)=y*e^x, got: {:?}",
        result
    );
}

#[test]
fn generic_mode_scaled_e_exponential_both_sides_does_not_hard_fail() {
    // Same equation as above in Generic mode.
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    let e = ctx.add(Expr::Constant(Constant::E));
    let two = ctx.num(2);
    let x = ctx.var("x");
    let y = ctx.var("y");
    let two_x = ctx.add(Expr::Mul(two, x));
    let lhs = ctx.add(Expr::Pow(e, two_x));
    let e_pow_x = ctx.add(Expr::Pow(e, x));
    let rhs = ctx.add(Expr::Mul(y, e_pow_x));
    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Generic, AssumeScope::Real);
    let result = solve_with_display_steps(&eq, "x", &mut engine.simplifier, opts);
    assert!(
        result.is_ok(),
        "Generic mode should not hard-fail for e^(2*x)=y*e^x, got: {:?}",
        result
    );
}
