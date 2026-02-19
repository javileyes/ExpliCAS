//! Contract tests for LinearCollect pedagogical improvements.
//!
//! These tests ensure the solver produces clean pedagogical traces
//! without circular steps or unnecessary divisions.

use cas_ast::{Equation, Expr, RelOp, SolutionSet};
use cas_engine::solver::solve;
use cas_engine::Simplifier;

// ============================================================================
// Helper functions
// ============================================================================

fn step_descriptions(steps: &[cas_engine::solver::SolveStep]) -> Vec<String> {
    steps.iter().map(|s| s.description.clone()).collect()
}

// ============================================================================
// Test: y = k*x/(x+c) should NOT divide by k
// ============================================================================

/// Build and solve y = k*x/(x+c) for x
fn solve_michaelis_menten() -> (SolutionSet, Vec<cas_engine::solver::SolveStep>, Simplifier) {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);

    let ctx = &mut simplifier.context;
    let x = ctx.var("x");
    let y = ctx.var("y");
    let k = ctx.var("k");
    let c = ctx.var("c");

    // k*x / (x + c)
    let k_times_x = ctx.add(Expr::Mul(k, x));
    let x_plus_c = ctx.add(Expr::Add(x, c));
    let rhs = ctx.add(Expr::Div(k_times_x, x_plus_c));

    let eq = Equation {
        lhs: y,
        rhs,
        op: RelOp::Eq,
    };

    let result = solve(&eq, "x", &mut simplifier).expect("Solver should succeed");
    (result.0, result.1, simplifier)
}

/// CONTRACT: Should NOT have "Divide both sides by k" step
#[test]
fn michaelis_menten_no_divide_by_k() {
    let (_, steps, _) = solve_michaelis_menten();
    let descs = step_descriptions(&steps);

    // Should NOT have divide by k step (that creates circular equation)
    let has_divide_by_k = descs.iter().any(|d| d.contains("Divide both sides by k"));
    assert!(
        !has_divide_by_k,
        "Should NOT have 'Divide both sides by k' step. Steps: {:?}",
        descs
    );
}

/// CONTRACT: Should have "Collect terms" step AFTER multiply
#[test]
fn michaelis_menten_has_collect_step() {
    let (_, steps, _) = solve_michaelis_menten();
    let descs = step_descriptions(&steps);

    let has_collect = descs.iter().any(|d| d.contains("Collect terms"));
    assert!(
        has_collect,
        "Should have 'Collect terms' step. Steps: {:?}",
        descs
    );
}

/// CONTRACT: Result should be x = c*y/(k-y)
#[test]
fn michaelis_menten_correct_solution() {
    let (solution_set, _, simplifier) = solve_michaelis_menten();

    let SolutionSet::Conditional(cases) = solution_set else {
        panic!("Expected Conditional solution");
    };

    let primary = &cases[0];
    let SolutionSet::Discrete(sols) = &primary.then.as_ref().solutions else {
        panic!("Expected Discrete solution in primary case");
    };

    let sol_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: sols[0],
        }
    );

    // Should contain c, y, k but not x
    assert!(
        !sol_str.contains("x"),
        "Solution should not contain x: {}",
        sol_str
    );
    assert!(
        sol_str.contains("y"),
        "Solution should contain y: {}",
        sol_str
    );
    assert!(
        sol_str.contains("k"),
        "Solution should contain k: {}",
        sol_str
    );
    assert!(
        sol_str.contains("c"),
        "Solution should contain c: {}",
        sol_str
    );
}

// ============================================================================
// Test: P*V/T = n*R should have 2-step decomposition
// ============================================================================

fn solve_ideal_gas_for_t() -> (SolutionSet, Vec<cas_engine::solver::SolveStep>, Simplifier) {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);

    let ctx = &mut simplifier.context;
    let p = ctx.var("P");
    let v = ctx.var("V");
    let t = ctx.var("T");
    let n = ctx.var("n");
    let r = ctx.var("R");

    // P*V / T
    let pv = ctx.add(Expr::Mul(p, v));
    let lhs = ctx.add(Expr::Div(pv, t));

    // n*R
    let rhs = ctx.add(Expr::Mul(n, r));

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "T", &mut simplifier).expect("Solver should succeed");
    (result.0, result.1, simplifier)
}

/// CONTRACT: Should have "Multiply both sides by T" step
#[test]
fn ideal_gas_has_multiply_by_t() {
    let (_, steps, _) = solve_ideal_gas_for_t();
    let descs = step_descriptions(&steps);

    let has_multiply = descs.iter().any(|d| d.contains("Multiply both sides by T"));
    assert!(
        has_multiply,
        "Should have 'Multiply both sides by T'. Steps: {:?}",
        descs
    );
}

/// CONTRACT: Should have "Divide both sides by" step for R*n
#[test]
fn ideal_gas_has_divide_step() {
    let (_, steps, _) = solve_ideal_gas_for_t();
    let descs = step_descriptions(&steps);

    // Should have divide step (not just "Isolate denominator")
    let has_divide = descs.iter().any(|d| d.starts_with("Divide both sides by"));
    assert!(
        has_divide,
        "Should have 'Divide both sides by...' step. Steps: {:?}",
        descs
    );
}

/// CONTRACT: Should NOT have single "Isolate denominator" step
#[test]
fn ideal_gas_no_magic_isolate() {
    let (_, steps, _) = solve_ideal_gas_for_t();
    let descs = step_descriptions(&steps);

    let has_isolate = descs.iter().any(|d| d.contains("Isolate denominator"));
    assert!(
        !has_isolate,
        "Should NOT have 'Isolate denominator' magic step. Steps: {:?}",
        descs
    );
}

// ============================================================================
// Test: a*x + b*x = c should factor correctly
// ============================================================================

fn solve_combine_like_terms() -> (SolutionSet, Vec<cas_engine::solver::SolveStep>, Simplifier) {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);

    let ctx = &mut simplifier.context;
    let a = ctx.var("a");
    let b = ctx.var("b");
    let c = ctx.var("c");
    let x = ctx.var("x");

    // a*x + b*x
    let ax = ctx.add(Expr::Mul(a, x));
    let bx = ctx.add(Expr::Mul(b, x));
    let lhs = ctx.add(Expr::Add(ax, bx));

    let eq = Equation {
        lhs,
        rhs: c,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut simplifier).expect("Solver should succeed");
    (result.0, result.1, simplifier)
}

/// CONTRACT: Should produce x = c/(a+b)
#[test]
fn combine_like_terms_correct_result() {
    let (solution_set, _, simplifier) = solve_combine_like_terms();

    let SolutionSet::Conditional(cases) = solution_set else {
        panic!("Expected Conditional solution");
    };

    let primary = &cases[0];
    let SolutionSet::Discrete(sols) = &primary.then.as_ref().solutions else {
        panic!("Expected Discrete solution");
    };

    let sol_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: sols[0],
        }
    );

    // Result should be c/(a+b) - not contain x
    assert!(
        !sol_str.contains("x"),
        "Solution should not contain x: {}",
        sol_str
    );
    assert!(
        sol_str.contains("c"),
        "Solution should contain c: {}",
        sol_str
    );
}

/// CONTRACT: Should NOT return Residual
#[test]
fn combine_like_terms_no_residual() {
    let (solution_set, _, _) = solve_combine_like_terms();
    assert!(
        !matches!(solution_set, SolutionSet::Residual(_)),
        "Should not return Residual"
    );
}

// ============================================================================
// Test: (x-1)/(x+1) = y should work
// ============================================================================

fn solve_fractional_linear() -> (SolutionSet, Vec<cas_engine::solver::SolveStep>, Simplifier) {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);

    let ctx = &mut simplifier.context;
    let x = ctx.var("x");
    let y = ctx.var("y");
    let one = ctx.num(1);

    // (x-1)/(x+1)
    let x_minus_1 = ctx.add(Expr::Sub(x, one));
    let one2 = ctx.num(1);
    let x_plus_1 = ctx.add(Expr::Add(x, one2));
    let lhs = ctx.add(Expr::Div(x_minus_1, x_plus_1));

    let eq = Equation {
        lhs,
        rhs: y,
        op: RelOp::Eq,
    };
    let result = solve(&eq, "x", &mut simplifier).expect("Solver should succeed");
    (result.0, result.1, simplifier)
}

/// CONTRACT: Should NOT return Residual
#[test]
fn fractional_linear_no_residual() {
    let (solution_set, _, _) = solve_fractional_linear();
    assert!(
        !matches!(solution_set, SolutionSet::Residual(_)),
        "Should not return Residual"
    );
}

/// CONTRACT: Result should be (1+y)/(1-y) form
#[test]
fn fractional_linear_correct_form() {
    let (solution_set, _, simplifier) = solve_fractional_linear();

    let SolutionSet::Conditional(cases) = solution_set else {
        panic!("Expected Conditional solution");
    };

    let primary = &cases[0];
    let SolutionSet::Discrete(sols) = &primary.then.as_ref().solutions else {
        panic!("Expected Discrete solution");
    };

    let sol_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: sols[0],
        }
    );

    // Should contain y and be a fraction
    assert!(
        sol_str.contains("y"),
        "Solution should contain y: {}",
        sol_str
    );
    assert!(
        sol_str.contains("/"),
        "Solution should be a fraction: {}",
        sol_str
    );
    assert!(
        !sol_str.contains("x"),
        "Solution should not contain x: {}",
        sol_str
    );
}
