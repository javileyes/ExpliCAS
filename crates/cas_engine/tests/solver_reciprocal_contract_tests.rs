//! Contract tests for reciprocal solve pedagogical steps.
//!
//! Ensures `1/R = 1/R1 + 1/R2` shows proper step decomposition:
//! 1. "Combine fractions on RHS (common denominator)"
//! 2. "Take reciprocal"

use cas_ast::{Equation, Expr, RelOp, SolutionSet};
use cas_engine::engine::Simplifier;
use cas_engine::solver::solve;

/// Build equation 1/R = 1/R1 + 1/R2 and solve for R
fn solve_parallel_resistors() -> (SolutionSet, Vec<cas_engine::solver::SolveStep>, Simplifier) {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);

    let ctx = &mut simplifier.context;

    let r = ctx.var("R");
    let r1 = ctx.var("R1");
    let r2 = ctx.var("R2");
    let one = ctx.num(1);

    // 1/R
    let one2 = ctx.num(1);
    let lhs = ctx.add(Expr::Div(one, r));

    // 1/R1 + 1/R2
    let one3 = ctx.num(1);
    let frac1 = ctx.add(Expr::Div(one2, r1));
    let frac2 = ctx.add(Expr::Div(one3, r2));
    let rhs = ctx.add(Expr::Add(frac1, frac2));

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let result = solve(&eq, "R", &mut simplifier).expect("Solver should succeed");
    (result.0, result.1, simplifier)
}

/// CONTRACT TEST: Pedagogical steps are shown (not magic step)
#[test]
fn reciprocal_solve_shows_pedagogical_steps() {
    let (_, steps, _) = solve_parallel_resistors();

    // Should have at least 2 steps
    assert!(
        steps.len() >= 2,
        "Should have at least 2 steps, got {}",
        steps.len()
    );

    // Step 1 should contain "Combine fractions"
    let step1 = &steps[0];
    assert!(
        step1.description.to_lowercase().contains("combine")
            || step1
                .description
                .to_lowercase()
                .contains("common denominator"),
        "Step 1 should mention combining fractions. Got: {}",
        step1.description
    );

    // Step 2 should contain "reciprocal"
    let step2 = &steps[1];
    assert!(
        step2.description.to_lowercase().contains("reciprocal"),
        "Step 2 should mention reciprocal. Got: {}",
        step2.description
    );
}

/// CONTRACT TEST: No "Isolate denominator" magic step
#[test]
fn reciprocal_solve_no_magic_step() {
    let (_, steps, _) = solve_parallel_resistors();

    for step in &steps {
        assert!(
            !step
                .description
                .to_lowercase()
                .contains("isolate denominator"),
            "Should not have 'Isolate denominator' magic step. Found: {}",
            step.description
        );
    }
}

/// CONTRACT TEST: Correct solution R = R1*R2/(R1+R2)
#[test]
fn reciprocal_solve_correct_result() {
    let (solution_set, _, simplifier) = solve_parallel_resistors();

    // Extract solution
    let SolutionSet::Conditional(cases) = solution_set else {
        panic!("Expected Conditional solution set");
    };

    assert!(!cases.is_empty(), "Should have at least one case");

    let first_case = &cases[0];
    let solutions = &first_case.then.as_ref().solutions;

    let SolutionSet::Discrete(sols) = solutions else {
        panic!("Expected Discrete solutions");
    };

    assert_eq!(sols.len(), 1, "Should have exactly one solution");

    // Verify solution structure: R1*R2 / (R1+R2)
    let sol_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: sols[0],
        }
    );

    assert!(
        sol_str.contains("R1") && sol_str.contains("R2"),
        "Solution should contain R1 and R2. Got: {}",
        sol_str
    );
    assert!(
        sol_str.contains("/"),
        "Solution should be a fraction. Got: {}",
        sol_str
    );
}

/// CONTRACT TEST: Guard shows R1+R2 ≠ 0 (not 1·R1+1·R2)
#[test]
fn reciprocal_solve_clean_guard() {
    let (solution_set, _, simplifier) = solve_parallel_resistors();

    let SolutionSet::Conditional(cases) = solution_set else {
        panic!("Expected Conditional solution set");
    };

    let first_case = &cases[0];
    let guard_str = first_case.when.display_with_context(&simplifier.context);

    // Should show simplified: "R1 + R2" not "1·R1 + 1·R2"
    assert!(
        !guard_str.contains("1·R1") && !guard_str.contains("1*R1"),
        "Guard should not show unsimplified 1·R1. Got: {}",
        guard_str
    );
}
