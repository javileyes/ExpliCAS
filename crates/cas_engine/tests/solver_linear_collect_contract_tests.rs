//! Regression tests for LinearCollectStrategy.
//!
//! These tests verify that the solver correctly handles equations where the
//! target variable appears in multiple additive terms, requiring factorization.
//!
//! Example: A = P + P*r*t → P = A / (1 + r*t)

use cas_ast::{Equation, Expr, RelOp, SolutionSet};
use cas_engine::solver::{contains_var, solve};
use cas_engine::Simplifier;

/// Helper to create A = P + P*r*t equation and run solver
fn solve_interest_equation() -> (SolutionSet, Vec<cas_engine::solver::SolveStep>, Simplifier) {
    let mut simplifier = Simplifier::new();
    let ctx = &mut simplifier.context;

    let a = ctx.var("A");
    let p = ctx.var("P");
    let r = ctx.var("r");
    let t = ctx.var("t");

    // P*r*t
    let rt = ctx.add(Expr::Mul(r, t));
    let prt = ctx.add(Expr::Mul(p, rt));

    // P + P*r*t
    let rhs = ctx.add(Expr::Add(p, prt));

    let eq = Equation {
        lhs: a,
        rhs,
        op: RelOp::Eq,
    };

    // Debug: print equation structure
    eprintln!(
        "Equation: {} = {}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: eq.lhs
        },
        cas_formatter::DisplayExpr {
            context: ctx,
            id: eq.rhs
        }
    );
    eprintln!("RHS structure: {:?}", ctx.get(rhs));

    let result = solve(&eq, "P", &mut simplifier).expect("Solver should succeed");
    (result.0, result.1, simplifier)
}

/// CONTRACT TEST: No circular solutions
///
/// A solution like "P = A - P*r*t" is NOT valid because P appears on RHS.
/// This was the original bug that LinearCollect was designed to fix.
#[test]
fn linear_collect_no_circular_solution() {
    let (solution_set, _steps, simplifier) = solve_interest_equation();

    // Debug: print what we got
    eprintln!("Solution set: {:?}", solution_set);

    // Extract solutions and verify none contain P
    match &solution_set {
        SolutionSet::Discrete(sols) => {
            for sol in sols {
                assert!(
                    !contains_var(&simplifier.context, *sol, "P"),
                    "Solution must not contain target variable P (circular solution bug)"
                );
            }
        }
        SolutionSet::Conditional(cases) => {
            for case in cases {
                // case.then is Box<SolveResult> which is a struct with .solutions field
                let solve_result = case.then.as_ref();
                if let SolutionSet::Discrete(sols) = &solve_result.solutions {
                    for sol in sols {
                        assert!(
                            !contains_var(&simplifier.context, *sol, "P"),
                            "Conditional solution must not contain target variable P"
                        );
                    }
                }
            }
        }
        SolutionSet::Residual(residual_expr) => {
            // Display the residual to understand what happened
            let residual_str = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id: *residual_expr,
                }
            );
            panic!(
                "Should not return Residual - LinearCollect should handle this. Residual: {}",
                residual_str
            );
        }
        _ => {} // AllReals, Empty are okay
    }
}

/// CONTRACT TEST: Correct result is A/(1+r*t)
///
/// The solution must be A / (1 + r*t), not -(A/(1+r*t)) or any other variant.
#[test]
fn linear_collect_correct_solution() {
    let (solution_set, _, simplifier) = solve_interest_equation();

    // Should be Conditional with primary case
    let SolutionSet::Conditional(cases) = solution_set else {
        panic!("Expected Conditional solution set");
    };

    assert!(
        cases.len() >= 2,
        "Should have at least 2 cases (main + degenerate)"
    );

    // First case should be the main solution
    let main_case = &cases[0];
    let solve_result = main_case.then.as_ref();

    let SolutionSet::Discrete(sols) = &solve_result.solutions else {
        panic!("First case should have Discrete solutions");
    };

    assert_eq!(sols.len(), 1, "Should have exactly one solution");

    // Verify it's A/(1+r*t) by checking structure
    let sol = sols[0];
    let sol_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: sol,
        }
    );

    // The solution should contain "A" in numerator and "(1 + r·t)" in denominator
    assert!(
        sol_str.contains("A") && (sol_str.contains("/") || sol_str.contains("÷")),
        "Solution should be A divided by something, got: {}",
        sol_str
    );

    // Verify it does NOT start with negative sign (no sign error)
    assert!(
        !sol_str.starts_with("-") && !sol_str.starts_with("-("),
        "Solution should not be negated, got: {}",
        sol_str
    );
}

/// CONTRACT TEST: Degenerate case handling
///
/// When coeff = 0 (i.e., 1 + r*t = 0) AND A = 0, solution should be AllReals.
#[test]
fn linear_collect_degenerate_all_reals() {
    let (solution_set, _, _simplifier) = solve_interest_equation();

    let SolutionSet::Conditional(cases) = solution_set else {
        panic!("Expected Conditional");
    };

    // Should have an AllReals case for coeff=0 ∧ A=0
    let has_all_reals = cases.iter().any(|case| {
        let solve_result = case.then.as_ref();
        matches!(solve_result.solutions, SolutionSet::AllReals)
    });

    assert!(
        has_all_reals,
        "Should have AllReals case for degenerate coeff=0 ∧ A=0"
    );
}

/// CONTRACT TEST: Step flow is pedagogically clean
///
/// The steps should NOT include a "Subtract P*r*t from both sides" step
/// that creates a circular equation. Instead, should go directly to factoring.
#[test]
fn linear_collect_clean_step_flow() {
    // Use with_default_rules to enable step collection
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);

    let ctx = &mut simplifier.context;
    let a = ctx.var("A");
    let p = ctx.var("P");
    let r = ctx.var("r");
    let t = ctx.var("t");
    let rt = ctx.add(Expr::Mul(r, t));
    let prt = ctx.add(Expr::Mul(p, rt));
    let rhs = ctx.add(Expr::Add(p, prt));
    let eq = Equation {
        lhs: a,
        rhs,
        op: RelOp::Eq,
    };

    let result = solve(&eq, "P", &mut simplifier);
    assert!(result.is_ok());

    let (_solution_set, steps) = result.unwrap();

    // Check that we don't have the circular subtract step
    for step in &steps {
        // If there's a subtract step, it should not be subtracting a term with P
        if step.description.to_lowercase().contains("subtract") {
            assert!(
                !step.description.contains("P·r·t") && !step.description.contains("P*r*t"),
                "Should not have 'Subtract P*r*t' step that creates circular equation. Step: {}",
                step.description
            );
        }
    }

    // Should have a "Collect terms" or "factor" step
    let has_factor_step = steps.iter().any(|step| {
        step.description.to_lowercase().contains("collect")
            || step.description.to_lowercase().contains("factor")
    });

    assert!(
        has_factor_step,
        "Should have a 'Collect/Factor' step in the solution"
    );
}
