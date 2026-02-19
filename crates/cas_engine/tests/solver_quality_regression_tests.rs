//! Non-regression tests for solver improvements (2026-01-07).
//!
//! These tests ensure solver output quality improvements are preserved:
//! 1. Absolute value equations have no spurious Requires
//! 2. Quadratic discriminants are simplified (GCD extraction)
//! 3. Cubic roots don't cause simplification cycles

use cas_ast::{Equation, Expr, RelOp, SolutionSet};
use cas_engine::solver::{solve, solve_with_display_steps, SolverOptions};
use cas_engine::Simplifier;

// ============================================================================
// Test 1: |2x+1| = 5 should have NO Requires
// ============================================================================

#[test]
fn abs_equation_no_spurious_requires() {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);
    let ctx = &mut simplifier.context;

    // Build |2x + 1| = 5
    let x = ctx.var("x");
    let two = ctx.num(2);
    let one = ctx.num(1);
    let five = ctx.num(5);

    let two_x = ctx.add(Expr::Mul(two, x));
    let two_x_plus_1 = ctx.add(Expr::Add(two_x, one));
    let abs_expr = ctx.call_builtin(cas_ast::BuiltinFn::Abs, vec![two_x_plus_1]);

    let eq = Equation {
        lhs: abs_expr,
        rhs: five,
        op: RelOp::Eq,
    };

    // Use solve_with_display_steps to get diagnostics in-band
    let opts = SolverOptions::default();
    let (solution_set, _steps, diagnostics) =
        solve_with_display_steps(&eq, "x", &mut simplifier, opts).expect("Solver should succeed");

    // Should have 2 solutions: {2, -3}
    let SolutionSet::Discrete(sols) = solution_set else {
        panic!("Expected Discrete solution set");
    };
    assert_eq!(sols.len(), 2, "Should have exactly 2 solutions");

    // Importantly: there should be NO required conditions from the solver
    // (The spurious |1+2x| > 0 should not appear)
    let required = diagnostics.required;
    for cond in &required {
        let cond_str = cond.display(&simplifier.context);
        // Should NOT contain "abs" or "|" in any require
        assert!(
            !cond_str.contains("abs") && !cond_str.contains("|"),
            "Absolute value equation should not produce requires with abs. Got: {}",
            cond_str
        );
    }
}

// ============================================================================
// Test 2: c² = a² + b² solved for a should have simplified discriminant
// ============================================================================

#[test]
fn pythagorean_theorem_simplified_discriminant() {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);
    let ctx = &mut simplifier.context;

    // Build c² = a² + b²
    let a = ctx.var("a");
    let b = ctx.var("b");
    let c = ctx.var("c");
    let two = ctx.num(2);

    let a2 = ctx.add(Expr::Pow(a, two));
    let b2 = ctx.add(Expr::Pow(b, two));
    let c2 = ctx.add(Expr::Pow(c, two));
    let a2_plus_b2 = ctx.add(Expr::Add(a2, b2));

    let eq = Equation {
        lhs: c2,
        rhs: a2_plus_b2,
        op: RelOp::Eq,
    };

    let (solution_set, _) = solve(&eq, "a", &mut simplifier).expect("Solver should succeed");

    let SolutionSet::Discrete(sols) = solution_set else {
        panic!("Expected Discrete solution set");
    };
    assert_eq!(sols.len(), 2, "Should have 2 solutions (±√(c²-b²))");

    // Check solutions are simplified - should NOT contain "4*c^2" or "4*b^2"
    for sol in &sols {
        let sol_str = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: *sol,
            }
        );

        // Should NOT contain unsimplified "4·c" or "4·b" (the GCD should be extracted)
        assert!(
            !sol_str.contains("4·c")
                && !sol_str.contains("4·b")
                && !sol_str.contains("4 * c")
                && !sol_str.contains("4 * b"),
            "Solution should not contain GCD factor 4. Got: {}",
            sol_str
        );
    }
}

// ============================================================================
// Test 3: x² - 4x + 4 = y should simplify to 2 ± √y
// ============================================================================

#[test]
fn perfect_square_quadratic_simplified() {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);
    let ctx = &mut simplifier.context;

    // Build x² - 4x + 4 = y
    let x = ctx.var("x");
    let y = ctx.var("y");
    let two = ctx.num(2);
    let four = ctx.num(4);

    let x2 = ctx.add(Expr::Pow(x, two));
    let four_x = ctx.add(Expr::Mul(four, x));
    let four_const = ctx.num(4);
    let x2_minus_4x = ctx.add(Expr::Sub(x2, four_x));
    let lhs = ctx.add(Expr::Add(x2_minus_4x, four_const));

    let eq = Equation {
        lhs,
        rhs: y,
        op: RelOp::Eq,
    };

    let (solution_set, _) = solve(&eq, "x", &mut simplifier).expect("Solver should succeed");

    let SolutionSet::Discrete(sols) = solution_set else {
        panic!("Expected Discrete solution set");
    };
    assert_eq!(sols.len(), 2, "Should have 2 solutions");

    // Check that solutions contain "2" (the vertex constant)
    let mut has_two = false;
    for sol in &sols {
        let sol_str = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: *sol,
            }
        );
        if sol_str.contains("2") {
            has_two = true;
        }

        // Should NOT contain the unsimplified form "1/2·(4"
        assert!(
            !sol_str.contains("1/2·(4") && !sol_str.contains("16 + 4"),
            "Solution should be simplified. Got: {}",
            sol_str
        );
    }
    assert!(
        has_two,
        "At least one solution should contain the constant 2"
    );
}

// ============================================================================
// Test 4: V = (4/3)*pi*r³ should NOT cause depth_overflow or cycles
// ============================================================================

#[test]
fn sphere_volume_no_cycle() {
    let mut simplifier = Simplifier::with_default_rules();
    simplifier.set_collect_steps(true);
    let ctx = &mut simplifier.context;

    // Build V = (4/3) * pi * r³
    let r = ctx.var("r");
    let v = ctx.var("V");
    let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
    let three = ctx.num(3);
    let four = ctx.num(4);

    let four_thirds = ctx.add(Expr::Div(four, three));
    let r3 = ctx.add(Expr::Pow(r, three));
    let pi_r3 = ctx.add(Expr::Mul(pi, r3));
    let rhs = ctx.add(Expr::Mul(four_thirds, pi_r3));

    let eq = Equation {
        lhs: v,
        rhs,
        op: RelOp::Eq,
    };

    // This should complete without timeout or cycle
    let result = solve(&eq, "r", &mut simplifier);

    assert!(result.is_ok(), "Solver should complete without error");

    let (solution_set, _) = result.unwrap();

    // Should have exactly 1 solution (cube root, not quadratic)
    let SolutionSet::Discrete(sols) = solution_set else {
        panic!("Expected Discrete solution set");
    };
    assert_eq!(sols.len(), 1, "Should have 1 solution for r");

    // Check solution contains cube root (1/3 exponent)
    let sol_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: sols[0],
        }
    );

    assert!(
        sol_str.contains("1/3") || sol_str.contains("(1/3)"),
        "Solution should contain cube root (1/3). Got: {}",
        sol_str
    );

    // Should NOT contain problematic patterns from cycles
    assert!(
        !sol_str.contains("pi^(1/3)·V^(1/3)") && !sol_str.contains("pi / pi"),
        "Solution should not contain cycle remnants. Got: {}",
        sol_str
    );
}
