//! Contract tests for Solver Requires Propagation (V2.2+)
//!
//! These tests verify that:
//! 1. `derive_requires_from_equation` correctly infers `y > 0` from `2^x = sqrt(y)`
//! 2. The solver does NOT create Conditional branches when conditions are in env.required
//! 3. Required conditions propagate correctly to the output

use cas_ast::{Equation, Expr, RelOp, SolutionSet};
use cas_engine::domain::DomainMode;
use cas_engine::semantics::ValueDomain;
use cas_engine::solver::{
    solve_with_display_steps, take_solver_required, SolveBudget, SolverOptions,
};
use cas_engine::Simplifier;

fn make_solver_opts(mode: DomainMode) -> SolverOptions {
    SolverOptions {
        value_domain: ValueDomain::RealOnly,
        domain_mode: mode,
        assume_scope: cas_engine::semantics::AssumeScope::Real,
        budget: SolveBudget::default(),
        ..Default::default()
    }
}

/// Test: 2^x = sqrt(y) should NOT produce Conditional in Generic mode
/// because derive_requires_from_equation infers y > 0 from 2^x > 0.
#[test]
fn sqrt_rhs_avoids_conditional_branch() {
    let mut simplifier = Simplifier::with_default_rules();
    let ctx = &mut simplifier.context;

    // Build: 2^x = sqrt(y)
    let two = ctx.num(2);
    let x = ctx.var("x");
    let y = ctx.var("y");
    let lhs = ctx.add(Expr::Pow(two, x)); // 2^x
    let rhs = ctx.call("sqrt", vec![y]); // sqrt(y)

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let opts = make_solver_opts(DomainMode::Generic);
    let result = solve_with_display_steps(&eq, "x", &mut simplifier, opts);

    assert!(result.is_ok(), "Solver should succeed");
    let (solution_set, _steps) = result.unwrap();

    // Key assertion: NOT Conditional (because y > 0 derived from structure)
    assert!(
        !matches!(solution_set, SolutionSet::Conditional(_)),
        "Expected direct solution (not Conditional) because sqrt(y) implies y > 0 which is derived from 2^x > 0"
    );

    // Verify required conditions were collected
    let required = take_solver_required();
    assert!(
        !required.is_empty(),
        "Should have required conditions from sqrt(y)"
    );

    // Check that y > 0 is in the required set (as Positive(y))
    let has_y_positive = required.iter().any(|c| {
        matches!(c, cas_engine::implicit_domain::ImplicitCondition::Positive(e)
            if matches!(simplifier.context.get(*e), Expr::Variable(sym_id) if simplifier.context.sym_name(*sym_id) == "y"))
    });
    assert!(
        has_y_positive,
        "Required conditions should include y > 0 (Positive(y))"
    );
}

/// Test: 2^x = y (without sqrt) behavior
/// In Generic mode, this should either be Conditional OR direct with required y > 0
#[test]
fn plain_y_rhs_requires_or_conditional() {
    let mut simplifier = Simplifier::with_default_rules();
    let ctx = &mut simplifier.context;

    // Build: 2^x = y
    let two = ctx.num(2);
    let x = ctx.var("x");
    let y = ctx.var("y");
    let lhs = ctx.add(Expr::Pow(two, x)); // 2^x
    let rhs = y;

    let eq = Equation {
        lhs,
        rhs,
        op: RelOp::Eq,
    };

    let opts = make_solver_opts(DomainMode::Generic);
    let result = solve_with_display_steps(&eq, "x", &mut simplifier, opts);

    // The solver may succeed or fail depending on mode
    // In Generic mode without proof of y > 0, it should be Conditional or Unsupported
    if let Ok((solution_set, _steps)) = result {
        // If it solved, check that:
        // - Either it's Conditional (because can't prove y > 0)
        // - Or the required set contains y > 0 (from ln(y) in solution)
        let required = take_solver_required();

        let is_conditional = matches!(solution_set, SolutionSet::Conditional(_));
        let has_y_positive = required.iter().any(|c| {
            matches!(c, cas_engine::implicit_domain::ImplicitCondition::Positive(e)
                if matches!(simplifier.context.get(*e), Expr::Variable(sym_id) if simplifier.context.sym_name(*sym_id) == "y"))
        });

        assert!(
            is_conditional || has_y_positive,
            "Plain y RHS should either be Conditional or have y > 0 in required (got neither)"
        );
    }
    // If it errored (Unsupported), that's also acceptable for Generic mode
}

/// Test: is_trivial correctly identifies constant conditions
#[test]
fn is_trivial_filters_constants() {
    use cas_engine::implicit_domain::ImplicitCondition;

    let mut simplifier = Simplifier::with_default_rules();
    let ctx = &mut simplifier.context;

    let two = ctx.num(2);
    let y = ctx.var("y");

    let trivial = ImplicitCondition::Positive(two);
    let non_trivial = ImplicitCondition::Positive(y);

    assert!(trivial.is_trivial(ctx), "Positive(2) should be trivial");
    assert!(
        !non_trivial.is_trivial(ctx),
        "Positive(y) should not be trivial"
    );
}
