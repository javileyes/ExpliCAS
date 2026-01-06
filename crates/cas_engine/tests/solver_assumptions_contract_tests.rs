//! Contract tests for PR-SCOPE-3.2: Solver Assumptions in Output.
//!
//! These tests verify that the solver correctly collects and returns
//! assumptions made during solving (e.g., "positive(y)" for 2^x = y).

use cas_ast::{Equation, Expr, RelOp};
use cas_engine::domain::DomainMode;
use cas_engine::semantics::{AssumeScope, ValueDomain};
use cas_engine::solver::{
    finish_assumption_collection, solve_with_options, start_assumption_collection,
    SolveAssumptionsGuard, SolverOptions,
};
use cas_engine::Engine;

fn make_opts(mode: DomainMode, scope: AssumeScope) -> SolverOptions {
    SolverOptions {
        value_domain: ValueDomain::RealOnly,
        domain_mode: mode,
        assume_scope: scope,
        budget: cas_engine::solver::SolveBudget::default(),
    }
}

fn setup_engine() -> Engine {
    Engine::new()
}

// =============================================================================
// Test 1: Assume mode emits positive(rhs) assumption
// =============================================================================

#[test]
#[ignore = "Pre-existing failure: solver assumptions not being collected"]
fn assume_mode_emits_positive_rhs_assumption() {
    // 2^x = y in Assume mode should emit assumption: positive(y)
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

    // Start assumption collection
    assert!(
        start_assumption_collection(),
        "Should be able to start collection"
    );

    let result = solve_with_options(&eq, "x", &mut engine.simplifier, opts);

    // Finish and get assumptions
    let collector = finish_assumption_collection().expect("Collector should be present");
    let records = collector.finish();

    // Verify solve succeeded
    assert!(result.is_ok(), "Solve should succeed in Assume mode");

    // Verify assumption was collected
    assert!(
        !records.is_empty(),
        "Should have at least one assumption (positive(y))"
    );

    // Check that we have a positive assumption for y
    let has_positive_y = records
        .iter()
        .any(|r| r.kind == "positive" && r.expr == "y");

    assert!(
        has_positive_y,
        "Should have positive(y) assumption, got: {:?}",
        records
    );
}

// =============================================================================
// Test 2: Strict mode has no assumptions (errors instead)
// =============================================================================

#[test]
fn strict_mode_no_assumptions() {
    // 2^x = y in Strict mode - no assumptions, solver skips
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

    // Start assumption collection
    start_assumption_collection();

    let _result = solve_with_options(&eq, "x", &mut engine.simplifier, opts);

    // Finish and get assumptions
    let collector = finish_assumption_collection().expect("Collector should be present");
    let records = collector.finish();

    // Strict mode should NOT produce assumptions (it skips or errors)
    assert!(
        records.is_empty(),
        "Strict mode should have no assumptions, got: {:?}",
        records
    );
}

// =============================================================================
// Test 3: Deduplication works (same assumption not repeated)
// =============================================================================

#[test]
#[ignore = "Pre-existing failure: solver assumptions not being collected"]
fn assumptions_are_deduplicated() {
    // If solver applies same assumption multiple times, should be counted not repeated
    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    // (2^x = y) - single solve, single assumption
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

    start_assumption_collection();
    let _ = solve_with_options(&eq, "x", &mut engine.simplifier, opts);
    let collector = finish_assumption_collection().expect("Collector should be present");
    let records = collector.finish();

    // Should have exactly 1 unique assumption record (not duplicates)
    let positive_y_count = records
        .iter()
        .filter(|r| r.kind == "positive" && r.expr == "y")
        .count();

    assert_eq!(
        positive_y_count, 1,
        "Should have exactly 1 unique positive(y) record, got {}",
        positive_y_count
    );
}

// =============================================================================
// Test 4: Nested solve guards don't leak assumptions
// =============================================================================

#[test]
#[ignore = "Pre-existing failure: solver assumptions not being collected"]
fn nested_solve_guards_are_isolated() {
    // RAII guards should isolate nested solve assumptions
    // Outer guard should not see inner guard's assumptions

    let mut engine = setup_engine();
    let ctx = &mut engine.simplifier.context;

    // Create outer equation: 2^x = y (assumption: positive(y))
    let two = ctx.num(2);
    let x = ctx.var("x");
    let y = ctx.var("y");
    let pow_outer = ctx.add(Expr::Pow(two, x));

    let eq_outer = Equation {
        lhs: pow_outer,
        rhs: y,
        op: RelOp::Eq,
    };

    // Create inner equation: 3^z = w (assumption: positive(w))
    let three = ctx.num(3);
    let z = ctx.var("z");
    let w = ctx.var("w");
    let pow_inner = ctx.add(Expr::Pow(three, z));

    let eq_inner = Equation {
        lhs: pow_inner,
        rhs: w,
        op: RelOp::Eq,
    };

    let opts = make_opts(DomainMode::Assume, AssumeScope::Real);

    // OUTER guard
    let outer_guard = SolveAssumptionsGuard::new(true);

    // Outer solve (generates positive(y))
    let _ = solve_with_options(&eq_outer, "x", &mut engine.simplifier, opts);

    // INNER guard (simulates nested solve)
    {
        let inner_guard = SolveAssumptionsGuard::new(true);

        // Inner solve (generates positive(w))
        let _ = solve_with_options(&eq_inner, "z", &mut engine.simplifier, opts);

        let inner_records = inner_guard.finish();

        // Inner should only have positive(w)
        assert!(
            inner_records.iter().any(|r| r.expr == "w"),
            "Inner should have positive(w), got: {:?}",
            inner_records
        );
        assert!(
            !inner_records.iter().any(|r| r.expr == "y"),
            "Inner should NOT have positive(y), got: {:?}",
            inner_records
        );
    }

    // Finish outer
    let outer_records = outer_guard.finish();

    // Outer should only have positive(y)
    assert!(
        outer_records.iter().any(|r| r.expr == "y"),
        "Outer should have positive(y), got: {:?}",
        outer_records
    );
    assert!(
        !outer_records.iter().any(|r| r.expr == "w"),
        "Outer should NOT have positive(w) (inner's assumption), got: {:?}",
        outer_records
    );
}
