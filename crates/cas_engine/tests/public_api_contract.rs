//! V2.1 Issue #4: Public API Contract Test
//!
//! This test ensures the stable API remains accessible.
//! If this test fails to compile, the public API contract has been broken.

use cas_engine::api::{
    // Functions
    solve,
    solve_with_options,
    // Domain types
    BoundType,
    Case,
    ConditionPredicate,
    ConditionSet,
    // Display types
    DisplayExpr,
    Interval,
    LaTeXExpr,
    SolutionSet,
    // Solver types
    SolveBudget,
    SolveResult,
    SolverOptions,
};

/// Test that SolutionSet variants are accessible
#[test]
fn test_solution_set_variants_accessible() {
    // Empty
    let _empty = SolutionSet::Empty;

    // AllReals
    let _all = SolutionSet::AllReals;

    // Discrete (would need ExprId, just check pattern)
    if let SolutionSet::Discrete(ref _sols) = SolutionSet::Empty {
        // Pattern matching works
    }

    // Conditional (would need Case, just check pattern)
    if let SolutionSet::Conditional(ref _cases) = SolutionSet::Empty {
        // Pattern matching works
    }
}

/// Test that ConditionPredicate variants are accessible
#[test]
fn test_condition_predicate_display() {
    // Can't create without ExprId, but test the Display method exists
    // Just verify the type is accessible
    fn _accepts_predicate(_pred: &ConditionPredicate) {}
}

/// Test that ConditionSet methods are accessible
#[test]
fn test_condition_set_methods() {
    let cs = ConditionSet::empty();
    assert!(cs.is_empty());
    assert!(cs.is_otherwise());
    let _ = cs.predicates();
}

/// Test that SolveBudget is accessible and constructible
#[test]
fn test_solve_budget_construction() {
    let budget = SolveBudget::default();
    assert!(budget.can_branch()); // Default allows at least 1 branch

    let no_branch = SolveBudget::none();
    assert!(!no_branch.can_branch());
}

/// Test that SolverOptions is constructible
#[test]
fn test_solver_options_default() {
    let _opts = SolverOptions::default();
}

/// Test that solve functions are importable (signature check)
#[test]
fn test_solve_functions_exist() {
    // Just verify they're callable signatures by storing function pointers
    let _ = solve as fn(&_, &str, &mut _) -> _;
    let _ = solve_with_options as fn(&_, &str, &mut _, _) -> _;
}

/// Test that display wrappers are accessible
#[test]
fn test_display_types_exist() {
    // Just verify the types are accessible
    fn _accepts_display_expr(_d: &DisplayExpr) {}
    fn _accepts_latex_expr(_l: &LaTeXExpr) {}
}

/// Test that Case type is accessible
#[test]
fn test_case_type_accessible() {
    fn _accepts_case(_c: &Case) {}
}

/// Test that SolveResult type is accessible
#[test]
fn test_solve_result_accessible() {
    fn _accepts_solve_result(_r: &SolveResult) {}
    // Test SolveResult constructor
    let result = SolveResult::solved(SolutionSet::Empty);
    assert!(!result.has_solutions());
}

/// Test that BoundType and Interval are accessible
#[test]
fn test_interval_types_accessible() {
    fn _accepts_bound_type(_b: &BoundType) {}
    fn _accepts_interval(_i: &Interval) {}
}
