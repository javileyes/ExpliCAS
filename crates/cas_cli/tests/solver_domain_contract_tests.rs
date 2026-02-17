//! Contract tests for solver domain guards.
//!
//! These tests verify that the solver correctly handles exponential equations
//! based on ValueDomain and DomainMode settings. See POLICY.md for details.

use cas_ast::SolutionSet;
use cas_engine::engine::Simplifier;
use cas_solver::{solve_with_display_steps, SolverOptions};

/// Helper to solve an equation string and return the solution set.
fn solve_equation(eq_str: &str) -> Result<SolutionSet, cas_engine::error::CasError> {
    let mut simplifier = Simplifier::default();
    let stmt = cas_parser::parse_statement(eq_str, &mut simplifier.context)
        .expect("Failed to parse equation");

    let eq = match stmt {
        cas_parser::Statement::Equation(eq) => eq,
        _ => panic!("Expected equation, got expression"),
    };

    let opts = SolverOptions::default(); // RealOnly, Generic
    let (set, _steps, _diagnostics) = solve_with_display_steps(&eq, "x", &mut simplifier, opts)?;
    Ok(set)
}

/// Helper to check if solve returns an error with given substring.
fn solve_returns_error_containing(eq_str: &str, expected_substr: &str) -> bool {
    let mut simplifier = Simplifier::default();
    let stmt = cas_parser::parse_statement(eq_str, &mut simplifier.context)
        .expect("Failed to parse equation");

    let eq = match stmt {
        cas_parser::Statement::Equation(eq) => eq,
        _ => panic!("Expected equation, got expression"),
    };

    let opts = SolverOptions::default();
    match solve_with_display_steps(&eq, "x", &mut simplifier, opts) {
        Err(e) => e.to_string().contains(expected_substr),
        Ok(_) => false,
    }
}

// =============================================================================
// Contract Tests for Exponential Solver Domain Guards
// =============================================================================

/// Test 1: (2/3)^x = -5 in RealOnly → Empty
/// Base 2/3 > 0, so (2/3)^x > 0 for all real x.
/// RHS = -5 < 0, so no real solution exists.
#[test]
fn exponential_positive_base_negative_rhs_returns_empty() {
    let result = solve_equation("(2/3)^x = -5").expect("Should not error");
    assert!(
        matches!(result, SolutionSet::Empty),
        "Expected Empty set for (2/3)^x = -5, got {:?}",
        result
    );
}

/// Test 2: (-2)^x = 5 in RealOnly → Error (UnsupportedInRealDomain)
/// Base -2 < 0, cannot take real log of negative base.
/// Should return error suggesting complex mode.
#[test]
fn exponential_negative_base_returns_error() {
    // Check for either message format
    let has_expected_error = solve_returns_error_containing("(-2)^x = 5", "not positive")
        || solve_returns_error_containing("(-2)^x = 5", "not provably positive")
        || solve_returns_error_containing("(-2)^x = 5", "real domain");
    assert!(
        has_expected_error,
        "Expected error about base not positive for (-2)^x = 5"
    );
}

/// Test 3: 2^x = 8 in RealOnly → Solution x = 3
/// Base 2 > 0, RHS 8 > 0, standard case.
#[test]
fn exponential_valid_case_returns_solution() {
    let result = solve_equation("2^x = 8").expect("Should not error");
    assert!(
        matches!(result, SolutionSet::Discrete(ref sols) if sols.len() == 1),
        "Expected single solution for 2^x = 8, got {:?}",
        result
    );
}

/// Test 4: 1^x = 5 in RealOnly → Empty
/// 1^x = 1 for all x, never equals 5.
/// TODO: This works in REPL but solver::solve_with_display_steps has different path.
/// The identity 1^x→1 detection requires simplification integration.
#[test]
#[ignore = "base=1 handling requires REPL-level identity detection"]
fn exponential_base_one_wrong_rhs_returns_empty() {
    let result = solve_equation("1^x = 5").expect("Should not error");
    assert!(
        matches!(result, SolutionSet::Empty),
        "Expected Empty set for 1^x = 5, got {:?}",
        result
    );
}

/// Test 5: 1^x = 1 in RealOnly → AllReals
/// 1^x = 1 for all real x.
/// TODO: This works in REPL but solver::solve_with_display_steps has different path.
#[test]
#[ignore = "base=1 handling requires REPL-level identity detection"]
fn exponential_base_one_rhs_one_returns_all_reals() {
    let result = solve_equation("1^x = 1").expect("Should not error");
    assert!(
        matches!(result, SolutionSet::AllReals),
        "Expected AllReals for 1^x = 1, got {:?}",
        result
    );
}

/// Test 6: (1/2)^x = 4 in RealOnly → Solution x = -2
/// Base 1/2 > 0, RHS 4 > 0, valid exponential equation.
/// (1/2)^(-2) = 2^2 = 4
#[test]
fn exponential_fraction_base_returns_solution() {
    let result = solve_equation("(1/2)^x = 4").expect("Should not error");
    assert!(
        matches!(result, SolutionSet::Discrete(ref sols) if sols.len() == 1),
        "Expected single solution for (1/2)^x = 4, got {:?}",
        result
    );
}
