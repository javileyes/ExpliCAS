//! Regression tests for `solve_system` command (2x2 linear systems)
//!
//! These tests verify the core functionality of the linear systems solver
//! using Cramer's rule with exact rational arithmetic.

use assert_cmd::cargo;
use assert_cmd::Command;
use predicates::prelude::*;

/// Helper to run cas_cli with given input
fn run_cas(input: &str) -> assert_cmd::assert::Assert {
    Command::new(cargo::cargo_bin!("cas_cli"))
        .write_stdin(input)
        .assert()
}

#[test]
fn test_solve_system_unique_simple() {
    // x + y = 3
    // x - y = 1
    // Solution: x = 2, y = 1
    run_cas("solve_system(x+y=3; x-y=1; x; y)\n")
        .success()
        .stdout(predicate::str::contains("x = 2"))
        .stdout(predicate::str::contains("y = 1"));
}

#[test]
fn test_solve_system_unique_with_coefficients() {
    // 2x + 3y = 7
    // x - y = 1
    // Solution: x = 2, y = 1
    run_cas("solve_system(2*x+3*y=7; x-y=1; x; y)\n")
        .success()
        .stdout(predicate::str::contains("x = 2"))
        .stdout(predicate::str::contains("y = 1"));
}

#[test]
fn test_solve_system_degenerate_det_zero() {
    // x + y = 2
    // 2x + 2y = 4  (same line, infinite solutions)
    // Should report det=0 error
    run_cas("solve_system(x+y=2; 2*x+2*y=4; x; y)\n")
        .success()
        .stdout(predicate::str::contains("no unique solution"));
}

#[test]
fn test_solve_system_non_linear() {
    // x * y = 1  (non-linear!)
    // x = 2
    // Should reject as non-linear
    run_cas("solve_system(x*y=1; x=2; x; y)\n")
        .success()
        .stdout(predicate::str::contains("non-linear"));
}

#[test]
fn test_solve_system_swapped_vars() {
    // Same system as test 1, but variables swapped in output
    // x + y = 3, x - y = 1
    // Asking for (y, x) instead of (x, y)
    run_cas("solve_system(x+y=3; x-y=1; y; x)\n")
        .success()
        .stdout(predicate::str::contains("y = 1"))
        .stdout(predicate::str::contains("x = 2"));
}
