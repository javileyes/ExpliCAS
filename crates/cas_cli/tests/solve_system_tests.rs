//! Regression tests for `solve_system` command (2x2 and 3x3 linear systems)
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

// =============================================================================
// 2x2 Systems
// =============================================================================

#[test]
fn test_solve_system_2x2_unique_simple() {
    // x + y = 3
    // x - y = 1
    // Solution: x = 2, y = 1
    run_cas("solve_system(x+y=3; x-y=1; x; y)\n")
        .success()
        .stdout(predicate::str::contains("x = 2"))
        .stdout(predicate::str::contains("y = 1"));
}

#[test]
fn test_solve_system_2x2_unique_with_coefficients() {
    // 2x + 3y = 7
    // x - y = 1
    // Solution: x = 2, y = 1
    run_cas("solve_system(2*x+3*y=7; x-y=1; x; y)\n")
        .success()
        .stdout(predicate::str::contains("x = 2"))
        .stdout(predicate::str::contains("y = 1"));
}

#[test]
fn test_solve_system_2x2_infinite_solutions() {
    // x + y = 2
    // 2x + 2y = 4  (same line, infinite solutions)
    run_cas("solve_system(x+y=2; 2*x+2*y=4; x; y)\n")
        .success()
        .stdout(predicate::str::contains("infinitely many solutions"));
}

#[test]
fn test_solve_system_2x2_no_solution() {
    // x + y = 2
    // x + y = 3  (parallel lines, no solution)
    run_cas("solve_system(x+y=2; x+y=3; x; y)\n")
        .success()
        .stdout(predicate::str::contains("no solution"));
}

#[test]
fn test_solve_system_2x2_zero_coefficient_row_is_inconsistent() {
    // 0·x + 0·y = 1 is the contradiction 0 = 1, so the system has NO solution. The all-zero
    // coefficient matrix made the proportionality cross-products (d1·b2 == d2·b1, d1·a2 == d2·a1)
    // reduce to 0 == 0 vacuously, so it was wrongly reported as "infinitely many solutions".
    run_cas("solve_system(0*x+0*y=1; 0*x+0*y=0; x; y)\n")
        .success()
        .stdout(predicate::str::contains("no solution"));
    // Two contradictory zero rows (0 = 2 and 0 = 3) are likewise inconsistent.
    run_cas("solve_system(0*x+0*y=2; 0*x+0*y=3; x; y)\n")
        .success()
        .stdout(predicate::str::contains("no solution"));
    // Control: both rows are the trivial 0 = 0, so EVERY (x, y) is a solution — infinitely many.
    run_cas("solve_system(0*x+0*y=0; 0*x+0*y=0; x; y)\n")
        .success()
        .stdout(predicate::str::contains("infinitely many solutions"));
}

#[test]
fn test_solve_system_2x2_non_linear() {
    // x * y = 1 with x = 2: since S2 (frente sistemas) this SOLVES by
    // isolate-substitute-verify composition — x = 2 isolates, x·y = 1
    // becomes 2·y = 1. The old "reject as non-linear" contract graduated.
    run_cas("solve_system(x*y=1; x=2; x; y)\n")
        .success()
        .stdout(predicate::str::contains("x = 2"))
        .stdout(predicate::str::contains("y = 1/2"));
}

#[test]
fn test_solve_system_2x2_swapped_vars() {
    // Same system as test 1, but variables swapped in output
    // x + y = 3, x - y = 1
    // Asking for (y, x) instead of (x, y)
    run_cas("solve_system(x+y=3; x-y=1; y; x)\n")
        .success()
        .stdout(predicate::str::contains("y = 1"))
        .stdout(predicate::str::contains("x = 2"));
}

// =============================================================================
// 3x3 Systems
// =============================================================================

#[test]
fn test_solve_system_3x3_unique_simple() {
    // x + y + z = 6
    // x - y = 0
    // y + z = 4
    // Solution: x = 2, y = 2, z = 2
    run_cas("solve_system(x+y+z=6; x-y=0; y+z=4; x; y; z)\n")
        .success()
        .stdout(predicate::str::contains("x = 2"))
        .stdout(predicate::str::contains("y = 2"))
        .stdout(predicate::str::contains("z = 2"));
}

#[test]
fn test_solve_system_3x3_with_negative() {
    // x + y + z = 1
    // 2x + y = 3
    // x + z = 2
    // Solution: x = 2, y = -1, z = 0
    run_cas("solve_system(x+y+z=1; 2*x+y=3; x+z=2; x; y; z)\n")
        .success()
        .stdout(predicate::str::contains("x = 2"))
        .stdout(predicate::str::contains("y = -1"))
        .stdout(predicate::str::contains("z = 0"));
}

#[test]
fn test_solve_system_3x3_infinite_solutions() {
    // x + y + z = 1
    // x + y + z = 1  (duplicate)
    // x + y + z = 1  (duplicate)
    // All same plane → infinite solutions
    run_cas("solve_system(x+y+z=1; x+y+z=1; x+y+z=1; x; y; z)\n")
        .success()
        .stdout(predicate::str::contains("infinitely many solutions"));
}

#[test]
fn test_solve_system_3x3_rank_two_dependent_is_infinite() {
    // z = 1 and x + y = 0, so the system is dependent, not inconsistent.
    run_cas("solve_system(x+y+z=1; x+y+2*z=2; x+y+3*z=3; x; y; z)\n")
        .success()
        .stdout(predicate::str::contains("infinitely many solutions"))
        .stdout(predicate::str::contains("dependent"));
}

#[test]
fn test_solve_system_3x3_no_solution() {
    // x + y + z = 1
    // x + y + z = 2
    // x + y + z = 3
    // Parallel planes, inconsistent → no solution
    run_cas("solve_system(x+y+z=1; x+y+z=2; x+y+z=3; x; y; z)\n")
        .success()
        .stdout(predicate::str::contains("no solution"));
}

#[test]
fn test_solve_system_3x3_non_linear() {
    // x*y = 1 (non-linear!)
    run_cas("solve_system(x*y=1; x=2; y=3; x; y; z)\n")
        .success()
        .stdout(predicate::str::contains("non-linear"));
}

// =============================================================================
// 4x4 Systems (Gaussian elimination)
// =============================================================================

#[test]
fn test_solve_system_4x4_unique() {
    // w + x + y + z = 4
    // w = 1, x = 1, y = 1
    // Solution: w = 1, x = 1, y = 1, z = 1
    run_cas("solve_system(w+x+y+z=4; w=1; x=1; y=1; w; x; y; z)\n")
        .success()
        .stdout(predicate::str::contains("w = 1"))
        .stdout(predicate::str::contains("x = 1"))
        .stdout(predicate::str::contains("y = 1"))
        .stdout(predicate::str::contains("z = 1"));
}

#[test]
fn test_solve_system_4x4_simple_substitution() {
    // a + b + c + d = 10
    // a = 1, b = 2, c = 3
    // Solution: a = 1, b = 2, c = 3, d = 4
    run_cas("solve_system(a+b+c+d=10; a=1; b=2; c=3; a; b; c; d)\n")
        .success()
        .stdout(predicate::str::contains("a = 1"))
        .stdout(predicate::str::contains("b = 2"))
        .stdout(predicate::str::contains("c = 3"))
        .stdout(predicate::str::contains("d = 4"));
}

#[test]
fn test_solve_system_4x4_infinite() {
    // All trivial: a=a, b=b, c=c, d=d
    run_cas("solve_system(a=a; b=b; c=c; d=d; a; b; c; d)\n")
        .success()
        .stdout(predicate::str::contains("infinitely many solutions"));
}

#[test]
fn test_solve_system_4x4_inconsistent() {
    // Circular system: a+b=3, b+c=4, c+d=5, d+a=6
    // Sum: 2(a+b+c+d)=18 but a+b=3, c+d=5 → only 8 ≠ 9
    run_cas("solve_system(a+b=3; b+c=4; c+d=5; d+a=6; a; b; c; d)\n")
        .success()
        .stdout(predicate::str::contains("no solution"));
}

// =============================================================================
// `solve([eqs], [vars])` list form — the natural surface routed to the same solver
// =============================================================================

#[test]
fn test_solve_list_form_2x2_unique() {
    // solve([x+y=3, x-y=1], [x, y]) -> x = 2, y = 1 (same as solve_system).
    run_cas("solve([x+y=3, x-y=1], [x, y])\n")
        .success()
        .stdout(predicate::str::contains("x = 2"))
        .stdout(predicate::str::contains("y = 1"));
}

#[test]
fn test_solve_list_form_bare_expressions_read_as_zero() {
    // A bare expression (no `=`) is read as `expr = 0`.
    run_cas("solve([x+y-3, x-y-1], [x, y])\n")
        .success()
        .stdout(predicate::str::contains("x = 2"))
        .stdout(predicate::str::contains("y = 1"));
}

#[test]
fn test_solve_list_form_3x3_unique() {
    // x + y + z = 6, x - y = 0, z = 3 -> x = 3/2, y = 3/2, z = 3.
    run_cas("solve([x+y+z=6, x-y=0, z=3], [x, y, z])\n")
        .success()
        .stdout(predicate::str::contains("x = 3/2"))
        .stdout(predicate::str::contains("y = 3/2"))
        .stdout(predicate::str::contains("z = 3"));
}

#[test]
fn test_solve_list_form_infinite_and_inconsistent() {
    run_cas("solve([x+y=1, 2*x+2*y=2], [x, y])\n")
        .success()
        .stdout(predicate::str::contains("infinitely many solutions"));
    run_cas("solve([x+y=1, x+y=2], [x, y])\n")
        .success()
        .stdout(predicate::str::contains("no solution"));
}

#[test]
fn test_solve_list_form_nonlinear_solves_by_substitution() {
    // Since S2, parabola-line systems solve with VERIFIED surd pairs (the
    // old honest-decline contract graduated to capability). Soundness is
    // preserved by the per-pair exact verification gate.
    run_cas("solve([x^2+y=1, x-y=0], [x, y])\n")
        .success()
        .stdout(predicate::str::contains("sqrt(5)"))
        .stdout(predicate::str::contains(" or "));
}

#[test]
fn test_single_equation_solve_unaffected_by_list_routing() {
    // The single-equation `solve(eq, var)` form still works (not captured by the list parser).
    run_cas("solve(x^2-4=0, x)\n")
        .success()
        .stdout(predicate::str::contains("-2"))
        .stdout(predicate::str::contains("2"));
}
