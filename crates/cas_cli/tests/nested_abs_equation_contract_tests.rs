//! Contracts for the F5 cycle (frontier-audit 2026-07-14): nested multi-abs
//! equations `|E| = c` with E combining ≥ 2 inner abs terms.
//!
//! The generic isolation recursed per-branch through the NARROW single-abs
//! isolation, silently dropping one branch's roots (`||x|−|x−2|| = 1`
//! returned `{3/2}`, losing the partner root `1/2`) or a whole flat-region
//! ray (the `= 2` case leaked a guarded `[2, ∞)` and lost `(−∞, 0]`). The
//! recovery now re-solves both branches `E = ±c` through the FULL solver
//! (which owns the exact sum-of-abs segment machinery) and (a) unions FULL
//! solution sets — ray branches included — and (b) fires proactively for
//! nested multi-abs even when the narrow path produced a clean-looking but
//! incomplete Discrete. Region-by-region hand verification for every pin.

use assert_cmd::cargo;
use assert_cmd::Command;
use serde_json::Value;

fn solve(input: &str) -> String {
    let out = Command::new(cargo::cargo_bin!("cas_cli"))
        .args(["eval", &format!("solve({input}, x)"), "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
    wire["result"].as_str().unwrap_or("").to_string()
}

#[test]
fn nested_difference_recovers_both_roots() {
    // ||x|−|x−2|| = 1 ⟺ |x|−|x−2| = ±1 → {3/2} ∪ {1/2} (F5 P0: lost 1/2).
    assert_eq!(solve("abs(abs(x)-abs(x-2)) = 1"), "{ 3/2, 1/2 }");
    // Shifted twin: ||x−1|−|x−3|| = 1 on the middle segment [1,3].
    assert_eq!(solve("abs(abs(x-1)-abs(x-3)) = 1"), "{ 5/2, 3/2 }");
    // Symmetric spread: ||x−2|−|x+2|| = 2 → ±1.
    assert_eq!(solve("abs(abs(x-2)-abs(x+2)) = 2"), "{ -1, 1 }");
}

#[test]
fn flat_region_branches_union_full_rays() {
    // ||x|−|x−2|| = 2: the argument is CONSTANT ±2 on the outer segments, so
    // the branches are rays `[2, ∞)` and `(−∞, 0]` — the Discrete-only
    // collection used to bail into a leaked-guard Conditional missing the
    // whole left ray.
    assert_eq!(
        solve("abs(abs(x)-abs(x-2)) = 2"),
        "(-infinity, 0] U [2, infinity)"
    );
}

#[test]
fn coefficient_and_scaled_arguments_enumerate_all_regions() {
    // |2|x|−|x−1|| = 1: three regions, roots −2 (x<0), 0 and 2/3 ([0,1)).
    assert_eq!(solve("abs(2*abs(x)-abs(x-1)) = 1"), "{ -2, 2/3, 0 }");
    // ||2x|−|x−3|| = 1: four roots across the breakpoints.
    assert_eq!(solve("abs(abs(2*x)-abs(x-3)) = 1"), "{ -4, 4/3, -2, 2/3 }");
}

#[test]
fn narrow_recursion_wrong_empty_is_recovered() {
    // Adversarial-sweep find: the narrow per-branch recursion dropped EVERY
    // root for the coefficient family at larger thresholds (wrong Empty).
    assert_eq!(solve("abs(2*abs(x)-abs(x-1)) = 2"), "{ -3, 1 }");
    assert_eq!(solve("abs(2*abs(x)-abs(x-1)) = 3"), "{ -4, 2 }");
    assert_eq!(solve("abs(abs(2*x)-abs(x-3)) = 5"), "{ -8, 8/3 }");
}

#[test]
fn out_of_range_constants_settle_empty_with_branch_completeness() {
    // range(|x|−|x−2|) = [−2, 2]: both branches Empty ⇒ honestly No solution.
    assert_eq!(solve("abs(abs(x)-abs(x-2)) = 5"), "No solution");
    // |x|+|x−2| ≥ 2 > 1 everywhere.
    assert_eq!(solve("abs(abs(x)+abs(x-2)) = 1"), "No solution");
}

#[test]
fn abs_ratio_with_nested_numerator_recovers_the_twin() {
    // F5 members 5-6: `|N|/|D| = c` cleared to `|N| = c·|D|` with the exact
    // `D ≠ 0` filter (`||x|−2|/|x| = 1` returned `{−1}`, losing `1`).
    assert_eq!(solve("abs(abs(x)-2)/abs(x) = 1"), "{ -1, 1 }");
    assert_eq!(solve("abs(abs(x)-3)/abs(x) = 1"), "{ -3/2, 3/2 }");
    assert_eq!(solve("abs(abs(x)-2)/abs(x) = 2"), "{ -2/3, 2/3 }");
    // c = 0 keeps the D ≠ 0 filter (`x = ±2` are fine, no 0/0 point enters).
    assert_eq!(solve("abs(abs(x)-2)/abs(x) = 0"), "{ 2, -2 }");
    // A negative ratio of absolute values is impossible wherever defined.
    assert_eq!(solve("abs(abs(x)-2)/abs(x) = -1"), "No solution");
    // Plain (non-nested) ratios keep their working owner.
    assert_eq!(solve("abs(x+1)/abs(x-1) = 2"), "{ 3, 1/3 }");
    assert_eq!(solve("abs(x^2-4)/abs(x) = 3"), "{ -1, 4, -4, 1 }");
}

#[test]
fn single_inner_abs_and_plain_forms_keep_their_owner() {
    // Single-inner-abs (2 abs total, below the ≥ 3 gate): isolation path.
    assert_eq!(solve("abs(abs(x)-3) = 1"), "{ 4, -4, 2, -2 }");
    assert_eq!(solve("abs(abs(x)-2) = 1"), "{ 3, -3, 1, -1 }");
    // Triple-nested with c = 0 collapses through the single branch.
    assert_eq!(solve("abs(abs(abs(x)-1)-1) = 0"), "{ 2, -2, 0 }");
    // Quadratic-argument recovery family (the pre-existing pinned owner).
    assert_eq!(solve("abs(x^2-2*x) = 3"), "{ -1, 3 }");
    assert_eq!(solve("abs(x-1) = 3"), "{ 4, -2 }");
}
