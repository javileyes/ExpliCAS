//! Contracts for the F10 cycle (frontier-audit 2026-07-14): the extraneous
//! root of a SYMBOLIC-parameter radical equation is filtered by the same
//! recorded range condition the numeric path already enforces.
//!
//! `√(a−x) = x` squares to `x² + x − a = 0`; the algebraic root
//! `(−√(4a+1)−1)/2` is `< 0` WHENEVER it is real (principal `√ ≥ 0` makes the
//! numerator `≤ −1`), while the equation requires `x ≥ 0` — extraneous for
//! every parameter value, so it must be dropped. The sign is decided by the
//! affine-over-radicals collector / structural range walk under the
//! root-filter premise (a candidate real solution has real radicals), never
//! by a float estimate. Roots that are only CONDITIONALLY negative
//! (`(1−√(4a+1))/2` is a genuine solution of `√(a+x) = x` for a ∈ [−1/4, 0])
//! must stay.

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
fn always_negative_parametric_root_is_dropped() {
    // The − root is ≤ −1/2 for every a where it is real; x ≥ 0 is recorded.
    assert_eq!(solve("sqrt(a-x) = x"), "{ 1/2·(sqrt(4·a + 1) - 1) }");
    // Same with a squared parameter radicand.
    assert_eq!(solve("sqrt(a^2-x) = x"), "{ 1/2·(sqrt(4·a^2 + 1) - 1) }");
}

#[test]
fn unsimplified_affine_shift_is_still_decided() {
    // Condition target `x − 1` at the − root is `(1−√(4a−3))/2 − 1` — an
    // UNDISTRIBUTED affine shift the recursive walk alone cannot sign; the
    // affine-over-radicals collector folds it to `−1/2 − √(4a−3)/2 < 0`.
    assert_eq!(solve("sqrt(a-x) = x-1"), "{ 1/2·(sqrt(4·a - 3) + 1) }");
}

#[test]
fn conditionally_valid_parametric_roots_are_kept() {
    // √(a+x) = x: the second root (1−√(4a+1))/2 is a GENUINE solution for
    // a ∈ [−1/4, 0] (e.g. a = −1/4 → x = 1/2). Pos + NonPos = Unknown ⇒ keep.
    assert_eq!(
        solve("sqrt(a+x) = x"),
        "{ 1/2·(sqrt(4·a + 1) + 1), 1/2·(1 - sqrt(4·a + 1)) }"
    );
    // √(a−x) = −x requires x ≤ 0: now the NEGATIVE root is the always-valid
    // one and the + root is only conditionally extraneous — both stay.
    assert_eq!(
        solve("sqrt(a-x) = -x"),
        "{ 1/2·(sqrt(4·a + 1) - 1), 1/2·(-sqrt(4·a + 1) - 1) }"
    );
}

#[test]
fn radicand_dominance_drops_the_symbol_coefficient_root() {
    // √(bx+1) = x: the − root (b−√(b²+4))/2 is negative for EVERY b because
    // √(b²+4) > √(b²) = |b| ≥ b — the dominance rule (radicand = q·v² + d,
    // d > 0, q·t² ≥ s²) proves it without knowing sign(b).
    assert_eq!(solve("sqrt(b*x+1) = x"), "{ 1/2·(sqrt(b^2 + 4) + b) }");
    // Scaled sibling: √(2bx+1) = x → x² − 2bx − 1 = 0, same dominance shape.
    assert_eq!(solve("sqrt(2*b*x+1) = x"), "{ sqrt(b^2 + 1) + b }");
}

fn solve_conditions(input: &str) -> Vec<String> {
    let out = Command::new(cargo::cargo_bin!("cas_cli"))
        .args(["eval", &format!("solve({input}, x)"), "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
    wire["required_conditions"]
        .as_array()
        .map(|cs| {
            cs.iter()
                .filter_map(|c| c["expr_display"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn parameter_only_range_condition_is_published() {
    // F10 member 3: the leftover range condition of a SHIFTED radical
    // (`√x + 3 = y` ⟹ `√x = y − 3 ≥ 0`) contains no solve variable, so it
    // can never act as a root filter — it must be PUBLISHED. The isolated
    // spelling already did; the shifted spellings now have parity.
    assert!(solve_conditions("sqrt(x)+3 = y").contains(&"y - 3".to_string()));
    assert!(solve_conditions("y = sqrt(x)+3").contains(&"y - 3".to_string()));
    assert!(solve_conditions("sqrt(x)-y = 2").contains(&"y + 2".to_string()));
    // Parity target (the isolated spelling, pre-existing owner).
    assert!(solve_conditions("sqrt(x) = y-3").contains(&"y - 3".to_string()));
    // Numeric thresholds decide by value — no noise condition is published.
    assert!(!solve_conditions("sqrt(x)+3 = 5")
        .iter()
        .any(|c| c == "2" || c == "5 - 3"));
}

#[test]
fn numeric_and_constant_controls_are_untouched() {
    // Numeric radicand: rational back-substitution already filtered these.
    assert_eq!(solve("sqrt(2-x) = x"), "{ 1 }");
    assert_eq!(solve("sqrt(x+1) = x-1"), "{ 3 }");
    // Golden-ratio constant root: const_value_bounds path (pre-existing).
    assert_eq!(solve("sqrt(x+1) = -x"), "{ 1/2·(1 - sqrt(5)) }");
    // Log-domain filter (the doc-comment example of the shared filter).
    assert_eq!(solve("ln(x)+ln(x+5) = 0"), "{ 1/2·(sqrt(29) - 5) }");
}
