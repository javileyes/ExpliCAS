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

fn solve_result_and_conditions(input: &str) -> (String, Vec<String>) {
    let out = Command::new(cargo::cargo_bin!("cas_cli"))
        .args(["eval", &format!("solve({input}, x)"), "--format", "json"])
        .output()
        .expect("Failed to run CLI");
    let wire: Value = serde_json::from_slice(&out.stdout).expect("Invalid wire output");
    let result = wire["result"].as_str().unwrap_or("").to_string();
    let conds = wire["required_conditions"]
        .as_array()
        .map(|cs| {
            cs.iter()
                .filter_map(|c| c["expr_display"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    (result, conds)
}

#[test]
fn bounded_domain_inverse_identity_carries_the_interval() {
    // F10 member 4: `asin(x) + acos(x) = π/2` is an identity ON ITS DOMAIN
    // [−1, 1]; the bare «All real numbers» was an over-claim. The condition
    // `1 − x² ≥ 0` (displayed as the interval) now rides the same channel
    // `√x = √x` uses for «ℝ if x ≥ 0».
    let (res, conds) = solve_result_and_conditions("asin(x)+acos(x) = pi/2");
    assert_eq!(res, "All real numbers if -1 ≤ x ≤ 1");
    assert!(conds.contains(&"1 - x^2".to_string()));
    // Affine argument scales the interval through the quadratic display.
    let (res2, _) = solve_result_and_conditions("asin(2*x)+acos(2*x) = pi/2");
    assert_eq!(res2, "All real numbers if -1/2 ≤ x ≤ 1/2");
    // atanh's domain is OPEN.
    let (res3, _) = solve_result_and_conditions("atanh(x) = atanh(x)");
    assert_eq!(res3, "All real numbers if -1 < x < 1");
    // Discrete answers stay clean (no noise line) and unconditioned
    // identities stay bare ℝ.
    let (res4, conds4) = solve_result_and_conditions("asin(x) = pi/6");
    assert_eq!(res4, "{ 1/2 }");
    assert!(conds4.is_empty());
    let (res5, conds5) = solve_result_and_conditions("x = x");
    assert_eq!(res5, "All real numbers");
    assert!(conds5.is_empty());
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

#[test]
fn rational_coefficient_radical_publishes_the_range_condition() {
    // Frontier-audit 2026-07-14 F10 m3, named stepping stone «coeficiente ≠1»:
    // the affine spellings `a·√f + b = y` published the squared root WITHOUT
    // the parameter-range condition, so for `y` outside the range the root is
    // extraneous (`2√x + 1 = y` at `y = 0` gives `x = 1/4`, but
    // `2·√(1/4) + 1 = 2 ≠ 0`). The split collector now carries the rational
    // coefficient as data and the publisher divides by it — the sign stays
    // folded, so a negative coefficient flips the displayed bound direction.
    let (res, conds) = solve_result_and_conditions("2*sqrt(x)+1 = y");
    assert_eq!(res, "{ (1/2·(y - 1))^2 }");
    assert!(conds.contains(&"y - 1".to_string()), "conds: {conds:?}");

    let (_, conds) = solve_result_and_conditions("2*sqrt(x)-1 = y");
    assert!(conds.contains(&"y + 1".to_string()), "conds: {conds:?}");

    // Negative coefficient: `−2√x + 5 = y` ⟹ `√x = (5−y)/2 ≥ 0` ⟹ `y ≤ 5`.
    let (_, conds) = solve_result_and_conditions("-2*sqrt(x)+5 = y");
    assert!(conds.contains(&"5 - y".to_string()), "conds: {conds:?}");

    // Coefficient born inside the radicand (`√(4x) = 2√x`).
    let (_, conds) = solve_result_and_conditions("sqrt(4*x)+1 = y");
    assert!(conds.contains(&"y - 1".to_string()), "conds: {conds:?}");
}

#[test]
fn factored_normal_form_is_peeled_before_the_split() {
    // `√x/2 + 1 − y` simplifies to the FACTORED `(1/2)·(√x + 2 − 2y)`, which
    // hid the radical from the term walk — the publisher peels the positive
    // rational factor (a `d = 0` invariant) before splitting.
    let (res, conds) = solve_result_and_conditions("sqrt(x)/2 + 1 = y");
    assert_eq!(res, "{ (2·y - 2)^2 }");
    assert!(conds.contains(&"y - 1".to_string()), "conds: {conds:?}");

    let (_, conds) = solve_result_and_conditions("(sqrt(x) - 2*y)/2 = 0");
    assert!(conds.contains(&"y".to_string()), "conds: {conds:?}");
}

#[test]
fn coefficient_spellings_out_of_scope_stay_untouched() {
    // SYMBOLIC coefficient: not a rational scale — the term stays in `rest`
    // and no range condition is invented (its sound condition needs a sign
    // split on `y`; named stepping stone, next cycle's candidate).
    let (res, conds) = solve_result_and_conditions("y*sqrt(x) = 2");
    assert_eq!(res, "{ 4 / y^2 }");
    // The pre-existing `y ≠ 0` (division) stays; what must NOT appear is a
    // range condition built from the symbolic division (`2/y ≥ 0`), which the
    // rational-only collector cannot justify.
    assert!(
        !conds.iter().any(|c| c.contains("/ y") || c.contains("2/y")),
        "no unsound blanket condition may appear: {conds:?}"
    );
    // Numeric thresholds decide by value, not by published condition.
    assert_eq!(solve("2*sqrt(x)+1 = 0"), "No solution");
    assert_eq!(solve("-2*sqrt(x)+5 = 7"), "No solution");
    // The coefficiented inequality keeps its upstream owner (`[0, 4)`), and
    // the unit inequality keeps the case-split owner.
    assert_eq!(solve("2*sqrt(x) < 4"), "[0, 4)");
    assert_eq!(solve("sqrt(x+1) <= x"), "[phi, infinity)");
}

#[test]
fn symbolic_coefficient_radical_publishes_the_sign_coupling_condition() {
    // Cycle-2 sibling of the rational-coefficient fix: `c·√f = g` with a
    // SYMBOLIC sqrt-free, var-free cofactor requires `g/c ≥ 0` no matter the
    // sign of `c` (a genuine root has `g/c = √f ≥ 0`). Without it,
    // `y·√x = 2 → {4/y²}` was spurious for every `y < 0` (at `y = −1`,
    // `x = 4` gives `−1·2 = −2 ≠ 2`). The display normalizer sharpens the
    // quotient to the sign bound where it can.
    let (res, conds) = solve_result_and_conditions("y*sqrt(x) = 2");
    assert_eq!(res, "{ 4 / y^2 }");
    assert!(conds.contains(&"y".to_string()), "conds: {conds:?}");

    // Negative RHS flips the satisfiable side: `−2/y ≥ 0` ⟺ `y < 0`.
    let (res, conds) = solve_result_and_conditions("y*sqrt(x) = -2");
    assert_eq!(res, "{ 4 / y^2 }");
    assert!(
        conds.iter().any(|c| c.contains("-2 / y")),
        "conds: {conds:?}"
    );

    // Fully general: `a·√x = b` carries `b/a ≥ 0`.
    let (res, conds) = solve_result_and_conditions("a*sqrt(x) = b");
    assert_eq!(res, "{ b^2 / a^2 }");
    assert!(
        conds.iter().any(|c| c.contains("b / a")),
        "conds: {conds:?}"
    );
}

#[test]
fn provably_vacuous_symbolic_condition_is_not_published() {
    // `a²·√x = 1` ⟹ `√x = 1/a²`, and `1/a² ≥ 0` is provably always true —
    // publishing it would be noise the sign prover gates out.
    let (res, conds) = solve_result_and_conditions("a^2*sqrt(x) = 1");
    assert_eq!(res, "{ 1 / a^4 }");
    assert!(
        !conds.iter().any(|c| c.contains("1 / a")),
        "vacuous condition published: {conds:?}"
    );
}

#[test]
fn symbolic_coefficient_neighbors_stay_untouched() {
    // Parameter cancels: the factored `y·(√(x+1) − 1)` is not a scaled-radical
    // TERM (the cofactor multiplies a sum), so no condition is invented.
    let (res, conds) = solve_result_and_conditions("y*sqrt(x+1) = y");
    assert_eq!(res, "{ 0 }");
    assert!(
        !conds.iter().any(|c| c.contains('/')),
        "no quotient condition may appear: {conds:?}"
    );
    // A cofactor CONTAINING the solve variable is not a coefficient.
    let (res, _) = solve_result_and_conditions("y*sqrt(x) = x");
    assert!(res.starts_with("solve("), "stays residual: {res}");
    // The symbolic-coefficient INEQUALITY keeps declining (empty display is
    // the pre-existing parametric-inequality presentation, unchanged).
    let (_, conds) = solve_result_and_conditions("y*sqrt(x) < 2");
    assert!(
        !conds.iter().any(|c| c.contains("2 / y")),
        "the inequality consumer must not divide by a symbolic sign: {conds:?}"
    );
}
