//! Contract: the Pythagorean sec/tan and csc/cot identities are simplified by the
//! LIVE rule set.
//!
//! # Background
//!
//! `rules::trig_canonicalization` used to carry a `register_pythagorean_identities`
//! helper that added six rules (`SecTan`/`CscCot`Pythagorean, `TanToSec`/`CotToCsc`
//! Pythagorean, and the two `MinusOne` identity-zero variants). That helper was never
//! called from anywhere in the workspace, so those six rules never fired. It was
//! removed (2026-07-02) after confirming the behavior below is already provided by the
//! rules registered in `rules::trigonometry::identities` — namely
//! `SecTanPythagoreanRule` / `CscCotPythagoreanRule` (from `identities/values_rules.rs`)
//! and `pythagorean::RecognizeSecSquaredRule` / `RecognizeCscSquaredRule` (n-ary, so
//! strictly more general than the removed binary-only rules), plus arithmetic.
//!
//! This test locks that coverage in: if the live rules are ever dropped, these
//! assertions fail instead of silently regressing.

use cas_parser::parse;
use cas_solver::runtime::Simplifier;

/// Simplify with the full default rule set and return the formatted result with all
/// whitespace stripped (so assertions are insensitive to display spacing).
fn simplify_compact(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(input, &mut simplifier.context).expect("parse failed");
    let (result, _) = simplifier.simplify(expr);
    format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    )
    .split_whitespace()
    .collect::<String>()
}

#[test]
fn sec_squared_minus_tan_squared_is_one() {
    assert_eq!(simplify_compact("sec(x)^2 - tan(x)^2"), "1");
}

#[test]
fn csc_squared_minus_cot_squared_is_one() {
    assert_eq!(simplify_compact("csc(x)^2 - cot(x)^2"), "1");
}

#[test]
fn sec_squared_minus_tan_squared_minus_one_is_zero() {
    assert_eq!(simplify_compact("sec(x)^2 - tan(x)^2 - 1"), "0");
}

#[test]
fn csc_squared_minus_cot_squared_minus_one_is_zero() {
    assert_eq!(simplify_compact("csc(x)^2 - cot(x)^2 - 1"), "0");
}

#[test]
fn one_plus_tan_squared_is_sec_squared() {
    // The engine's canonical form for sec is 1/cos (SecToRecipCosRule), so the
    // Pythagorean contraction `1 + tan²(x) → sec²(x)` surfaces as `1/cos(x)^2`.
    // Both are exactly `sec²(x)`.
    assert_eq!(simplify_compact("1 + tan(x)^2"), "1/cos(x)^2");
}

#[test]
fn one_plus_cot_squared_is_csc_squared() {
    // Likewise `1 + cot²(x) → csc²(x)` surfaces as `1/sin(x)^2` (CscToRecipSinRule).
    assert_eq!(simplify_compact("1 + cot(x)^2"), "1/sin(x)^2");
}
