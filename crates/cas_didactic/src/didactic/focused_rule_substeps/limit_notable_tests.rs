//! Tests de `focused_rule_substeps`: `limit_notable_tests`, extraídos del módulo.

use super::{generate_limit_residual_substeps, generate_limit_substeps};
use crate::runtime::Step;
use cas_ast::Context;
use cas_math::limit_types::Approach;
use cas_parser::parse;
// The canonical rule name, not the literal: dispatch matches on this constant
// (`rule_dispatch.rs`), so the wire anchor lives in `rule_names.rs` — these
// tests call the generator directly and gain nothing from re-spelling it.
use cas_solver_core::rule_names::RULE_CONSERVAR_LIMITE_RESIDUAL;

fn substep_titles(before_src: &str, after_src: &str) -> Vec<String> {
    substep_titles_for_rule("Evaluar límite finito", before_src, after_src)
}

fn substep_titles_at_infinity(before_src: &str, after_src: &str) -> Vec<String> {
    substep_titles_for_rule("Evaluar límite en infinito", before_src, after_src)
}

fn substeps_at_infinity(before_src: &str, after_src: &str) -> Vec<super::SubStep> {
    let mut ctx = Context::new();
    let before = parse(before_src, &mut ctx).expect("parse before");
    let after = parse(after_src, &mut ctx).expect("parse after");
    let mut step = Step::new_compact("desc", "Evaluar límite en infinito", before, after);
    // Mirror the emitter: a limit step carries its approach, sign included.
    step.meta_mut().limit_approach = Some(Approach::PosInfinity);
    generate_limit_substeps(&ctx, &step)
}

fn substep_titles_for_rule(rule: &str, before_src: &str, after_src: &str) -> Vec<String> {
    let mut ctx = Context::new();
    let before = parse(before_src, &mut ctx).expect("parse before");
    let after = parse(after_src, &mut ctx).expect("parse after");
    let mut step = Step::new_compact("desc", rule, before, after);
    if rule.contains("infinito") {
        step.meta_mut().limit_approach = Some(Approach::PosInfinity);
    }
    generate_limit_substeps(&ctx, &step)
        .into_iter()
        .map(|s| s.description)
        .collect()
}

fn substep_titles_finite_at_point(
    before_src: &str,
    after_src: &str,
    point_src: &str,
) -> Vec<String> {
    let mut ctx = Context::new();
    let before = parse(before_src, &mut ctx).expect("parse before");
    let after = parse(after_src, &mut ctx).expect("parse after");
    let point = parse(point_src, &mut ctx).expect("parse point");
    let mut step = Step::new_compact("desc", "Evaluar límite finito", before, after);
    step.meta_mut().limit_point = Some(point);
    step.meta_mut().limit_approach = Some(Approach::Finite(point));
    generate_limit_substeps(&ctx, &step)
        .into_iter()
        .map(|s| s.description)
        .collect()
}

fn substeps_finite_at_point(
    before_src: &str,
    after_src: &str,
    point_src: &str,
) -> Vec<super::SubStep> {
    let mut ctx = Context::new();
    let before = parse(before_src, &mut ctx).expect("parse before");
    let after = parse(after_src, &mut ctx).expect("parse after");
    let point = parse(point_src, &mut ctx).expect("parse point");
    let mut step = Step::new_compact("desc", "Evaluar límite finito", before, after);
    step.meta_mut().limit_point = Some(point);
    step.meta_mut().limit_approach = Some(Approach::Finite(point));
    generate_limit_substeps(&ctx, &step)
}

#[test]
fn factor_cancel_deepens_into_factor_cancel_substitute_chain() {
    // With the limit point known (as the engine always sets for a finite
    // limit), factor-and-cancel SHOWS the work: factor → cancel → substitute,
    // each substep carrying its before/after expression.
    let subs = substeps_finite_at_point("(x^2-1)/(x-1)", "2", "1");
    let titles: Vec<&str> = subs.iter().map(|s| s.description.as_str()).collect();
    assert_eq!(
        titles.len(),
        3,
        "factor-cancel should show three substeps: {titles:?}"
    );
    assert!(titles[0].contains("Factoriza"), "{titles:?}");
    assert!(titles[1].contains("Cancela el factor común"), "{titles:?}");
    assert!(
        titles[1].contains("x - 1"),
        "cancel names the factor: {titles:?}"
    );
    assert!(titles[2].contains("Sustituye"), "{titles:?}");
    // The work is visible in the before/after expressions.
    assert_eq!(subs[0].before_expr, "(x^2 - 1) / (x - 1)");
    assert_eq!(subs[0].after_expr, "(x - 1) * (x + 1) / (x - 1)");
    assert_eq!(subs[1].after_expr, "x + 1");
    assert_eq!(subs[2].before_expr, "x + 1");
    assert_eq!(subs[2].after_expr, "2");

    // A residual denominator survives the cancellation.
    let subs2 = substeps_finite_at_point("(x^2-3*x+2)/(x^2-4)", "1/4", "2");
    assert_eq!(subs2.len(), 3);
    assert_eq!(subs2[1].after_expr, "(x - 1) / (x + 2)");
    assert_eq!(subs2[2].after_expr, "1 / 4");

    // Numerator IS the common factor: cofactor 1 is not printed, leaving 1/den.
    let subs3 = substeps_finite_at_point("(x-1)/(x^2-1)", "1/2", "1");
    assert_eq!(subs3.len(), 3);
    assert_eq!(subs3[1].after_expr, "1 / (x + 1)");
    assert_eq!(subs3[2].after_expr, "1 / 2");

    // Without a known point the deepened builder declines and the one-line
    // technique name is preserved (graceful fallback).
    let flat = substep_titles_for_rule("Evaluar límite finito", "(x^2-1)/(x-1)", "2");
    assert_eq!(flat.len(), 1);
    assert!(flat[0].contains("cancelar el factor común"));
}

/// C1.8 with teeth: the closing line asserts `EvalAt`, and the two sides of
/// that assertion come from different places — the cancelled form is rebuilt
/// here, the value is the engine's. Hand it a value that does not match and
/// the whole deepened chain must refuse to publish, falling back to the
/// one-line technique name.
///
/// This is what separates a declared relation from a decorative one: the
/// corpus never produced a mismatch, but nothing in the code guaranteed
/// that before the claim was checked.
#[test]
fn factor_cancel_declines_when_the_substitution_misses_the_engine_value() {
    assert_eq!(
        substeps_finite_at_point("(x^2-1)/(x-1)", "2", "1").len(),
        3,
        "the honest chain publishes"
    );
    let subs = substeps_finite_at_point("(x^2-1)/(x-1)", "7", "1");
    let titles: Vec<&str> = subs.iter().map(|s| s.description.as_str()).collect();
    assert!(
        !titles.iter().any(|t| t.contains("Sustituye")),
        "substituting x = 1 into x + 1 is not 7; the chain must decline: {titles:?}"
    );
}

#[test]
fn inf_minus_inf_conjugate_at_infinity_deepens_into_three_substeps() {
    // √(x²+x) − x → 1/2 by conjugate rationalization SHOWS the work:
    // indeterminate ∞−∞ → multiply/divide by conjugate → divide by the
    // dominant power and evaluate.
    let subs = substeps_at_infinity("sqrt(x^2+x)-x", "1/2");
    let titles: Vec<&str> = subs.iter().map(|s| s.description.as_str()).collect();
    assert_eq!(
        titles.len(),
        3,
        "conjugate should show three substeps: {titles:?}"
    );
    assert!(titles[0].contains("∞−∞"), "{titles:?}");
    assert!(
        titles[1].contains("conjugado") && titles[1].contains("sqrt(x^2 + x) + x"),
        "names the conjugate: {titles:?}"
    );
    assert!(titles[2].contains("límite es 1 / 2"), "{titles:?}");
    // The rationalized quotient is visible: √(x²+x) − x = x/(√(x²+x) + x).
    assert_eq!(subs[1].after_expr, "x / (sqrt(x^2 + x) + x)");
    assert_eq!(subs[2].after_expr, "1 / 2");

    // The reverse order `x − √(x²−x)` (surd second) rationalizes the same way.
    let rev = substeps_at_infinity("x-sqrt(x^2-x)", "1/2");
    assert_eq!(rev.len(), 3, "{rev:?}");
    assert!(
        rev[1].description.contains("conjugado"),
        "{:?}",
        rev[1].description
    );
    assert_eq!(rev[1].after_expr, "x / (sqrt(x^2 - x) + x)");

    // A radicand with no linear term (b = 0) rationalizes to 1/(√(x²+1)+x) → 0.
    let zero = substeps_at_infinity("sqrt(x^2+1)-x", "0");
    assert_eq!(zero.len(), 3, "{zero:?}");
    assert_eq!(zero[1].after_expr, "1 / (sqrt(x^2 + 1) + x)");
    assert_eq!(zero[2].after_expr, "0");
}

#[test]
fn inf_minus_inf_common_denominator_at_finite_point_deepens() {
    // `1/x − 1/sin(x) → 0` (∞−∞ at 0) SHOWS the work: indeterminate ∞−∞ →
    // combine over a common denominator → the resulting 0/0 (L'Hôpital/Taylor).
    let subs = substeps_finite_at_point("1/x - 1/sin(x)", "0", "0");
    let titles: Vec<&str> = subs.iter().map(|s| s.description.as_str()).collect();
    assert_eq!(titles.len(), 3, "{titles:?}");
    assert!(titles[0].contains("∞−∞"), "{titles:?}");
    assert!(titles[1].contains("común denominador"), "{titles:?}");
    assert!(
        titles[2].contains("0/0") && titles[2].contains("Hôpital"),
        "{titles:?}"
    );
    // The combined single fraction is visible.
    assert_eq!(subs[1].after_expr, "(sin(x) - x) / (x * sin(x))");
    assert_eq!(subs[2].after_expr, "0");

    // The audit's OPEN peldaño `1/tan²x − 1/x² → −2/3` now narrates (three
    // substeps, the combine step reaching a single fraction).
    let tan2 = substeps_finite_at_point("1/tan(x)^2 - 1/x^2", "-2/3", "0");
    assert_eq!(tan2.len(), 3, "{:?}", tan2[0].description);
    assert!(tan2[1].description.contains("común denominador"));
    assert_eq!(tan2[2].after_expr, "-2 / 3");
}

#[test]
fn inf_minus_inf_common_denominator_declines_out_of_scope() {
    // Divergent same-sign ∞−∞ (result ±∞, not a finite rational): the
    // after-oracle (P3) rejects it, so no common-denominator narration.
    let divergent = substep_titles_finite_at_point("1/x - 1/x^2", "infinity", "0");
    assert!(
        divergent.iter().all(|t| !t.contains("común denominador")),
        "{divergent:?}"
    );
    // A term that does not blow up at the point (`sin(x)` bounded): P5 rejects.
    let bounded = substep_titles_finite_at_point("1/x - sin(x)", "0", "0");
    assert!(
        bounded.iter().all(|t| !t.contains("común denominador")),
        "{bounded:?}"
    );
    // A single quotient (`Div`, not a `Sub` of two reciprocals) never reaches
    // this recognizer — it stays a notable/L'Hôpital narration.
    let quotient = substep_titles_finite_at_point("sin(x)/x", "1", "0");
    assert!(
        quotient.iter().all(|t| !t.contains("común denominador")),
        "{quotient:?}"
    );
}

#[test]
fn inf_minus_inf_conjugate_declines_out_of_scope() {
    // Radicand not degree 2 (√(x⁴+x) − x²): the recognizer declines, so no
    // conjugate substeps are emitted (honest empty, not false narration).
    assert!(substep_titles_at_infinity("sqrt(x^4+x)-x^2", "0").is_empty());
    // Leading terms do NOT cancel (√(x²+x) − 2x: leading 1 ≠ 2²): declines.
    assert!(substep_titles_at_infinity("sqrt(x^2+x)-2*x", "0").is_empty());
    // A genuine ∞/∞ quotient still routes to dominance, never to conjugate.
    let dom = substep_titles_at_infinity("(3*x^2+1)/(x^2-5)", "3");
    assert!(
        dom.iter().any(|t| t.contains("Dominancia")),
        "quotient stays dominance: {dom:?}"
    );
}

#[test]
fn direct_substitution_names_the_specific_point() {
    // Direct substitution is atomic (one substep); with the point known it
    // names the specific evaluation point rather than a generic "en el punto".
    let titles = substep_titles_finite_at_point("x^2+3*x+1", "11", "2");
    assert_eq!(titles.len(), 1, "{titles:?}");
    assert!(titles[0].contains("Sustitución directa"), "{titles:?}");
    assert!(titles[0].contains("x = 2"), "names the point: {titles:?}");
    // A negative point renders cleanly in the title.
    let titles_neg = substep_titles_finite_at_point("x^3-2*x", "1", "-1");
    assert_eq!(titles_neg.len(), 1, "{titles_neg:?}");
    assert!(titles_neg[0].contains("x = -1"), "{titles_neg:?}");
    // Without a known point the generic continuity title is preserved.
    let flat = substep_titles_for_rule("Evaluar límite finito", "x^2+1", "5");
    assert_eq!(flat.len(), 1);
    assert!(flat[0].contains("Sustitución directa"), "{flat:?}");
}

#[test]
fn names_generic_zero_over_zero_at_origin() {
    // Genuine 0/0 at 0 not matched by a specific notable: a `u^k` denominator with a finite
    // result. The denominator vanishes at 0 and the finite result forces the numerator to
    // vanish too, so the form is provably 0/0 — narrate L'Hôpital / Taylor. (Values verified
    // against sympy: 1/6, 1/2, 1/3, -1/6.)
    for (before, after) in [
        ("(x-sin(x))/x^3", "1/6"),
        ("(exp(x)-1-x)/x^2", "1/2"),
        ("(tan(x)-x)/x^3", "1/3"),
        ("(sin(x)-x)/x^3", "-1/6"),
    ] {
        // A polynomial denominator → the L'Hôpital iteration is reconstructed
        // (≥ 2 substeps): differentiate until the denominator no longer
        // vanishes, then substitute.
        let titles = substep_titles_finite_at_point(before, after, "0");
        assert!(
            titles.len() >= 2,
            "{before}: iteration expected: {titles:?}"
        );
        assert!(
            titles[0].contains("0/0")
                && titles[0].contains("Hôpital")
                && titles[0].contains("x = 0"),
            "{before}: {titles:?}"
        );
        assert!(
            titles.last().unwrap().contains("sustituye"),
            "{before}: ends with substitution: {titles:?}"
        );
    }
}

#[test]
fn lhopital_iteration_differentiates_until_determinate() {
    // `(x − sin x)/x³ → 1/6` needs three L'Hôpital steps (the denominator x³
    // has a triple root at 0), then a substitution — four substeps that show
    // each differentiated form, ending at the engine's result.
    let subs = substeps_finite_at_point("(x-sin(x))/x^3", "1/6", "0");
    let titles: Vec<&str> = subs.iter().map(|s| s.description.as_str()).collect();
    assert_eq!(subs.len(), 4, "{titles:?}");
    assert!(titles[0].contains("L'Hôpital"), "{titles:?}");
    assert!(titles[1].contains("Sigue siendo 0/0"), "{titles:?}");
    assert!(titles[2].contains("Sigue siendo 0/0"), "{titles:?}");
    assert!(titles[3].contains("sustituye"), "{titles:?}");
    assert_eq!(subs[0].before_expr, "(x - sin(x)) / x^3");
    assert_eq!(subs[0].after_expr, "(1 - cos(x)) / (3 * x^2)");
    assert_eq!(subs[1].after_expr, "sin(x) / (6 * x)");
    assert_eq!(subs[2].after_expr, "cos(x) / 6");
    assert_eq!(subs[3].after_expr, "1 / 6");

    // `ln(x)/(x−1)` needs a single step (simple root); `num'/1` prints as `1/x`.
    let one = substeps_finite_at_point("ln(x)/(x-1)", "1", "1");
    assert_eq!(one.len(), 2, "{one:?}");
    assert_eq!(one[0].after_expr, "1 / x");
    assert_eq!(one[1].after_expr, "1");

    // A transcendental denominator cannot give an exact step count, so the
    // iteration declines to the one-line L'Hôpital/Taylor name.
    let trig = substep_titles_finite_at_point("x^2/sin(x)", "0", "0");
    assert_eq!(trig.len(), 1, "{trig:?}");
    assert!(
        trig[0].contains("L'Hôpital") || trig[0].contains("Taylor"),
        "{trig:?}"
    );
}

#[test]
fn names_generic_zero_over_zero_at_shifted_point() {
    // The denominator is a polynomial that vanishes at the (nonzero) limit point, so the form
    // is a genuine 0/0 there — narrate it with the actual point. (sympy: 1, 1/2, -1/6.)
    for (before, after) in [
        ("ln(x)/(x-1)", "1"),
        ("(1-cos(x-1))/(x-1)^2", "1/2"),
        ("(sin(x-1)-(x-1))/(x-1)^3", "-1/6"),
    ] {
        let titles = substep_titles_finite_at_point(before, after, "1");
        assert!(
            titles.len() >= 2,
            "{before}: iteration expected: {titles:?}"
        );
        assert!(
            titles[0].contains("0/0") && titles[0].contains("x = 1"),
            "{before}: {titles:?}"
        );
        assert!(
            titles.last().unwrap().contains("sustituye"),
            "{before}: {titles:?}"
        );
    }
}

#[test]
fn names_generic_zero_over_zero_with_trig_denominator() {
    // The denominator is a zero-at-the-origin function (or a power of one) of an argument that
    // vanishes at 0, so it tends to 0 there — a genuine 0/0. (sympy: 0, 0, 0, 1/6.)
    for (before, after) in [
        ("(cos(x)-1)/sin(x)", "0"),
        ("(1-cos(x))/tan(x)", "0"),
        ("x^2/sin(x)", "0"),
        ("(x-sin(x))/sin(x)^3", "1/6"),
    ] {
        let titles = substep_titles_finite_at_point(before, after, "0");
        assert_eq!(titles.len(), 1, "{before} should narrate a 0/0 technique");
        assert!(
            titles[0].contains("0/0") && titles[0].contains("x=0"),
            "{before}: expected the 0/0 narration at x=0 in `{}`",
            titles[0]
        );
    }
    // cos does NOT vanish at 0, so 1/cos(x) (which is 1, not 0/0) must decline.
    assert!(substep_titles_finite_at_point("1/cos(x)", "1", "0").is_empty());
}

#[test]
fn declines_generic_zero_over_zero_unless_denominator_vanishes_at_the_point() {
    // SOUNDNESS: the denominator must vanish AT THE LIMIT POINT. A point where it does not
    // vanish is plain direct substitution (not 0/0) and must NOT be narrated.
    assert!(substep_titles_finite_at_point("(x+1)/x", "3/2", "2").is_empty());
    assert!(substep_titles_finite_at_point("sin(pi*x)/x", "0", "1").is_empty());
    // `x-1` vanishes at 1, not at 0 — so `ln(x)/(x-1)` at 0 (which is +∞, not 0/0) declines.
    assert!(substep_titles_finite_at_point("ln(x)/(x-1)", "1", "0").is_empty());
    // Without point information (no limit_point on the step) the 0/0 claim cannot be justified.
    assert!(substep_titles("(x-sin(x))/x^3", "1/6").is_empty());
}

#[test]
fn generic_zero_over_zero_does_not_shadow_specific_notables() {
    // The specific notables / factor-cancel must still win at the origin (they
    // return earlier). They are now narrated as 0/0 → apply, so the notable is
    // the SECOND substep, not a generic L'Hôpital fallback.
    let sin = substep_titles_finite_at_point("sin(x)/x", "1", "0");
    assert!(sin.len() == 2 && sin[1].contains("sin(u)/u = 1"), "{sin:?}");
    assert!(sin[0].contains("indeterminación 0/0"), "{sin:?}");
    let cos = substep_titles_finite_at_point("(1-cos(x))/x^2", "1/2", "0");
    assert!(
        cos.len() == 2 && cos[1].contains("(1 − cos(u))/u² = 1/2"),
        "{cos:?}"
    );
}

#[test]
fn names_infinity_dominance_methods() {
    for (before, after, needle) in [
        ("(x^2+1)/(2*x^2-3)", "1/2", "coeficientes líderes"),
        ("(3*x^2+1)/(x^2-5)", "3", "coeficientes líderes"),
        ("(x+1)/(x^2+1)", "0", "denominador tiene mayor grado"),
        ("(x^3+1)/(x^2+1)", "infinity", "numerador tiene mayor grado"),
        ("x^2+1", "infinity", "polinomio de grado ≥ 1"),
    ] {
        // A genuine ∞/∞ quotient is narrated as two substeps (∞/∞ → dominance);
        // a bare polynomial stays a single substep. Either way the dominance
        // method appears in SOME substep (see
        // `dominance_quotient_shows_infinity_over_infinity` for the chain).
        let titles = substep_titles_at_infinity(before, after);
        assert!(
            titles.iter().any(|t| t.contains(needle)),
            "{before}: expected `{needle}` in `{titles:?}`"
        );
    }
}

#[test]
fn dominance_quotient_shows_infinity_over_infinity() {
    // A genuine ∞/∞ quotient (growth-class or both-polynomial) prepends the
    // indeterminate form; a `1/∞` (constant numerator) and a bare polynomial
    // do NOT (they are not ∞/∞).
    for (before, after, needle) in [
        ("ln(x)/x", "0", "el logaritmo crece más despacio"),
        ("exp(x)/x^3", "infinity", "la exponencial crece más rápido"),
        ("(2*x^2+1)/(x^2+3)", "2", "coeficientes líderes"),
        ("(x^2+1)/(x^3+x)", "0", "denominador tiene mayor grado"),
    ] {
        let titles = substep_titles_at_infinity(before, after);
        assert_eq!(titles.len(), 2, "{before}: {titles:?}");
        assert!(titles[0].contains("∞/∞"), "{before}: {titles:?}");
        assert!(titles[1].contains(needle), "{before}: {titles:?}");
    }
    // `1/x` is `1/∞`, not `∞/∞` — single substep, no ∞/∞ claim.
    let recip = substep_titles_at_infinity("1/x", "0");
    assert_eq!(recip.len(), 1, "{recip:?}");
    assert!(!recip[0].contains("∞/∞"), "{recip:?}");
    // A bare polynomial is not a quotient.
    let poly = substep_titles_at_infinity("x^2+1", "infinity");
    assert_eq!(poly.len(), 1, "{poly:?}");
}

#[test]
fn residual_limit_suggests_one_sided_limits_without_claiming_dne() {
    // A finite-point limit the safe policy keeps unresolved gets an HONEST
    // method hint (compute one-sided limits), never a non-existence claim —
    // the residual lumps together genuine DNE and existent-but-undecided cases.
    let mut ctx = Context::new();
    let before = parse("1/x", &mut ctx).expect("parse");
    let point = parse("0", &mut ctx).expect("parse");
    let mut step = Step::new_compact("desc", RULE_CONSERVAR_LIMITE_RESIDUAL, before, before);
    step.meta_mut().limit_point = Some(point);
    let subs = generate_limit_residual_substeps(&ctx, &step);
    assert_eq!(
        subs.len(),
        1,
        "{:?}",
        subs.iter().map(|s| &s.description).collect::<Vec<_>>()
    );
    let title = &subs[0].description;
    assert!(title.contains("límites laterales"), "{title}");
    assert!(title.contains("x = 0"), "{title}");
    // SOUNDNESS: the statement is a CONDITIONAL method (si coinciden … / si
    // difieren …), not an unconditional DNE claim.
    assert!(title.contains("si coinciden"), "{title}");
    assert!(title.contains("si difieren"), "{title}");

    // An infinity / one-sided residual carries no finite limit_point → no hint.
    let step2 = Step::new_compact("desc", RULE_CONSERVAR_LIMITE_RESIDUAL, before, before);
    assert!(
        generate_limit_residual_substeps(&ctx, &step2).is_empty(),
        "no hint without a finite point"
    );
}

#[test]
fn names_the_e_limit_at_infinity() {
    // The infinity-side notable: the definition of e. Narrated as the 1^∞
    // indeterminate form first, then the standard limit.
    for before in ["(1+1/x)^x", "(1+1/n)^n"] {
        let titles = substep_titles_at_infinity(before, "e");
        assert_eq!(titles.len(), 2, "{before}: {titles:?}");
        assert!(titles[0].contains("1^∞"), "{before}: {titles:?}");
        assert!(
            titles[1].contains("(1 + 1/x)^x = e"),
            "{before}: expected the e notable in `{titles:?}`"
        );
    }
    // Wrong structure → wrong value: (1+2/x)^x → e² and (1+1/x)^(2x) → e² must decline (both
    // structurally and because the result is not e).
    assert!(substep_titles_at_infinity("(1+2/x)^x", "e^2").is_empty());
    assert!(substep_titles_at_infinity("(1+1/x)^(2*x)", "e^2").is_empty());
}

#[test]
fn names_growth_class_dominance_at_infinity() {
    // The hierarchy ln(x) ≪ x^a ≪ e^x: the higher class wins, narrated by the result (0 or ∞).
    for (before, after, needle) in [
        (
            "ln(x)/x",
            "0",
            "el logaritmo crece más despacio que la potencia",
        ),
        (
            "ln(x)^2/x",
            "0",
            "el logaritmo crece más despacio que la potencia",
        ),
        (
            "sqrt(x)/ln(x)",
            "infinity",
            "la potencia crece más rápido que el logaritmo",
        ),
        (
            "x^2/exp(x)",
            "0",
            "la potencia crece más despacio que la exponencial",
        ),
        (
            "exp(x)/x^3",
            "infinity",
            "la exponencial crece más rápido que la potencia",
        ),
        (
            "ln(x)/exp(x)",
            "0",
            "el logaritmo crece más despacio que la exponencial",
        ),
    ] {
        // Growth-class quotients are genuine ∞/∞, so they get a two-substep
        // narrative; the dominance method appears in SOME substep.
        let titles = substep_titles_at_infinity(before, after);
        assert!(
            titles.iter().any(|t| t.contains(needle)),
            "{before}: expected `{needle}` in `{titles:?}`"
        );
    }
    // Outside the hierarchy (bounded sin) or a wrong/structure-mismatched result must decline.
    assert!(substep_titles_at_infinity("sin(x)/x", "0").is_empty());
    // e^{-x} decays (negative leading exponent), so x^2/e^{-x} is not classified as exp.
    assert!(substep_titles_at_infinity("x^2/exp(-x)", "infinity").is_empty());
    // Product form `p(x)·e^{-q(x)} → 0`: the decaying exponential beats the polynomial.
    for before in ["x^2*exp(-x)", "x^5*exp(-2*x)", "exp(-x^2)*x^3"] {
        let titles = substep_titles_at_infinity(before, "0");
        assert_eq!(titles.len(), 1, "{before} should narrate product decay");
        assert!(
            titles[0].contains("la exponencial decae"),
            "{before}: expected product-decay dominance in `{}`",
            titles[0]
        );
    }
    // A GROWING exponential product (x²·e^x → ∞) and a bounded·decay product (sin(x)·e^{-x},
    // sin not sub-exponential) must not narrate product decay.
    assert!(substep_titles_at_infinity("x^2*exp(x)", "infinity").is_empty());
    assert!(substep_titles_at_infinity("sin(x)*exp(-x)", "0").is_empty());
}

#[test]
fn dominance_rejects_mismatched_results_and_does_not_fire_at_finite_points() {
    // Equal degrees but a fabricated ratio (real ratio is 1/2) must decline.
    assert!(substep_titles_at_infinity("(x^2+1)/(2*x^2-3)", "7").is_empty());
    // At a FINITE point the dominance narrator must not fire (the rule routes elsewhere).
    let finite = substep_titles("(x^2+1)/(2*x^2-3)", "1/2");
    assert!(
        finite.iter().all(|t| !t.contains("Dominancia")),
        "dominance must not narrate a finite-point limit: {finite:?}"
    );
}

#[test]
fn names_the_standard_notable_limits() {
    for (before, after, needle) in [
        ("sin(x)/x", "1", "sin(u)/u = 1"),
        ("tan(x)/x", "1", "tan(u)/u = 1"),
        ("arcsin(x)/x", "1", "arcsin(u)/u = 1"),
        ("arctan(x)/x", "1", "arctan(u)/u = 1"),
        ("sinh(x)/x", "1", "sinh(u)/u = 1"),
        ("tanh(x)/x", "1", "tanh(u)/u = 1"),
        // Scaled arguments f(a·u)/u → a (a = 1 is the bare form above).
        ("sin(3*x)/x", "3", "sin(3·u)/u = 3"),
        ("tan(2*x)/x", "2", "tan(2·u)/u = 2"),
        ("arctan(5*x)/x", "5", "arctan(5·u)/u = 5"),
        ("sin(-2*x)/x", "-2", "sin(-2·u)/u = -2"),
        // Cross / scaled-denominator forms f(a·u)/g(b·u) → a/b (one or both sides notable).
        ("sin(3*x)/(2*x)", "3/2", "sin(3·u)/(2·u) = 3/2"),
        ("sin(x)/(2*x)", "1/2", "sin(u)/(2·u) = 1/2"),
        ("tan(3*x)/sin(2*x)", "3/2", "tan(3·u)/sin(2·u) = 3/2"),
        ("arcsin(x)/(5*x)", "1/5", "arcsin(u)/(5·u) = 1/5"),
        ("(exp(x)-1)/x", "1", "(e^u − 1)/u = 1"),
        ("ln(1+x)/x", "1", "ln(1+u)/u = 1"),
        ("(1-cos(x))/x^2", "1/2", "(1 − cos(u))/u² = 1/2"),
        ("(2^x-1)/x", "ln(2)", "(aᵘ − 1)/u = ln(a)"),
        ("(3^x-1)/x", "ln(3)", "(aᵘ − 1)/u = ln(a)"),
        ("(1+x)^(1/x)", "e", "(1 + u)^(1/u) = e"),
        // Binomial / root first-order equivalent ((1+u)^a − 1)/u → a.
        ("(sqrt(1+x)-1)/x", "1/2", "((1+u)^(1/2) − 1)/u = 1/2"),
        ("((1+x)^3-1)/x", "3", "((1+u)^(3) − 1)/u = 3"),
        ("x*sin(1/x)", "0", "teorema del sándwich"),
        ("x^2*cos(1/x)", "0", "teorema del sándwich"),
        ("x/sin(x)", "1", "u/sin(u) = 1"),
        ("x/tan(x)", "1", "u/tan(u) = 1"),
        ("x/(exp(x)-1)", "1", "u/(e^u − 1) = 1"),
        ("x/arcsin(x)", "1", "u/arcsin(u) = 1"),
    ] {
        // The technique is recognized (it appears in SOME substep). A 0/0
        // quotient notable is now narrated as two substeps (0/0 → apply), while
        // the squeeze and the 1^∞ `= e` forms stay a single substep — checked
        // precisely in `notable_zero_over_zero_shows_indeterminate_form_first`.
        let titles = substep_titles(before, after);
        assert!(
            titles.iter().any(|t| t.contains(needle)),
            "{before}: expected `{needle}` in `{titles:?}`"
        );
    }
}

#[test]
fn notable_zero_over_zero_shows_indeterminate_form_first() {
    // A 0/0 quotient notable is narrated as two substeps: show that direct
    // substitution gives 0/0, then apply the standard limit.
    for (before, after, needle) in [
        ("sin(x)/x", "1", "sin(u)/u = 1"),
        ("sin(3*x)/x", "3", "sin(3·u)/u = 3"),
        ("(exp(x)-1)/x", "1", "(e^u − 1)/u = 1"),
        ("ln(1+x)/x", "1", "ln(1+u)/u = 1"),
        ("x/sin(x)", "1", "u/sin(u) = 1"),
        ("(1-cos(x))/x^2", "1/2", "(1 − cos(u))/u² = 1/2"),
    ] {
        let titles = substep_titles(before, after);
        assert_eq!(titles.len(), 2, "{before}: {titles:?}");
        assert!(
            titles[0].contains("indeterminación 0/0"),
            "{before}: {titles:?}"
        );
        assert!(titles[1].contains(needle), "{before}: {titles:?}");
    }
    // The 1^∞ `= e` form is NOT 0/0 (it gets the 1^∞ narrative instead, see
    // `notable_one_to_infinity_shows_indeterminate_form_first`), and the
    // squeeze theorem (see `squeeze_shows_the_bounding_argument`) — neither
    // claims 0/0.
    let e_form = substep_titles("(1+x)^(1/x)", "e");
    assert!(
        !e_form.iter().any(|t| t.contains("indeterminación 0/0")),
        "{e_form:?}"
    );
    let squeeze = substep_titles("x*sin(1/x)", "0");
    assert!(
        !squeeze.iter().any(|t| t.contains("indeterminación 0/0")),
        "{squeeze:?}"
    );
    assert!(
        squeeze.iter().any(|t| t.contains("teorema del sándwich")),
        "{squeeze:?}"
    );
}

#[test]
fn squeeze_shows_the_bounding_argument() {
    // The squeeze theorem is narrated as two substeps: bound the oscillator
    // (`|sin/cos| ≤ 1` ⇒ `|uᵏ·osc| ≤ |uᵏ|`), then `|uᵏ| → 0`.
    let subs = substeps_finite_at_point("x*sin(1/x)", "0", "0");
    assert_eq!(
        subs.len(),
        2,
        "{:?}",
        subs.iter().map(|s| &s.description).collect::<Vec<_>>()
    );
    assert!(subs[0].description.contains("Acota el factor oscilante"));
    assert!(subs[0].description.contains("|sin(1 / x)| ≤ 1"));
    assert_eq!(subs[0].before_expr, "x * sin(1 / x)");
    assert_eq!(subs[0].after_expr, "|x|");
    assert!(subs[1].description.contains("teorema del sándwich"));
    assert_eq!(subs[1].before_expr, "|x|");
    assert_eq!(subs[1].after_expr, "0");

    // A higher power bounds by `|xᵏ|`.
    let subs2 = substeps_finite_at_point("x^2*cos(1/x)", "0", "0");
    assert_eq!(subs2.len(), 2);
    assert_eq!(subs2[0].after_expr, "|x^2|");
    assert_eq!(subs2[1].after_expr, "0");
}

#[test]
fn notable_one_to_infinity_shows_indeterminate_form_first() {
    // The `(1+u)^(1/u)` / `(1+1/x)^x → e` notables are 1^∞ (not 0/0): narrate
    // the indeterminate form, then cite the definition of e.
    let finite = substep_titles("(1+x)^(1/x)", "e");
    assert_eq!(finite.len(), 2, "{finite:?}");
    assert!(finite[0].contains("1^∞"), "{finite:?}");
    assert!(!finite[0].contains("0/0"), "must not claim 0/0: {finite:?}");
    assert!(finite[1].contains("(1 + u)^(1/u) = e"), "{finite:?}");

    let at_inf = substep_titles_at_infinity("(1+1/x)^x", "e");
    assert_eq!(at_inf.len(), 2, "{at_inf:?}");
    assert!(at_inf[0].contains("1^∞"), "{at_inf:?}");
    assert!(at_inf[1].contains("(1 + 1/x)^x = e"), "{at_inf:?}");
}

#[test]
fn names_continuity_and_factor_cancel_methods() {
    for (before, after, needle) in [
        ("(x^2-1)/(x-1)", "2", "cancelar el factor común"),
        ("(x^2-4)/(x-2)", "4", "cancelar el factor común"),
        ("(x^3-1)/(x-1)", "3", "cancelar el factor común"),
        ("x^2+1", "5", "Sustitución directa"),
        ("3*x-1", "14", "Sustitución directa"),
    ] {
        let titles = substep_titles(before, after);
        assert_eq!(titles.len(), 1, "{before} should name one method");
        assert!(
            titles[0].contains(needle),
            "{before}: expected `{needle}` in `{}`",
            titles[0]
        );
    }
}

#[test]
fn declines_method_narration_for_coprime_rational_and_constant() {
    // (x^2+1)/(x-1): no shared factor (genuine pole) — not factor-and-cancel.
    assert!(substep_titles("(x^2+1)/(x-1)", "3").is_empty());
    // A constant has no variable, so the continuity narration does not apply.
    assert!(substep_titles("5", "5").is_empty());
    // A non-finite result (the symbolic limit form) is never narrated.
    assert!(substep_titles("x^2+1", "limit(x^2+1, x, 2)").is_empty());
}

#[test]
fn declines_when_result_is_not_the_notable_value() {
    // sin(x)/x at a non-zero point evaluates to sin(5)/5, NOT 1 — must not be narrated.
    assert!(substep_titles("sin(x)/x", "sin(5)/5").is_empty());
    // Scaled argument sin(a·u)/u narrates ONLY when the result equals the scale: sin(2x)/x → 3
    // is fabricated (real limit is 2) and must decline, while sin(2x)/x → 2 is narrated (see
    // names_the_standard_notable_limits). A constant offset (sin(x+1)/x) is not the notable form.
    assert!(substep_titles("sin(2*x)/x", "3").is_empty());
    assert!(substep_titles("sin(x+1)/x", "1").is_empty());
    // Cross-form must have a genuine notable on at least one side and the exact a/b result:
    // `cos(2x)/(3x)` (cos has no first-order equivalent) and a fabricated ratio both decline.
    assert!(substep_titles("cos(2*x)/(3*x)", "2/3").is_empty());
    assert!(substep_titles("sin(3*x)/(2*x)", "5").is_empty());
    // Binomial equivalent must be exactly `(1+u)^a − 1`: a different shift (√(2+x)) and a
    // fabricated value both decline.
    assert!(substep_titles("(sqrt(2+x)-1)/x", "1/2").is_empty());
    assert!(substep_titles("(sqrt(1+x)-1)/x", "3").is_empty());
    // Right structural form but wrong value (fabricated) must decline.
    assert!(substep_titles("sin(x)/x", "2").is_empty());
    // x·sin(x) → 0 is continuity, NOT the squeeze theorem (the argument is not a reciprocal).
    assert!(substep_titles("x*sin(x)", "0").is_empty());
    // (2^x − 1)/x → ln(3) would be a fabricated base; the result must be ln of the base.
    assert!(substep_titles("(2^x-1)/x", "ln(3)").is_empty());
}
