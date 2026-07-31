//! Tests de `focused_rule_substeps`: `named_identity_matcher_tests`, extraídos del módulo.

use super::{
    generate_cos_2x_additive_contraction_substeps, generate_double_angle_contraction_substeps,
    generate_double_angle_expansion_substeps, generate_half_angle_square_identity_substeps,
    generate_half_angle_tangent_substeps, generate_pythagorean_factor_form_substeps,
    generate_reciprocal_product_identity_substeps, generate_reciprocal_pythagorean_substeps,
    generate_reciprocal_trig_identity_substeps, generate_sec_csc_squared_contraction_substeps,
    generate_sec_csc_squared_expansion_substeps, generate_split_log_exponents_substeps,
    generate_trig_quotient_substeps,
};
use crate::runtime::Step;
use cas_ast::Context;
use cas_parser::parse;

fn run<F>(generator: F, rule: &str, before_src: &str, after_src: &str) -> Vec<super::SubStep>
where
    F: Fn(&Context, &Step) -> Vec<super::SubStep>,
{
    let mut ctx = Context::new();
    let before = parse(before_src, &mut ctx).expect("parse before");
    let after = parse(after_src, &mut ctx).expect("parse after");
    let step = Step::new_compact("desc", rule, before, after);
    generator(&ctx, &step)
}

/// The TRUE application publishes, byte-identical to what it always said.
#[test]
fn tan_cot_still_narrates_its_genuine_instances() {
    let subs = run(
        generate_reciprocal_product_identity_substeps,
        "Reciprocal Product Identity",
        "tan(x^2) * cot(x^2)",
        "1",
    );
    assert_eq!(subs.len(), 1);
    assert_eq!(subs[0].description, "Usar tan(u) · cot(u) = 1");
}

/// The audit's first named lie: the same title used to publish over ANY
/// pair this generator received. A pair that instantiates nothing now
/// publishes nothing.
#[test]
fn tan_cot_declines_a_pair_that_is_no_instance() {
    let subs = run(
        generate_reciprocal_product_identity_substeps,
        "Reciprocal Product Identity",
        "sin(x) + 1",
        "1",
    );
    assert!(
        subs.is_empty(),
        "a non-instance pair must not be narrated as tan·cot = 1: {:?}",
        subs.iter().map(|s| &s.description).collect::<Vec<_>>()
    );
}

/// The second named lie lived in the `None` branch. Its GENUINE case is
/// the pair whose doubling already folded away (`u = x/2` makes `2u` read
/// `x`): no variant recognizer fires, yet the pair truly instantiates the
/// identity — the directed matcher rebuilds σ(rhs) in the instance context
/// and lets the engine's own equality own the comparison.
#[test]
fn half_angle_none_branch_narrates_a_genuine_tan_pair() {
    let subs = run(
        generate_half_angle_tangent_substeps,
        "Half-Angle Tangent Identity",
        "tan(x/2)",
        "(1 - cos(x)) / sin(x)",
    );
    assert_eq!(subs.len(), 1);
    assert_eq!(subs[0].description, "Usar tan(u) = (1 - cos(2u)) / sin(2u)");
}

/// A pair with the doubling EXPLICIT is a recognized variant — and the
/// orientation-agnostic matcher accepts it in the expansion direction too
/// (the variant was detected on `after`, so the pair applies the identity
/// right-to-left).
#[test]
fn half_angle_variant_accepts_the_expansion_direction() {
    let subs = run(
        generate_half_angle_tangent_substeps,
        "Half-Angle Tangent Identity",
        "tan(x/2)",
        "(1 - cos(2*(x/2))) / sin(2*(x/2))",
    );
    assert_eq!(subs.len(), 1);
    assert_eq!(subs[0].description, "Usar (1 - cos(2u)) / sin(2u) = tan(u)");
}

/// …and the unrecognized-variant pair the audit saw declines.
#[test]
fn half_angle_none_branch_declines_what_it_did_not_recognize() {
    let subs = run(
        generate_half_angle_tangent_substeps,
        "Half-Angle Tangent Identity",
        "(1 - cos(x)) / (1 + cos(x))",
        "tan(x/2)^2",
    );
    assert!(
        subs.is_empty(),
        "the branch that recognized no variant must not cite the identity: {:?}",
        subs.iter().map(|s| &s.description).collect::<Vec<_>>()
    );
}

/// The recognized variants keep narrating exactly as before.
#[test]
fn half_angle_recognized_variants_still_narrate() {
    let subs = run(
        generate_half_angle_tangent_substeps,
        "Half-Angle Tangent Identity",
        "(1 - cos(2*x)) / sin(2*x)",
        "tan(x)",
    );
    assert_eq!(subs.len(), 1);
    assert_eq!(subs[0].description, "Usar (1 - cos(2u)) / sin(2u) = tan(u)");
}

/// The shadow pass measured the old Trig Quotient emitter citing
/// «sin(u)/cos(u) = tan(u)» over EVERY pair of the rule — cot, sec and csc
/// included. Each pair now narrates ITS OWN census-adjudicated template.
#[test]
fn trig_quotient_narrates_each_definitional_pair() {
    for (before, after, expected) in [
        (
            "sin(x^2) / cos(x^2)",
            "tan(x^2)",
            "Usar sin(u) / cos(u) = tan(u)",
        ),
        ("cos(x) / sin(x)", "cot(x)", "Usar cos(u) / sin(u) = cot(u)"),
        ("1 / cos(2*x)", "sec(2*x)", "Usar 1 / cos(u) = sec(u)"),
        ("1 / sin(x)", "csc(x)", "Usar 1 / sin(u) = csc(u)"),
    ] {
        let subs = run(
            generate_trig_quotient_substeps,
            "Trig Quotient",
            before,
            after,
        );
        assert_eq!(subs.len(), 1, "pair {before} ⟹ {after} must narrate");
        assert_eq!(
            subs[0].description, expected,
            "each pair narrates its OWN identity, never a neighbour's"
        );
    }
}

/// The derive-route shadow measured this rule 6/6 census-covered; the
/// engine's application description routes each pair to its ORIENTED
/// template, and the matcher gates the instance.
#[test]
fn reciprocal_trig_identity_narrates_each_described_pair() {
    let run_desc = |desc: &str, before_src: &str, after_src: &str| {
        let mut ctx = Context::new();
        let before = parse(before_src, &mut ctx).expect("parse before");
        let after = parse(after_src, &mut ctx).expect("parse after");
        let step = Step::new_compact(desc, "Reciprocal Trig Identity", before, after);
        generate_reciprocal_trig_identity_substeps(&ctx, &step)
    };
    for (desc, before, after, expected) in [
        (
            "Expand sec(u) as 1 / cos(u)",
            "sec(x)",
            "1/cos(x)",
            "Usar sec(u) = 1 / cos(u)",
        ),
        (
            "Recognize 1 / sin(u) as csc(u)",
            "1/sin(2*x)",
            "csc(2*x)",
            "Usar 1 / sin(u) = csc(u)",
        ),
        (
            "Expand cot(u) as cos(u) / sin(u)",
            "cot(x^2)",
            "cos(x^2)/sin(x^2)",
            "Usar cot(u) = cos(u) / sin(u)",
        ),
    ] {
        let subs = run_desc(desc, before, after);
        assert_eq!(subs.len(), 1, "pair {before} ⟹ {after} must narrate");
        assert_eq!(subs[0].description, expected);
    }
    // Described but NOT an instance: the engine's description says sec
    // expansion, the pair is something else — the matcher declines what
    // the old emitter would have cited.
    let subs = run_desc("Expand sec(u) as 1 / cos(u)", "sin(x) + 1", "1/cos(x)");
    assert!(
        subs.is_empty(),
        "a described-but-non-instance pair must decline: {:?}",
        subs.iter().map(|s| &s.description).collect::<Vec<_>>()
    );
}

/// The four sec²/csc² rules are RULE-NAME routed (one identity each), so
/// the narration is description-agnostic: the derive spelling and the
/// simplify spelling of the same application both narrate, and a
/// non-instance pair declines regardless of what the description claims.
#[test]
fn sec_csc_squared_emitters_narrate_by_rule_name_and_decline_non_instances() {
    let run = |rule: &str, desc: &str, before_src: &str, after_src: &str| {
        let mut ctx = Context::new();
        let before = parse(before_src, &mut ctx).expect("parse before");
        let after = parse(after_src, &mut ctx).expect("parse after");
        let step = Step::new_compact(desc, rule, before, after);
        if rule.starts_with("Expand") {
            generate_sec_csc_squared_expansion_substeps(&ctx, &step)
        } else {
            generate_sec_csc_squared_contraction_substeps(&ctx, &step)
        }
    };
    for (rule, desc, before, after, expected) in [
        (
            "Expand Secant Squared",
            "Expand sec²(u) as 1 + tan(u)^2",
            "sec(x)^2",
            "1 + tan(x)^2",
            "Usar sec(u)^2 = 1 + tan(u)^2",
        ),
        (
            "Expand Cosecant Squared",
            "Expand csc²(u) as 1 + cot(u)^2",
            "csc(2*x)^2",
            "1 + cot(2*x)^2",
            "Usar csc(u)^2 = 1 + cot(u)^2",
        ),
        // The SIMPLIFY route spells the same application differently —
        // the rule-name routing must not care.
        (
            "Recognize Secant Squared",
            "1 + tan²(x) = sec²(x)",
            "tan(x)^2 + 1",
            "sec(x)^2",
            "Usar 1 + tan(u)^2 = sec(u)^2",
        ),
        (
            "Recognize Cosecant Squared",
            "Recognize 1 + cot²(u) as csc²(u)",
            "cot(x^2)^2 + 1",
            "csc(x^2)^2",
            "Usar 1 + cot(u)^2 = csc(u)^2",
        ),
    ] {
        let subs = run(rule, desc, before, after);
        assert_eq!(subs.len(), 1, "pair {before} ⟹ {after} must narrate");
        assert_eq!(subs[0].description, expected);
    }
    // A pair that is NOT an instance declines even under the right rule.
    let subs = run(
        "Recognize Secant Squared",
        "Recognize 1 + tan²(u) as sec²(u)",
        "tan(x)^2 + 2",
        "sec(x)^2",
    );
    assert!(
        subs.is_empty(),
        "a non-instance must decline: {:?}",
        subs.iter().map(|s| &s.description).collect::<Vec<_>>()
    );
}

/// `Double Angle Expansion` publishes an identity in BOTH directions
/// under the same engine description, so orientation can only come from
/// the pair. The decisive case is the `cos(2u)` family: its two rows are
/// equivalent by Pythagoras, and a directed match would accept either —
/// each pair must cite the form on screen, not its equivalent.
#[test]
fn double_angle_expansion_cites_the_orientation_on_screen() {
    let run = |before_src: &str, after_src: &str| {
        let mut ctx = Context::new();
        let before = parse(before_src, &mut ctx).expect("parse before");
        let after = parse(after_src, &mut ctx).expect("parse after");
        let step = Step::new_compact(
            "Expand double-angle sine",
            "Double Angle Expansion",
            before,
            after,
        );
        generate_double_angle_expansion_substeps(&ctx, &step)
    };
    for (before, after, expected) in [
        (
            "sin(2*x)",
            "2*sin(x)*cos(x)",
            "Usar sin(2u) = 2 · sin(u) · cos(u)",
        ),
        (
            "2*sin(x)*cos(x)",
            "sin(2*x)",
            "Usar 2·sin(u)·cos(u) = sin(2u)",
        ),
        (
            "cos(2*x)",
            "1-2*sin(x)^2",
            "Usar cos(2u) = 1 - 2 · sin(u)^2",
        ),
        (
            "cos(2*x)",
            "2*cos(x)^2-1",
            "Usar cos(2u) = 2 · cos(u)^2 - 1",
        ),
        ("1-2*sin(x)^2", "cos(2*x)", "Usar 1 - 2·sin(u)^2 = cos(2u)"),
        ("2*cos(x)^2-1", "cos(2*x)", "Usar 2·cos(u)^2 - 1 = cos(2u)"),
    ] {
        let subs = run(before, after);
        assert_eq!(subs.len(), 1, "pair {before} ⟹ {after} must narrate");
        assert_eq!(
            subs[0].description, expected,
            "the cited form must be the one printed, never its Pythagorean twin"
        );
    }
    // Non-instances decline: the half-scaled contraction the shadow lists
    // as uncovered, and a fabricated pair.
    assert!(run("sin(x)*cos(x)", "sin(2*x)/2").is_empty());
    assert!(run("sin(2*x)", "2*sin(x)*sin(x)").is_empty());
}

/// `Split Log Exponents` cites plain exponential algebra — the identity
/// is FREE of domain conditions even though the rule lives among the log
/// rules — and the inline `e^(log_e x) ⟹ x` fold is rescued by the
/// directed pass. A pair that is not an instance at all declines.
#[test]
fn split_log_exponents_narrates_the_exponential_split_and_declines_non_instances() {
    let run = |before_src: &str, after_src: &str| {
        let mut ctx = Context::new();
        let before = parse(before_src, &mut ctx).expect("parse before");
        let after = parse(after_src, &mut ctx).expect("parse after");
        let step = Step::new_compact(
            "e^(a+b) -> e^a * e^b (log cancellation)",
            "Split Log Exponents",
            before,
            after,
        );
        generate_split_log_exponents_substeps(&ctx, &step)
    };
    // The printed instance: both factors survive the split.
    let subs = run("e^(2*ln(x) + 3*ln(y))", "e^(2*ln(x))*e^(3*ln(y))");
    assert_eq!(subs.len(), 1);
    assert_eq!(subs[0].description, "Usar e^(A+B) = e^A · e^B");
    // The FOLDED instance: `e^(log_e x)` collapsed to `x` on the way out,
    // so the pair no longer looks like the identity — the directed pass
    // rescues it because the equality still holds.
    let subs = run("e^(log(e,x) + 5)", "x*e^5");
    assert_eq!(subs.len(), 1, "the folded instance must still narrate");
    assert_eq!(subs[0].description, "Usar e^(A+B) = e^A · e^B");
    // Not an instance under any reading: the sum did not become a product.
    let subs = run("e^(a + b)", "e^a + e^b");
    assert!(
        subs.is_empty(),
        "a non-instance must decline: {:?}",
        subs.iter().map(|s| &s.description).collect::<Vec<_>>()
    );
}

/// Half-angle squares: the four pairs the derive-route shadow measured
/// (structural 4/4) narrate their OWN oriented identity — expansion and
/// recognition are different gestures and must not borrow each other's
/// title — and the `cos(2u)` arms keep their shape-of-`after` routing.
/// A described-but-non-instance pair declines, which is what leaving the
/// silenced list is allowed to cost.
#[test]
fn half_angle_square_identity_narrates_each_oriented_gesture() {
    let run = |rule: &str, desc: &str, before_src: &str, after_src: &str| {
        let mut ctx = Context::new();
        let before = parse(before_src, &mut ctx).expect("parse before");
        let after = parse(after_src, &mut ctx).expect("parse after");
        let step = Step::new_compact(desc, rule, before, after);
        generate_half_angle_square_identity_substeps(&ctx, &step)
    };
    for (rule, desc, before, after, expected) in [
        (
            "Half-Angle Square Identity",
            "Expand sin²(u) as (1 - cos(2u))/2",
            "sin(x)^2",
            "(1 - cos(2*x))/2",
            "Usar sin²(u) = (1 - cos(2u)) / 2",
        ),
        (
            "Half-Angle Square Identity",
            "Expand cos²(u) as (1 + cos(2u))/2",
            "cos(3*x)^2",
            "(1 + cos(6*x))/2",
            "Usar cos²(u) = (1 + cos(2u)) / 2",
        ),
        // The INVERSE gesture cites the inverse orientation, not the one
        // it undoes: recognition is what the reader is watching.
        (
            "Half-Angle Square Identity",
            "Recognize (1 - cos(2u))/2 as sin²(u)",
            "(1 - cos(2*x))/2",
            "sin(x)^2",
            "Usar (1 - cos(2u)) / 2 = sin²(u)",
        ),
        (
            "Half-Angle Square Identity",
            "Recognize (1 + cos(2u))/2 as cos²(u)",
            "(1 + cos(2*x))/2",
            "cos(x)^2",
            "Usar (1 + cos(2u)) / 2 = cos²(u)",
        ),
        // The two `cos(2u)` right-hand sides are EQUIVALENT, so the engine
        // description cannot disambiguate them: the shape of `after` does,
        // and the matcher still gates the instance.
        (
            "Angle Consistency (Half-Angle)",
            "Half-Angle Expansion",
            "cos(2*x)",
            "2*cos(x)^2 - 1",
            "Usar cos(2u) = 2 · cos(u)^2 - 1",
        ),
        (
            "Angle Consistency (Half-Angle)",
            "Half-Angle Expansion",
            "cos(2*x)",
            "1 - 2*sin(x)^2",
            "Usar cos(2u) = 1 - 2 · sin(u)^2",
        ),
    ] {
        let subs = run(rule, desc, before, after);
        assert_eq!(subs.len(), 1, "pair {before} ⟹ {after} must narrate");
        assert_eq!(
            subs[0].description, expected,
            "each gesture cites the orientation the reader sees"
        );
    }
    // Described as a half-angle expansion, but the pair is not an
    // instance: the matcher declines instead of publishing the title the
    // pre-migration emitter would have printed unconditionally.
    let subs = run(
        "Half-Angle Square Identity",
        "Expand sin²(u) as (1 - cos(2u))/2",
        "sin(x)^2",
        "(1 - cos(3*x))/2",
    );
    assert!(
        subs.is_empty(),
        "a described-but-non-instance pair must decline: {:?}",
        subs.iter().map(|s| &s.description).collect::<Vec<_>>()
    );
}

/// Factor-form and reciprocal-Pythagorean tables: every oriented family
/// the derive route produces narrates its OWN censused identity — the
/// expansion and negated variants the old prefix-routed emitter missed
/// included — and non-instances decline.
#[test]
fn pythagorean_factor_and_reciprocal_tables_narrate_each_family() {
    let run_pff = |before_src: &str, after_src: &str| {
        let mut ctx = Context::new();
        let before = parse(before_src, &mut ctx).expect("parse before");
        let after = parse(after_src, &mut ctx).expect("parse after");
        let step = Step::new_compact("any", "Pythagorean Factor Form", before, after);
        generate_pythagorean_factor_form_substeps(&ctx, &step)
    };
    for (before, after, expected) in [
        ("1 - sin(x)^2", "cos(x)^2", "Usar 1 - sin(u)^2 = cos(u)^2"),
        // Expansion direction (the shadow's uncovered witness).
        ("sin(x)^2", "1 - cos(x)^2", "Usar 1 - cos(u)^2 = sin(u)^2"),
        // Negated variant.
        ("sin(x)^2 - 1", "-cos(x)^2", "Usar sin(u)^2 - 1 = -cos(u)^2"),
    ] {
        let subs = run_pff(before, after);
        assert_eq!(subs.len(), 1, "pair {before} ⟹ {after} must narrate");
        assert_eq!(subs[0].description, expected);
    }
    let subs = run_pff("1 - sin(x)^2", "cos(2*x)^2");
    assert!(
        subs.is_empty(),
        "a non-instance must decline: {:?}",
        subs.iter().map(|s| &s.description).collect::<Vec<_>>()
    );

    let run_rpi = |before_src: &str, after_src: &str| {
        let mut ctx = Context::new();
        let before = parse(before_src, &mut ctx).expect("parse before");
        let after = parse(after_src, &mut ctx).expect("parse after");
        let step = Step::new_compact("any", "Reciprocal Pythagorean Identity", before, after);
        generate_reciprocal_pythagorean_substeps(&ctx, &step)
    };
    for (before, after, expected) in [
        ("sec(x)^2 - tan(x)^2", "1", "Usar sec(u)^2 - tan(u)^2 = 1"),
        ("csc(x)^2 - 1", "cot(x)^2", "Usar csc(u)^2 - 1 = cot(u)^2"),
        // Expansion arm: same censused row, inverted application.
        ("tan(x)^2", "sec(x)^2 - 1", "Usar sec(u)^2 - 1 = tan(u)^2"),
        // Negated family.
        ("1 - sec(x)^2", "-tan(x)^2", "Usar 1 - sec(u)^2 = -tan(u)^2"),
    ] {
        let subs = run_rpi(before, after);
        assert_eq!(subs.len(), 1, "pair {before} ⟹ {after} must narrate");
        assert_eq!(subs[0].description, expected);
    }
    let subs = run_rpi("sec(x)^2 - tan(x)^2", "2");
    assert!(
        subs.is_empty(),
        "a non-instance must decline: {:?}",
        subs.iter().map(|s| &s.description).collect::<Vec<_>>()
    );
}

/// A pair that instantiates none of the quotient templates publishes
/// nothing — the honest silence the migration policy demands.
#[test]
fn trig_quotient_declines_a_pair_that_is_no_instance() {
    let subs = run(
        generate_trig_quotient_substeps,
        "Trig Quotient",
        "sin(x) + 1",
        "tan(x)",
    );
    assert!(
        subs.is_empty(),
        "a non-instance pair must not cite any quotient identity: {:?}",
        subs.iter().map(|s| &s.description).collect::<Vec<_>>()
    );
}

/// The extended shadow pass caught the double-angle emitter citing the
/// SINE identity over the cosine pair. Each pair now narrates its own.
#[test]
fn double_angle_contraction_narrates_each_pair_with_its_own_identity() {
    for (before, after, expected) in [
        (
            "2*sin(x)*cos(x)",
            "sin(2*x)",
            "Usar 2·sin(u)·cos(u) = sin(2u)",
        ),
        (
            "cos(x)^2 - sin(x)^2",
            "cos(2*x)",
            "Usar cos(u)^2 - sin(u)^2 = cos(2u)",
        ),
    ] {
        let subs = run(
            generate_double_angle_contraction_substeps,
            "Double Angle Contraction",
            before,
            after,
        );
        assert_eq!(subs.len(), 1, "pair {before} ⟹ {after} must narrate");
        assert_eq!(subs[0].description, expected);
    }
}

/// The old Cos-2x emitter picked its template by substring-sniffing the
/// display, so the SCALED pair `4·cos²−2 ⟹ 2·cos(2x)` was cited without
/// being an instance. With coefficient peeling (pass 3) it narrates the
/// identity it genuinely uses; a mismatched-scale pair still declines.
#[test]
fn cos_2x_additive_contraction_narrates_instances_and_declines_the_scaled_pair() {
    for (before, after, expected) in [
        (
            "2*cos(x)^2 - 1",
            "cos(2*x)",
            "Usar 2·cos(u)^2 - 1 = cos(2u)",
        ),
        (
            "1 - 2*sin(x)^2",
            "cos(2*x)",
            "Usar 1 - 2·sin(u)^2 = cos(2u)",
        ),
    ] {
        let subs = run(
            generate_cos_2x_additive_contraction_substeps,
            "Cos 2x Additive Contraction",
            before,
            after,
        );
        assert_eq!(subs.len(), 1, "pair {before} ⟹ {after} must narrate");
        assert_eq!(subs[0].description, expected);
    }
    // The SCALED pair applies the identity inside a linear combination —
    // pass 3 (coefficient peeling) recognizes it and names the identity
    // actually used, where the old emitter mis-cited and the first
    // migration declined.
    let subs = run(
        generate_cos_2x_additive_contraction_substeps,
        "Cos 2x Additive Contraction",
        "4*cos(x)^2 - 2",
        "2*cos(2*x)",
    );
    assert_eq!(subs.len(), 1, "the scaled pair must narrate via peeling");
    assert_eq!(subs[0].description, "Usar 2·cos(u)^2 - 1 = cos(2u)");
    // A pair whose sides shed DIFFERENT factors is not an application.
    let subs = run(
        generate_cos_2x_additive_contraction_substeps,
        "Cos 2x Additive Contraction",
        "4*cos(x)^2 - 2",
        "3*cos(2*x)",
    );
    assert!(
        subs.is_empty(),
        "mismatched scale factors must decline: {:?}",
        subs.iter().map(|s| &s.description).collect::<Vec<_>>()
    );
}
