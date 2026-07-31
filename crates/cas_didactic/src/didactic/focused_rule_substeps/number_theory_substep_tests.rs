//! Tests de `focused_rule_substeps`: `number_theory_substep_tests`, extraídos del módulo.

use super::{
    number_theory_divisors_substeps, number_theory_fibonacci_substeps, number_theory_gcd_substeps,
    number_theory_lcm_substeps, number_theory_sigma_substeps, number_theory_totient_substeps,
};

#[test]
fn gcd_shows_euclidean_remainder_chain() {
    let subs = number_theory_gcd_substeps(48, 36);
    assert_eq!(subs.len(), 1);
    assert_eq!(subs[0].before_expr, "gcd(48, 36)");
    assert_eq!(subs[0].after_expr, "gcd(36, 12) = gcd(12, 0) = 12");
}

#[test]
fn lcm_uses_product_over_gcd() {
    let subs = number_theory_lcm_substeps(4, 6);
    assert_eq!(subs[0].after_expr, "(4 · 6) / gcd(4, 6) = 24 / 2 = 12");
}

#[test]
fn totient_factorizes_then_applies_euler_formula() {
    let subs = number_theory_totient_substeps(12);
    assert_eq!(subs.len(), 2);
    assert_eq!(subs[0].after_expr, "12 = 2^2 · 3");
    assert_eq!(subs[1].after_expr, "12 · (1 - 1/2) · (1 - 1/3) = 4");
}

#[test]
fn divisors_factorizes_then_lists() {
    let subs = number_theory_divisors_substeps(12);
    assert_eq!(subs[0].before_expr, "12 = 2^2 · 3");
    assert_eq!(subs[0].after_expr, "[1, 2, 3, 4, 6, 12]");
}

#[test]
fn sigma_sums_divisors() {
    let subs = number_theory_sigma_substeps(12);
    assert_eq!(subs[0].after_expr, "1 + 2 + 3 + 4 + 6 + 12 = 28");
}

#[test]
fn fibonacci_shows_sequence() {
    let subs = number_theory_fibonacci_substeps(10);
    assert_eq!(
        subs[0].after_expr,
        "0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55 → 55"
    );
}

#[test]
fn out_of_range_inputs_emit_no_substep() {
    assert!(number_theory_fibonacci_substeps(200).is_empty());
    assert!(number_theory_totient_substeps(1).is_empty());
}
