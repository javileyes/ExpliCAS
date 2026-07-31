//! Tests de las reglas aritméticas: familia `pairing` (troceo P1).

use super::*;

#[test]
fn direct_core_equivalence_rewrite_matches_grouped_symbolic_scale_sum_distribution_pair() {
    let mut ctx = Context::new();
    let lhs = parse("a*x^2 + c*x^2 + e*x^2 + b*x + d*x", &mut ctx)
        .unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs =
        parse("x*(b + d) + x^2*(a + c + e)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn direct_core_equivalence_rewrite_matches_reordered_additive_noncall_pair() {
    let mut ctx = Context::new();
    let lhs = parse("x + y + z", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("z + x + y", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn direct_core_equivalence_rewrite_rejects_atomic_noncall_pair_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("a", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("-b", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}
