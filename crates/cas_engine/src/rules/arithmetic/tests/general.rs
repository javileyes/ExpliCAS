//! Tests de las reglas aritméticas: familia `general` (troceo P1).

use super::*;

#[test]
fn term_has_matrix_product_factor_flags_only_matrix_products() {
    // A matrix literal used as a factor of a multiplication is a
    // non-commutative product: the cancellation/reorder machinery must not
    // treat such a term as a commutative-ring element.
    let flagged = [
        "[[1,2],[3,4]]*[[5,6],[7,8]]",
        "[[1,2],[3,4]]*[[5,6],[7,8]] - [[5,6],[7,8]]*[[1,2],[3,4]]",
        "2*[[1,2],[3,4]]", // scalar·matrix still has a matrix factor
        "[[1,2],[3,4]]^2", // matrix power expands to repeated products
        "x*([[1,0],[0,1]]*[[1,2],[3,4]])",
    ];
    for input in flagged {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx).unwrap_or_else(|err| panic!("parse {input}: {err}"));
        assert!(
            term_has_matrix_product_factor(&ctx, expr),
            "{input}: expected matrix product factor to be flagged"
        );
    }

    // Matrices that appear only as additive terms or only inside a function
    // argument are NOT non-commutative products, so they stay unflagged
    // (commutative add-reordering and scalar matchers remain enabled).
    let unflagged = [
        "[[1,2],[3,4]] - [[1,2],[3,4]]", // matrices in a difference, not a product
        "[[1,2],[3,4]] + [[5,6],[7,8]]",
        "det([[1,2],[3,4]])*x", // matrix only inside det(...) argument
        "x*y - y*x",            // no matrices at all
        "(x+1)*(x-1)",
    ];
    for input in unflagged {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx).unwrap_or_else(|err| panic!("parse {input}: {err}"));
        assert!(
            !term_has_matrix_product_factor(&ctx, expr),
            "{input}: expected NO matrix product factor"
        );
    }
}
#[test]
fn subtraction_self_cancel_rule_matches_abs_sub_mirror_in_generic() {
    let mut ctx = Context::new();
    let expr = parse(
        "abs((2*u)/(u^2 - 1) - 1) - abs(1 - 2*u/(u^2 - 1))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = SubSelfToZeroRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
}
#[test]
fn integrate_prep_candidate_accepts_dirichlet_raw_difference() {
    let mut ctx = Context::new();
    let expr = parse(
        "sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::super::maybe_integrate_prep_exact_additive_candidate(
        &mut ctx, expr
    ));
}
#[test]
fn direct_core_equivalence_rewrite_rejects_scaled_symbolic_atom_mismatch_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("x", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("2*x", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}
#[test]
fn direct_core_equivalence_rewrite_keeps_signed_scaled_symbolic_atom_match_before_default_simplify()
{
    let mut ctx = Context::new();
    let lhs = parse("-x", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("-1*x", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
}
#[test]
fn direct_core_equivalence_rewrite_rejects_product_division_shared_scale_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("a * x", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("-(a * d / c)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}
#[test]
fn direct_core_equivalence_rewrite_keeps_cancelable_product_division_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("a * x", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("(a * x * y) / y", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_some());
}
#[test]
fn direct_core_equivalence_rewrite_keeps_atomic_direct_match_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("a", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("a", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_some());
}
#[test]
fn shared_passthrough_difference_shape_gate_rejects_additive_vs_single_term_residual() {
    let mut ctx = Context::new();
    let expr = parse("(x + x) - (2*x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        !super::super::has_plausible_shared_additive_passthrough_difference_shape(&mut ctx, expr)
    );
    assert!(
        super::super::try_build_exact_zero_shared_passthrough_difference_rewrite(&mut ctx, expr)
            .is_none()
    );
}
#[test]
fn direct_finite_sum_equivalence_rewrite_matches_telescoping_sum() {
    let mut ctx = Context::new();
    let lhs =
        parse("sum(1/(k*(k+1)), k, 1, n)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("1 - 1/(n+1)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_finite_sum_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Finite Telescoping Sum");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn direct_core_equivalence_rewrite_matches_telescoping_sum_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs =
        parse("sum(1/(k*(k+1)), k, 1, n)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("1 - 1/(n+1)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Finite Telescoping Sum");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn extract_scaled_double_sine_product_for_cancellation_matches_canonical_order() {
    let mut ctx = Context::new();
    let expr = parse("2*sin(2*x)*sin(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let (scale, arg) = extract_scaled_double_sine_product_for_cancellation(&mut ctx, expr)
        .unwrap_or_else(|| panic!("extract"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: scale
            }
        ),
        "1"
    );
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: arg
            }
        ),
        "x"
    );
}
