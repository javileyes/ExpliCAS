use super::*;
use cas_ast::{BuiltinFn, Expr};
use cas_math::multipoly_modp::MultiPolyModP;
use cas_math::poly_store::{clear_thread_local_store, thread_local_insert, PolyMeta};

fn poly_result_expr(ctx: &mut Context, id: i64) -> ExprId {
    let id_expr = ctx.num(id);
    ctx.call_builtin(BuiltinFn::PolyResult, vec![id_expr])
}

fn insert_test_poly() -> PolyResultId {
    clear_thread_local_store();
    let meta = PolyMeta {
        modulus: 101,
        n_terms: 1,
        n_vars: 1,
        max_total_degree: 0,
        var_names: vec!["x".to_string()],
    };
    let poly = MultiPolyModP::constant(1, 101, 1);
    thread_local_insert(meta, poly)
}

#[test]
fn parse_poly_stats_call_matches_expected_shape() {
    let mut ctx = Context::new();
    let arg = poly_result_expr(&mut ctx, 7);
    let expr = ctx.call("poly_stats", vec![arg]);
    let call = try_parse_poly_stats_call(&ctx, expr).expect("expected match");
    assert_eq!(call.poly_id, 7);
}

#[test]
fn parse_poly_stats_call_rejects_bad_shapes() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let expr = ctx.call("poly_stats", vec![x]);
    assert!(try_parse_poly_stats_call(&ctx, expr).is_none());
}

#[test]
fn parse_poly_to_expr_uses_default_when_limit_missing_or_invalid() {
    let mut ctx = Context::new();
    let arg = poly_result_expr(&mut ctx, 3);
    let bare = ctx.call("poly_to_expr", vec![arg]);
    let parsed = try_parse_poly_to_expr_call(&ctx, bare, 50_000).expect("parse bare");
    assert_eq!(parsed.max_terms, 50_000);

    let k = ctx.var("k");
    let bad = ctx.call("poly_to_expr", vec![arg, k]);
    let parsed_bad = try_parse_poly_to_expr_call(&ctx, bad, 50_000).expect("parse with bad limit");
    assert_eq!(parsed_bad.max_terms, 50_000);
}

#[test]
fn parse_poly_print_reads_numeric_limit() {
    let mut ctx = Context::new();
    let arg = poly_result_expr(&mut ctx, 11);
    let max_terms = ctx.num(123);
    let expr = ctx.call("poly_print", vec![arg, max_terms]);
    let parsed = try_parse_poly_print_call(&ctx, expr, 1000).expect("parse");
    assert_eq!(parsed.poly_id, 11);
    assert_eq!(parsed.max_terms, 123);
}

#[test]
fn build_poly_info_expr_has_expected_shape() {
    let mut ctx = Context::new();
    let expr = build_poly_info_expr(&mut ctx, 9, 17, 3);
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        assert_eq!(ctx.sym_name(*fn_id), "poly_info");
        assert_eq!(args.len(), 4);
    } else {
        panic!("expected poly_info call");
    }
}

#[test]
fn format_materialization_note_switches_on_limit() {
    let small = format_materialization_note(2, 100, 1_000);
    let large = format_materialization_note(2, 2_000, 1_000);
    assert!(small.contains("poly_to_expr(poly_result(2))"));
    assert_eq!(large, "too large to materialize");
}

#[test]
fn rewrite_poly_stats_call_with_materialize_limit_builds_poly_info() {
    let mut ctx = Context::new();
    let id = insert_test_poly();
    let poly = poly_result_expr(&mut ctx, id as i64);
    let expr = ctx.call("poly_stats", vec![poly]);

    let rewritten =
        rewrite_poly_stats_call_with_materialize_limit(&mut ctx, expr, 50_000).expect("rewrite");
    if let Expr::Function(fn_id, args) = ctx.get(rewritten.0) {
        assert_eq!(ctx.sym_name(*fn_id), "poly_info");
        assert_eq!(args.len(), 4);
    } else {
        panic!("expected poly_info call");
    }
    assert!(rewritten.1.contains("Poly #"));
}

#[test]
fn rewrite_poly_to_expr_call_with_default_limit_materializes_poly() {
    let mut ctx = Context::new();
    let id = insert_test_poly();
    let poly = poly_result_expr(&mut ctx, id as i64);
    let expr = ctx.call("poly_to_expr", vec![poly]);

    let rewritten =
        rewrite_poly_to_expr_call_with_default_limit(&mut ctx, expr, 50_000).expect("rewrite");
    assert!(rewritten.1.contains("Materialized polynomial"));
}

#[test]
fn rewrite_poly_print_call_with_default_limit_formats_without_ast() {
    let mut ctx = Context::new();
    let id = insert_test_poly();
    let poly = poly_result_expr(&mut ctx, id as i64);
    let expr = ctx.call("poly_print", vec![poly]);

    let rewritten =
        rewrite_poly_print_call_with_default_limit(&mut ctx, expr, 1000).expect("rewrite");
    if let Expr::Variable(sym_id) = ctx.get(rewritten.0) {
        assert!(!ctx.sym_name(*sym_id).is_empty());
    } else {
        panic!("expected variable symbol carrying rendered text");
    }
    assert!(rewritten.1.contains("Formatted polynomial"));
}

#[test]
fn rewrite_poly_latex_call_with_default_limit_formats_latex() {
    let mut ctx = Context::new();
    let id = insert_test_poly();
    let poly = poly_result_expr(&mut ctx, id as i64);
    let expr = ctx.call("poly_latex", vec![poly]);

    let rewritten =
        rewrite_poly_latex_call_with_default_limit(&mut ctx, expr, 100).expect("rewrite");
    if let Expr::Variable(sym_id) = ctx.get(rewritten.0) {
        assert!(!ctx.sym_name(*sym_id).is_empty());
    } else {
        panic!("expected variable symbol carrying rendered LaTeX");
    }
    assert!(rewritten.1.contains("LaTeX polynomial"));
}
