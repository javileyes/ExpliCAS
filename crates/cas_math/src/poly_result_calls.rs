//! Shared helpers for `poly_result(...)` consumer calls.
//!
//! Keeps rule-level parsing and small expression builders out of `cas_engine`.

use crate::poly_result::{parse_poly_result_id, PolyResultId};
use cas_ast::{Context, Expr, ExprId};

/// Parsed `poly_stats(poly_result(id))` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolyStatsCall {
    pub poly_id: PolyResultId,
}

/// Parsed `poly_to_expr/poly_print/poly_latex(poly_result(id), max_terms?)` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolyResultWithLimitCall {
    pub poly_id: PolyResultId,
    pub max_terms: usize,
}

/// Parse `poly_stats(poly_result(id))`.
pub fn try_parse_poly_stats_call(ctx: &Context, expr: ExprId) -> Option<PolyStatsCall> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if ctx.sym_name(*fn_id) != "poly_stats" || args.len() != 1 {
        return None;
    }
    Some(PolyStatsCall {
        poly_id: parse_poly_result_id(ctx, args[0])?,
    })
}

/// Parse `poly_to_expr(poly_result(id))` or `poly_to_expr(poly_result(id), max_terms)`.
pub fn try_parse_poly_to_expr_call(
    ctx: &Context,
    expr: ExprId,
    default_max_terms: usize,
) -> Option<PolyResultWithLimitCall> {
    try_parse_poly_result_call_with_optional_limit(ctx, expr, "poly_to_expr", default_max_terms)
}

/// Parse `poly_print(poly_result(id))` or `poly_print(poly_result(id), max_terms)`.
pub fn try_parse_poly_print_call(
    ctx: &Context,
    expr: ExprId,
    default_max_terms: usize,
) -> Option<PolyResultWithLimitCall> {
    try_parse_poly_result_call_with_optional_limit(ctx, expr, "poly_print", default_max_terms)
}

/// Parse `poly_latex(poly_result(id))` or `poly_latex(poly_result(id), max_terms)`.
pub fn try_parse_poly_latex_call(
    ctx: &Context,
    expr: ExprId,
    default_max_terms: usize,
) -> Option<PolyResultWithLimitCall> {
    try_parse_poly_result_call_with_optional_limit(ctx, expr, "poly_latex", default_max_terms)
}

/// Build `poly_info(id, terms, vars, modp)` expression.
pub fn build_poly_info_expr(
    ctx: &mut Context,
    poly_id: PolyResultId,
    n_terms: usize,
    n_vars: usize,
) -> ExprId {
    let poly_id_expr = ctx.num(poly_id as i64);
    let terms = ctx.num(n_terms as i64);
    let vars = ctx.num(n_vars as i64);
    let repr = ctx.var("modp");
    ctx.call("poly_info", vec![poly_id_expr, terms, vars, repr])
}

/// Human-readable materialization capability note.
pub fn format_materialization_note(
    poly_id: PolyResultId,
    n_terms: usize,
    materialize_limit: usize,
) -> String {
    if n_terms <= materialize_limit {
        format!("materializable via poly_to_expr(poly_result({poly_id}))")
    } else {
        "too large to materialize".to_string()
    }
}

fn try_parse_poly_result_call_with_optional_limit(
    ctx: &Context,
    expr: ExprId,
    fn_name: &str,
    default_max_terms: usize,
) -> Option<PolyResultWithLimitCall> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if ctx.sym_name(*fn_id) != fn_name || args.is_empty() || args.len() > 2 {
        return None;
    }

    let poly_id = parse_poly_result_id(ctx, args[0])?;
    let max_terms = if args.len() == 2 {
        parse_max_terms_arg(ctx, args[1]).unwrap_or(default_max_terms)
    } else {
        default_max_terms
    };

    Some(PolyResultWithLimitCall { poly_id, max_terms })
}

fn parse_max_terms_arg(ctx: &Context, expr: ExprId) -> Option<usize> {
    match ctx.get(expr) {
        // Keep previous behavior: to_integer truncation + fallible usize cast.
        Expr::Number(n) => n.to_integer().try_into().ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{BuiltinFn, Expr};

    fn poly_result_expr(ctx: &mut Context, id: i64) -> ExprId {
        let id_expr = ctx.num(id);
        ctx.call_builtin(BuiltinFn::PolyResult, vec![id_expr])
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
        let parsed_bad =
            try_parse_poly_to_expr_call(&ctx, bad, 50_000).expect("parse with bad limit");
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
}
