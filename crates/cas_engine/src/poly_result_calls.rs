//! Shared helpers for `poly_result(...)` consumer calls used by engine rules.

use cas_ast::{Context, Expr, ExprId};
use cas_math::poly_result::{parse_poly_result_id, PolyResultId};

/// Default limit for showing poly materialization hint in `poly_stats`.
pub const DEFAULT_POLY_STATS_MATERIALIZE_LIMIT: usize = 50_000;
/// Default max terms for `poly_to_expr(poly_result(id))`.
pub const DEFAULT_POLY_TO_EXPR_MAX_TERMS: usize = 50_000;
/// Default max terms for `poly_print(poly_result(id))`.
pub const DEFAULT_POLY_PRINT_MAX_TERMS: usize = 1000;
/// Default max terms for `poly_latex(poly_result(id))`.
pub const DEFAULT_POLY_LATEX_MAX_TERMS: usize = 100;

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

/// Rewrite helper for `poly_stats(poly_result(id)) -> poly_info(...)`.
///
/// Returns rewritten expression and human-readable description.
pub fn rewrite_poly_stats_call_with_materialize_limit(
    ctx: &mut Context,
    expr: ExprId,
    materialize_limit: usize,
) -> Option<(ExprId, String)> {
    let call = try_parse_poly_stats_call(ctx, expr)?;
    let meta = cas_math::poly_store::thread_local_meta(call.poly_id)?;
    let rewritten = build_poly_info_expr(ctx, call.poly_id, meta.n_terms, meta.n_vars);
    let materialize_note =
        format_materialization_note(call.poly_id, meta.n_terms, materialize_limit);
    let desc = format!(
        "Poly #{}: {} terms, {} vars, repr=modp, {}",
        call.poly_id, meta.n_terms, meta.n_vars, materialize_note
    );
    Some((rewritten, desc))
}

/// Rewrite helper for `poly_to_expr(poly_result(id), max_terms?)`.
///
/// Returns rewritten expression and human-readable description.
///
/// If stored polynomial exceeds `max_terms`, returns an inline error symbol
/// expression to preserve previous user-facing behavior.
pub fn rewrite_poly_to_expr_call_with_default_limit(
    ctx: &mut Context,
    expr: ExprId,
    default_max_terms: usize,
) -> Option<(ExprId, String)> {
    let call = try_parse_poly_to_expr_call(ctx, expr, default_max_terms)?;
    let meta = cas_math::poly_store::thread_local_meta(call.poly_id)?;
    if meta.n_terms > call.max_terms {
        let message_expr = ctx.var(&format!(
            "Error: {} terms exceeds limit {}",
            meta.n_terms, call.max_terms
        ));
        let desc = format!(
            "poly_to_expr: {} terms > limit {}",
            meta.n_terms, call.max_terms
        );
        return Some((message_expr, desc));
    }

    let materialized = cas_math::poly_store::materialize_poly_result_expr(ctx, call.poly_id)?;
    let desc = format!("Materialized polynomial: {} terms", meta.n_terms);
    Some((materialized, desc))
}

/// Rewrite helper for `poly_print(poly_result(id), max_terms?)`.
///
/// Returns rewritten expression and human-readable description.
pub fn rewrite_poly_print_call_with_default_limit(
    ctx: &mut Context,
    expr: ExprId,
    default_max_terms: usize,
) -> Option<(ExprId, String)> {
    let call = try_parse_poly_print_call(ctx, expr, default_max_terms)?;
    let meta = cas_math::poly_store::thread_local_meta(call.poly_id)?;
    let formatted = cas_math::poly_store::render_poly_result(call.poly_id, call.max_terms)?;
    let rewritten = ctx.var(&formatted);
    let desc = if meta.n_terms > call.max_terms {
        format!(
            "Formatted polynomial: {} of {} terms shown (no AST)",
            call.max_terms, meta.n_terms
        )
    } else {
        format!("Formatted polynomial: {} terms (no AST)", meta.n_terms)
    };
    Some((rewritten, desc))
}

/// Rewrite helper for `poly_latex(poly_result(id), max_terms?)`.
///
/// Returns rewritten expression and human-readable description.
pub fn rewrite_poly_latex_call_with_default_limit(
    ctx: &mut Context,
    expr: ExprId,
    default_max_terms: usize,
) -> Option<(ExprId, String)> {
    let call = try_parse_poly_latex_call(ctx, expr, default_max_terms)?;
    let meta = cas_math::poly_store::thread_local_meta(call.poly_id)?;
    let formatted = cas_math::poly_store::render_poly_result_latex(call.poly_id, call.max_terms)?;
    let rewritten = ctx.var(&formatted);
    let desc = format!(
        "LaTeX polynomial: {} terms",
        meta.n_terms.min(call.max_terms)
    );
    Some((rewritten, desc))
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
#[path = "poly_result_calls_tests.rs"]
mod tests;
