//! Statistics and truncation helpers for eval-json output.

use cas_api_models::ExprStatsJson;
use cas_ast::{Context, Expr, ExprId};
use cas_formatter::DisplayExpr;

/// Render expression for eval-json with max length truncation.
pub(crate) fn format_expr_limited_eval_json(
    ctx: &Context,
    expr: ExprId,
    max_chars: usize,
) -> (String, bool, usize) {
    if let Some(poly_str) = cas_solver::try_render_poly_result(ctx, expr) {
        let len = poly_str.chars().count();
        if len <= max_chars {
            return (poly_str, false, len);
        }
        let truncated: String = poly_str.chars().take(max_chars).collect();
        return (format!("{truncated} … <truncated>"), true, len);
    }

    let full = format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: expr
        }
    );
    let len = full.chars().count();

    if len <= max_chars {
        return (full, false, len);
    }

    let truncated: String = full.chars().take(max_chars).collect();
    (format!("{truncated} … <truncated>"), true, len)
}

fn count_add_terms(ctx: &Context, expr: ExprId) -> Option<usize> {
    let inner_expr = match ctx.get(expr) {
        Expr::Function(name, args)
            if ctx.is_builtin(*name, cas_ast::BuiltinFn::Hold) && args.len() == 1 =>
        {
            args[0]
        }
        _ => expr,
    };

    if !matches!(ctx.get(inner_expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let mut count = 0usize;
    let mut stack = vec![inner_expr];

    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Add(l, r) | Expr::Sub(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            _ => {
                count += 1;
            }
        }
    }

    if count > 1 {
        Some(count)
    } else {
        None
    }
}

/// Compute node/depth/term stats for eval-json payload.
pub(crate) fn expr_stats_eval_json(ctx: &Context, expr: ExprId) -> ExprStatsJson {
    let (node_count, depth) = cas_ast::traversal::count_nodes_and_max_depth(ctx, expr);
    let term_count = cas_solver::try_get_poly_result_term_count(ctx, expr)
        .or_else(|| count_add_terms(ctx, expr));

    ExprStatsJson {
        node_count,
        depth,
        term_count,
    }
}

fn hash_expr_recursive<H: std::hash::Hasher>(ctx: &Context, expr: ExprId, hasher: &mut H) {
    use std::hash::Hash;

    match ctx.get(expr) {
        Expr::Number(n) => {
            0u8.hash(hasher);
            n.numer().to_string().hash(hasher);
            n.denom().to_string().hash(hasher);
        }
        Expr::Variable(name) => {
            1u8.hash(hasher);
            name.hash(hasher);
        }
        Expr::Constant(c) => {
            2u8.hash(hasher);
            format!("{:?}", c).hash(hasher);
        }
        Expr::SessionRef(id) => {
            11u8.hash(hasher);
            id.hash(hasher);
        }
        Expr::Hold(inner) => {
            12u8.hash(hasher);
            hash_expr_recursive(ctx, *inner, hasher);
        }
        Expr::Add(l, r) => {
            3u8.hash(hasher);
            hash_expr_recursive(ctx, *l, hasher);
            hash_expr_recursive(ctx, *r, hasher);
        }
        Expr::Sub(l, r) => {
            4u8.hash(hasher);
            hash_expr_recursive(ctx, *l, hasher);
            hash_expr_recursive(ctx, *r, hasher);
        }
        Expr::Mul(l, r) => {
            5u8.hash(hasher);
            hash_expr_recursive(ctx, *l, hasher);
            hash_expr_recursive(ctx, *r, hasher);
        }
        Expr::Div(l, r) => {
            6u8.hash(hasher);
            hash_expr_recursive(ctx, *l, hasher);
            hash_expr_recursive(ctx, *r, hasher);
        }
        Expr::Pow(l, r) => {
            7u8.hash(hasher);
            hash_expr_recursive(ctx, *l, hasher);
            hash_expr_recursive(ctx, *r, hasher);
        }
        Expr::Neg(inner) => {
            8u8.hash(hasher);
            hash_expr_recursive(ctx, *inner, hasher);
        }
        Expr::Function(name, args) => {
            9u8.hash(hasher);
            name.hash(hasher);
            for arg in args {
                hash_expr_recursive(ctx, *arg, hasher);
            }
        }
        Expr::Matrix { rows, cols, data } => {
            10u8.hash(hasher);
            rows.hash(hasher);
            cols.hash(hasher);
            for elem in data {
                hash_expr_recursive(ctx, *elem, hasher);
            }
        }
    }
}

/// Stable hash used for truncated eval-json responses.
pub(crate) fn expr_hash_eval_json(ctx: &Context, expr: ExprId) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    let mut hasher = DefaultHasher::new();
    hash_expr_recursive(ctx, expr, &mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::{expr_hash_eval_json, expr_stats_eval_json, format_expr_limited_eval_json};

    #[test]
    fn format_expr_limited_eval_json_truncates_when_needed() {
        let mut ctx = cas_ast::Context::new();
        let expr = cas_parser::parse("x + x + x + x + x", &mut ctx).expect("parse");
        let (rendered, truncated, original_len) = format_expr_limited_eval_json(&ctx, expr, 5);
        assert!(truncated);
        assert!(rendered.contains("<truncated>"));
        assert!(original_len > 5);
    }

    #[test]
    fn expr_hash_eval_json_is_stable_for_same_expr() {
        let mut ctx = cas_ast::Context::new();
        let expr = cas_parser::parse("x^2 + 1", &mut ctx).expect("parse");
        let h1 = expr_hash_eval_json(&ctx, expr);
        let h2 = expr_hash_eval_json(&ctx, expr);
        assert_eq!(h1, h2);

        let stats = expr_stats_eval_json(&ctx, expr);
        assert!(stats.node_count >= 1);
    }
}
