use cas_api_models::ExprStatsWire;
use cas_ast::{Context, Expr, ExprId};

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

/// Compute node/depth/term stats for output payloads.
pub(crate) fn expr_output_stats(ctx: &Context, expr: ExprId) -> ExprStatsWire {
    let (node_count, depth) = cas_ast::traversal::count_nodes_and_max_depth(ctx, expr);
    let term_count =
        crate::try_get_poly_result_term_count(ctx, expr).or_else(|| count_add_terms(ctx, expr));

    ExprStatsWire {
        node_count,
        depth,
        term_count,
    }
}
