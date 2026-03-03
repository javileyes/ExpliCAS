use cas_ast::{Context, Expr, ExprId};

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
