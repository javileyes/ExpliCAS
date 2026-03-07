mod composite;
mod leaf;

use cas_ast::{Context, Expr, ExprId};

fn hash_expr_recursive<H: std::hash::Hasher>(ctx: &Context, expr: ExprId, hasher: &mut H) {
    match ctx.get(expr) {
        Expr::Number(n) => leaf::hash_number(n, hasher),
        Expr::Variable(name) => leaf::hash_variable(name, hasher),
        Expr::Constant(c) => leaf::hash_constant(c, hasher),
        Expr::SessionRef(id) => leaf::hash_session_ref(id, hasher),
        Expr::Hold(inner) => composite::hash_hold(ctx, *inner, hasher, hash_expr_recursive::<H>),
        Expr::Add(l, r) => composite::hash_add(ctx, *l, *r, hasher, hash_expr_recursive::<H>),
        Expr::Sub(l, r) => composite::hash_sub(ctx, *l, *r, hasher, hash_expr_recursive::<H>),
        Expr::Mul(l, r) => composite::hash_mul(ctx, *l, *r, hasher, hash_expr_recursive::<H>),
        Expr::Div(l, r) => composite::hash_div(ctx, *l, *r, hasher, hash_expr_recursive::<H>),
        Expr::Pow(l, r) => composite::hash_pow(ctx, *l, *r, hasher, hash_expr_recursive::<H>),
        Expr::Neg(inner) => composite::hash_neg(ctx, *inner, hasher, hash_expr_recursive::<H>),
        Expr::Function(name, args) => {
            composite::hash_function(ctx, name, args, hasher, hash_expr_recursive::<H>)
        }
        Expr::Matrix { rows, cols, data } => {
            composite::hash_matrix(ctx, rows, cols, data, hasher, hash_expr_recursive::<H>)
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
