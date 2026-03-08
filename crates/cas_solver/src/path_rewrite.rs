mod binary;
mod function;
mod unary;

use cas_ast::{Context, Expr, ExprId};

/// Reconstruct an expression by replacing the sub-expression at `path`.
///
/// `path` is expressed in engine `PathStep` units.
pub fn reconstruct_global_expr(
    context: &mut Context,
    root: ExprId,
    path: &[crate::PathStep],
    replacement: ExprId,
) -> ExprId {
    if path.is_empty() {
        return replacement;
    }

    let current_step = &path[0];
    let remaining_path = &path[1..];
    let expr = context.get(root).clone();

    match (expr, current_step) {
        (Expr::Add(l, r), step) => {
            binary::reconstruct_add(context, root, l, r, step, remaining_path, replacement)
        }
        (Expr::Sub(l, r), step) => {
            binary::reconstruct_sub(context, root, l, r, step, remaining_path, replacement)
        }
        (Expr::Mul(l, r), step) => {
            binary::reconstruct_mul(context, root, l, r, step, remaining_path, replacement)
        }
        (Expr::Div(l, r), step) => {
            binary::reconstruct_div(context, root, l, r, step, remaining_path, replacement)
        }
        (Expr::Pow(b, e), step) => {
            binary::reconstruct_pow(context, root, b, e, step, remaining_path, replacement)
        }
        (Expr::Neg(e), step) => {
            unary::reconstruct_neg(context, root, e, step, remaining_path, replacement)
        }
        (Expr::Function(name, args), step) => function::reconstruct_function(
            context,
            root,
            name,
            args,
            step,
            remaining_path,
            replacement,
        ),
        (Expr::Hold(inner), step) => {
            unary::reconstruct_hold(context, root, inner, step, remaining_path, replacement)
        }
        _ => root,
    }
}
