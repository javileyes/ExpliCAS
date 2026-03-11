//! AST rewrite helpers addressed by numeric child paths.
//!
//! This module centralizes "replace subtree at path" logic so runtime crates
//! can reuse one implementation with either canonicalizing or raw node builders.

use cas_ast::{Context, Expr, ExprId};

/// Rewrite one subtree at `path` and rebuild ancestors using `add_expr`.
///
/// Path semantics:
/// - binary nodes: `0 = left/base/numerator`, `1 = right/exponent/denominator`
/// - unary nodes: `0 = inner`
/// - function nodes: `i = argument index`
///
/// On invalid path segments, this returns the original `root` unchanged.
pub fn rewrite_at_expr_path_with<FAddExpr>(
    ctx: &mut Context,
    root: ExprId,
    path: &[u8],
    replacement: ExprId,
    add_expr: &mut FAddExpr,
) -> ExprId
where
    FAddExpr: FnMut(&mut Context, Expr) -> ExprId,
{
    if path.is_empty() {
        return replacement;
    }

    let child = path[0] as usize;
    let rest = &path[1..];
    let expr = ctx.get(root).clone();

    match expr {
        Expr::Add(l, r) => match child {
            0 => {
                let new_l = rewrite_at_expr_path_with(ctx, l, rest, replacement, add_expr);
                add_expr(ctx, Expr::Add(new_l, r))
            }
            1 => {
                let new_r = rewrite_at_expr_path_with(ctx, r, rest, replacement, add_expr);
                add_expr(ctx, Expr::Add(l, new_r))
            }
            _ => root,
        },
        Expr::Sub(l, r) => match child {
            0 => {
                let new_l = rewrite_at_expr_path_with(ctx, l, rest, replacement, add_expr);
                add_expr(ctx, Expr::Sub(new_l, r))
            }
            1 => {
                let new_r = rewrite_at_expr_path_with(ctx, r, rest, replacement, add_expr);
                add_expr(ctx, Expr::Sub(l, new_r))
            }
            _ => root,
        },
        Expr::Mul(l, r) => match child {
            0 => {
                let new_l = rewrite_at_expr_path_with(ctx, l, rest, replacement, add_expr);
                add_expr(ctx, Expr::Mul(new_l, r))
            }
            1 => {
                let new_r = rewrite_at_expr_path_with(ctx, r, rest, replacement, add_expr);
                add_expr(ctx, Expr::Mul(l, new_r))
            }
            _ => root,
        },
        Expr::Div(l, r) => match child {
            0 => {
                let new_l = rewrite_at_expr_path_with(ctx, l, rest, replacement, add_expr);
                add_expr(ctx, Expr::Div(new_l, r))
            }
            1 => {
                let new_r = rewrite_at_expr_path_with(ctx, r, rest, replacement, add_expr);
                add_expr(ctx, Expr::Div(l, new_r))
            }
            _ => root,
        },
        Expr::Pow(b, e) => match child {
            0 => {
                let new_b = rewrite_at_expr_path_with(ctx, b, rest, replacement, add_expr);
                add_expr(ctx, Expr::Pow(new_b, e))
            }
            1 => {
                let new_e = rewrite_at_expr_path_with(ctx, e, rest, replacement, add_expr);
                add_expr(ctx, Expr::Pow(b, new_e))
            }
            _ => root,
        },
        Expr::Neg(inner) => {
            if child == 0 {
                let new_inner = rewrite_at_expr_path_with(ctx, inner, rest, replacement, add_expr);
                add_expr(ctx, Expr::Neg(new_inner))
            } else {
                root
            }
        }
        Expr::Hold(inner) => {
            if child == 0 {
                let new_inner = rewrite_at_expr_path_with(ctx, inner, rest, replacement, add_expr);
                add_expr(ctx, Expr::Hold(new_inner))
            } else {
                root
            }
        }
        Expr::Function(name, mut args) => {
            if child < args.len() {
                args[child] =
                    rewrite_at_expr_path_with(ctx, args[child], rest, replacement, add_expr);
                add_expr(ctx, Expr::Function(name, args))
            } else {
                root
            }
        }
        Expr::Number(_)
        | Expr::Constant(_)
        | Expr::Variable(_)
        | Expr::SessionRef(_)
        | Expr::Matrix { .. } => root,
    }
}

/// Rewrite with `Context::add` rebuilding behavior.
pub fn rewrite_at_expr_path(
    ctx: &mut Context,
    root: ExprId,
    path: &[u8],
    replacement: ExprId,
) -> ExprId {
    rewrite_at_expr_path_with(ctx, root, path, replacement, &mut |ctx, expr| ctx.add(expr))
}

/// Rewrite with `Context::add_raw` rebuilding behavior.
pub fn rewrite_at_expr_path_raw(
    ctx: &mut Context,
    root: ExprId,
    path: &[u8],
    replacement: ExprId,
) -> ExprId {
    rewrite_at_expr_path_with(ctx, root, path, replacement, &mut |ctx, expr| {
        ctx.add_raw(expr)
    })
}

#[cfg(test)]
mod tests {
    use super::{rewrite_at_expr_path, rewrite_at_expr_path_raw};
    use cas_ast::{Context, Expr};
    use cas_parser::parse;

    #[test]
    fn rewrite_at_expr_path_updates_nested_subtree() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let sum = ctx.add_raw(Expr::Add(a, b));
        let root = ctx.add_raw(Expr::Mul(sum, c));
        let replacement = ctx.var("x");
        let path = vec![0, 1]; // Mul.left then Add.right
        let rewritten = rewrite_at_expr_path(&mut ctx, root, &path, replacement);

        assert!(matches!(ctx.get(rewritten), Expr::Mul(_, _)));
        assert!(contains_var_name(&ctx, rewritten, "x"));
        assert!(!contains_var_name(&ctx, rewritten, "b"));
    }

    #[test]
    fn rewrite_at_expr_path_raw_preserves_raw_mul_shape() {
        let mut ctx = Context::new();
        let a = parse("a", &mut ctx).expect("parse a");
        let b = parse("b", &mut ctx).expect("parse b");
        let root = ctx.add_raw(Expr::Mul(a, b));
        let replacement = parse("x", &mut ctx).expect("parse replacement");
        let rewritten = rewrite_at_expr_path_raw(&mut ctx, root, &[0], replacement);
        assert!(matches!(ctx.get(rewritten), Expr::Mul(_, _)));
    }

    #[test]
    fn rewrite_at_expr_path_returns_root_on_invalid_path() {
        let mut ctx = Context::new();
        let root = parse("a+b", &mut ctx).expect("parse root");
        let replacement = parse("x", &mut ctx).expect("parse replacement");
        let rewritten = rewrite_at_expr_path(&mut ctx, root, &[2], replacement);
        assert_eq!(rewritten, root);
    }

    fn contains_var_name(ctx: &Context, expr: cas_ast::ExprId, var: &str) -> bool {
        match ctx.get(expr) {
            Expr::Variable(sym) => ctx.sym_name(*sym) == var,
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => contains_var_name(ctx, *l, var) || contains_var_name(ctx, *r, var),
            Expr::Neg(inner) | Expr::Hold(inner) => contains_var_name(ctx, *inner, var),
            Expr::Function(_, args) => args.iter().any(|arg| contains_var_name(ctx, *arg, var)),
            Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) | Expr::Matrix { .. } => {
                false
            }
        }
    }
}
