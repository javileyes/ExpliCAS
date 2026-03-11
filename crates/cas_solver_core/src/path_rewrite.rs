use crate::step_types::PathStep;
use cas_ast::{symbol::SymbolId, Context, Expr, ExprId};

/// Reconstruct an expression by replacing the sub-expression at `path`.
///
/// `path` is expressed in core `PathStep` units.
pub fn reconstruct_global_expr(
    context: &mut Context,
    root: ExprId,
    path: &[PathStep],
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
            reconstruct_add(context, root, l, r, step, remaining_path, replacement)
        }
        (Expr::Sub(l, r), step) => reconstruct_left_right_binary(
            context,
            root,
            l,
            r,
            step,
            remaining_path,
            replacement,
            Expr::Sub,
        ),
        (Expr::Mul(l, r), step) => reconstruct_left_right_binary(
            context,
            root,
            l,
            r,
            step,
            remaining_path,
            replacement,
            Expr::Mul,
        ),
        (Expr::Div(l, r), step) => reconstruct_left_right_binary(
            context,
            root,
            l,
            r,
            step,
            remaining_path,
            replacement,
            Expr::Div,
        ),
        (Expr::Pow(base, exp), step) => {
            reconstruct_pow(context, root, base, exp, step, remaining_path, replacement)
        }
        (Expr::Neg(inner), step) => reconstruct_unary(
            context,
            root,
            inner,
            step,
            remaining_path,
            replacement,
            Expr::Neg,
        ),
        (Expr::Hold(inner), step) => reconstruct_unary(
            context,
            root,
            inner,
            step,
            remaining_path,
            replacement,
            Expr::Hold,
        ),
        (Expr::Function(name, args), step) => {
            reconstruct_function(context, root, name, args, step, remaining_path, replacement)
        }
        _ => root,
    }
}

fn reconstruct_add(
    context: &mut Context,
    root: ExprId,
    l: ExprId,
    r: ExprId,
    step: &PathStep,
    remaining_path: &[PathStep],
    replacement: ExprId,
) -> ExprId {
    match step {
        PathStep::Left => {
            if let Expr::Neg(inner) = context.get(l).clone() {
                let new_inner =
                    reconstruct_global_expr(context, inner, remaining_path, replacement);
                let new_neg = context.add(Expr::Neg(new_inner));
                context.add(Expr::Add(new_neg, r))
            } else {
                let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
                context.add(Expr::Add(new_l, r))
            }
        }
        PathStep::Right => {
            if let Expr::Neg(inner) = context.get(r).clone() {
                let new_inner =
                    reconstruct_global_expr(context, inner, remaining_path, replacement);
                let new_neg = context.add(Expr::Neg(new_inner));
                context.add(Expr::Add(l, new_neg))
            } else {
                let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
                context.add(Expr::Add(l, new_r))
            }
        }
        _ => root,
    }
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_left_right_binary(
    context: &mut Context,
    root: ExprId,
    l: ExprId,
    r: ExprId,
    step: &PathStep,
    remaining_path: &[PathStep],
    replacement: ExprId,
    build: fn(ExprId, ExprId) -> Expr,
) -> ExprId {
    match step {
        PathStep::Left => {
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(build(new_l, r))
        }
        PathStep::Right => {
            let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
            context.add(build(l, new_r))
        }
        _ => root,
    }
}

fn reconstruct_pow(
    context: &mut Context,
    root: ExprId,
    base: ExprId,
    exp: ExprId,
    step: &PathStep,
    remaining_path: &[PathStep],
    replacement: ExprId,
) -> ExprId {
    match step {
        PathStep::Base => {
            let new_base = reconstruct_global_expr(context, base, remaining_path, replacement);
            context.add(Expr::Pow(new_base, exp))
        }
        PathStep::Exponent => {
            let new_exp = reconstruct_global_expr(context, exp, remaining_path, replacement);
            context.add(Expr::Pow(base, new_exp))
        }
        _ => root,
    }
}

fn reconstruct_unary(
    context: &mut Context,
    root: ExprId,
    inner: ExprId,
    step: &PathStep,
    remaining_path: &[PathStep],
    replacement: ExprId,
    build: fn(ExprId) -> Expr,
) -> ExprId {
    match step {
        PathStep::Inner => {
            let new_inner = reconstruct_global_expr(context, inner, remaining_path, replacement);
            context.add(build(new_inner))
        }
        _ => root,
    }
}

fn reconstruct_function(
    context: &mut Context,
    root: ExprId,
    name: SymbolId,
    args: Vec<ExprId>,
    step: &PathStep,
    remaining_path: &[PathStep],
    replacement: ExprId,
) -> ExprId {
    match step {
        PathStep::Arg(idx) => {
            let mut new_args = args;
            if *idx < new_args.len() {
                new_args[*idx] =
                    reconstruct_global_expr(context, new_args[*idx], remaining_path, replacement);
                context.add(Expr::Function(name, new_args))
            } else {
                root
            }
        }
        _ => root,
    }
}

#[cfg(test)]
mod tests {
    use super::reconstruct_global_expr;
    use crate::step_types::PathStep;
    use cas_ast::{Context, Expr};

    #[test]
    fn reconstruct_replaces_right_add_branch() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let add = ctx.add(Expr::Add(x, y));
        let out = reconstruct_global_expr(&mut ctx, add, &[PathStep::Right], z);
        assert_eq!(
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &ctx,
                    id: out
                }
            ),
            "x + z"
        );
    }

    #[test]
    fn reconstruct_preserves_neg_wrapper_on_left_add_branch() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let neg_x = ctx.add_raw(Expr::Neg(x));
        let add = ctx.add_raw(Expr::Add(neg_x, y));
        let out = reconstruct_global_expr(&mut ctx, add, &[PathStep::Left], z);
        let rendered = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &ctx,
                id: out
            }
        );
        assert_eq!(rendered, "y - z");
    }

    #[test]
    fn reconstruct_descends_into_hold_inner() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let hold_x = ctx.add(Expr::Hold(x));
        let out = reconstruct_global_expr(&mut ctx, hold_x, &[PathStep::Inner], y);
        assert!(matches!(ctx.get(out), Expr::Hold(inner) if *inner == y));
    }
}
