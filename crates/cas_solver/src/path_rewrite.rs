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
        (Expr::Add(l, r), crate::PathStep::Left) => {
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(Expr::Add(new_l, r))
        }
        // Special case: Sub(a,b) may be canonicalized as Add(a, Neg(b)).
        (Expr::Add(l, r), crate::PathStep::Right) => {
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
        (Expr::Sub(l, r), crate::PathStep::Left) => {
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(Expr::Sub(new_l, r))
        }
        (Expr::Sub(l, r), crate::PathStep::Right) => {
            let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
            context.add(Expr::Sub(l, new_r))
        }
        (Expr::Mul(l, r), crate::PathStep::Left) => {
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(Expr::Mul(new_l, r))
        }
        (Expr::Mul(l, r), crate::PathStep::Right) => {
            let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
            context.add(Expr::Mul(l, new_r))
        }
        (Expr::Div(l, r), crate::PathStep::Left) => {
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(Expr::Div(new_l, r))
        }
        (Expr::Div(l, r), crate::PathStep::Right) => {
            let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
            context.add(Expr::Div(l, new_r))
        }
        (Expr::Pow(b, e), crate::PathStep::Base) => {
            let new_b = reconstruct_global_expr(context, b, remaining_path, replacement);
            context.add(Expr::Pow(new_b, e))
        }
        (Expr::Pow(b, e), crate::PathStep::Exponent) => {
            let new_e = reconstruct_global_expr(context, e, remaining_path, replacement);
            context.add(Expr::Pow(b, new_e))
        }
        (Expr::Neg(e), crate::PathStep::Inner) => {
            let new_e = reconstruct_global_expr(context, e, remaining_path, replacement);
            context.add(Expr::Neg(new_e))
        }
        (Expr::Function(name, args), crate::PathStep::Arg(idx)) => {
            let mut new_args = args.clone();
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
    use cas_ast::{Context, Expr};

    #[test]
    fn reconstruct_replaces_right_add_branch() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let add = ctx.add(Expr::Add(x, y));
        let out = reconstruct_global_expr(&mut ctx, add, &[crate::PathStep::Right], z);
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
}
