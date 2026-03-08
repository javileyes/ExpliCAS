#[cfg(test)]
mod tests {
    use crate::path_rewrite::reconstruct_global_expr;
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

    #[test]
    fn reconstruct_preserves_neg_wrapper_on_left_add_branch() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let z = ctx.var("z");
        let neg_x = ctx.add_raw(Expr::Neg(x));
        let add = ctx.add_raw(Expr::Add(neg_x, y));
        let out = reconstruct_global_expr(&mut ctx, add, &[crate::PathStep::Left], z);
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
        let out = reconstruct_global_expr(&mut ctx, hold_x, &[crate::PathStep::Inner], y);
        assert!(matches!(ctx.get(out), Expr::Hold(inner) if *inner == y));
    }
}
