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
}
