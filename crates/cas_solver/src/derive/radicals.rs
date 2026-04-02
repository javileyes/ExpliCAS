use cas_ast::{Context, ExprId};

pub(crate) fn try_rewrite_odd_half_power_target_aware(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    if let Some(rewrite) = cas_math::root_forms::try_rewrite_odd_half_power_expr(ctx, expr) {
        return Some(rewrite.rewritten);
    }

    let normalized = cas_math::canonical_forms::normalize_core(ctx, expr);
    if normalized == expr {
        return None;
    }

    cas_math::root_forms::try_rewrite_odd_half_power_expr(ctx, normalized)
        .map(|rewrite| rewrite.rewritten)
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_odd_half_power_target_aware;
    use cas_ast::{Context, Expr};
    use cas_formatter::DisplayExpr;

    #[test]
    fn rewrites_direct_odd_half_power() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let exp = ctx.rational(3, 2);
        let expr = ctx.add(Expr::Pow(x, exp));
        let rewritten = try_rewrite_odd_half_power_target_aware(&mut ctx, expr).expect("rewrite");
        let text = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewritten
            }
        );
        assert!(text.contains("sqrt"));
        assert!(text.contains("|x|") || text.contains("abs"));
    }
}
