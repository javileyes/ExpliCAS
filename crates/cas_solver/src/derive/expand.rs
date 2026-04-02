use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpandRewriteKind {
    BinomialPower,
    General,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExpandRewrite {
    pub(crate) rewritten: ExprId,
    pub(crate) kind: ExpandRewriteKind,
}

pub(crate) fn try_rewrite_expanded_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<ExpandRewrite> {
    let rewritten = cas_math::expand_ops::expand(ctx, source_expr);
    if rewritten == source_expr || !matches_expanded_target(ctx, rewritten, target_expr) {
        return None;
    }

    Some(ExpandRewrite {
        rewritten,
        kind: classify_expand_rewrite_kind(ctx, source_expr),
    })
}

fn matches_expanded_target(ctx: &mut Context, actual: ExprId, target: ExprId) -> bool {
    if super::strong_target_match(ctx, actual, target) {
        return true;
    }

    let difference = ctx.add(Expr::Sub(actual, target));
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (simplified, _steps, _stats) =
        simplifier.simplify_with_stats(difference, crate::SimplifyOptions::default());
    std::mem::swap(&mut simplifier.context, ctx);

    let zero = ctx.num(0);
    if super::strong_target_match(ctx, simplified, zero) {
        return true;
    }

    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let equivalent = matches!(
        simplifier.are_equivalent_extended(actual, target),
        crate::EquivalenceResult::True | crate::EquivalenceResult::ConditionalTrue { .. }
    );
    std::mem::swap(&mut simplifier.context, ctx);
    equivalent
}

fn classify_expand_rewrite_kind(ctx: &Context, expr: ExprId) -> ExpandRewriteKind {
    match ctx.get(expr) {
        Expr::Pow(base, exp)
            if matches!(ctx.get(*base), Expr::Add(_, _) | Expr::Sub(_, _))
                && matches!(ctx.get(*exp), Expr::Number(_)) =>
        {
            ExpandRewriteKind::BinomialPower
        }
        _ => ExpandRewriteKind::General,
    }
}

#[cfg(test)]
mod tests {
    use super::{try_rewrite_expanded_target_aware, ExpandRewriteKind};

    #[test]
    fn expands_symbolic_binomial_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("(a + b)^2", &mut ctx).expect("source");
        let target = cas_parser::parse("a^2 + 2*a*b + b^2", &mut ctx).expect("target");
        let rewrite = try_rewrite_expanded_target_aware(&mut ctx, source, target).expect("rewrite");
        assert_eq!(rewrite.kind, ExpandRewriteKind::BinomialPower);
    }

    #[test]
    fn rejects_non_matching_expansion_target() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("(a + b)^2", &mut ctx).expect("source");
        let target = cas_parser::parse("a^2 + b^2", &mut ctx).expect("target");
        assert!(try_rewrite_expanded_target_aware(&mut ctx, source, target).is_none());
    }
}
