use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::collect_by_var_support::try_rewrite_collect_by_var_expr;
use cas_math::collect_rule_support::try_rewrite_collect_like_terms_expr;

define_rule!(CollectRule, "Collect Terms", |ctx, expr| {
    let rewrite = try_rewrite_collect_by_var_expr(ctx, expr)?;
    Some(
        Rewrite::new(rewrite.rewritten)
            .desc_lazy(|| format!("collect({}, {})", rewrite.target_expr, rewrite.var_name)),
    )
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    #[test]
    fn test_collect_basic() {
        let mut ctx = Context::new();
        let rule = CollectRule;
        // collect(a*x + b*x, x) -> (a+b)*x
        let expr = parse("collect(a*x + b*x, x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        // Result could be (a+b)*x or (b+a)*x
        let s = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        assert!(s.contains("x"));
        assert!(s.contains("a + b") || s.contains("b + a"));
    }

    #[test]
    fn test_collect_with_constants() {
        let mut ctx = Context::new();
        let rule = CollectRule;
        // collect(a*x + 2*x + 5, x) -> (a+2)*x + 5
        let expr = parse("collect(a*x + 2*x + 5, x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        let s = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        assert!(s.contains("a + 2") || s.contains("2 + a"));
        assert!(s.contains("5"));
    }

    #[test]
    fn test_collect_powers() {
        let mut ctx = Context::new();
        let rule = CollectRule;
        // collect(3*x^2 + y*x^2 + x, x) -> (3+y)*x^2 + x
        let expr = parse("collect(3*x^2 + y*x^2 + x, x)", &mut ctx).unwrap();
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        let s = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        assert!(s.contains("3 + y") || s.contains("y + 3"));
        assert!(s.contains("x^2"));
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(CollectRule));
    // simplifier.add_rule(Box::new(CollectLikeTermsRule));
}

define_rule!(CollectLikeTermsRule, "Collect Like Terms", |ctx, expr| {
    let new_expr = try_rewrite_collect_like_terms_expr(ctx, expr)?;
    Some(Rewrite::new(new_expr).desc("Collect like terms"))
});
