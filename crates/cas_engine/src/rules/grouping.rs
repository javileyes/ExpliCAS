use crate::collect_by_var_support::try_rewrite_collect_by_var_expr;
use crate::collect_rule_support::try_rewrite_collect_like_terms_identity_expr;
use crate::define_rule;
use crate::rule::Rewrite;

define_rule!(CollectRule, "Collect Terms", |ctx, expr| {
    let rewrite = try_rewrite_collect_by_var_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc_lazy(|| rewrite.desc()))
});

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(CollectRule));
    // simplifier.add_rule(Box::new(CollectLikeTermsRule));
}

define_rule!(CollectLikeTermsRule, "Collect Like Terms", |ctx, expr| {
    let rewrite = try_rewrite_collect_like_terms_identity_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
});
