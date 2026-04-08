use super::strong_target_match;
use cas_ast::{Expr, ExprId};
use cas_math::expr_nary::{add_terms_signed, Sign};
use cas_math::number_theory_support::{
    try_rewrite_consecutive_factorial_ratio_expr, ConsecutiveFactorialRatioRewrite,
};

fn apply_sign_to_term(ctx: &mut cas_ast::Context, term: ExprId, sign: Sign) -> ExprId {
    match sign {
        Sign::Pos => term,
        Sign::Neg => ctx.add(Expr::Neg(term)),
    }
}

fn rebuild_additive_terms_with_rewritten_term(
    ctx: &mut cas_ast::Context,
    terms: &[(ExprId, Sign)],
    rewritten_index: usize,
    rewritten_term: ExprId,
) -> ExprId {
    let mut rebuilt_terms = Vec::with_capacity(terms.len());
    for (index, (term, sign)) in terms.iter().enumerate() {
        let current = if index == rewritten_index {
            rewritten_term
        } else {
            apply_sign_to_term(ctx, *term, *sign)
        };
        rebuilt_terms.push(current);
    }

    let mut iter = rebuilt_terms.into_iter();
    let Some(first) = iter.next() else {
        return ctx.num(0);
    };

    iter.fold(first, |acc, term| ctx.add(Expr::Add(acc, term)))
}

pub(crate) fn try_rewrite_consecutive_factorial_ratio_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<ConsecutiveFactorialRatioRewrite> {
    if let Some(rewrite) = try_rewrite_consecutive_factorial_ratio_expr(ctx, expr) {
        if strong_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(rewrite);
        }
    }

    let source_terms = add_terms_signed(ctx, expr);
    if source_terms.len() <= 1 {
        return None;
    }

    for (source_index, (source_term, source_sign)) in source_terms.iter().copied().enumerate() {
        let Some(rewrite) = try_rewrite_consecutive_factorial_ratio_expr(ctx, source_term) else {
            continue;
        };
        let signed_rewritten_term = apply_sign_to_term(ctx, rewrite.rewritten, source_sign);
        let rewritten = rebuild_additive_terms_with_rewritten_term(
            ctx,
            &source_terms,
            source_index,
            signed_rewritten_term,
        );
        if strong_target_match(ctx, rewritten, target_expr) {
            return Some(ConsecutiveFactorialRatioRewrite {
                rewritten: target_expr,
                factorial_arg_requires_nonnegative: rewrite.factorial_arg_requires_nonnegative,
            });
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_consecutive_factorial_ratio_target_aware;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn rewrites_consecutive_factorial_ratio_target_aware() {
        let mut ctx = Context::new();
        let source = parse("(n+1)!/n!", &mut ctx).expect("source");
        let target = parse("n+1", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_consecutive_factorial_ratio_target_aware(&mut ctx, source, target)
                .expect("rewrite");

        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn rewrites_consecutive_factorial_ratio_with_passthrough_target_aware() {
        let mut ctx = Context::new();
        let source = parse("(n+1)!/n!+a", &mut ctx).expect("source");
        let target = parse("n+1+a", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_consecutive_factorial_ratio_target_aware(&mut ctx, source, target)
                .expect("rewrite");

        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn rewrites_gap_two_factorial_ratio_target_aware() {
        let mut ctx = Context::new();
        let source = parse("(n+1)!/(n-1)!", &mut ctx).expect("source");
        let target = parse("n*(n+1)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_consecutive_factorial_ratio_target_aware(&mut ctx, source, target)
                .expect("rewrite");

        assert_eq!(rewrite.rewritten, target);
    }
}
