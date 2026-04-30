use super::strong_target_match;
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_nary::{add_terms_signed, build_balanced_add, Sign};
use num_traits::Signed;
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FactorRewrite {
    pub(crate) rewritten: ExprId,
    pub(crate) focus_before: Option<ExprId>,
    pub(crate) focus_after: Option<ExprId>,
}

pub(crate) fn try_rewrite_factored_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<FactorRewrite> {
    if let Some(rewritten) = try_rewrite_factor_focus_target_aware(ctx, source_expr, target_expr) {
        return Some(FactorRewrite {
            rewritten,
            focus_before: None,
            focus_after: None,
        });
    }

    try_rewrite_factor_additive_target_aware(ctx, source_expr, target_expr)
}

fn try_rewrite_factor_additive_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<FactorRewrite> {
    let source_terms = signed_additive_terms(ctx, source_expr);
    let target_terms = signed_additive_terms(ctx, target_expr);
    if source_terms.len() < 2 || source_terms.len() > 8 || target_terms.len() < 2 {
        return None;
    }

    for (target_index, target_focus) in target_terms.iter().copied().enumerate() {
        let target_passthrough =
            collect_passthrough_terms_excluding_index(&target_terms, target_index);

        for source_mask in 1usize..(1usize << source_terms.len()) {
            let source_passthrough =
                collect_passthrough_terms_excluding_mask(&source_terms, source_mask);
            if !additive_term_multiset_matches(ctx, &source_passthrough, &target_passthrough) {
                continue;
            }

            let source_focus_terms = collect_terms_by_mask(&source_terms, source_mask);
            let source_focus = build_additive_focus(ctx, &source_focus_terms)?;
            if try_rewrite_factor_focus_target_aware(ctx, source_focus, target_focus).is_some() {
                return Some(FactorRewrite {
                    rewritten: target_expr,
                    focus_before: Some(source_focus),
                    focus_after: Some(target_focus),
                });
            }
        }
    }

    None
}

fn try_rewrite_factor_focus_target_aware(
    ctx: &mut Context,
    source_focus: ExprId,
    target_focus: ExprId,
) -> Option<ExprId> {
    if let Some(rewritten) = try_rewrite_factor_direct_target_aware(ctx, source_focus, target_focus)
    {
        return Some(rewritten);
    }

    let (source_inner, target_inner) = match (ctx.get(source_focus), ctx.get(target_focus)) {
        (Expr::Neg(source_inner), Expr::Neg(target_inner)) => (*source_inner, *target_inner),
        _ => return None,
    };
    try_rewrite_factor_direct_target_aware(ctx, source_inner, target_inner)?;
    Some(target_focus)
}

fn try_rewrite_factor_direct_target_aware(
    ctx: &mut Context,
    source_focus: ExprId,
    target_focus: ExprId,
) -> Option<ExprId> {
    if !looks_like_factor_focus_target(ctx, target_focus) {
        return None;
    }

    let factored = cas_math::factor::factor(ctx, source_focus);
    (factored != source_focus && factor_target_match(ctx, factored, target_focus))
        .then_some(target_focus)
}

fn factor_target_match(ctx: &mut Context, factored: ExprId, target_focus: ExprId) -> bool {
    strong_target_match(ctx, factored, target_focus)
        || simplified_difference_matches_zero(ctx, factored, target_focus)
}

fn simplified_difference_matches_zero(ctx: &mut Context, left: ExprId, right: ExprId) -> bool {
    let zero = ctx.num(0);
    let difference = ctx.add(Expr::Sub(left, right));
    let simplified = run_default_simplify(ctx, difference);
    strong_target_match(ctx, simplified, zero)
}

fn run_default_simplify(ctx: &mut Context, expr: ExprId) -> ExprId {
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (rewritten, _steps, _stats) = simplifier.simplify_with_stats(
        expr,
        crate::SimplifyOptions {
            collect_steps: false,
            suppress_depth_overflow_warnings: true,
            ..crate::SimplifyOptions::default()
        },
    );
    std::mem::swap(&mut simplifier.context, ctx);
    rewritten
}

fn looks_like_factor_focus_target(ctx: &mut Context, expr: ExprId) -> bool {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            return looks_like_factor_focus_target(ctx, inner);
        }
        Expr::Pow(base, exp) => {
            return is_additive_factor_shape(ctx, base) && is_positive_integer_exponent(ctx, exp);
        }
        _ => {}
    }

    if !ctx.is_mul_commutative(expr) {
        return false;
    }

    let factors = cas_math::trig_roots_flatten::flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return false;
    }

    let mut non_numeric_factors = 0usize;
    let mut has_additive_factor = false;
    for factor in factors {
        if matches!(ctx.get(factor), Expr::Number(_) | Expr::Constant(_)) {
            continue;
        }

        non_numeric_factors += 1;
        if is_additive_factor_shape(ctx, factor) {
            has_additive_factor = true;
        }
    }

    has_additive_factor && non_numeric_factors >= 2
}

fn is_additive_factor_shape(ctx: &mut Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
}

fn is_positive_integer_exponent(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(value) => value.is_integer() && value.to_integer() > 1.into(),
        _ => false,
    }
}

fn signed_additive_terms(ctx: &mut Context, expr: ExprId) -> Vec<ExprId> {
    add_terms_signed(ctx, expr)
        .into_iter()
        .map(|(term, sign)| apply_sign_to_term(ctx, term, sign))
        .collect()
}

fn apply_sign_to_term(ctx: &mut Context, term: ExprId, sign: Sign) -> ExprId {
    match sign {
        Sign::Pos => term,
        Sign::Neg => ctx.add(Expr::Neg(term)),
    }
}

fn build_additive_focus(ctx: &mut Context, terms: &[ExprId]) -> Option<ExprId> {
    match terms {
        [] => None,
        [term] => Some(*term),
        [left, right] => {
            if let Some(rewritten) = build_binary_subtraction_focus(ctx, *left, *right) {
                return Some(rewritten);
            }
            Some(build_balanced_add(ctx, terms))
        }
        _ => Some(build_balanced_add(ctx, terms)),
    }
}

fn build_binary_subtraction_focus(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    if let Expr::Neg(right_inner) = ctx.get(right).clone() {
        return Some(ctx.add(Expr::Sub(left, right_inner)));
    }
    if let Expr::Number(value) = ctx.get(right).clone() {
        if value.is_negative() {
            let positive = ctx.add(Expr::Number(-value));
            return Some(ctx.add(Expr::Sub(left, positive)));
        }
    }

    if let Expr::Neg(left_inner) = ctx.get(left).clone() {
        return Some(ctx.add(Expr::Sub(right, left_inner)));
    }
    if let Expr::Number(value) = ctx.get(left).clone() {
        if value.is_negative() {
            let positive = ctx.add(Expr::Number(-value));
            return Some(ctx.add(Expr::Sub(right, positive)));
        }
    }

    None
}

fn collect_terms_by_mask(terms: &[ExprId], mask: usize) -> Vec<ExprId> {
    terms
        .iter()
        .enumerate()
        .filter_map(|(index, term)| ((mask & (1usize << index)) != 0).then_some(*term))
        .collect()
}

fn collect_passthrough_terms_excluding_index(
    terms: &[ExprId],
    excluded_index: usize,
) -> Vec<ExprId> {
    terms
        .iter()
        .enumerate()
        .filter_map(|(index, term)| (index != excluded_index).then_some(*term))
        .collect()
}

fn collect_passthrough_terms_excluding_mask(terms: &[ExprId], mask: usize) -> Vec<ExprId> {
    terms
        .iter()
        .enumerate()
        .filter_map(|(index, term)| ((mask & (1usize << index)) == 0).then_some(*term))
        .collect()
}

fn additive_term_multiset_matches(
    ctx: &mut Context,
    lhs_terms: &[ExprId],
    rhs_terms: &[ExprId],
) -> bool {
    if lhs_terms.len() != rhs_terms.len() {
        return false;
    }

    let mut lhs = lhs_terms.to_vec();
    let mut rhs = rhs_terms.to_vec();
    lhs.sort_by(|left, right| compare_expr(ctx, *left, *right));
    rhs.sort_by(|left, right| compare_expr(ctx, *left, *right));

    lhs.iter()
        .zip(rhs.iter())
        .all(|(left, right)| compare_expr(ctx, *left, *right) == Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_factored_target_aware;
    use cas_math::semantic_equality::SemanticEqualityChecker;

    #[test]
    fn factors_direct_targets_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("x^2-1", &mut ctx).expect("source");
        let target = cas_parser::parse("(x-1)*(x+1)", &mut ctx).expect("target");
        let rewrite = try_rewrite_factored_target_aware(&mut ctx, source, target).expect("rewrite");
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn factors_multi_term_focus_with_additive_passthrough() {
        let cases = [("a+x^2-1", "a+(x-1)*(x+1)"), ("a-(x^2-1)", "a-(x-1)*(x+1)")];

        for (source, target) in cases {
            let mut ctx = cas_ast::Context::new();
            let source = cas_parser::parse(source, &mut ctx).expect("source");
            let target = cas_parser::parse(target, &mut ctx).expect("target");
            let rewrite =
                try_rewrite_factored_target_aware(&mut ctx, source, target).expect("rewrite");
            let checker = SemanticEqualityChecker::new(&ctx);
            assert!(
                checker.are_equal(rewrite.rewritten, target),
                "expected additive factor rewrite to equal target"
            );
        }
    }
}
