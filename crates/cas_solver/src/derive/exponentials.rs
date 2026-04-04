use cas_ast::{BuiltinFn, Expr, ExprId};
use cas_math::expr_destructure::{as_div, as_mul, as_pow};
use cas_math::expr_nary::{add_terms_signed, Sign};
use cas_math::expr_relations::extract_negated_inner;
use num_traits::{One, Signed};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeriveExponentialRewriteKind {
    ExpandSumDiff,
    ContractSumDiff,
    ExpandReciprocal,
    ContractReciprocal,
    ExpandPower,
    ContractPower,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeriveExponentialRewrite {
    pub(crate) rewritten: ExprId,
    pub(crate) kind: DeriveExponentialRewriteKind,
}

impl DeriveExponentialRewriteKind {
    pub(crate) fn description(self) -> &'static str {
        match self {
            Self::ExpandSumDiff => {
                "Expand exp(u ± v ± ...) into products/quotients of exponentials"
            }
            Self::ContractSumDiff => {
                "Contract products/quotients of exponentials into exp(u ± v ± ...)"
            }
            Self::ExpandReciprocal => "Expand exp(-u) as 1 / exp(u)",
            Self::ContractReciprocal => "Recognize 1 / exp(u) as exp(-u)",
            Self::ExpandPower => "Expand exp(n·u) as exp(u)^n",
            Self::ContractPower => "Recognize exp(u)^n as exp(n·u)",
        }
    }

    pub(crate) fn rule_name(self) -> &'static str {
        match self {
            Self::ExpandSumDiff | Self::ContractSumDiff => "Exponential Sum/Difference Identity",
            Self::ExpandReciprocal | Self::ContractReciprocal => "Exponential Reciprocal Identity",
            Self::ExpandPower | Self::ContractPower => "Exponential Power Identity",
        }
    }
}

pub(crate) fn try_rewrite_exponential_sum_diff_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveExponentialRewrite> {
    if let Some(rewrite) =
        try_rewrite_exponential_sum_diff_signed_term_target_aware(ctx, expr, target_expr)
    {
        return Some(rewrite);
    }

    try_rewrite_exponential_sum_diff_additive_passthrough_target_aware(ctx, expr, target_expr)
}

fn try_rewrite_exponential_sum_diff_signed_term_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveExponentialRewrite> {
    if let Some(rewrite) = try_rewrite_exponential_sum_diff_core(ctx, expr, target_expr) {
        return Some(rewrite);
    }

    match (ctx.get(expr), ctx.get(target_expr)) {
        (Expr::Neg(inner_expr), Expr::Neg(inner_target)) => {
            try_rewrite_exponential_sum_diff_core(ctx, *inner_expr, *inner_target).map(|rewrite| {
                DeriveExponentialRewrite {
                    rewritten: target_expr,
                    kind: rewrite.kind,
                }
            })
        }
        _ => None,
    }
}

fn try_rewrite_exponential_sum_diff_additive_passthrough_target_aware(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveExponentialRewrite> {
    let source_terms = add_terms_signed(ctx, expr);
    let target_terms = add_terms_signed(ctx, target_expr);
    if source_terms.len() <= 1 || target_terms.len() <= 1 {
        return None;
    }

    for (source_index, (source_term, source_sign)) in source_terms.iter().enumerate() {
        let signed_source = apply_sign_to_term(ctx, *source_term, *source_sign);
        for (target_term, target_sign) in &target_terms {
            let signed_target = apply_sign_to_term(ctx, *target_term, *target_sign);
            let Some(rewrite) = try_rewrite_exponential_sum_diff_signed_term_target_aware(
                ctx,
                signed_source,
                signed_target,
            ) else {
                continue;
            };

            let rebuilt = rebuild_additive_terms_with_rewritten_term(
                ctx,
                &source_terms,
                source_index,
                rewrite.rewritten,
            );
            if super::strong_target_match(ctx, rebuilt, target_expr) {
                return Some(DeriveExponentialRewrite {
                    rewritten: target_expr,
                    kind: rewrite.kind,
                });
            }
        }
    }

    None
}

fn try_rewrite_exponential_sum_diff_core(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<DeriveExponentialRewrite> {
    if let Some(rewritten) = expand_exponential_sum_diff_expr(ctx, expr) {
        if matches_target_modulo_simplify(ctx, rewritten, target_expr) {
            return Some(DeriveExponentialRewrite {
                rewritten: target_expr,
                kind: DeriveExponentialRewriteKind::ExpandSumDiff,
            });
        }
    }

    if let Some(rewritten) = contract_exponential_sum_diff_expr(ctx, expr) {
        if matches_target_modulo_simplify(ctx, rewritten, target_expr) {
            return Some(DeriveExponentialRewrite {
                rewritten: target_expr,
                kind: DeriveExponentialRewriteKind::ContractSumDiff,
            });
        }
    }

    if let Some(rewritten) = expand_exponential_reciprocal_expr(ctx, expr) {
        if matches_target_modulo_simplify(ctx, rewritten, target_expr) {
            return Some(DeriveExponentialRewrite {
                rewritten: target_expr,
                kind: DeriveExponentialRewriteKind::ExpandReciprocal,
            });
        }
    }

    if let Some(rewritten) = contract_exponential_reciprocal_expr(ctx, expr) {
        if matches_target_modulo_simplify(ctx, rewritten, target_expr) {
            return Some(DeriveExponentialRewrite {
                rewritten: target_expr,
                kind: DeriveExponentialRewriteKind::ContractReciprocal,
            });
        }
    }

    if let Some(rewritten) = expand_exponential_power_expr(ctx, expr) {
        if matches_target_modulo_simplify(ctx, rewritten, target_expr) {
            return Some(DeriveExponentialRewrite {
                rewritten: target_expr,
                kind: DeriveExponentialRewriteKind::ExpandPower,
            });
        }
    }

    if let Some(rewritten) = contract_exponential_power_expr(ctx, expr) {
        if matches_target_modulo_simplify(ctx, rewritten, target_expr) {
            return Some(DeriveExponentialRewrite {
                rewritten: target_expr,
                kind: DeriveExponentialRewriteKind::ContractPower,
            });
        }
    }

    None
}

fn expand_exponential_sum_diff_expr(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let arg = cas_math::expr_extract::extract_exp_argument(ctx, expr)?;

    let mut positive_terms = Vec::new();
    let mut negative_terms = Vec::new();
    collect_signed_terms(ctx, arg, true, &mut positive_terms, &mut negative_terms);

    if positive_terms.len() + negative_terms.len() < 2 {
        return None;
    }

    let numerator = build_exp_product(ctx, &positive_terms);
    let denominator = build_exp_product(ctx, &negative_terms);

    match (numerator, denominator) {
        (Some(num), Some(den)) => Some(ctx.add(Expr::Div(num, den))),
        (Some(num), None) => Some(num),
        (None, Some(den)) => {
            let one = ctx.num(1);
            Some(ctx.add(Expr::Div(one, den)))
        }
        (None, None) => None,
    }
}

fn contract_exponential_sum_diff_expr(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let mut positive_terms = Vec::new();
    let mut negative_terms = Vec::new();
    if !collect_exp_factors(ctx, expr, true, &mut positive_terms, &mut negative_terms) {
        return None;
    }

    if positive_terms.is_empty() && negative_terms.is_empty() {
        return None;
    }

    if positive_terms.len() + negative_terms.len() < 2 {
        return None;
    }

    let arg = build_signed_sum(ctx, &positive_terms, &negative_terms)?;
    Some(ctx.call_builtin(BuiltinFn::Exp, vec![arg]))
}

fn collect_signed_terms(
    ctx: &cas_ast::Context,
    expr: ExprId,
    positive: bool,
    positive_terms: &mut Vec<ExprId>,
    negative_terms: &mut Vec<ExprId>,
) {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            collect_signed_terms(ctx, *left, positive, positive_terms, negative_terms);
            collect_signed_terms(ctx, *right, positive, positive_terms, negative_terms);
        }
        Expr::Sub(left, right) => {
            collect_signed_terms(ctx, *left, positive, positive_terms, negative_terms);
            collect_signed_terms(ctx, *right, !positive, positive_terms, negative_terms);
        }
        Expr::Neg(inner) => {
            collect_signed_terms(ctx, *inner, !positive, positive_terms, negative_terms);
        }
        _ => {
            if positive {
                positive_terms.push(expr);
            } else {
                negative_terms.push(expr);
            }
        }
    }
}

fn collect_exp_factors(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
    positive: bool,
    positive_terms: &mut Vec<ExprId>,
    negative_terms: &mut Vec<ExprId>,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Mul(left, right) => {
            collect_exp_factors(ctx, left, positive, positive_terms, negative_terms)
                && collect_exp_factors(ctx, right, positive, positive_terms, negative_terms)
        }
        Expr::Div(left, right) => {
            collect_exp_factors(ctx, left, positive, positive_terms, negative_terms)
                && collect_exp_factors(ctx, right, !positive, positive_terms, negative_terms)
        }
        Expr::Number(n) if n.is_one() => true,
        _ => {
            let Some(arg) = extract_exp_factor_argument(ctx, expr) else {
                return false;
            };
            if positive {
                positive_terms.push(arg);
            } else {
                negative_terms.push(arg);
            }
            true
        }
    }
}

fn extract_exp_factor_argument(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    if let Some(arg) = cas_math::expr_extract::extract_exp_argument(ctx, expr) {
        return Some(arg);
    }

    let (base, exponent) = as_pow(ctx, expr)?;
    if !matches!(ctx.get(exponent), Expr::Number(_)) {
        return None;
    }

    let arg = cas_math::expr_extract::extract_exp_argument(ctx, base)?;
    Some(ctx.add(Expr::Mul(exponent, arg)))
}

fn build_exp_product(ctx: &mut cas_ast::Context, terms: &[ExprId]) -> Option<ExprId> {
    let mut iter = terms.iter().copied();
    let first = iter.next()?;
    let mut product = ctx.call_builtin(BuiltinFn::Exp, vec![first]);
    for term in iter {
        let exp_term = ctx.call_builtin(BuiltinFn::Exp, vec![term]);
        product = ctx.add(Expr::Mul(product, exp_term));
    }
    Some(product)
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

fn apply_sign_to_term(ctx: &mut cas_ast::Context, term: ExprId, sign: Sign) -> ExprId {
    match sign {
        Sign::Pos => term,
        Sign::Neg => ctx.add(Expr::Neg(term)),
    }
}

fn expand_exponential_reciprocal_expr(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    if let Some(inner_expr) = extract_negated_inner(ctx, expr) {
        let reciprocal = expand_exponential_reciprocal_expr(ctx, inner_expr)?;
        return Some(negate_expr(ctx, reciprocal));
    }

    let arg = cas_math::expr_extract::extract_exp_argument(ctx, expr)?;
    let inner = extract_negated_inner(ctx, arg)?;
    let denominator = build_exp_power_from_negated_arg(ctx, inner);
    Some(build_signed_reciprocal(ctx, denominator, false))
}

fn contract_exponential_reciprocal_expr(
    ctx: &mut cas_ast::Context,
    expr: ExprId,
) -> Option<ExprId> {
    let (numerator, denominator) = as_div(ctx, expr)?;
    let numerator_sign = signed_unit_numerator(ctx, numerator)?;
    let arg = denominator_exp_argument(ctx, denominator)?;
    let exp_expr = build_exp_with_negated_arg(ctx, arg);

    Some(if numerator_sign.is_negative() {
        negate_expr(ctx, exp_expr)
    } else {
        exp_expr
    })
}

fn expand_exponential_power_expr(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    let arg = cas_math::expr_extract::extract_exp_argument(ctx, expr)?;
    let (lhs, rhs) = as_mul(ctx, arg)?;
    let (exponent, inner) = match (ctx.get(lhs), ctx.get(rhs)) {
        (Expr::Number(_), _) => (lhs, rhs),
        (_, Expr::Number(_)) => (rhs, lhs),
        _ => return None,
    };
    let base = ctx.call_builtin(BuiltinFn::Exp, vec![inner]);
    Some(ctx.add(Expr::Pow(base, exponent)))
}

fn contract_exponential_power_expr(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    if let Some((numerator, denominator)) = as_div(ctx, expr) {
        let numerator_sign = signed_unit_numerator(ctx, numerator)?;
        let (base, exponent) = as_pow(ctx, denominator)?;
        if !matches!(ctx.get(exponent), Expr::Number(_)) {
            return None;
        }
        let arg = cas_math::expr_extract::extract_exp_argument(ctx, base)?;
        let exp_expr = build_exp_with_negated_scaled_arg(ctx, arg, exponent);
        return Some(if numerator_sign.is_negative() {
            negate_expr(ctx, exp_expr)
        } else {
            exp_expr
        });
    }

    let (base, exponent) = as_pow(ctx, expr)?;
    if !matches!(ctx.get(exponent), Expr::Number(_)) {
        return None;
    }
    let arg = cas_math::expr_extract::extract_exp_argument(ctx, base)?;
    let scaled = ctx.add(Expr::Mul(arg, exponent));
    Some(ctx.call_builtin(BuiltinFn::Exp, vec![scaled]))
}

fn build_signed_sum(
    ctx: &mut cas_ast::Context,
    positive_terms: &[ExprId],
    negative_terms: &[ExprId],
) -> Option<ExprId> {
    let positive = build_add_chain(ctx, positive_terms);
    let negative = build_add_chain(ctx, negative_terms);

    match (positive, negative) {
        (Some(pos), Some(neg)) => Some(ctx.add(Expr::Sub(pos, neg))),
        (Some(pos), None) => Some(pos),
        (None, Some(neg)) => Some(ctx.add(Expr::Neg(neg))),
        (None, None) => None,
    }
}

fn build_add_chain(ctx: &mut cas_ast::Context, terms: &[ExprId]) -> Option<ExprId> {
    let mut iter = terms.iter().copied();
    let first = iter.next()?;
    let mut sum = first;
    for term in iter {
        sum = ctx.add(Expr::Add(sum, term));
    }
    Some(sum)
}

fn signed_unit_numerator(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Option<num_rational::BigRational> {
    match ctx.get(expr) {
        Expr::Number(n) if n.is_one() || n.is_negative() && (-n.clone()).is_one() => {
            Some(n.clone())
        }
        _ => None,
    }
}

fn denominator_exp_argument(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<ExprId> {
    if let Some(arg) = cas_math::expr_extract::extract_exp_argument(ctx, expr) {
        return Some(arg);
    }

    let (base, exponent) = as_pow(ctx, expr)?;
    if !matches!(ctx.get(exponent), Expr::Number(_)) {
        return None;
    }
    let arg = cas_math::expr_extract::extract_exp_argument(ctx, base)?;
    Some(ctx.add(Expr::Mul(exponent, arg)))
}

fn build_exp_power_from_negated_arg(ctx: &mut cas_ast::Context, arg: ExprId) -> ExprId {
    if let Some((lhs, rhs)) = as_mul(ctx, arg) {
        let (exponent, inner) = match (ctx.get(lhs), ctx.get(rhs)) {
            (Expr::Number(n), _) if n.is_positive() => (lhs, rhs),
            (_, Expr::Number(n)) if n.is_positive() => (rhs, lhs),
            _ => {
                let denominator = ctx.call_builtin(BuiltinFn::Exp, vec![arg]);
                return denominator;
            }
        };
        let base = ctx.call_builtin(BuiltinFn::Exp, vec![inner]);
        return ctx.add(Expr::Pow(base, exponent));
    }

    ctx.call_builtin(BuiltinFn::Exp, vec![arg])
}

fn build_exp_with_negated_arg(ctx: &mut cas_ast::Context, arg: ExprId) -> ExprId {
    let negated = ctx.add(Expr::Neg(arg));
    ctx.call_builtin(BuiltinFn::Exp, vec![negated])
}

fn build_exp_with_negated_scaled_arg(
    ctx: &mut cas_ast::Context,
    arg: ExprId,
    exponent: ExprId,
) -> ExprId {
    let scaled = ctx.add(Expr::Mul(exponent, arg));
    build_exp_with_negated_arg(ctx, scaled)
}

fn build_signed_reciprocal(
    ctx: &mut cas_ast::Context,
    denominator: ExprId,
    negative: bool,
) -> ExprId {
    let numerator = if negative { ctx.num(-1) } else { ctx.num(1) };
    ctx.add(Expr::Div(numerator, denominator))
}

fn negate_expr(ctx: &mut cas_ast::Context, expr: ExprId) -> ExprId {
    ctx.add(Expr::Neg(expr))
}

fn matches_target_modulo_simplify(ctx: &mut cas_ast::Context, left: ExprId, right: ExprId) -> bool {
    super::strong_target_match(ctx, left, right)
        || simplified_difference_matches_zero(ctx, left, right)
}

fn simplified_difference_matches_zero(
    ctx: &mut cas_ast::Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    let zero = ctx.num(0);
    let difference = ctx.add(cas_ast::Expr::Sub(left, right));
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (simplified, _steps, _stats) = simplifier.simplify_with_stats(
        difference,
        crate::SimplifyOptions {
            suppress_depth_overflow_warnings: true,
            ..crate::SimplifyOptions::default()
        },
    );
    std::mem::swap(&mut simplifier.context, ctx);
    super::strong_target_match(ctx, simplified, zero)
}

#[cfg(test)]
mod tests {
    use super::{try_rewrite_exponential_sum_diff_target_aware, DeriveExponentialRewriteKind};

    #[test]
    fn rewrites_exponential_sum_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("exp(x+y)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("exp(x)*exp(y)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_exponential_sum_diff_target_aware(&mut ctx, source, target)
            .expect("expected exponential target-aware rewrite");

        assert_eq!(rewrite.kind, DeriveExponentialRewriteKind::ExpandSumDiff);
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn rewrites_exponential_difference_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("exp(x-y)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("exp(x)/exp(y)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_exponential_sum_diff_target_aware(&mut ctx, source, target)
            .expect("expected exponential target-aware rewrite");

        assert_eq!(rewrite.kind, DeriveExponentialRewriteKind::ExpandSumDiff);
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn rewrites_exponential_product_contraction_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("exp(x)*exp(y)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("exp(x+y)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_exponential_sum_diff_target_aware(&mut ctx, source, target)
            .expect("expected exponential contraction");

        assert_eq!(rewrite.kind, DeriveExponentialRewriteKind::ContractSumDiff);
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn rewrites_exponential_reciprocal_contraction_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("1/exp(x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("exp(-x)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_exponential_sum_diff_target_aware(&mut ctx, source, target)
            .expect("expected exponential reciprocal contraction");

        assert_eq!(
            rewrite.kind,
            DeriveExponentialRewriteKind::ContractReciprocal
        );
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn rewrites_exponential_reciprocal_expansion_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("exp(-x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("1/exp(x)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_exponential_sum_diff_target_aware(&mut ctx, source, target)
            .expect("expected exponential reciprocal expansion");

        assert_eq!(rewrite.kind, DeriveExponentialRewriteKind::ExpandReciprocal);
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn rewrites_exponential_power_contraction_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("exp(x)^3", &mut ctx).expect("parse source");
        let target = cas_parser::parse("exp(3*x)", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_exponential_sum_diff_target_aware(&mut ctx, source, target)
            .expect("expected exponential power contraction");

        assert_eq!(rewrite.kind, DeriveExponentialRewriteKind::ContractPower);
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn rewrites_exponential_power_expansion_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("exp(3*x)", &mut ctx).expect("parse source");
        let target = cas_parser::parse("exp(x)^3", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_exponential_sum_diff_target_aware(&mut ctx, source, target)
            .expect("expected exponential power expansion");

        assert_eq!(rewrite.kind, DeriveExponentialRewriteKind::ExpandPower);
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn rewrites_exponential_family_variants_target_aware() {
        let cases = [
            (
                "exp(x)*exp(y)/exp(z)",
                "exp(x+y-z)",
                DeriveExponentialRewriteKind::ContractSumDiff,
            ),
            (
                "exp(x)/exp(y)^2",
                "exp(x-2*y)",
                DeriveExponentialRewriteKind::ContractSumDiff,
            ),
            (
                "exp(x)^2/exp(y)^3",
                "exp(2*x-3*y)",
                DeriveExponentialRewriteKind::ContractSumDiff,
            ),
            (
                "-1/exp(x)",
                "-exp(-x)",
                DeriveExponentialRewriteKind::ContractReciprocal,
            ),
            (
                "1/exp(x)^2",
                "exp(-2*x)",
                DeriveExponentialRewriteKind::ContractReciprocal,
            ),
        ];

        for (source_text, target_text, expected_kind) in cases {
            let mut ctx = cas_ast::Context::new();
            let source = cas_parser::parse(source_text, &mut ctx).expect("parse source");
            let target = cas_parser::parse(target_text, &mut ctx).expect("parse target");
            let rewrite = try_rewrite_exponential_sum_diff_target_aware(&mut ctx, source, target)
                .expect("expected exponential family rewrite");

            assert_eq!(rewrite.kind, expected_kind);
            assert_eq!(rewrite.rewritten, target);
        }
    }

    #[test]
    fn rewrites_exponential_sum_with_passthrough_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("exp(x+y)+a", &mut ctx).expect("parse source");
        let target = cas_parser::parse("exp(x)*exp(y)+a", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_exponential_sum_diff_target_aware(&mut ctx, source, target)
            .expect("expected exponential passthrough expansion");

        assert_eq!(rewrite.kind, DeriveExponentialRewriteKind::ExpandSumDiff);
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn rewrites_exponential_product_with_passthrough_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("exp(x)*exp(y)+a", &mut ctx).expect("parse source");
        let target = cas_parser::parse("exp(x+y)+a", &mut ctx).expect("parse target");
        let rewrite = try_rewrite_exponential_sum_diff_target_aware(&mut ctx, source, target)
            .expect("expected exponential passthrough contraction");

        assert_eq!(rewrite.kind, DeriveExponentialRewriteKind::ContractSumDiff);
        assert_eq!(rewrite.rewritten, target);
    }
}
