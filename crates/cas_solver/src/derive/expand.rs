use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_nary::{add_terms_signed, Sign};
use cas_math::expr_relations::conjugate_nary_add_sub_pair;
use cas_math::trig_roots_flatten::flatten_mul_chain;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpandRewriteKind {
    BinomialPower,
    SophieGermainProduct,
    HyperbolicAngleSumDiff,
    HyperbolicProductSum,
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
    if try_rewrite_sophie_germain_expansion_target_aware(ctx, source_expr, target_expr).is_some() {
        return Some(ExpandRewrite {
            rewritten: target_expr,
            kind: ExpandRewriteKind::SophieGermainProduct,
        });
    }

    if let Some(kind) =
        super::try_rewrite_hyperbolic_expansion_target_aware(ctx, source_expr, target_expr)
    {
        return Some(ExpandRewrite {
            rewritten: target_expr,
            kind: match kind {
                super::hyperbolic::DeriveHyperbolicRewriteKind::ProductToSumSinhCosh
                | super::hyperbolic::DeriveHyperbolicRewriteKind::ProductToSumCoshCosh
                | super::hyperbolic::DeriveHyperbolicRewriteKind::ProductToSumSinhSinh
                | super::hyperbolic::DeriveHyperbolicRewriteKind::SumToProductSinhCosh
                | super::hyperbolic::DeriveHyperbolicRewriteKind::SumToProductCoshCosh
                | super::hyperbolic::DeriveHyperbolicRewriteKind::SumToProductSinhSinh => {
                    ExpandRewriteKind::HyperbolicProductSum
                }
                _ => ExpandRewriteKind::HyperbolicAngleSumDiff,
            },
        });
    }

    if let Some(rewrite) =
        try_rewrite_hyperbolic_additive_product_sum_target_aware(ctx, source_expr, target_expr)
    {
        return Some(rewrite);
    }

    let rewritten = run_engine_expand_target_aware(ctx, source_expr, target_expr)?;

    Some(ExpandRewrite {
        rewritten,
        kind: classify_expand_rewrite_kind(ctx, source_expr),
    })
}

fn try_rewrite_sophie_germain_expansion_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<ExprId> {
    let (left, right) = match ctx.get(source_expr) {
        Expr::Mul(left, right) => (*left, *right),
        _ => return None,
    };
    let (u, v) = conjugate_nary_add_sub_pair(ctx, left, right)?;
    let (a, b) = extract_sophie_germain_components(ctx, u, v)?;

    let four = ctx.num(4);
    let a_fourth = ctx.add(Expr::Pow(a, four));
    let b_fourth = ctx.add(Expr::Pow(b, four));
    let scaled_b_fourth = ctx.add(Expr::Mul(four, b_fourth));
    let candidate = ctx.add(Expr::Add(a_fourth, scaled_b_fourth));

    super::strong_target_match(ctx, candidate, target_expr).then_some(target_expr)
}

fn run_engine_expand_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<ExprId> {
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (rewritten, _steps) = simplifier.expand(source_expr);
    std::mem::swap(&mut simplifier.context, ctx);

    if rewritten == source_expr || !matches_expanded_target(ctx, rewritten, target_expr) {
        return None;
    }

    Some(rewritten)
}

fn try_rewrite_hyperbolic_additive_product_sum_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<ExpandRewrite> {
    for rewrite in super::generate_hyperbolic_additive_term_bridge_rewrites(ctx, source_expr) {
        if !matches!(
            rewrite.kind,
            super::hyperbolic::DeriveHyperbolicRewriteKind::ProductToSumSinhCosh
                | super::hyperbolic::DeriveHyperbolicRewriteKind::ProductToSumCoshCosh
                | super::hyperbolic::DeriveHyperbolicRewriteKind::ProductToSumSinhSinh
                | super::hyperbolic::DeriveHyperbolicRewriteKind::SumToProductSinhCosh
                | super::hyperbolic::DeriveHyperbolicRewriteKind::SumToProductCoshCosh
                | super::hyperbolic::DeriveHyperbolicRewriteKind::SumToProductSinhSinh
        ) {
            continue;
        }

        if matches_expanded_target(ctx, rewrite.rewritten, target_expr) {
            return Some(ExpandRewrite {
                rewritten: target_expr,
                kind: ExpandRewriteKind::HyperbolicProductSum,
            });
        }
    }

    None
}

fn matches_expanded_target(ctx: &mut Context, actual: ExprId, target: ExprId) -> bool {
    if super::strong_target_match(ctx, actual, target) {
        return true;
    }

    let difference = ctx.add(Expr::Sub(actual, target));
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
    fn contains_binomial_power(ctx: &Context, expr: ExprId) -> bool {
        match ctx.get(expr) {
            Expr::Pow(base, exp)
                if matches!(ctx.get(*base), Expr::Add(_, _) | Expr::Sub(_, _))
                    && matches!(ctx.get(*exp), Expr::Number(_)) =>
            {
                true
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                contains_binomial_power(ctx, *l) || contains_binomial_power(ctx, *r)
            }
            Expr::Neg(inner) | Expr::Hold(inner) => contains_binomial_power(ctx, *inner),
            Expr::Function(_, args) => args
                .iter()
                .copied()
                .any(|arg| contains_binomial_power(ctx, arg)),
            Expr::Pow(base, exp) => {
                contains_binomial_power(ctx, *base) || contains_binomial_power(ctx, *exp)
            }
            Expr::Matrix { data, .. } => data
                .iter()
                .copied()
                .any(|item| contains_binomial_power(ctx, item)),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
        }
    }

    match ctx.get(expr) {
        _ if contains_binomial_power(ctx, expr) => ExpandRewriteKind::BinomialPower,
        _ => ExpandRewriteKind::General,
    }
}

fn extract_integer_power_base(ctx: &Context, expr: ExprId, exponent: i64) -> Option<ExprId> {
    let (base, power) = match ctx.get(expr) {
        Expr::Pow(base, power) => (*base, *power),
        _ => return None,
    };

    is_integer_literal(ctx, power, exponent).then_some(base)
}

fn extract_scaled_power_base(
    ctx: &mut Context,
    expr: ExprId,
    coefficient: i64,
    exponent: i64,
) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut coefficient_seen = false;
    let mut power_base = None;

    for factor in factors {
        if !coefficient_seen && is_integer_literal(ctx, factor, coefficient) {
            coefficient_seen = true;
            continue;
        }

        if power_base.is_none() {
            if let Some(base) = extract_integer_power_base(ctx, factor, exponent) {
                power_base = Some(base);
                continue;
            }
        }

        return None;
    }

    if coefficient_seen {
        power_base
    } else {
        None
    }
}

fn extract_scaled_binary_product_pair(
    ctx: &mut Context,
    expr: ExprId,
    coefficient: i64,
) -> Option<(ExprId, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut coefficient_seen = false;
    let mut unsigned_factors = Vec::new();

    for factor in factors {
        if !coefficient_seen && is_integer_literal(ctx, factor, coefficient) {
            coefficient_seen = true;
            continue;
        }
        unsigned_factors.push(factor);
    }

    if coefficient_seen && unsigned_factors.len() == 2 {
        Some((unsigned_factors[0], unsigned_factors[1]))
    } else {
        None
    }
}

fn extract_sophie_germain_components(
    ctx: &mut Context,
    u: ExprId,
    v: ExprId,
) -> Option<(ExprId, ExprId)> {
    let u_terms = add_terms_signed(ctx, u);
    if u_terms.len() != 2 || u_terms.iter().any(|(_, sign)| *sign != Sign::Pos) {
        return None;
    }

    let mut squared_base = None;
    let mut scaled_squared_base = None;
    for (term, _) in u_terms {
        if let Some(base) = extract_integer_power_base(ctx, term, 2) {
            squared_base = Some(base);
            continue;
        }

        if let Some(base) = extract_scaled_power_base(ctx, term, 2, 2) {
            scaled_squared_base = Some(base);
            continue;
        }

        return None;
    }

    let a = squared_base?;
    let b = scaled_squared_base?;
    let (left, right) = extract_scaled_binary_product_pair(ctx, v, 2)?;
    let direct_match =
        super::strong_target_match(ctx, left, a) && super::strong_target_match(ctx, right, b);
    let swapped_match =
        super::strong_target_match(ctx, left, b) && super::strong_target_match(ctx, right, a);

    (direct_match || swapped_match).then_some((a, b))
}

fn is_integer_literal(ctx: &Context, expr: ExprId, value: i64) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(number) if number.is_integer() && number.to_integer() == value.into()
    )
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
    fn expands_fractional_binomial_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("(x + 1/2)^2", &mut ctx).expect("source");
        let target = cas_parser::parse("x^2 + x + 1/4", &mut ctx).expect("target");
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

    #[test]
    fn expands_hyperbolic_sinh_sum_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("sinh(x+y)", &mut ctx).expect("source");
        let target =
            cas_parser::parse("sinh(x)*cosh(y) + cosh(x)*sinh(y)", &mut ctx).expect("target");
        let rewrite = try_rewrite_expanded_target_aware(&mut ctx, source, target).expect("rewrite");
        assert_eq!(rewrite.kind, ExpandRewriteKind::HyperbolicAngleSumDiff);
    }

    #[test]
    fn expands_recursive_hyperbolic_sinh_six_x_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("sinh(6*x)", &mut ctx).expect("source");
        let target =
            cas_parser::parse("sinh(5*x)*cosh(x) + cosh(5*x)*sinh(x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_expanded_target_aware(&mut ctx, source, target).expect("rewrite");
        assert_eq!(rewrite.kind, ExpandRewriteKind::HyperbolicAngleSumDiff);
    }

    #[test]
    fn expands_hyperbolic_product_to_sum_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("2*sinh(2*x)*cosh(x)", &mut ctx).expect("source");
        let target = cas_parser::parse("sinh(3*x)+sinh(x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_expanded_target_aware(&mut ctx, source, target).expect("rewrite");
        assert_eq!(rewrite.kind, ExpandRewriteKind::HyperbolicProductSum);
    }

    #[test]
    fn expands_sophie_germain_product_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("(x^2 - 2*x*y + 2*y^2)*(x^2 + 2*x*y + 2*y^2)", &mut ctx)
            .expect("source");
        let target = cas_parser::parse("x^4 + 4*y^4", &mut ctx).expect("target");
        let rewrite = try_rewrite_expanded_target_aware(&mut ctx, source, target).expect("rewrite");
        assert_eq!(rewrite.kind, ExpandRewriteKind::SophieGermainProduct);
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn contracts_hyperbolic_sum_to_product_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("sinh(3*x)-sinh(x)", &mut ctx).expect("source");
        let target = cas_parser::parse("2*cosh(2*x)*sinh(x)", &mut ctx).expect("target");
        let rewrite = try_rewrite_expanded_target_aware(&mut ctx, source, target).expect("rewrite");
        assert_eq!(rewrite.kind, ExpandRewriteKind::HyperbolicProductSum);
    }

    #[test]
    fn expands_hyperbolic_product_to_sum_with_passthrough_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("2*sinh(2*x)*sinh(x)+a", &mut ctx).expect("source");
        let target = cas_parser::parse("4*cosh(x)^3-4*cosh(x)+a", &mut ctx).expect("target");
        let rewrite = try_rewrite_expanded_target_aware(&mut ctx, source, target).expect("rewrite");
        assert_eq!(rewrite.kind, ExpandRewriteKind::HyperbolicProductSum);
        assert_eq!(rewrite.rewritten, target);
    }

    #[test]
    fn classifies_binomial_power_inside_expression_as_binomial_expansion() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("(a+b)^2 - a^2 - 2*a*b", &mut ctx).expect("source");
        let target = cas_parser::parse("b^2", &mut ctx).expect("target");
        let rewrite = try_rewrite_expanded_target_aware(&mut ctx, source, target).expect("rewrite");
        assert_eq!(rewrite.kind, ExpandRewriteKind::BinomialPower);
    }
}
