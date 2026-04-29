use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_destructure::as_div;
use cas_math::expr_nary::{build_balanced_mul, mul_leaves};
use cas_math::poly_compare::poly_eq;
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PowerMergeRewrite {
    pub(crate) canonicalized: Option<ExprId>,
    pub(crate) rewritten: ExprId,
}

pub(crate) fn try_rewrite_power_merge_target_aware(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> Option<PowerMergeRewrite> {
    if let Some(rewrite) = try_combine_same_base_powers_to_target(ctx, source_expr, target_expr) {
        return Some(PowerMergeRewrite {
            canonicalized: None,
            rewritten: rewrite.rewritten,
        });
    }

    let factors = mul_leaves(ctx, source_expr);
    if factors.len() < 2 {
        return None;
    }

    let mut changed = false;
    let rewritten_factors = factors
        .iter()
        .copied()
        .map(|factor| {
            cas_math::root_forms::try_rewrite_canonical_root_expr(ctx, factor)
                .map(|rewrite| {
                    changed = true;
                    rewrite.rewritten
                })
                .unwrap_or(factor)
        })
        .collect::<Vec<_>>();

    if !changed {
        return None;
    }

    let canonicalized = build_balanced_mul(ctx, rewritten_factors.as_slice());
    let rewrite = try_combine_same_base_powers_to_target(ctx, canonicalized, target_expr)?;

    Some(PowerMergeRewrite {
        canonicalized: Some(canonicalized),
        rewritten: rewrite.rewritten,
    })
}

fn try_combine_same_base_powers_to_target(
    ctx: &mut Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<cas_math::power_product_support::PowerProductRewrite> {
    if let Some(rewrite) = try_combine_same_base_quotient_powers_to_target(ctx, expr, target_expr) {
        return Some(rewrite);
    }

    if let Some(rewrite) = try_combine_same_base_powers_generic(ctx, expr) {
        if power_merge_matches_target(ctx, rewrite.rewritten, target_expr) {
            return Some(retarget_power_merge_rewrite(ctx, rewrite, target_expr));
        }
    }

    let rewrite = try_combine_same_base_powers_symbolic(ctx, expr)?;
    power_merge_matches_target(ctx, rewrite.rewritten, target_expr)
        .then_some(retarget_power_merge_rewrite(ctx, rewrite, target_expr))
}

fn try_combine_same_base_quotient_powers_to_target(
    ctx: &mut Context,
    expr: ExprId,
    target_expr: ExprId,
) -> Option<cas_math::power_product_support::PowerProductRewrite> {
    let (numerator, denominator) = as_div(ctx, expr)?;
    let (num_base, num_exp) = extract_power_factor(ctx, numerator)?;
    let (den_base, den_exp) = extract_power_factor(ctx, denominator)?;

    if cas_ast::ordering::compare_expr(ctx, num_base, den_base) != Ordering::Equal {
        return None;
    }

    let raw_exponent = ctx.add(Expr::Sub(num_exp, den_exp));
    let exponent = normalize_merged_exponent(ctx, raw_exponent);
    let zero = ctx.num(0);
    let one = ctx.num(1);
    let rewritten = if cas_ast::ordering::compare_expr(ctx, exponent, zero) == Ordering::Equal {
        one
    } else if cas_ast::ordering::compare_expr(ctx, exponent, one) == Ordering::Equal {
        num_base
    } else {
        ctx.add(Expr::Pow(num_base, exponent))
    };

    let rewrite = cas_math::power_product_support::PowerProductRewrite {
        rewritten,
        kind: cas_math::power_product_support::PowerProductRewriteKind::SameBaseNary,
    };

    power_merge_matches_target(ctx, rewrite.rewritten, target_expr)
        .then_some(retarget_power_merge_rewrite(ctx, rewrite, target_expr))
}

fn try_combine_same_base_powers_generic(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<cas_math::power_product_support::PowerProductRewrite> {
    cas_math::power_product_support::try_rewrite_mul_nary_combine_powers_expr(ctx, expr)
}

fn try_combine_same_base_powers_symbolic(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<cas_math::power_product_support::PowerProductRewrite> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 || factors.len() > 12 {
        return None;
    }

    let mut grouped: Vec<(ExprId, ExprId)> = Vec::new();
    let mut any_combined = false;
    let one = ctx.num(1);
    let zero = ctx.num(0);

    for factor in factors {
        let Some((base, exp)) = extract_power_factor(ctx, factor) else {
            continue;
        };

        if let Some((_, existing_exp)) = grouped.iter_mut().find(|(existing_base, _)| {
            cas_ast::ordering::compare_expr(ctx, *existing_base, base) == Ordering::Equal
        }) {
            let combined = cas_math::exponents_support::add_exp(ctx, *existing_exp, exp);
            *existing_exp = normalize_merged_exponent(ctx, combined);
            any_combined = true;
        } else {
            grouped.push((base, exp));
        }
    }

    if !any_combined {
        return None;
    }

    let mut rebuilt = Vec::new();
    for (base, exp) in grouped {
        if cas_ast::ordering::compare_expr(ctx, exp, zero) == Ordering::Equal {
            rebuilt.push(one);
        } else if cas_ast::ordering::compare_expr(ctx, exp, one) == Ordering::Equal {
            rebuilt.push(base);
        } else {
            rebuilt.push(ctx.add(cas_ast::Expr::Pow(base, exp)));
        }
    }

    Some(cas_math::power_product_support::PowerProductRewrite {
        rewritten: build_balanced_mul(ctx, rebuilt.as_slice()),
        kind: cas_math::power_product_support::PowerProductRewriteKind::SameBaseNary,
    })
}

fn extract_power_factor(ctx: &mut Context, factor: ExprId) -> Option<(ExprId, ExprId)> {
    if let cas_ast::Expr::Pow(base, exp) = ctx.get(factor) {
        return Some((*base, normalize_merged_exponent(ctx, *exp)));
    }

    if matches!(ctx.get(factor), cas_ast::Expr::Number(_)) {
        return None;
    }

    Some((factor, ctx.num(1)))
}

fn normalize_merged_exponent(ctx: &mut Context, exponent: ExprId) -> ExprId {
    if let Some(value) = cas_ast::views::as_rational_const(ctx, exponent, 8) {
        return ctx.add(cas_ast::Expr::Number(value));
    }

    cas_math::canonical_forms::normalize_core(ctx, exponent)
}

fn power_merge_matches_target(ctx: &mut Context, rewritten: ExprId, target_expr: ExprId) -> bool {
    if super::strong_target_match(ctx, rewritten, target_expr) {
        return true;
    }

    let (rewritten_base, rewritten_exp) = match ctx.get(rewritten) {
        cas_ast::Expr::Pow(base, exp) => (*base, *exp),
        _ => return false,
    };
    let (target_base, target_exp) = match ctx.get(target_expr) {
        cas_ast::Expr::Pow(base, exp) => (*base, *exp),
        _ => return false,
    };

    (super::strong_target_match(ctx, rewritten_base, target_base)
        || poly_eq(ctx, rewritten_base, target_base))
        && (super::strong_target_match(ctx, rewritten_exp, target_exp)
            || poly_eq(ctx, rewritten_exp, target_exp))
}

fn retarget_power_merge_rewrite(
    ctx: &mut Context,
    rewrite: cas_math::power_product_support::PowerProductRewrite,
    target_expr: ExprId,
) -> cas_math::power_product_support::PowerProductRewrite {
    if super::strong_target_match(ctx, rewrite.rewritten, target_expr) {
        return rewrite;
    }

    cas_math::power_product_support::PowerProductRewrite {
        rewritten: target_expr,
        kind: rewrite.kind,
    }
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_power_merge_target_aware;
    use cas_math::semantic_equality::SemanticEqualityChecker;

    #[test]
    fn merges_tabulated_power_targets_aware() {
        let cases = [
            ("x^(1/2)*x^(2/3)", "x^(7/6)", false),
            ("x^(3/4)*x^(1/4)", "x", false),
            ("x*x^(1/3)", "x^(4/3)", false),
            ("x^a*x^b", "x^(a+b)", false),
            ("x*x^a", "x^(a+1)", false),
            ("x^a*x^b*x^c*x^d", "x^(a+b+c+d)", false),
            ("x^a/x^b", "x^(a-b)", false),
            ("2^a/2^b", "2^(a-b)", false),
            ("sqrt(x)*x^(1/3)", "x^(5/6)", true),
            ("sqrt(x)*x^a", "x^(a+1/2)", true),
        ];

        for (source, target, expect_canonicalized) in cases {
            let mut ctx = cas_ast::Context::new();
            let source = cas_parser::parse(source, &mut ctx).expect("source");
            let target = cas_parser::parse(target, &mut ctx).expect("target");
            let rewrite =
                try_rewrite_power_merge_target_aware(&mut ctx, source, target).expect("rewrite");
            assert_eq!(
                rewrite.canonicalized.is_some(),
                expect_canonicalized,
                "unexpected canonicalization flag for target-aware merge"
            );
            let checker = SemanticEqualityChecker::new(&ctx);
            assert!(
                checker.are_equal(rewrite.rewritten, target),
                "expected rewrite to equal target"
            );
        }
    }
}
