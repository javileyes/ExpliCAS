use cas_ast::{Context, ExprId};
use cas_math::expr_nary::{build_balanced_mul, mul_leaves};
use num_rational::BigRational;
use num_traits::{One, Zero};

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
    if let Some(rewrite) = try_combine_same_base_powers(ctx, source_expr) {
        if super::strong_target_match(ctx, rewrite.rewritten, target_expr) {
            return Some(PowerMergeRewrite {
                canonicalized: None,
                rewritten: rewrite.rewritten,
            });
        }
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
    let rewrite = try_combine_same_base_powers(ctx, canonicalized)?;
    if !super::strong_target_match(ctx, rewrite.rewritten, target_expr) {
        return None;
    }

    Some(PowerMergeRewrite {
        canonicalized: Some(canonicalized),
        rewritten: rewrite.rewritten,
    })
}

fn try_combine_same_base_powers(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<cas_math::power_product_support::PowerProductRewrite> {
    if let Some(rewrite) =
        cas_math::power_product_support::try_rewrite_mul_nary_combine_powers_expr(ctx, expr)
    {
        return Some(rewrite);
    }

    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 || factors.len() > 12 {
        return None;
    }

    let mut grouped: Vec<(ExprId, BigRational)> = Vec::new();
    let mut residual = Vec::new();
    let mut any_combined = false;

    for factor in factors {
        let Some((base, exp)) = extract_power_factor(ctx, factor) else {
            residual.push(factor);
            continue;
        };

        if let Some((_, existing_exp)) = grouped.iter_mut().find(|(existing_base, _)| {
            cas_ast::ordering::compare_expr(ctx, *existing_base, base).is_eq()
        }) {
            *existing_exp += exp;
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
        if exp.is_zero() {
            rebuilt.push(ctx.num(1));
        } else if exp.is_one() {
            rebuilt.push(base);
        } else {
            let exp_expr = ctx.add(cas_ast::Expr::Number(exp));
            rebuilt.push(ctx.add(cas_ast::Expr::Pow(base, exp_expr)));
        }
    }
    rebuilt.extend(residual);

    Some(cas_math::power_product_support::PowerProductRewrite {
        rewritten: build_balanced_mul(ctx, rebuilt.as_slice()),
        kind: cas_math::power_product_support::PowerProductRewriteKind::SameBaseNary,
    })
}

fn extract_power_factor(ctx: &Context, factor: ExprId) -> Option<(ExprId, BigRational)> {
    match ctx.get(factor) {
        cas_ast::Expr::Pow(base, exp) => {
            let exp_value = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
            Some((*base, exp_value))
        }
        cas_ast::Expr::Number(_) => None,
        _ => Some((factor, BigRational::one())),
    }
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_power_merge_target_aware;
    use cas_formatter::DisplayExpr;

    #[test]
    fn merges_same_base_fractional_powers_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("x^(1/2)*x^(2/3)", &mut ctx).expect("source");
        let target = cas_parser::parse("x^(7/6)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_power_merge_target_aware(&mut ctx, source, target).expect("rewrite");
        assert!(rewrite.canonicalized.is_none());
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.rewritten,
                }
            ),
            "x^(7/6)"
        );
    }

    #[test]
    fn canonicalizes_root_then_merges_power_target_aware() {
        let mut ctx = cas_ast::Context::new();
        let source = cas_parser::parse("sqrt(x)*x^(2/3)", &mut ctx).expect("source");
        let target = cas_parser::parse("x^(7/6)", &mut ctx).expect("target");
        let rewrite =
            try_rewrite_power_merge_target_aware(&mut ctx, source, target).expect("rewrite");
        assert!(rewrite.canonicalized.is_some());
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.rewritten,
                }
            ),
            "x^(7/6)"
        );
    }
}
