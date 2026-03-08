//! Legacy-compatible multiplicative distribution support.
//!
//! This module mirrors the historical `DistributeRule` multiplication branches
//! from `cas_engine`, preserving guard ordering and behavior.

use crate::cube_identity_support::is_cube_identity_product;
use crate::distribution_guard_support::{
    is_binomial_expr, should_block_fractional_coeff_over_binomial,
    should_distribute_factor_over_additive, should_skip_distribution_for_factor,
};
use crate::expr_destructure::{as_add, as_mul, as_sub};
use crate::expr_relations::is_conjugate_add_sub;
use crate::expr_rewrite::smart_mul;
use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistributionMulRewrite {
    pub rewritten: ExprId,
}

/// Plan multiplicative distribution preserving legacy guard semantics.
///
/// Handles:
/// - `a * (b + c)` / `a * (b - c)`
/// - `(b + c) * a` / `(b - c) * a`
///
/// Returns `None` when any anti-oscillation or anti-explosion guard triggers.
pub fn try_rewrite_mul_distribution_legacy_expr(
    ctx: &mut Context,
    expr: ExprId,
    parent_mul_terms: Option<(ExprId, ExprId)>,
) -> Option<DistributionMulRewrite> {
    let (l, r) = as_mul(ctx, expr)?;

    if crate::expr_predicates::is_one_expr(ctx, l) || crate::expr_predicates::is_one_expr(ctx, r) {
        return None;
    }

    // a * (b + c) -> a*b + a*c
    if let Some((b, c)) = as_add(ctx, r) {
        if should_skip_distribution_for_factor(ctx, l, r) {
            return None;
        }
        if !should_distribute_factor_over_additive(ctx, l, r, expr) {
            return None;
        }
        if is_conjugate_add_sub(ctx, l, r) {
            return None;
        }
        if let Some((pl, pr)) = parent_mul_terms {
            if is_conjugate_add_sub(ctx, r, pl) || is_conjugate_add_sub(ctx, r, pr) {
                return None;
            }
        }
        if is_binomial_expr(ctx, l)
            && is_binomial_expr(ctx, r)
            && !is_cube_identity_product(ctx, l, r)
        {
            return None;
        }
        if should_block_fractional_coeff_over_binomial(ctx, l, r) {
            return None;
        }

        let ab = smart_mul(ctx, l, b);
        let ac = smart_mul(ctx, l, c);
        return Some(DistributionMulRewrite {
            rewritten: ctx.add(Expr::Add(ab, ac)),
        });
    }

    // a * (b - c) -> a*b - a*c
    if let Some((b, c)) = as_sub(ctx, r) {
        if should_skip_distribution_for_factor(ctx, l, r) {
            return None;
        }
        if !should_distribute_factor_over_additive(ctx, l, r, expr) {
            return None;
        }
        if is_conjugate_add_sub(ctx, l, r) {
            return None;
        }
        if let Some((pl, pr)) = parent_mul_terms {
            if is_conjugate_add_sub(ctx, r, pl) || is_conjugate_add_sub(ctx, r, pr) {
                return None;
            }
        }
        if is_binomial_expr(ctx, l)
            && is_binomial_expr(ctx, r)
            && !is_cube_identity_product(ctx, l, r)
        {
            return None;
        }
        if should_block_fractional_coeff_over_binomial(ctx, l, r) {
            return None;
        }

        let ab = smart_mul(ctx, l, b);
        let ac = smart_mul(ctx, l, c);
        return Some(DistributionMulRewrite {
            rewritten: ctx.add(Expr::Sub(ab, ac)),
        });
    }

    // (b + c) * a -> b*a + c*a
    if let Some((b, c)) = as_add(ctx, l) {
        if should_skip_distribution_for_factor(ctx, r, l) {
            return None;
        }
        if !should_distribute_factor_over_additive(ctx, r, l, expr) {
            return None;
        }
        if is_conjugate_add_sub(ctx, l, r) {
            return None;
        }
        if let Some((pl, pr)) = parent_mul_terms {
            if is_conjugate_add_sub(ctx, l, pl) || is_conjugate_add_sub(ctx, l, pr) {
                return None;
            }
        }
        if is_binomial_expr(ctx, l)
            && is_binomial_expr(ctx, r)
            && !is_cube_identity_product(ctx, l, r)
        {
            return None;
        }
        if should_block_fractional_coeff_over_binomial(ctx, r, l) {
            return None;
        }

        let ba = smart_mul(ctx, b, r);
        let ca = smart_mul(ctx, c, r);
        return Some(DistributionMulRewrite {
            rewritten: ctx.add(Expr::Add(ba, ca)),
        });
    }

    // (b - c) * a -> b*a - c*a
    if let Some((b, c)) = as_sub(ctx, l) {
        if should_skip_distribution_for_factor(ctx, r, l) {
            return None;
        }
        if !should_distribute_factor_over_additive(ctx, r, l, expr) {
            return None;
        }
        if is_conjugate_add_sub(ctx, l, r) {
            return None;
        }
        if let Some((pl, pr)) = parent_mul_terms {
            if is_conjugate_add_sub(ctx, l, pl) || is_conjugate_add_sub(ctx, l, pr) {
                return None;
            }
        }
        if is_binomial_expr(ctx, l)
            && is_binomial_expr(ctx, r)
            && !is_cube_identity_product(ctx, l, r)
        {
            return None;
        }
        if should_block_fractional_coeff_over_binomial(ctx, r, l) {
            return None;
        }

        let ba = smart_mul(ctx, b, r);
        let ca = smart_mul(ctx, c, r);
        return Some(DistributionMulRewrite {
            rewritten: ctx.add(Expr::Sub(ba, ca)),
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_mul_distribution_legacy_expr;
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    #[test]
    fn rewrites_simple_mul_add_distribution() {
        let mut ctx = Context::new();
        let expr = parse("x^2 * (x + 3)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_mul_distribution_legacy_expr(&mut ctx, expr, None)
            .expect("rewrite expected");
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.rewritten
                }
            ),
            "x^2 * x + x^2 * 3"
        );
    }

    #[test]
    fn blocks_conjugate_distribution() {
        let mut ctx = Context::new();
        let expr = parse("(a+b) * (a-b)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_mul_distribution_legacy_expr(&mut ctx, expr, None);
        assert!(rewrite.is_none());
    }
}
