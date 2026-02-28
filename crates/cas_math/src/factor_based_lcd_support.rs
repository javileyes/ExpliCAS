//! Factor-based LCD combination support for additive fractions.

use crate::build::mul2_raw;
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy)]
pub struct FactorBasedLcdRewrite {
    pub rewritten: ExprId,
}

/// Combine 3+ fractions using LCD built from normalized binomial factors.
///
/// This intentionally mirrors engine behavior used by `FactorBasedLCDRule`.
pub fn try_rewrite_factor_based_lcd_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<FactorBasedLcdRewrite> {
    let normalize_binomial = |ctx: &mut Context, e: ExprId| -> (ExprId, bool) {
        let add_parts = match ctx.get(e) {
            Expr::Add(l, r) => Some((*l, *r)),
            _ => None,
        };
        let sub_parts = match ctx.get(e) {
            Expr::Sub(l, r) => Some((*l, *r)),
            _ => None,
        };

        if let Some((l, r)) = add_parts {
            let neg_inner = match ctx.get(r) {
                Expr::Neg(inner) => Some(*inner),
                _ => None,
            };
            if let Some(inner) = neg_inner {
                if compare_expr(ctx, l, inner) == Ordering::Less {
                    (e, false)
                } else {
                    let neg_l = ctx.add(Expr::Neg(l));
                    let canonical = ctx.add(Expr::Add(inner, neg_l));
                    (canonical, true)
                }
            } else {
                (e, false)
            }
        } else if let Some((l, r)) = sub_parts {
            if compare_expr(ctx, l, r) == Ordering::Less {
                (e, false)
            } else {
                let canonical = ctx.add(Expr::Sub(r, l));
                (canonical, true)
            }
        } else {
            (e, false)
        }
    };

    let get_factors = |ctx: &Context, e: ExprId| -> Vec<ExprId> {
        let mut factors = Vec::new();
        let mut stack = vec![e];
        while let Some(curr) = stack.pop() {
            match ctx.get(curr) {
                Expr::Mul(l, r) => {
                    stack.push(*l);
                    stack.push(*r);
                }
                _ => factors.push(curr),
            }
        }
        factors
    };

    let is_binomial = |ctx: &Context, e: ExprId| -> bool {
        match ctx.get(e) {
            Expr::Add(_, r) => matches!(ctx.get(*r), Expr::Neg(_)),
            Expr::Sub(_, _) => true,
            _ => false,
        }
    };

    let expr_eq = |ctx: &Context, a: ExprId, b: ExprId| compare_expr(ctx, a, b) == Ordering::Equal;

    let mut terms = Vec::new();
    let mut stack = vec![expr];
    while let Some(curr) = stack.pop() {
        match ctx.get(curr) {
            Expr::Add(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            _ => terms.push(curr),
        }
    }

    if terms.len() < 3 {
        return None;
    }

    let mut fractions: Vec<(ExprId, ExprId)> = Vec::new();
    for term in &terms {
        match ctx.get(*term) {
            Expr::Div(num, den) => fractions.push((*num, *den)),
            _ => return None,
        }
    }

    let mut all_factor_sets: Vec<Vec<(ExprId, bool)>> = Vec::new();
    for (_, den) in &fractions {
        let raw_factors = get_factors(ctx, *den);
        let mut normalized = Vec::new();
        for f in raw_factors {
            if !is_binomial(ctx, f) {
                return None;
            }
            let (canonical, flipped) = normalize_binomial(ctx, f);
            normalized.push((canonical, flipped));
        }
        if normalized.is_empty() {
            return None;
        }
        all_factor_sets.push(normalized);
    }

    let mut unique_factors: Vec<ExprId> = Vec::new();
    for factor_set in &all_factor_sets {
        for (canonical, _) in factor_set {
            let exists = unique_factors.iter().any(|u| expr_eq(ctx, *u, *canonical));
            if !exists {
                unique_factors.push(*canonical);
            }
        }
    }

    let all_same = all_factor_sets.iter().all(|fs| {
        fs.len() == unique_factors.len()
            && unique_factors
                .iter()
                .all(|uf| fs.iter().any(|(cf, _)| expr_eq(ctx, *cf, *uf)))
    });
    if all_same && fractions.len() == terms.len() {
        return None;
    }

    let lcd = if unique_factors.len() == 1 {
        unique_factors[0]
    } else {
        let (&first, rest) = unique_factors.split_first()?;
        rest.iter()
            .copied()
            .fold(first, |acc, f| mul2_raw(ctx, acc, f))
    };

    let mut numerator_terms: Vec<ExprId> = Vec::new();
    for (i, (num, _den)) in fractions.iter().enumerate() {
        let factor_set = &all_factor_sets[i];
        let sign_flips: usize = factor_set.iter().filter(|(_, f)| *f).count();
        let is_negative = sign_flips % 2 == 1;

        let mut missing: Vec<ExprId> = Vec::new();
        for uf in &unique_factors {
            let present = factor_set.iter().any(|(cf, _)| expr_eq(ctx, *cf, *uf));
            if !present {
                missing.push(*uf);
            }
        }

        let mut contribution = *num;
        for mf in missing {
            contribution = mul2_raw(ctx, contribution, mf);
        }
        if is_negative {
            contribution = ctx.add(Expr::Neg(contribution));
        }
        numerator_terms.push(contribution);
    }

    let total_num = if numerator_terms.len() == 1 {
        numerator_terms[0]
    } else {
        let (&first, rest) = numerator_terms.split_first()?;
        rest.iter()
            .copied()
            .fold(first, |acc, term| ctx.add(Expr::Add(acc, term)))
    };

    Some(FactorBasedLcdRewrite {
        rewritten: ctx.add(Expr::Div(total_num, lcd)),
    })
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_factor_based_lcd_expr;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn combines_three_fraction_terms_with_factor_lcd() {
        let mut ctx = Context::new();
        let expr = parse("1/(x-1) + 1/(x-2) + 1/(x-1)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_factor_based_lcd_expr(&mut ctx, expr);
        assert!(rewrite.is_some());
    }
}
