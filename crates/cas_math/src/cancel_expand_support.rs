//! Context-aware expansion support for equation-level cancellation.

use crate::cancel_normalization_support::OriginSafety;
use crate::cancel_support::{
    collect_additive_terms_signed as collect_additive_terms, mul_preview,
    structural_expr_fingerprint as term_fingerprint,
};
use cas_ast::{Context, Expr, ExprId};
use num_traits::{Signed, ToPrimitive};
use std::collections::HashSet;

/// Expand selected additive structures in-place when overlap with the
/// opposing side is detected via structural fingerprints.
///
/// Returns `true` if any term was expanded.
pub fn try_expand_for_cancel_with<F>(
    ctx: &mut Context,
    terms: &mut Vec<(ExprId, bool, OriginSafety)>,
    opposing_fps: &HashSet<u64>,
    mut fallback_expand: F,
) -> bool
where
    F: FnMut(&mut Context, ExprId) -> ExprId,
{
    if opposing_fps.is_empty() {
        return false;
    }

    const MAX_EXP: u32 = 4;
    const MAX_BASE_K: usize = 6;
    const MAX_PRED_TERMS: usize = 35;
    const MAX_BASE_NODES: usize = 25;
    const MAX_MUL_FACTOR_TERMS: usize = 4;
    const MAX_MUL_PRODUCT_TERMS: usize = 16;

    let mut expanded_any = false;
    let mut i = 0;

    // Phase A: Pow(Add|Sub, n) expansion
    while i < terms.len() {
        let (term_id, term_pos, term_safety) = terms[i];

        let pow_info = match ctx.get(term_id) {
            Expr::Pow(base, exp) => {
                let base = *base;
                let exp = *exp;
                let n = match ctx.get(exp) {
                    Expr::Number(num) => {
                        if !num.is_integer() || num.is_negative() {
                            i += 1;
                            continue;
                        }
                        match num.to_integer().to_u32() {
                            Some(v) if (2..=MAX_EXP).contains(&v) => v,
                            _ => {
                                i += 1;
                                continue;
                            }
                        }
                    }
                    _ => {
                        i += 1;
                        continue;
                    }
                };
                let is_additive = matches!(ctx.get(base), Expr::Add(_, _) | Expr::Sub(_, _));
                if !is_additive {
                    i += 1;
                    continue;
                }
                Some((base, exp, n))
            }
            _ => None,
        };

        let (base, exp, n) = match pow_info {
            Some(info) => info,
            None => {
                i += 1;
                continue;
            }
        };

        let mut base_terms_vec = Vec::new();
        collect_additive_terms(ctx, base, true, &mut base_terms_vec);
        let k = base_terms_vec.len();
        if !(2..=MAX_BASE_K).contains(&k) {
            i += 1;
            continue;
        }

        let pred_terms =
            match crate::multinomial_expand::multinomial_term_count(n, k, MAX_PRED_TERMS) {
                Some(t) if t <= MAX_PRED_TERMS => t,
                _ => {
                    i += 1;
                    continue;
                }
            };
        let _ = pred_terms;

        let base_nodes = cas_ast::traversal::count_all_nodes(ctx, base);
        if base_nodes > MAX_BASE_NODES {
            i += 1;
            continue;
        }

        let budget = crate::multinomial_expand::MultinomialExpandBudget {
            max_exp: MAX_EXP,
            max_base_terms: MAX_BASE_K,
            max_vars: MAX_BASE_K,
            max_output_terms: MAX_PRED_TERMS,
        };
        let expanded =
            match crate::multinomial_expand::try_expand_multinomial_direct(ctx, base, exp, &budget)
            {
                Some(e) => match ctx.get(e) {
                    Expr::Hold(inner) => *inner,
                    _ => e,
                },
                None => {
                    let pow_expr = ctx.add(Expr::Pow(base, exp));
                    let expanded = fallback_expand(ctx, pow_expr);
                    if expanded == pow_expr {
                        i += 1;
                        continue;
                    }
                    expanded
                }
            };

        let mut exp_terms = Vec::new();
        collect_additive_terms(ctx, expanded, true, &mut exp_terms);

        let has_overlap = exp_terms.iter().any(|(t, _)| {
            let fp = term_fingerprint(ctx, *t);
            opposing_fps.contains(&fp)
        });

        if !has_overlap {
            i += 1;
            continue;
        }

        terms.remove(i);
        for (et, ep) in exp_terms {
            let final_pos = if term_pos { ep } else { !ep };
            terms.insert(i, (et, final_pos, term_safety));
            i += 1;
        }
        expanded_any = true;
    }

    // Phase B: generalized Mul-factor expansion
    let mut j = 0;
    while j < terms.len() {
        let (term_id, term_pos, term_safety) = terms[j];

        if !matches!(ctx.get(term_id), Expr::Mul(_, _)) {
            j += 1;
            continue;
        }

        let mut factors: Vec<ExprId> = Vec::new();
        {
            let mut stack = vec![term_id];
            let max_depth = 8;
            let max_factors = 7;
            let mut depth = 0;
            while let Some(node) = stack.pop() {
                depth += 1;
                if depth > max_depth || factors.len() >= max_factors {
                    factors.push(node);
                    continue;
                }
                match ctx.get(node) {
                    Expr::Mul(a, b) => {
                        stack.push(*b);
                        stack.push(*a);
                    }
                    _ => {
                        factors.push(node);
                    }
                }
            }
        }

        if factors.len() < 2 {
            j += 1;
            continue;
        }

        let mut add_like_indices: Vec<usize> = Vec::new();
        for (fi, &fid) in factors.iter().enumerate() {
            let is_additive = matches!(ctx.get(fid), Expr::Add(_, _) | Expr::Sub(_, _));
            if is_additive {
                add_like_indices.push(fi);
            }
        }

        if add_like_indices.len() < 2 {
            j += 1;
            continue;
        }

        let mut add_term_lists: Vec<(usize, Vec<(ExprId, bool)>)> = Vec::new();
        let mut bail = false;
        for &fi in &add_like_indices {
            let mut ft = Vec::new();
            collect_additive_terms(ctx, factors[fi], true, &mut ft);
            if ft.len() > MAX_MUL_FACTOR_TERMS {
                bail = true;
                break;
            }
            add_term_lists.push((fi, ft));
        }
        if bail || add_term_lists.len() < 2 {
            j += 1;
            continue;
        }

        add_term_lists.sort_by_key(|(_, terms_list)| terms_list.len());
        let (idx_a, ref a_add_terms) = add_term_lists[0];
        let (idx_b, ref b_add_terms) = add_term_lists[1];

        if a_add_terms.len() * b_add_terms.len() > MAX_MUL_PRODUCT_TERMS {
            j += 1;
            continue;
        }

        let mut distributed: Vec<(ExprId, bool)> = Vec::new();
        for (at, ap) in a_add_terms {
            for (bt, bp) in b_add_terms {
                let prod = mul_preview(ctx, *at, *bt);
                let sign = *ap == *bp;
                distributed.push((prod, sign));
            }
        }

        let scalar_indices: Vec<usize> = (0..factors.len())
            .filter(|i| *i != idx_a && *i != idx_b)
            .collect();

        let distributed_overlap = !scalar_indices.is_empty()
            && distributed.iter().any(|(t, _)| {
                let fp = term_fingerprint(ctx, *t);
                opposing_fps.contains(&fp)
            });

        let wrapped_terms: Vec<(ExprId, bool)> = if scalar_indices.is_empty() {
            distributed
        } else {
            distributed
                .iter()
                .map(|(dt, dp)| {
                    let mut wrapped = *dt;
                    for &si in &scalar_indices {
                        wrapped = ctx.add(Expr::Mul(factors[si], wrapped));
                    }
                    (wrapped, *dp)
                })
                .collect()
        };

        let fp_overlap = distributed_overlap
            || wrapped_terms.iter().any(|(t, _)| {
                let fp = term_fingerprint(ctx, *t);
                opposing_fps.contains(&fp)
            });

        let count_overlap = !fp_overlap
            && !scalar_indices.is_empty()
            && opposing_fps.len() >= a_add_terms.len() * b_add_terms.len();

        if !fp_overlap && !count_overlap {
            j += 1;
            continue;
        }

        terms.remove(j);
        for (wt, wp) in wrapped_terms {
            let final_pos = if term_pos { wp } else { !wp };
            terms.insert(j, (wt, final_pos, term_safety));
            j += 1;
        }
        expanded_any = true;
    }

    expanded_any
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cancel_normalization_support::OriginSafety;
    use cas_parser::parse;

    #[test]
    fn no_op_when_no_opposing_fingerprint() {
        let mut ctx = Context::new();
        let pow = parse("(x+1)^2", &mut ctx).expect("parse");
        let mut terms = vec![(pow, true, OriginSafety::DefinabilityPreserving)];
        let changed =
            try_expand_for_cancel_with(&mut ctx, &mut terms, &HashSet::new(), |_ctx, expr| expr);
        assert!(!changed);
        assert_eq!(terms.len(), 1);
    }
}
