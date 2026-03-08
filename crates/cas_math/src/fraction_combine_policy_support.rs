use crate::expr_rewrite::count_div_nodes;
use crate::trig_roots_flatten::flatten_add_sub_chain;
use cas_ast::{count_nodes, ordering::compare_expr, Context, Expr, ExprId};
use std::cmp::Ordering;
use std::collections::HashMap;

/// Domain-policy decision for combining fractions with a shared denominator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombineSameDenominatorPolicy {
    /// Do not apply rewrite in the current domain policy.
    Block,
    /// Apply rewrite; may need an assumption when nonzero is not proven.
    Apply { assume_denominator_nonzero: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FractionCombineDomainMode {
    Assume,
    Strict,
    Generic,
}

fn fraction_combine_domain_mode_from_flags(
    assume_mode: bool,
    strict_mode: bool,
) -> FractionCombineDomainMode {
    if assume_mode {
        FractionCombineDomainMode::Assume
    } else if strict_mode {
        FractionCombineDomainMode::Strict
    } else {
        FractionCombineDomainMode::Generic
    }
}

/// Decide whether same-denominator fraction combination can be applied.
///
/// Behavior:
/// - Strict: requires proven denominator nonzero.
/// - Assume: applies; when nonzero is unproven, marks assumption requirement.
/// - Generic: applies without additional assumption marker.
pub fn decide_combine_same_denominator_policy(
    assume_mode: bool,
    strict_mode: bool,
    denominator_is_proven_nonzero: bool,
) -> CombineSameDenominatorPolicy {
    let mode = fraction_combine_domain_mode_from_flags(assume_mode, strict_mode);
    match mode {
        FractionCombineDomainMode::Strict => {
            if denominator_is_proven_nonzero {
                CombineSameDenominatorPolicy::Apply {
                    assume_denominator_nonzero: false,
                }
            } else {
                CombineSameDenominatorPolicy::Block
            }
        }
        FractionCombineDomainMode::Assume => CombineSameDenominatorPolicy::Apply {
            assume_denominator_nonzero: !denominator_is_proven_nonzero,
        },
        FractionCombineDomainMode::Generic => CombineSameDenominatorPolicy::Apply {
            assume_denominator_nonzero: false,
        },
    }
}

/// Fraction term extracted from an additive chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FractionTermWithSign {
    pub numerator: ExprId,
    pub denominator: ExprId,
    pub is_negated: bool,
}

pub type FractionGroupEntry = (usize, ExprId, bool);
pub type FractionDenGroups = HashMap<ExprId, Vec<FractionGroupEntry>>;

#[derive(Debug, Clone)]
pub struct SameDenominatorPreparation {
    pub terms: Vec<ExprId>,
    pub denom_groups: FractionDenGroups,
    pub non_fraction_indices: Vec<usize>,
    pub common_den: ExprId,
    pub group: Vec<FractionGroupEntry>,
}

#[derive(Debug, Clone, Copy)]
pub struct SameDenominatorCombinationBuild {
    pub result: ExprId,
    pub focus_before: ExprId,
    pub focus_after: ExprId,
}

#[derive(Debug, Clone, Copy)]
pub struct SameDenominatorCombinationPlan {
    pub build: SameDenominatorCombinationBuild,
    pub assume_denominator_nonzero: bool,
}

/// Extract fraction-like term information from:
/// - `a / d`
/// - `-(a / d)`
pub fn extract_fraction_term_with_sign(
    ctx: &Context,
    term: ExprId,
) -> Option<FractionTermWithSign> {
    match ctx.get(term) {
        Expr::Div(num, den) => Some(FractionTermWithSign {
            numerator: *num,
            denominator: *den,
            is_negated: false,
        }),
        Expr::Neg(inner) => {
            if let Expr::Div(num, den) = ctx.get(*inner) {
                Some(FractionTermWithSign {
                    numerator: *num,
                    denominator: *den,
                    is_negated: true,
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Group additive terms by structurally equal denominators.
///
/// Returns:
/// - `FractionDenGroups`: denominator -> list of `(term_index, numerator, is_negated)`
/// - `Vec<usize>`: indices of non-fraction terms.
pub fn group_fraction_terms_by_denominator(
    ctx: &Context,
    terms: &[ExprId],
) -> (FractionDenGroups, Vec<usize>) {
    let mut denom_groups: FractionDenGroups = HashMap::new();
    let mut non_fraction_indices: Vec<usize> = Vec::new();

    for (idx, &term) in terms.iter().enumerate() {
        if let Some(fr) = extract_fraction_term_with_sign(ctx, term) {
            let mut found_key = None;
            for existing_den in denom_groups.keys() {
                if compare_expr(ctx, *existing_den, fr.denominator) == Ordering::Equal {
                    found_key = Some(*existing_den);
                    break;
                }
            }

            if let Some(key) = found_key {
                if let Some(v) = denom_groups.get_mut(&key) {
                    v.push((idx, fr.numerator, fr.is_negated));
                } else {
                    // Should be unreachable because `key` comes from map keys.
                    denom_groups.insert(key, vec![(idx, fr.numerator, fr.is_negated)]);
                }
            } else {
                denom_groups.insert(fr.denominator, vec![(idx, fr.numerator, fr.is_negated)]);
            }
        } else {
            non_fraction_indices.push(idx);
        }
    }

    (denom_groups, non_fraction_indices)
}

/// Prepare all structural inputs needed to combine one same-denominator group.
pub fn prepare_same_denominator_combination(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<SameDenominatorPreparation> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let terms = flatten_add_sub_chain(ctx, expr);
    if terms.len() < 2 {
        return None;
    }

    let (denom_groups, non_fraction_indices) = group_fraction_terms_by_denominator(ctx, &terms);
    let (common_den, group) = first_combinable_denominator_group(&denom_groups)?;

    Some(SameDenominatorPreparation {
        terms,
        denom_groups,
        non_fraction_indices,
        common_den,
        group,
    })
}

/// Return the first denominator group that has at least two fraction terms.
pub fn first_combinable_denominator_group(
    denom_groups: &FractionDenGroups,
) -> Option<(ExprId, Vec<FractionGroupEntry>)> {
    for (den, group) in denom_groups.iter() {
        if group.len() >= 2 {
            return Some((*den, group.clone()));
        }
    }
    None
}

/// Build rewrite payload for combining one same-denominator fraction group.
///
/// Returns `None` when the rebuilt expression is a structural/no-op downgrade
/// (no reduced division count and no node-count win).
pub fn build_same_denominator_combination(
    ctx: &mut Context,
    original_expr: ExprId,
    terms: &[ExprId],
    denom_groups: &FractionDenGroups,
    non_fraction_indices: &[usize],
    common_den: ExprId,
    group: &[FractionGroupEntry],
) -> Option<SameDenominatorCombinationBuild> {
    let combined_num_terms: Vec<ExprId> = group
        .iter()
        .map(|(_, num, is_neg)| {
            if *is_neg {
                ctx.add(Expr::Neg(*num))
            } else {
                *num
            }
        })
        .collect();

    let combined_num = if combined_num_terms.len() == 1 {
        combined_num_terms[0]
    } else {
        let mut acc = combined_num_terms[0];
        for term in &combined_num_terms[1..] {
            acc = ctx.add(Expr::Add(acc, *term));
        }
        acc
    };
    let combined_fraction = ctx.add(Expr::Div(combined_num, common_den));

    let mut new_terms: Vec<ExprId> = Vec::new();
    for &idx in non_fraction_indices {
        new_terms.push(terms[idx]);
    }
    for (den, single_group) in denom_groups.iter() {
        if single_group.len() == 1 && compare_expr(ctx, *den, common_den) != Ordering::Equal {
            let (idx, _, _) = single_group[0];
            new_terms.push(terms[idx]);
        }
    }
    new_terms.push(combined_fraction);

    if new_terms.is_empty() {
        return None;
    }
    let result = if new_terms.len() == 1 {
        new_terms[0]
    } else {
        let mut acc = new_terms[0];
        for term in &new_terms[1..] {
            acc = ctx.add(Expr::Add(acc, *term));
        }
        acc
    };

    if count_nodes(ctx, result) >= count_nodes(ctx, original_expr) {
        let old_divs = count_div_nodes(ctx, original_expr);
        let new_divs = count_div_nodes(ctx, result);
        if new_divs >= old_divs {
            return None;
        }
    }

    let original_fractions: Vec<ExprId> = group.iter().map(|&(idx, _, _)| terms[idx]).collect();
    let focus_before = if original_fractions.len() == 1 {
        original_fractions[0]
    } else {
        let mut acc = original_fractions[0];
        for &term in &original_fractions[1..] {
            acc = ctx.add(Expr::Add(acc, term));
        }
        acc
    };

    Some(SameDenominatorCombinationBuild {
        result,
        focus_before,
        focus_after: combined_fraction,
    })
}

/// Try to fully plan a same-denominator combination rewrite.
///
/// This performs:
/// 1) structural preparation of a combinable denominator group,
/// 2) domain-policy gating (strict/assume/generic),
/// 3) construction of rewrite result and focus payload.
pub fn try_plan_same_denominator_combination_with<F>(
    ctx: &mut Context,
    expr: ExprId,
    assume_mode: bool,
    strict_mode: bool,
    mut denominator_is_proven_nonzero: F,
) -> Option<SameDenominatorCombinationPlan>
where
    F: FnMut(&mut Context, ExprId) -> bool,
{
    let prep = prepare_same_denominator_combination(ctx, expr)?;
    let policy = decide_combine_same_denominator_policy(
        assume_mode,
        strict_mode,
        denominator_is_proven_nonzero(ctx, prep.common_den),
    );
    let assume_denominator_nonzero = match policy {
        CombineSameDenominatorPolicy::Block => return None,
        CombineSameDenominatorPolicy::Apply {
            assume_denominator_nonzero,
        } => assume_denominator_nonzero,
    };

    let build = build_same_denominator_combination(
        ctx,
        expr,
        &prep.terms,
        &prep.denom_groups,
        &prep.non_fraction_indices,
        prep.common_den,
        &prep.group,
    )?;

    Some(SameDenominatorCombinationPlan {
        build,
        assume_denominator_nonzero,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        build_same_denominator_combination, decide_combine_same_denominator_policy,
        extract_fraction_term_with_sign, first_combinable_denominator_group,
        group_fraction_terms_by_denominator, prepare_same_denominator_combination,
        try_plan_same_denominator_combination_with, CombineSameDenominatorPolicy,
    };
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn strict_blocks_when_denominator_nonzero_is_unproven() {
        let out = decide_combine_same_denominator_policy(false, true, false);
        assert_eq!(out, CombineSameDenominatorPolicy::Block);
    }

    #[test]
    fn strict_allows_when_denominator_nonzero_is_proven() {
        let out = decide_combine_same_denominator_policy(false, true, true);
        assert_eq!(
            out,
            CombineSameDenominatorPolicy::Apply {
                assume_denominator_nonzero: false,
            }
        );
    }

    #[test]
    fn assume_allows_and_marks_assumption_when_unproven() {
        let out = decide_combine_same_denominator_policy(true, false, false);
        assert_eq!(
            out,
            CombineSameDenominatorPolicy::Apply {
                assume_denominator_nonzero: true,
            }
        );
    }

    #[test]
    fn generic_allows_without_assumption() {
        let out = decide_combine_same_denominator_policy(false, false, false);
        assert_eq!(
            out,
            CombineSameDenominatorPolicy::Apply {
                assume_denominator_nonzero: false,
            }
        );
    }

    #[test]
    fn assume_priority_over_strict() {
        let out = decide_combine_same_denominator_policy(true, true, false);
        assert_eq!(
            out,
            CombineSameDenominatorPolicy::Apply {
                assume_denominator_nonzero: true,
            }
        );
    }

    #[test]
    fn extracts_fraction_term_and_negated_fraction_term() {
        let mut ctx = Context::new();
        let pos = parse("a/d", &mut ctx).expect("parse");
        let neg = parse("-(b/d)", &mut ctx).expect("parse");

        let p = extract_fraction_term_with_sign(&ctx, pos).expect("pos");
        assert!(!p.is_negated);

        let n = extract_fraction_term_with_sign(&ctx, neg).expect("neg");
        assert!(n.is_negated);
    }

    #[test]
    fn groups_terms_by_structural_denominator() {
        let mut ctx = Context::new();
        let t1 = parse("a/d", &mut ctx).expect("parse");
        let t2 = parse("-(b/d)", &mut ctx).expect("parse");
        let t3 = parse("c/e", &mut ctx).expect("parse");
        let t4 = parse("x", &mut ctx).expect("parse");
        let terms = vec![t1, t2, t3, t4];

        let (groups, non_fracs) = group_fraction_terms_by_denominator(&ctx, &terms);
        assert_eq!(non_fracs, vec![3]);
        assert_eq!(groups.len(), 2);
        assert!(groups.values().any(|v| v.len() == 2));
        assert!(groups.values().any(|v| v.len() == 1));
    }

    #[test]
    fn first_combinable_group_finds_multi_fraction_denominator() {
        let mut ctx = Context::new();
        let terms = vec![
            parse("a/d", &mut ctx).expect("parse"),
            parse("-(b/d)", &mut ctx).expect("parse"),
            parse("x", &mut ctx).expect("parse"),
        ];
        let (groups, _) = group_fraction_terms_by_denominator(&ctx, &terms);
        let got = first_combinable_denominator_group(&groups);
        assert!(got.is_some());
    }

    #[test]
    fn build_same_denominator_combination_produces_focus_payload() {
        let mut ctx = Context::new();
        let expr = parse("1 + a/d - b/d", &mut ctx).expect("parse");
        let terms = vec![
            parse("1", &mut ctx).expect("parse"),
            parse("a/d", &mut ctx).expect("parse"),
            parse("-(b/d)", &mut ctx).expect("parse"),
        ];
        let (groups, non_fracs) = group_fraction_terms_by_denominator(&ctx, &terms);
        let (common_den, group) =
            first_combinable_denominator_group(&groups).expect("combinable group");
        let built = build_same_denominator_combination(
            &mut ctx, expr, &terms, &groups, &non_fracs, common_den, &group,
        )
        .expect("build");
        assert_ne!(built.result, expr);
        assert_ne!(built.focus_before, built.focus_after);
    }

    #[test]
    fn prepare_same_denominator_combination_extracts_group() {
        let mut ctx = Context::new();
        let expr = parse("1 + a/d - b/d", &mut ctx).expect("parse");
        let prep = prepare_same_denominator_combination(&mut ctx, expr).expect("prep");
        assert!(prep.group.len() >= 2);
    }

    #[test]
    fn prepare_same_denominator_combination_rejects_non_add_sub() {
        let mut ctx = Context::new();
        let expr = parse("a/d", &mut ctx).expect("parse");
        assert!(prepare_same_denominator_combination(&mut ctx, expr).is_none());
    }

    #[test]
    fn plan_same_denominator_combination_blocks_in_strict_when_unproven() {
        let mut ctx = Context::new();
        let expr = parse("1 + a/d - b/d", &mut ctx).expect("parse");
        let plan =
            try_plan_same_denominator_combination_with(&mut ctx, expr, false, true, |_c, _den| {
                false
            });
        assert!(plan.is_none());
    }

    #[test]
    fn plan_same_denominator_combination_marks_assumption_in_assume_mode() {
        let mut ctx = Context::new();
        let expr = parse("1 + a/d - b/d", &mut ctx).expect("parse");
        let plan =
            try_plan_same_denominator_combination_with(&mut ctx, expr, true, false, |_c, _den| {
                false
            })
            .expect("plan");
        assert!(plan.assume_denominator_nonzero);
        assert_ne!(plan.build.result, expr);
    }
}
