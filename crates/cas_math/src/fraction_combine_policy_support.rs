use cas_ast::{ordering::compare_expr, Context, Expr, ExprId};
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

#[cfg(test)]
mod tests {
    use super::{
        decide_combine_same_denominator_policy, extract_fraction_term_with_sign,
        group_fraction_terms_by_denominator, CombineSameDenominatorPolicy,
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
}
