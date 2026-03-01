use crate::expr_nary::build_balanced_add;
use cas_ast::{Context, Expr, ExprId};
use num_traits::One;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InverseTrigCompositionKind {
    SinArcsin,
    CosArccos,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InverseTrigCompositionPlan {
    pub desc: &'static str,
    pub assume_defined: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InverseTrigCompositionMode {
    Assume,
    Strict,
    Generic,
}

fn inverse_trig_composition_mode_from_flags(
    assume_mode: bool,
    strict_mode: bool,
) -> InverseTrigCompositionMode {
    if assume_mode {
        InverseTrigCompositionMode::Assume
    } else if strict_mode {
        InverseTrigCompositionMode::Strict
    } else {
        InverseTrigCompositionMode::Generic
    }
}

fn default_desc(kind: InverseTrigCompositionKind) -> &'static str {
    match kind {
        InverseTrigCompositionKind::SinArcsin => "sin(arcsin(x)) = x",
        InverseTrigCompositionKind::CosArccos => "cos(arccos(x)) = x",
    }
}

fn assume_desc(kind: InverseTrigCompositionKind) -> &'static str {
    match kind {
        InverseTrigCompositionKind::SinArcsin => "sin(arcsin(x)) = x (assuming x ∈ [-1, 1])",
        InverseTrigCompositionKind::CosArccos => "cos(arccos(x)) = x (assuming x ∈ [-1, 1])",
    }
}

/// Domain-policy planner for `sin(arcsin(x))` and `cos(arccos(x))`.
///
/// `arg_in_unit_interval_proven` should be true only when strict mode can prove
/// the inverse-function input lies in `[-1, 1]`.
pub fn plan_inverse_trig_composition_with_mode_flags(
    kind: InverseTrigCompositionKind,
    arg_in_unit_interval_proven: bool,
    assume_mode: bool,
    strict_mode: bool,
) -> Option<InverseTrigCompositionPlan> {
    let mode = inverse_trig_composition_mode_from_flags(assume_mode, strict_mode);
    match mode {
        InverseTrigCompositionMode::Strict => {
            if arg_in_unit_interval_proven {
                Some(InverseTrigCompositionPlan {
                    desc: default_desc(kind),
                    assume_defined: false,
                })
            } else {
                None
            }
        }
        InverseTrigCompositionMode::Generic => Some(InverseTrigCompositionPlan {
            desc: default_desc(kind),
            assume_defined: false,
        }),
        InverseTrigCompositionMode::Assume => Some(InverseTrigCompositionPlan {
            desc: assume_desc(kind),
            assume_defined: true,
        }),
    }
}

/// Strict-mode helper: checks whether `expr` is a numeric literal in `[-1, 1]`.
pub fn is_number_in_unit_interval(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        let one = num_rational::BigRational::one();
        let neg_one = -one.clone();
        *n >= neg_one && *n <= one
    } else {
        false
    }
}

/// Build sum of all terms except indices `skip_i` and `skip_j`.
///
/// Returns `None` when no terms remain.
pub fn build_sum_without(
    ctx: &mut Context,
    terms: &[ExprId],
    skip_i: usize,
    skip_j: usize,
) -> Option<ExprId> {
    let remaining: Vec<ExprId> = terms
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx != skip_i && *idx != skip_j)
        .map(|(_, &term)| term)
        .collect();

    match remaining.len() {
        0 => None,
        _ => Some(build_balanced_add(ctx, &remaining)),
    }
}

/// Combine optional additive base with a new term.
pub fn combine_with_term(ctx: &mut Context, base: Option<ExprId>, new_term: ExprId) -> ExprId {
    match base {
        None => new_term,
        Some(b) => ctx.add(Expr::Add(b, new_term)),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_sum_without, combine_with_term, is_number_in_unit_interval,
        plan_inverse_trig_composition_with_mode_flags, InverseTrigCompositionKind,
        InverseTrigCompositionPlan,
    };
    use cas_ast::{Context, Expr};
    use cas_parser::parse;

    #[test]
    fn strict_requires_interval_proof() {
        let out = plan_inverse_trig_composition_with_mode_flags(
            InverseTrigCompositionKind::SinArcsin,
            false,
            false,
            true,
        );
        assert_eq!(out, None);
    }

    #[test]
    fn strict_accepts_proven_interval() {
        let out = plan_inverse_trig_composition_with_mode_flags(
            InverseTrigCompositionKind::CosArccos,
            true,
            false,
            true,
        );
        assert_eq!(
            out,
            Some(InverseTrigCompositionPlan {
                desc: "cos(arccos(x)) = x",
                assume_defined: false,
            })
        );
    }

    #[test]
    fn assume_always_applies_with_assumption() {
        let out = plan_inverse_trig_composition_with_mode_flags(
            InverseTrigCompositionKind::SinArcsin,
            false,
            true,
            false,
        );
        assert_eq!(
            out,
            Some(InverseTrigCompositionPlan {
                desc: "sin(arcsin(x)) = x (assuming x ∈ [-1, 1])",
                assume_defined: true,
            })
        );
    }

    #[test]
    fn generic_applies_without_assumption() {
        let out = plan_inverse_trig_composition_with_mode_flags(
            InverseTrigCompositionKind::SinArcsin,
            false,
            false,
            false,
        );
        assert_eq!(
            out,
            Some(InverseTrigCompositionPlan {
                desc: "sin(arcsin(x)) = x",
                assume_defined: false,
            })
        );
    }

    #[test]
    fn assume_priority_over_strict() {
        let out = plan_inverse_trig_composition_with_mode_flags(
            InverseTrigCompositionKind::CosArccos,
            false,
            true,
            true,
        );
        assert_eq!(
            out,
            Some(InverseTrigCompositionPlan {
                desc: "cos(arccos(x)) = x (assuming x ∈ [-1, 1])",
                assume_defined: true,
            })
        );
    }

    #[test]
    fn unit_interval_detection_accepts_and_rejects_expected_values() {
        let mut ctx = Context::new();
        let half = ctx.add(Expr::Number(num_rational::BigRational::new(
            1.into(),
            2.into(),
        )));
        let minus_two = ctx.num(-2);
        let x = ctx.var("x");

        assert!(is_number_in_unit_interval(&ctx, half));
        assert!(!is_number_in_unit_interval(&ctx, minus_two));
        assert!(!is_number_in_unit_interval(&ctx, x));

        // Explicit bound checks
        let minus_one = ctx.num(-1);
        let plus_one = ctx.num(1);
        let three = ctx.num(3);
        let two = ctx.num(2);
        let over_one = ctx.add(Expr::Div(three, two));
        assert!(is_number_in_unit_interval(&ctx, minus_one));
        assert!(is_number_in_unit_interval(&ctx, plus_one));
        assert!(!is_number_in_unit_interval(&ctx, over_one));
    }

    #[test]
    fn sum_helpers_build_remaining_sum_and_append_term() {
        let mut ctx = Context::new();
        let a = parse("a", &mut ctx).expect("a");
        let b = parse("b", &mut ctx).expect("b");
        let c = parse("c", &mut ctx).expect("c");
        let d = parse("d", &mut ctx).expect("d");
        let terms = vec![a, b, c, d];

        let remaining = build_sum_without(&mut ctx, &terms, 1, 3).expect("remaining");
        let appended = combine_with_term(&mut ctx, Some(remaining), b);

        let expected_remaining = parse("a+c", &mut ctx).expect("a+c");
        let expected_appended = parse("(a+c)+b", &mut ctx).expect("(a+c)+b");

        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, remaining, expected_remaining),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            cas_ast::ordering::compare_expr(&ctx, appended, expected_appended),
            std::cmp::Ordering::Equal
        );
        assert_eq!(combine_with_term(&mut ctx, None, c), c);
    }
}
