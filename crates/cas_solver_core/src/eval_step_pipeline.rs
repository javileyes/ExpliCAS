//! Generic eval-step cleanup pipeline shared across crates.

use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_predicates::{is_one_expr, is_zero_expr};

/// Clean raw eval steps for display:
/// 1) Remove no-op steps (no global or focused local change),
/// 2) Repair global-before/global-after chain coherence.
#[allow(clippy::too_many_arguments)]
pub fn clean_eval_steps<
    T,
    FBefore,
    FAfter,
    FBeforeLocal,
    FAfterLocal,
    FGlobalAfter,
    FSetGlobalBefore,
>(
    raw_steps: Vec<T>,
    mut before_of: FBefore,
    mut after_of: FAfter,
    mut before_local_of: FBeforeLocal,
    mut after_local_of: FAfterLocal,
    mut global_after_of: FGlobalAfter,
    mut set_global_before: FSetGlobalBefore,
) -> Vec<T>
where
    FBefore: FnMut(&T) -> ExprId,
    FAfter: FnMut(&T) -> ExprId,
    FBeforeLocal: FnMut(&T) -> Option<ExprId>,
    FAfterLocal: FnMut(&T) -> Option<ExprId>,
    FGlobalAfter: FnMut(&T) -> Option<ExprId>,
    FSetGlobalBefore: FnMut(&mut T, ExprId),
{
    let mut cleaned: Vec<T> = raw_steps
        .into_iter()
        .filter(|step| {
            let global_changed = before_of(step) != after_of(step);
            let local_changed = match (before_local_of(step), after_local_of(step)) {
                (Some(bl), Some(al)) => bl != al,
                _ => false,
            };
            global_changed || local_changed
        })
        .collect();

    for i in 0..cleaned.len().saturating_sub(1) {
        if let Some(after_i) = global_after_of(&cleaned[i]) {
            set_global_before(&mut cleaned[i + 1], after_i);
        }
    }

    cleaned
}

/// Canonical conversion from raw eval steps to display-ready cleaned steps.
pub fn to_display_eval_steps(
    raw_steps: Vec<crate::step_model::Step>,
) -> crate::display_steps::DisplaySteps<crate::step_model::Step> {
    let cleaned = clean_eval_steps(
        raw_steps,
        |s: &crate::step_model::Step| s.before,
        |s: &crate::step_model::Step| s.after,
        |s: &crate::step_model::Step| s.before_local(),
        |s: &crate::step_model::Step| s.after_local(),
        |s: &crate::step_model::Step| s.global_after,
        |s: &mut crate::step_model::Step, gb| s.global_before = Some(gb),
    );
    let cleaned = remove_redundant_post_calculus_rationalization(cleaned);
    crate::display_steps::DisplaySteps(cleaned)
}

fn remove_redundant_post_calculus_rationalization(
    steps: Vec<crate::step_model::Step>,
) -> Vec<crate::step_model::Step> {
    let mut filtered = Vec::with_capacity(steps.len());
    let mut index = 0;

    while index < steps.len() {
        if let Some(next_index) = redundant_post_calculus_rationalization_end_index(&steps, index) {
            index = next_index + 1;
            continue;
        }
        if redundant_pre_presentation_cleanup_after_diff(&steps, index) {
            index += 1;
            continue;
        }
        filtered.push(steps[index].clone());
        index += 1;
    }

    repair_step_chain(&mut filtered);
    filtered
}

fn redundant_post_calculus_rationalization_end_index(
    steps: &[crate::step_model::Step],
    start: usize,
) -> Option<usize> {
    if start == 0 || steps[start - 1].rule_name.as_str() != "Symbolic Differentiation" {
        return None;
    }

    let rationalize = steps.get(start)?;
    if rationalize.rule_name.as_str() != "Rationalize Product Denominator" {
        return None;
    }

    let mut presentation_index = start + 1;
    while steps
        .get(presentation_index)
        .is_some_and(is_post_calculus_presentation_noise_step)
    {
        presentation_index += 1;
    }

    let presentation = steps.get(presentation_index)?;
    if presentation.rule_name.as_str() == "Present calculus result in compact form" {
        Some(presentation_index)
    } else {
        None
    }
}

fn is_post_calculus_presentation_noise_step(step: &crate::step_model::Step) -> bool {
    matches!(
        step.rule_name.as_str(),
        "Combine Constants"
            | "Distributive Property"
            | "Expand"
            | "Identity Property of Multiplication"
            | "N-ary Mul Combine Powers"
            | "Normalize Negation in Product"
            | "Pull Constant From Fraction"
            | "Rationalize Product Denominator"
            | "Simplify Multiplication with Division"
    )
}

fn redundant_pre_presentation_cleanup_after_diff(
    steps: &[crate::step_model::Step],
    index: usize,
) -> bool {
    if !steps
        .get(index)
        .is_some_and(is_post_calculus_presentation_noise_step)
    {
        return false;
    }

    let mut next_index = index + 1;
    while steps
        .get(next_index)
        .is_some_and(is_post_calculus_presentation_noise_step)
    {
        next_index += 1;
    }

    steps
        .get(next_index)
        .is_some_and(|step| step.rule_name.as_str() == "Present calculus result in compact form")
        && has_prior_symbolic_differentiation_anchor(steps, index)
}

fn has_prior_symbolic_differentiation_anchor(
    steps: &[crate::step_model::Step],
    before_index: usize,
) -> bool {
    steps[..before_index]
        .iter()
        .rev()
        .take_while(|step| step.rule_name.as_str() != "Present calculus result in compact form")
        .any(|step| step.rule_name.as_str() == "Symbolic Differentiation")
}

fn repair_step_chain(steps: &mut [crate::step_model::Step]) {
    for i in 0..steps.len().saturating_sub(1) {
        if let Some(after_i) = steps[i].global_after {
            steps[i + 1].global_before = Some(after_i);
        }
    }
}

pub fn normalize_expr_for_display(ctx: &mut Context, expr: ExprId) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => {
            let lhs = normalize_expr_for_display(ctx, lhs);
            let rhs = normalize_expr_for_display(ctx, rhs);
            if is_zero_expr(ctx, lhs) {
                rhs
            } else if is_zero_expr(ctx, rhs) {
                lhs
            } else {
                ctx.add(Expr::Add(lhs, rhs))
            }
        }
        Expr::Sub(lhs, rhs) => {
            let lhs = normalize_expr_for_display(ctx, lhs);
            let rhs = normalize_expr_for_display(ctx, rhs);
            if is_zero_expr(ctx, rhs) {
                lhs
            } else {
                ctx.add(Expr::Sub(lhs, rhs))
            }
        }
        Expr::Mul(lhs, rhs) => {
            let lhs = normalize_expr_for_display(ctx, lhs);
            let rhs = normalize_expr_for_display(ctx, rhs);
            if is_zero_expr(ctx, lhs) || is_zero_expr(ctx, rhs) {
                ctx.num(0)
            } else if is_one_expr(ctx, lhs) {
                rhs
            } else if is_one_expr(ctx, rhs) {
                lhs
            } else {
                ctx.add(Expr::Mul(lhs, rhs))
            }
        }
        Expr::Div(lhs, rhs) => {
            let lhs = normalize_expr_for_display(ctx, lhs);
            let rhs = normalize_expr_for_display(ctx, rhs);
            if is_one_expr(ctx, rhs) {
                lhs
            } else {
                ctx.add(Expr::Div(lhs, rhs))
            }
        }
        Expr::Pow(base, exp) => {
            let base = normalize_expr_for_display(ctx, base);
            let exp = normalize_expr_for_display(ctx, exp);
            ctx.add(Expr::Pow(base, exp))
        }
        Expr::Neg(inner) => {
            let inner = normalize_expr_for_display(ctx, inner);
            if is_zero_expr(ctx, inner) {
                inner
            } else {
                ctx.add(Expr::Neg(inner))
            }
        }
        Expr::Function(fn_id, args) => {
            let args = args
                .into_iter()
                .map(|arg| normalize_expr_for_display(ctx, arg))
                .collect();
            ctx.add(Expr::Function(fn_id, args))
        }
        Expr::Matrix { rows, cols, data } => {
            let data = data
                .into_iter()
                .map(|item| normalize_expr_for_display(ctx, item))
                .collect();
            ctx.add(Expr::Matrix { rows, cols, data })
        }
        Expr::Hold(inner) => {
            let inner = normalize_expr_for_display(ctx, inner);
            ctx.add(Expr::Hold(inner))
        }
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => expr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{Context, Expr};

    #[derive(Clone, Debug)]
    struct TestStep {
        before: ExprId,
        after: ExprId,
        before_local: Option<ExprId>,
        after_local: Option<ExprId>,
        global_before: Option<ExprId>,
        global_after: Option<ExprId>,
    }

    #[test]
    fn removes_global_noops_without_local_change() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(2);

        let raw = vec![
            TestStep {
                before: a,
                after: a,
                before_local: None,
                after_local: None,
                global_before: None,
                global_after: Some(a),
            },
            TestStep {
                before: a,
                after: b,
                before_local: None,
                after_local: None,
                global_before: None,
                global_after: Some(b),
            },
        ];

        let cleaned = clean_eval_steps(
            raw,
            |s| s.before,
            |s| s.after,
            |s| s.before_local,
            |s| s.after_local,
            |s| s.global_after,
            |s, gb| s.global_before = Some(gb),
        );

        assert_eq!(cleaned.len(), 1);
        assert_eq!(cleaned[0].before, a);
        assert_eq!(cleaned[0].after, b);
    }

    #[test]
    fn preserves_local_changes_when_global_is_equal() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(2);

        let raw = vec![TestStep {
            before: a,
            after: a,
            before_local: Some(a),
            after_local: Some(b),
            global_before: None,
            global_after: Some(a),
        }];

        let cleaned = clean_eval_steps(
            raw,
            |s| s.before,
            |s| s.after,
            |s| s.before_local,
            |s| s.after_local,
            |s| s.global_after,
            |s, gb| s.global_before = Some(gb),
        );

        assert_eq!(cleaned.len(), 1);
    }

    #[test]
    fn repairs_global_before_chain_from_previous_after() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(2);
        let c = ctx.num(3);

        let raw = vec![
            TestStep {
                before: a,
                after: b,
                before_local: None,
                after_local: None,
                global_before: None,
                global_after: Some(c),
            },
            TestStep {
                before: b,
                after: c,
                before_local: None,
                after_local: None,
                global_before: None,
                global_after: Some(b),
            },
        ];

        let cleaned = clean_eval_steps(
            raw,
            |s| s.before,
            |s| s.after,
            |s| s.before_local,
            |s| s.after_local,
            |s| s.global_after,
            |s, gb| s.global_before = Some(gb),
        );

        assert_eq!(cleaned.len(), 2);
        assert_eq!(cleaned[1].global_before, Some(c));
    }

    #[test]
    fn to_display_eval_steps_removes_noop() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let step = crate::step_model::Step::new_compact("test", "rule", one, two);
        let out = super::to_display_eval_steps(vec![step]);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn to_display_eval_steps_removes_redundant_diff_rationalize_presentation_round_trip() {
        let mut ctx = Context::new();
        let source = ctx.num(1);
        let compact = ctx.num(2);
        let rationalized = ctx.num(3);
        let identity_cleaned = ctx.num(4);
        let constant_pulled = ctx.num(5);
        let negation_normalized = ctx.num(6);
        let constants_combined = ctx.num(7);
        let powers_combined = ctx.num(8);
        let follow_after = ctx.num(9);

        let mut diff = crate::step_model::Step::new_compact(
            "diff",
            "Symbolic Differentiation",
            source,
            compact,
        );
        diff.global_after = Some(compact);
        let mut rationalize = crate::step_model::Step::new_compact(
            "rationalize",
            "Rationalize Product Denominator",
            compact,
            rationalized,
        );
        rationalize.global_after = Some(rationalized);
        let mut identity = crate::step_model::Step::new_compact(
            "identity",
            "Identity Property of Multiplication",
            rationalized,
            identity_cleaned,
        );
        identity.global_after = Some(identity_cleaned);
        let mut pull_constant = crate::step_model::Step::new_compact(
            "pull constant",
            "Pull Constant From Fraction",
            identity_cleaned,
            constant_pulled,
        );
        pull_constant.global_after = Some(constant_pulled);
        let mut normalize_negation = crate::step_model::Step::new_compact(
            "normalize negation",
            "Normalize Negation in Product",
            constant_pulled,
            negation_normalized,
        );
        normalize_negation.global_after = Some(negation_normalized);
        let mut combine_constants = crate::step_model::Step::new_compact(
            "combine constants",
            "Combine Constants",
            negation_normalized,
            constants_combined,
        );
        combine_constants.global_after = Some(constants_combined);
        let mut combine_powers = crate::step_model::Step::new_compact(
            "combine powers",
            "N-ary Mul Combine Powers",
            constants_combined,
            powers_combined,
        );
        combine_powers.global_after = Some(powers_combined);
        let mut present = crate::step_model::Step::new_compact(
            "present",
            "Present calculus result in compact form",
            powers_combined,
            compact,
        );
        present.global_after = Some(compact);
        let follow =
            crate::step_model::Step::new_compact("follow", "Follow-up", compact, follow_after);

        let out = super::to_display_eval_steps(vec![
            diff,
            rationalize,
            identity,
            pull_constant,
            normalize_negation,
            combine_constants,
            combine_powers,
            present,
            follow,
        ]);
        let rules = out
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(rules, vec!["Symbolic Differentiation", "Follow-up"]);
        assert_eq!(out[1].global_before, Some(compact));
    }

    #[test]
    fn to_display_eval_steps_keeps_diff_presentation_while_removing_prior_cleanup() {
        let mut ctx = Context::new();
        let source = ctx.num(1);
        let diffed = ctx.num(2);
        let pulled = ctx.num(3);
        let combined = ctx.num(4);
        let presented = ctx.num(5);

        let diff = crate::step_model::Step::new_compact(
            "diff",
            "Symbolic Differentiation",
            source,
            diffed,
        );
        let pull = crate::step_model::Step::new_compact(
            "pull constant",
            "Pull Constant From Fraction",
            diffed,
            pulled,
        );
        let combine = crate::step_model::Step::new_compact(
            "combine",
            "Simplify Multiplication with Division",
            pulled,
            combined,
        );
        let present = crate::step_model::Step::new_compact(
            "present",
            "Present calculus result in compact form",
            combined,
            presented,
        );

        let out = super::to_display_eval_steps(vec![diff, pull, combine, present]);
        let rules = out
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            rules,
            vec![
                "Symbolic Differentiation",
                "Present calculus result in compact form"
            ]
        );
    }

    #[test]
    fn to_display_eval_steps_keeps_meaningful_post_diff_step_before_presentation_cleanup() {
        let mut ctx = Context::new();
        let source = ctx.num(1);
        let diffed = ctx.num(2);
        let root_simplified = ctx.num(3);
        let pulled = ctx.num(4);
        let combined = ctx.num(5);
        let presented = ctx.num(6);

        let diff = crate::step_model::Step::new_compact(
            "diff",
            "Symbolic Differentiation",
            source,
            diffed,
        );
        let root = crate::step_model::Step::new_compact(
            "root",
            "Undo Root And Power",
            diffed,
            root_simplified,
        );
        let pull = crate::step_model::Step::new_compact(
            "pull constant",
            "Pull Constant From Fraction",
            root_simplified,
            pulled,
        );
        let combine = crate::step_model::Step::new_compact(
            "combine",
            "Simplify Multiplication with Division",
            pulled,
            combined,
        );
        let present = crate::step_model::Step::new_compact(
            "present",
            "Present calculus result in compact form",
            combined,
            presented,
        );

        let out = super::to_display_eval_steps(vec![diff, root, pull, combine, present]);
        let rules = out
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            rules,
            vec![
                "Symbolic Differentiation",
                "Undo Root And Power",
                "Present calculus result in compact form"
            ]
        );
    }

    #[test]
    fn to_display_eval_steps_removes_post_diff_expand_round_trip_before_presentation() {
        let mut ctx = Context::new();
        let source = ctx.num(1);
        let arccot_rewritten = ctx.num(2);
        let diffed = ctx.num(3);
        let expanded_once = ctx.num(4);
        let expanded_twice = ctx.num(5);
        let presented = ctx.num(6);

        let rewrite = crate::step_model::Step::new_compact(
            "rewrite arccot",
            "Rewrite Arccot As Arctan Reciprocal",
            source,
            arccot_rewritten,
        );
        let diff = crate::step_model::Step::new_compact(
            "diff",
            "Symbolic Differentiation",
            arccot_rewritten,
            diffed,
        );
        let expand_once = crate::step_model::Step::new_compact(
            "expand",
            "Distributive Property",
            diffed,
            expanded_once,
        );
        let expand_twice = crate::step_model::Step::new_compact(
            "expand again",
            "Distributive Property",
            expanded_once,
            expanded_twice,
        );
        let present = crate::step_model::Step::new_compact(
            "present",
            "Present calculus result in compact form",
            expanded_twice,
            presented,
        );

        let out =
            super::to_display_eval_steps(vec![rewrite, diff, expand_once, expand_twice, present]);
        let rules = out
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            rules,
            vec![
                "Rewrite Arccot As Arctan Reciprocal",
                "Symbolic Differentiation",
                "Present calculus result in compact form"
            ]
        );
    }

    #[test]
    fn to_display_eval_steps_removes_delayed_post_diff_rationalize_round_trip() {
        let mut ctx = Context::new();
        let source = ctx.num(1);
        let rewritten = ctx.num(2);
        let diffed = ctx.num(3);
        let pulled = ctx.num(4);
        let rationalized = ctx.num(5);
        let presented = ctx.num(6);

        let rewrite = crate::step_model::Step::new_compact(
            "rewrite arccot",
            "Rewrite Arccot As Arctan Reciprocal",
            source,
            rewritten,
        );
        let diff = crate::step_model::Step::new_compact(
            "diff",
            "Symbolic Differentiation",
            rewritten,
            diffed,
        );
        let pull = crate::step_model::Step::new_compact(
            "pull constant",
            "Pull Constant From Fraction",
            diffed,
            pulled,
        );
        let rationalize = crate::step_model::Step::new_compact(
            "rationalize",
            "Rationalize Product Denominator",
            pulled,
            rationalized,
        );
        let present = crate::step_model::Step::new_compact(
            "present",
            "Present calculus result in compact form",
            rationalized,
            presented,
        );

        let out = super::to_display_eval_steps(vec![rewrite, diff, pull, rationalize, present]);
        let rules = out
            .iter()
            .map(|step| step.rule_name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            rules,
            vec![
                "Rewrite Arccot As Arctan Reciprocal",
                "Symbolic Differentiation",
                "Present calculus result in compact form"
            ]
        );
    }

    #[test]
    fn normalize_expr_for_display_collapses_identity_noise() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let one = ctx.num(1);
        let x_plus_zero = ctx.add(Expr::Add(x, zero));
        let zero_plus_zero = ctx.add(Expr::Add(zero, zero));
        let x_times_one = ctx.add(Expr::Mul(x, one));

        assert_eq!(normalize_expr_for_display(&mut ctx, x_plus_zero), x);
        assert_eq!(normalize_expr_for_display(&mut ctx, zero_plus_zero), zero);
        assert_eq!(normalize_expr_for_display(&mut ctx, x_times_one), x);
    }
}
