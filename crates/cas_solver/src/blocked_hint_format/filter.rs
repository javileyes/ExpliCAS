use cas_ast::{Context, Expr, ExprId};
use num_traits::Zero;
use std::collections::BTreeSet;

fn implicit_domain_condition_displays(ctx: &Context, expr: ExprId) -> BTreeSet<String> {
    crate::infer_implicit_domain(ctx, expr, crate::ValueDomain::RealOnly)
        .conditions()
        .iter()
        .map(|condition| condition.display(ctx))
        .collect()
}

fn explicit_condition_displays(
    ctx: &Context,
    conditions: &[crate::ImplicitCondition],
) -> BTreeSet<String> {
    conditions
        .iter()
        .map(|condition| condition.display(ctx))
        .collect()
}

fn cycle_defined_hint_is_covered_by_result_domain(
    ctx: &Context,
    result: ExprId,
    required_conditions: &[crate::ImplicitCondition],
    hint: &crate::BlockedHint,
) -> bool {
    if hint.key.kind() != "defined" || !hint.suggestion.starts_with("cycle detected") {
        return false;
    }

    let hint_conditions = implicit_domain_condition_displays(ctx, hint.expr_id);
    if hint_conditions.is_empty() {
        return false;
    }

    let result_conditions = implicit_domain_condition_displays(ctx, result);
    let explicit_conditions = explicit_condition_displays(ctx, required_conditions);
    hint_conditions.iter().all(|condition| {
        result_conditions.contains(condition) || explicit_conditions.contains(condition)
    })
}

/// Filter blocked hints for eval display.
///
/// When the resolved result is `Undefined`, drops `defined` hints because
/// they are often cycle-artifacts and not actionable.
pub fn filter_blocked_hints_for_eval(
    ctx: &Context,
    resolved: ExprId,
    required_conditions: &[crate::ImplicitCondition],
    hints: &[crate::BlockedHint],
) -> Vec<crate::BlockedHint> {
    let result_is_undefined = matches!(
        ctx.get(resolved),
        cas_ast::Expr::Constant(cas_ast::Constant::Undefined)
    );

    let result_is_zero = matches!(ctx.get(resolved), Expr::Number(value) if value.is_zero());

    hints
        .iter()
        .filter(|hint| {
            !(hint.key.kind() == "defined"
                && (result_is_undefined
                    || (result_is_zero && hint.suggestion.starts_with("cycle detected"))
                    || cycle_defined_hint_is_covered_by_result_domain(
                        ctx,
                        resolved,
                        required_conditions,
                        hint,
                    )))
        })
        .cloned()
        .collect()
}
