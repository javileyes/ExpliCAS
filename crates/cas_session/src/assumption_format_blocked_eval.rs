use cas_ast::{Context, ExprId};

use crate::assumption_format_blocked_support::{
    blocked_hint_suggestion, format_blocked_hint_condition, group_blocked_hint_conditions_by_rule,
};

/// Filter blocked hints for eval display.
///
/// When the resolved result is `Undefined`, drops `defined` hints because
/// they are often cycle-artifacts and not actionable.
pub fn filter_blocked_hints_for_eval(
    ctx: &Context,
    resolved: ExprId,
    hints: &[cas_solver::BlockedHint],
) -> Vec<cas_solver::BlockedHint> {
    let result_is_undefined = matches!(
        ctx.get(resolved),
        cas_ast::Expr::Constant(cas_ast::Constant::Undefined)
    );

    hints
        .iter()
        .filter(|hint| !(result_is_undefined && hint.key.kind() == "defined"))
        .cloned()
        .collect()
}

/// Render blocked hints with eval-oriented messaging.
///
/// Uses a compact single-line format when there is only one hint.
pub fn format_eval_blocked_hints_lines(
    ctx: &Context,
    hints: &[cas_solver::BlockedHint],
    domain_mode: cas_solver::DomainMode,
) -> Vec<String> {
    if hints.is_empty() {
        return Vec::new();
    }

    let grouped = group_blocked_hint_conditions_by_rule(ctx, hints);
    let suggestion = blocked_hint_suggestion(domain_mode, true);

    if grouped.len() == 1 && hints.len() == 1 {
        let hint = &hints[0];
        return vec![
            format!(
                "ℹ️  Blocked: requires {} [{}]",
                format_blocked_hint_condition(ctx, hint),
                hint.rule
            ),
            format!("   {suggestion}"),
        ];
    }

    let mut lines = vec!["ℹ️  Some simplifications were blocked:".to_string()];
    for (rule, conditions) in grouped {
        lines.push(format!(" - Requires {}  [{}]", conditions.join(", "), rule));
    }
    lines.push(format!("   Tip: {suggestion}"));
    lines
}
