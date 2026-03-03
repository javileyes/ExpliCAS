use cas_ast::Context;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn format_blocked_hint_condition(
    ctx: &Context,
    hint: &cas_solver::BlockedHint,
) -> String {
    let expr_str = cas_formatter::DisplayExpr {
        context: ctx,
        id: hint.expr_id,
    }
    .to_string();
    match hint.key.kind() {
        "positive" => format!("{expr_str} > 0"),
        "nonzero" => format!("{expr_str} ≠ 0"),
        "nonnegative" => format!("{expr_str} ≥ 0"),
        _ => format!("{expr_str} ({})", hint.key.kind()),
    }
}

pub(crate) fn group_blocked_hint_conditions_by_rule(
    ctx: &Context,
    hints: &[cas_solver::BlockedHint],
) -> Vec<(String, Vec<String>)> {
    let mut grouped: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for hint in hints {
        grouped
            .entry(hint.rule.clone())
            .or_default()
            .insert(format_blocked_hint_condition(ctx, hint));
    }

    grouped
        .into_iter()
        .map(|(rule, conditions)| (rule, conditions.into_iter().collect()))
        .collect()
}

pub(crate) fn blocked_hint_suggestion(
    domain_mode: cas_solver::DomainMode,
    mention_analytic: bool,
) -> &'static str {
    match domain_mode {
        cas_solver::DomainMode::Strict => "use `domain generic` or `domain assume` to allow",
        cas_solver::DomainMode::Generic => {
            if mention_analytic {
                "use `semantics set domain assume` to allow analytic assumptions"
            } else {
                "use `semantics set domain assume` to allow"
            }
        }
        cas_solver::DomainMode::Assume => "assumptions already enabled",
    }
}
