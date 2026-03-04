use cas_ast::{Context, ExprId};

/// Filter blocked hints for eval display.
///
/// When the resolved result is `Undefined`, drops `defined` hints because
/// they are often cycle-artifacts and not actionable.
pub fn filter_blocked_hints_for_eval(
    ctx: &Context,
    resolved: ExprId,
    hints: &[crate::BlockedHint],
) -> Vec<crate::BlockedHint> {
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
    hints: &[crate::BlockedHint],
    domain_mode: crate::DomainMode,
) -> Vec<String> {
    if hints.is_empty() {
        return Vec::new();
    }

    let grouped = crate::group_blocked_hint_conditions_by_rule(ctx, hints);
    let suggestion = crate::blocked_hint_suggestion(domain_mode, true);

    if grouped.len() == 1 && hints.len() == 1 {
        let hint = &hints[0];
        return vec![
            format!(
                "ℹ️  Blocked: requires {} [{}]",
                crate::format_blocked_hint_condition(ctx, hint),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolveAssumptionSectionConfig {
    pub debug_mode: bool,
    pub hints_enabled: bool,
    pub domain_mode: crate::DomainMode,
}

/// Render optional solve assumption/blocked sections according to CLI flags.
pub fn format_solve_assumption_and_blocked_sections(
    ctx: &Context,
    assumption_records: &[crate::AssumptionRecord],
    blocked_hints: &[crate::BlockedHint],
    config: SolveAssumptionSectionConfig,
) -> Vec<String> {
    let has_assumptions = !assumption_records.is_empty();
    let has_blocked = !blocked_hints.is_empty();

    if config.debug_mode && (has_assumptions || has_blocked) {
        let mut lines = vec![String::new()];
        if has_assumptions {
            lines.extend(crate::format_assumption_records_section_lines(
                assumption_records,
                "ℹ️ Assumptions used:",
                "  - ",
            ));
        }
        if has_blocked {
            lines.extend(crate::format_blocked_simplifications_section_lines(
                ctx,
                blocked_hints,
                config.domain_mode,
            ));
        }
        return lines;
    }

    if has_blocked && config.hints_enabled {
        let mut lines = vec![String::new()];
        lines.extend(crate::format_blocked_simplifications_section_lines(
            ctx,
            blocked_hints,
            config.domain_mode,
        ));
        return lines;
    }

    Vec::new()
}
