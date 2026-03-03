use cas_ast::{Context, ExprId};
use std::collections::{BTreeMap, BTreeSet, HashSet};

/// Render required-conditions as plain display lines using a line prefix.
pub fn format_required_condition_lines(
    ctx: &Context,
    conditions: &[cas_solver::ImplicitCondition],
    line_prefix: &str,
) -> Vec<String> {
    conditions
        .iter()
        .map(|cond| format!("{line_prefix}{}", cond.display(ctx)))
        .collect()
}

/// Render domain warnings as display lines using a line prefix.
///
/// When `include_rule` is true, appends `(from <rule>)`.
pub fn format_domain_warning_lines(
    warnings: &[cas_solver::DomainWarning],
    include_rule: bool,
    line_prefix: &str,
) -> Vec<String> {
    warnings
        .iter()
        .map(|warning| {
            if include_rule {
                format!(
                    "{line_prefix}{} (from {})",
                    warning.message, warning.rule_name
                )
            } else {
                format!("{line_prefix}{}", warning.message)
            }
        })
        .collect()
}

/// Render blocked hints as compact rule/suggestion lines using a line prefix.
pub fn format_blocked_hint_lines(
    hints: &[cas_solver::BlockedHint],
    line_prefix: &str,
) -> Vec<String> {
    hints
        .iter()
        .map(|hint| format!("{line_prefix}{} (hint: {})", hint.rule, hint.suggestion))
        .collect()
}

/// Render normalized required-conditions as REPL bullet lines.
pub fn format_normalized_condition_lines(
    ctx: &mut Context,
    conditions: &[cas_solver::ImplicitCondition],
    debug_mode: bool,
) -> Vec<String> {
    let normalized_conditions = cas_solver::normalize_and_dedupe_conditions(ctx, conditions);
    normalized_conditions
        .iter()
        .map(|cond| {
            if debug_mode {
                format!("  • {} (normalized)", cond.display(ctx))
            } else {
                format!("  • {}", cond.display(ctx))
            }
        })
        .collect()
}

/// Render display lines for `Diagnostics::requires` after witness filtering.
pub fn format_diagnostics_requires_lines(
    ctx: &mut Context,
    diagnostics: &cas_solver::Diagnostics,
    result_expr: Option<ExprId>,
    display_level: cas_solver::RequiresDisplayLevel,
    debug_mode: bool,
) -> Vec<String> {
    let filtered: Vec<_> = if let Some(result) = result_expr {
        diagnostics.filter_requires_for_display(ctx, result, display_level)
    } else {
        diagnostics.requires.iter().collect()
    };

    if filtered.is_empty() {
        return Vec::new();
    }

    let conditions: Vec<_> = filtered.iter().map(|item| item.cond.clone()).collect();
    format_normalized_condition_lines(ctx, &conditions, debug_mode)
}

fn dedupe_fingerprint(key: &cas_solver::AssumptionKey) -> u64 {
    match key {
        cas_solver::AssumptionKey::NonZero { expr_fingerprint } => *expr_fingerprint,
        cas_solver::AssumptionKey::Positive { expr_fingerprint } => {
            expr_fingerprint.wrapping_add(1_000_000)
        }
        cas_solver::AssumptionKey::NonNegative { expr_fingerprint } => {
            expr_fingerprint.wrapping_add(2_000_000)
        }
        cas_solver::AssumptionKey::Defined { expr_fingerprint } => {
            expr_fingerprint.wrapping_add(3_000_000)
        }
        cas_solver::AssumptionKey::InvTrigPrincipalRange {
            arg_fingerprint, ..
        } => arg_fingerprint.wrapping_add(4_000_000),
        cas_solver::AssumptionKey::ComplexPrincipalBranch {
            arg_fingerprint, ..
        } => arg_fingerprint.wrapping_add(5_000_000),
    }
}

fn condition_text(key: &cas_solver::AssumptionKey, expr_display: &str) -> String {
    match key {
        cas_solver::AssumptionKey::NonZero { .. } => format!("{expr_display} ≠ 0"),
        cas_solver::AssumptionKey::Positive { .. } => format!("{expr_display} > 0"),
        cas_solver::AssumptionKey::NonNegative { .. } => format!("{expr_display} ≥ 0"),
        cas_solver::AssumptionKey::Defined { .. } => format!("{expr_display} is defined"),
        cas_solver::AssumptionKey::InvTrigPrincipalRange { func, .. } => {
            format!("{expr_display} in {func} principal range")
        }
        cas_solver::AssumptionKey::ComplexPrincipalBranch { func, .. } => {
            format!("{func}({expr_display}) principal branch")
        }
    }
}

/// Collect assumptions used from simplification steps.
///
/// Returns `(condition_text, rule_name)` items, deduplicated by
/// `(assumption_kind, expr_fingerprint)` to avoid cascades.
pub fn collect_assumed_conditions_from_steps(steps: &[cas_solver::Step]) -> Vec<(String, String)> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();

    for step in steps {
        let assumption_events = cas_solver::assumption_events_from_step(step);
        for event in &assumption_events {
            let fp = dedupe_fingerprint(&event.key);
            if seen.insert(fp) {
                result.push((
                    condition_text(&event.key, &event.expr_display),
                    step.rule_name.clone(),
                ));
            }
        }
    }

    result
}

/// Group `(condition, rule)` assumed-condition pairs by rule name.
///
/// Conditions are sorted and deduplicated inside each rule group.
pub fn group_assumed_conditions_by_rule(
    conditions: &[(String, String)],
) -> Vec<(String, Vec<String>)> {
    let mut grouped: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (condition, rule) in conditions {
        grouped
            .entry(rule.clone())
            .or_default()
            .insert(condition.clone());
    }

    grouped
        .into_iter()
        .map(|(rule, conditions)| (rule, conditions.into_iter().collect()))
        .collect()
}

/// Format "assumptions used" report lines for REPL display.
pub fn format_assumed_conditions_report_lines(conditions: &[(String, String)]) -> Vec<String> {
    if conditions.is_empty() {
        return Vec::new();
    }

    if conditions.len() == 1 {
        let (cond, rule) = &conditions[0];
        return vec![format!(
            "ℹ️  Assumptions used (assumed): {} [{}]",
            cond, rule
        )];
    }

    let mut lines = vec!["ℹ️  Assumptions used (assumed):".to_string()];
    for (rule, conds) in group_assumed_conditions_by_rule(conditions) {
        lines.push(format!("   - {} [{}]", conds.join(", "), rule));
    }
    lines
}

fn format_blocked_hint_condition(ctx: &Context, hint: &cas_solver::BlockedHint) -> String {
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

fn group_blocked_hint_conditions_by_rule(
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

fn blocked_hint_suggestion(
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

fn assumption_record_summary_item(record: &cas_solver::AssumptionRecord) -> String {
    if record.count > 1 {
        format!("{}({}) (×{})", record.kind, record.expr, record.count)
    } else {
        format!("{}({})", record.kind, record.expr)
    }
}

fn assumption_record_condition(record: &cas_solver::AssumptionRecord) -> String {
    match record.kind.to_ascii_lowercase().as_str() {
        "positive" => format!("{} > 0", record.expr),
        "nonzero" => format!("{} ≠ 0", record.expr),
        "nonnegative" => format!("{} ≥ 0", record.expr),
        "defined" => format!("{} is defined", record.expr),
        _ => format!("{} ({})", record.expr, record.kind),
    }
}

/// Format assumptions summary payload for REPL/UI.
///
/// Returns only the right side content (without the `⚠ Assumptions:` prefix).
pub fn format_assumption_records_summary(
    records: &[cas_solver::AssumptionRecord],
) -> Option<String> {
    if records.is_empty() {
        return None;
    }
    let items: Vec<String> = records.iter().map(assumption_record_summary_item).collect();
    Some(items.join(", "))
}

fn format_assumption_records_conditions(records: &[cas_solver::AssumptionRecord]) -> Vec<String> {
    let mut items: Vec<String> = records.iter().map(assumption_record_condition).collect();
    items.sort();
    items.dedup();
    items
}

fn format_assumption_records_section_lines(
    records: &[cas_solver::AssumptionRecord],
    header: &str,
    line_prefix: &str,
) -> Vec<String> {
    if records.is_empty() {
        return Vec::new();
    }

    let mut lines = vec![header.to_string()];
    for cond in format_assumption_records_conditions(records) {
        lines.push(format!("{line_prefix}{cond}"));
    }
    lines
}

fn collect_blocked_hint_items(
    ctx: &Context,
    hints: &[cas_solver::BlockedHint],
) -> Vec<(String, String)> {
    let mut items: Vec<(String, String)> = hints
        .iter()
        .map(|hint| (format_blocked_hint_condition(ctx, hint), hint.rule.clone()))
        .collect();
    items.sort();
    items.dedup();
    items
}

fn format_blocked_simplifications_section_lines(
    ctx: &Context,
    hints: &[cas_solver::BlockedHint],
    domain_mode: cas_solver::DomainMode,
) -> Vec<String> {
    if hints.is_empty() {
        return Vec::new();
    }

    let mut lines = vec!["ℹ️ Blocked simplifications:".to_string()];
    for (cond, rule) in collect_blocked_hint_items(ctx, hints) {
        lines.push(format!("  - requires {}  [{}]", cond, rule));
    }
    lines.push(format!(
        "  tip: {}",
        blocked_hint_suggestion(domain_mode, false)
    ));
    lines
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolveAssumptionSectionConfig {
    pub debug_mode: bool,
    pub hints_enabled: bool,
    pub domain_mode: cas_solver::DomainMode,
}

/// Render optional solve assumption/blocked sections according to CLI flags.
pub fn format_solve_assumption_and_blocked_sections(
    ctx: &Context,
    assumption_records: &[cas_solver::AssumptionRecord],
    blocked_hints: &[cas_solver::BlockedHint],
    config: SolveAssumptionSectionConfig,
) -> Vec<String> {
    let has_assumptions = !assumption_records.is_empty();
    let has_blocked = !blocked_hints.is_empty();

    if config.debug_mode && (has_assumptions || has_blocked) {
        let mut lines = vec![String::new()];
        if has_assumptions {
            lines.extend(format_assumption_records_section_lines(
                assumption_records,
                "ℹ️ Assumptions used:",
                "  - ",
            ));
        }
        if has_blocked {
            lines.extend(format_blocked_simplifications_section_lines(
                ctx,
                blocked_hints,
                config.domain_mode,
            ));
        }
        return lines;
    }

    if has_blocked && config.hints_enabled {
        let mut lines = vec![String::new()];
        lines.extend(format_blocked_simplifications_section_lines(
            ctx,
            blocked_hints,
            config.domain_mode,
        ));
        return lines;
    }

    Vec::new()
}

/// Format displayable assumption events into compact single-line strings.
///
/// Output format: `"<icon> <label>: <message>"`.
pub fn format_displayable_assumption_lines(events: &[cas_solver::AssumptionEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| {
            let kind = event.kind;
            if kind.should_display() {
                Some(format!(
                    "{} {}: {}",
                    kind.icon(),
                    kind.label(),
                    event.message
                ))
            } else {
                None
            }
        })
        .collect()
}

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
