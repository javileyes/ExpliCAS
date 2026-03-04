use cas_ast::{Context, ExprId};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{Hash, Hasher};

/// Classification of assumptions for display filtering and UI presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AssumptionKind {
    /// Redundant with requires from input - not displayed
    DerivedFromRequires,
    /// New constraint necessary for equivalence
    #[default]
    RequiresIntroduced,
    /// Heuristic for simplification, user convenience choice
    HeuristicAssumption,
    /// Choosing one branch of multi-valued function
    BranchChoice,
    /// Extending domain (e.g. R -> C)
    DomainExtension,
}

impl AssumptionKind {
    /// Should this assumption be displayed to the user?
    pub fn should_display(&self) -> bool {
        !matches!(self, Self::DerivedFromRequires)
    }

    /// Get the display icon for this kind.
    pub fn icon(&self) -> &'static str {
        match self {
            Self::DerivedFromRequires => "",
            Self::RequiresIntroduced => "ℹ️",
            Self::HeuristicAssumption => "⚠️",
            Self::BranchChoice => "🔀",
            Self::DomainExtension => "🧿",
        }
    }

    /// Get the display label for this kind.
    pub fn label(&self) -> &'static str {
        match self {
            Self::DerivedFromRequires => "Derived",
            Self::RequiresIntroduced => "Requires",
            Self::HeuristicAssumption => "Assumes",
            Self::BranchChoice => "Branch",
            Self::DomainExtension => "Domain",
        }
    }
}

/// Aggregated assumption record produced by solver/engine flows.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AssumptionRecord {
    pub kind: String,
    pub expr: String,
    pub message: String,
    pub count: u32,
}

/// Hashable key for assumption deduplication.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AssumptionKey {
    /// Assumed expression is non-zero.
    NonZero { expr_fingerprint: u64 },
    /// Assumed expression is positive.
    Positive { expr_fingerprint: u64 },
    /// Assumed expression is non-negative.
    NonNegative { expr_fingerprint: u64 },
    /// Assumed expression is defined.
    Defined { expr_fingerprint: u64 },
    /// Assumed argument is in principal range for inverse trig composition.
    InvTrigPrincipalRange {
        func: &'static str,
        arg_fingerprint: u64,
    },
    /// Assumed principal branch for complex multi-valued functions.
    ComplexPrincipalBranch {
        func: &'static str,
        arg_fingerprint: u64,
    },
}

impl AssumptionKey {
    /// Get the kind as a string for JSON output.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::NonZero { .. } => "nonzero",
            Self::Positive { .. } => "positive",
            Self::NonNegative { .. } => "nonnegative",
            Self::Defined { .. } => "defined",
            Self::InvTrigPrincipalRange { .. } => "principal_range",
            Self::ComplexPrincipalBranch { .. } => "principal_branch",
        }
    }

    /// Get the condition class for domain-mode gating.
    pub fn class(&self) -> crate::solve_safety_policy::ConditionClass {
        match self {
            Self::NonZero { .. } | Self::Defined { .. } => {
                crate::solve_safety_policy::ConditionClass::Definability
            }
            Self::Positive { .. }
            | Self::NonNegative { .. }
            | Self::InvTrigPrincipalRange { .. }
            | Self::ComplexPrincipalBranch { .. } => {
                crate::solve_safety_policy::ConditionClass::Analytic
            }
        }
    }

    /// Create a positive key from an expression.
    pub fn positive_key(ctx: &Context, expr: ExprId) -> Self {
        Self::Positive {
            expr_fingerprint: expr_fingerprint(ctx, expr),
        }
    }

    /// Create a non-zero key from an expression.
    pub fn nonzero_key(ctx: &Context, expr: ExprId) -> Self {
        Self::NonZero {
            expr_fingerprint: expr_fingerprint(ctx, expr),
        }
    }

    /// Create a non-negative key from an expression.
    pub fn nonnegative_key(ctx: &Context, expr: ExprId) -> Self {
        Self::NonNegative {
            expr_fingerprint: expr_fingerprint(ctx, expr),
        }
    }

    /// Get a human-readable display for the required condition.
    pub fn condition_display(&self) -> &'static str {
        match self {
            Self::NonZero { .. } => "≠ 0 (NonZero)",
            Self::Positive { .. } => "> 0 (Positive)",
            Self::NonNegative { .. } => "≥ 0 (NonNegative)",
            Self::Defined { .. } => "is defined",
            Self::InvTrigPrincipalRange { func, .. } => match *func {
                "asin" | "acos" => "∈ [-1, 1]",
                "atan" => "∈ ℝ",
                _ => "in principal range",
            },
            Self::ComplexPrincipalBranch { .. } => "principal branch",
        }
    }
}

/// Compute a stable fingerprint for an expression based on canonical display.
pub fn expr_fingerprint(ctx: &Context, expr: ExprId) -> u64 {
    let display = cas_formatter::render_expr(ctx, expr);
    let mut hasher = DefaultHasher::new();
    display.hash(&mut hasher);
    hasher.finish()
}

/// One assumption event emitted during rewriting.
#[derive(Debug, Clone)]
pub struct AssumptionEvent {
    pub key: AssumptionKey,
    pub expr_display: String,
    pub message: String,
    pub kind: AssumptionKind,
    pub expr_id: Option<ExprId>,
}

impl AssumptionEvent {
    /// Create a non-zero assumption event.
    pub fn nonzero(ctx: &Context, expr: ExprId) -> Self {
        let display = cas_formatter::render_expr(ctx, expr);
        Self {
            key: AssumptionKey::NonZero {
                expr_fingerprint: expr_fingerprint(ctx, expr),
            },
            expr_display: display.clone(),
            message: format!("{display} ≠ 0"),
            kind: AssumptionKind::DerivedFromRequires,
            expr_id: Some(expr),
        }
    }

    /// Create a positive assumption event.
    pub fn positive(ctx: &Context, expr: ExprId) -> Self {
        let display = cas_formatter::render_expr(ctx, expr);
        Self {
            key: AssumptionKey::Positive {
                expr_fingerprint: expr_fingerprint(ctx, expr),
            },
            expr_display: display.clone(),
            message: format!("{display} > 0"),
            kind: AssumptionKind::RequiresIntroduced,
            expr_id: Some(expr),
        }
    }

    /// Create a positive-assumed assumption event.
    pub fn positive_assumed(ctx: &Context, expr: ExprId) -> Self {
        let display = cas_formatter::render_expr(ctx, expr);
        Self {
            key: AssumptionKey::Positive {
                expr_fingerprint: expr_fingerprint(ctx, expr),
            },
            expr_display: display.clone(),
            message: format!("{display} > 0"),
            kind: AssumptionKind::HeuristicAssumption,
            expr_id: Some(expr),
        }
    }

    /// Create a non-negative assumption event.
    pub fn nonnegative(ctx: &Context, expr: ExprId) -> Self {
        let display = cas_formatter::render_expr(ctx, expr);
        Self {
            key: AssumptionKey::NonNegative {
                expr_fingerprint: expr_fingerprint(ctx, expr),
            },
            expr_display: display.clone(),
            message: format!("{display} ≥ 0"),
            kind: AssumptionKind::DerivedFromRequires,
            expr_id: Some(expr),
        }
    }

    /// Create a definedness assumption event.
    pub fn defined(ctx: &Context, expr: ExprId) -> Self {
        let display = cas_formatter::render_expr(ctx, expr);
        Self {
            key: AssumptionKey::Defined {
                expr_fingerprint: expr_fingerprint(ctx, expr),
            },
            expr_display: display.clone(),
            message: format!("{display} is defined"),
            kind: AssumptionKind::DerivedFromRequires,
            expr_id: Some(expr),
        }
    }

    /// Create an inverse-trig principal-range assumption event.
    pub fn inv_trig_principal_range(ctx: &Context, func: &'static str, arg: ExprId) -> Self {
        let display = cas_formatter::render_expr(ctx, arg);
        Self {
            key: AssumptionKey::InvTrigPrincipalRange {
                func,
                arg_fingerprint: expr_fingerprint(ctx, arg),
            },
            expr_display: display.clone(),
            message: format!("{display} in {func} principal range"),
            kind: AssumptionKind::BranchChoice,
            expr_id: Some(arg),
        }
    }

    /// Create a complex principal-branch assumption event.
    pub fn complex_principal_branch(ctx: &Context, func: &'static str, arg: ExprId) -> Self {
        let display = cas_formatter::render_expr(ctx, arg);
        Self {
            key: AssumptionKey::ComplexPrincipalBranch {
                func,
                arg_fingerprint: expr_fingerprint(ctx, arg),
            },
            expr_display: display.clone(),
            message: format!("{func}({display}) principal branch"),
            kind: AssumptionKind::BranchChoice,
            expr_id: Some(arg),
        }
    }

    /// Parse a legacy string into an assumption event.
    pub fn from_legacy_string(message: &str) -> Self {
        let (kind, expr_display) = if message.contains("≠ 0") || message.contains("!= 0") {
            let expr = if message.contains("denominator") {
                "denominator"
            } else if message.contains("x") {
                "x"
            } else {
                "expr"
            };
            ("nonzero", expr.to_string())
        } else if message.contains("defined") {
            ("defined", "expression".to_string())
        } else if message.contains("> 0") || message.contains("positive") {
            ("positive", "expr".to_string())
        } else if message.contains("principal") || message.contains("range") {
            ("principal_range", "arg".to_string())
        } else {
            ("defined", "expression".to_string())
        };

        let mut hasher = DefaultHasher::new();
        message.hash(&mut hasher);
        let fp = hasher.finish();

        let key = match kind {
            "nonzero" => AssumptionKey::NonZero {
                expr_fingerprint: fp,
            },
            "positive" => AssumptionKey::Positive {
                expr_fingerprint: fp,
            },
            "principal_range" => AssumptionKey::InvTrigPrincipalRange {
                func: "trig",
                arg_fingerprint: fp,
            },
            _ => AssumptionKey::Defined {
                expr_fingerprint: fp,
            },
        };

        Self {
            key,
            expr_display,
            message: message.to_string(),
            kind: AssumptionKind::RequiresIntroduced,
            expr_id: None,
        }
    }
}

/// Condition kind derivable from an assumption event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssumptionConditionKind {
    NonZero,
    Positive,
    NonNegative,
}

/// Convert an assumption event into a condition-kind + expression-id pair.
///
/// Returns `None` for assumptions that are not modeled as explicit implicit
/// conditions (e.g. branch/principal-range choices).
pub fn assumption_condition_kind(
    event: &AssumptionEvent,
) -> Option<(AssumptionConditionKind, ExprId)> {
    let expr_id = event.expr_id?;
    let kind = match event.key {
        AssumptionKey::NonZero { .. } => AssumptionConditionKind::NonZero,
        AssumptionKey::Positive { .. } => AssumptionConditionKind::Positive,
        AssumptionKey::NonNegative { .. } => AssumptionConditionKind::NonNegative,
        AssumptionKey::Defined { .. }
        | AssumptionKey::InvTrigPrincipalRange { .. }
        | AssumptionKey::ComplexPrincipalBranch { .. } => return None,
    };
    Some((kind, expr_id))
}

/// Canonical assumption-kind reclassification policy shared by engine/solver.
///
/// `condition_implied` must be:
/// - `None` when the event cannot be mapped to a condition predicate.
/// - `Some(true)` when a mapped condition is already implied by known requires.
/// - `Some(false)` when a mapped condition is new.
///
/// Returns `(new_kind, should_introduce_requirement)`.
pub fn classify_assumption_kind(
    original_kind: AssumptionKind,
    condition_implied: Option<bool>,
) -> (AssumptionKind, bool) {
    let Some(is_implied) = condition_implied else {
        return (original_kind, false);
    };

    match original_kind {
        AssumptionKind::BranchChoice | AssumptionKind::DomainExtension => (original_kind, false),
        _ if is_implied => (AssumptionKind::DerivedFromRequires, false),
        AssumptionKind::HeuristicAssumption => (AssumptionKind::HeuristicAssumption, false),
        _ => (AssumptionKind::RequiresIntroduced, true),
    }
}

/// Classify one event plus its optional mapped condition using canonical policy.
///
/// Returns `(new_kind, condition_to_introduce)`, where the second value is
/// populated only when the condition is not implied and policy requires adding
/// it to introduced-requires.
pub fn classify_assumption_with_condition<T>(
    event: &AssumptionEvent,
    condition: Option<T>,
    mut is_condition_implied: impl FnMut(&T) -> bool,
) -> (AssumptionKind, Option<T>) {
    let Some(cond) = condition else {
        let (new_kind, _) = classify_assumption_kind(event.kind, None);
        return (new_kind, None);
    };

    let implied = is_condition_implied(&cond);
    let (new_kind, should_introduce) = classify_assumption_kind(event.kind, Some(implied));
    if should_introduce {
        (new_kind, Some(cond))
    } else {
        (new_kind, None)
    }
}

/// Collects assumption events with deduplication.
///
/// Multiple events with the same key are counted, not repeated.
#[derive(Debug, Clone, Default)]
pub struct AssumptionCollector {
    map: HashMap<AssumptionKey, (AssumptionEvent, u32)>,
}

impl AssumptionCollector {
    /// Create an empty collector.
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Register one assumption event.
    pub fn note(&mut self, event: AssumptionEvent) {
        let key = event.key.clone();
        self.map
            .entry(key)
            .and_modify(|(_, count)| *count += 1)
            .or_insert((event, 1));
    }

    /// Returns true when collector has no assumptions.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Number of unique assumptions collected.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Finish and return deduped assumption records.
    pub fn finish(self) -> Vec<AssumptionRecord> {
        let mut records: Vec<AssumptionRecord> = self
            .map
            .into_values()
            .map(|(event, count)| AssumptionRecord {
                kind: event.key.kind().to_string(),
                expr: event.expr_display,
                message: event.message,
                count,
            })
            .collect();
        records.sort_by(|a, b| a.kind.cmp(&b.kind).then_with(|| a.expr.cmp(&b.expr)));
        records
    }

    /// Human-readable summary line.
    pub fn summary_line(&self) -> Option<String> {
        if self.map.is_empty() {
            return None;
        }
        let mut items: Vec<String> = self
            .map
            .values()
            .map(|(event, count)| {
                if *count > 1 {
                    format!("{}({}) (×{})", event.key.kind(), event.expr_display, count)
                } else {
                    format!("{}({})", event.key.kind(), event.expr_display)
                }
            })
            .collect();
        items.sort();
        Some(format!("⚠ Assumptions: {}", items.join(", ")))
    }
}

/// Aggregate assumption events into sorted assumption records.
pub fn collect_assumption_records(events: &[AssumptionEvent]) -> Vec<AssumptionRecord> {
    collect_assumption_records_from_iter(events.iter().cloned())
}

/// Aggregate assumption events from iterator into sorted assumption records.
pub fn collect_assumption_records_from_iter<I>(events: I) -> Vec<AssumptionRecord>
where
    I: IntoIterator<Item = AssumptionEvent>,
{
    let mut collector = AssumptionCollector::new();
    for event in events {
        collector.note(event);
    }
    collector.finish()
}

/// Format one blocked hint condition as a human-readable predicate string.
pub fn format_blocked_hint_condition(
    ctx: &Context,
    hint: &crate::blocked_hint::BlockedHint,
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

/// Group blocked hints by rule, with sorted and deduplicated condition lists.
pub fn group_blocked_hint_conditions_by_rule(
    ctx: &Context,
    hints: &[crate::blocked_hint::BlockedHint],
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

/// Collect `(condition, rule)` blocked-hint items sorted and deduplicated.
pub fn collect_blocked_hint_items(
    ctx: &Context,
    hints: &[crate::blocked_hint::BlockedHint],
) -> Vec<(String, String)> {
    let mut items: Vec<(String, String)> = hints
        .iter()
        .map(|hint| (format_blocked_hint_condition(ctx, hint), hint.rule.clone()))
        .collect();
    items.sort();
    items.dedup();
    items
}

/// Suggestion text for blocked simplifications by domain mode.
///
/// Set `mention_analytic` when UI copy should explicitly mention analytic assumptions.
pub fn blocked_hint_suggestion(
    domain_mode: crate::domain_mode::DomainMode,
    mention_analytic: bool,
) -> &'static str {
    match domain_mode {
        crate::domain_mode::DomainMode::Strict => {
            "use `domain generic` or `domain assume` to allow"
        }
        crate::domain_mode::DomainMode::Generic => {
            if mention_analytic {
                "use `semantics set domain assume` to allow analytic assumptions"
            } else {
                "use `semantics set domain assume` to allow"
            }
        }
        crate::domain_mode::DomainMode::Assume => "assumptions already enabled",
    }
}

/// Render a full blocked-simplifications section for CLI output.
pub fn format_blocked_simplifications_section_lines(
    ctx: &Context,
    hints: &[crate::blocked_hint::BlockedHint],
    domain_mode: crate::domain_mode::DomainMode,
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

fn assumption_record_condition(record: &AssumptionRecord) -> String {
    match record.kind.to_ascii_lowercase().as_str() {
        "positive" => format!("{} > 0", record.expr),
        "nonzero" => format!("{} ≠ 0", record.expr),
        "nonnegative" => format!("{} ≥ 0", record.expr),
        "defined" => format!("{} is defined", record.expr),
        _ => format!("{} ({})", record.expr, record.kind),
    }
}

/// Format assumptions as condition strings for debug explain blocks.
///
/// Returned values are sorted and deduplicated for stable output.
pub fn format_assumption_records_conditions(records: &[AssumptionRecord]) -> Vec<String> {
    let mut items: Vec<String> = records.iter().map(assumption_record_condition).collect();
    items.sort();
    items.dedup();
    items
}

/// Format an assumptions section from aggregated assumption records.
pub fn format_assumption_records_section_lines(
    records: &[AssumptionRecord],
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

/// Map one logarithmic assumption into an assumption event.
pub fn assumption_event_from_log_assumption(
    ctx: &Context,
    assumption: crate::log_domain::LogAssumption,
    base: ExprId,
    rhs: ExprId,
) -> AssumptionEvent {
    crate::log_assumptions::map_log_assumption_target_with(
        ctx,
        assumption,
        base,
        rhs,
        AssumptionEvent::positive,
    )
}

/// Convert one blocked logarithmic hint into a blocked-hint payload.
pub fn map_log_blocked_hint(
    ctx: &Context,
    hint: crate::solve_outcome::LogBlockedHintRecord,
) -> crate::blocked_hint::BlockedHint {
    let mapped =
        crate::log_assumptions::map_log_blocked_hint_with(ctx, hint, AssumptionEvent::positive);
    crate::blocked_hint::BlockedHint {
        key: mapped.event.key,
        expr_id: mapped.expr_id,
        rule: mapped.rule.to_string(),
        suggestion: mapped.suggestion,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        assumption_condition_kind, blocked_hint_suggestion, classify_assumption_kind,
        classify_assumption_with_condition, collect_blocked_hint_items,
        format_assumption_records_conditions, format_blocked_hint_condition,
        AssumptionConditionKind, AssumptionEvent, AssumptionKey, AssumptionKind, AssumptionRecord,
    };

    #[test]
    fn assumption_kind_metadata() {
        assert!(AssumptionKind::RequiresIntroduced.should_display());
        assert!(!AssumptionKind::DerivedFromRequires.should_display());
        assert_eq!(AssumptionKind::BranchChoice.icon(), "🔀");
        assert_eq!(AssumptionKind::DomainExtension.label(), "Domain");
    }

    #[test]
    fn assumption_key_classification() {
        assert_eq!(
            AssumptionKey::NonZero {
                expr_fingerprint: 1
            }
            .class(),
            crate::solve_safety_policy::ConditionClass::Definability
        );
        assert_eq!(
            AssumptionKey::Positive {
                expr_fingerprint: 1
            }
            .class(),
            crate::solve_safety_policy::ConditionClass::Analytic
        );
        assert_eq!(
            AssumptionKey::InvTrigPrincipalRange {
                func: "asin",
                arg_fingerprint: 1
            }
            .condition_display(),
            "∈ [-1, 1]"
        );
    }

    #[test]
    fn classify_assumption_kind_matches_engine_policy() {
        assert_eq!(
            classify_assumption_kind(AssumptionKind::RequiresIntroduced, None),
            (AssumptionKind::RequiresIntroduced, false)
        );
        assert_eq!(
            classify_assumption_kind(AssumptionKind::RequiresIntroduced, Some(true)),
            (AssumptionKind::DerivedFromRequires, false)
        );
        assert_eq!(
            classify_assumption_kind(AssumptionKind::RequiresIntroduced, Some(false)),
            (AssumptionKind::RequiresIntroduced, true)
        );
        assert_eq!(
            classify_assumption_kind(AssumptionKind::HeuristicAssumption, Some(true)),
            (AssumptionKind::DerivedFromRequires, false)
        );
        assert_eq!(
            classify_assumption_kind(AssumptionKind::HeuristicAssumption, Some(false)),
            (AssumptionKind::HeuristicAssumption, false)
        );
        assert_eq!(
            classify_assumption_kind(AssumptionKind::BranchChoice, Some(false)),
            (AssumptionKind::BranchChoice, false)
        );
    }

    #[test]
    fn assumption_condition_kind_extracts_conditional_events() {
        let mut ctx = cas_ast::Context::default();
        let x = ctx.var("x");
        let nonzero = AssumptionEvent::nonzero(&ctx, x);
        let positive = AssumptionEvent::positive(&ctx, x);
        let defined = AssumptionEvent::defined(&ctx, x);

        assert_eq!(
            assumption_condition_kind(&nonzero),
            Some((AssumptionConditionKind::NonZero, x))
        );
        assert_eq!(
            assumption_condition_kind(&positive),
            Some((AssumptionConditionKind::Positive, x))
        );
        assert_eq!(assumption_condition_kind(&defined), None);
    }

    #[test]
    fn classify_assumption_with_condition_returns_expected_payload() {
        let mut ctx = cas_ast::Context::default();
        let x = ctx.var("x");
        let event = AssumptionEvent::positive(&ctx, x);

        let (kind_none, cond_none) =
            classify_assumption_with_condition(&event, None::<u8>, |_| false);
        assert_eq!(kind_none, AssumptionKind::RequiresIntroduced);
        assert_eq!(cond_none, None);

        let (kind_implied, cond_implied) =
            classify_assumption_with_condition(&event, Some(7u8), |_| true);
        assert_eq!(kind_implied, AssumptionKind::DerivedFromRequires);
        assert_eq!(cond_implied, None);

        let (kind_new, cond_new) = classify_assumption_with_condition(&event, Some(9u8), |_| false);
        assert_eq!(kind_new, AssumptionKind::RequiresIntroduced);
        assert_eq!(cond_new, Some(9u8));
    }

    #[test]
    fn blocked_hint_formatting_and_grouping_are_stable() {
        let mut ctx = cas_ast::Context::default();
        let x = ctx.var("x");
        let hint = crate::blocked_hint::BlockedHint {
            key: AssumptionKey::NonZero {
                expr_fingerprint: 11,
            },
            expr_id: x,
            rule: "RuleA".to_string(),
            suggestion: "tip",
        };

        assert_eq!(format_blocked_hint_condition(&ctx, &hint), "x ≠ 0");
        let items = collect_blocked_hint_items(&ctx, &[hint]);
        assert_eq!(items, vec![("x ≠ 0".to_string(), "RuleA".to_string())]);
    }

    #[test]
    fn blocked_hint_suggestion_matrix_matches_contract() {
        use crate::domain_mode::DomainMode;

        assert_eq!(
            blocked_hint_suggestion(DomainMode::Strict, false),
            "use `domain generic` or `domain assume` to allow"
        );
        assert_eq!(
            blocked_hint_suggestion(DomainMode::Generic, true),
            "use `semantics set domain assume` to allow analytic assumptions"
        );
        assert_eq!(
            blocked_hint_suggestion(DomainMode::Assume, false),
            "assumptions already enabled"
        );
    }

    #[test]
    fn format_assumption_records_conditions_sorts_and_dedups() {
        let records = vec![
            AssumptionRecord {
                kind: "nonzero".to_string(),
                expr: "x".to_string(),
                message: "x != 0".to_string(),
                count: 1,
            },
            AssumptionRecord {
                kind: "nonzero".to_string(),
                expr: "x".to_string(),
                message: "x != 0".to_string(),
                count: 2,
            },
            AssumptionRecord {
                kind: "positive".to_string(),
                expr: "y".to_string(),
                message: "y > 0".to_string(),
                count: 1,
            },
        ];

        assert_eq!(
            format_assumption_records_conditions(&records),
            vec!["x ≠ 0".to_string(), "y > 0".to_string()]
        );
    }
}
