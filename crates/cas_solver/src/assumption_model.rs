use cas_ast::{Context, ExprId};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};

fn expr_fingerprint(ctx: &Context, expr: ExprId) -> u64 {
    let display = cas_formatter::render_expr(ctx, expr);
    let mut hasher = DefaultHasher::new();
    display.hash(&mut hasher);
    hasher.finish()
}

/// Classification of assumptions for display filtering and UI presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AssumptionKind {
    /// Redundant with requires from input - NOT displayed
    DerivedFromRequires,
    /// New constraint necessary for equivalence, not deducible from input
    #[default]
    RequiresIntroduced,
    /// Heuristic for simplification, user convenience choice
    HeuristicAssumption,
    /// Choosing one branch of multi-valued function
    BranchChoice,
    /// Extending domain (e.g. ℝ -> ℂ)
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

impl From<cas_engine::AssumptionKind> for AssumptionKind {
    fn from(value: cas_engine::AssumptionKind) -> Self {
        match value {
            cas_engine::AssumptionKind::DerivedFromRequires => Self::DerivedFromRequires,
            cas_engine::AssumptionKind::RequiresIntroduced => Self::RequiresIntroduced,
            cas_engine::AssumptionKind::HeuristicAssumption => Self::HeuristicAssumption,
            cas_engine::AssumptionKind::BranchChoice => Self::BranchChoice,
            cas_engine::AssumptionKind::DomainExtension => Self::DomainExtension,
        }
    }
}

impl From<AssumptionKind> for cas_engine::AssumptionKind {
    fn from(value: AssumptionKind) -> Self {
        match value {
            AssumptionKind::DerivedFromRequires => Self::DerivedFromRequires,
            AssumptionKind::RequiresIntroduced => Self::RequiresIntroduced,
            AssumptionKind::HeuristicAssumption => Self::HeuristicAssumption,
            AssumptionKind::BranchChoice => Self::BranchChoice,
            AssumptionKind::DomainExtension => Self::DomainExtension,
        }
    }
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

    /// Get the condition class for DomainMode gating.
    pub fn class(&self) -> crate::ConditionClass {
        match self {
            Self::NonZero { .. } | Self::Defined { .. } => crate::ConditionClass::Definability,
            Self::Positive { .. }
            | Self::NonNegative { .. }
            | Self::InvTrigPrincipalRange { .. }
            | Self::ComplexPrincipalBranch { .. } => crate::ConditionClass::Analytic,
        }
    }

    /// Create a Positive key from an expression.
    pub fn positive_key(ctx: &Context, expr: ExprId) -> Self {
        Self::Positive {
            expr_fingerprint: expr_fingerprint(ctx, expr),
        }
    }

    /// Create a NonZero key from an expression.
    pub fn nonzero_key(ctx: &Context, expr: ExprId) -> Self {
        Self::NonZero {
            expr_fingerprint: expr_fingerprint(ctx, expr),
        }
    }

    /// Create a NonNegative key from an expression.
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

impl From<cas_engine::AssumptionKey> for AssumptionKey {
    fn from(value: cas_engine::AssumptionKey) -> Self {
        match value {
            cas_engine::AssumptionKey::NonZero { expr_fingerprint } => {
                Self::NonZero { expr_fingerprint }
            }
            cas_engine::AssumptionKey::Positive { expr_fingerprint } => {
                Self::Positive { expr_fingerprint }
            }
            cas_engine::AssumptionKey::NonNegative { expr_fingerprint } => {
                Self::NonNegative { expr_fingerprint }
            }
            cas_engine::AssumptionKey::Defined { expr_fingerprint } => {
                Self::Defined { expr_fingerprint }
            }
            cas_engine::AssumptionKey::InvTrigPrincipalRange {
                func,
                arg_fingerprint,
            } => Self::InvTrigPrincipalRange {
                func,
                arg_fingerprint,
            },
            cas_engine::AssumptionKey::ComplexPrincipalBranch {
                func,
                arg_fingerprint,
            } => Self::ComplexPrincipalBranch {
                func,
                arg_fingerprint,
            },
        }
    }
}

impl From<AssumptionKey> for cas_engine::AssumptionKey {
    fn from(value: AssumptionKey) -> Self {
        match value {
            AssumptionKey::NonZero { expr_fingerprint } => Self::NonZero { expr_fingerprint },
            AssumptionKey::Positive { expr_fingerprint } => Self::Positive { expr_fingerprint },
            AssumptionKey::NonNegative { expr_fingerprint } => {
                Self::NonNegative { expr_fingerprint }
            }
            AssumptionKey::Defined { expr_fingerprint } => Self::Defined { expr_fingerprint },
            AssumptionKey::InvTrigPrincipalRange {
                func,
                arg_fingerprint,
            } => Self::InvTrigPrincipalRange {
                func,
                arg_fingerprint,
            },
            AssumptionKey::ComplexPrincipalBranch {
                func,
                arg_fingerprint,
            } => Self::ComplexPrincipalBranch {
                func,
                arg_fingerprint,
            },
        }
    }
}

/// One assumption event emitted during rewriting.
///
/// This is owned by `cas_solver` but keeps field types aligned with the engine
/// to preserve broad compatibility while decoupling call sites progressively.
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

    /// Converts to engine representation.
    #[inline]
    pub fn into_engine(self) -> cas_engine::AssumptionEvent {
        self.into()
    }
}

impl From<cas_engine::AssumptionEvent> for AssumptionEvent {
    fn from(value: cas_engine::AssumptionEvent) -> Self {
        Self {
            key: value.key.into(),
            expr_display: value.expr_display,
            message: value.message,
            kind: value.kind.into(),
            expr_id: value.expr_id,
        }
    }
}

impl From<AssumptionEvent> for cas_engine::AssumptionEvent {
    fn from(value: AssumptionEvent) -> Self {
        Self {
            key: value.key.into(),
            expr_display: value.expr_display,
            message: value.message,
            kind: value.kind.into(),
            expr_id: value.expr_id,
        }
    }
}

/// Deduplicating assumption collector facade.
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
    pub fn finish(self) -> Vec<crate::AssumptionRecord> {
        let mut records: Vec<crate::AssumptionRecord> = self
            .map
            .into_values()
            .map(|(event, count)| crate::AssumptionRecord {
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
pub fn collect_assumption_records(events: &[AssumptionEvent]) -> Vec<crate::AssumptionRecord> {
    collect_assumption_records_from_iter(events.iter().cloned())
}

/// Aggregate assumption events from iterator into sorted assumption records.
pub fn collect_assumption_records_from_iter<I>(events: I) -> Vec<crate::AssumptionRecord>
where
    I: IntoIterator<Item = AssumptionEvent>,
{
    let mut collector = AssumptionCollector::new();
    for event in events {
        collector.note(event);
    }
    collector.finish()
}

fn dedupe_fingerprint(key: &AssumptionKey) -> u64 {
    match key {
        AssumptionKey::NonZero { expr_fingerprint } => *expr_fingerprint,
        AssumptionKey::Positive { expr_fingerprint } => expr_fingerprint.wrapping_add(1_000_000),
        AssumptionKey::NonNegative { expr_fingerprint } => expr_fingerprint.wrapping_add(2_000_000),
        AssumptionKey::Defined { expr_fingerprint } => expr_fingerprint.wrapping_add(3_000_000),
        AssumptionKey::InvTrigPrincipalRange {
            arg_fingerprint, ..
        } => arg_fingerprint.wrapping_add(4_000_000),
        AssumptionKey::ComplexPrincipalBranch {
            arg_fingerprint, ..
        } => arg_fingerprint.wrapping_add(5_000_000),
    }
}

fn condition_text(key: &AssumptionKey, expr_display: &str) -> String {
    match key {
        AssumptionKey::NonZero { .. } => format!("{expr_display} ≠ 0"),
        AssumptionKey::Positive { .. } => format!("{expr_display} > 0"),
        AssumptionKey::NonNegative { .. } => format!("{expr_display} ≥ 0"),
        AssumptionKey::Defined { .. } => format!("{expr_display} is defined"),
        AssumptionKey::InvTrigPrincipalRange { func, .. } => {
            format!("{expr_display} in {func} principal range")
        }
        AssumptionKey::ComplexPrincipalBranch { func, .. } => {
            format!("{func}({expr_display}) principal branch")
        }
    }
}

/// Collect assumptions used from simplification steps.
///
/// Returns `(condition_text, rule_name)` items, deduplicated by
/// `(assumption_kind, expr_fingerprint)` to avoid cascades.
pub fn collect_assumed_conditions_from_steps(steps: &[crate::Step]) -> Vec<(String, String)> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();

    for step in steps {
        for event in step.assumption_events() {
            let key = AssumptionKey::from(event.key.clone());
            let fp = dedupe_fingerprint(&key);
            if seen.insert(fp) {
                result.push((
                    condition_text(&key, &event.expr_display),
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

/// Format displayable assumption events into compact single-line strings.
///
/// Output format: `"<icon> <label>: <message>"`.
pub fn format_displayable_assumption_lines(events: &[cas_engine::AssumptionEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| {
            let kind = AssumptionKind::from(event.kind);
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

/// Format one blocked hint condition as a human-readable predicate string.
pub fn format_blocked_hint_condition(ctx: &Context, hint: &crate::BlockedHint) -> String {
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
    hints: &[crate::BlockedHint],
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
    hints: &[crate::BlockedHint],
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
    domain_mode: crate::DomainMode,
    mention_analytic: bool,
) -> &'static str {
    match domain_mode {
        crate::DomainMode::Strict => "use `domain generic` or `domain assume` to allow",
        crate::DomainMode::Generic => {
            if mention_analytic {
                "use `semantics set domain assume` to allow analytic assumptions"
            } else {
                "use `semantics set domain assume` to allow"
            }
        }
        crate::DomainMode::Assume => "assumptions already enabled",
    }
}

/// Convert engine solver-assumption payload into solver-owned records.
pub fn assumption_records_from_engine(
    records: &[cas_engine::AssumptionRecord],
) -> Vec<crate::AssumptionRecord> {
    records.iter().cloned().map(Into::into).collect()
}

/// Render normalized required-conditions as REPL bullet lines.
pub fn format_normalized_condition_lines(
    ctx: &mut Context,
    conditions: &[crate::ImplicitCondition],
    debug_mode: bool,
) -> Vec<String> {
    let normalized_conditions = crate::normalize_and_dedupe_conditions(ctx, conditions);
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
    diagnostics: &cas_engine::Diagnostics,
    result_expr: Option<ExprId>,
    display_level: crate::RequiresDisplayLevel,
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

/// Render required-conditions as plain display lines using a line prefix.
pub fn format_required_condition_lines(
    ctx: &Context,
    conditions: &[crate::ImplicitCondition],
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
    warnings: &[crate::DomainWarning],
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
pub fn format_blocked_hint_lines(hints: &[crate::BlockedHint], line_prefix: &str) -> Vec<String> {
    hints
        .iter()
        .map(|hint| format!("{line_prefix}{} (hint: {})", hint.rule, hint.suggestion))
        .collect()
}

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

/// Render a full blocked-simplifications section for CLI output.
pub fn format_blocked_simplifications_section_lines(
    ctx: &Context,
    hints: &[crate::BlockedHint],
    domain_mode: crate::DomainMode,
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

fn assumption_record_summary_item(record: &crate::AssumptionRecord) -> String {
    if record.count > 1 {
        format!("{}({}) (×{})", record.kind, record.expr, record.count)
    } else {
        format!("{}({})", record.kind, record.expr)
    }
}

fn assumption_record_condition(record: &crate::AssumptionRecord) -> String {
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
pub fn format_assumption_records_summary(records: &[crate::AssumptionRecord]) -> Option<String> {
    if records.is_empty() {
        return None;
    }
    let items: Vec<String> = records.iter().map(assumption_record_summary_item).collect();
    Some(items.join(", "))
}

/// Format assumptions as condition strings for debug explain blocks.
///
/// Returned values are sorted and deduplicated for stable output.
pub fn format_assumption_records_conditions(records: &[crate::AssumptionRecord]) -> Vec<String> {
    let mut items: Vec<String> = records.iter().map(assumption_record_condition).collect();
    items.sort();
    items.dedup();
    items
}

/// Format an assumptions section from aggregated assumption records.
pub fn format_assumption_records_section_lines(
    records: &[crate::AssumptionRecord],
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

/// Rendering config for solve assumption/blocked sections.
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

/// Format a simple requires section from textual conditions.
pub fn format_text_requires_lines(requires: &[String]) -> Vec<String> {
    if requires.is_empty() {
        return Vec::new();
    }

    let mut lines = vec!["ℹ️ Requires:".to_string()];
    for req in requires {
        lines.push(format!("  • {req}"));
    }
    lines
}

/// Labels and prefixes for eval metadata section rendering.
#[derive(Debug, Clone, Copy)]
pub struct EvalMetadataSectionLabels<'a> {
    pub required_header: &'a str,
    pub assumed_header: &'a str,
    pub blocked_header: &'a str,
    pub line_prefix: &'a str,
}

/// Render standard eval metadata sections for CLI/UI:
/// `Requires`, `Assumed` and `Blocked`.
pub fn format_eval_metadata_sections(
    ctx: &Context,
    required_conditions: &[crate::ImplicitCondition],
    domain_warnings: &[crate::DomainWarning],
    blocked_hints: &[crate::BlockedHint],
    labels: EvalMetadataSectionLabels<'_>,
) -> Vec<String> {
    let mut lines = Vec::new();

    if !required_conditions.is_empty() {
        lines.push(labels.required_header.to_string());
        lines.extend(format_required_condition_lines(
            ctx,
            required_conditions,
            labels.line_prefix,
        ));
    }

    if !domain_warnings.is_empty() {
        lines.push(labels.assumed_header.to_string());
        lines.extend(format_domain_warning_lines(
            domain_warnings,
            false,
            labels.line_prefix,
        ));
    }

    if !blocked_hints.is_empty() {
        lines.push(labels.blocked_header.to_string());
        lines.extend(format_blocked_hint_lines(blocked_hints, labels.line_prefix));
    }

    lines
}

/// Map one solver-core logarithmic assumption into a solver assumption event.
pub(crate) fn assumption_event_from_log_assumption(
    ctx: &Context,
    assumption: cas_solver_core::log_domain::LogAssumption,
    base: ExprId,
    rhs: ExprId,
) -> AssumptionEvent {
    cas_solver_core::log_assumptions::map_log_assumption_target_with(
        ctx,
        assumption,
        base,
        rhs,
        AssumptionEvent::positive,
    )
}

/// Convert one blocked log hint from solver-core into the engine-compatible payload.
pub(crate) fn map_log_blocked_hint(
    ctx: &Context,
    hint: cas_solver_core::solve_outcome::LogBlockedHintRecord,
) -> crate::BlockedHint {
    let mapped = cas_solver_core::log_assumptions::map_log_blocked_hint_with(
        ctx,
        hint,
        AssumptionEvent::positive,
    );
    crate::BlockedHint {
        key: mapped.event.key.into(),
        expr_id: mapped.expr_id,
        rule: mapped.rule.to_string(),
        suggestion: mapped.suggestion,
    }
}

/// Convert and register one blocked log hint in the global blocked-hints registry.
pub(crate) fn register_log_blocked_hint(
    ctx: &Context,
    hint: cas_solver_core::solve_outcome::LogBlockedHintRecord,
) {
    crate::register_blocked_hint(map_log_blocked_hint(ctx, hint));
}

/// Classify one assumption event against domain context.
pub fn classify_assumption(
    ctx: &Context,
    dc: &crate::DomainContext,
    event: &AssumptionEvent,
) -> (AssumptionKind, Option<crate::ImplicitCondition>) {
    match event.kind {
        AssumptionKind::BranchChoice | AssumptionKind::DomainExtension => {
            return (event.kind, None);
        }
        _ => {}
    }

    match assumption_to_condition(event) {
        Some(cond) => {
            if dc.is_condition_implied(ctx, &cond) {
                (AssumptionKind::DerivedFromRequires, None)
            } else {
                match event.kind {
                    AssumptionKind::HeuristicAssumption => {
                        (AssumptionKind::HeuristicAssumption, None)
                    }
                    _ => (AssumptionKind::RequiresIntroduced, Some(cond)),
                }
            }
        }
        None => (event.kind, None),
    }
}

fn assumption_to_condition(event: &AssumptionEvent) -> Option<crate::ImplicitCondition> {
    let expr_id = event.expr_id?;

    match &event.key {
        AssumptionKey::NonZero { .. } => Some(crate::ImplicitCondition::NonZero(expr_id)),
        AssumptionKey::Positive { .. } => Some(crate::ImplicitCondition::Positive(expr_id)),
        AssumptionKey::NonNegative { .. } => Some(crate::ImplicitCondition::NonNegative(expr_id)),
        AssumptionKey::Defined { .. } => None,
        AssumptionKey::InvTrigPrincipalRange { .. } => None,
        AssumptionKey::ComplexPrincipalBranch { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{format_solve_assumption_and_blocked_sections, SolveAssumptionSectionConfig};

    #[test]
    fn format_solve_assumption_and_blocked_sections_empty_when_no_data() {
        let ctx = cas_ast::Context::new();
        let lines = format_solve_assumption_and_blocked_sections(
            &ctx,
            &[],
            &[],
            SolveAssumptionSectionConfig {
                debug_mode: false,
                hints_enabled: false,
                domain_mode: crate::DomainMode::Generic,
            },
        );
        assert!(lines.is_empty());
    }

    #[test]
    fn format_solve_assumption_and_blocked_sections_includes_assumptions_in_debug_mode() {
        let ctx = cas_ast::Context::new();
        let assumptions = vec![crate::AssumptionRecord {
            kind: "nonzero".to_string(),
            expr: "x".to_string(),
            message: "x != 0".to_string(),
            count: 1,
        }];
        let lines = format_solve_assumption_and_blocked_sections(
            &ctx,
            &assumptions,
            &[],
            SolveAssumptionSectionConfig {
                debug_mode: true,
                hints_enabled: false,
                domain_mode: crate::DomainMode::Generic,
            },
        );
        assert!(lines.iter().any(|line| line.contains("Assumptions used")));
        assert!(lines.iter().any(|line| line.contains("x ≠ 0")));
    }
}
