use cas_ast::{Context, ExprId};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet, HashMap};
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

fn assumption_kind_from_engine(value: cas_engine::AssumptionKind) -> AssumptionKind {
    match value {
        cas_engine::AssumptionKind::DerivedFromRequires => AssumptionKind::DerivedFromRequires,
        cas_engine::AssumptionKind::RequiresIntroduced => AssumptionKind::RequiresIntroduced,
        cas_engine::AssumptionKind::HeuristicAssumption => AssumptionKind::HeuristicAssumption,
        cas_engine::AssumptionKind::BranchChoice => AssumptionKind::BranchChoice,
        cas_engine::AssumptionKind::DomainExtension => AssumptionKind::DomainExtension,
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

pub(crate) fn assumption_key_from_engine(value: cas_engine::AssumptionKey) -> AssumptionKey {
    match value {
        cas_engine::AssumptionKey::NonZero { expr_fingerprint } => {
            AssumptionKey::NonZero { expr_fingerprint }
        }
        cas_engine::AssumptionKey::Positive { expr_fingerprint } => {
            AssumptionKey::Positive { expr_fingerprint }
        }
        cas_engine::AssumptionKey::NonNegative { expr_fingerprint } => {
            AssumptionKey::NonNegative { expr_fingerprint }
        }
        cas_engine::AssumptionKey::Defined { expr_fingerprint } => {
            AssumptionKey::Defined { expr_fingerprint }
        }
        cas_engine::AssumptionKey::InvTrigPrincipalRange {
            func,
            arg_fingerprint,
        } => AssumptionKey::InvTrigPrincipalRange {
            func,
            arg_fingerprint,
        },
        cas_engine::AssumptionKey::ComplexPrincipalBranch {
            func,
            arg_fingerprint,
        } => AssumptionKey::ComplexPrincipalBranch {
            func,
            arg_fingerprint,
        },
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
}

fn assumption_event_from_engine(value: cas_engine::AssumptionEvent) -> AssumptionEvent {
    AssumptionEvent {
        key: assumption_key_from_engine(value.key),
        expr_display: value.expr_display,
        message: value.message,
        kind: assumption_kind_from_engine(value.kind),
        expr_id: value.expr_id,
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
pub(crate) fn assumption_records_from_engine(
    records: &[cas_engine::AssumptionRecord],
) -> Vec<crate::AssumptionRecord> {
    records
        .iter()
        .cloned()
        .map(crate::assumption_types::assumption_record_from_engine)
        .collect()
}

/// Convert engine step-assumption events into solver-owned events.
pub(crate) fn assumption_events_from_engine(
    events: &[cas_engine::AssumptionEvent],
) -> Vec<crate::AssumptionEvent> {
    events
        .iter()
        .cloned()
        .map(assumption_event_from_engine)
        .collect()
}

/// Convert assumption events from one step into solver-owned events.
pub fn assumption_events_from_step(step: &crate::Step) -> Vec<crate::AssumptionEvent> {
    assumption_events_from_engine(step.assumption_events())
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

fn assumption_record_condition(record: &crate::AssumptionRecord) -> String {
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
        key: mapped.event.key,
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

pub(crate) fn assumption_key_to_engine(key: AssumptionKey) -> cas_engine::AssumptionKey {
    match key {
        AssumptionKey::NonZero { expr_fingerprint } => {
            cas_engine::AssumptionKey::NonZero { expr_fingerprint }
        }
        AssumptionKey::Positive { expr_fingerprint } => {
            cas_engine::AssumptionKey::Positive { expr_fingerprint }
        }
        AssumptionKey::NonNegative { expr_fingerprint } => {
            cas_engine::AssumptionKey::NonNegative { expr_fingerprint }
        }
        AssumptionKey::Defined { expr_fingerprint } => {
            cas_engine::AssumptionKey::Defined { expr_fingerprint }
        }
        AssumptionKey::InvTrigPrincipalRange {
            func,
            arg_fingerprint,
        } => cas_engine::AssumptionKey::InvTrigPrincipalRange {
            func,
            arg_fingerprint,
        },
        AssumptionKey::ComplexPrincipalBranch {
            func,
            arg_fingerprint,
        } => cas_engine::AssumptionKey::ComplexPrincipalBranch {
            func,
            arg_fingerprint,
        },
    }
}
