//! Evaluation engine: `Engine.eval()` entry point and supporting types.
//!
//! This module is the top-level orchestrator that takes a parsed expression,
//! resolves session references, dispatches to the appropriate action handler
//! (simplify, solve, expand, equiv, limit), assembles diagnostics, and
//! returns the final `EvalOutput`.

mod diagnostics;
mod dispatch;

/// Result type for individual action handlers in `Engine.eval()`.
///
/// Fields: (result, domain_warnings, steps, solve_steps, solver_assumptions, output_scopes, solver_required)
pub(crate) type ActionResult = (
    EvalResult,
    Vec<DomainWarning>,
    Vec<crate::Step>,
    Vec<crate::solver::SolveStep>,
    Vec<crate::assumptions::AssumptionRecord>,
    Vec<cas_ast::display_transforms::ScopeTag>,
    Vec<crate::implicit_domain::ImplicitCondition>,
);

use crate::Simplifier;
use cas_ast::{BuiltinFn, Equation, Expr, ExprId, RelOp};

pub(crate) type CoreSimplifyCacheKey =
    cas_session_core::cache::SimplifyCacheKey<crate::domain::DomainMode>;
pub(crate) type CoreSimplifiedCache = cas_session_core::cache::SimplifiedCache<
    crate::domain::DomainMode,
    crate::diagnostics::RequiredItem,
    crate::step::Step,
>;
pub(crate) type CoreCacheHitTrace =
    cas_session_core::cache::CacheHitTrace<crate::diagnostics::RequiredItem>;

/// Engine-local cache key used by eval/session abstraction.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SimplifyCacheKey {
    pub domain: crate::domain::DomainMode,
    pub ruleset_rev: u64,
}

impl SimplifyCacheKey {
    pub fn from_context(domain: crate::domain::DomainMode) -> Self {
        Self {
            domain,
            ruleset_rev: 1,
        }
    }
}

impl From<SimplifyCacheKey> for CoreSimplifyCacheKey {
    fn from(value: SimplifyCacheKey) -> Self {
        Self {
            domain: value.domain,
            ruleset_rev: value.ruleset_rev,
        }
    }
}

/// Engine-local cached simplification payload.
#[derive(Debug, Clone)]
pub struct SimplifiedCache {
    pub key: SimplifyCacheKey,
    pub expr: ExprId,
    pub requires: Vec<crate::diagnostics::RequiredItem>,
    pub steps: Option<std::sync::Arc<Vec<crate::step::Step>>>,
}

impl SimplifiedCache {
    pub(crate) fn into_core(self) -> CoreSimplifiedCache {
        CoreSimplifiedCache {
            key: self.key.into(),
            expr: self.expr,
            requires: self.requires,
            steps: self.steps,
        }
    }
}

/// Engine-local cache hit trace used by eval/session abstraction.
#[derive(Debug, Clone)]
pub struct CacheHitTrace {
    pub entry_id: u64,
    pub before_ref_expr: ExprId,
    pub after_expr: ExprId,
    pub requires: Vec<crate::diagnostics::RequiredItem>,
}

impl From<CoreCacheHitTrace> for CacheHitTrace {
    fn from(value: CoreCacheHitTrace) -> Self {
        Self {
            entry_id: value.entry_id,
            before_ref_expr: value.before_ref_expr,
            after_expr: value.after_expr,
            requires: value.requires,
        }
    }
}

/// Engine-level session/reference resolution error.
///
/// This keeps `EvalSession` decoupled from any concrete session implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalResolveError {
    NotFound(u64),
    CircularReference(u64),
}

impl std::fmt::Display for EvalResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "Session reference #{} not found", id),
            Self::CircularReference(id) => {
                write!(f, "Circular session reference detected at #{}", id)
            }
        }
    }
}

impl std::error::Error for EvalResolveError {}

/// The central Engine struct that wraps the core Simplifier and potentially other components.
///
/// # Example
///
/// ```
/// use cas_engine::Engine;
/// use cas_parser::parse;
///
/// let mut engine = Engine::new();
/// let expr = parse("x + x", &mut engine.simplifier.context).unwrap();
/// let (result, _steps) = engine.simplifier.simplify(expr);
///
/// // Result is simplified
/// use cas_ast::display::DisplayExpr;
/// let output = format!("{}", DisplayExpr { context: &engine.simplifier.context, id: result });
/// assert!(output.contains("x")); // Contains x
/// ```
pub struct Engine {
    pub simplifier: Simplifier,
}

/// Session abstraction for `Engine::eval`, allowing callers to provide any
/// state container that exposes the required eval components.
pub trait EvalStore {
    fn push_raw_input(&mut self, ctx: &cas_ast::Context, parsed: ExprId, raw_input: String) -> u64;
    fn touch_cached(&mut self, entry_id: u64);
    fn update_diagnostics(&mut self, id: u64, diagnostics: crate::diagnostics::Diagnostics);
    fn update_simplified(&mut self, id: u64, cache: SimplifiedCache);
}

impl EvalStore
    for cas_session_core::store::SessionStore<crate::diagnostics::Diagnostics, CoreSimplifiedCache>
{
    fn push_raw_input(&mut self, ctx: &cas_ast::Context, parsed: ExprId, raw_input: String) -> u64 {
        let kind = if let Some((lhs, rhs)) = cas_ast::eq::unwrap_eq(ctx, parsed) {
            cas_session_core::types::EntryKind::Eq { lhs, rhs }
        } else {
            cas_session_core::types::EntryKind::Expr(parsed)
        };
        self.push(kind, raw_input)
    }

    fn touch_cached(&mut self, entry_id: u64) {
        self.touch_cached(entry_id);
    }

    fn update_diagnostics(&mut self, id: u64, diagnostics: crate::diagnostics::Diagnostics) {
        self.update_diagnostics(id, diagnostics);
    }

    fn update_simplified(&mut self, id: u64, cache: SimplifiedCache) {
        self.update_simplified(id, cache.into_core());
    }
}

pub trait EvalSession {
    type Store: EvalStore;

    fn store_mut(&mut self) -> &mut Self::Store;
    fn options(&self) -> &crate::options::EvalOptions;
    fn profile_cache_mut(&mut self) -> &mut crate::profile_cache::ProfileCache;

    fn resolve_all(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<ExprId, EvalResolveError>;

    fn resolve_all_with_diagnostics(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
    ) -> Result<(ExprId, crate::diagnostics::Diagnostics, Vec<CacheHitTrace>), EvalResolveError>;
}

impl Engine {
    /// Create a new Engine with default rules.
    ///
    /// # Example
    ///
    /// ```
    /// use cas_engine::Engine;
    ///
    /// let engine = Engine::new();
    /// // Engine is ready to simplify expressions
    /// ```
    pub fn new() -> Self {
        Self {
            simplifier: Simplifier::with_default_rules(),
        }
    }

    /// Create an Engine with a pre-populated Context (for session restoration).
    pub fn with_context(context: cas_ast::Context) -> Self {
        Self {
            simplifier: Simplifier::with_context(context),
        }
    }

    /// Determine effective options, resolving Auto modes based on expression content.
    /// - ContextMode::Auto → IntegratePrep if contains integrate(), else Standard
    /// - ComplexMode::Auto → On if contains i, else Off
    /// - expand_policy forced Off in Solve mode (anti-contamination)
    fn effective_options(
        &self,
        opts: &crate::options::EvalOptions,
        expr: ExprId,
    ) -> crate::options::EvalOptions {
        use crate::options::{ComplexMode, ContextMode};
        use crate::phase::ExpandPolicy;

        let mut effective = opts.clone();

        // Resolve ContextMode::Auto
        if opts.shared.context_mode == ContextMode::Auto {
            if crate::helpers::contains_integral(&self.simplifier.context, expr) {
                effective.shared.context_mode = ContextMode::IntegratePrep;
            } else {
                effective.shared.context_mode = ContextMode::Standard;
            }
        }

        // Resolve ComplexMode::Auto
        if opts.complex_mode == ComplexMode::Auto {
            if crate::helpers::contains_i(&self.simplifier.context, expr) {
                effective.complex_mode = ComplexMode::On;
            } else {
                effective.complex_mode = ComplexMode::Off;
            }
        }

        // CRITICAL: Force expand_policy Off in Solve mode
        // Auto-expansion can interfere with equation solving strategies
        if effective.shared.context_mode == ContextMode::Solve {
            effective.shared.expand_policy = ExpandPolicy::Off;
        }

        effective
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub enum EvalAction {
    Simplify,
    Expand,
    // Solve for a variable.
    Solve {
        var: String,
    },
    // Check equivalence between two expressions
    Equiv {
        other: ExprId,
    },
    // Compute limit as variable approaches a value
    Limit {
        var: String,
        approach: crate::limits::Approach,
    },
}

#[derive(Clone, Debug)]
pub struct EvalRequest {
    pub raw_input: String,
    pub parsed: ExprId,
    pub action: EvalAction,
    pub auto_store: bool,
}

#[derive(Clone, Debug)]
pub enum EvalResult {
    Expr(ExprId),
    Set(Vec<ExprId>),                  // For Solve multiple roots (legacy)
    SolutionSet(cas_ast::SolutionSet), // V2.0: Full solution set including Conditional
    Bool(bool),                        // For Equiv
    None,                              // For commands with no output
}

/// A domain assumption warning with its source rule.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DomainWarning {
    pub message: String,
    pub rule_name: String,
}

#[derive(Clone, Debug)]
pub struct EvalOutput {
    pub stored_id: Option<u64>,
    pub parsed: ExprId,
    pub resolved: ExprId,
    pub result: EvalResult,
    /// Domain warnings with deduplication and rule source.
    pub domain_warnings: Vec<DomainWarning>,
    pub steps: crate::step::DisplayEvalSteps,
    pub solve_steps: Vec<crate::solver::SolveStep>,
    /// Assumptions made during solver operations (for Assume mode).
    pub solver_assumptions: Vec<crate::assumptions::AssumptionRecord>,
    /// Scopes for context-aware display (e.g., QuadraticFormula -> sqrt display).
    pub output_scopes: Vec<cas_ast::display_transforms::ScopeTag>,
    /// Required conditions for validity - implicit domain constraints from input.
    /// NOT assumptions! These were already required by the input expression.
    /// Sorted and deduplicated for stable display.
    pub required_conditions: Vec<crate::implicit_domain::ImplicitCondition>,
    /// Blocked hints - operations that could not proceed in Strict/Generic mode.
    /// These suggest using Assume mode to enable certain transformations.
    pub blocked_hints: Vec<crate::domain::BlockedHint>,
    /// V2.2+: Unified diagnostics with origin tracking.
    /// Consolidates requires, assumed, and blocked in one container.
    pub diagnostics: crate::diagnostics::Diagnostics,
}

/// Collect domain warnings from steps with deduplication.
/// Collects structured assumption_events from each step.
/// Note: Only events that are NOT RequiresIntroduced become DomainWarnings (⚠️).
/// RequiresIntroduced events are displayed in steps with ℹ️ icon instead.
pub(crate) fn collect_domain_warnings(steps: &[crate::Step]) -> Vec<DomainWarning> {
    use std::collections::HashSet;

    let mut seen: HashSet<String> = HashSet::new();
    let mut warnings = Vec::new();

    for step in steps {
        // Collect structured assumption_events
        for event in step.assumption_events() {
            // Skip RequiresIntroduced - these show in steps with ℹ️ icon, not as ⚠️ warnings
            // Skip DerivedFromRequires - these are implied by existing requires, don't show
            if matches!(
                event.kind,
                crate::assumptions::AssumptionKind::RequiresIntroduced
                    | crate::assumptions::AssumptionKind::DerivedFromRequires
            ) {
                continue;
            }
            let msg_str = event.message.clone();
            if !seen.contains(&msg_str) {
                seen.insert(msg_str.clone());
                warnings.push(DomainWarning {
                    message: msg_str,
                    rule_name: step.rule_name.clone(),
                });
            }
        }
    }

    warnings
}

/// V2.15.36: Build a synthetic timeline step showing cache hits.
///
/// Creates a single aggregated step when `#N` references were resolved
/// from cache. This provides traceability ("Used cached result from #1, #3")
/// without repeating the full derivation steps.
pub(crate) fn build_cache_hit_step(
    ctx: &cas_ast::Context,
    original_expr: cas_ast::ExprId,
    resolved_expr: cas_ast::ExprId,
    cache_hits: &[CacheHitTrace],
) -> Option<crate::Step> {
    if cache_hits.is_empty() {
        return None;
    }

    // Collect and sort entry IDs for deterministic output
    let mut ids: Vec<u64> = cache_hits.iter().map(|h| h.entry_id).collect();
    ids.sort();

    // Format the description with truncation for readability
    let shown: Vec<String> = ids.iter().take(6).map(|id| format!("#{}", id)).collect();
    let suffix = if ids.len() > 6 {
        format!(" (+{})", ids.len() - 6)
    } else {
        String::new()
    };

    let description = format!(
        "Used cached simplified result from {}{}",
        shown.join(", "),
        suffix
    );

    let mut step = crate::Step::new(
        &description,        // label
        "Use cached result", // rule_name
        original_expr,       // before: the original parsed expression with #N
        resolved_expr,       // after: with #N replaced by cached simplified result
        Vec::new(),          // child_steps
        Some(ctx),           // context for display
    );
    // V2.15.36: Set to Medium so it appears in the timeline
    step.importance = crate::step::ImportanceLevel::Medium;
    step.category = crate::step::StepCategory::Substitute;
    Some(step)
}
