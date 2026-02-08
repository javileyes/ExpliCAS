use crate::parent_context::ParentContext;
use crate::phase::PhaseMask;
use crate::step::{ImportanceLevel, StepCategory};
use crate::target_kind::TargetKindSet;
use cas_ast::{Context, ExprId};
use std::borrow::Cow;
use std::cell::Cell;

// =============================================================================
// Steps-enabled thread-local flag
// =============================================================================
//
// When steps_mode == Off, rules still construct Rewrite objects with .desc().
// This flag gates description computation so that format!() closures in
// desc_lazy() are never evaluated, and desc() drops incoming Strings immediately.
// Set by the engine in orchestration.rs before each simplification pass.

thread_local! {
    static STEPS_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Set whether step descriptions should be computed.
/// Called by the engine before simplification passes.
pub fn set_steps_enabled(enabled: bool) {
    STEPS_ENABLED.with(|s| s.set(enabled));
}

/// Check if step descriptions are enabled.
#[inline]
pub fn steps_enabled() -> bool {
    STEPS_ENABLED.with(|s| s.get())
}

// =============================================================================
// SoundnessLabel: Mathematical soundness classification for Rule transformations
// =============================================================================

/// Mathematical soundness classification for rules.
///
/// This enum classifies how a rule transformation relates to mathematical equivalence,
/// used for:
/// - Auditing rules during review
/// - Property-based testing (only Equivalence* rules should preserve numeric value)
/// - Per-step numeric validation in debug mode
/// - UX feedback (showing assumption types to users)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SoundnessLabel {
    /// Equivalence for all values in the implicit domain of the input.
    /// The transformation preserves mathematical identity.
    /// Example: x + 0 → x, sin²(x) + cos²(x) → 1
    #[default]
    Equivalence,

    /// Equivalence, but requires additional conditions (typically RequiresIntroduced).
    /// Example: log(ab) → log(a) + log(b) (requires a > 0, b > 0)
    EquivalenceUnderIntroducedRequires,

    /// The rule chooses one branch of a multi-valued expression (principal branch).
    /// Example: sqrt(x²) → x (choosing positive root), arcsin(sin(x)) → x
    BranchChoice,

    /// Extends the domain (R → C, or similar).
    /// Example: (-1)^(1/2) → i (moving from real to complex)
    DomainExtension,

    /// Heuristic: aims to "improve" the expression but doesn't guarantee
    /// global equivalence. Used for simplification strategies.
    /// Example: auto-expand with budget decisions
    Heuristic,
}

// =============================================================================
// ChainedRewrite: A subsequent transformation following the main Rewrite
// =============================================================================

/// A chained transformation following the main Rewrite.
///
/// The engine constructs Steps from this with correct before/after:
/// - Main Rewrite: before=original, after=rewrite.new_expr
/// - ChainedRewrite[0]: before=rewrite.new_expr, after=chain[0].after
/// - ChainedRewrite[n]: before=chain[n-1].after, after=chain[n].after
///
/// This guarantees sequential coherence: each step's `after` equals the next step's `before`.
#[derive(Debug, Clone)]
pub struct ChainedRewrite {
    /// The result expression after this transformation
    pub after: ExprId,
    /// Human-readable description of this step
    pub description: Cow<'static, str>,
    /// Optional: Focus the "Rule:" line on this sub-expression
    pub before_local: Option<ExprId>,
    /// Optional: Focus the "Rule:" line on this result
    pub after_local: Option<ExprId>,
    /// Required conditions for this step
    pub required_conditions: Vec<crate::implicit_domain::ImplicitCondition>,
    /// Assumptions made by this step
    pub assumption_events: smallvec::SmallVec<[crate::assumptions::AssumptionEvent; 1]>,
    /// Optional polynomial proof data
    pub poly_proof: Option<crate::multipoly_display::PolynomialProofData>,
    /// Optional importance override (defaults to rule's importance)
    pub importance: Option<ImportanceLevel>,
}

impl ChainedRewrite {
    /// Create a new ChainedRewrite with the given result expression.
    #[must_use]
    pub fn new(after: ExprId) -> Self {
        Self {
            after,
            description: Cow::Borrowed(""),
            before_local: None,
            after_local: None,
            required_conditions: vec![],
            assumption_events: Default::default(),
            poly_proof: None,
            importance: None,
        }
    }

    /// Set the description of this chained step.
    /// Gated: when steps are disabled, skips storage to avoid overhead.
    #[must_use]
    pub fn desc(mut self, description: impl Into<Cow<'static, str>>) -> Self {
        if steps_enabled() {
            self.description = description.into();
        }
        self
    }

    /// Lazy description — closure is only evaluated when steps mode is on.
    /// Use instead of `.desc(format!(...))` to avoid heap allocation when steps are off.
    #[must_use]
    pub fn desc_lazy(mut self, f: impl FnOnce() -> String) -> Self {
        if steps_enabled() {
            self.description = Cow::Owned(f());
        }
        self
    }

    /// Set explicit local before/after expressions for the "Rule:" line.
    #[must_use]
    pub fn local(mut self, before: ExprId, after: ExprId) -> Self {
        self.before_local = Some(before);
        self.after_local = Some(after);
        self
    }

    /// Set importance level (overrides rule's default).
    #[must_use]
    pub fn importance(mut self, level: ImportanceLevel) -> Self {
        self.importance = Some(level);
        self
    }

    /// Add a required condition.
    #[must_use]
    pub fn requires(mut self, cond: crate::implicit_domain::ImplicitCondition) -> Self {
        self.required_conditions.push(cond);
        self
    }

    /// Add an assumption event.
    #[must_use]
    pub fn assume(mut self, ev: crate::assumptions::AssumptionEvent) -> Self {
        self.assumption_events.push(ev);
        self
    }
}

// =============================================================================
// Rewrite: Result of a rule application
// =============================================================================

/// Result of a rule application containing the new expression and metadata
pub struct Rewrite {
    /// The transformed expression
    pub new_expr: ExprId,
    /// Human-readable description of the transformation
    pub description: Cow<'static, str>,
    /// Optional: The specific local expression before the rule (for n-ary rules)
    /// If set, CLI uses this for "Rule: before -> after" instead of full expression
    pub before_local: Option<ExprId>,
    /// Optional: The specific local result after the rule (for n-ary rules)
    pub after_local: Option<ExprId>,
    /// Optional: Domain assumption used by this rule (e.g., "x > 0 for ln(x)")
    /// LEGACY: use assumption_events for structured emission, this is fallback.
    /// Structured assumption events (preferred over domain_assumption string)
    /// Multiple events allowed for rules that make several assumptions.
    pub assumption_events: smallvec::SmallVec<[crate::assumptions::AssumptionEvent; 1]>,
    /// Required conditions for validity (implicit domain preservation) - NOT assumptions!
    /// These are conditions that were already implicitly required by the input expression.
    /// Used when a rewrite makes implicit domain constraints explicit (e.g., sqrt(x)^2 → x requires x ≥ 0).
    pub required_conditions: Vec<crate::implicit_domain::ImplicitCondition>,
    /// Optional: Polynomial proof data for identity cancellation (PolyZero airbag)
    /// Used for didactic display of the normalization process
    pub poly_proof: Option<crate::multipoly_display::PolynomialProofData>,
    /// Chained sequential rewrites following this one.
    /// Engine processes these in order, constructing Steps with correct before/after.
    /// See `ChainedRewrite` documentation for the sequential coherence guarantee.
    pub chained: Vec<ChainedRewrite>,
    /// V2.14.45: Educational sub-steps explaining rule application.
    /// These are metadata only - don't participate in the rewrite loop.
    /// Propagated to Step.substeps during step creation.
    pub substeps: Vec<crate::step::SubStep>,
}

impl Default for Rewrite {
    fn default() -> Self {
        Self {
            new_expr: cas_ast::ExprId::from_raw(0), // Will be set by new()
            description: Cow::Borrowed(""),
            before_local: None,
            after_local: None,
            assumption_events: Default::default(),
            required_conditions: vec![],
            poly_proof: None,
            chained: vec![],
            substeps: vec![],
        }
    }
}

impl Rewrite {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Create a new Rewrite with the given result expression.
    /// Use fluent setters to add description and other fields.
    ///
    /// # Example
    /// ```ignore
    /// Some(Rewrite::new(result).desc("Simplify"))
    /// ```
    #[must_use]
    pub fn new(new_expr: ExprId) -> Self {
        Self {
            new_expr,
            ..Default::default()
        }
    }

    /// Create a simple rewrite (most common case - local transform = global transform)
    #[must_use]
    pub fn simple(new_expr: ExprId, description: impl Into<Cow<'static, str>>) -> Self {
        Self::new(new_expr).desc(description)
    }

    /// Create a rewrite with explicit local before/after (for n-ary rules)
    /// Use when the rule transforms a subpattern within a larger expression
    #[must_use]
    pub fn with_local(
        new_expr: ExprId,
        description: impl Into<Cow<'static, str>>,
        before_local: ExprId,
        after_local: ExprId,
    ) -> Self {
        Self::new(new_expr)
            .desc(description)
            .local(before_local, after_local)
    }

    /// Create a rewrite with domain assumption warning
    /// DEPRECATED: Use `.assume()` fluent method instead
    #[must_use]
    pub fn with_domain_assumption(
        new_expr: ExprId,
        description: impl Into<Cow<'static, str>>,
        _assumption: &'static str,
    ) -> Self {
        Self::new(new_expr).desc(description)
    }

    /// Create a rewrite with polynomial proof data (for PolyZero airbag)
    /// Used by PolynomialIdentityZeroRule to attach normalization info for didactic display
    #[must_use]
    pub fn with_poly_proof(
        new_expr: ExprId,
        description: impl Into<Cow<'static, str>>,
        poly_proof: crate::multipoly_display::PolynomialProofData,
    ) -> Self {
        Self::new(new_expr).desc(description).poly_proof(poly_proof)
    }

    // =========================================================================
    // Fluent Setters (Builder Pattern)
    // =========================================================================

    /// Set the description of this rewrite.
    /// Gated: when steps are disabled, skips storage to avoid overhead.
    #[must_use]
    pub fn desc(mut self, description: impl Into<Cow<'static, str>>) -> Self {
        if steps_enabled() {
            self.description = description.into();
        }
        self
    }

    /// Lazy description — closure is only evaluated when steps mode is on.
    /// Use instead of `.desc(format!(...))` to avoid heap allocation when steps are off.
    #[must_use]
    pub fn desc_lazy(mut self, f: impl FnOnce() -> String) -> Self {
        if steps_enabled() {
            self.description = Cow::Owned(f());
        }
        self
    }

    /// Set explicit local before/after expressions (for n-ary rules).
    /// Use when the rule transforms a subpattern within a larger expression.
    #[must_use]
    pub fn local(mut self, before: ExprId, after: ExprId) -> Self {
        self.before_local = Some(before);
        self.after_local = Some(after);
        self
    }

    /// Add a required condition for validity.
    /// These are conditions that were already implicitly required by the input expression.
    #[must_use]
    pub fn requires(mut self, cond: crate::implicit_domain::ImplicitCondition) -> Self {
        self.required_conditions.push(cond);
        self
    }

    /// Add multiple required conditions for validity.
    #[must_use]
    pub fn requires_all<I>(mut self, conds: I) -> Self
    where
        I: IntoIterator<Item = crate::implicit_domain::ImplicitCondition>,
    {
        self.required_conditions.extend(conds);
        self
    }

    /// Add an assumption event (structured domain assumption).
    #[must_use]
    pub fn assume(mut self, ev: crate::assumptions::AssumptionEvent) -> Self {
        self.assumption_events.push(ev);
        self
    }

    /// Add multiple assumption events.
    #[must_use]
    pub fn assume_all<I>(mut self, evs: I) -> Self
    where
        I: IntoIterator<Item = crate::assumptions::AssumptionEvent>,
    {
        self.assumption_events.extend(evs);
        self
    }

    /// Set polynomial proof data (for PolyZero airbag didactic display).
    #[must_use]
    pub fn poly_proof(mut self, proof: crate::multipoly_display::PolynomialProofData) -> Self {
        self.poly_proof = Some(proof);
        self
    }

    /// Add a chained sequential rewrite.
    /// The engine will process these in order, creating separate Steps with correct before/after.
    /// Example: Factor step followed by Cancel step.
    #[must_use]
    pub fn chain(mut self, r: ChainedRewrite) -> Self {
        self.chained.push(r);
        self
    }

    /// Add an educational sub-step explaining rule application (V2.14.45).
    /// These are metadata only - don't participate in the rewrite loop.
    ///
    /// # Example
    /// ```ignore
    /// Rewrite::new(result)
    ///     .desc("Triple Tangent Product")
    ///     .substep("Pattern Recognition", vec![
    ///         "Detected 3 tan(·) factors".to_string(),
    ///     ])
    /// ```
    #[must_use]
    pub fn substep(mut self, title: impl Into<String>, lines: Vec<String>) -> Self {
        self.substeps.push(crate::step::SubStep::new(title, lines));
        self
    }
}

/// Simplified Rule trait for backward compatibility
/// Most rules should implement this for simplicity
pub trait SimpleRule {
    fn name(&self) -> &str;

    /// Apply rule without parent context (legacy API).
    /// Most rules implement only this method.
    fn apply_simple(&self, context: &mut Context, expr: ExprId) -> Option<Rewrite>;

    /// Apply rule with parent context for domain-aware rules.
    ///
    /// Override this method in rules that need access to `DomainMode` or other
    /// context from `ParentContext`. Default implementation ignores parent_ctx
    /// and calls `apply_simple()`.
    ///
    /// # Example
    /// ```ignore
    /// fn apply_with_context(
    ///     &self,
    ///     ctx: &mut Context,
    ///     expr: ExprId,
    ///     parent_ctx: &ParentContext,
    /// ) -> Option<Rewrite> {
    ///     let mode = parent_ctx.domain_mode();
    ///     // ... domain-aware logic ...
    /// }
    /// ```
    fn apply_with_context(
        &self,
        context: &mut Context,
        expr: ExprId,
        _parent_ctx: &ParentContext,
    ) -> Option<Rewrite> {
        self.apply_simple(context, expr)
    }

    fn target_types(&self) -> Option<TargetKindSet> {
        None
    }
    /// Phases this rule is allowed to run in (default: Core + PostCleanup)
    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::CORE | PhaseMask::POST
    }

    /// Priority for rule ordering (higher = applied first). Default: 0
    fn priority(&self) -> i32 {
        0
    }

    /// Step importance level for this rule. Default: Medium (visible in normal mode)
    /// Override to Low for canonicalization and other internal transformations
    fn importance(&self) -> ImportanceLevel {
        ImportanceLevel::Medium
    }

    /// Step category for grouping. Default: General
    /// Override for specific rule types (Expand, Factor, etc.)
    fn category(&self) -> StepCategory {
        StepCategory::General
    }

    /// Safety classification for use in equation solving.
    /// Default: Always (safe in solver pre-pass).
    /// Override to NeedsCondition for rules requiring assumptions (e.g., cancellation).
    fn solve_safety(&self) -> crate::solve_safety::SolveSafety {
        crate::solve_safety::SolveSafety::Always
    }

    /// Mathematical soundness classification for this rule.
    /// Default: Equivalence (preserves mathematical identity).
    /// Override for rules that introduce requirements, choose branches, etc.
    fn soundness(&self) -> SoundnessLabel {
        SoundnessLabel::Equivalence
    }
}

/// Main Rule trait with parent-context awareness
/// Only implement this directly for rules that need parent context
pub trait Rule {
    fn name(&self) -> &str;

    /// Apply rule with parent context information
    fn apply(
        &self,
        context: &mut Context,
        expr: ExprId,
        parent_ctx: &ParentContext,
    ) -> Option<Rewrite>;

    // Optional: Return the set of Expr variant kinds this rule targets.
    // If None, the rule is applied to all nodes (global rule).
    fn target_types(&self) -> Option<TargetKindSet> {
        None
    }

    /// Phases this rule is allowed to run in (default: Core + PostCleanup)
    /// Override to TRANSFORM for expansion/distribution rules
    /// Override to RATIONALIZE for rationalization rules
    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::CORE | PhaseMask::POST
    }

    /// Priority for rule ordering (higher = applied first). Default: 0
    /// Use higher values for rules that should match before more general rules.
    fn priority(&self) -> i32 {
        0
    }

    /// Step importance level for this rule. Default: Medium
    fn importance(&self) -> ImportanceLevel {
        ImportanceLevel::Medium
    }

    /// Step category for grouping. Default: General
    fn category(&self) -> StepCategory {
        StepCategory::General
    }

    /// Safety classification for use in equation solving.
    fn solve_safety(&self) -> crate::solve_safety::SolveSafety {
        crate::solve_safety::SolveSafety::Always
    }

    /// Mathematical soundness classification for this rule.
    /// Default: Equivalence (preserves mathematical identity).
    fn soundness(&self) -> SoundnessLabel {
        SoundnessLabel::Equivalence
    }
}

/// Auto-implement Rule for any SimpleRule
/// This allows existing rules to work without modification
impl<T: SimpleRule> Rule for T {
    fn name(&self) -> &str {
        SimpleRule::name(self)
    }

    fn apply(
        &self,
        context: &mut Context,
        expr: ExprId,
        parent_ctx: &ParentContext,
    ) -> Option<Rewrite> {
        // Call apply_with_context to enable domain-aware rules
        self.apply_with_context(context, expr, parent_ctx)
    }

    fn target_types(&self) -> Option<TargetKindSet> {
        SimpleRule::target_types(self)
    }

    fn allowed_phases(&self) -> PhaseMask {
        SimpleRule::allowed_phases(self)
    }

    fn priority(&self) -> i32 {
        SimpleRule::priority(self)
    }

    fn importance(&self) -> ImportanceLevel {
        SimpleRule::importance(self)
    }

    fn category(&self) -> StepCategory {
        SimpleRule::category(self)
    }

    fn solve_safety(&self) -> crate::solve_safety::SolveSafety {
        SimpleRule::solve_safety(self)
    }

    fn soundness(&self) -> SoundnessLabel {
        SimpleRule::soundness(self)
    }
}
