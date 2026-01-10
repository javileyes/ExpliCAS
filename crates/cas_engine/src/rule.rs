use crate::parent_context::ParentContext;
use crate::phase::PhaseMask;
use crate::step::{ImportanceLevel, StepCategory};
use cas_ast::{Context, ExprId};

/// Result of a rule application containing the new expression and metadata
pub struct Rewrite {
    /// The transformed expression
    pub new_expr: ExprId,
    /// Human-readable description of the transformation
    pub description: String,
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
}

impl Default for Rewrite {
    fn default() -> Self {
        Self {
            new_expr: cas_ast::ExprId::from_raw(0), // Will be set by new()
            description: String::new(),
            before_local: None,
            after_local: None,
            assumption_events: Default::default(),
            required_conditions: vec![],
            poly_proof: None,
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
    pub fn simple(new_expr: ExprId, description: impl Into<String>) -> Self {
        Self::new(new_expr).desc(description)
    }

    /// Create a rewrite with explicit local before/after (for n-ary rules)
    /// Use when the rule transforms a subpattern within a larger expression
    #[must_use]
    pub fn with_local(
        new_expr: ExprId,
        description: impl Into<String>,
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
        description: impl Into<String>,
        _assumption: &'static str,
    ) -> Self {
        Self::new(new_expr).desc(description)
    }

    /// Create a rewrite with polynomial proof data (for PolyZero airbag)
    /// Used by PolynomialIdentityZeroRule to attach normalization info for didactic display
    #[must_use]
    pub fn with_poly_proof(
        new_expr: ExprId,
        description: impl Into<String>,
        poly_proof: crate::multipoly_display::PolynomialProofData,
    ) -> Self {
        Self::new(new_expr).desc(description).poly_proof(poly_proof)
    }

    // =========================================================================
    // Fluent Setters (Builder Pattern)
    // =========================================================================

    /// Set the description of this rewrite.
    #[must_use]
    pub fn desc(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
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

    fn target_types(&self) -> Option<Vec<&str>> {
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

    // Optional: Return list of Expr variant names this rule targets.
    // If None, the rule is applied to all nodes.
    // Common variants: "Add", "Sub", "Mul", "Div", "Pow", "Neg", "Function", "Variable", "Number", "Constant"
    fn target_types(&self) -> Option<Vec<&str>> {
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

    fn target_types(&self) -> Option<Vec<&str>> {
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
}
