use crate::parent_context::ParentContext;
use crate::phase::PhaseMask;
use crate::step::ImportanceLevel;
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
    /// When set, CLI/timeline can display warnings about implicit assumptions.
    /// The assumption collector will convert this to structured AssumptionEvent.
    pub domain_assumption: Option<&'static str>,
}

impl Rewrite {
    /// Create a simple rewrite (most common case - local transform = global transform)
    pub fn simple(new_expr: ExprId, description: impl Into<String>) -> Self {
        Rewrite {
            new_expr,
            description: description.into(),
            before_local: None,
            after_local: None,
            domain_assumption: None,
        }
    }

    /// Create a rewrite with explicit local before/after (for n-ary rules)
    /// Use when the rule transforms a subpattern within a larger expression
    pub fn with_local(
        new_expr: ExprId,
        description: impl Into<String>,
        before_local: ExprId,
        after_local: ExprId,
    ) -> Self {
        Rewrite {
            new_expr,
            description: description.into(),
            before_local: Some(before_local),
            after_local: Some(after_local),
            domain_assumption: None,
        }
    }

    /// Create a rewrite with domain assumption warning
    pub fn with_domain_assumption(
        new_expr: ExprId,
        description: impl Into<String>,
        assumption: &'static str,
    ) -> Self {
        Rewrite {
            new_expr,
            description: description.into(),
            before_local: None,
            after_local: None,
            domain_assumption: Some(assumption),
        }
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

    /// Step importance level for this rule. Default: Low (hidden in normal mode)
    /// Override to Medium for pedagogically valuable transformations
    fn importance(&self) -> ImportanceLevel {
        ImportanceLevel::Low
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

    /// Step importance level for this rule. Default: Low
    fn importance(&self) -> ImportanceLevel {
        ImportanceLevel::Low
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
}
