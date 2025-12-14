use crate::parent_context::ParentContext;
use crate::phase::PhaseMask;
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
    /// When set, CLI/timeline can display warnings about implicit assumptions
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
    fn apply_simple(&self, context: &mut Context, expr: ExprId) -> Option<Rewrite>;
    fn target_types(&self) -> Option<Vec<&str>> {
        None
    }
    /// Phases this rule is allowed to run in (default: Core + PostCleanup)
    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::CORE | PhaseMask::POST
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
        _parent_ctx: &ParentContext,
    ) -> Option<Rewrite> {
        self.apply_simple(context, expr)
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        SimpleRule::target_types(self)
    }

    fn allowed_phases(&self) -> PhaseMask {
        SimpleRule::allowed_phases(self)
    }
}
