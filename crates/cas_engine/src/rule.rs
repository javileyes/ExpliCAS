use crate::parent_context::ParentContext;
use cas_ast::{Context, ExprId};

pub struct Rewrite {
    pub new_expr: ExprId,
    pub description: String,
}

/// Simplified Rule trait for backward compatibility
/// Most rules should implement this for simplicity
pub trait SimpleRule {
    fn name(&self) -> &str;
    fn apply_simple(&self, context: &mut Context, expr: ExprId) -> Option<Rewrite>;
    fn target_types(&self) -> Option<Vec<&str>> {
        None
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
}
