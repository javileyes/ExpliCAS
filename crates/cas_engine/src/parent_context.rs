use cas_ast::{Context, ExprId};

/// ParentContext tracks the ancestor chain of an expression without holding a Context reference.
/// Used by context-aware rules to detect special patterns like Pythagorean identities.
#[derive(Clone)]
pub struct ParentContext {
    /// IDs of ancestor expressions, from closest to furthest
    pub(crate) ancestors: Vec<ExprId>,
    /// Pre-scanned pattern marks for context-aware guards
    pub(crate) pattern_marks: Option<crate::pattern_marks::PatternMarks>,
    /// Whether we're in "expand mode" - forces aggressive distribution/expansion
    pub(crate) expand_mode: bool,
    /// Whether auto-expand is enabled (expand cheap cases within budget)
    pub(crate) auto_expand: bool,
    /// Budget for auto-expand (only used when auto_expand=true)
    pub(crate) auto_expand_budget: Option<crate::phase::ExpandBudget>,
}

impl ParentContext {
    /// Create empty context for root expressions
    pub fn root() -> Self {
        Self {
            ancestors: Vec::new(),
            pattern_marks: None,
            expand_mode: false,
            auto_expand: false,
            auto_expand_budget: None,
        }
    }

    /// Create context with single parent
    pub fn with_parent(parent: ExprId) -> Self {
        Self {
            ancestors: vec![parent],
            pattern_marks: None,
            expand_mode: false,
            auto_expand: false,
            auto_expand_budget: None,
        }
    }

    /// Create context with pattern marks
    pub fn with_marks(pattern_marks: crate::pattern_marks::PatternMarks) -> Self {
        Self {
            ancestors: Vec::new(),
            pattern_marks: Some(pattern_marks),
            expand_mode: false,
            auto_expand: false,
            auto_expand_budget: None,
        }
    }

    /// Create context with expand_mode enabled
    pub fn with_expand_mode(
        pattern_marks: crate::pattern_marks::PatternMarks,
        expand_mode: bool,
    ) -> Self {
        Self {
            ancestors: Vec::new(),
            pattern_marks: Some(pattern_marks),
            expand_mode,
            auto_expand: false,
            auto_expand_budget: None,
        }
    }

    /// Extend context by adding a new parent
    /// This is used when recursing down the tree
    pub fn extend(&self, parent_id: ExprId) -> Self {
        let mut new_ancestors = self.ancestors.clone();
        new_ancestors.push(parent_id);
        Self {
            ancestors: new_ancestors,
            pattern_marks: self.pattern_marks.clone(),
            expand_mode: self.expand_mode,
            auto_expand: self.auto_expand,
            auto_expand_budget: self.auto_expand_budget,
        }
    }

    /// Check if any ancestor matches the given predicate
    pub fn has_ancestor_matching<F>(&self, ctx: &Context, predicate: F) -> bool
    where
        F: Fn(&Context, ExprId) -> bool,
    {
        self.ancestors
            .iter()
            .any(|&ancestor| predicate(ctx, ancestor))
    }

    pub fn pattern_marks(&self) -> Option<&crate::pattern_marks::PatternMarks> {
        self.pattern_marks.as_ref()
    }

    /// Check if we're in expand mode (aggressive distribution/expansion)
    pub fn is_expand_mode(&self) -> bool {
        self.expand_mode
    }

    /// Set expand_mode flag, returning a new context
    /// Used to propagate expand_mode from initial context during rule application
    pub fn with_expand_mode_flag(mut self, expand_mode: bool) -> Self {
        self.expand_mode = expand_mode;
        self
    }

    /// Check if auto-expand is enabled
    pub fn is_auto_expand(&self) -> bool {
        self.auto_expand
    }

    /// Get auto-expand budget, if set
    pub fn auto_expand_budget(&self) -> Option<&crate::phase::ExpandBudget> {
        self.auto_expand_budget.as_ref()
    }

    /// Set auto-expand with budget
    pub fn with_auto_expand(mut self, budget: crate::phase::ExpandBudget) -> Self {
        self.auto_expand = true;
        self.auto_expand_budget = Some(budget);
        self
    }

    /// Set auto-expand flag only (for propagation)
    pub fn with_auto_expand_flag(
        mut self,
        auto_expand: bool,
        budget: Option<crate::phase::ExpandBudget>,
    ) -> Self {
        self.auto_expand = auto_expand;
        self.auto_expand_budget = budget;
        self
    }

    /// Get immediate parent, if exists
    pub fn immediate_parent(&self) -> Option<ExprId> {
        self.ancestors.last().copied() // Last element is most recent parent
    }

    /// Get all ancestors in order from immediate parent to root
    pub fn all_ancestors(&self) -> &[ExprId] {
        &self.ancestors
    }

    /// Get depth in the tree (number of ancestors)
    pub fn depth(&self) -> usize {
        self.ancestors.len()
    }

    /// Check if we're inside an auto-expand context (marked Div/Sub where expansion helps)
    /// This is more robust than checking individual Pow nodes, as those may get rewritten.
    pub fn in_auto_expand_context(&self) -> bool {
        if let Some(marks) = &self.pattern_marks {
            self.ancestors
                .iter()
                .any(|id| marks.is_auto_expand_context(*id))
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Expr;
    use num_traits::ToPrimitive;

    #[test]
    fn test_root_context() {
        let parent_ctx = ParentContext::root();

        assert_eq!(parent_ctx.immediate_parent(), None);
        assert_eq!(parent_ctx.all_ancestors().len(), 0);
        assert_eq!(parent_ctx.depth(), 0);
    }

    #[test]
    fn test_with_parent() {
        let mut ctx = Context::new();
        let parent_id = ctx.num(42);
        let parent_ctx = ParentContext::with_parent(parent_id);

        assert_eq!(parent_ctx.immediate_parent(), Some(parent_id));
        assert_eq!(parent_ctx.all_ancestors().len(), 1);
        assert_eq!(parent_ctx.depth(), 1);
    }

    #[test]
    fn test_extend() {
        let mut ctx = Context::new();
        let grandparent = ctx.num(1);
        let parent = ctx.num(2);

        let ctx1 = ParentContext::with_parent(grandparent);
        let ctx2 = ctx1.extend(parent);

        assert_eq!(ctx2.immediate_parent(), Some(parent));
        assert_eq!(ctx2.depth(), 2);
        assert_eq!(ctx2.all_ancestors(), &[grandparent, parent]);
    }

    #[test]
    fn test_has_ancestor_matching() {
        let mut ctx = Context::new();
        let target = ctx.num(42);
        let _other = ctx.num(99);

        let parent_ctx = ParentContext::with_parent(target);

        assert!(parent_ctx.has_ancestor_matching(&ctx, |c, id| {
            matches!(c.get(id), Expr::Number(n) if n.to_i32() == Some(42))
        }));

        assert!(!parent_ctx.has_ancestor_matching(&ctx, |c, id| {
            matches!(c.get(id), Expr::Number(n) if n.to_i32() == Some(99))
        }));
    }
}
