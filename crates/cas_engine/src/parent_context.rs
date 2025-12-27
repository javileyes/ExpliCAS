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
    /// Domain assumption mode for factor cancellation
    pub(crate) domain_mode: crate::domain::DomainMode,
    /// Inverse trig composition policy
    pub(crate) inv_trig: crate::semantics::InverseTrigPolicy,
    /// Value domain for constants (RealOnly, ComplexEnabled)
    pub(crate) value_domain: crate::semantics::ValueDomain,
    /// Branch policy for multi-valued functions
    pub(crate) branch: crate::semantics::BranchPolicy,
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
            domain_mode: crate::domain::DomainMode::default(),
            inv_trig: crate::semantics::InverseTrigPolicy::default(),
            value_domain: crate::semantics::ValueDomain::default(),
            branch: crate::semantics::BranchPolicy::default(),
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
            domain_mode: crate::domain::DomainMode::default(),
            inv_trig: crate::semantics::InverseTrigPolicy::default(),
            value_domain: crate::semantics::ValueDomain::default(),
            branch: crate::semantics::BranchPolicy::default(),
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
            domain_mode: crate::domain::DomainMode::default(),
            inv_trig: crate::semantics::InverseTrigPolicy::default(),
            value_domain: crate::semantics::ValueDomain::default(),
            branch: crate::semantics::BranchPolicy::default(),
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
            domain_mode: crate::domain::DomainMode::default(),
            inv_trig: crate::semantics::InverseTrigPolicy::default(),
            value_domain: crate::semantics::ValueDomain::default(),
            branch: crate::semantics::BranchPolicy::default(),
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
            domain_mode: self.domain_mode,
            inv_trig: self.inv_trig,
            value_domain: self.value_domain,
            branch: self.branch,
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

    /// Get the domain assumption mode
    pub fn domain_mode(&self) -> crate::domain::DomainMode {
        self.domain_mode
    }

    /// Set domain_mode flag, returning a new context
    pub fn with_domain_mode(mut self, mode: crate::domain::DomainMode) -> Self {
        self.domain_mode = mode;
        self
    }

    /// Get inverse trig policy
    pub fn inv_trig_policy(&self) -> crate::semantics::InverseTrigPolicy {
        self.inv_trig
    }

    /// Set inverse trig policy, returning a new context
    pub fn with_inv_trig(mut self, policy: crate::semantics::InverseTrigPolicy) -> Self {
        self.inv_trig = policy;
        self
    }

    /// Get value domain
    pub fn value_domain(&self) -> crate::semantics::ValueDomain {
        self.value_domain
    }

    /// Set value domain, returning a new context
    pub fn with_value_domain(mut self, domain: crate::semantics::ValueDomain) -> Self {
        self.value_domain = domain;
        self
    }

    /// Get branch policy
    pub fn branch_policy(&self) -> crate::semantics::BranchPolicy {
        self.branch
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

    /// Check if the given expr_id or any ancestor is in auto-expand context.
    /// This is used when the rule is applied directly to the marked node.
    pub fn in_auto_expand_context_for_expr(&self, expr_id: ExprId) -> bool {
        if let Some(marks) = &self.pattern_marks {
            // Check current node
            if marks.is_auto_expand_context(expr_id) {
                return true;
            }
            // Check ancestors
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
