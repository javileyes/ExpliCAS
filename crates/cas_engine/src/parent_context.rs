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
    /// Transformation goal (controls which inverse rules are gated)
    pub(crate) goal: crate::semantics::NormalFormGoal,
    /// Root expression for this simplification pass (for implicit domain)
    pub(crate) root_expr: Option<ExprId>,
    /// Cached implicit domain (computed once per pass)
    pub(crate) implicit_domain: Option<crate::implicit_domain::ImplicitDomain>,
    /// Context mode (Standard, Solve, etc.)
    pub(crate) context_mode: crate::options::ContextMode,
    /// Purpose of simplification (Eval, SolvePrepass, SolveTactic)
    pub(crate) simplify_purpose: crate::solve_safety::SimplifyPurpose,
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
            goal: crate::semantics::NormalFormGoal::default(),
            root_expr: None,
            implicit_domain: None,
            context_mode: crate::options::ContextMode::default(),
            simplify_purpose: crate::solve_safety::SimplifyPurpose::default(),
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
            goal: crate::semantics::NormalFormGoal::default(),
            root_expr: None,
            implicit_domain: None,
            context_mode: crate::options::ContextMode::default(),
            simplify_purpose: crate::solve_safety::SimplifyPurpose::default(),
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
            goal: crate::semantics::NormalFormGoal::default(),
            root_expr: None,
            implicit_domain: None,
            context_mode: crate::options::ContextMode::default(),
            simplify_purpose: crate::solve_safety::SimplifyPurpose::default(),
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
            goal: crate::semantics::NormalFormGoal::default(),
            root_expr: None,
            implicit_domain: None,
            context_mode: crate::options::ContextMode::default(),
            simplify_purpose: crate::solve_safety::SimplifyPurpose::default(),
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
            goal: self.goal,
            root_expr: self.root_expr,
            implicit_domain: self.implicit_domain.clone(),
            context_mode: self.context_mode,
            simplify_purpose: self.simplify_purpose,
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

    /// Get transformation goal
    pub fn goal(&self) -> crate::semantics::NormalFormGoal {
        self.goal
    }

    /// Set transformation goal, returning a new context
    pub fn with_goal(mut self, goal: crate::semantics::NormalFormGoal) -> Self {
        self.goal = goal;
        self
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

    /// Get root expression for this pass (used for implicit domain)
    pub fn root_expr(&self) -> Option<ExprId> {
        self.root_expr
    }

    /// Set root expression and compute implicit domain
    pub fn with_root_expr(mut self, ctx: &Context, root: ExprId) -> Self {
        use crate::implicit_domain::infer_implicit_domain;
        self.root_expr = Some(root);
        self.implicit_domain = Some(infer_implicit_domain(ctx, root, self.value_domain));
        self
    }

    /// Set root expression only (without computing implicit domain)
    /// V2.14.21: Used for propagation to rules that need root_expr for lazy computation
    pub fn with_root_expr_only(mut self, root: ExprId) -> Self {
        self.root_expr = Some(root);
        self
    }

    /// Get cached implicit domain
    pub fn implicit_domain(&self) -> Option<&crate::implicit_domain::ImplicitDomain> {
        self.implicit_domain.as_ref()
    }

    /// Set implicit domain (for propagation during rule execution)
    /// V2.14.20: Used to propagate pre-computed implicit domain from initial context
    pub fn with_implicit_domain(
        mut self,
        domain: Option<crate::implicit_domain::ImplicitDomain>,
    ) -> Self {
        self.implicit_domain = domain;
        self
    }

    /// Get simplify purpose (Eval, SolvePrepass, SolveTactic)
    pub fn simplify_purpose(&self) -> crate::solve_safety::SimplifyPurpose {
        self.simplify_purpose
    }

    /// Set simplify purpose, returning a new context
    pub fn with_simplify_purpose(mut self, purpose: crate::solve_safety::SimplifyPurpose) -> Self {
        self.simplify_purpose = purpose;
        self
    }

    /// Get context mode (Standard, Solve, etc.)
    pub fn context_mode(&self) -> crate::options::ContextMode {
        self.context_mode
    }

    /// Set context mode, returning a new context
    pub fn with_context_mode(mut self, mode: crate::options::ContextMode) -> Self {
        self.context_mode = mode;
        self
    }

    /// Check if we're in a Solve context (via context_mode or simplify_purpose)
    pub fn is_solve_context(&self) -> bool {
        // Check context_mode (explicit Solve mode from options)
        if self.context_mode == crate::options::ContextMode::Solve {
            return true;
        }
        // Check simplify_purpose (solver pre-pass or tactic)
        matches!(
            self.simplify_purpose,
            crate::solve_safety::SimplifyPurpose::SolvePrepass
                | crate::solve_safety::SimplifyPurpose::SolveTactic
        )
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
