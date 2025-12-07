use cas_ast::{Context, ExprId};

/// ParentContext tracks the ancestor chain of an expression without holding a Context reference.
/// Used by context-aware rules to detect special patterns like Pythagorean identities.
#[derive(Clone)]
pub struct ParentContext {
    /// IDs of ancestor expressions, from closest to furthest
    pub(crate) ancestors: Vec<ExprId>,
    /// Pre-scanned pattern marks for context-aware guards
    pub(crate) pattern_marks: Option<crate::pattern_marks::PatternMarks>,
}

impl ParentContext {
    /// Create empty context for root expressions
    pub fn root() -> Self {
        Self {
            ancestors: Vec::new(),
            pattern_marks: None,
        }
    }

    /// Create context with single parent
    pub fn with_parent(parent: ExprId) -> Self {
        Self {
            ancestors: vec![parent],
            pattern_marks: None,
        }
    }

    /// Create context with pattern marks
    pub fn with_marks(pattern_marks: crate::pattern_marks::PatternMarks) -> Self {
        Self {
            ancestors: Vec::new(),
            pattern_marks: Some(pattern_marks),
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

    /// Get immediate parent, if exists
    pub fn immediate_parent(&self) -> Option<ExprId> {
        self.ancestors.first().copied()
    }

    /// Get all ancestors in order from immediate parent to root
    pub fn all_ancestors(&self) -> &[ExprId] {
        &self.ancestors
    }

    /// Get depth in the tree (number of ancestors)
    pub fn depth(&self) -> usize {
        self.ancestors.len()
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
