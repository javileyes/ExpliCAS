use cas_ast::ExprId;
use std::collections::HashSet;

/// Marks for expressions that are part of special patterns.
/// This is populated by a pre-analysis pass before simplification.
#[derive(Clone, Debug, Default)]
pub struct PatternMarks {
    /// ExprIds that are part of Pythagorean identity patterns
    /// (e.g., tan in sec²-tan², cot in csc²-cot²)
    pub pythagorean_protected: HashSet<ExprId>,
}

impl PatternMarks {
    pub fn new() -> Self {
        Self {
            pythagorean_protected: HashSet::new(),
        }
    }

    /// Check if an expression is marked as part of a Pythagorean pattern
    pub fn is_pythagorean_protected(&self, expr: ExprId) -> bool {
        self.pythagorean_protected.contains(&expr)
    }

    /// Mark an expression as part of a Pythagorean pattern
    pub fn mark_pythagorean(&mut self, expr: ExprId) {
        self.pythagorean_protected.insert(expr);
    }
}
