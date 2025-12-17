use cas_ast::ExprId;
use std::collections::HashSet;

/// Marks for expressions that are part of special patterns.
/// This is populated by a pre-analysis pass before simplification.
#[derive(Clone, Debug, Default)]
pub struct PatternMarks {
    /// ExprIds that are part of Pythagorean identity patterns
    /// (e.g., tan in sec²-tan², cot in csc²-cot²)
    pub pythagorean_protected: HashSet<ExprId>,
    /// ExprIds that are bases inside sqrt(u²) or sqrt(u*u) patterns
    /// Protected from binomial expansion to allow sqrt(u²) → |u| shortcut
    pub sqrt_square_protected: HashSet<ExprId>,
    /// ExprIds of sin/cos Function nodes that are part of sin²(u)+cos²(u)=1 patterns
    /// Protected from angle expansion (AngleIdentityRule) to preserve Pythagorean identity
    pub trig_square_protected: HashSet<ExprId>,
}

impl PatternMarks {
    pub fn new() -> Self {
        Self {
            pythagorean_protected: HashSet::new(),
            sqrt_square_protected: HashSet::new(),
            trig_square_protected: HashSet::new(),
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

    /// Check if an expression is protected as a sqrt-square base
    /// (should not be expanded by BinomialExpansionRule)
    pub fn is_sqrt_square_protected(&self, expr: ExprId) -> bool {
        self.sqrt_square_protected.contains(&expr)
    }

    /// Mark an expression as a sqrt-square base (e.g., u² in sqrt(u²))
    pub fn mark_sqrt_square(&mut self, expr: ExprId) {
        self.sqrt_square_protected.insert(expr);
    }

    /// Check if a sin/cos function is protected as part of sin²+cos²=1 pattern
    /// (should not be expanded by AngleIdentityRule)
    pub fn is_trig_square_protected(&self, expr: ExprId) -> bool {
        self.trig_square_protected.contains(&expr)
    }

    /// Mark a sin/cos function as part of sin²+cos²=1 pattern
    pub fn mark_trig_square(&mut self, expr: ExprId) {
        self.trig_square_protected.insert(expr);
    }
}
