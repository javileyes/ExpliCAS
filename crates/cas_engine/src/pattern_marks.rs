use cas_ast::ExprId;
use std::collections::HashSet;

/// Marks for expressions that are part of special patterns.
/// This is populated by a pre-analysis pass before simplification.
///
/// # Lifecycle: Build-Then-Freeze Pattern
///
/// PatternMarks is designed with a "build-then-freeze" lifecycle:
/// 1. **Build phase**: Created and mutated during pre-analysis (in `orchestrator.rs`)
/// 2. **Freeze phase**: Wrapped in `Rc<PatternMarks>` when passed to `ParentContext`
/// 3. **Read-only phase**: Shared across all recursive calls via `Rc` clone (O(1))
///
/// **IMPORTANT**: Once wrapped in `Rc` and stored in `ParentContext`, this struct
/// MUST NOT be mutated. All `mark_*` methods should only be called during the
/// build phase, before wrapping in `Rc`.
///
/// This pattern is critical for stack frame size: without `Rc`, cloning the 7
/// `HashSet`s on every recursive call caused ~150KB frames and stack overflow.
/// With `Rc`, clone is O(1) and frames are safe at depth 50+.
///
/// # Thread Safety
///
/// Uses `Rc` (not `Arc`) because `Simplifier` is single-threaded.
/// Switch to `Arc` if parallelizing simplification in the future.
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
    /// ExprIds of context nodes (Div/Sub) where auto-expand is beneficial.
    /// When processing is inside one of these contexts, Pow(Add(..), n) will be expanded.
    /// This is more robust than marking individual Pow nodes, as rewrites may change ExprIds.
    pub auto_expand_contexts: HashSet<ExprId>,
    /// ExprIds of tan/sin/cos nodes that are inside arctan(tan(x)) or similar inverse-trig patterns.
    /// Protected from conversion to sin/cos to allow principal value simplification.
    pub inverse_trig_protected: HashSet<ExprId>,
    /// ExprIds of sin/cos Function nodes inside (sin(A)+sin(B))/(cos(A)+cos(B)) patterns.
    /// Protected from TripleAngleRule to allow sum-to-product simplification.
    pub sum_quotient_protected: HashSet<ExprId>,
    /// ExprIds of tan() nodes that are part of tan(u)·tan(π/3+u)·tan(π/3-u) triple product.
    /// Protected from TanToSinCosRule expansion to allow TanTripleProductRule to fire.
    pub tan_triple_product_protected: HashSet<ExprId>,
}

impl PatternMarks {
    pub fn new() -> Self {
        Self {
            pythagorean_protected: HashSet::new(),
            sqrt_square_protected: HashSet::new(),
            trig_square_protected: HashSet::new(),
            auto_expand_contexts: HashSet::new(),
            inverse_trig_protected: HashSet::new(),
            sum_quotient_protected: HashSet::new(),
            tan_triple_product_protected: HashSet::new(),
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

    /// Check if an expression is marked as an auto-expand context
    /// (e.g., Div/Sub nodes where expansion helps cancellation)
    pub fn is_auto_expand_context(&self, expr: ExprId) -> bool {
        self.auto_expand_contexts.contains(&expr)
    }

    /// Mark an expression as an auto-expand context
    /// This should be a Div or Sub node where expanding inner Pow nodes helps cancellation
    pub fn mark_auto_expand_context(&mut self, expr: ExprId) {
        self.auto_expand_contexts.insert(expr);
    }

    /// Check if any auto-expand contexts have been marked
    pub fn has_auto_expand_contexts(&self) -> bool {
        !self.auto_expand_contexts.is_empty()
    }

    /// Check if an expression is protected as part of inverse-trig pattern
    /// (e.g., tan(x) in arctan(tan(x)) should not be converted to sin/cos)
    pub fn is_inverse_trig_protected(&self, expr: ExprId) -> bool {
        self.inverse_trig_protected.contains(&expr)
    }

    /// Mark an expression as part of inverse-trig pattern
    pub fn mark_inverse_trig(&mut self, expr: ExprId) {
        self.inverse_trig_protected.insert(expr);
    }

    /// Check if an expression is protected as part of sum-quotient pattern
    /// (e.g., sin(3x) in (sin(x)+sin(3x))/(cos(x)+cos(3x)) should not be expanded by TripleAngleRule)
    pub fn is_sum_quotient_protected(&self, expr: ExprId) -> bool {
        self.sum_quotient_protected.contains(&expr)
    }

    /// Mark an expression as part of sum-quotient pattern
    pub fn mark_sum_quotient(&mut self, expr: ExprId) {
        self.sum_quotient_protected.insert(expr);
    }

    /// Check if an expression is protected as part of tan triple product pattern
    /// (e.g., tan(x) in tan(x)·tan(π/3+x)·tan(π/3-x) should not be converted to sin/cos)
    pub fn is_tan_triple_product_protected(&self, expr: ExprId) -> bool {
        self.tan_triple_product_protected.contains(&expr)
    }

    /// Mark an expression as part of tan triple product pattern
    pub fn mark_tan_triple_product(&mut self, expr: ExprId) {
        self.tan_triple_product_protected.insert(expr);
    }
}
