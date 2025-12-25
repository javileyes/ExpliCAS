//! # N-ary Expression Views
//!
//! This module provides shape-independent views for additive and multiplicative
//! expressions. Rules operate on flat lists of terms/factors instead of pattern
//! matching binary tree structure.
//!
//! ## Key Types
//!
//! - [`AddView`]: Flattened view of Add/Sub/Neg chains with signed terms
//! - [`MulView`]: Flattened view of Mul chains with commutativity tracking
//!
//! ## Benefits
//!
//! - **Shape-independence**: `a+(b+c)`, `(a+b)+c`, and balanced trees all produce same terms
//! - **Heap-free for small expressions**: Uses `SmallVec` for common cases (≤8 terms)
//! - **Canonical rebuild**: Rebuilds as balanced trees using `build_balanced_*`
//! - **Commutativity-aware**: MulView respects matrix non-commutativity
//!
//! ## Example
//!
//! ```ignore
//! // Rule that finds atan(r) + atan(1/r) pairs - works regardless of tree shape
//! let view = AddView::from_expr(ctx, expr);
//! for i in 0..view.terms.len() {
//!     if let Some(r) = is_atan(ctx, view.terms[i].0) {
//!         for j in (i+1)..view.terms.len() {
//!             if is_atan_reciprocal(ctx, view.terms[j].0, r) {
//!                 // Found pair! Remove both and add π/2
//!             }
//!         }
//!     }
//! }
//! ```

use cas_ast::{Context, Expr, ExprId};
use smallvec::SmallVec;

/// Sign of a term in an additive expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sign {
    /// Positive term: +x
    Pos,
    /// Negative term: -x
    Neg,
}

impl Sign {
    /// Convert to multiplier for numeric operations.
    #[inline]
    pub fn to_i32(self) -> i32 {
        match self {
            Sign::Pos => 1,
            Sign::Neg => -1,
        }
    }

    /// Negate the sign.
    #[inline]
    pub fn negate(self) -> Sign {
        match self {
            Sign::Pos => Sign::Neg,
            Sign::Neg => Sign::Pos,
        }
    }
}

// ============================================================================
// Public Builder Functions
// ============================================================================

/// Build a balanced Add tree from a slice of terms.
///
/// This is useful for rules that need to reconstruct n-ary sums from a list of terms.
/// - Empty slice → 0
/// - Single term → that term
/// - Multiple terms → balanced binary Add tree
pub fn build_balanced_add(ctx: &mut Context, terms: &[ExprId]) -> ExprId {
    match terms.len() {
        0 => ctx.num(0),
        1 => terms[0],
        2 => ctx.add(Expr::Add(terms[0], terms[1])),
        n => {
            let mid = n / 2;
            let left = build_balanced_add(ctx, &terms[..mid]);
            let right = build_balanced_add(ctx, &terms[mid..]);
            ctx.add(Expr::Add(left, right))
        }
    }
}

/// Build a balanced Mul tree from a slice of factors.
///
/// Wrapper around `Context::build_balanced_mul` with handling for empty case.
///
/// # Edge cases
/// - Empty slice → 1 (multiplicative identity)
/// - Single factor → that factor
/// - Multiple factors → balanced binary Mul tree
///
/// # See also
/// - `Context::build_balanced_mul` (canonical implementation)
/// - `MulBuilder` for right-fold construction with exponents
pub fn build_balanced_mul(ctx: &mut Context, factors: &[ExprId]) -> ExprId {
    match factors.len() {
        0 => ctx.num(1), // Multiplicative identity (not handled by Context)
        _ => ctx.build_balanced_mul(factors),
    }
}

/// N-ary view of an additive expression (flattened Add/Sub/Neg chain).
///
/// Shape-independent: works identically regardless of tree structure.
/// A left-associative `((a+b)+c)+d`, right-associative `a+(b+(c+d))`,
/// or balanced tree all produce the same `terms` list.
#[derive(Debug, Clone)]
pub struct AddView {
    /// Original root expression (for debugging/steps).
    pub root: ExprId,
    /// Flattened terms with signs.
    /// SmallVec avoids heap allocation for expressions with ≤8 terms.
    pub terms: SmallVec<[(ExprId, Sign); 8]>,
}

/// N-ary view of a multiplicative expression (flattened Mul chain).
///
/// Shape-independent: works identically regardless of tree structure.
/// Respects commutativity: if any factor is non-commutative (e.g., matrix),
/// the `commutative` flag is `false` and `rebuild` won't sort factors.
#[derive(Debug, Clone)]
pub struct MulView {
    /// Original root expression (for debugging/steps).
    pub root: ExprId,
    /// Flattened factors (order preserved if non-commutative).
    /// SmallVec avoids heap allocation for expressions with ≤8 factors.
    pub factors: SmallVec<[ExprId; 8]>,
    /// Whether factors can be reordered during rebuild.
    /// Based on `Context::is_mul_commutative` for each factor.
    pub commutative: bool,
}

// ============================================================================
// AddView Implementation
// ============================================================================

impl AddView {
    /// Create an AddView from any expression.
    ///
    /// This always returns a view:
    /// - Add/Sub/Neg chains are flattened into terms with signs
    /// - Other expressions become a single-term sum `[(expr, Pos)]`
    ///
    /// # Shape Independence
    /// Different tree structures produce identical term lists:
    /// - `(a+b)+c` → `[(a, +), (b, +), (c, +)]`
    /// - `a+(b+c)` → `[(a, +), (b, +), (c, +)]`
    pub fn from_expr(ctx: &Context, root: ExprId) -> Self {
        let mut terms = SmallVec::new();
        Self::collect_terms(ctx, root, Sign::Pos, &mut terms);
        AddView { root, terms }
    }

    /// Iterative term collector (stack-safe for deep expressions).
    ///
    /// NOTE: This collector unwraps __hold barriers per the Hold Contract
    /// (see ARCHITECTURE.md "Canonical Utilities Registry"). This makes
    /// AddView transparent to internal barriers used by autoexpand.
    fn collect_terms(
        ctx: &Context,
        root: ExprId,
        initial_sign: Sign,
        out: &mut SmallVec<[(ExprId, Sign); 8]>,
    ) {
        // Worklist: (expr, sign)
        let mut stack = vec![(root, initial_sign)];

        while let Some((id, sign)) = stack.pop() {
            // Unwrap __hold barrier per Hold Contract (transparency for algebra)
            let id = cas_ast::hold::unwrap_hold(ctx, id);

            match ctx.get(id) {
                Expr::Add(l, r) => {
                    stack.push((*r, sign));
                    stack.push((*l, sign));
                }
                Expr::Sub(l, r) => {
                    stack.push((*r, sign.negate()));
                    stack.push((*l, sign));
                }
                Expr::Neg(inner) => {
                    stack.push((*inner, sign.negate()));
                }
                // Any other expression is a leaf term
                _ => {
                    out.push((id, sign));
                }
            }
        }
    }

    /// Rebuild the expression as a canonical balanced tree.
    ///
    /// - Empty → 0
    /// - Single term → that term (with sign applied)
    /// - Multiple terms → balanced Add tree
    ///
    /// Signs are applied by wrapping negative terms in `Neg`.
    pub fn rebuild(self, ctx: &mut Context) -> ExprId {
        if self.terms.is_empty() {
            return ctx.num(0);
        }

        // Apply signs and collect final terms
        let signed_terms: SmallVec<[ExprId; 8]> = self
            .terms
            .into_iter()
            .map(|(term, sign)| match sign {
                Sign::Pos => term,
                Sign::Neg => ctx.add(Expr::Neg(term)),
            })
            .collect();

        if signed_terms.len() == 1 {
            return signed_terms[0];
        }

        // Use balanced builder
        build_balanced_add(ctx, &signed_terms)
    }

    /// Check if the view is empty (no terms).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Number of terms.
    #[inline]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Find indices of terms matching a predicate.
    pub fn find<F>(&self, ctx: &Context, mut pred: F) -> SmallVec<[usize; 4]>
    where
        F: FnMut(&Context, ExprId, Sign) -> bool,
    {
        self.terms
            .iter()
            .enumerate()
            .filter_map(
                |(i, &(term, sign))| {
                    if pred(ctx, term, sign) {
                        Some(i)
                    } else {
                        None
                    }
                },
            )
            .collect()
    }

    /// Remove term at index, returning it.
    pub fn remove(&mut self, idx: usize) -> (ExprId, Sign) {
        self.terms.remove(idx)
    }

    /// Retain only terms matching a predicate.
    pub fn retain<F>(&mut self, mut pred: F)
    where
        F: FnMut(ExprId, Sign) -> bool,
    {
        self.terms.retain(|item| pred(item.0, item.1));
    }
}

// ============================================================================
// MulView Implementation
// ============================================================================

impl MulView {
    /// Create a MulView from any expression.
    ///
    /// This always returns a view:
    /// - Mul chains are flattened into factors
    /// - Other expressions become a single-factor product `[expr]`
    ///
    /// The `commutative` flag is computed based on whether all factors
    /// are commutative (no matrices, etc.).
    pub fn from_expr(ctx: &Context, root: ExprId) -> Self {
        let mut factors = SmallVec::new();
        Self::collect_factors(ctx, root, &mut factors);

        // Compute commutativity from all factors
        let commutative = factors.iter().all(|&f| ctx.is_mul_commutative(f));

        MulView {
            root,
            factors,
            commutative,
        }
    }

    /// Iterative factor collector (stack-safe for deep expressions).
    ///
    /// NOTE: This collector unwraps __hold barriers per the Hold Contract.
    fn collect_factors(ctx: &Context, root: ExprId, out: &mut SmallVec<[ExprId; 8]>) {
        let mut stack = vec![root];

        while let Some(id) = stack.pop() {
            // Unwrap __hold barrier per Hold Contract (transparency for algebra)
            let id = cas_ast::hold::unwrap_hold(ctx, id);

            match ctx.get(id) {
                Expr::Mul(l, r) => {
                    // Push right first so left is processed first (preserves order)
                    stack.push(*r);
                    stack.push(*l);
                }
                // Any other expression is a leaf factor
                _ => {
                    out.push(id);
                }
            }
        }
    }

    /// Rebuild the expression as a canonical balanced tree.
    ///
    /// - Empty → 1
    /// - Single factor → that factor
    /// - Multiple factors → balanced Mul tree
    ///
    /// If `commutative`, factors are sorted before building.
    /// If non-commutative, original order is preserved.
    pub fn rebuild(mut self, ctx: &mut Context) -> ExprId {
        if self.factors.is_empty() {
            return ctx.num(1);
        }

        if self.factors.len() == 1 {
            return self.factors[0];
        }

        // Sort only if commutative
        if self.commutative {
            self.factors
                .sort_by(|a, b| crate::ordering::compare_expr(ctx, *a, *b));
        }

        Self::build_balanced_mul(ctx, &self.factors)
    }

    /// Build a balanced Mul tree from a slice of factors.
    fn build_balanced_mul(ctx: &mut Context, factors: &[ExprId]) -> ExprId {
        match factors.len() {
            0 => ctx.num(1),
            1 => factors[0],
            2 => ctx.add(Expr::Mul(factors[0], factors[1])),
            n => {
                let mid = n / 2;
                let left = Self::build_balanced_mul(ctx, &factors[..mid]);
                let right = Self::build_balanced_mul(ctx, &factors[mid..]);
                ctx.add(Expr::Mul(left, right))
            }
        }
    }

    /// Check if the view is empty (no factors).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }

    /// Number of factors.
    #[inline]
    pub fn len(&self) -> usize {
        self.factors.len()
    }

    /// Find indices of factors matching a predicate.
    pub fn find<F>(&self, ctx: &Context, mut pred: F) -> SmallVec<[usize; 4]>
    where
        F: FnMut(&Context, ExprId) -> bool,
    {
        self.factors
            .iter()
            .enumerate()
            .filter_map(
                |(i, &factor)| {
                    if pred(ctx, factor) {
                        Some(i)
                    } else {
                        None
                    }
                },
            )
            .collect()
    }

    /// Remove factor at index, returning it.
    pub fn remove(&mut self, idx: usize) -> ExprId {
        self.factors.remove(idx)
    }

    /// Retain only factors matching a predicate.
    pub fn retain<F>(&mut self, mut pred: F)
    where
        F: FnMut(ExprId) -> bool,
    {
        self.factors.retain(|factor| pred(*factor));
    }
}

// ============================================================================
// Convenience Helpers (for migrating from flatten_add*/flatten_mul*)
// ============================================================================

/// Flatten an Add chain into terms, ignoring signs.
///
/// This is a drop-in replacement for local `flatten_add()` functions.
/// Uses `AddView` internally for shape-independence and __hold transparency.
///
/// # Example
/// ```ignore
/// let terms = add_terms_no_sign(ctx, expr);
/// for term in terms { ... }
/// ```
pub fn add_terms_no_sign(ctx: &Context, root: ExprId) -> SmallVec<[ExprId; 8]> {
    AddView::from_expr(ctx, root)
        .terms
        .into_iter()
        .map(|(t, _)| t)
        .collect()
}

/// Flatten an Add/Sub/Neg chain into signed terms.
///
/// This is a drop-in replacement for `flatten_add_sub_chain()` and
/// `flatten_add_signed()` functions.
///
/// # Example
/// ```ignore
/// let terms = add_terms_signed(ctx, expr);
/// for (term, sign) in terms { ... }
/// ```
pub fn add_terms_signed(ctx: &Context, root: ExprId) -> SmallVec<[(ExprId, Sign); 8]> {
    AddView::from_expr(ctx, root).terms
}

/// Flatten a Mul chain into factors.
///
/// This is a drop-in replacement for local `flatten_mul()` functions.
/// Uses `MulView` internally for shape-independence and __hold transparency.
pub fn mul_factors(ctx: &Context, root: ExprId) -> SmallVec<[ExprId; 8]> {
    MulView::from_expr(ctx, root).factors
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Helper: Convert AddView terms to a multiset (term -> net sign count)
    fn terms_to_multiset(_ctx: &Context, view: &AddView) -> HashMap<ExprId, i32> {
        let mut map = HashMap::new();
        for &(term, sign) in &view.terms {
            *map.entry(term).or_insert(0) += sign.to_i32();
        }
        // Normalize away zero-sum terms
        map.retain(|_, v| *v != 0);
        map
    }

    /// Helper: Convert MulView factors to a multiset (factor -> count)
    fn factors_to_multiset(view: &MulView) -> HashMap<ExprId, i32> {
        let mut map = HashMap::new();
        for &factor in &view.factors {
            *map.entry(factor).or_insert(0) += 1;
        }
        map
    }

    #[test]
    fn test_add_view_shape_independence() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let d = ctx.var("d");

        // Build same sum with different shapes (using add_raw to bypass canonicalization)
        // Left-associative: ((a+b)+c)+d
        let ab = ctx.add_raw(Expr::Add(a, b));
        let abc = ctx.add_raw(Expr::Add(ab, c));
        let left_assoc = ctx.add_raw(Expr::Add(abc, d));

        // Right-associative: a+(b+(c+d))
        let cd = ctx.add_raw(Expr::Add(c, d));
        let bcd = ctx.add_raw(Expr::Add(b, cd));
        let right_assoc = ctx.add_raw(Expr::Add(a, bcd));

        // Balanced: (a+b)+(c+d)
        let ab2 = ctx.add_raw(Expr::Add(a, b));
        let cd2 = ctx.add_raw(Expr::Add(c, d));
        let balanced = ctx.add_raw(Expr::Add(ab2, cd2));

        // All should produce same terms as multiset
        let view1 = AddView::from_expr(&ctx, left_assoc);
        let view2 = AddView::from_expr(&ctx, right_assoc);
        let view3 = AddView::from_expr(&ctx, balanced);

        let ms1 = terms_to_multiset(&ctx, &view1);
        let ms2 = terms_to_multiset(&ctx, &view2);
        let ms3 = terms_to_multiset(&ctx, &view3);

        assert_eq!(ms1, ms2, "Left-assoc vs right-assoc should have same terms");
        assert_eq!(ms2, ms3, "Right-assoc vs balanced should have same terms");
        assert_eq!(view1.len(), 4);
        assert_eq!(view2.len(), 4);
        assert_eq!(view3.len(), 4);
    }

    #[test]
    fn test_add_view_with_subtraction() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");

        // a - b represented as Sub
        let sub_expr = ctx.add_raw(Expr::Sub(a, b));
        let view = AddView::from_expr(&ctx, sub_expr);

        assert_eq!(view.len(), 2);

        // Check that a is positive and b is negative
        let ms = terms_to_multiset(&ctx, &view);
        assert_eq!(ms.get(&a), Some(&1), "a should have sign +1");
        assert_eq!(ms.get(&b), Some(&-1), "b should have sign -1");
    }

    #[test]
    fn test_add_view_single_term() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        // Single variable (not an Add)
        let view = AddView::from_expr(&ctx, x);

        assert_eq!(view.len(), 1);
        assert_eq!(view.terms[0], (x, Sign::Pos));
    }

    #[test]
    fn test_add_view_rebuild_roundtrip() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");

        // Original: a + b + c
        let bc = ctx.add(Expr::Add(b, c));
        let original = ctx.add(Expr::Add(a, bc));

        // Roundtrip
        let view = AddView::from_expr(&ctx, original);
        let rebuilt = view.rebuild(&mut ctx);

        // Rebuild as view again - should have same terms
        let view2 = AddView::from_expr(&ctx, rebuilt);
        assert_eq!(view2.len(), 3);
    }

    #[test]
    fn test_mul_view_shape_independence() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");

        // Different shapes
        let ab = ctx.add_raw(Expr::Mul(a, b));
        let left = ctx.add_raw(Expr::Mul(ab, c));
        let bc = ctx.add_raw(Expr::Mul(b, c));
        let right = ctx.add_raw(Expr::Mul(a, bc));

        let view1 = MulView::from_expr(&ctx, left);
        let view2 = MulView::from_expr(&ctx, right);

        let ms1 = factors_to_multiset(&view1);
        let ms2 = factors_to_multiset(&view2);

        assert_eq!(ms1, ms2, "Different shapes should have same factors");
        assert!(view1.commutative, "Scalar factors should be commutative");
        assert!(view2.commutative);
    }

    #[test]
    fn test_mul_view_with_matrix() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let two = ctx.num(2);
        let three = ctx.num(3);
        let four = ctx.num(4);
        let matrix = ctx.matrix(2, 2, vec![one, two, three, four]).unwrap();

        // x * M
        let xm = ctx.add_raw(Expr::Mul(x, matrix));
        let view = MulView::from_expr(&ctx, xm);

        assert_eq!(view.len(), 2);
        assert!(
            !view.commutative,
            "Matrix expression should be non-commutative"
        );
    }

    #[test]
    fn test_mul_view_rebuild_respects_commutativity() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let three = ctx.num(3);
        let four = ctx.num(4);
        let mat_a = ctx.matrix(2, 2, vec![one, two, three, four]).unwrap();
        let mat_b = ctx.matrix(2, 2, vec![four, three, two, one]).unwrap();

        // A * B (non-commutative)
        let ab = ctx.add_raw(Expr::Mul(mat_a, mat_b));
        let view_ab = MulView::from_expr(&ctx, ab);
        let rebuilt_ab = view_ab.clone().rebuild(&mut ctx);

        // B * A (non-commutative)
        let ba = ctx.add_raw(Expr::Mul(mat_b, mat_a));
        let view_ba = MulView::from_expr(&ctx, ba);
        let rebuilt_ba = view_ba.clone().rebuild(&mut ctx);

        // Rebuilt expressions should preserve order (not be equal)
        let final_ab = MulView::from_expr(&ctx, rebuilt_ab);
        let final_ba = MulView::from_expr(&ctx, rebuilt_ba);

        // Factor order should be preserved
        assert_eq!(final_ab.factors[0], mat_a);
        assert_eq!(final_ab.factors[1], mat_b);
        assert_eq!(final_ba.factors[0], mat_b);
        assert_eq!(final_ba.factors[1], mat_a);
    }

    #[test]
    fn test_add_view_empty_rebuild() {
        let mut ctx = Context::new();
        let view = AddView {
            root: ctx.num(0),
            terms: SmallVec::new(),
        };
        let rebuilt = view.rebuild(&mut ctx);

        // Should produce 0
        assert!(matches!(ctx.get(rebuilt), Expr::Number(n) if n.to_integer() == 0.into()));
    }

    #[test]
    fn test_mul_view_empty_rebuild() {
        let mut ctx = Context::new();
        let view = MulView {
            root: ctx.num(1),
            factors: SmallVec::new(),
            commutative: true,
        };
        let rebuilt = view.rebuild(&mut ctx);

        // Should produce 1
        assert!(matches!(ctx.get(rebuilt), Expr::Number(n) if n.to_integer() == 1.into()));
    }

    #[test]
    fn test_add_view_find_and_remove() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");

        let bc = ctx.add(Expr::Add(b, c));
        let expr = ctx.add(Expr::Add(a, bc));
        let mut view = AddView::from_expr(&ctx, expr);

        // Find 'b'
        let found = view.find(&ctx, |_ctx, term, _sign| term == b);
        assert_eq!(found.len(), 1);

        // Remove 'b'
        let removed = view.remove(found[0]);
        assert_eq!(removed.0, b);
        assert_eq!(view.len(), 2);
    }

    #[test]
    fn test_build_balanced_add_creates_balanced_tree() {
        let mut ctx = Context::new();
        let terms: Vec<ExprId> = (0..8).map(|i| ctx.var(&format!("x{}", i))).collect();

        let result = build_balanced_add(&mut ctx, &terms);

        // Flatten back and verify we get all 8 terms
        let view = AddView::from_expr(&ctx, result);
        assert_eq!(view.len(), 8);

        // Verify all original terms are present
        for term in &terms {
            assert!(view.terms.iter().any(|(t, _)| t == term));
        }
    }

    #[test]
    fn test_build_balanced_mul_creates_balanced_tree() {
        let mut ctx = Context::new();
        let factors: Vec<ExprId> = (0..4).map(|i| ctx.var(&format!("y{}", i))).collect();

        let result = build_balanced_mul(&mut ctx, &factors);

        // Flatten back and verify we get all 4 factors
        let view = MulView::from_expr(&ctx, result);
        assert_eq!(view.len(), 4);

        // Verify all original factors are present
        for factor in &factors {
            assert!(view.factors.iter().any(|f| f == factor));
        }
    }
}
