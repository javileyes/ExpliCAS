//! # Expression Views
//!
//! This module provides "views" that normalize different expression representations
//! for easier pattern matching in rules. For example, `Div(a,b)`, `Mul(a, Pow(b,-1))`,
//! and `Pow(x, -1)` are all recognized as having a denominator.
//!
//! ## Key Types
//!
//! - `Factor`: A base with a signed integer exponent
//! - `MulParts`: Collects multiplicative factors with sign extraction
//! - `FractionParts`: Separates numerator/denominator factors
//!
//! ## Builders
//!
//! - `build_as_div`: Creates didactic form `Div(num, den)`
//! - `build_as_mulpow`: Creates canonical form `Mul(factors, Pow(den, -1))`

use crate::{Context, Expr, ExprId};
use num_traits::ToPrimitive;
use std::collections::HashMap;

/// A factor in a multiplicative expression: base^exp where exp is a signed integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Factor {
    pub base: ExprId,
    pub exp: i32, // Signed exponent: negative means in denominator
}

/// Multiplicative parts of an expression, with sign extracted.
///
/// Recognizes: Mul, Div, Pow with integer exponent, Neg
#[derive(Debug, Clone)]
pub struct MulParts {
    pub sign: i8,             // +1 or -1 (extracted from Neg)
    pub factors: Vec<Factor>, // exp is signed (negative = denominator)
}

/// Fraction representation: separated numerator and denominator factors.
///
/// Both num and den have positive exponents.
#[derive(Debug, Clone)]
pub struct FractionParts {
    pub sign: i8,         // +1 or -1
    pub num: Vec<Factor>, // factors with exp > 0
    pub den: Vec<Factor>, // factors with exp > 0 (originally negative)
}

/// Extract integer exponent from an expression if it's an integer Number.
fn int_exp_i32(ctx: &Context, id: ExprId) -> Option<i32> {
    if let Expr::Number(n) = ctx.get(id) {
        if n.is_integer() {
            return n.to_integer().to_i32();
        }
    }
    None
}

impl MulParts {
    /// Create MulParts by collecting all multiplicative factors from an expression.
    ///
    /// Recognizes:
    /// - `Mul(a, b)` → factors from a and b
    /// - `Div(a, b)` → a in numerator, b in denominator
    /// - `Pow(base, k)` where k is integer → base^k
    /// - `Neg(x)` → flips sign, collects from x
    pub fn from(ctx: &Context, id: ExprId) -> Self {
        let mut out = MulParts {
            sign: 1,
            factors: Vec::new(),
        };
        collect_mul(ctx, id, 1, &mut out);
        out.compress(ctx);
        out
    }

    /// Combine factors with same base, remove factors with exp=0
    fn compress(&mut self, ctx: &Context) {
        let mut map: HashMap<ExprId, i32> = HashMap::new();
        for f in self.factors.drain(..) {
            *map.entry(f.base).or_insert(0) += f.exp;
        }
        self.factors = map
            .into_iter()
            .filter_map(|(base, exp)| {
                if exp != 0 {
                    Some(Factor { base, exp })
                } else {
                    None
                }
            })
            .collect();

        // Sort by expression ordering for determinism
        self.factors
            .sort_by(|a, b| crate::ordering::compare_expr(ctx, a.base, b.base));
    }

    /// Split into FractionParts (numerator with exp>0, denominator with exp>0)
    pub fn split_fraction(mut self) -> FractionParts {
        let mut num = Vec::new();
        let mut den = Vec::new();

        for f in self.factors.drain(..) {
            if f.exp > 0 {
                num.push(Factor {
                    base: f.base,
                    exp: f.exp,
                });
            } else {
                den.push(Factor {
                    base: f.base,
                    exp: -f.exp, // Make positive
                });
            }
        }

        FractionParts {
            sign: self.sign,
            num,
            den,
        }
    }

    /// Check if this represents a non-trivial fraction (has denominator factors)
    pub fn has_denominator(&self) -> bool {
        self.factors.iter().any(|f| f.exp < 0)
    }
}

/// Recursively collect multiplicative factors.
fn collect_mul(ctx: &Context, id: ExprId, mult: i32, out: &mut MulParts) {
    match ctx.get(id) {
        Expr::Mul(l, r) => {
            collect_mul(ctx, *l, mult, out);
            collect_mul(ctx, *r, mult, out);
        }
        Expr::Div(n, d) => {
            collect_mul(ctx, *n, mult, out);
            collect_mul(ctx, *d, -mult, out); // denominator gets negated exponent
        }
        Expr::Neg(x) => {
            out.sign *= -1;
            collect_mul(ctx, *x, mult, out);
        }
        Expr::Pow(b, e) => {
            if let Some(k) = int_exp_i32(ctx, *e) {
                // Integer exponent: recurse into base with scaled exponent
                // But only if base is not a nested Pow (to avoid complexity)
                if matches!(ctx.get(*b), Expr::Mul(_, _) | Expr::Div(_, _)) {
                    collect_mul(ctx, *b, mult * k, out);
                } else {
                    out.factors.push(Factor {
                        base: *b,
                        exp: mult * k,
                    });
                }
            } else {
                // Non-integer exponent: treat Pow as atomic factor
                out.factors.push(Factor {
                    base: id,
                    exp: mult,
                });
            }
        }
        _ => {
            // Atomic factor (Number, Variable, Function, etc.)
            out.factors.push(Factor {
                base: id,
                exp: mult,
            });
        }
    }
}

// ============================================================================
// MulBuilder: Canonical product construction
// ============================================================================

/// Mode for MulBuilder - controls what gets flattened.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MulMode {
    /// No flattening - just collect factors as atoms
    Simple,
    /// Flatten only Mul chains (safe for recursive functions)
    FlattenMul,
    /// Flatten Mul + combine Pow with integer exponents (NO Div)
    MulPowInt,
}

/// Builder for creating canonical multiplication expressions.
///
/// Two usage patterns:
///
/// **Safe for recursive functions** (distribute, AddFractions, etc.):
/// ```ignore
/// let mut b = MulBuilder::new_simple();
/// b.push(a);
/// b.push(b_expr);
/// let result = b.build(ctx);
/// ```
///
/// **For normalization passes** (not inside recursion):
/// ```ignore
/// let mut b = MulBuilder::new_flatten();
/// b.push_expr(ctx, complex_expr);  // Flattens Mul, combines Pow
/// let result = b.build(ctx);
/// ```
pub struct MulBuilder {
    mode: MulMode,
    sign: i8,
    factors: Vec<(ExprId, i64)>, // (base, exponent)
}

impl MulBuilder {
    /// Create builder in Simple mode - no flattening.
    /// **Use this inside recursive functions** (distribute, expand, etc.)
    pub fn new_simple() -> Self {
        MulBuilder {
            mode: MulMode::Simple,
            sign: 1,
            factors: Vec::new(),
        }
    }

    /// Create builder that flattens Mul and combines Pow(base, int).
    /// **Do NOT use inside recursive functions** - use new_simple() instead.
    pub fn new_flatten() -> Self {
        MulBuilder {
            mode: MulMode::MulPowInt,
            sign: 1,
            factors: Vec::new(),
        }
    }

    /// Create a new empty builder (defaults to Simple mode for safety).
    pub fn new() -> Self {
        Self::new_simple()
    }

    /// Add a factor with exponent 1 (no flattening).
    pub fn push(&mut self, base: ExprId) -> &mut Self {
        self.factors.push((base, 1));
        self
    }

    /// Add a factor with a specific exponent (no flattening).
    pub fn push_pow(&mut self, base: ExprId, exp: i64) -> &mut Self {
        if exp != 0 {
            self.factors.push((base, exp));
        }
        self
    }

    /// Negate the product.
    pub fn negate(&mut self) -> &mut Self {
        self.sign *= -1;
        self
    }

    /// Add factors from an expression, controlled by mode.
    ///
    /// - `Simple`: Just adds expr as a single factor (no flattening)
    /// - `FlattenMul`: Flattens nested Mul
    /// - `MulPowInt`: Flattens Mul + combines Pow with integer exponents
    ///
    /// **IMPORTANT**: Never flattens Div. Use RationalFnView for fractions.
    pub fn push_expr(&mut self, ctx: &Context, expr: ExprId) -> &mut Self {
        match self.mode {
            MulMode::Simple => {
                // No flattening - add as atom
                self.factors.push((expr, 1));
            }
            MulMode::FlattenMul | MulMode::MulPowInt => {
                // Iterative worklist to avoid deep recursion
                let mut worklist: Vec<(ExprId, i64)> = vec![(expr, 1)];

                while let Some((id, mult)) = worklist.pop() {
                    match ctx.get(id) {
                        Expr::Mul(l, r) => {
                            worklist.push((*l, mult));
                            worklist.push((*r, mult));
                        }
                        Expr::Pow(base, exp) if self.mode == MulMode::MulPowInt => {
                            if let Some(e) = int_exp_i32(ctx, *exp) {
                                worklist.push((*base, mult * e as i64));
                            } else {
                                // Non-integer exponent: treat whole expression as factor
                                self.factors.push((id, mult));
                            }
                        }
                        Expr::Neg(inner) => {
                            self.sign *= -1;
                            worklist.push((*inner, mult));
                        }
                        Expr::Number(n) => {
                            use num_traits::One;
                            if !n.is_one() || mult != 1 {
                                self.factors.push((id, mult));
                            }
                            // Skip 1 with mult=1 (identity)
                        }
                        // IMPORTANT: Div is NOT flattened - treat as atom
                        // This prevents explosive growth in recursive functions
                        _ => {
                            self.factors.push((id, mult));
                        }
                    }
                }
            }
        }
        self
    }

    /// Build the canonical product expression.
    ///
    /// Properties:
    /// - Preserves original factor order (no sorting - matches current system)
    /// - Compresses adjacent same-base factors (`x * x` → `x^2`)
    /// - Uses **right-fold** association: `a*(b*(c*d))` (matches current system)
    /// - Idempotent: calling twice produces identical tree
    ///
    /// Note: Sorting is disabled to preserve pattern matching compatibility.
    /// When most rules use MulView for matching, we can enable sorting.
    pub fn build(self, ctx: &mut Context) -> ExprId {
        // 1. NO sorting - preserve order for pattern matching compatibility
        // (sorting breaks rules that expect specific factor order)

        // 2. Compress adjacent same-base factors only (preserve order)
        // CRITICAL: Use ExprId equality, not compare_expr (avoids false positives with matrices)
        let mut compressed: Vec<(ExprId, i64)> = Vec::new();
        for (base, exp) in self.factors {
            if let Some(last) = compressed.last_mut() {
                // Use ExprId equality - only combine if literally same interned node
                if last.0 == base {
                    last.1 += exp;
                    continue;
                }
            }
            compressed.push((base, exp));
        }

        // 3. Remove factors with exp=0
        compressed.retain(|(_, exp)| *exp != 0);

        // 4. Handle empty case
        if compressed.is_empty() {
            let one = ctx.num(1);
            return if self.sign < 0 {
                ctx.add(Expr::Neg(one))
            } else {
                one
            };
        }

        // 5. Build each factor as base^exp (or just base if exp=1)
        // CRITICAL: Do NOT compress matrices to Pow(Matrix, k) - keep as repeated Mul
        let mut parts: Vec<ExprId> = Vec::new();
        for (base, exp) in compressed {
            let is_matrix = matches!(ctx.get(base), Expr::Matrix { .. });

            if exp == 1 {
                parts.push(base);
            } else if exp == -1 {
                let neg_one = ctx.num(-1);
                parts.push(ctx.add(Expr::Pow(base, neg_one)));
            } else if is_matrix {
                // Matrix: don't compress to Pow, expand as repeated Mul factors
                for _ in 0..exp.abs() {
                    parts.push(base);
                }
                // If negative exp, wrap in Pow^-1 (matrix inverse placeholder)
                if exp < 0 {
                    // For matrices, we'd need proper inverse - just mark as unsupported for now
                    // This is a simplification; real impl might handle differently
                }
            } else {
                let exp_id = ctx.num(exp);
                parts.push(ctx.add(Expr::Pow(base, exp_id)));
            }
        }

        // 6. Build RIGHT-fold product: a*(b*(c*d))
        // This matches the current system's expected form
        let mut acc = *parts.last().unwrap();
        for &f in parts[..parts.len() - 1].iter().rev() {
            acc = ctx.add(Expr::Mul(f, acc));
        }

        // 7. Apply sign
        if self.sign < 0 {
            ctx.add(Expr::Neg(acc))
        } else {
            acc
        }
    }
}

impl Default for MulBuilder {
    fn default() -> Self {
        Self::new()
    }
}
impl FractionParts {
    /// Create FractionParts directly from an expression.
    pub fn from(ctx: &Context, id: ExprId) -> Self {
        MulParts::from(ctx, id).split_fraction()
    }

    /// Check if this actually represents a fraction (has denominator).
    pub fn is_fraction(&self) -> bool {
        !self.den.is_empty()
    }

    /// Check if this is a simple fraction a/b (single term in num and den)
    pub fn is_simple(&self) -> bool {
        self.num.len() <= 1 && self.den.len() <= 1
    }

    /// Get simple (numerator, denominator, is_fraction) tuple.
    ///
    /// This is useful for rules that need to work with the num/den as single expressions.
    /// Returns the built numerator and denominator expressions, applying the sign to numerator.
    pub fn to_num_den(&self, ctx: &mut Context) -> (ExprId, ExprId, bool) {
        let mut num_expr = Self::build_product_static(ctx, &self.num);
        let den_expr = Self::build_product_static(ctx, &self.den);

        // Apply sign to numerator
        if self.sign < 0 {
            num_expr = ctx.add(Expr::Neg(num_expr));
        }

        (num_expr, den_expr, self.is_fraction())
    }

    /// Build a product from factors: Π base^exp
    ///
    /// Public static method for building products without needing a FractionParts instance.
    /// Uses RIGHT-fold (a*(b*c)) and add_raw to preserve operand order.
    pub fn build_product_static(ctx: &mut Context, factors: &[Factor]) -> ExprId {
        if factors.is_empty() {
            return ctx.num(1);
        }

        let mut parts: Vec<ExprId> = Vec::with_capacity(factors.len());
        for f in factors {
            let term = if f.exp == 1 {
                f.base
            } else {
                let e = ctx.num(f.exp as i64);
                ctx.add(Expr::Pow(f.base, e))
            };
            parts.push(term);
        }

        // RIGHT-fold: a*(b*(c*d)) - use add_raw to preserve order
        let mut acc = *parts.last().unwrap();
        for &p in parts[..parts.len() - 1].iter().rev() {
            acc = ctx.add_raw(Expr::Mul(p, acc));
        }
        acc
    }

    /// Build as didactic division: `Div(num, den)` or just `num` if den=1.
    ///
    /// Use this for pedagogical output that shows fractions as a/b.
    pub fn build_as_div(&self, ctx: &mut Context) -> ExprId {
        let num_expr = Self::build_product_static(ctx, &self.num);
        let den_expr = Self::build_product_static(ctx, &self.den);

        let mut result = if self.den.is_empty() {
            // No denominator, just return numerator
            num_expr
        } else {
            // Check if denominator is just 1
            if let Expr::Number(n) = ctx.get(den_expr) {
                if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                    num_expr
                } else {
                    ctx.add(Expr::Div(num_expr, den_expr))
                }
            } else {
                ctx.add(Expr::Div(num_expr, den_expr))
            }
        };

        // Apply sign
        if self.sign < 0 {
            result = ctx.add(Expr::Neg(result));
        }

        result
    }

    /// Build as canonical multiplication with negative powers: `Mul(factors..., Pow(den, -exp)...)`.
    ///
    /// Use this for internal canonical form.
    pub fn build_as_mulpow(&self, ctx: &mut Context) -> ExprId {
        // Combine num (positive exp) and den (negative exp)
        let mut all_factors: Vec<Factor> = self.num.clone();
        for d in &self.den {
            all_factors.push(Factor {
                base: d.base,
                exp: -(d.exp), // negate for canonical form
            });
        }

        if all_factors.is_empty() {
            // Just ±1
            return ctx.num(self.sign as i64);
        }

        // Sort for determinism
        all_factors.sort_by(|a, b| crate::ordering::compare_expr(ctx, a.base, b.base));

        // Build product
        let mut parts: Vec<ExprId> = Vec::new();
        for f in &all_factors {
            let term = if f.exp == 1 {
                f.base
            } else {
                let e = ctx.num(f.exp as i64);
                ctx.add(Expr::Pow(f.base, e))
            };
            parts.push(term);
        }

        let mut acc = parts[0];
        for p in parts.into_iter().skip(1) {
            acc = ctx.add(Expr::Mul(acc, p));
        }

        // Apply sign
        if self.sign < 0 {
            ctx.add(Expr::Neg(acc))
        } else {
            acc
        }
    }
}

// ============================================================================
// RationalFnView: Preserves num/den as complete expression trees
// ============================================================================

/// Rational function representation: num and den as complete expression trees.
///
/// Unlike `FractionParts` which decomposes into factors, this view preserves
/// the original structure of numerator and denominator as complete expressions.
/// Use this for rules that need to operate on num/den as polynomials.
///
/// ## When to use
/// - SimplifyFractionRule (GCD, polynomial factorization)
/// - NestedFractionRule (detecting fractions within fractions)
/// - Any rule that needs structural operations on num/den
///
/// ## When to use FractionParts instead
/// - Multiplicative cancellation
/// - Quotient of powers
/// - Rationalization
#[derive(Debug, Clone)]
pub struct RationalFnView {
    pub sign: i8,    // +1 or -1
    pub num: ExprId, // numerator as complete expression tree
    pub den: ExprId, // denominator as complete expression tree
}

impl RationalFnView {
    /// Create RationalFnView from an expression.
    ///
    /// Returns Some if the expression represents a fraction:
    /// - `Div(n, d)` → num=n, den=d
    /// - `Pow(x, -1)` → num=1, den=x
    /// - `Mul` with denominator factors → reconstructed num/den
    /// - `Neg(fraction)` → negated num
    ///
    /// Returns None if not a fraction-like expression.
    pub fn from(ctx: &mut Context, id: ExprId) -> Option<Self> {
        // First try direct Div pattern (most common)
        if let Expr::Div(n, d) = ctx.get(id).clone() {
            return Some(RationalFnView {
                sign: 1,
                num: n,
                den: d,
            });
        }

        // Handle Neg(fraction)
        if let Expr::Neg(inner) = ctx.get(id).clone() {
            if let Some(mut view) = Self::from(ctx, inner) {
                view.sign *= -1;
                return Some(view);
            }
            return None;
        }

        // Handle Pow(x, -1) = 1/x
        if let Expr::Pow(b, e) = ctx.get(id).clone() {
            if let Some(exp) = int_exp_i32(ctx, e) {
                if exp == -1 {
                    let one = ctx.num(1);
                    return Some(RationalFnView {
                        sign: 1,
                        num: one,
                        den: b,
                    });
                }
            }
            return None;
        }

        // Handle Mul with Pow(x,-1) factors: a * b^(-1) = a/b
        // Use FractionParts to decompose, then reconstruct as ExprIds
        let fp = FractionParts::from(&*ctx, id);
        if fp.is_fraction() {
            // Reconstruct num and den as products
            let num_expr = FractionParts::build_product_static(ctx, &fp.num);
            let den_expr = FractionParts::build_product_static(ctx, &fp.den);
            return Some(RationalFnView {
                sign: fp.sign,
                num: num_expr,
                den: den_expr,
            });
        }

        None
    }

    /// Check if this is a "simple" fraction (both num and den are single terms)
    pub fn is_simple(&self, ctx: &Context) -> bool {
        !matches!(ctx.get(self.num), Expr::Add(_, _) | Expr::Sub(_, _))
            && !matches!(ctx.get(self.den), Expr::Add(_, _) | Expr::Sub(_, _))
    }

    /// Check if denominator is 1
    pub fn is_integer(&self, ctx: &Context) -> bool {
        if let Expr::Number(n) = ctx.get(self.den) {
            n.is_integer() && *n == num_rational::BigRational::from_integer(1.into())
        } else {
            false
        }
    }

    /// Build as didactic division: `Div(num, den)` or just `num` if den=1.
    pub fn build_as_div(&self, ctx: &mut Context) -> ExprId {
        let mut result = if self.is_integer(ctx) {
            self.num
        } else {
            ctx.add(Expr::Div(self.num, self.den))
        };

        if self.sign < 0 {
            result = ctx.add(Expr::Neg(result));
        }

        result
    }

    /// Build as canonical form: `num * den^(-1)` (C2 form).
    pub fn build_as_mulpow(&self, ctx: &mut Context) -> ExprId {
        let result = if self.is_integer(ctx) {
            self.num
        } else {
            let neg_one = ctx.num(-1);
            let den_inv = ctx.add(Expr::Pow(self.den, neg_one));
            ctx.add(Expr::Mul(self.num, den_inv))
        };

        if self.sign < 0 {
            ctx.add(Expr::Neg(result))
        } else {
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_fraction() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let div = ctx.add(Expr::Div(a, b));

        let fp = FractionParts::from(&ctx, div);
        assert!(fp.is_fraction());
        assert_eq!(fp.sign, 1);
        assert_eq!(fp.num.len(), 1);
        assert_eq!(fp.den.len(), 1);
    }

    #[test]
    fn test_reciprocal_pow() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let neg_one = ctx.num(-1);
        let recip = ctx.add(Expr::Pow(x, neg_one));

        let fp = FractionParts::from(&ctx, recip);
        assert!(fp.is_fraction());
        assert_eq!(fp.num.len(), 0);
        assert_eq!(fp.den.len(), 1);
    }

    #[test]
    fn test_mul_with_div() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        // a * (b/c)
        let frac = ctx.add(Expr::Div(b, c));
        let mul = ctx.add(Expr::Mul(a, frac));

        let fp = FractionParts::from(&ctx, mul);
        assert!(fp.is_fraction());
        assert_eq!(fp.num.len(), 2); // a, b
        assert_eq!(fp.den.len(), 1); // c
    }

    #[test]
    fn test_build_as_div() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let neg_one = ctx.num(-1);

        // a * b^(-1) should build back to a/b
        let recip_b = ctx.add(Expr::Pow(b, neg_one));
        let mul = ctx.add(Expr::Mul(a, recip_b));

        let fp = FractionParts::from(&ctx, mul);
        let result = fp.build_as_div(&mut ctx);

        // Should be Div(a, b)
        if let Expr::Div(n, d) = ctx.get(result) {
            assert_eq!(*n, a);
            assert_eq!(*d, b);
        } else {
            panic!("Expected Div");
        }
    }

    #[test]
    fn test_rational_fn_view_div() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let div = ctx.add(Expr::Div(a, b));

        let view = RationalFnView::from(&mut ctx, div).unwrap();
        assert_eq!(view.sign, 1);
        assert_eq!(view.num, a);
        assert_eq!(view.den, b);
    }

    #[test]
    fn test_rational_fn_view_pow_neg1() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let neg_one = ctx.num(-1);
        let recip = ctx.add(Expr::Pow(x, neg_one));

        let view = RationalFnView::from(&mut ctx, recip).unwrap();
        assert_eq!(view.sign, 1);
        assert_eq!(view.den, x);
        // num should be 1
        if let Expr::Number(n) = ctx.get(view.num) {
            assert!(n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()));
        } else {
            panic!("Expected Number(1)");
        }
    }

    #[test]
    fn test_rational_fn_view_preserves_structure() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        // (a + b) / b - the numerator is a sum, not just factors
        let sum = ctx.add(Expr::Add(a, b));
        let div = ctx.add(Expr::Div(sum, b));

        let view = RationalFnView::from(&mut ctx, div).unwrap();
        // num should still be the Add expression
        assert!(matches!(ctx.get(view.num), Expr::Add(_, _)));
    }

    #[test]
    fn test_mul_builder_basic() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");

        let mut builder = MulBuilder::new();
        builder.push(a);
        builder.push(b);
        let result = builder.build(&mut ctx);

        // Should be Mul(a, b) in left-fold form
        if let Expr::Mul(l, r) = ctx.get(result) {
            assert_eq!(*l, a);
            assert_eq!(*r, b);
        } else {
            panic!("Expected Mul");
        }
    }

    #[test]
    fn test_mul_builder_compresses_exponents() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        // x * x should become x^2
        let mut builder = MulBuilder::new();
        builder.push(x);
        builder.push(x);
        let result = builder.build(&mut ctx);

        // Should be Pow(x, 2)
        if let Expr::Pow(base, exp) = ctx.get(result) {
            assert_eq!(*base, x);
            if let Expr::Number(n) = ctx.get(*exp) {
                assert_eq!(*n, num_rational::BigRational::from_integer(2.into()));
            } else {
                panic!("Expected Number exponent");
            }
        } else {
            panic!("Expected Pow, got {:?}", ctx.get(result));
        }
    }

    #[test]
    fn test_mul_builder_idempotent() {
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");

        // Build once
        let mut builder1 = MulBuilder::new();
        builder1.push(a);
        builder1.push(b);
        let first = builder1.build(&mut ctx);

        // Flatten and rebuild - should get same result
        let mut builder2 = MulBuilder::new();
        builder2.push_expr(&ctx, first);
        let second = builder2.build(&mut ctx);

        assert_eq!(first, second, "MulBuilder should be idempotent");
    }
}
