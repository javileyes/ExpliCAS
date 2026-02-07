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

/// Extract a BigRational constant from an expression, recognizing various forms.
///
/// Supports (with depth limit to prevent explosion):
/// - `Number(q)` - direct rational
/// - `Neg(x)` - negated rational
/// - `Div(a, b)` - fraction of rationals
///
/// Returns None if the expression isn't a constant rational.
pub fn as_rational_const(
    ctx: &Context,
    id: ExprId,
    depth: u8,
) -> Option<num_rational::BigRational> {
    use num_rational::BigRational;

    if depth == 0 {
        return None;
    }

    match ctx.get(id) {
        // Direct rational number
        Expr::Number(n) => Some(n.clone()),

        // Negation: -x
        Expr::Neg(inner) => as_rational_const(ctx, *inner, depth - 1).map(|r| -r),

        // Division: a/b where both are rationals
        Expr::Div(num, den) => {
            let num_val = as_rational_const(ctx, *num, depth - 1)?;
            let den_val = as_rational_const(ctx, *den, depth - 1)?;
            // Avoid division by zero
            if den_val == BigRational::from_integer(0.into()) {
                return None;
            }
            Some(num_val / den_val)
        }

        // Other expressions are not constant rationals
        _ => None,
    }
}

/// Check if an expression is "surd-free" (contains no square roots).
///
/// Returns `false` if the expression contains:
/// - `Pow(_, exp)` where exp equals 1/2 (robust to Div(1,2))
/// - `Function("sqrt", _)`
///
/// Uses iterative worklist with budget to prevent explosion.
pub fn is_surd_free(ctx: &Context, id: ExprId, budget: usize) -> bool {
    use num_rational::BigRational;

    let half = BigRational::new(1.into(), 2.into());
    let mut worklist = vec![id];
    let mut visited = 0;

    while let Some(curr) = worklist.pop() {
        visited += 1;
        if visited > budget {
            // Conservative: if we hit budget, assume not surd-free
            return false;
        }

        match ctx.get(curr) {
            // Check for sqrt as Pow(_, 1/2)
            Expr::Pow(base, exp) => {
                if let Some(exp_val) = as_rational_const(ctx, *exp, 8) {
                    if exp_val == half {
                        return false; // Found a surd!
                    }
                }
                worklist.push(*base);
                worklist.push(*exp);
            }

            // Check for sqrt() function
            Expr::Function(fn_id, args) => {
                if ctx.sym_name(*fn_id) == "sqrt" {
                    return false; // Found a surd!
                }
                worklist.extend(args.iter().copied());
            }

            // Recurse into other expression types
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                worklist.push(*l);
                worklist.push(*r);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => worklist.push(*inner),
            Expr::Matrix { data, .. } => worklist.extend(data.iter().copied()),

            // Atoms are surd-free
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    true // No surds found
}

/// Count distinct numeric surds in an expression.
///
/// Returns the number of distinct numeric radicands found as `Pow(Number(n), 1/2)` or `sqrt(Number(n))`.
/// Used for multi-surd guard in Level 1/1.5 rationalization.
///
/// If budget is exceeded, returns a conservative high count (100) to block rationalization.
pub fn count_distinct_numeric_surds(ctx: &Context, id: ExprId, budget: usize) -> usize {
    use num_rational::BigRational;
    use std::collections::HashSet;

    let half = BigRational::new(1.into(), 2.into());
    let mut worklist = vec![id];
    let mut visited = 0;
    let mut distinct_radicands: HashSet<i64> = HashSet::new();

    while let Some(curr) = worklist.pop() {
        visited += 1;
        if visited > budget {
            // Conservative: if we hit budget, return high count to block
            return 100;
        }

        match ctx.get(curr) {
            // Check for Pow(Number(n), 1/2)
            Expr::Pow(base, exp) => {
                if let Some(exp_val) = as_rational_const(ctx, *exp, 8) {
                    if exp_val == half {
                        if let Expr::Number(n) = ctx.get(*base) {
                            if n.is_integer() {
                                if let Ok(radicand) = n.numer().try_into() {
                                    distinct_radicands.insert(radicand);
                                }
                            }
                        }
                    }
                }
                worklist.push(*base);
                worklist.push(*exp);
            }

            // Check for sqrt(Number(n))
            Expr::Function(fn_id, args) => {
                if ctx.sym_name(*fn_id) == "sqrt" && !args.is_empty() {
                    if let Expr::Number(n) = ctx.get(args[0]) {
                        if n.is_integer() {
                            if let Ok(radicand) = n.numer().try_into() {
                                distinct_radicands.insert(radicand);
                            }
                        }
                    }
                }
                worklist.extend(args.iter().copied());
            }

            // Recurse into other expression types
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                worklist.push(*l);
                worklist.push(*r);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => worklist.push(*inner),
            Expr::Matrix { data, .. } => worklist.extend(data.iter().copied()),

            // Atoms have no surds
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }

    distinct_radicands.len()
}

/// Check if an Add chain has all negative terms (for display sign normalization).
///
/// Returns `true` if every term in the Add chain is:
/// - `Neg(x)`
/// - `Number(n)` where n < 0
/// - `Mul(Number(n), _)` where n < 0
///
/// This is used to factor out a common `-1` for cleaner display:
/// `(-3) + (-2*√5)` → `-(3 + 2*√5)`
///
/// Only applies to genuine Add chains (not single terms).
pub fn has_all_negative_terms(ctx: &Context, id: ExprId) -> bool {
    use num_rational::BigRational;
    let zero = BigRational::from_integer(0.into());

    // Collect all terms in the Add chain
    let mut terms = Vec::new();
    collect_add_terms(ctx, id, &mut terms);

    // Need at least 2 terms for this to matter
    if terms.len() < 2 {
        return false;
    }

    // Check if ALL terms are negative
    for term in terms {
        if !is_negative_term(ctx, term, &zero) {
            return false;
        }
    }

    true
}

/// Collect all terms in an Add chain (flattening nested Adds).
///
/// __hold wrappers are transparent: collects terms from inside __hold.
fn collect_add_terms(ctx: &Context, id: ExprId, terms: &mut Vec<ExprId>) {
    // Unwrap __hold for transparency (algebra sees through holds)
    let id = crate::hold::unwrap_hold(ctx, id);
    match ctx.get(id) {
        Expr::Add(l, r) => {
            collect_add_terms(ctx, *l, terms);
            collect_add_terms(ctx, *r, terms);
        }
        _ => terms.push(id),
    }
}

/// Check if a single term is "negative" (starts with - or negative coefficient).
fn is_negative_term(ctx: &Context, id: ExprId, zero: &num_rational::BigRational) -> bool {
    match ctx.get(id) {
        // Explicit Neg
        Expr::Neg(_) => true,

        // Negative number
        Expr::Number(n) => n < zero,

        // Mul with negative leading factor
        Expr::Mul(l, _) => {
            if let Expr::Number(n) = ctx.get(*l) {
                n < zero
            } else {
                matches!(ctx.get(*l), Expr::Neg(_))
            }
        }

        _ => false,
    }
}

/// Negate a term for display purposes (return the absolute value).
///
/// Given a negative term, returns its positive counterpart's ExprId.
/// This is read-only (for display) - doesn't create new nodes.
pub fn get_term_absolute_value(ctx: &Context, id: ExprId) -> Option<ExprId> {
    match ctx.get(id) {
        // Neg(x) -> x
        Expr::Neg(inner) => Some(*inner),

        // Number(n < 0) -> would need to create new node (not supported here)
        Expr::Number(n) => {
            if n < &num_rational::BigRational::from_integer(0.into()) {
                None // Can't return absolute without creating new node
            } else {
                Some(id)
            }
        }

        // Mul(Neg(x), y) -> return id of inner product without neg
        // This case requires examining the structure
        Expr::Mul(l, r) => {
            if let Expr::Neg(inner) = ctx.get(*l) {
                // -x * y -> the positive part would need reconstruction
                // For now, we'll mark this as needing special handling
                let _ = (inner, r);
                None
            } else if let Expr::Number(n) = ctx.get(*l) {
                if n < &num_rational::BigRational::from_integer(0.into()) {
                    // Negative coefficient - can't extract without new node
                    None
                } else {
                    Some(id)
                }
            } else {
                Some(id)
            }
        }

        _ => Some(id),
    }
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

    /// Create MulParts only if expression is purely commutative (no matrices).
    ///
    /// Returns `None` if any factor is a Matrix, since matrix multiplication
    /// is not commutative and compress/reorder operations would be incorrect.
    pub fn from_commutative(ctx: &Context, id: ExprId) -> Option<Self> {
        let parts = Self::from(ctx, id);

        // Check if any factor base is a Matrix
        for f in &parts.factors {
            if matches!(ctx.get(f.base), Expr::Matrix { .. }) {
                return None;
            }
        }

        Some(parts)
    }
}

// ============================================================================
// MulChainView: Order-preserving multiplication linearization
// ============================================================================

/// Simple linearization of multiplication chain, preserving order.
///
/// Unlike MulParts, this:
/// - Does NOT parse Div or Pow
/// - Does NOT compress same-base factors
/// - PRESERVES factor order (works with matrices)
///
/// Use this when you need order-preserving iteration over Mul factors.
#[derive(Debug, Clone)]
pub struct MulChainView {
    pub factors: Vec<ExprId>,
}

impl MulChainView {
    /// Linearize a Mul chain preserving order.
    ///
    /// `a * (b * c)` → `[a, b, c]` (order preserved)
    pub fn from(ctx: &Context, id: ExprId) -> Self {
        let mut factors = Vec::new();
        Self::collect_factors(ctx, id, &mut factors);
        MulChainView { factors }
    }

    fn collect_factors(ctx: &Context, id: ExprId, out: &mut Vec<ExprId>) {
        // Unwrap __hold for transparency (algebra sees through holds)
        let id = crate::hold::unwrap_hold(ctx, id);
        match ctx.get(id) {
            Expr::Mul(l, r) => {
                Self::collect_factors(ctx, *l, out);
                Self::collect_factors(ctx, *r, out);
            }
            _ => {
                out.push(id);
            }
        }
    }

    /// Check if any factor is a matrix.
    pub fn contains_matrix(&self, ctx: &Context) -> bool {
        self.factors
            .iter()
            .any(|&f| matches!(ctx.get(f), Expr::Matrix { .. }))
    }

    /// Rebuild as right-associative Mul chain using add_raw.
    pub fn build_raw(self, ctx: &mut Context) -> ExprId {
        if let Some((&last, rest)) = self.factors.split_last() {
            let mut acc = last;
            for &f in rest.iter().rev() {
                acc = ctx.add_raw(Expr::Mul(f, acc));
            }
            acc
        } else {
            ctx.num(1)
        }
    }
}

/// Recursively collect multiplicative factors.
///
/// __hold wrappers are transparent: collects factors from inside __hold.
fn collect_mul(ctx: &Context, id: ExprId, mult: i32, out: &mut MulParts) {
    // Unwrap __hold for transparency (algebra sees through holds)
    let id = crate::hold::unwrap_hold(ctx, id);
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
        // Treat all numbers as atomic factors.
        // Previously, fractional numbers with numerator 1 or -1 were decomposed.
        Expr::Number(_n) => {
            out.factors.push(Factor {
                base: id,
                exp: mult,
            });
        }
        _ => {
            // Atomic factor (Variable, Function, etc.)
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
        let acc = if let Some((&last, rest)) = parts.split_last() {
            let mut acc = last;
            for &f in rest.iter().rev() {
                acc = ctx.add(Expr::Mul(f, acc));
            }
            acc
        } else {
            // parts is empty — all exponents cancelled
            return if self.sign < 0 {
                let one = ctx.num(1);
                ctx.add(Expr::Neg(one))
            } else {
                ctx.num(1)
            };
        };

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
        if let Some((&last, rest)) = parts.split_last() {
            let mut acc = last;
            for &p in rest.iter().rev() {
                acc = ctx.add_raw(Expr::Mul(p, acc));
            }
            acc
        } else {
            ctx.num(1)
        }
    }

    /// Build as didactic division: `Div(num, den)` or just `num` if den=1.
    ///
    /// Use this for pedagogical output that shows fractions as a/b.
    /// NOTE: Sign is applied to the numerator `(-a)/b` rather than wrapping `-(a/b)`
    /// to match canonical form and avoid infinite loops with canonicalization rules.
    pub fn build_as_div(&self, ctx: &mut Context) -> ExprId {
        let num_expr = Self::build_product_static(ctx, &self.num);
        let den_expr = Self::build_product_static(ctx, &self.den);

        // Apply sign to numerator BEFORE building the division
        // This produces (-a)/b instead of -(a/b), matching canonical form
        let signed_num = if self.sign < 0 {
            ctx.add(Expr::Neg(num_expr))
        } else {
            num_expr
        };

        if self.den.is_empty() {
            // No denominator, just return signed numerator
            signed_num
        } else {
            // Check if denominator is just 1
            if let Expr::Number(n) = ctx.get(den_expr) {
                if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                    signed_num
                } else {
                    ctx.add(Expr::Div(signed_num, den_expr))
                }
            } else {
                ctx.add(Expr::Div(signed_num, den_expr))
            }
        }
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

// ============================================================================
// SurdSumView: Recognize sums of rational + surds for rationalization
// ============================================================================

use num_rational::BigRational;

/// A surd term: coeff * √(radicand) where radicand is square-free.
///
/// Example: 3√2 → SurdAtom { radicand: 2, coeff: 3 }
#[derive(Debug, Clone, PartialEq)]
pub struct SurdAtom {
    /// Square-free positive integer under the radical
    pub radicand: i64,
    /// Rational coefficient (can be negative)
    pub coeff: BigRational,
}

/// Sum of surds: constant + Σ(coeff_i * √n_i)
///
/// Domain v1 (strict):
/// - Only `Pow(Number(n), Number(1/2))` with n > 0 integer
/// - Only rational coefficients (Number, Neg, Mul)
/// - No variables, matrices, or functions
///
/// Example: `1 + 2√3 + √5` → SurdSumView { constant: 1, surds: [SurdAtom(3,2), SurdAtom(5,1)] }
#[derive(Debug, Clone)]
pub struct SurdSumView {
    /// Rational constant part
    pub constant: BigRational,
    /// Surd terms, combined by radicand (normalized to square-free)
    pub surds: Vec<SurdAtom>,
}

impl SurdSumView {
    /// Try to interpret an expression as a sum of surds.
    ///
    /// Returns `None` if:
    /// - Expression contains variables
    /// - Expression contains matrices or functions
    /// - Radicand is not a positive integer
    /// - Any term is not recognizable as rational or surd
    pub fn from(ctx: &Context, id: ExprId) -> Option<Self> {
        let mut constant = BigRational::from_integer(0.into());
        let mut surd_map: HashMap<i64, BigRational> = HashMap::new();

        // Collect all additive terms
        let mut worklist = vec![(id, BigRational::from_integer(1.into()))]; // (expr, sign_coeff)

        while let Some((curr, outer_coeff)) = worklist.pop() {
            match ctx.get(curr) {
                // Reject immediately: matrices, variables, constants
                Expr::Matrix { .. } => return None,
                Expr::Variable(_) => return None,
                Expr::Constant(_) => return None, // π, e not allowed in v1

                // Addition: recurse into both sides
                Expr::Add(l, r) => {
                    worklist.push((*l, outer_coeff.clone()));
                    worklist.push((*r, outer_coeff));
                }

                // Subtraction: right side gets negative coeff
                Expr::Sub(l, r) => {
                    worklist.push((*l, outer_coeff.clone()));
                    worklist.push((*r, -outer_coeff));
                }

                // Negation: flip sign
                Expr::Neg(inner) => {
                    worklist.push((*inner, -outer_coeff));
                }

                // Pure number: add to constant
                Expr::Number(n) => {
                    constant += n.clone() * outer_coeff;
                }

                // Potential surd: Pow(n, 1/2)
                Expr::Pow(base, exp) => {
                    if let Some(atom) = Self::as_surd(ctx, *base, *exp) {
                        let scaled_coeff = atom.coeff * outer_coeff;
                        *surd_map
                            .entry(atom.radicand)
                            .or_insert_with(|| BigRational::from_integer(0.into())) += scaled_coeff;
                    } else {
                        return None; // Not a recognized surd
                    }
                }

                // Multiplication: could be coeff * surd
                Expr::Mul(l, r) => {
                    if let Some((inner_coeff, atom)) = Self::as_coeff_times_surd(ctx, *l, *r) {
                        let scaled_coeff = atom.coeff * inner_coeff * outer_coeff;
                        *surd_map
                            .entry(atom.radicand)
                            .or_insert_with(|| BigRational::from_integer(0.into())) += scaled_coeff;
                    } else {
                        return None; // Not a recognized pattern
                    }
                }

                // Function: sqrt(n) is also a surd
                Expr::Function(fn_id, args) => {
                    if ctx.sym_name(*fn_id) == "sqrt" && args.len() == 1 {
                        if let Some(atom) = Self::as_sqrt_function(ctx, args[0]) {
                            let scaled_coeff = atom.coeff * outer_coeff;
                            *surd_map
                                .entry(atom.radicand)
                                .or_insert_with(|| BigRational::from_integer(0.into())) +=
                                scaled_coeff;
                        } else {
                            return None; // sqrt of non-integer
                        }
                    } else {
                        return None; // Other functions not supported
                    }
                }

                // Division: not supported in v1
                Expr::Div(_, _) => return None,

                // SessionRef/Hold: not supported in surd parsing (should be resolved first)
                Expr::SessionRef(_) | Expr::Hold(_) => return None,
            }
        }

        // Convert map to vec, removing zero coefficients
        // CRITICAL: Sort by radicand for deterministic conjugate selection
        let mut surds: Vec<SurdAtom> = surd_map
            .into_iter()
            .filter(|(_, coeff)| *coeff != BigRational::from_integer(0.into()))
            .map(|(radicand, coeff)| SurdAtom { radicand, coeff })
            .collect();
        surds.sort_by_key(|s| s.radicand);

        Some(SurdSumView { constant, surds })
    }

    /// Check if expression is a pure surd: Pow(base, 1/2) with base a positive integer.
    ///
    /// Now robust to both `Number(1/2)` and `Div(1, 2)` exponent forms.
    fn as_surd(ctx: &Context, base: ExprId, exp: ExprId) -> Option<SurdAtom> {
        use num_rational::BigRational;

        // Use robust rational extraction (depth 8 is plenty for simple exponents)
        let exp_val = as_rational_const(ctx, exp, 8)?;

        // Check if exponent equals 1/2
        let half = BigRational::new(1.into(), 2.into());
        if exp_val != half {
            return None;
        }

        // Base must be a positive integer
        if let Expr::Number(n) = ctx.get(base) {
            if n.is_integer() {
                let n_int = n.numer().to_i64()?;
                if n_int > 0 {
                    // Apply square-free decomposition
                    let (outer, inner) = square_free_decompose(n_int);
                    return Some(SurdAtom {
                        radicand: inner,
                        coeff: BigRational::from_integer(outer.into()),
                    });
                }
            }
        }
        None
    }

    /// Check if Mul(l, r) is coeff * surd or surd * coeff.
    fn as_coeff_times_surd(ctx: &Context, l: ExprId, r: ExprId) -> Option<(BigRational, SurdAtom)> {
        // Try l=coeff, r=surd (Pow form)
        if let Expr::Number(coeff) = ctx.get(l) {
            if let Expr::Pow(base, exp) = ctx.get(r) {
                if let Some(atom) = Self::as_surd(ctx, *base, *exp) {
                    return Some((coeff.clone(), atom));
                }
            }
            // Try sqrt function form
            if let Expr::Function(fn_id, args) = ctx.get(r) {
                if ctx.sym_name(*fn_id) == "sqrt" && args.len() == 1 {
                    if let Some(atom) = Self::as_sqrt_function(ctx, args[0]) {
                        return Some((coeff.clone(), atom));
                    }
                }
            }
        }
        // Try l=surd, r=coeff (Pow form)
        if let Expr::Number(coeff) = ctx.get(r) {
            if let Expr::Pow(base, exp) = ctx.get(l) {
                if let Some(atom) = Self::as_surd(ctx, *base, *exp) {
                    return Some((coeff.clone(), atom));
                }
            }
            // Try sqrt function form
            if let Expr::Function(fn_id, args) = ctx.get(l) {
                if ctx.sym_name(*fn_id) == "sqrt" && args.len() == 1 {
                    if let Some(atom) = Self::as_sqrt_function(ctx, args[0]) {
                        return Some((coeff.clone(), atom));
                    }
                }
            }
        }
        None
    }

    /// Check if Function("sqrt", [n]) is a surd where n is a positive integer.
    fn as_sqrt_function(ctx: &Context, arg: ExprId) -> Option<SurdAtom> {
        if let Expr::Number(n) = ctx.get(arg) {
            if n.is_integer() {
                let n_int = n.numer().to_i64()?;
                if n_int > 0 {
                    let (outer, inner) = square_free_decompose(n_int);
                    return Some(SurdAtom {
                        radicand: inner,
                        coeff: BigRational::from_integer(outer.into()),
                    });
                }
            }
        }
        None
    }

    /// Number of surd terms (non-zero)
    pub fn surd_count(&self) -> usize {
        self.surds.len()
    }

    /// Check if this is purely rational (no surds)
    pub fn is_rational(&self) -> bool {
        self.surds.is_empty()
    }
}

/// Decompose n into (outer, inner) such that n = outer² * inner and inner is square-free.
///
/// Example: 12 = 4 * 3 = 2² * 3 → (2, 3)
/// Example: 8 = 4 * 2 = 2² * 2 → (2, 2)
/// Example: 18 = 9 * 2 = 3² * 2 → (3, 2)
pub fn square_free_decompose(mut n: i64) -> (i64, i64) {
    if n <= 0 {
        return (1, n); // Invalid, but don't panic
    }

    let mut outer = 1i64;
    let mut factor = 2i64;

    while factor * factor <= n {
        while n % (factor * factor) == 0 {
            outer *= factor;
            n /= factor * factor;
        }
        factor += 1;
    }

    (outer, n)
}

#[cfg(test)]
mod surd_tests {
    use super::*;

    #[test]
    fn test_square_free_decompose() {
        assert_eq!(square_free_decompose(1), (1, 1));
        assert_eq!(square_free_decompose(2), (1, 2));
        assert_eq!(square_free_decompose(4), (2, 1));
        assert_eq!(square_free_decompose(8), (2, 2));
        assert_eq!(square_free_decompose(12), (2, 3));
        assert_eq!(square_free_decompose(18), (3, 2));
        assert_eq!(square_free_decompose(72), (6, 2)); // 72 = 36*2 = 6²*2
    }

    #[test]
    fn test_surd_sum_view_basic() {
        let mut ctx = Context::new();

        // 1 + √2
        let one = ctx.num(1);
        let two = ctx.num(2);
        let half = ctx.rational(1, 2);
        let sqrt2 = ctx.add(Expr::Pow(two, half));
        let sum = ctx.add(Expr::Add(one, sqrt2));

        let view = SurdSumView::from(&ctx, sum).expect("Should parse");
        assert_eq!(view.constant, BigRational::from_integer(1.into()));
        assert_eq!(view.surd_count(), 1);
        assert_eq!(view.surds[0].radicand, 2);
    }

    #[test]
    fn test_surd_sum_view_rejects_variable() {
        let mut ctx = Context::new();

        // 1 + √x (should reject)
        let one = ctx.num(1);
        let x = ctx.var("x");
        let half = ctx.rational(1, 2);
        let sqrt_x = ctx.add(Expr::Pow(x, half));
        let sum = ctx.add(Expr::Add(one, sqrt_x));

        assert!(SurdSumView::from(&ctx, sum).is_none());
    }

    #[test]
    fn test_surd_sum_view_square_free_normalization() {
        let mut ctx = Context::new();

        // √8 = 2√2
        let eight = ctx.num(8);
        let half = ctx.rational(1, 2);
        let sqrt8 = ctx.add(Expr::Pow(eight, half));

        let view = SurdSumView::from(&ctx, sqrt8).expect("Should parse");
        assert_eq!(view.surd_count(), 1);
        assert_eq!(view.surds[0].radicand, 2);
        assert_eq!(view.surds[0].coeff, BigRational::from_integer(2.into()));
    }

    #[test]
    fn test_surd_sum_view_sqrt_function() {
        let mut ctx = Context::new();

        // 1 + sqrt(2) + sqrt(3) using Function representation
        let one = ctx.num(1);
        let two = ctx.num(2);
        let three = ctx.num(3);
        let sqrt2_fn = ctx.call("sqrt", vec![two]);
        let sqrt3_fn = ctx.call("sqrt", vec![three]);
        let sum1 = ctx.add(Expr::Add(one, sqrt2_fn));
        let sum2 = ctx.add(Expr::Add(sum1, sqrt3_fn));

        let view = SurdSumView::from(&ctx, sum2).expect("Should parse sqrt function form");
        assert_eq!(view.constant, BigRational::from_integer(1.into()));
        assert_eq!(view.surd_count(), 2);
        // Check both surds present (order may vary due to HashMap)
        let radicands: Vec<i64> = view.surds.iter().map(|s| s.radicand).collect();
        assert!(radicands.contains(&2));
        assert!(radicands.contains(&3));
    }

    #[test]
    fn test_surd_sum_view_div_exponent_form() {
        let mut ctx = Context::new();

        // 1 + 2^(1/2) + 3^(1/2) using Div form for exponent
        let one_expr = ctx.num(1);
        let two = ctx.num(2);
        let three = ctx.num(3);
        let one = ctx.num(1);
        let two_denom = ctx.num(2);
        // Create Div(1, 2) as exponent
        let half_div = ctx.add(Expr::Div(one, two_denom));
        let sqrt2_div = ctx.add(Expr::Pow(two, half_div));

        let one2 = ctx.num(1);
        let two_denom2 = ctx.num(2);
        let half_div2 = ctx.add(Expr::Div(one2, two_denom2));
        let sqrt3_div = ctx.add(Expr::Pow(three, half_div2));

        let sum1 = ctx.add(Expr::Add(one_expr, sqrt2_div));
        let sum2 = ctx.add(Expr::Add(sum1, sqrt3_div));

        let view = SurdSumView::from(&ctx, sum2).expect("Should parse Div(1,2) exponent form");
        assert_eq!(view.constant, BigRational::from_integer(1.into()));
        assert_eq!(view.surd_count(), 2);
        // Check both surds present
        let radicands: Vec<i64> = view.surds.iter().map(|s| s.radicand).collect();
        assert!(radicands.contains(&2));
        assert!(radicands.contains(&3));
    }
}
