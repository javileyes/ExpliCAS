use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::ToPrimitive;
use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::builtin::BuiltinFn;
use crate::symbol::{SymbolId, SymbolTable};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprId(u32);

impl ExprId {
    pub const INDEX_MASK: u32 = 0x1FFFFFFF;
    pub const TAG_SHIFT: u32 = 29;

    pub const TAG_NUMBER: u32 = 0;
    pub const TAG_ATOM: u32 = 1; // Variable, Constant
    pub const TAG_UNARY: u32 = 2; // Neg
    pub const TAG_BINARY: u32 = 3; // Add, Sub, Mul, Div, Pow
    pub const TAG_NARY: u32 = 4; // Function, Matrix

    #[inline]
    pub fn new(index: u32, tag: u32) -> Self {
        debug_assert!(index <= Self::INDEX_MASK, "ExprId index overflow");
        debug_assert!(tag <= 7, "ExprId tag overflow");
        ExprId((tag << Self::TAG_SHIFT) | (index & Self::INDEX_MASK))
    }

    #[inline]
    pub fn index(self) -> usize {
        (self.0 & Self::INDEX_MASK) as usize
    }

    #[inline]
    pub fn tag(self) -> u32 {
        self.0 >> Self::TAG_SHIFT
    }

    #[inline]
    pub fn is_atom(self) -> bool {
        let t = self.tag();
        t == Self::TAG_NUMBER || t == Self::TAG_ATOM
    }

    /// Create an ExprId from a raw u32 value (for testing purposes)
    #[inline]
    pub fn from_raw(val: u32) -> Self {
        ExprId(val)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Constant {
    Pi,
    E,
    Infinity,
    Undefined,
    /// Imaginary unit i (where i² = -1)
    I,
    /// Golden ratio φ = (1+√5)/2 ≈ 1.618
    Phi,
}

/// Check if a function is known to return a commutative (scalar) result.
///
/// These functions produce scalar outputs even when given matrix arguments,
/// so their presence doesn't make surrounding multiplication non-commutative.
///
/// # Currently recognized
/// - `det` - determinant: det(M) → scalar
/// - `trace` - trace: trace(M) → scalar
/// - `norm` - matrix/vector norm: norm(M) → scalar
/// - `rank` - matrix rank: rank(M) → scalar
///
/// # Extension point
/// Add new scalar-returning functions here when implementing matrix operations.
#[inline]
fn function_returns_commutative(ctx: &Context, fn_id: SymbolId) -> bool {
    let name = ctx.sym_name(fn_id);
    matches!(name, "det" | "trace" | "norm" | "rank")
}

/// Multiplication commutativity kind for an expression.
///
/// Determines whether factors in a `Mul` can be safely reordered.
/// Used by canonicalization to decide sorting strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MulCommutativity {
    /// Commutative multiplication: a*b = b*a
    /// (numbers, polynomials, scalars)
    Commutative,
    /// Non-commutative multiplication: a*b ≠ b*a
    /// (matrices, operators, quaternions)
    NonCommutative,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    Number(BigRational),
    Constant(Constant),
    Variable(SymbolId),
    Add(ExprId, ExprId),
    Sub(ExprId, ExprId),
    Mul(ExprId, ExprId),
    Div(ExprId, ExprId),
    Pow(ExprId, ExprId),
    Neg(ExprId),
    Function(SymbolId, Vec<ExprId>),
    /// Matrix: Unified representation for matrices and vectors
    /// - Vector column (nx1): rows=n, cols=1
    /// - Vector row (1xn): rows=1, cols=n
    /// - Matrix (mxn): rows=m, cols=n
    /// - Data stored in row-major order: data[row * cols + col]
    Matrix {
        rows: usize,
        cols: usize,
        data: Vec<ExprId>, // Length must be rows * cols
    },
    /// Session reference: refers to a previously stored expression by ID (#1, #2, etc.)
    /// Resolved before simplification using SessionStore
    SessionRef(u64),
    /// Hold barrier: blocks expansive/structural rules but transparent to basic algebra.
    /// Like Neg, this is a unary wrapper (TAG_UNARY) for consistent storage/layout.
    /// Phase 1: capacity only - wrap_hold still produces Function(__hold, [x])
    Hold(ExprId),
}

/// Statistics for tracking Context resource usage.
///
/// This is used by the budget system to detect real growth
/// (not just intern calls, but actual new node creation).
#[derive(Debug, Default, Clone, Copy)]
pub struct ContextStats {
    /// Number of nodes actually created (not deduplicated).
    /// Increments only when `nodes.push` happens, not on cache hits.
    pub nodes_created: u64,
}

#[derive(Clone)]
pub struct Context {
    pub nodes: Vec<Expr>,
    /// Interner: maps hash to bucket of ExprIds with that hash.
    /// Using Vec<ExprId> instead of single ExprId to properly handle hash collisions
    /// without losing deduplication for expressions that share the same hash.
    pub interner: FxHashMap<u64, SmallVec<[ExprId; 1]>>,
    /// Symbol table for interned variable names
    pub symbols: SymbolTable,
    /// Statistics for budget tracking
    stats: ContextStats,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    #[inline]
    fn expr_hash(&self, value: &Expr) -> u64 {
        let mut hasher = FxHasher::default();
        match value {
            Expr::Number(n) => {
                hasher.write_u8(0);
                if let (Some(num), Some(den)) = (n.numer().to_i64(), n.denom().to_i64()) {
                    hasher.write_i64(num);
                    hasher.write_i64(den);
                } else {
                    n.numer().hash(&mut hasher);
                    n.denom().hash(&mut hasher);
                }
            }
            Expr::Constant(c) => {
                hasher.write_u8(1);
                let tag = match c {
                    Constant::Pi => 0u8,
                    Constant::E => 1,
                    Constant::Infinity => 2,
                    Constant::Undefined => 3,
                    Constant::I => 4,
                    Constant::Phi => 5,
                };
                hasher.write_u8(tag);
            }
            Expr::Variable(id) => {
                hasher.write_u8(2);
                hasher.write_usize(*id);
            }
            Expr::Add(l, r) => {
                hasher.write_u8(3);
                hasher.write_u32(l.0);
                hasher.write_u32(r.0);
            }
            Expr::Sub(l, r) => {
                hasher.write_u8(4);
                hasher.write_u32(l.0);
                hasher.write_u32(r.0);
            }
            Expr::Mul(l, r) => {
                hasher.write_u8(5);
                hasher.write_u32(l.0);
                hasher.write_u32(r.0);
            }
            Expr::Div(l, r) => {
                hasher.write_u8(6);
                hasher.write_u32(l.0);
                hasher.write_u32(r.0);
            }
            Expr::Pow(l, r) => {
                hasher.write_u8(7);
                hasher.write_u32(l.0);
                hasher.write_u32(r.0);
            }
            Expr::Neg(inner) => {
                hasher.write_u8(8);
                hasher.write_u32(inner.0);
            }
            Expr::Function(name, args) => {
                hasher.write_u8(9);
                hasher.write_usize(*name);
                hasher.write_usize(args.len());
                for arg in args {
                    hasher.write_u32(arg.0);
                }
            }
            Expr::Matrix { rows, cols, data } => {
                hasher.write_u8(10);
                hasher.write_usize(*rows);
                hasher.write_usize(*cols);
                hasher.write_usize(data.len());
                for item in data {
                    hasher.write_u32(item.0);
                }
            }
            Expr::SessionRef(id) => {
                hasher.write_u8(11);
                hasher.write_u64(*id);
            }
            Expr::Hold(inner) => {
                hasher.write_u8(12);
                hasher.write_u32(inner.0);
            }
        }
        hasher.finish()
    }

    pub fn new() -> Self {
        let extra_symbol_capacity = 8;
        let mut interner = FxHashMap::default();
        interner.reserve(64);
        Self {
            // Most short-lived parse/eval contexts stay below a few dozen nodes.
            nodes: Vec::with_capacity(32),
            interner,
            symbols: SymbolTable::with_capacity(extra_symbol_capacity),
            stats: ContextStats::default(),
        }
    }

    /// Create a context sized for snapshot restore.
    ///
    /// Snapshot restore reconstructs nodes directly into the arena without
    /// canonicalization, so the main win is avoiding repeated growth of the
    /// node arena and symbol table.
    pub fn with_restore_capacity(node_capacity: usize) -> Self {
        let extra_symbol_capacity = node_capacity.saturating_div(4).clamp(8, 256);
        let mut interner = FxHashMap::default();
        interner.reserve(64);
        Self {
            nodes: Vec::with_capacity(node_capacity),
            interner,
            symbols: SymbolTable::with_capacity(extra_symbol_capacity),
            stats: ContextStats::default(),
        }
    }

    // =========================================================================
    // Symbol interning API
    // =========================================================================

    /// Intern a symbol name, returning its SymbolId.
    #[inline]
    pub fn intern_symbol(&mut self, name: &str) -> SymbolId {
        self.symbols.intern(name)
    }

    /// Resolve a SymbolId to its string name.
    #[inline]
    pub fn sym_name(&self, id: SymbolId) -> &str {
        self.symbols.resolve(id)
    }

    /// Check if an expression is a variable with the given name.
    ///
    /// Note: requires &mut self because intern_symbol may insert.
    pub fn is_var(&mut self, expr: ExprId, name: &str) -> bool {
        // Extract sym_id first to avoid borrow conflict
        let sym_id = match self.get(expr) {
            Expr::Variable(id) => Some(*id),
            _ => None,
        };
        match sym_id {
            Some(id) => self.sym_is(id, name),
            None => false,
        }
    }

    /// Compare a SymbolId with a string name efficiently.
    ///
    /// When Expr::Variable uses SymbolId internally, this allows O(1)
    /// comparison without resolving to string.
    ///
    /// Note: requires &mut self because intern_symbol may insert.
    #[inline]
    pub fn sym_is(&mut self, id: SymbolId, name: &str) -> bool {
        id == self.intern_symbol(name)
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    /// Get current statistics (for budget tracking).
    #[inline]
    pub fn stats(&self) -> ContextStats {
        self.stats
    }

    /// Check if a term is "negative" for Add ordering purposes
    /// Returns true for: Neg(x), Mul(-n, x), Number(-n)
    fn is_negative_term(&self, id: ExprId) -> bool {
        use num_traits::Signed;
        match self.get(id) {
            Expr::Neg(_) => true,
            Expr::Mul(a, _) => {
                // If left factor is a negative number, the term is negative
                if let Expr::Number(n) = self.get(*a) {
                    return n.is_negative();
                }
                false
            }
            Expr::Number(n) => n.is_negative(),
            _ => false,
        }
    }

    /// Compare Add terms with positive-first ordering
    /// Positive terms come before negative terms for prettier display (x² - y² not -y² + x²)
    fn compare_add_terms(&self, a: ExprId, b: ExprId) -> std::cmp::Ordering {
        let a_neg = self.is_negative_term(a);
        let b_neg = self.is_negative_term(b);
        match (a_neg, b_neg) {
            (false, true) => std::cmp::Ordering::Less, // positive < negative
            (true, false) => std::cmp::Ordering::Greater, // negative > positive
            _ => crate::ordering::compare_expr(self, a, b), // same sign: regular order
        }
    }

    pub fn add(&mut self, expr: Expr) -> ExprId {
        // FLATTEN + CANONICALIZE: For Add and Mul, collect all terms, sort, rebuild
        // This ensures Add(Add(a,b),c) and Add(a,Add(b,c)) produce identical trees
        let canonical_expr = match expr {
            // Neg canonicalization: prevent Neg(Number) and Neg(Neg(x))
            Expr::Neg(inner) => {
                match self.get(inner) {
                    // Neg(Number(n)) → Number(-n)
                    Expr::Number(n) => {
                        return self.add(Expr::Number(-n.clone()));
                    }
                    // Neg(Neg(x)) → x
                    Expr::Neg(double_inner) => {
                        return *double_inner;
                    }
                    _ => Expr::Neg(inner),
                }
            }
            Expr::Add(l, r) => {
                if !matches!(self.get(l), Expr::Add(_, _))
                    && !matches!(self.get(r), Expr::Add(_, _))
                {
                    let (left, right) = if self.compare_add_terms(l, r).is_gt() {
                        (r, l)
                    } else {
                        (l, r)
                    };
                    Expr::Add(left, right)
                } else {
                    // Collect all additive terms (flatten nested Adds)
                    let mut terms = SmallVec::<[ExprId; 8]>::new();
                    self.collect_add_terms(l, &mut terms);
                    self.collect_add_terms(r, &mut terms);
                    // Sort terms: positive first, then by compare_expr
                    terms.sort_unstable_by(|a, b| self.compare_add_terms(*a, *b));
                    // Build right-associative tree: a + (b + (c + d))
                    self.build_balanced_add(&terms)
                }
            }
            Expr::Mul(l, r) => {
                let flat_binary = !matches!(self.get(l), Expr::Mul(_, _))
                    && !matches!(self.get(r), Expr::Mul(_, _));
                // Check if multiplication is non-commutative (e.g., contains matrices)
                if !self.is_mul_commutative_pair(l, r) {
                    if flat_binary {
                        return self.add_raw(Expr::Mul(l, r));
                    }
                    // Non-commutative: flatten for associativity but do NOT sort
                    let mut factors = SmallVec::<[ExprId; 8]>::new();
                    self.collect_mul_factors(l, &mut factors);
                    self.collect_mul_factors(r, &mut factors);
                    // Rebuild balanced (preserves order) - returns ExprId directly
                    return self.build_balanced_mul(&factors);
                } else {
                    if flat_binary {
                        let (left, right) = if crate::ordering::compare_expr(self, l, r).is_gt() {
                            (r, l)
                        } else {
                            (l, r)
                        };
                        return self.add_raw(Expr::Mul(left, right));
                    }
                    // Commutative: flatten + sort (using order_key to avoid recursive compare)
                    let mut factors = SmallVec::<[ExprId; 8]>::new();
                    self.collect_mul_factors(l, &mut factors);
                    self.collect_mul_factors(r, &mut factors);
                    // Sort by structural comparison (balanced tree prevents deep recursion)
                    factors.sort_unstable_by(|a, b| crate::ordering::compare_expr(self, *a, *b));
                    // Returns ExprId directly
                    return self.build_balanced_mul(&factors);
                }
            }
            // Non-commutative operations and atoms: keep as-is
            other => other,
        };

        // Expression Interning: Deduplicate expressions
        let hash = self.expr_hash(&canonical_expr);

        // Determine tag based on expression type
        let tag = match &canonical_expr {
            Expr::Number(_) => ExprId::TAG_NUMBER,
            Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => ExprId::TAG_ATOM,
            Expr::Neg(_) | Expr::Hold(_) => ExprId::TAG_UNARY,
            Expr::Add(_, _)
            | Expr::Sub(_, _)
            | Expr::Mul(_, _)
            | Expr::Div(_, _)
            | Expr::Pow(_, _) => ExprId::TAG_BINARY,
            Expr::Function(_, _) | Expr::Matrix { .. } => ExprId::TAG_NARY,
        };

        // Check if we already have this expression in the bucket
        if let Some(bucket) = self.interner.get(&hash) {
            // Search all entries in the bucket for an exact match
            for &id in bucket {
                if self.nodes[id.index()] == canonical_expr {
                    return id; // Found existing expression
                }
            }
            // Hash collision: same hash but different content
            // We'll add to the bucket below
        }

        let index = self.nodes.len() as u32;
        let id = ExprId::new(index, tag);
        self.nodes.push(canonical_expr);
        self.stats.nodes_created += 1; // Track real creation (not cache hit)

        // Add to bucket (create new bucket if needed)
        self.interner.entry(hash).or_default().push(id);
        id
    }

    /// Add expression WITHOUT canonicalization (preserves operand order).
    ///
    /// Use this for:
    /// - Non-commutative operations (matrix multiplication)
    /// - Raw construction where order matters
    /// - Internal builders like `mul2_raw`
    ///
    /// Unlike `add()`, this does NOT swap Mul/Add operands.
    pub fn add_raw(&mut self, expr: Expr) -> ExprId {
        // DEBUG: Catch Variables being created via add_raw (should use ctx.var() instead)
        debug_assert!(
            !matches!(expr, Expr::Variable(_)),
            "add_raw must not be used for Variable; use ctx.var() or ctx.add() instead"
        );

        // Expression Interning: Deduplicate expressions
        let hash = self.expr_hash(&expr);

        let tag = match &expr {
            Expr::Number(_) => ExprId::TAG_NUMBER,
            Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => ExprId::TAG_ATOM,
            Expr::Neg(_) | Expr::Hold(_) => ExprId::TAG_UNARY,
            Expr::Add(_, _)
            | Expr::Sub(_, _)
            | Expr::Mul(_, _)
            | Expr::Div(_, _)
            | Expr::Pow(_, _) => ExprId::TAG_BINARY,
            Expr::Function(_, _) | Expr::Matrix { .. } => ExprId::TAG_NARY,
        };

        // Check if we already have this expression in the bucket
        if let Some(bucket) = self.interner.get(&hash) {
            for &id in bucket {
                if self.nodes[id.index()] == expr {
                    return id; // Found existing expression
                }
            }
        }

        let index = self.nodes.len() as u32;
        let id = ExprId::new(index, tag);
        self.nodes.push(expr);
        self.stats.nodes_created += 1; // Track real creation (not cache hit)

        self.interner.entry(hash).or_default().push(id);
        id
    }

    /// Collect all additive terms by flattening nested Add (iterative)
    fn collect_add_terms(&self, id: ExprId, terms: &mut SmallVec<[ExprId; 8]>) {
        let mut stack = SmallVec::<[ExprId; 8]>::new();
        stack.push(id);
        while let Some(current) = stack.pop() {
            match self.get(current) {
                Expr::Add(l, r) => {
                    // Push right first so left is processed first
                    stack.push(*r);
                    stack.push(*l);
                }
                _ => terms.push(current),
            }
        }
    }

    /// Build balanced Add tree iteratively: [a,b,c,d] -> Add(Add(a,b), Add(c,d))
    fn build_balanced_add(&mut self, terms: &[ExprId]) -> Expr {
        match terms.len() {
            0 => {
                // Empty terms is a pipeline bug, but we recover gracefully in release
                debug_assert!(false, "build_balanced_add called with empty terms");
                Expr::Number(BigRational::from_integer(BigInt::from(0))) // Identity for addition
            }
            1 => self.get(terms[0]).clone(),
            2 => Expr::Add(terms[0], terms[1]),
            _ => {
                // Build pairs bottom-up iteratively
                let mut current: SmallVec<[ExprId; 8]> = terms.iter().copied().collect();
                while current.len() > 2 {
                    let mut next =
                        SmallVec::<[ExprId; 8]>::with_capacity(current.len().div_ceil(2));
                    let mut i = 0;
                    while i < current.len() {
                        if i + 1 < current.len() {
                            let pair = Expr::Add(current[i], current[i + 1]);
                            next.push(self.add_raw(pair));
                            i += 2;
                        } else {
                            next.push(current[i]);
                            i += 1;
                        }
                    }
                    current = next;
                }
                if current.len() == 2 {
                    Expr::Add(current[0], current[1])
                } else {
                    self.get(current[0]).clone()
                }
            }
        }
    }

    /// Collect all multiplicative factors by flattening nested Mul (iterative)
    fn collect_mul_factors(&self, id: ExprId, factors: &mut SmallVec<[ExprId; 8]>) {
        let mut stack = SmallVec::<[ExprId; 8]>::new();
        stack.push(id);
        while let Some(current) = stack.pop() {
            match self.get(current) {
                Expr::Mul(l, r) => {
                    // Push right first so left is processed first (maintains order)
                    stack.push(*r);
                    stack.push(*l);
                }
                _ => factors.push(current),
            }
        }
    }

    /// Build balanced Mul tree from factors: [a,b,c,d] → Mul(Mul(a,b), Mul(c,d))
    ///
    /// **CANONICAL balanced multiplication builder.**
    ///
    /// # When to use
    /// - Expansion/collection passes with many factors (O(log n) depth)
    /// - Multinomial expansion, polynomial operations
    /// - Any context where tree depth matters more than shape stability
    ///
    /// # For pattern matching / didactic transformations
    /// Use `MulBuilder` (right-fold) instead - it produces stable shape `a*(b*(c*d))`
    ///
    /// # Edge cases
    /// - `[]` → panics (empty product undefined at this level)
    /// - `[x]` → returns `x` directly (no wrapper)
    /// - `[a,b]` → `Mul(a, b)`
    /// - `[a,b,c,d]` → `Mul(Mul(a,b), Mul(c,d))` (balanced)
    ///
    /// # See also
    /// - `cas_ast::views::MulBuilder` for right-fold construction
    /// - POLICY.md "Builders Contract" for contribution rules
    pub fn build_balanced_mul(&mut self, factors: &[ExprId]) -> ExprId {
        match factors.len() {
            0 => {
                // Empty factors is a pipeline bug, but we recover gracefully in release
                debug_assert!(false, "build_balanced_mul called with empty factors");
                self.num(1) // Identity for multiplication
            }
            1 => factors[0],
            2 => self.add_raw(Expr::Mul(factors[0], factors[1])),
            _ => {
                // Build pairs bottom-up iteratively
                let mut current: SmallVec<[ExprId; 8]> = factors.iter().copied().collect();
                while current.len() > 2 {
                    let mut next =
                        SmallVec::<[ExprId; 8]>::with_capacity(current.len().div_ceil(2));
                    let mut i = 0;
                    while i < current.len() {
                        if i + 1 < current.len() {
                            let pair = Expr::Mul(current[i], current[i + 1]);
                            next.push(self.add_raw(pair));
                            i += 2;
                        } else {
                            next.push(current[i]);
                            i += 1;
                        }
                    }
                    current = next;
                }
                if current.len() == 2 {
                    self.add_raw(Expr::Mul(current[0], current[1]))
                } else {
                    current[0]
                }
            }
        }
    }

    /// Multiplication commutativity for an expression subtree.
    ///
    /// # Purpose
    /// Determines whether it is safe to **reorder factors** in a `Mul` containing
    /// this expression without changing the result.
    ///
    /// # Returns
    /// - `Commutative`: factors can be sorted (a*b = b*a)
    /// - `NonCommutative`: factor order matters (a*b ≠ b*a)
    ///
    /// # Semantics
    /// - **Conservative**: if uncertain, returns `NonCommutative`
    /// - Does NOT indicate "scalar" vs "matrix" type
    /// - Only concerns multiplication ordering
    ///
    /// # When to use
    /// Call this before any `sort_by` on Mul factors. If NonCommutative,
    /// preserve the original factor order.
    ///
    /// # Performance
    /// O(n) traversal where n = subtree size. Called from few places during
    /// canonicalization. If profiling shows hotspot, add cache by ExprId.
    ///
    /// # Future extensions
    /// - Functions like `det(M)` or `trace(M)` return scalars even with matrix args
    /// - Could add function table to override (TODO: function result type overrides)
    pub fn mul_commutativity(&self, id: ExprId) -> MulCommutativity {
        // Iterative traversal - stack-safe for deep expressions
        let mut stack = SmallVec::<[ExprId; 8]>::new();
        stack.push(id);
        while let Some(current) = stack.pop() {
            match self.get(current) {
                // Explicitly non-commutative types
                Expr::Matrix { .. } => return MulCommutativity::NonCommutative,

                // Future: add more non-commutative types here
                // Expr::Operator(_) => return MulCommutativity::NonCommutative,
                // Expr::Quaternion(_) => return MulCommutativity::NonCommutative,

                // Binary ops: check children
                Expr::Mul(l, r)
                | Expr::Add(l, r)
                | Expr::Sub(l, r)
                | Expr::Div(l, r)
                | Expr::Pow(l, r) => {
                    stack.push(*l);
                    stack.push(*r);
                }
                Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),

                // Functions: check arguments unless function is known to return scalar
                Expr::Function(fn_id, args) => {
                    // Check for functions known to return commutative outputs
                    if function_returns_commutative(self, *fn_id) {
                        continue; // Skip checking args - output is scalar
                    }
                    stack.extend(args.iter().copied());
                }

                // Scalars are commutative
                Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
            }
        }
        MulCommutativity::Commutative
    }

    /// Convenience: check if Mul is commutative for this expression
    #[inline]
    pub fn is_mul_commutative(&self, id: ExprId) -> bool {
        self.mul_commutativity(id) == MulCommutativity::Commutative
    }

    /// Check if a pair of expressions can be reordered in multiplication.
    ///
    /// This is useful during construction of Mul nodes, where we don't
    /// yet have the combined ExprId.
    #[inline]
    pub fn is_mul_commutative_pair(&self, l: ExprId, r: ExprId) -> bool {
        self.is_mul_commutative(l) && self.is_mul_commutative(r)
    }

    pub fn get(&self, id: ExprId) -> &Expr {
        &self.nodes[id.index()]
    }

    // Helper constructors that add to context immediately
    pub fn num(&mut self, n: i64) -> ExprId {
        self.add(Expr::Number(BigRational::from_integer(BigInt::from(n))))
    }

    pub fn rational(&mut self, num: i64, den: i64) -> ExprId {
        self.add(Expr::Number(BigRational::new(
            BigInt::from(num),
            BigInt::from(den),
        )))
    }

    pub fn var(&mut self, name: &str) -> ExprId {
        let sym = self.intern_symbol(name);
        self.add(Expr::Variable(sym))
    }

    // =========================================================================
    // Function call helpers
    // =========================================================================

    /// Create a function call expression: ctx.call("sqrt", vec![x])
    pub fn call(&mut self, name: &str, args: Vec<ExprId>) -> ExprId {
        let fn_id = self.intern_symbol(name);
        self.add(Expr::Function(fn_id, args))
    }

    /// Get the name of a function call, if this is a function expression.
    /// Returns None if not a function.
    pub fn fn_name(&self, expr: ExprId) -> Option<&str> {
        match self.get(expr) {
            Expr::Function(fn_id, _) => Some(self.sym_name(*fn_id)),
            _ => None,
        }
    }

    /// Check if an expression is a function call with the given name.
    /// Uses symbol interning for efficient O(1) comparison.
    ///
    /// Note: requires &mut self because intern_symbol may insert.
    pub fn is_call_named(&mut self, expr: ExprId, name: &str) -> bool {
        let target = self.intern_symbol(name);
        matches!(self.get(expr), Expr::Function(fn_id, _) if *fn_id == target)
    }

    // =========================================================================
    // Builtin function helpers (O(1) comparison, no string allocation)
    // =========================================================================

    /// Get the cached SymbolId for a builtin function.
    ///
    /// In the standard `Context` layout builtins occupy the fixed logical prefix
    /// of `SymbolId`, so this is a direct cast.
    ///
    /// # Example
    /// ```rust,ignore
    /// // Instead of: ctx.sym_name(*fn_id) == "sqrt"
    /// if *fn_id == ctx.builtin_id(BuiltinFn::Sqrt) { ... }
    /// ```
    #[inline]
    pub fn builtin_id(&self, builtin: BuiltinFn) -> SymbolId {
        builtin as SymbolId
    }

    /// Check if an expression is a call to a specific builtin function.
    ///
    /// This is the preferred way to check function identity in rules.
    ///
    /// # Example
    /// ```rust,ignore
    /// if ctx.is_builtin_call(expr, BuiltinFn::Sqrt) {
    ///     // Handle sqrt(...)
    /// }
    /// ```
    #[inline]
    pub fn is_builtin_call(&self, expr: ExprId, builtin: BuiltinFn) -> bool {
        match self.get(expr) {
            Expr::Function(fn_id, _) => *fn_id == self.builtin_id(builtin),
            _ => false,
        }
    }

    /// Check if a function SymbolId matches a builtin.
    ///
    /// Use this when you already have the fn_id extracted from Expr::Function.
    ///
    /// # Example
    /// ```rust,ignore
    /// if let Expr::Function(fn_id, args) = ctx.get(expr) {
    ///     if ctx.is_builtin(*fn_id, BuiltinFn::Sin) { ... }
    /// }
    /// ```
    #[inline]
    pub fn is_builtin(&self, fn_id: SymbolId, builtin: BuiltinFn) -> bool {
        fn_id == self.builtin_id(builtin)
    }

    /// Reverse lookup: get the BuiltinFn variant for a SymbolId (O(1)).
    ///
    /// Returns `None` if the SymbolId doesn't correspond to any builtin.
    ///
    /// # Example
    /// ```rust,ignore
    /// if let Expr::Function(fn_id, args) = ctx.get(expr) {
    ///     match ctx.builtin_of(*fn_id) {
    ///         Some(BuiltinFn::Sin) => { /* handle sin */ }
    ///         Some(BuiltinFn::Cos) => { /* handle cos */ }
    ///         _ => {}
    ///     }
    /// }
    /// ```
    #[inline]
    pub fn builtin_of(&self, fn_id: SymbolId) -> Option<BuiltinFn> {
        (fn_id < BuiltinFn::COUNT).then(|| crate::builtin::ALL_BUILTINS[fn_id])
    }

    /// Create a function call using a builtin.
    ///
    /// Slightly more efficient than `call("sqrt", args)` since it uses
    /// the cached SymbolId.
    #[inline]
    pub fn call_builtin(&mut self, builtin: BuiltinFn, args: Vec<ExprId>) -> ExprId {
        let fn_id = self.builtin_id(builtin);
        self.add(Expr::Function(fn_id, args))
    }

    // Matrix helpers

    /// Create a matrix with given dimensions and data.
    /// Returns Err if data.len() != rows * cols.
    pub fn matrix(
        &mut self,
        rows: usize,
        cols: usize,
        data: Vec<ExprId>,
    ) -> Result<ExprId, crate::error::AstError> {
        if data.len() != rows * cols {
            return Err(crate::error::AstError::InvalidMatrix {
                reason: format!(
                    "data length {} does not match dimensions {}x{}",
                    data.len(),
                    rows,
                    cols
                ),
            });
        }
        Ok(self.add(Expr::Matrix { rows, cols, data }))
    }

    /// Get element at (row, col) from a matrix (0-indexed)
    /// Returns None if id is not a matrix or indices are out of bounds
    pub fn matrix_element(&self, id: ExprId, row: usize, col: usize) -> Option<ExprId> {
        if let Expr::Matrix { rows, cols, data } = self.get(id) {
            if row < *rows && col < *cols {
                Some(data[row * cols + col])
            } else {
                None
            }
        } else {
            None
        }
    }

    // ... other helpers would need &mut self ...
}

impl fmt::Display for ExprId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Expr#{}", self.index())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul_commutativity_scalar() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let num = ctx.num(42);
        let pi = ctx.add(Expr::Constant(Constant::Pi));

        assert_eq!(ctx.mul_commutativity(x), MulCommutativity::Commutative);
        assert_eq!(ctx.mul_commutativity(y), MulCommutativity::Commutative);
        assert_eq!(ctx.mul_commutativity(num), MulCommutativity::Commutative);
        assert_eq!(ctx.mul_commutativity(pi), MulCommutativity::Commutative);

        // Mul of scalars is commutative
        let xy = ctx.add(Expr::Mul(x, y));
        assert!(ctx.is_mul_commutative(xy));
    }

    #[test]
    fn test_mul_commutativity_matrix() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(2);
        let c = ctx.num(3);
        let d = ctx.num(4);

        let matrix = ctx.matrix(2, 2, vec![a, b, c, d]).unwrap();

        assert_eq!(
            ctx.mul_commutativity(matrix),
            MulCommutativity::NonCommutative
        );
        assert!(!ctx.is_mul_commutative(matrix));
    }

    #[test]
    fn test_mul_commutativity_mixed() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let a = ctx.num(1);
        let b = ctx.num(2);
        let c = ctx.num(3);
        let d = ctx.num(4);
        let matrix = ctx.matrix(2, 2, vec![a, b, c, d]).unwrap();

        // x * M should be non-commutative
        let xm = ctx.add(Expr::Mul(x, matrix));
        assert!(!ctx.is_mul_commutative(xm));

        // Pair check
        assert!(ctx.is_mul_commutative_pair(x, x));
        assert!(!ctx.is_mul_commutative_pair(x, matrix));
        assert!(!ctx.is_mul_commutative_pair(matrix, matrix));
    }

    #[test]
    fn test_mul_commutativity_function_with_matrix() {
        let mut ctx = Context::new();
        let a = ctx.num(1);
        let b = ctx.num(2);
        let c = ctx.num(3);
        let d = ctx.num(4);
        let matrix = ctx.matrix(2, 2, vec![a, b, c, d]).unwrap();

        // sin(M) should be non-commutative (contains matrix)
        let sin_m = ctx.call("sin", vec![matrix]);
        assert!(!ctx.is_mul_commutative(sin_m));

        // sin(x) should be commutative
        let x = ctx.var("x");
        let sin_x = ctx.call("sin", vec![x]);
        assert!(ctx.is_mul_commutative(sin_x));
    }

    /// Anti-regression test: commutativity guards preserve canonicity correctly.
    ///
    /// - Matrices: A*B ≠ B*A (order preserved, not canonicalized to same form)
    /// - Scalars: x*y = y*x (both canonicalize to same form)
    #[test]
    fn test_canonicity_preserved_with_commutativity_guards() {
        let mut ctx = Context::new();

        // Matrix case: A*B and B*A should NOT be equal after canonicalization
        let a = ctx.num(1);
        let b = ctx.num(2);
        let c = ctx.num(3);
        let d = ctx.num(4);
        let mat_a = ctx.matrix(2, 2, vec![a, b, c, d]).unwrap();
        let mat_b = ctx.matrix(2, 2, vec![d, c, b, a]).unwrap();

        let ab = ctx.add(Expr::Mul(mat_a, mat_b)); // A*B
        let ba = ctx.add(Expr::Mul(mat_b, mat_a)); // B*A

        // Different ExprIds (order preserved, not sorted)
        assert_ne!(ab, ba, "Matrix multiplication must preserve order");

        // Scalar case: x*y and y*x should canonicalize to same form
        let x = ctx.var("x");
        let y = ctx.var("y");

        let xy = ctx.add(Expr::Mul(x, y)); // x*y
        let yx = ctx.add(Expr::Mul(y, x)); // y*x

        // Same ExprId (canonicalized to same order)
        assert_eq!(
            xy, yx,
            "Scalar multiplication must canonicalize to same form"
        );
    }

    #[test]
    fn test_binary_add_fast_path_preserves_canonical_order() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);

        let x_plus_one = ctx.add(Expr::Add(x, one));
        let one_plus_x = ctx.add(Expr::Add(one, x));

        assert_eq!(x_plus_one, one_plus_x);
        assert_eq!(ctx.get(x_plus_one), &Expr::Add(one, x));
    }

    #[test]
    fn test_binary_add_fast_path_keeps_positive_first() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let neg_y = ctx.add(Expr::Neg(y));

        let x_minus_y = ctx.add(Expr::Add(x, neg_y));
        let minus_y_plus_x = ctx.add(Expr::Add(neg_y, x));

        assert_eq!(x_minus_y, minus_y_plus_x);
        assert_eq!(ctx.get(x_minus_y), &Expr::Add(x, neg_y));
    }

    #[test]
    fn test_context_new_preloads_builtin_identity_layout() {
        let ctx = Context::new();
        assert_eq!(ctx.builtin_id(BuiltinFn::Sin), BuiltinFn::Sin as SymbolId);
        assert_eq!(ctx.builtin_id(BuiltinFn::Sqrt), BuiltinFn::Sqrt as SymbolId);
        assert_eq!(ctx.sym_name(ctx.builtin_id(BuiltinFn::Hold)), "__hold");
        assert_eq!(
            ctx.builtin_of(ctx.builtin_id(BuiltinFn::Expand)),
            Some(BuiltinFn::Expand)
        );
    }
}
