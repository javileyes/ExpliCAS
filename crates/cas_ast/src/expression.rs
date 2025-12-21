use num_bigint::BigInt;
use num_rational::BigRational;
use std::collections::HashMap;
use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};

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
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    Number(BigRational),
    Constant(Constant),
    Variable(String),
    Add(ExprId, ExprId),
    Sub(ExprId, ExprId),
    Mul(ExprId, ExprId),
    Div(ExprId, ExprId),
    Pow(ExprId, ExprId),
    Neg(ExprId),
    Function(String, Vec<ExprId>),
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
}

#[derive(Default, Clone)]
pub struct Context {
    pub nodes: Vec<Expr>,
    /// Interner: maps hash to bucket of ExprIds with that hash.
    /// Using Vec<ExprId> instead of single ExprId to properly handle hash collisions
    /// without losing deduplication for expressions that share the same hash.
    pub interner: HashMap<u64, Vec<ExprId>>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            interner: HashMap::new(),
        }
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
                // Collect all additive terms (flatten nested Adds)
                let mut terms = Vec::new();
                self.collect_add_terms(l, &mut terms);
                self.collect_add_terms(r, &mut terms);
                // Sort terms: positive first, then by compare_expr
                terms.sort_by(|a, b| self.compare_add_terms(*a, *b));
                // Build right-associative tree: a + (b + (c + d))
                self.build_balanced_add(&terms)
            }
            Expr::Mul(l, r) => {
                // Check if any factor contains a matrix (non-commutative multiplication)
                if self.contains_matrix(l) || self.contains_matrix(r) {
                    // Non-commutative: flatten for associativity but do NOT sort
                    let mut factors = Vec::new();
                    self.collect_mul_factors(l, &mut factors);
                    self.collect_mul_factors(r, &mut factors);
                    // Rebuild balanced (preserves order)
                    self.build_balanced_mul(&factors)
                } else {
                    // Commutative: flatten + sort (using order_key to avoid recursive compare)
                    let mut factors = Vec::new();
                    self.collect_mul_factors(l, &mut factors);
                    self.collect_mul_factors(r, &mut factors);
                    // Sort by structural comparison (balanced tree prevents deep recursion)
                    factors.sort_by(|a, b| crate::ordering::compare_expr(self, *a, *b));
                    self.build_balanced_mul(&factors)
                }
            }
            // Non-commutative operations and atoms: keep as-is
            other => other,
        };

        // Expression Interning: Deduplicate expressions
        let mut hasher = DefaultHasher::new();
        canonical_expr.hash(&mut hasher);
        let hash = hasher.finish();

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

        // Determine tag based on expression type
        let tag = match &canonical_expr {
            Expr::Number(_) => ExprId::TAG_NUMBER,
            Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => ExprId::TAG_ATOM,
            Expr::Neg(_) => ExprId::TAG_UNARY,
            Expr::Add(_, _)
            | Expr::Sub(_, _)
            | Expr::Mul(_, _)
            | Expr::Div(_, _)
            | Expr::Pow(_, _) => ExprId::TAG_BINARY,
            Expr::Function(_, _) | Expr::Matrix { .. } => ExprId::TAG_NARY,
        };

        let id = ExprId::new(index, tag);
        self.nodes.push(canonical_expr);

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
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // DEBUG: Catch Variables being created via add_raw (should use ctx.var() instead)
        debug_assert!(
            !matches!(expr, Expr::Variable(_)),
            "add_raw must not be used for Variable; use ctx.var() or ctx.add() instead"
        );

        // Expression Interning: Deduplicate expressions
        let mut hasher = DefaultHasher::new();
        expr.hash(&mut hasher);
        let hash = hasher.finish();

        // Check if we already have this expression in the bucket
        if let Some(bucket) = self.interner.get(&hash) {
            for &id in bucket {
                if self.nodes[id.index()] == expr {
                    return id; // Found existing expression
                }
            }
        }

        let index = self.nodes.len() as u32;

        let tag = match &expr {
            Expr::Number(_) => ExprId::TAG_NUMBER,
            Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => ExprId::TAG_ATOM,
            Expr::Neg(_) => ExprId::TAG_UNARY,
            Expr::Add(_, _)
            | Expr::Sub(_, _)
            | Expr::Mul(_, _)
            | Expr::Div(_, _)
            | Expr::Pow(_, _) => ExprId::TAG_BINARY,
            Expr::Function(_, _) | Expr::Matrix { .. } => ExprId::TAG_NARY,
        };

        let id = ExprId::new(index, tag);
        self.nodes.push(expr);

        self.interner.entry(hash).or_default().push(id);
        id
    }

    /// Collect all additive terms by flattening nested Add (iterative)
    fn collect_add_terms(&self, id: ExprId, terms: &mut Vec<ExprId>) {
        let mut stack = vec![id];
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
            0 => panic!("Cannot build Add from empty terms"),
            1 => return self.get(terms[0]).clone(),
            2 => Expr::Add(terms[0], terms[1]),
            _ => {
                // Build pairs bottom-up iteratively
                let mut current: Vec<ExprId> = terms.to_vec();
                while current.len() > 2 {
                    let mut next = Vec::with_capacity((current.len() + 1) / 2);
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
    fn collect_mul_factors(&self, id: ExprId, factors: &mut Vec<ExprId>) {
        let mut stack = vec![id];
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

    /// Build balanced Mul tree iteratively: [a,b,c,d] -> Mul(Mul(a,b), Mul(c,d))
    fn build_balanced_mul(&mut self, factors: &[ExprId]) -> Expr {
        match factors.len() {
            0 => panic!("Cannot build Mul from empty factors"),
            1 => return self.get(factors[0]).clone(),
            2 => Expr::Mul(factors[0], factors[1]),
            _ => {
                // Build pairs bottom-up iteratively
                let mut current: Vec<ExprId> = factors.to_vec();
                while current.len() > 2 {
                    let mut next = Vec::with_capacity((current.len() + 1) / 2);
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
                    Expr::Mul(current[0], current[1])
                } else {
                    self.get(current[0]).clone()
                }
            }
        }
    }

    /// Check if an expression contains a Matrix anywhere in its tree (iterative)
    fn contains_matrix(&self, id: ExprId) -> bool {
        let mut stack = vec![id];
        while let Some(current) = stack.pop() {
            match self.get(current) {
                Expr::Matrix { .. } => return true,
                Expr::Mul(l, r)
                | Expr::Add(l, r)
                | Expr::Sub(l, r)
                | Expr::Div(l, r)
                | Expr::Pow(l, r) => {
                    stack.push(*l);
                    stack.push(*r);
                }
                Expr::Neg(inner) => stack.push(*inner),
                Expr::Function(_, args) => stack.extend(args.iter().copied()),
                _ => {}
            }
        }
        false
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
        self.add(Expr::Variable(name.to_string()))
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
