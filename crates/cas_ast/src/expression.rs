use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;
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
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Constant {
    Pi,
    E,
    Infinity,
    Undefined,
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
}

#[derive(Default, Clone)]
pub struct Context {
    pub nodes: Vec<Expr>,
    pub interner: HashMap<u64, ExprId>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            interner: HashMap::new(),
        }
    }

    pub fn add(&mut self, expr: Expr) -> ExprId {
        // Canonicalize commutative operations BEFORE adding to context
        // This ensures deterministic ordering: always left < right for Mul and Add
        let canonical_expr = match expr {
            Expr::Mul(l, r) => {
                // Only canonicalize if operands are out of order
                if crate::ordering::compare_expr(self, l, r) == std::cmp::Ordering::Greater {
                    Expr::Mul(r, l) // Swap to canonical order
                } else {
                    Expr::Mul(l, r) // Already canonical
                }
            }
            Expr::Add(l, r) => {
                if crate::ordering::compare_expr(self, l, r) == std::cmp::Ordering::Greater {
                    Expr::Add(r, l) // Swap to canonical order
                } else {
                    Expr::Add(l, r) // Already canonical
                }
            }
            // Non-commutative operations and atoms: keep as-is
            other => other,
        };

        // Expression Interning: Deduplicate expressions
        let mut hasher = DefaultHasher::new();
        canonical_expr.hash(&mut hasher);
        let hash = hasher.finish();

        if let Some(&id) = self.interner.get(&hash) {
            // Verify equality to handle hash collisions
            // We can safely panic on access out of bounds, but it shouldn't happen
            if self.nodes[id.index()] == canonical_expr {
                return id;
            }
            // Collision detected (same hash, different content)
            // We fall through to add a new node, effectively overwriting the map entry for this hash
            // This degrades deduplication for this bucket but maintains correctness
        }

        let index = self.nodes.len() as u32;

        // Determine tag based on expression type
        let tag = match &canonical_expr {
            Expr::Number(_) => ExprId::TAG_NUMBER,
            Expr::Variable(_) | Expr::Constant(_) => ExprId::TAG_ATOM,
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
        self.interner.insert(hash, id);
        id
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

    /// Create a matrix with given dimensions and data
    /// Panics if data.len() != rows * cols
    pub fn matrix(&mut self, rows: usize, cols: usize, data: Vec<ExprId>) -> ExprId {
        assert_eq!(
            data.len(),
            rows * cols,
            "Matrix data length {} does not match dimensions {}x{}",
            data.len(),
            rows,
            cols
        );
        self.add(Expr::Matrix { rows, cols, data })
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

// We need a way to display Expr with Context, or just Expr if it doesn't recurse.
// But Expr DOES recurse via IDs. So we can't implement Display for Expr easily without Context.
// We can implement a helper struct for display.

pub struct DisplayExpr<'a> {
    pub context: &'a Context,
    pub id: ExprId,
}

impl<'a> fmt::Display for DisplayExpr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let expr = self.context.get(self.id);
        match expr {
            Expr::Number(n) => write!(f, "{}", n),
            Expr::Constant(c) => match c {
                Constant::Pi => write!(f, "pi"),
                Constant::E => write!(f, "e"),
                Constant::Infinity => write!(f, "infinity"),
                Constant::Undefined => write!(f, "undefined"),
            },
            Expr::Variable(s) => write!(f, "{}", s),
            Expr::Add(_, _) => {
                // Flatten Add chain to handle mixed signs gracefully
                let mut terms = collect_add_terms(self.context, self.id);

                // Reorder for display: put positive terms first
                // This makes -√2 + √5 display as √5 - √2 which is cleaner
                if terms.len() >= 2 {
                    let (is_first_neg, _, _) = check_negative(self.context, terms[0]);
                    if is_first_neg {
                        // Find the first positive term and swap it to the front
                        if let Some(first_positive_idx) = terms
                            .iter()
                            .position(|t| !check_negative(self.context, *t).0)
                        {
                            terms.swap(0, first_positive_idx);
                        }
                    }
                }

                for (i, term) in terms.iter().enumerate() {
                    let (is_neg, _, _) = check_negative(self.context, *term);

                    if i == 0 {
                        // First term: print as is
                        write!(
                            f,
                            "{}",
                            DisplayExpr {
                                context: self.context,
                                id: *term
                            }
                        )?;
                    } else {
                        if is_neg {
                            // Print " - " then absolute value
                            write!(f, " - ")?;

                            // Re-check locally to extract positive part
                            match self.context.get(*term) {
                                Expr::Neg(inner) => {
                                    write!(
                                        f,
                                        "{}",
                                        DisplayExpr {
                                            context: self.context,
                                            id: *inner
                                        }
                                    )?;
                                }
                                Expr::Number(n) => {
                                    write!(f, "{}", -n)?;
                                }
                                Expr::Mul(a, b) => {
                                    if let Expr::Number(n) = self.context.get(*a) {
                                        let pos_n = -n;
                                        // Print pos_n * b
                                        let b_prec = precedence(self.context, *b);
                                        write!(f, "{} * ", pos_n)?;
                                        if b_prec < 2 {
                                            write!(
                                                f,
                                                "({})",
                                                DisplayExpr {
                                                    context: self.context,
                                                    id: *b
                                                }
                                            )?;
                                        } else {
                                            write!(
                                                f,
                                                "{}",
                                                DisplayExpr {
                                                    context: self.context,
                                                    id: *b
                                                }
                                            )?;
                                        }
                                    } else {
                                        // Should not happen if check_negative is correct
                                        write!(
                                            f,
                                            "{}",
                                            DisplayExpr {
                                                context: self.context,
                                                id: *term
                                            }
                                        )?;
                                    }
                                }
                                _ => {
                                    // Should not happen
                                    write!(
                                        f,
                                        "{}",
                                        DisplayExpr {
                                            context: self.context,
                                            id: *term
                                        }
                                    )?;
                                }
                            }
                        } else {
                            write!(
                                f,
                                " + {}",
                                DisplayExpr {
                                    context: self.context,
                                    id: *term
                                }
                            )?;
                        }
                    }
                }
                Ok(())
            }
            Expr::Sub(l, r) => {
                let rhs_prec = precedence(self.context, *r);
                let op_prec = 1; // Sub precedence

                // Check if RHS is Neg to wrap in parens: a - (-b)
                let rhs_is_neg = matches!(self.context.get(*r), Expr::Neg(_));

                if rhs_prec <= op_prec || rhs_is_neg {
                    write!(
                        f,
                        "{} - ({})",
                        DisplayExpr {
                            context: self.context,
                            id: *l
                        },
                        DisplayExpr {
                            context: self.context,
                            id: *r
                        }
                    )
                } else {
                    write!(
                        f,
                        "{} - {}",
                        DisplayExpr {
                            context: self.context,
                            id: *l
                        },
                        DisplayExpr {
                            context: self.context,
                            id: *r
                        }
                    )
                }
            }
            Expr::Mul(l, r) => {
                let lhs_prec = precedence(self.context, *l);
                let rhs_prec = precedence(self.context, *r);
                let op_prec = 2; // Mul precedence

                if lhs_prec < op_prec {
                    write!(
                        f,
                        "({})",
                        DisplayExpr {
                            context: self.context,
                            id: *l
                        }
                    )?
                } else {
                    write!(
                        f,
                        "{}",
                        DisplayExpr {
                            context: self.context,
                            id: *l
                        }
                    )?
                }

                write!(f, " * ")?;

                if rhs_prec < op_prec {
                    write!(
                        f,
                        "({})",
                        DisplayExpr {
                            context: self.context,
                            id: *r
                        }
                    )
                } else {
                    write!(
                        f,
                        "{}",
                        DisplayExpr {
                            context: self.context,
                            id: *r
                        }
                    )
                }
            }
            Expr::Div(l, r) => {
                let lhs_prec = precedence(self.context, *l);
                let rhs_prec = precedence(self.context, *r);
                let op_prec = 2; // Div precedence (same as Mul)

                if lhs_prec < op_prec {
                    write!(
                        f,
                        "({})",
                        DisplayExpr {
                            context: self.context,
                            id: *l
                        }
                    )?
                } else {
                    write!(
                        f,
                        "{}",
                        DisplayExpr {
                            context: self.context,
                            id: *l
                        }
                    )?
                }

                write!(f, " / ")?;

                // RHS of div always needs parens if it's Mul/Div or lower to be unambiguous?
                // a / b * c -> (a / b) * c usually.
                // a / (b * c).
                // If RHS is Mul/Div, we need parens: a / (b * c) vs a / b * c.
                if rhs_prec <= op_prec {
                    write!(
                        f,
                        "({})",
                        DisplayExpr {
                            context: self.context,
                            id: *r
                        }
                    )
                } else {
                    write!(
                        f,
                        "{}",
                        DisplayExpr {
                            context: self.context,
                            id: *r
                        }
                    )
                }
            }
            Expr::Pow(b, e) => {
                let base_prec = precedence(self.context, *b);
                let op_prec = 3; // Pow precedence

                if base_prec < op_prec {
                    write!(
                        f,
                        "({})",
                        DisplayExpr {
                            context: self.context,
                            id: *b
                        }
                    )?
                } else {
                    write!(
                        f,
                        "{}",
                        DisplayExpr {
                            context: self.context,
                            id: *b
                        }
                    )?
                }

                write!(f, "^")?;

                // Exponent usually doesn't need parens if it's simple, but for clarity maybe?
                // x^(a+b) needs parens.
                // x^2 doesn't.
                // If exponent is complex, wrap in parens.
                let exp_prec = precedence(self.context, *e);
                let needs_parens = if exp_prec <= 4 {
                    true
                } else if let Expr::Number(n) = self.context.get(*e) {
                    !n.is_integer() || *n < num_rational::BigRational::zero() // If fraction or negative, add parens: x^(1/2), x^(-1)
                } else {
                    false
                };

                if needs_parens {
                    write!(
                        f,
                        "({})",
                        DisplayExpr {
                            context: self.context,
                            id: *e
                        }
                    )
                } else {
                    write!(
                        f,
                        "{}",
                        DisplayExpr {
                            context: self.context,
                            id: *e
                        }
                    )
                }
            }
            Expr::Neg(e) => {
                let inner_prec = precedence(self.context, *e);
                // Check if inner is Neg to wrap in parens: -(-x)
                let inner_is_neg = matches!(self.context.get(*e), Expr::Neg(_));

                if inner_prec < 4 || inner_is_neg {
                    // Neg precedence
                    write!(
                        f,
                        "-({})",
                        DisplayExpr {
                            context: self.context,
                            id: *e
                        }
                    )
                } else {
                    write!(
                        f,
                        "-{}",
                        DisplayExpr {
                            context: self.context,
                            id: *e
                        }
                    )
                }
            }
            Expr::Function(name, args) => {
                if name == "abs" && args.len() == 1 {
                    write!(
                        f,
                        "|{}|",
                        DisplayExpr {
                            context: self.context,
                            id: args[0]
                        }
                    )
                } else if name == "log" && args.len() == 2 {
                    // Check if base is 'e'
                    let base = self.context.get(args[0]);
                    if let Expr::Constant(Constant::E) = base {
                        write!(
                            f,
                            "ln({})",
                            DisplayExpr {
                                context: self.context,
                                id: args[1]
                            }
                        )
                    } else {
                        write!(f, "{}(", name)?;
                        for (i, arg) in args.iter().enumerate() {
                            if i > 0 {
                                write!(f, ", ")?;
                            }
                            write!(
                                f,
                                "{}",
                                DisplayExpr {
                                    context: self.context,
                                    id: *arg
                                }
                            )?;
                        }
                        write!(f, ")")
                    }
                } else if name == "factored" {
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            write!(f, " * ")?;
                        }
                        write!(
                            f,
                            "{}",
                            DisplayExpr {
                                context: self.context,
                                id: *arg
                            }
                        )?;
                    }
                    Ok(())
                } else if name == "factored_pow" && args.len() == 2 {
                    write!(
                        f,
                        "{}^{}",
                        DisplayExpr {
                            context: self.context,
                            id: args[0]
                        },
                        DisplayExpr {
                            context: self.context,
                            id: args[1]
                        }
                    )
                } else {
                    write!(f, "{}(", name)?;
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(
                            f,
                            "{}",
                            DisplayExpr {
                                context: self.context,
                                id: *arg
                            }
                        )?;
                    }
                    write!(f, ")")
                }
            }
            Expr::Matrix { rows, cols, data } => {
                if *rows == 1 {
                    // Row vector: [a, b, c]
                    write!(f, "[")?;
                    for (i, elem) in data.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(
                            f,
                            "{}",
                            DisplayExpr {
                                context: self.context,
                                id: *elem
                            }
                        )?;
                    }
                    write!(f, "]")
                } else {
                    // Matrix or column vector: [[a, b], [c, d]]
                    write!(f, "[")?;
                    for r in 0..*rows {
                        if r > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "[")?;
                        for c in 0..*cols {
                            if c > 0 {
                                write!(f, ", ")?;
                            }
                            let idx = r * cols + c;
                            write!(
                                f,
                                "{}",
                                DisplayExpr {
                                    context: self.context,
                                    id: data[idx]
                                }
                            )?;
                        }
                        write!(f, "]")?;
                    }
                    write!(f, "]")
                }
            }
        }
    }
}

fn precedence(ctx: &Context, id: ExprId) -> i32 {
    match ctx.get(id) {
        Expr::Add(_, _) | Expr::Sub(_, _) => 1,
        Expr::Mul(_, _) | Expr::Div(_, _) => 2,
        Expr::Pow(_, _) => 3,
        Expr::Neg(_) => 4,
        Expr::Function(_, _)
        | Expr::Variable(_)
        | Expr::Number(_)
        | Expr::Constant(_)
        | Expr::Matrix { .. } => 5,
    }
}

fn collect_add_terms(ctx: &Context, id: ExprId) -> Vec<ExprId> {
    let mut terms = Vec::new();
    collect_add_terms_recursive(ctx, id, &mut terms);
    terms
}

fn collect_add_terms_recursive(ctx: &Context, id: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(id) {
        Expr::Add(l, r) => {
            collect_add_terms_recursive(ctx, *l, terms);
            collect_add_terms_recursive(ctx, *r, terms);
        }
        _ => terms.push(id),
    }
}

fn check_negative(ctx: &Context, id: ExprId) -> (bool, Option<ExprId>, Option<BigRational>) {
    match ctx.get(id) {
        Expr::Neg(inner) => (true, Some(*inner), None),
        Expr::Number(n) => {
            if *n < num_rational::BigRational::zero() {
                (true, None, Some(n.clone()))
            } else {
                (false, None, None)
            }
        }
        Expr::Mul(a, _) => {
            if let Expr::Number(n) = ctx.get(*a) {
                if *n < num_rational::BigRational::zero() {
                    (true, None, Some(n.clone()))
                } else {
                    (false, None, None)
                }
            } else {
                (false, None, None)
            }
        }
        _ => (false, None, None),
    }
}

pub fn count_nodes(context: &Context, id: ExprId) -> usize {
    match context.get(id) {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            1 + count_nodes(context, *l) + count_nodes(context, *r)
        }
        Expr::Neg(e) => 1 + count_nodes(context, *e),
        Expr::Function(_, args) => 1 + args.iter().map(|a| count_nodes(context, *a)).sum::<usize>(),
        Expr::Matrix { data, .. } => {
            1 + data.iter().map(|e| count_nodes(context, *e)).sum::<usize>()
        }
        _ => 1,
    }
}

pub struct RawDisplayExpr<'a> {
    pub context: &'a Context,
    pub id: ExprId,
}

impl<'a> fmt::Display for RawDisplayExpr<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let expr = self.context.get(self.id);
        match expr {
            Expr::Number(n) => write!(f, "{}", n),
            Expr::Constant(c) => match c {
                Constant::Pi => write!(f, "pi"),
                Constant::E => write!(f, "e"),
                Constant::Infinity => write!(f, "infinity"),
                Constant::Undefined => write!(f, "undefined"),
            },
            Expr::Variable(s) => write!(f, "{}", s),
            Expr::Add(l, r) => write!(
                f,
                "{} + {}",
                RawDisplayExpr {
                    context: self.context,
                    id: *l
                },
                RawDisplayExpr {
                    context: self.context,
                    id: *r
                }
            ),
            Expr::Sub(l, r) => write!(
                f,
                "{} - {}",
                RawDisplayExpr {
                    context: self.context,
                    id: *l
                },
                RawDisplayExpr {
                    context: self.context,
                    id: *r
                }
            ),
            Expr::Mul(l, r) => write!(
                f,
                "({}) * ({})",
                RawDisplayExpr {
                    context: self.context,
                    id: *l
                },
                RawDisplayExpr {
                    context: self.context,
                    id: *r
                }
            ),
            Expr::Div(l, r) => write!(
                f,
                "({}) / ({})",
                RawDisplayExpr {
                    context: self.context,
                    id: *l
                },
                RawDisplayExpr {
                    context: self.context,
                    id: *r
                }
            ),
            Expr::Pow(b, e) => write!(
                f,
                "({})^({})",
                RawDisplayExpr {
                    context: self.context,
                    id: *b
                },
                RawDisplayExpr {
                    context: self.context,
                    id: *e
                }
            ),
            Expr::Neg(e) => {
                let inner = self.context.get(*e);
                match inner {
                    Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) => write!(
                        f,
                        "-{}",
                        RawDisplayExpr {
                            context: self.context,
                            id: *e
                        }
                    ),
                    _ => write!(
                        f,
                        "-({})",
                        RawDisplayExpr {
                            context: self.context,
                            id: *e
                        }
                    ),
                }
            }
            Expr::Function(name, args) => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(
                        f,
                        "{}",
                        RawDisplayExpr {
                            context: self.context,
                            id: *arg
                        }
                    )?;
                }
                write!(f, ")")
            }
            Expr::Matrix { rows, cols, data } => {
                // Raw display: show structure explicitly
                write!(f, "Matrix({}x{}, [", rows, cols)?;
                for (i, elem) in data.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(
                        f,
                        "{}",
                        RawDisplayExpr {
                            context: self.context,
                            id: *elem
                        }
                    )?;
                }
                write!(f, "])")
            }
        }
    }
}

/// DisplayExpr with hints for preserving original notation (e.g., sqrt)
pub struct DisplayExprWithHints<'a> {
    pub context: &'a Context,
    pub id: ExprId,
    pub hints: &'a crate::display_context::DisplayContext,
}

impl<'a> DisplayExprWithHints<'a> {
    fn fmt_internal(&self, f: &mut fmt::Formatter<'_>, id: ExprId) -> fmt::Result {
        // Check for display hint first
        if let Some(hint) = self.hints.get(id) {
            match hint {
                crate::display_context::DisplayHint::AsRoot { index } => {
                    // Render as root notation
                    if let Expr::Pow(base, exp) = self.context.get(id) {
                        // Get the numerator of the exponent (for x^(k/n), k is the power inside the root)
                        let inner_power = if let Expr::Number(n) = self.context.get(*exp) {
                            let numer: i64 = n.numer().try_into().unwrap_or(1);
                            numer
                        } else {
                            1
                        };

                        if *index == 2 {
                            write!(f, "√(")?;
                        } else {
                            write!(f, "{}√(", index)?;
                        }

                        // If numerator > 1, show base^k inside the root
                        if inner_power != 1 {
                            self.fmt_internal(f, *base)?;
                            write!(f, "^{}", inner_power)?;
                        } else {
                            self.fmt_internal(f, *base)?;
                        }
                        return write!(f, ")");
                    }
                }
            }
        }

        // Regular formatting (delegate to DisplayExpr logic)
        let expr = self.context.get(id);
        match expr {
            Expr::Number(n) => write!(f, "{}", n),
            Expr::Constant(c) => match c {
                Constant::Pi => write!(f, "pi"),
                Constant::E => write!(f, "e"),
                Constant::Infinity => write!(f, "infinity"),
                Constant::Undefined => write!(f, "undefined"),
            },
            Expr::Variable(s) => write!(f, "{}", s),
            Expr::Add(_, _) => {
                // Flatten Add chain to handle mixed signs gracefully
                let mut terms = collect_add_terms(self.context, id);

                // Reorder for display: put positive terms first
                // This makes -x + y display as y - x which is cleaner
                if terms.len() >= 2 {
                    let (is_first_neg, _, _) = check_negative(self.context, terms[0]);
                    if is_first_neg {
                        // Find the first positive term and swap it to the front
                        if let Some(first_positive_idx) = terms
                            .iter()
                            .position(|t| !check_negative(self.context, *t).0)
                        {
                            terms.swap(0, first_positive_idx);
                        }
                    }
                }

                for (i, term) in terms.iter().enumerate() {
                    let (is_neg, _, _) = check_negative(self.context, *term);

                    if i == 0 {
                        // First term: print as is
                        self.fmt_internal(f, *term)?;
                    } else {
                        if is_neg {
                            // Print " - " then absolute value
                            write!(f, " - ")?;
                            // Extract positive part
                            match self.context.get(*term) {
                                Expr::Neg(inner) => {
                                    self.fmt_internal(f, *inner)?;
                                }
                                Expr::Number(n) => {
                                    write!(f, "{}", -n)?;
                                }
                                Expr::Mul(a, b) => {
                                    if let Expr::Number(n) = self.context.get(*a) {
                                        let pos_n = -n;
                                        write!(f, "{} * ", pos_n)?;
                                        self.fmt_internal(f, *b)?;
                                    } else {
                                        self.fmt_internal(f, *term)?;
                                    }
                                }
                                _ => {
                                    self.fmt_internal(f, *term)?;
                                }
                            }
                        } else {
                            write!(f, " + ")?;
                            self.fmt_internal(f, *term)?;
                        }
                    }
                }
                Ok(())
            }
            Expr::Sub(l, r) => {
                self.fmt_internal(f, *l)?;
                write!(f, " - ")?;
                self.fmt_internal(f, *r)
            }
            Expr::Mul(l, r) => {
                let lhs_prec = precedence(self.context, *l);
                let rhs_prec = precedence(self.context, *r);
                let op_prec = 2; // Mul precedence

                if lhs_prec < op_prec {
                    write!(f, "(")?;
                    self.fmt_internal(f, *l)?;
                    write!(f, ")")?;
                } else {
                    self.fmt_internal(f, *l)?;
                }

                write!(f, " * ")?;

                if rhs_prec < op_prec {
                    write!(f, "(")?;
                    self.fmt_internal(f, *r)?;
                    write!(f, ")")?;
                    Ok(())
                } else {
                    self.fmt_internal(f, *r)
                }
            }
            Expr::Div(l, r) => {
                let lhs_prec = precedence(self.context, *l);
                let rhs_prec = precedence(self.context, *r);
                let op_prec = 2; // Div precedence (same as Mul)

                if lhs_prec < op_prec {
                    write!(f, "(")?;
                    self.fmt_internal(f, *l)?;
                    write!(f, ")")?;
                } else {
                    self.fmt_internal(f, *l)?;
                }

                write!(f, " / ")?;

                // RHS of div always needs parens if it's Mul/Div or lower to be unambiguous
                if rhs_prec <= op_prec {
                    write!(f, "(")?;
                    self.fmt_internal(f, *r)?;
                    write!(f, ")")?;
                    Ok(())
                } else {
                    self.fmt_internal(f, *r)
                }
            }
            Expr::Pow(b, e) => {
                // Add parentheses around base if it's a binary operation
                let base_expr = self.context.get(*b);
                let needs_parens = matches!(
                    base_expr,
                    Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Mul(_, _) | Expr::Div(_, _)
                );
                if needs_parens {
                    write!(f, "(")?;
                }
                self.fmt_internal(f, *b)?;
                if needs_parens {
                    write!(f, ")")?;
                }
                write!(f, "^(")?;
                self.fmt_internal(f, *e)?;
                write!(f, ")")
            }
            Expr::Neg(e) => {
                write!(f, "-")?;
                self.fmt_internal(f, *e)
            }
            Expr::Function(name, args) => {
                // Special handling for sqrt/root functions - always render as √
                if name == "sqrt" && args.len() >= 1 {
                    let index = if args.len() == 2 {
                        if let Expr::Number(n) = self.context.get(args[1]) {
                            n.to_integer().try_into().unwrap_or(2u32)
                        } else {
                            2u32
                        }
                    } else {
                        2u32
                    };
                    if index == 2 {
                        write!(f, "√(")?;
                    } else {
                        write!(f, "{}√(", index)?;
                    }
                    self.fmt_internal(f, args[0])?;
                    return write!(f, ")");
                } else if name == "root" && args.len() == 2 {
                    let index = if let Expr::Number(n) = self.context.get(args[1]) {
                        n.to_integer().try_into().unwrap_or(2u32)
                    } else {
                        2u32
                    };
                    if index == 2 {
                        write!(f, "√(")?;
                    } else {
                        write!(f, "{}√(", index)?;
                    }
                    self.fmt_internal(f, args[0])?;
                    return write!(f, ")");
                }
                // Regular function format
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    self.fmt_internal(f, *arg)?;
                }
                write!(f, ")")
            }
            Expr::Matrix { rows, cols, data } => {
                write!(f, "Matrix({}x{}, [", rows, cols)?;
                for (i, elem) in data.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    self.fmt_internal(f, *elem)?;
                }
                write!(f, "])")
            }
        }
    }
}

impl<'a> fmt::Display for DisplayExprWithHints<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_internal(f, self.id)
    }
}
