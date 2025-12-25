//! Display formatting for expressions
//!
//! This module provides display implementations for expressions:
//! - `DisplayExpr`: Basic expression display
//! - `DisplayExprWithHints`: Display with rendering hints (roots, etc.)
//! - `RawDisplayExpr`: Debug-style display

use crate::{Constant, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{ToPrimitive, Zero};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

// ============================================================================
// Pretty Output Configuration
// ============================================================================

/// Global flag controlling pretty Unicode output (∛, x², ·) vs ASCII (sqrt, ^, *)
/// Default: false (ASCII mode for tests compatibility)
/// CLI enables pretty mode for end users
static PRETTY_OUTPUT: AtomicBool = AtomicBool::new(false);

/// Enable pretty Unicode output (∛, x², ·)
pub fn enable_pretty_output() {
    PRETTY_OUTPUT.store(true, Ordering::SeqCst);
}

/// Disable pretty output, use ASCII (sqrt, ^, *)
pub fn disable_pretty_output() {
    PRETTY_OUTPUT.store(false, Ordering::SeqCst);
}

/// Check if pretty output is enabled
pub fn is_pretty_output() -> bool {
    PRETTY_OUTPUT.load(Ordering::SeqCst)
}

/// Get the multiplication symbol based on pretty mode
pub fn mul_symbol() -> &'static str {
    if PRETTY_OUTPUT.load(Ordering::SeqCst) {
        "·"
    } else {
        " * "
    }
}

// ============================================================================
// Unicode Pretty Output Helpers
// ============================================================================

/// Convert a digit (0-9) to its Unicode superscript equivalent
fn digit_to_superscript(d: u32) -> char {
    match d {
        0 => '⁰',
        1 => '¹',
        2 => '²',
        3 => '³',
        4 => '⁴',
        5 => '⁵',
        6 => '⁶',
        7 => '⁷',
        8 => '⁸',
        9 => '⁹',
        _ => '?',
    }
}

/// Convert an integer to a Unicode superscript string
/// Examples: 2 → "²", 12 → "¹²", 100 → "¹⁰⁰"
pub fn number_to_superscript(n: u64) -> String {
    if n == 0 {
        return "⁰".to_string();
    }

    let mut result = String::new();
    let mut num = n;
    let mut digits = Vec::new();

    while num > 0 {
        digits.push((num % 10) as u32);
        num /= 10;
    }

    for d in digits.into_iter().rev() {
        result.push(digit_to_superscript(d));
    }

    result
}

/// Get the root prefix for a given index
/// Pretty mode: 2 → "√", 3 → "∛", 4 → "∜", 5 → "⁵√"
/// ASCII mode: 2 → "sqrt", 3 → "cbrt", n → "root(,n)"
pub fn unicode_root_prefix(index: u64) -> String {
    if !is_pretty_output() {
        // ASCII mode - will be handled by the caller using sqrt() format
        return match index {
            2 => "sqrt".to_string(),
            n => format!("{}√", n), // Fallback, normally caller handles ASCII
        };
    }

    match index {
        2 => "√".to_string(),
        3 => "∛".to_string(),
        4 => "∜".to_string(),
        n => format!("{}√", number_to_superscript(n)),
    }
}

// We need a way to display Expr with Context, or just Expr if it doesn't recurse.
// But Expr DOES recurse via IDs. So we can't implement Display for Expr easily without Context.
// We can implement a helper struct for display.

/// Factor with exponent for display.
#[derive(Debug, Clone, Copy)]
pub struct DisplayFactor {
    pub base: ExprId,
    pub exp: i32, // Always positive for display purposes
}

/// Read-only view of an expression as a fraction for display.
///
/// Collects numerator/denominator factors WITHOUT building new AST nodes.
/// Uses &Context (immutable) only.
#[derive(Debug)]
pub struct FractionDisplayView {
    pub sign: i8,
    pub num: Vec<DisplayFactor>, // Factors in numerator (exp > 0)
    pub den: Vec<DisplayFactor>, // Factors in denominator (exp originally < 0)
}

impl FractionDisplayView {
    /// Try to interpret expression as a fraction.
    ///
    /// Returns None if:
    /// - Not a Mul or Pow expression
    /// - Contains matrices (non-commutative)
    /// - Has no denominator factors
    pub fn from(ctx: &Context, id: ExprId) -> Option<Self> {
        // Only process Mul or Pow
        match ctx.get(id) {
            Expr::Mul(_, _) | Expr::Pow(_, _) => {}
            _ => return None,
        }

        let mut sign: i8 = 1;
        let mut num = Vec::new();
        let mut den = Vec::new();
        let mut worklist = vec![id];

        while let Some(curr) = worklist.pop() {
            match ctx.get(curr) {
                Expr::Matrix { .. } => return None, // Matrix blocks fraction display

                Expr::Neg(inner) => {
                    sign *= -1;
                    worklist.push(*inner);
                }

                Expr::Mul(l, r) => {
                    // Check for matrices in either operand
                    if matches!(ctx.get(*l), Expr::Matrix { .. })
                        || matches!(ctx.get(*r), Expr::Matrix { .. })
                    {
                        return None;
                    }
                    worklist.push(*l);
                    worklist.push(*r);
                }

                Expr::Pow(base, exp_id) => {
                    if matches!(ctx.get(*base), Expr::Matrix { .. }) {
                        return None;
                    }

                    // Check if exponent is integer
                    if let Some(exp) = Self::as_int(ctx, *exp_id) {
                        if exp < 0 {
                            den.push(DisplayFactor {
                                base: *base,
                                exp: -exp,
                            });
                        } else if exp > 0 {
                            num.push(DisplayFactor { base: *base, exp });
                        }
                        // exp == 0 means factor is 1, skip
                    } else {
                        // Non-integer exponent: treat as single num factor
                        num.push(DisplayFactor { base: curr, exp: 1 });
                    }
                }

                Expr::Div(n, d) => {
                    // Add numerator factors, denominator factors
                    worklist.push(*n);
                    // Denominator goes to den with exp=1
                    if matches!(ctx.get(*d), Expr::Matrix { .. }) {
                        return None;
                    }
                    den.push(DisplayFactor { base: *d, exp: 1 });
                }

                // Atoms: add as numerator factor
                _ => {
                    num.push(DisplayFactor { base: curr, exp: 1 });
                }
            }
        }

        // Only return if there's actually a denominator
        if den.is_empty() {
            return None;
        }

        Some(FractionDisplayView { sign, num, den })
    }

    /// Extract integer from an expression.
    fn as_int(ctx: &Context, id: ExprId) -> Option<i32> {
        if let Expr::Number(n) = ctx.get(id) {
            if n.is_integer() {
                return n.to_integer().to_i32();
            }
        }
        None
    }
}

/// Format a list of factors as a product for display.
fn format_factors(
    f: &mut fmt::Formatter<'_>,
    ctx: &Context,
    factors: &[DisplayFactor],
) -> fmt::Result {
    if factors.is_empty() {
        return write!(f, "1");
    }

    for (i, factor) in factors.iter().enumerate() {
        if i > 0 {
            write!(f, "{}", mul_symbol())?;
        }

        let base_prec = precedence(ctx, factor.base);
        let needs_parens = base_prec < 2; // Mul precedence

        if needs_parens {
            write!(
                f,
                "({})",
                DisplayExpr {
                    context: ctx,
                    id: factor.base
                }
            )?;
        } else {
            write!(
                f,
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: factor.base
                }
            )?;
        }

        if factor.exp != 1 {
            write!(f, "^{}", factor.exp)?;
        }
    }

    Ok(())
}

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
                Constant::I => write!(f, "i"),
            },
            Expr::Variable(s) => write!(f, "{}", s),
            Expr::Add(_, _) => {
                // Flatten Add chain to handle mixed signs gracefully
                let mut terms = collect_add_terms(self.context, self.id);

                // Reorder for display: put positive terms first, negative terms last
                // This makes -x + √x² display as √x² - x which is cleaner
                // Use stable_sort to preserve relative order within positive/negative groups
                terms.sort_by(|a, b| {
                    let a_neg = check_negative(self.context, *a).0;
                    let b_neg = check_negative(self.context, *b).0;
                    match (a_neg, b_neg) {
                        (false, true) => std::cmp::Ordering::Less, // positive before negative
                        (true, false) => std::cmp::Ordering::Greater, // negative after positive
                        _ => std::cmp::Ordering::Equal,            // same sign: keep order
                    }
                });

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
                    } else if is_neg {
                        // Print " - " then absolute value
                        write!(f, " - ")?;

                        // Re-check locally to extract positive part
                        match self.context.get(*term) {
                            Expr::Neg(inner) => {
                                // If inner is Add/Sub, wrap in parentheses to preserve grouping
                                // e.g., -(a - b) should display as "-(a - b)" not "-a - b"
                                let inner_is_add_sub = matches!(
                                    self.context.get(*inner),
                                    Expr::Add(_, _) | Expr::Sub(_, _)
                                );
                                if inner_is_add_sub {
                                    write!(
                                        f,
                                        "({})",
                                        DisplayExpr {
                                            context: self.context,
                                            id: *inner
                                        }
                                    )?;
                                } else {
                                    write!(
                                        f,
                                        "{}",
                                        DisplayExpr {
                                            context: self.context,
                                            id: *inner
                                        }
                                    )?;
                                }
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
                Ok(())
            }
            Expr::Sub(l, r) => {
                let rhs_prec = precedence(self.context, *r);
                let op_prec = 1; // Sub precedence

                // Check if RHS needs parens:
                // 1. RHS is Neg: a - (-b) needs parens
                // 2. RHS is Add/Sub: a - (b + c) or a - (b - c) needs parens to preserve associativity
                // 3. RHS has lower/equal precedence
                let rhs_is_neg = matches!(self.context.get(*r), Expr::Neg(_));
                let rhs_is_add_sub =
                    matches!(self.context.get(*r), Expr::Add(_, _) | Expr::Sub(_, _));

                if rhs_prec <= op_prec || rhs_is_neg || rhs_is_add_sub {
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
                // P3: Try to display as fraction using FractionDisplayView
                if let Some(frac) = FractionDisplayView::from(self.context, self.id) {
                    // Handle sign
                    if frac.sign < 0 {
                        write!(f, "-")?;
                    }

                    // Format numerator
                    let needs_num_parens =
                        frac.num.len() > 1 || frac.num.iter().any(|f| f.exp != 1);
                    if frac.num.is_empty() {
                        write!(f, "1")?;
                    } else if needs_num_parens && frac.num.len() > 1 {
                        write!(f, "(")?;
                        format_factors(f, self.context, &frac.num)?;
                        write!(f, ")")?;
                    } else {
                        format_factors(f, self.context, &frac.num)?;
                    }

                    write!(f, "/")?;

                    // Format denominator (always needs parens if multiple factors)
                    if frac.den.len() > 1 || frac.den.iter().any(|d| d.exp != 1) {
                        write!(f, "(")?;
                        format_factors(f, self.context, &frac.den)?;
                        return write!(f, ")");
                    } else {
                        // Single factor - check if it needs parens based on precedence
                        let den_base_prec = precedence(self.context, frac.den[0].base);
                        if den_base_prec <= 2 {
                            write!(f, "(")?;
                            format_factors(f, self.context, &frac.den)?;
                            return write!(f, ")");
                        } else {
                            return format_factors(f, self.context, &frac.den);
                        }
                    }
                }

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

                write!(f, "{}", mul_symbol())?;

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
                // If exponent is 1, just display the base (no ^1)
                if let Expr::Number(n) = self.context.get(*e) {
                    if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                        return write!(
                            f,
                            "{}",
                            DisplayExpr {
                                context: self.context,
                                id: *b
                            }
                        );
                    }
                }

                // If base is 1, just display "1" (1^n = 1)
                if let Expr::Number(n) = self.context.get(*b) {
                    if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                        return write!(f, "1");
                    }
                }

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
                // __hold is an internal invisible barrier - just display the inner
                if name == "__hold" && args.len() == 1 {
                    return write!(
                        f,
                        "{}",
                        DisplayExpr {
                            context: self.context,
                            id: args[0]
                        }
                    );
                }
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
                            write!(f, "{}", mul_symbol())?;
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
            Expr::SessionRef(id) => write!(f, "#{}", id),
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
        | Expr::Matrix { .. }
        | Expr::SessionRef(_) => 5,
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
        Expr::Mul(a, b) => {
            // Check left factor for negative number
            if let Expr::Number(n) = ctx.get(*a) {
                if *n < num_rational::BigRational::zero() {
                    return (true, None, Some(n.clone()));
                }
            }
            // Check right factor for negative number (in case of canonicalization order)
            if let Expr::Number(n) = ctx.get(*b) {
                if *n < num_rational::BigRational::zero() {
                    return (true, None, Some(n.clone()));
                }
            }
            (false, None, None)
        }
        _ => (false, None, None),
    }
}

/// Compute the "polynomial degree" of a term for display ordering.
///
/// Returns the total degree of all power terms (sum of exponents).
/// Constants and numbers have degree 0, variables have degree 1.
/// Used to sort polynomial sums: x^3, x^2, x, 1
fn polynomial_degree(ctx: &Context, id: ExprId) -> i32 {
    match ctx.get(id) {
        // Constants have degree 0
        Expr::Number(_) | Expr::Constant(_) => 0,

        // Variables have degree 1
        Expr::Variable(_) => 1,

        // Power: get degree from exponent
        Expr::Pow(base, exp) => {
            // Check if exponent is numeric
            if let Expr::Number(n) = ctx.get(*exp) {
                if let Some(exp_int) = n.to_integer().to_i32() {
                    // For x^n, if base is variable, degree = n
                    // For expression^n, degree = base_degree * n
                    let base_deg = polynomial_degree(ctx, *base);
                    if base_deg > 0 {
                        return base_deg * exp_int;
                    }
                    // If base is number/constant, this is just a constant
                    return 0;
                }
            }
            // Non-integer exponent: treat base as having its own degree
            polynomial_degree(ctx, *base)
        }

        // Mul: sum the degrees (c * x^2 * y has degree 3 if y is variable)
        Expr::Mul(l, r) => {
            let ld = polynomial_degree(ctx, *l);
            let rd = polynomial_degree(ctx, *r);
            ld + rd
        }

        // Neg: same degree as inner
        Expr::Neg(inner) => polynomial_degree(ctx, *inner),

        // Div: numerator degree minus denominator degree
        Expr::Div(num, den) => polynomial_degree(ctx, *num) - polynomial_degree(ctx, *den),

        // Add/Sub: take max degree (shouldn't happen for terms, but just in case)
        Expr::Add(l, r) | Expr::Sub(l, r) => {
            std::cmp::max(polynomial_degree(ctx, *l), polynomial_degree(ctx, *r))
        }

        // Functions: treat as degree 0 for ordering purposes
        Expr::Function(_, _) | Expr::Matrix { .. } | Expr::SessionRef(_) => 0,
    }
}

/// Count all nodes in an expression tree.
///
/// Wrapper calling canonical `crate::traversal::count_all_nodes`.
/// (See POLICY.md "Traversal Contract")
#[deprecated(
    since = "0.1.0",
    note = "Use crate::traversal::count_all_nodes instead"
)]
pub fn count_nodes(context: &Context, id: ExprId) -> usize {
    crate::traversal::count_all_nodes(context, id)
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
                Constant::I => write!(f, "i"),
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
            Expr::SessionRef(id) => write!(f, "#{}", id),
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

                        write!(f, "{}(", unicode_root_prefix(*index as u64))?;

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
                Constant::I => write!(f, "i"),
            },
            Expr::Variable(s) => write!(f, "{}", s),
            Expr::Add(_, _) => {
                // Flatten Add chain to handle mixed signs gracefully
                let mut terms = collect_add_terms(self.context, id);

                // Reorder for display: put positive terms first, negative terms last
                // This makes -x + |x| display as |x| - x which is cleaner
                terms.sort_by(|a, b| {
                    let a_neg = check_negative(self.context, *a).0;
                    let b_neg = check_negative(self.context, *b).0;
                    match (a_neg, b_neg) {
                        (false, true) => std::cmp::Ordering::Less, // positive before negative
                        (true, false) => std::cmp::Ordering::Greater, // negative after positive
                        _ => std::cmp::Ordering::Equal,            // same sign: keep order
                    }
                });

                for (i, term) in terms.iter().enumerate() {
                    let (is_neg, _, _) = check_negative(self.context, *term);

                    if i == 0 {
                        // First term: print as is
                        self.fmt_internal(f, *term)?;
                    } else if is_neg {
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
                Ok(())
            }
            Expr::Sub(l, r) => {
                self.fmt_internal(f, *l)?;
                write!(f, " - ")?;
                self.fmt_internal(f, *r)
            }
            Expr::Mul(l, r) => {
                // Sign pull-out: if RHS is an Add with all negative terms,
                // display as -(l * |RHS|) for cleaner output
                // e.g., 1/11 * (-3 - 2√5) -> -(1/11 * (3 + 2√5)) -> -(3 + 2√5)/11
                if crate::views::has_all_negative_terms(self.context, *r) {
                    // Factor out the negative and display the positive version
                    return self.fmt_mul_with_sign_pullout(f, *l, *r);
                }
                // Same for LHS (less common but possible)
                if crate::views::has_all_negative_terms(self.context, *l) {
                    return self.fmt_mul_with_sign_pullout(f, *r, *l);
                }

                // P3: Try to display as fraction using FractionDisplayView
                if let Some(frac) = FractionDisplayView::from(self.context, self.id) {
                    // Handle sign
                    if frac.sign < 0 {
                        write!(f, "-")?;
                    }

                    // Format numerator using fmt_internal for hints
                    if frac.num.is_empty() {
                        write!(f, "1")?;
                    } else if frac.num.len() > 1 {
                        write!(f, "(")?;
                        for (i, factor) in frac.num.iter().enumerate() {
                            if i > 0 {
                                write!(f, "{}", mul_symbol())?;
                            }
                            self.fmt_internal(f, factor.base)?;
                            if factor.exp != 1 {
                                write!(f, "^{}", factor.exp)?;
                            }
                        }
                        write!(f, ")")?;
                    } else {
                        let factor = &frac.num[0];
                        self.fmt_internal(f, factor.base)?;
                        if factor.exp != 1 {
                            write!(f, "^{}", factor.exp)?;
                        }
                    }

                    write!(f, "/")?;

                    // Format denominator
                    if frac.den.len() > 1 || frac.den.iter().any(|d| d.exp != 1) {
                        write!(f, "(")?;
                        for (i, factor) in frac.den.iter().enumerate() {
                            if i > 0 {
                                write!(f, "{}", mul_symbol())?;
                            }
                            self.fmt_internal(f, factor.base)?;
                            if factor.exp != 1 {
                                write!(f, "^{}", factor.exp)?;
                            }
                        }
                        return write!(f, ")");
                    } else {
                        let den_base_prec = precedence(self.context, frac.den[0].base);
                        if den_base_prec <= 2 {
                            write!(f, "(")?;
                            self.fmt_internal(f, frac.den[0].base)?;
                            return write!(f, ")");
                        } else {
                            return self.fmt_internal(f, frac.den[0].base);
                        }
                    }
                }

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

                write!(f, "{}", mul_symbol())?;

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
                // If exponent is 1, just display the base (no ^1)
                if let Expr::Number(n) = self.context.get(*e) {
                    if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                        return self.fmt_internal(f, *b);
                    }
                }

                // If base is 1, just display "1" (1^n = 1)
                if let Expr::Number(n) = self.context.get(*b) {
                    if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                        return write!(f, "1");
                    }
                }

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
                if name == "sqrt" && !args.is_empty() {
                    let index = if args.len() == 2 {
                        if let Expr::Number(n) = self.context.get(args[1]) {
                            n.to_integer().try_into().unwrap_or(2u32)
                        } else {
                            2u32
                        }
                    } else {
                        2u32
                    };
                    write!(f, "{}(", unicode_root_prefix(index as u64))?;
                    self.fmt_internal(f, args[0])?;
                    return write!(f, ")");
                } else if name == "root" && args.len() == 2 {
                    let index = if let Expr::Number(n) = self.context.get(args[1]) {
                        n.to_integer().try_into().unwrap_or(2u32)
                    } else {
                        2u32
                    };
                    write!(f, "{}(", unicode_root_prefix(index as u64))?;
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
            Expr::SessionRef(id) => write!(f, "#{}", id),
        }
    }

    /// Format a Mul where one factor has all-negative Add terms.
    /// Pulls out the negative sign: `r * (-a - b)` -> `-(r * (a + b))`
    fn fmt_mul_with_sign_pullout(
        &self,
        f: &mut fmt::Formatter<'_>,
        coeff: ExprId,
        all_neg_add: ExprId,
    ) -> fmt::Result {
        use num_rational::BigRational;
        let zero = BigRational::from_integer(0.into());

        // Print leading minus
        write!(f, "-")?;

        // Check if coeff is a simple rational (for fraction display like -(a+b)/n)
        if let Expr::Number(n) = self.context.get(coeff) {
            if !n.is_integer() {
                // It's a fraction like 1/11 -> display as -(add)/denom
                let _numer = n.numer();
                let denom = n.denom();

                // Format the positive Add
                write!(f, "(")?;
                self.fmt_positive_add(f, all_neg_add, &zero)?;
                write!(f, ")/")?;

                // Just show denominator (numer already factored into display)
                return write!(f, "{}", denom);
            }
        }

        // General case: -(coeff * (positive_add))
        // Print coefficient
        let coeff_prec = precedence(self.context, coeff);
        if coeff_prec < 2 {
            write!(f, "(")?;
            self.fmt_internal(f, coeff)?;
            write!(f, ")")?;
        } else {
            self.fmt_internal(f, coeff)?;
        }

        write!(f, " * (")?;
        self.fmt_positive_add(f, all_neg_add, &zero)?;
        write!(f, ")")
    }

    /// Format an Add where all terms are negative, by printing their absolute values.
    /// `(-a) + (-b)` -> `a + b`
    fn fmt_positive_add(
        &self,
        f: &mut fmt::Formatter<'_>,
        id: ExprId,
        zero: &num_rational::BigRational,
    ) -> fmt::Result {
        // Collect terms
        let mut terms = Vec::new();
        self.collect_add_terms_for_pullout(id, &mut terms);

        for (i, term) in terms.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            // Print absolute value of term
            self.fmt_term_absolute(f, *term, zero)?;
        }
        Ok(())
    }

    fn collect_add_terms_for_pullout(&self, id: ExprId, terms: &mut Vec<ExprId>) {
        match self.context.get(id) {
            Expr::Add(l, r) => {
                self.collect_add_terms_for_pullout(*l, terms);
                self.collect_add_terms_for_pullout(*r, terms);
            }
            _ => terms.push(id),
        }
    }

    /// Format absolute value of a negative term.
    fn fmt_term_absolute(
        &self,
        f: &mut fmt::Formatter<'_>,
        id: ExprId,
        zero: &num_rational::BigRational,
    ) -> fmt::Result {
        match self.context.get(id) {
            // Neg(x) -> x
            Expr::Neg(inner) => self.fmt_internal(f, *inner),

            // Number(n < 0) -> |n|
            Expr::Number(n) if n < zero => {
                let abs_n = -n;
                write!(f, "{}", abs_n)
            }

            // Mul(Number(n < 0), rest) -> |n| * rest
            Expr::Mul(l, r) => {
                if let Expr::Number(n) = self.context.get(*l) {
                    if n < zero {
                        let abs_n = -n;
                        write!(f, "{} * ", abs_n)?;
                        return self.fmt_internal(f, *r);
                    }
                }
                // Not a negative-leading mul, print as-is
                self.fmt_internal(f, id)
            }

            // Everything else as-is
            _ => self.fmt_internal(f, id),
        }
    }
}

impl<'a> fmt::Display for DisplayExprWithHints<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_internal(f, self.id)
    }
}

// ============================================================================
// DisplayExprStyled: Global style-based formatting (replaces hints)
// ============================================================================

/// Display expression using global StylePreferences.
///
/// This is the recommended replacement for `DisplayExprWithHints`.
/// Instead of per-node hints, it uses a single set of preferences
/// that apply consistently to the entire output.
pub struct DisplayExprStyled<'a> {
    pub context: &'a Context,
    pub id: ExprId,
    pub style: &'a crate::root_style::StylePreferences,
}

impl<'a> DisplayExprStyled<'a> {
    /// Create a new styled display wrapper
    pub fn new(
        context: &'a Context,
        id: ExprId,
        style: &'a crate::root_style::StylePreferences,
    ) -> Self {
        Self { context, id, style }
    }

    fn fmt_internal(&self, f: &mut fmt::Formatter<'_>, id: ExprId) -> fmt::Result {
        let expr = self.context.get(id);
        let resolved_style = self.style.resolve();

        match expr {
            // Check for root-style Pow (should display as √ if style says so)
            Expr::Pow(base, exp) => {
                // Check if this is a fractional power that should be a root
                if resolved_style.root_style == crate::root_style::RootStyle::Radical {
                    if let Expr::Number(n) = self.context.get(*exp) {
                        // Check for 1/n form (nth root)
                        if !n.is_integer() && n.numer() == &1.into() {
                            if let Some(index) = n.denom().to_u32_digits().1.first() {
                                // Print as root
                                write!(f, "{}(", unicode_root_prefix(*index as u64))?;
                                self.fmt_internal(f, *base)?;
                                return write!(f, ")");
                            }
                        }
                        // Check for k/n form (kth power under nth root)
                        if !n.is_integer() {
                            let numer: i64 = n.numer().try_into().unwrap_or(1);
                            if let Some(denom) = n.denom().to_u32_digits().1.first() {
                                if *denom > 1 && numer != 1 {
                                    write!(f, "{}(", unicode_root_prefix(*denom as u64))?;
                                    self.fmt_internal(f, *base)?;
                                    write!(f, "^{})", numer)?;
                                    return Ok(());
                                }
                            }
                        }
                    }
                }
                // Default power display
                self.fmt_power(f, *base, *exp)
            }

            Expr::Number(n) => write!(f, "{}", n),
            Expr::Constant(c) => match c {
                Constant::Pi => write!(f, "pi"),
                Constant::E => write!(f, "e"),
                Constant::Infinity => write!(f, "infinity"),
                Constant::Undefined => write!(f, "undefined"),
                Constant::I => write!(f, "i"),
            },
            Expr::Variable(s) => write!(f, "{}", s),

            Expr::Add(_, _) => {
                // Collect terms and display with proper signs
                let mut terms = collect_add_terms(self.context, id);

                // Polynomial ordering: sort by degree descending (x^3, x^2, x, constant)
                // Applied FIRST since positive-first should take priority for display
                if resolved_style.polynomial_order {
                    terms.sort_by(|a, b| {
                        let deg_a = polynomial_degree(self.context, *a);
                        let deg_b = polynomial_degree(self.context, *b);
                        deg_b.cmp(&deg_a) // Descending
                    });
                }

                // Reorder for display: put positive terms first, negative terms last
                // This is applied LAST so it takes priority - makes -x + |x| display as |x| - x
                terms.sort_by(|a, b| {
                    let a_neg = check_negative(self.context, *a).0;
                    let b_neg = check_negative(self.context, *b).0;
                    match (a_neg, b_neg) {
                        (false, true) => std::cmp::Ordering::Less, // positive before negative
                        (true, false) => std::cmp::Ordering::Greater, // negative after positive
                        _ => std::cmp::Ordering::Equal,            // same sign: keep order
                    }
                });

                for (i, term) in terms.iter().enumerate() {
                    let (is_neg, _, _) = check_negative(self.context, *term);
                    if i == 0 {
                        self.fmt_internal(f, *term)?;
                    } else if is_neg {
                        write!(f, " - ")?;
                        self.fmt_term_abs(f, *term)?;
                    } else {
                        write!(f, " + ")?;
                        self.fmt_internal(f, *term)?;
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
                // PREFER_DIVISION: Convert 1/k * X → X/k (when enabled)
                if self.style.resolve().prefer_division {
                    // Check if left is Number(1/k) where k is integer > 1
                    if let Expr::Number(n) = self.context.get(*l) {
                        if !n.is_integer()
                            && *n.numer() == 1.into()
                            && n.denom() > &num_bigint::BigInt::from(1)
                        {
                            let denom = n.denom();
                            // Format as: X/k (with parens around X if needed)
                            let rhs_prec = precedence(self.context, *r);
                            if rhs_prec < 2 {
                                write!(f, "(")?;
                                self.fmt_internal(f, *r)?;
                                write!(f, ")")?;
                            } else {
                                self.fmt_internal(f, *r)?;
                            }
                            return write!(f, "/{}", denom);
                        }
                        // Also check for negative: -1/k * X → -X/k
                        if !n.is_integer()
                            && *n.numer() == (-1).into()
                            && n.denom() > &num_bigint::BigInt::from(1)
                        {
                            let denom = n.denom();
                            write!(f, "-")?;
                            let rhs_prec = precedence(self.context, *r);
                            if rhs_prec < 2 {
                                write!(f, "(")?;
                                self.fmt_internal(f, *r)?;
                                write!(f, ")")?;
                            } else {
                                self.fmt_internal(f, *r)?;
                            }
                            return write!(f, "/{}", denom);
                        }
                    }
                    // Check RHS for 1/k (in case of X * 1/k due to ordering)
                    if let Expr::Number(n) = self.context.get(*r) {
                        if !n.is_integer()
                            && *n.numer() == 1.into()
                            && n.denom() > &num_bigint::BigInt::from(1)
                        {
                            let denom = n.denom();
                            let lhs_prec = precedence(self.context, *l);
                            if lhs_prec < 2 {
                                write!(f, "(")?;
                                self.fmt_internal(f, *l)?;
                                write!(f, ")")?;
                            } else {
                                self.fmt_internal(f, *l)?;
                            }
                            return write!(f, "/{}", denom);
                        }
                    }
                }

                // Sign pull-out for all-negative Adds
                if crate::views::has_all_negative_terms(self.context, *r) {
                    write!(f, "-")?;
                    return self.fmt_mul_positive(f, *l, *r);
                }
                if crate::views::has_all_negative_terms(self.context, *l) {
                    write!(f, "-")?;
                    return self.fmt_mul_positive(f, *r, *l);
                }

                // Standard multiplication
                let lhs_prec = precedence(self.context, *l);
                let rhs_prec = precedence(self.context, *r);
                if lhs_prec < 2 {
                    write!(f, "(")?;
                    self.fmt_internal(f, *l)?;
                    write!(f, ")")?;
                } else {
                    self.fmt_internal(f, *l)?;
                }
                write!(f, "{}", mul_symbol())?;
                if rhs_prec < 2 {
                    write!(f, "(")?;
                    self.fmt_internal(f, *r)?;
                    write!(f, ")")
                } else {
                    self.fmt_internal(f, *r)
                }
            }

            Expr::Div(l, r) => {
                let lhs_prec = precedence(self.context, *l);
                let rhs_prec = precedence(self.context, *r);
                if lhs_prec < 2 {
                    write!(f, "(")?;
                    self.fmt_internal(f, *l)?;
                    write!(f, ")")?;
                } else {
                    self.fmt_internal(f, *l)?;
                }
                write!(f, " / ")?;
                if rhs_prec <= 2 {
                    write!(f, "(")?;
                    self.fmt_internal(f, *r)?;
                    write!(f, ")")
                } else {
                    self.fmt_internal(f, *r)
                }
            }

            Expr::Neg(inner) => {
                write!(f, "-")?;
                let prec = precedence(self.context, *inner);
                if prec < 3 {
                    write!(f, "(")?;
                    self.fmt_internal(f, *inner)?;
                    write!(f, ")")
                } else {
                    self.fmt_internal(f, *inner)
                }
            }

            Expr::Function(name, args) => {
                // Special case: abs(x) displays as |x|
                if name == "abs" && args.len() == 1 {
                    write!(f, "|")?;
                    self.fmt_internal(f, args[0])?;
                    return write!(f, "|");
                }

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
            Expr::SessionRef(id) => write!(f, "#{}", id),
        }
    }

    fn fmt_power(&self, f: &mut fmt::Formatter<'_>, base: ExprId, exp: ExprId) -> fmt::Result {
        // Skip ^1
        if let Expr::Number(n) = self.context.get(exp) {
            if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                return self.fmt_internal(f, base);
            }
        }

        // Format base with parens if needed
        let base_prec = precedence(self.context, base);
        if base_prec < 3 {
            write!(f, "(")?;
            self.fmt_internal(f, base)?;
            write!(f, ")")?;
        } else {
            self.fmt_internal(f, base)?;
        }

        // Check if exponent is a small positive integer (use superscript in pretty mode)
        if is_pretty_output() {
            if let Expr::Number(n) = self.context.get(exp) {
                if n.is_integer() {
                    if let Some(exp_i64) = n.to_integer().try_into().ok() as Option<i64> {
                        // Use superscript for positive integers 0-99
                        if (0..=99).contains(&exp_i64) {
                            return write!(f, "{}", number_to_superscript(exp_i64 as u64));
                        }
                    }
                }
            }
        }

        // Default: ^(exp) format
        write!(f, "^(")?;
        self.fmt_internal(f, exp)?;
        write!(f, ")")
    }

    fn fmt_term_abs(&self, f: &mut fmt::Formatter<'_>, id: ExprId) -> fmt::Result {
        match self.context.get(id) {
            Expr::Neg(inner) => self.fmt_internal(f, *inner),
            Expr::Number(n) => {
                let zero = num_rational::BigRational::from_integer(0.into());
                if n < &zero {
                    write!(f, "{}", -n)
                } else {
                    write!(f, "{}", n)
                }
            }
            Expr::Mul(l, r) => {
                if let Expr::Number(n) = self.context.get(*l) {
                    let zero = num_rational::BigRational::from_integer(0.into());
                    if n < &zero {
                        write!(f, "{} * ", -n)?;
                        return self.fmt_internal(f, *r);
                    }
                }
                self.fmt_internal(f, id)
            }
            _ => self.fmt_internal(f, id),
        }
    }

    fn fmt_mul_positive(
        &self,
        f: &mut fmt::Formatter<'_>,
        coeff: ExprId,
        neg_add: ExprId,
    ) -> fmt::Result {
        // If coeff is a rational, display as -(add)/denom
        if let Expr::Number(n) = self.context.get(coeff) {
            if !n.is_integer() {
                let denom = n.denom();
                write!(f, "(")?;
                self.fmt_positive_sum(f, neg_add)?;
                return write!(f, ")/{}", denom);
            }
        }
        // General: coeff * (positive_sum)
        self.fmt_internal(f, coeff)?;
        write!(f, " * (")?;
        self.fmt_positive_sum(f, neg_add)?;
        write!(f, ")")
    }

    fn fmt_positive_sum(&self, f: &mut fmt::Formatter<'_>, id: ExprId) -> fmt::Result {
        let mut terms = Vec::new();
        self.collect_terms(id, &mut terms);
        for (i, term) in terms.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            self.fmt_term_abs(f, *term)?;
        }
        Ok(())
    }

    fn collect_terms(&self, id: ExprId, terms: &mut Vec<ExprId>) {
        match self.context.get(id) {
            Expr::Add(l, r) => {
                self.collect_terms(*l, terms);
                self.collect_terms(*r, terms);
            }
            _ => terms.push(id),
        }
    }
}

impl<'a> fmt::Display for DisplayExprStyled<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_internal(f, self.id)
    }
}

#[cfg(test)]
mod hold_tests {
    use super::*;
    use crate::{Context, Expr};

    #[test]
    fn test_hold_transparency() {
        let mut ctx = Context::new();
        let x = ctx.add(Expr::Variable("x".to_string()));
        let held = ctx.add(Expr::Function("__hold".to_string(), vec![x]));
        let display = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: held
            }
        );
        assert_eq!(display, "x", "Expected 'x' but got '{}'", display);
    }
}
