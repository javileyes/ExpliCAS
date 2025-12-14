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
            write!(f, " * ")?;
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
                                write!(f, " * ")?;
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
                                write!(f, " * ")?;
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
                let numer = n.numer();
                let denom = n.denom();

                // Format the positive Add
                write!(f, "(")?;
                self.fmt_positive_add(f, all_neg_add, &zero)?;
                write!(f, ")/")?;

                // If numerator is 1, just show denominator
                if *numer == 1.into() {
                    return write!(f, "{}", denom);
                } else {
                    return write!(f, "{}", denom);
                }
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
