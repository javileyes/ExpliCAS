//! Core display implementations: `DisplayExpr` and `RawDisplayExpr`.

use crate::{Constant, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::Zero;
use std::fmt;

use super::mul_symbol;
use super::ordering::{cmp_term_for_display, format_factors, FractionDisplayView};

// ============================================================================
// DisplayExpr
// ============================================================================

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
                Constant::Phi => write!(f, "phi"),
            },
            Expr::Variable(sym_id) => write!(f, "{}", self.context.sym_name(*sym_id)),
            Expr::Add(_, _) => {
                // Flatten Add/Sub chain to handle mixed signs gracefully
                let mut terms = collect_signed_add_terms(self.context, self.id);

                // Unified canonical ordering: polynomial degree descending + compare_expr fallback
                terms.sort_by(|a, b| cmp_term_for_display(self.context, a.id, b.id));

                for (i, term) in terms.iter().enumerate() {
                    let raw_is_neg = check_negative(self.context, term.id).0;
                    let is_neg = term.invert_sign ^ raw_is_neg;

                    if i == 0 {
                        if is_neg {
                            write!(f, "-")?;
                            format_term_absolute(f, self.context, term.id)?;
                        } else if term.invert_sign && raw_is_neg {
                            format_term_absolute(f, self.context, term.id)?;
                        } else {
                            write!(
                                f,
                                "{}",
                                DisplayExpr {
                                    context: self.context,
                                    id: term.id
                                }
                            )?;
                        }
                    } else if is_neg {
                        write!(f, " - ")?;
                        format_term_absolute(f, self.context, term.id)?;
                    } else {
                        write!(
                            f,
                            " + {}",
                            DisplayExpr {
                                context: self.context,
                                id: term.id
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
                let rhs_is_add_sub = is_add_sub_after_internal_hold(self.context, *r);

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

                let left_neg = direct_negative_factor(self.context, *l);
                let right_neg = direct_negative_factor(self.context, *r);
                match (left_neg, right_neg) {
                    (Some((inner, coeff)), None) => {
                        write!(f, "-")?;
                        format_mul_absolute(
                            f,
                            self.context,
                            RenderFactor::Direct(inner, coeff),
                            RenderFactor::Expr(*r),
                        )?;
                        return Ok(());
                    }
                    (None, Some((inner, coeff))) => {
                        write!(f, "-")?;
                        format_mul_absolute(
                            f,
                            self.context,
                            RenderFactor::Expr(*l),
                            RenderFactor::Direct(inner, coeff),
                        )?;
                        return Ok(());
                    }
                    _ => {}
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

                // Check if base is "negative" (Neg expr, negative number, or leading negative factor)
                // These need parentheses: (-1)² not -1², (-x)² not -x²
                let (base_is_negative, _, _) = check_negative(self.context, *b);

                if base_prec < op_prec || base_is_negative {
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
            Expr::Function(fn_id, args) => {
                let name = self.context.sym_name(*fn_id);
                // ONLY internal __hold barrier is transparent for display
                // User-facing hold(...) should be displayed explicitly
                if crate::hold::is_internal_hold_name(name) && args.len() == 1 {
                    return write!(
                        f,
                        "{}",
                        DisplayExpr {
                            context: self.context,
                            id: args[0]
                        }
                    );
                }
                // __eq__ is an internal equation representation - display as "lhs = rhs"
                if crate::eq::is_eq_name(name) && args.len() == 2 {
                    return write!(
                        f,
                        "{} = {}",
                        DisplayExpr {
                            context: self.context,
                            id: args[0]
                        },
                        DisplayExpr {
                            context: self.context,
                            id: args[1]
                        }
                    );
                }
                if (name == "fact" || name == "factorial") && args.len() == 1 {
                    let needs_parens = matches!(
                        self.context.get(args[0]),
                        Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Mul(_, _) | Expr::Div(_, _)
                    );
                    if needs_parens {
                        write!(
                            f,
                            "({})!",
                            DisplayExpr {
                                context: self.context,
                                id: args[0]
                            }
                        )
                    } else {
                        write!(
                            f,
                            "{}!",
                            DisplayExpr {
                                context: self.context,
                                id: args[0]
                            }
                        )
                    }
                } else if name == "abs" && args.len() == 1 {
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
            // Hold is transparent for display - render inner directly
            Expr::Hold(inner) => write!(
                f,
                "{}",
                DisplayExpr {
                    context: self.context,
                    id: *inner
                }
            ),
        }
    }
}

// ============================================================================
// Shared helpers
// ============================================================================

pub(super) fn precedence(ctx: &Context, id: ExprId) -> i32 {
    let id = unwrap_internal_hold_for_display(ctx, id);
    match ctx.get(id) {
        Expr::Add(_, _) | Expr::Sub(_, _) => 1,
        Expr::Mul(_, _) | Expr::Div(_, _) => 2,
        Expr::Pow(_, _) => 3,
        Expr::Neg(_) | Expr::Hold(_) => 4, // Unary wrappers
        Expr::Function(_, _)
        | Expr::Variable(_)
        | Expr::Number(_)
        | Expr::Constant(_)
        | Expr::Matrix { .. }
        | Expr::SessionRef(_) => 5,
    }
}

pub(super) fn is_add_sub_after_internal_hold(ctx: &Context, id: ExprId) -> bool {
    let id = unwrap_internal_hold_for_display(ctx, id);
    matches!(ctx.get(id), Expr::Add(_, _) | Expr::Sub(_, _))
}

fn unwrap_internal_hold_for_display(ctx: &Context, id: ExprId) -> ExprId {
    let mut current = id;
    loop {
        match ctx.get(current) {
            Expr::Hold(inner) => current = *inner,
            Expr::Function(fn_id, args)
                if args.len() == 1 && crate::hold::is_internal_hold_name(ctx.sym_name(*fn_id)) =>
            {
                current = args[0];
            }
            _ => return current,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct SignedAddTerm {
    pub id: ExprId,
    pub invert_sign: bool,
}

pub(super) fn collect_signed_add_terms(ctx: &Context, id: ExprId) -> Vec<SignedAddTerm> {
    let mut terms = Vec::new();
    collect_signed_add_terms_recursive(ctx, id, false, &mut terms);
    terms
}

fn collect_signed_add_terms_recursive(
    ctx: &Context,
    id: ExprId,
    invert_sign: bool,
    terms: &mut Vec<SignedAddTerm>,
) {
    match ctx.get(id) {
        Expr::Add(l, r) => {
            collect_signed_add_terms_recursive(ctx, *l, invert_sign, terms);
            collect_signed_add_terms_recursive(ctx, *r, invert_sign, terms);
        }
        Expr::Sub(l, r) => {
            collect_signed_add_terms_recursive(ctx, *l, invert_sign, terms);
            collect_signed_add_terms_recursive(ctx, *r, !invert_sign, terms);
        }
        _ => terms.push(SignedAddTerm { id, invert_sign }),
    }
}

pub(super) fn check_negative(
    ctx: &Context,
    id: ExprId,
) -> (bool, Option<ExprId>, Option<BigRational>) {
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
            let left_neg = direct_negative_factor(ctx, *a);
            let right_neg = direct_negative_factor(ctx, *b);
            match (left_neg, right_neg) {
                (Some((_inner, coeff)), None) | (None, Some((_inner, coeff))) => {
                    (true, None, coeff)
                }
                _ => (false, None, None),
            }
        }
        _ => (false, None, None),
    }
}

pub(super) fn direct_negative_factor(
    ctx: &Context,
    id: ExprId,
) -> Option<(ExprId, Option<BigRational>)> {
    match ctx.get(id) {
        Expr::Neg(inner) => Some((*inner, None)),
        Expr::Number(n) if *n < num_rational::BigRational::zero() => Some((id, Some((-n).clone()))),
        _ => None,
    }
}

pub(super) fn format_term_absolute(
    f: &mut fmt::Formatter<'_>,
    ctx: &Context,
    id: ExprId,
) -> fmt::Result {
    match ctx.get(id) {
        Expr::Neg(inner) => {
            let inner_is_add_sub = is_add_sub_after_internal_hold(ctx, *inner);
            if inner_is_add_sub {
                write!(
                    f,
                    "({})",
                    DisplayExpr {
                        context: ctx,
                        id: *inner
                    }
                )
            } else {
                write!(
                    f,
                    "{}",
                    DisplayExpr {
                        context: ctx,
                        id: *inner
                    }
                )
            }
        }
        Expr::Number(n) if *n < num_rational::BigRational::zero() => write!(f, "{}", -n),
        Expr::Mul(a, b) => {
            let left_neg = direct_negative_factor(ctx, *a);
            let right_neg = direct_negative_factor(ctx, *b);
            match (left_neg, right_neg) {
                (Some((inner, coeff)), None) => format_mul_absolute(
                    f,
                    ctx,
                    RenderFactor::Direct(inner, coeff),
                    RenderFactor::Expr(*b),
                ),
                (None, Some((inner, coeff))) => format_mul_absolute(
                    f,
                    ctx,
                    RenderFactor::Expr(*a),
                    RenderFactor::Direct(inner, coeff),
                ),
                _ => write!(f, "{}", DisplayExpr { context: ctx, id }),
            }
        }
        _ => write!(f, "{}", DisplayExpr { context: ctx, id }),
    }
}

enum RenderFactor {
    Expr(ExprId),
    Direct(ExprId, Option<BigRational>),
}

fn format_mul_absolute(
    f: &mut fmt::Formatter<'_>,
    ctx: &Context,
    left: RenderFactor,
    right: RenderFactor,
) -> fmt::Result {
    format_abs_factor(f, ctx, left)?;
    write!(f, "{}", mul_symbol())?;
    format_abs_factor(f, ctx, right)
}

fn format_abs_factor(
    f: &mut fmt::Formatter<'_>,
    ctx: &Context,
    factor: RenderFactor,
) -> fmt::Result {
    match factor {
        RenderFactor::Direct(_inner, Some(coeff)) => write!(f, "{}", coeff),
        RenderFactor::Direct(inner, None) | RenderFactor::Expr(inner) => {
            let prec = precedence(ctx, inner);
            if prec < 2 {
                write!(
                    f,
                    "({})",
                    DisplayExpr {
                        context: ctx,
                        id: inner
                    }
                )
            } else {
                write!(
                    f,
                    "{}",
                    DisplayExpr {
                        context: ctx,
                        id: inner
                    }
                )
            }
        }
    }
}

// ============================================================================
// RawDisplayExpr
// ============================================================================

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
                Constant::Phi => write!(f, "phi"),
            },
            Expr::Variable(sym_id) => write!(f, "{}", self.context.sym_name(*sym_id)),
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
            // Hold is transparent for display - render inner directly
            Expr::Hold(inner) => write!(
                f,
                "{}",
                RawDisplayExpr {
                    context: self.context,
                    id: *inner
                }
            ),
        }
    }
}
