//! Core display implementations: `DisplayExpr` and `RawDisplayExpr`.

use crate::{Constant, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::fmt;

use super::mul_symbol;
use super::ordering::{cmp_term_for_display, format_factors, DisplayFactor, FractionDisplayView};

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
                if format_trig_polynomial_product(f, self.context, *l, *r)? {
                    return Ok(());
                }

                // P3: Try to display as fraction using FractionDisplayView
                if let Some(frac) = FractionDisplayView::from(self.context, self.id) {
                    // Handle sign
                    if frac.sign < 0 {
                        write!(f, "-")?;
                    }

                    // Format numerator
                    let needs_num_parens = frac.sign >= 0
                        && (frac.num.len() > 1 || frac.num.iter().any(|f| f.exp != 1));
                    if frac.num.is_empty() {
                        write!(f, "1")?;
                    } else if needs_num_parens && frac.num.len() > 1 {
                        write!(f, "(")?;
                        format_fraction_numerator(f, self.context, &frac.num, frac.sign < 0)?;
                        write!(f, ")")?;
                    } else {
                        format_fraction_numerator(f, self.context, &frac.num, frac.sign < 0)?;
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
                            if frac.den[0].exp == 1 && den_base_prec < 2 {
                                write!(
                                    f,
                                    "{}",
                                    DisplayExpr {
                                        context: self.context,
                                        id: frac.den[0].base
                                    }
                                )?;
                            } else {
                                format_factors(f, self.context, &frac.den)?;
                            }
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

                if is_positive_one_factor(self.context, *l) {
                    return write!(
                        f,
                        "{}",
                        DisplayExpr {
                            context: self.context,
                            id: *r
                        }
                    );
                }
                if is_positive_one_factor(self.context, *r) {
                    return write!(
                        f,
                        "{}",
                        DisplayExpr {
                            context: self.context,
                            id: *l
                        }
                    );
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
                let (lhs_is_neg, lhs_neg_inner, _) = check_negative(self.context, *l);
                let (rhs_is_neg, _, _) = check_negative(self.context, *r);

                if !lhs_is_neg && !rhs_is_neg {
                    if let Some((coefficient, radicand)) =
                        reciprocal_sqrt_numerator_for_display(self.context, *l)
                    {
                        return format_reciprocal_sqrt_div_for_display(
                            f,
                            self.context,
                            coefficient,
                            radicand,
                            *r,
                        );
                    }
                }

                if lhs_is_neg ^ rhs_is_neg {
                    write!(f, "-")?;
                }

                if lhs_prec < op_prec
                    || (lhs_is_neg && lhs_neg_inner.is_some() && lhs_prec <= op_prec)
                {
                    write!(f, "(")?;
                    if lhs_is_neg {
                        format_term_absolute(f, self.context, *l)?;
                    } else {
                        write!(
                            f,
                            "{}",
                            DisplayExpr {
                                context: self.context,
                                id: *l
                            }
                        )?;
                    }
                    write!(f, ")")?;
                } else if lhs_is_neg {
                    format_term_absolute(f, self.context, *l)?
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
                let rhs_product_display = (!rhs_is_neg)
                    .then(|| denominator_product_numeric_first_for_display(self.context, *r))
                    .flatten();

                if rhs_prec <= op_prec {
                    write!(f, "(")?;
                    if let Some(display) = rhs_product_display {
                        write!(f, "{display}")?;
                    } else if rhs_is_neg {
                        format_term_absolute(f, self.context, *r)?;
                    } else {
                        write!(
                            f,
                            "{}",
                            DisplayExpr {
                                context: self.context,
                                id: *r
                            }
                        )?;
                    }
                    write!(f, ")")
                } else if rhs_is_neg {
                    format_term_absolute(f, self.context, *r)
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

                if is_internal_hold_wrapped(self.context, *b)
                    && is_one_half_exponent(self.context, *e)
                {
                    return write!(
                        f,
                        "sqrt({})",
                        DisplayExpr {
                            context: self.context,
                            id: *b
                        }
                    );
                }

                // If base is 1, just display "1" (1^n = 1)
                if let Expr::Number(n) = self.context.get(*b) {
                    if n.is_integer() && *n == num_rational::BigRational::from_integer(1.into()) {
                        return write!(f, "1");
                    }
                    if n.is_positive() && is_one_half_exponent(self.context, *e) {
                        return write!(f, "sqrt({n})");
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
                    if let Some(exp) =
                        format_unit_fraction_scaled_expression_for_display(self.context, *e)
                    {
                        return write!(f, "({exp})");
                    }
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
                if let Expr::Div(l, r) = self.context.get(*e) {
                    let (lhs_is_neg, _, _) = check_negative(self.context, *l);
                    let (rhs_is_neg, _, _) = check_negative(self.context, *r);
                    if !(lhs_is_neg ^ rhs_is_neg) {
                        write!(f, "-")?;
                        return write!(
                            f,
                            "{}",
                            DisplayExpr {
                                context: self.context,
                                id: *e
                            }
                        );
                    }
                }

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
                } else if name == "integrate" && args.len() == 1 {
                    write!(
                        f,
                        "int {} , dx",
                        DisplayExpr {
                            context: self.context,
                            id: args[0]
                        }
                    )
                } else if args.len() == 1 && prefers_quotient_sum_function_arg(name) {
                    let arg = format_preferred_function_arg_for_display(self.context, args[0])
                        .unwrap_or_else(|| {
                            format!(
                                "{}",
                                DisplayExpr {
                                    context: self.context,
                                    id: args[0]
                                }
                            )
                        });
                    write!(f, "{}({})", name, arg)
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

fn is_internal_hold_wrapped(ctx: &Context, id: ExprId) -> bool {
    matches!(ctx.get(id), Expr::Hold(_))
        || matches!(
            ctx.get(id),
            Expr::Function(fn_id, args)
                if args.len() == 1 && crate::hold::is_internal_hold_name(ctx.sym_name(*fn_id))
        )
}

fn is_one_half_exponent(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(n) => *n == BigRational::new(1.into(), 2.into()),
        Expr::Div(num, den) => {
            matches!(ctx.get(*num), Expr::Number(n) if n.is_one())
                && matches!(ctx.get(*den), Expr::Number(n) if *n == BigRational::from_integer(2.into()))
        }
        _ => false,
    }
}

fn prefers_quotient_sum_function_arg(name: &str) -> bool {
    matches!(
        name,
        "sin"
            | "cos"
            | "tan"
            | "sec"
            | "csc"
            | "cot"
            | "sinh"
            | "cosh"
            | "tanh"
            | "exp"
            | "arcsin"
            | "asin"
            | "arccos"
            | "acos"
            | "arctan"
            | "atan"
            | "asinh"
            | "acosh"
            | "atanh"
    )
}

fn format_preferred_function_arg_for_display(ctx: &Context, id: ExprId) -> Option<String> {
    format_unit_fraction_scaled_expression_for_display(ctx, id)
        .or_else(|| format_scaled_half_power_function_arg_for_display(ctx, id))
        .or_else(|| format_half_power_function_arg_for_display(ctx, id))
}

fn format_scaled_half_power_function_arg_for_display(ctx: &Context, id: ExprId) -> Option<String> {
    let id = unwrap_internal_hold_for_display(ctx, id);
    let Expr::Mul(left, right) = ctx.get(id) else {
        return None;
    };

    let (scale, half_power) = match (
        positive_non_unit_number_factor(ctx, *left),
        half_power_base(ctx, *right),
    ) {
        (Some(scale), Some(base)) => (scale, base),
        _ => match (
            positive_non_unit_number_factor(ctx, *right),
            half_power_base(ctx, *left),
        ) {
            (Some(scale), Some(base)) => (scale, base),
            _ => return None,
        },
    };

    let base = format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: half_power
        }
    );
    Some(format!("{scale}{}sqrt({base})", mul_symbol()))
}

fn format_half_power_function_arg_for_display(ctx: &Context, id: ExprId) -> Option<String> {
    let id = unwrap_internal_hold_for_display(ctx, id);
    let base = half_power_base(ctx, id)?;
    let base = format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: base
        }
    );
    Some(format!("sqrt({base})"))
}

fn half_power_base(ctx: &Context, id: ExprId) -> Option<ExprId> {
    let id = unwrap_internal_hold_for_display(ctx, id);
    let Expr::Pow(base, exp) = ctx.get(id) else {
        return None;
    };
    if !is_one_half_exponent(ctx, *exp) {
        return None;
    }
    Some(*base)
}

fn positive_non_unit_number_factor(ctx: &Context, id: ExprId) -> Option<String> {
    let id = unwrap_internal_hold_for_display(ctx, id);
    let Expr::Number(n) = ctx.get(id) else {
        return None;
    };
    (n.is_positive() && !n.is_one()).then(|| n.to_string())
}

fn reciprocal_sqrt_numerator_for_display(
    ctx: &Context,
    id: ExprId,
) -> Option<(BigRational, ExprId)> {
    let mut factors = Vec::new();
    collect_mul_factors_for_display(ctx, id, &mut factors);

    let mut coefficient = BigRational::one();
    let mut radicand = None;

    for factor in factors {
        match ctx.get(factor) {
            Expr::Number(n) if n.is_positive() => coefficient *= n.clone(),
            Expr::Pow(base, exp) if is_negative_one_half_exponent(ctx, *exp) => {
                if radicand.replace(*base).is_some() {
                    return None;
                }
            }
            _ => return None,
        }
    }

    radicand.map(|radicand| (coefficient, radicand))
}

fn collect_mul_factors_for_display(ctx: &Context, id: ExprId, out: &mut Vec<ExprId>) {
    let id = unwrap_internal_hold_for_display(ctx, id);
    match ctx.get(id) {
        Expr::Mul(l, r) => {
            collect_mul_factors_for_display(ctx, *l, out);
            collect_mul_factors_for_display(ctx, *r, out);
        }
        _ => out.push(id),
    }
}

fn denominator_product_numeric_first_for_display(ctx: &Context, id: ExprId) -> Option<String> {
    let mut factors = Vec::new();
    collect_mul_factors_for_display(ctx, id, &mut factors);
    let reordered = numeric_first_product_factors(ctx, &factors)?;
    Some(
        reordered
            .iter()
            .map(|factor| display_factor_in_product(ctx, *factor))
            .collect::<Vec<_>>()
            .join(mul_symbol()),
    )
}

fn numeric_first_product_factors(ctx: &Context, factors: &[ExprId]) -> Option<Vec<ExprId>> {
    if factors.len() < 2 {
        return None;
    }

    let mut numeric = Vec::new();
    let mut rest = Vec::new();
    for factor in factors {
        if matches!(ctx.get(*factor), Expr::Number(n) if n.is_positive()) {
            numeric.push(*factor);
        } else {
            rest.push(*factor);
        }
    }

    if numeric.is_empty() || rest.is_empty() {
        return None;
    }

    let reordered = numeric.into_iter().chain(rest).collect::<Vec<_>>();
    (reordered != factors).then_some(reordered)
}

fn display_factor_in_product(ctx: &Context, id: ExprId) -> String {
    let rendered = format!("{}", DisplayExpr { context: ctx, id });
    if precedence(ctx, id) < 2 {
        format!("({rendered})")
    } else {
        rendered
    }
}

fn is_negative_one_half_exponent(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(n) => *n == BigRational::new((-1).into(), 2.into()),
        Expr::Div(num, den) => {
            matches!(ctx.get(*num), Expr::Number(n) if *n == BigRational::from_integer((-1).into()))
                && matches!(ctx.get(*den), Expr::Number(n) if *n == BigRational::from_integer(2.into()))
        }
        _ => false,
    }
}

fn format_reciprocal_sqrt_div_for_display(
    f: &mut fmt::Formatter<'_>,
    ctx: &Context,
    coefficient: BigRational,
    radicand: ExprId,
    denominator: ExprId,
) -> fmt::Result {
    write!(f, "{}", coefficient.numer())?;
    write!(f, " / (")?;

    let sqrt_display = format!(
        "sqrt({})",
        DisplayExpr {
            context: ctx,
            id: radicand
        }
    );

    let mut denominator_parts = Vec::new();
    if !coefficient.denom().is_one() {
        denominator_parts.push(coefficient.denom().to_string());
    }

    let mut denominator_factors = Vec::new();
    collect_mul_factors_for_display(ctx, denominator, &mut denominator_factors);
    let mut rest_parts = Vec::new();
    let mut only_sin_or_cos_sqrt_rest = true;
    for factor in denominator_factors {
        let rendered = display_factor_in_product(ctx, factor);
        if matches!(ctx.get(factor), Expr::Number(n) if n.is_positive()) {
            denominator_parts.push(rendered);
        } else {
            only_sin_or_cos_sqrt_rest &= is_sin_or_cos_of_display_sqrt_factor(ctx, factor);
            rest_parts.push(rendered);
        }
    }
    let sqrt_before_rest = rest_parts.len() == 1 && only_sin_or_cos_sqrt_rest;
    if sqrt_before_rest {
        denominator_parts.push(sqrt_display);
        denominator_parts.extend(rest_parts);
    } else {
        denominator_parts.extend(rest_parts);
        denominator_parts.push(sqrt_display);
    }
    write!(f, "{}", denominator_parts.join(mul_symbol()))?;
    write!(f, ")")
}

fn is_sin_or_cos_of_display_sqrt_factor(ctx: &Context, id: ExprId) -> bool {
    let id = unwrap_internal_hold_for_display(ctx, id);
    let Expr::Function(fn_id, args) = ctx.get(id) else {
        return false;
    };
    if args.len() != 1 || !matches!(ctx.sym_name(*fn_id), "sin" | "cos") {
        return false;
    }
    let arg = unwrap_internal_hold_for_display(ctx, args[0]);
    match ctx.get(arg) {
        Expr::Function(inner_fn, inner_args) => {
            inner_args.len() == 1 && ctx.sym_name(*inner_fn) == "sqrt"
        }
        Expr::Pow(_, exp) => is_one_half_exponent(ctx, *exp),
        _ => false,
    }
}

fn format_unit_fraction_scaled_expression_for_display(ctx: &Context, id: ExprId) -> Option<String> {
    let id = unwrap_internal_hold_for_display(ctx, id);
    let (numerator, denominator) = match ctx.get(id) {
        Expr::Mul(left, right) => {
            if let Some(denominator) = unit_fraction_denominator_for_display(ctx, *left) {
                if displayable_quotient_numerator(ctx, *right) {
                    return render_expression_over_denominator_for_display(
                        ctx,
                        *right,
                        denominator,
                    );
                }
            }
            if let Some(denominator) = unit_fraction_denominator_for_display(ctx, *right) {
                if displayable_quotient_numerator(ctx, *left) {
                    return render_expression_over_denominator_for_display(ctx, *left, denominator);
                }
            }

            let (scale, numerator) = match (ctx.get(*left), ctx.get(*right)) {
                (Expr::Number(scale), _) if displayable_quotient_numerator(ctx, *right) => {
                    (scale, *right)
                }
                (_, Expr::Number(scale)) if displayable_quotient_numerator(ctx, *left) => {
                    (scale, *left)
                }
                _ => return None,
            };
            if !scale.is_positive() || !scale.numer().is_one() || scale.denom().is_one() {
                return None;
            }
            (numerator, scale.denom().to_string())
        }
        Expr::Div(num, den) => {
            let Expr::Number(denominator) = ctx.get(*den) else {
                return None;
            };
            if !denominator.is_positive()
                || !denominator.is_integer()
                || denominator.is_one()
                || !denominator.denom().is_one()
            {
                return None;
            }
            let numerator = unit_mul_numerator_for_display(ctx, *num)?;
            (numerator, denominator.numer().to_string())
        }
        _ => return None,
    };

    let numerator = unwrap_internal_hold_for_display(ctx, numerator);
    render_expression_over_denominator_for_display(ctx, numerator, denominator)
}

fn unit_fraction_denominator_for_display(ctx: &Context, id: ExprId) -> Option<String> {
    let id = unwrap_internal_hold_for_display(ctx, id);
    match ctx.get(id) {
        Expr::Number(scale)
            if scale.is_positive() && scale.numer().is_one() && !scale.denom().is_one() =>
        {
            Some(scale.denom().to_string())
        }
        Expr::Div(num, den) => {
            let Expr::Number(numerator) = ctx.get(*num) else {
                return None;
            };
            let Expr::Number(denominator) = ctx.get(*den) else {
                return None;
            };
            if numerator.is_one()
                && denominator.is_positive()
                && denominator.is_integer()
                && !denominator.is_one()
                && denominator.denom().is_one()
            {
                Some(denominator.numer().to_string())
            } else {
                None
            }
        }
        _ => None,
    }
}

fn render_expression_over_denominator_for_display(
    ctx: &Context,
    numerator: ExprId,
    denominator: String,
) -> Option<String> {
    let numerator = unwrap_internal_hold_for_display(ctx, numerator);
    if !displayable_quotient_numerator(ctx, numerator) {
        return None;
    }
    let numerator_display = format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: numerator
        }
    );
    if quotient_numerator_needs_parens(ctx, numerator) {
        Some(format!("({numerator_display}) / {denominator}"))
    } else {
        Some(format!("{numerator_display} / {denominator}"))
    }
}

fn unit_mul_numerator_for_display(ctx: &Context, id: ExprId) -> Option<ExprId> {
    let id = unwrap_internal_hold_for_display(ctx, id);
    if displayable_quotient_numerator(ctx, id) {
        return Some(id);
    }

    let Expr::Mul(left, right) = ctx.get(id) else {
        return None;
    };
    match (ctx.get(*left), ctx.get(*right)) {
        (Expr::Number(n), _) if n.is_one() => {
            displayable_quotient_numerator(ctx, *right).then_some(*right)
        }
        (_, Expr::Number(n)) if n.is_one() => {
            displayable_quotient_numerator(ctx, *left).then_some(*left)
        }
        _ => None,
    }
}

fn displayable_quotient_numerator(ctx: &Context, id: ExprId) -> bool {
    let id = unwrap_internal_hold_for_display(ctx, id);
    !matches!(ctx.get(id), Expr::Number(_))
}

fn quotient_numerator_needs_parens(ctx: &Context, id: ExprId) -> bool {
    let id = unwrap_internal_hold_for_display(ctx, id);
    is_add_sub_after_internal_hold(ctx, id) || matches!(ctx.get(id), Expr::Div(_, _))
}

fn format_trig_polynomial_product(
    f: &mut fmt::Formatter<'_>,
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> Result<bool, fmt::Error> {
    let Some((poly, trig)) = trig_polynomial_product_factors(ctx, left, right) else {
        return Ok(false);
    };
    let mut terms = collect_signed_add_terms(ctx, poly);
    if !should_format_add_degree_first(ctx, &terms) {
        return Ok(false);
    }

    terms.sort_by(|a, b| compare_add_terms_degree_first(ctx, a, b));
    write!(f, "(")?;
    format_signed_add_terms_in_order(f, ctx, &terms)?;
    write!(
        f,
        "){}{}",
        mul_symbol(),
        DisplayExpr {
            context: ctx,
            id: trig
        }
    )?;
    Ok(true)
}

fn trig_polynomial_product_factors(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, ExprId)> {
    if is_add_sub_after_internal_hold(ctx, left) && is_display_trig_factor(ctx, right) {
        return Some((left, right));
    }
    if is_display_trig_factor(ctx, left) && is_add_sub_after_internal_hold(ctx, right) {
        return Some((right, left));
    }
    None
}

fn is_display_trig_factor(ctx: &Context, id: ExprId) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(id) else {
        return false;
    };
    args.len() == 1
        && matches!(
            ctx.builtin_of(*fn_id),
            Some(cas_ast::BuiltinFn::Sin | cas_ast::BuiltinFn::Cos)
        )
}

fn should_format_add_degree_first(ctx: &Context, terms: &[SignedAddTerm]) -> bool {
    if terms.len() < 3 {
        return false;
    }
    let mut max_degree = 0;
    for term in terms {
        let Some(degree) = display_polynomial_degree(ctx, term.id) else {
            return false;
        };
        max_degree = max_degree.max(degree);
    }
    max_degree >= 2
}

fn compare_add_terms_degree_first(
    ctx: &Context,
    a: &SignedAddTerm,
    b: &SignedAddTerm,
) -> std::cmp::Ordering {
    use crate::ordering::compare_expr;

    let deg_a = display_polynomial_degree(ctx, a.id).unwrap_or_default();
    let deg_b = display_polynomial_degree(ctx, b.id).unwrap_or_default();
    if deg_a != deg_b {
        return deg_b.cmp(&deg_a);
    }
    compare_expr(ctx, a.id, b.id)
}

fn display_polynomial_degree(ctx: &Context, id: ExprId) -> Option<i32> {
    match ctx.get(id) {
        Expr::Number(_) | Expr::Constant(_) => Some(0),
        Expr::Variable(_) => Some(1),
        Expr::Pow(base, exp) => {
            if matches!(ctx.get(*base), Expr::Variable(_)) {
                if let Expr::Number(n) = ctx.get(*exp) {
                    if n.is_integer() {
                        return n.to_integer().try_into().ok();
                    }
                }
            }
            None
        }
        Expr::Mul(a, b) => match (
            display_polynomial_degree(ctx, *a),
            display_polynomial_degree(ctx, *b),
        ) {
            (Some(left), Some(right)) => Some(left + right),
            _ => None,
        },
        Expr::Div(a, b) => {
            if display_polynomial_degree(ctx, *b) == Some(0) {
                display_polynomial_degree(ctx, *a)
            } else {
                None
            }
        }
        Expr::Neg(inner) => display_polynomial_degree(ctx, *inner),
        _ => None,
    }
}

fn format_signed_add_terms_in_order(
    f: &mut fmt::Formatter<'_>,
    ctx: &Context,
    terms: &[SignedAddTerm],
) -> fmt::Result {
    for (i, term) in terms.iter().enumerate() {
        let raw_is_neg = check_negative(ctx, term.id).0;
        let is_neg = term.invert_sign ^ raw_is_neg;

        if i == 0 {
            if is_neg {
                write!(f, "-")?;
                format_term_absolute(f, ctx, term.id)?;
            } else if term.invert_sign && raw_is_neg {
                format_term_absolute(f, ctx, term.id)?;
            } else {
                write!(
                    f,
                    "{}",
                    DisplayExpr {
                        context: ctx,
                        id: term.id
                    }
                )?;
            }
        } else if is_neg {
            write!(f, " - ")?;
            format_term_absolute(f, ctx, term.id)?;
        } else {
            write!(
                f,
                " + {}",
                DisplayExpr {
                    context: ctx,
                    id: term.id
                }
            )?;
        }
    }
    Ok(())
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
        Expr::Div(a, b) => {
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

fn is_positive_one_factor(ctx: &Context, id: ExprId) -> bool {
    matches!(ctx.get(id), Expr::Number(n) if n.is_one())
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
        Expr::Div(a, b) => {
            let left_neg = direct_negative_factor(ctx, *a);
            let right_neg = direct_negative_factor(ctx, *b);
            match (left_neg, right_neg) {
                (Some((inner, coeff)), None) => format_div_absolute(
                    f,
                    ctx,
                    RenderFactor::Direct(inner, coeff),
                    RenderFactor::Expr(*b),
                ),
                (None, Some((inner, coeff))) => format_div_absolute(
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

fn format_div_absolute(
    f: &mut fmt::Formatter<'_>,
    ctx: &Context,
    numerator: RenderFactor,
    denominator: RenderFactor,
) -> fmt::Result {
    let numerator_prec = match &numerator {
        RenderFactor::Direct(_, Some(_)) => 3,
        RenderFactor::Direct(id, None) | RenderFactor::Expr(id) => precedence(ctx, *id),
    };
    let denominator_prec = match &denominator {
        RenderFactor::Direct(_, Some(_)) => 3,
        RenderFactor::Direct(id, None) | RenderFactor::Expr(id) => precedence(ctx, *id),
    };

    if numerator_prec < 2 {
        write!(f, "(")?;
        format_render_factor_absolute(f, ctx, numerator)?;
        write!(f, ")")?;
    } else {
        format_render_factor_absolute(f, ctx, numerator)?;
    }

    write!(f, " / ")?;

    if denominator_prec <= 2 {
        write!(f, "(")?;
        format_render_factor_absolute(f, ctx, denominator)?;
        write!(f, ")")
    } else {
        format_render_factor_absolute(f, ctx, denominator)
    }
}

fn format_render_factor_absolute(
    f: &mut fmt::Formatter<'_>,
    ctx: &Context,
    factor: RenderFactor,
) -> fmt::Result {
    match factor {
        RenderFactor::Direct(_, Some(coeff)) => write!(f, "{}", coeff),
        RenderFactor::Direct(id, None) | RenderFactor::Expr(id) => {
            write!(f, "{}", DisplayExpr { context: ctx, id })
        }
    }
}

fn format_fraction_numerator(
    f: &mut fmt::Formatter<'_>,
    ctx: &Context,
    factors: &[DisplayFactor],
    prefer_numeric_first: bool,
) -> fmt::Result {
    if !prefer_numeric_first || factors.len() < 2 {
        return format_factors(f, ctx, factors);
    }

    let Some(numeric_idx) = factors
        .iter()
        .position(|factor| factor.exp == 1 && matches!(ctx.get(factor.base), Expr::Number(_)))
    else {
        return format_factors(f, ctx, factors);
    };

    if numeric_idx == 0 {
        return format_factors(f, ctx, factors);
    }

    let mut ordered = Vec::with_capacity(factors.len());
    ordered.push(factors[numeric_idx]);
    ordered.extend(factors[..numeric_idx].iter().copied());
    ordered.extend(factors[numeric_idx + 1..].iter().copied());
    format_factors(f, ctx, &ordered)
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
    if render_factor_is_unit(ctx, &left) {
        return format_abs_factor(f, ctx, right);
    }
    if render_factor_is_unit(ctx, &right) {
        return format_abs_factor(f, ctx, left);
    }

    format_abs_factor(f, ctx, left)?;
    write!(f, "{}", mul_symbol())?;
    format_abs_factor(f, ctx, right)
}

fn render_factor_is_unit(ctx: &Context, factor: &RenderFactor) -> bool {
    match factor {
        RenderFactor::Direct(_, Some(coeff)) => coeff.is_one(),
        RenderFactor::Direct(id, None) | RenderFactor::Expr(id) => is_positive_one_factor(ctx, *id),
    }
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
