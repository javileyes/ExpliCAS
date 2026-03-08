//! `DisplayExprWithHints`: Display with rendering hints (roots, fractions, etc.)

use crate::{Constant, Context, Expr, ExprId};
use std::fmt;

use super::expr::{check_negative, collect_add_terms, precedence};
use super::mul_symbol;
use super::ordering::{cmp_term_for_display, FractionDisplayView};
use super::unicode_root_prefix;

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
                crate::display_context::DisplayHint::PreferPower => {
                    // PreferPower hint: let regular formatting handle it (exponential form)
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
                Constant::Phi => write!(f, "phi"),
            },
            Expr::Variable(sym_id) => write!(f, "{}", self.context.sym_name(*sym_id)),
            Expr::Add(_, _) => {
                // Flatten Add chain to handle mixed signs gracefully
                let mut terms = collect_add_terms(self.context, id);

                // Sort by degree (descending) then sign (positive first) for polynomial order
                terms.sort_by(|a, b| cmp_term_for_display(self.context, *a, *b));

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
                                // Add parentheses when inner is Add/Sub to preserve grouping
                                // The "-" has already been printed above, so just print abs value
                                let inner_is_add_sub = matches!(
                                    self.context.get(*inner),
                                    Expr::Add(_, _) | Expr::Sub(_, _)
                                );
                                if inner_is_add_sub {
                                    write!(f, "(")?;
                                    self.fmt_internal(f, *inner)?;
                                    write!(f, ")")?;
                                } else {
                                    self.fmt_internal(f, *inner)?;
                                }
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
                // Add parentheses when inner is Add/Sub to preserve grouping
                // e.g., -(a + b) should display as "-(a + b)" not "-a + b"
                let inner_is_add_sub =
                    matches!(self.context.get(*e), Expr::Add(_, _) | Expr::Sub(_, _));
                if inner_is_add_sub {
                    write!(f, "(")?;
                    self.fmt_internal(f, *e)?;
                    write!(f, ")")
                } else {
                    self.fmt_internal(f, *e)
                }
            }
            Expr::Function(fn_id, args) => {
                let name = self.context.sym_name(*fn_id);
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
            // Hold is transparent for display - render inner directly
            Expr::Hold(inner) => self.fmt_internal(f, *inner),
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
