//! `DisplayExprStyled`: Global style-based formatting (replaces hints).

use crate::{Constant, Context, Expr, ExprId};
use std::fmt;

use super::expr::{check_negative, collect_add_terms, precedence};
use super::mul_symbol;
use super::ordering::cmp_term_for_display;
use super::{is_pretty_output, number_to_superscript, unicode_root_prefix};

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
                    use num_traits::ToPrimitive;

                    // Helper to extract (numerator, denominator) from exponent
                    let get_frac_parts = |exp_id: ExprId| -> Option<(i64, i64)> {
                        match self.context.get(exp_id) {
                            // Expr::Number with rational value
                            Expr::Number(n) => {
                                if !n.is_integer() {
                                    let numer = n.numer().to_i64()?;
                                    let denom = n.denom().to_i64()?;
                                    Some((numer, denom))
                                } else {
                                    None
                                }
                            }
                            // Expr::Div(num, den) form
                            Expr::Div(num, den) => {
                                if let (Expr::Number(n), Expr::Number(d)) =
                                    (self.context.get(*num), self.context.get(*den))
                                {
                                    if n.is_integer() && d.is_integer() {
                                        let numer = n.numer().to_i64()?;
                                        let denom = d.numer().to_i64()?;
                                        if denom > 1 {
                                            return Some((numer, denom));
                                        }
                                    }
                                }
                                None
                            }
                            _ => None,
                        }
                    };

                    if let Some((numer, denom)) = get_frac_parts(*exp) {
                        if denom > 1 {
                            if numer == 1 {
                                // Print as root: √(base)
                                write!(f, "{}(", unicode_root_prefix(denom as u64))?;
                                self.fmt_internal(f, *base)?;
                                return write!(f, ")");
                            } else if numer != 1 {
                                // Print as k-th power under d-th root: ∜(base^k)
                                write!(f, "{}(", unicode_root_prefix(denom as u64))?;
                                self.fmt_internal(f, *base)?;
                                write!(f, "^{})", numer)?;
                                return Ok(());
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
                Constant::Phi => write!(f, "phi"),
            },
            Expr::Variable(sym_id) => write!(f, "{}", self.context.sym_name(*sym_id)),

            Expr::Add(_, _) => {
                // Collect terms and display with proper signs
                let mut terms = collect_add_terms(self.context, id);

                // Sort by degree (descending) then sign (positive first) for polynomial order
                // cmp_term_for_display combines both criteria
                terms.sort_by(|a, b| cmp_term_for_display(self.context, *a, *b));

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
                // Check if RHS needs parens (same logic as DisplayExpr):
                // RHS is Add/Sub: a - (b + c) or a - (b - c) needs parens to preserve associativity
                let rhs_is_add_sub =
                    matches!(self.context.get(*r), Expr::Add(_, _) | Expr::Sub(_, _));
                self.fmt_internal(f, *l)?;
                write!(f, " - ")?;
                if rhs_is_add_sub {
                    write!(f, "(")?;
                    self.fmt_internal(f, *r)?;
                    write!(f, ")")
                } else {
                    self.fmt_internal(f, *r)
                }
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

            Expr::Function(fn_id, args) => {
                let name = self.context.sym_name(*fn_id);
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
            // Hold is transparent for display - render inner directly
            Expr::Hold(inner) => self.fmt_internal(f, *inner),
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
        // Check if base is "negative" (Neg expr, negative number, or leading negative factor)
        // These need parentheses: (-1)² not -1², (-x)² not -x²
        let (base_is_negative, _, _) = check_negative(self.context, base);

        if base_prec < 3 || base_is_negative {
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
            Expr::Neg(inner) => {
                // Add parentheses when inner is Add/Sub to preserve grouping
                // The "-" has already been printed by the caller
                let inner_is_add_sub =
                    matches!(self.context.get(*inner), Expr::Add(_, _) | Expr::Sub(_, _));
                if inner_is_add_sub {
                    write!(f, "(")?;
                    self.fmt_internal(f, *inner)?;
                    write!(f, ")")
                } else {
                    self.fmt_internal(f, *inner)
                }
            }
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
    use super::super::expr::DisplayExpr;
    use crate::{Context, Expr};

    #[test]
    fn test_hold_transparency() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let held = crate::hold::wrap_hold(&mut ctx, x);
        let display = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: held
            }
        );
        assert_eq!(display, "x", "Expected 'x' but got '{}'", display);
    }

    #[test]
    fn test_display_styled_pow_half_as_radical() {
        use crate::root_style::{RootStyle, StylePreferences};

        let mut ctx = Context::new();
        let two = ctx.num(2);
        let half = ctx.rational(1, 2);
        let sqrt2 = ctx.add(Expr::Pow(two, half));

        // With Radical style, should show sqrt (ASCII) or √ (pretty)
        // Tests run in ASCII mode, so check for "sqrt"
        let style = StylePreferences::with_root_style(RootStyle::Radical);
        let disp = format!("{}", super::DisplayExprStyled::new(&ctx, sqrt2, &style));

        // In ASCII mode (tests), unicode_root_prefix(2) returns "sqrt"
        // In pretty mode (CLI), it returns "√"
        assert!(
            disp.contains("sqrt") || disp.contains("√"),
            "Expected sqrt or √ in output, got: {}",
            disp
        );
        // Verify it's NOT ^(1/2) format
        assert!(
            !disp.contains("^(1/2)") && !disp.contains("^(1 / 2)"),
            "Should not use exponential format, got: {}",
            disp
        );
    }
}
