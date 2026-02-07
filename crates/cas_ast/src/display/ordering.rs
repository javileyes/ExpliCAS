//! Term ordering and fraction display view for expression display.

use crate::{Context, Expr, ExprId};
use num_traits::ToPrimitive;
use std::fmt;

use super::expr::{check_negative, precedence, DisplayExpr};
use super::mul_symbol;

// ============================================================================
// Unified Term Ordering for Add/Mul Display
// ============================================================================

/// Ordering mode for commutative operations (Add, Mul) in display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OrderingMode {
    /// Robust canonical ordering: polynomial degree (desc) + compare_expr fallback
    #[default]
    Canonical,
    /// No reordering (debug: display in AST creation order)
    None,
}

/// Fast polynomial degree heuristic (returns None for non-polynomial terms).
/// Does NOT recurse into Add - treats Add terms as atomic.
fn poly_degree_fast(ctx: &Context, id: ExprId) -> Option<i32> {
    match ctx.get(id) {
        Expr::Number(_) | Expr::Constant(_) => Some(0),
        Expr::Variable(_) => Some(1),
        Expr::Pow(base, exp) => {
            // Only integer exponents on variables
            if let Expr::Variable(_) = ctx.get(*base) {
                if let Expr::Number(n) = ctx.get(*exp) {
                    if n.is_integer() {
                        return n.to_i32();
                    }
                }
            }
            None // Non-integer exponent or non-variable base
        }
        Expr::Mul(a, b) => {
            // Sum of degrees
            match (poly_degree_fast(ctx, *a), poly_degree_fast(ctx, *b)) {
                (Some(da), Some(db)) => Some(da + db),
                _ => None,
            }
        }
        Expr::Neg(inner) => poly_degree_fast(ctx, *inner),
        // Everything else: not polynomial for display purposes
        _ => None,
    }
}

/// Compare terms for display ordering.
/// Primary: positive terms before negative (for "x + 1 - (...)" ordering)
/// Secondary: non-polynomial before polynomial within same sign (for "2^(1/2) - 1" ordering)
/// Tertiary: polynomial degree descending, then compare_expr for tie-breaking
pub fn cmp_term_for_display(ctx: &Context, a: ExprId, b: ExprId) -> std::cmp::Ordering {
    use crate::ordering::compare_expr;
    use std::cmp::Ordering;

    // PRIMARY: Positive terms before negative terms
    let (a_neg, _, _) = check_negative(ctx, a);
    let (b_neg, _, _) = check_negative(ctx, b);
    match (a_neg, b_neg) {
        (false, true) => return Ordering::Less, // positive < negative
        (true, false) => return Ordering::Greater, // negative > positive
        _ => {}                                 // Same sign: fall through to secondary
    }

    // SECONDARY: Within same sign, order by polynomial degree
    let deg_a = poly_degree_fast(ctx, a);
    let deg_b = poly_degree_fast(ctx, b);

    match (deg_a, deg_b) {
        (Some(da), Some(db)) => {
            // Both polynomial: higher degree first
            if da != db {
                return db.cmp(&da);
            }
        }
        // Non-polynomial before polynomial (for "2^(1/2) - 1" with root before constant)
        (None, Some(_)) => return Ordering::Less,
        (Some(_), None) => return Ordering::Greater,
        (None, None) => {} // Both non-polynomial: fall through
    }

    // TERTIARY: Structural comparison for total order
    compare_expr(ctx, a, b)
}

// ============================================================================
// Fraction Display View
// ============================================================================

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
pub(super) fn format_factors(
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
