//! Term ordering and fraction display view for expression display.

use crate::{Context, Expr, ExprId};
use num_traits::{One, Signed, ToPrimitive};
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

fn ln_power_display_degree(ctx: &Context, id: ExprId) -> Option<i32> {
    match ctx.get(id) {
        Expr::Number(_) => Some(0),
        Expr::Neg(inner) => ln_power_display_degree(ctx, *inner),
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(cas_ast::BuiltinFn::Ln) =>
        {
            Some(1)
        }
        Expr::Pow(base, exp) => {
            let Expr::Function(fn_id, args) = ctx.get(*base) else {
                return None;
            };
            if args.len() != 1 || ctx.builtin_of(*fn_id) != Some(cas_ast::BuiltinFn::Ln) {
                return None;
            }
            let Expr::Number(n) = ctx.get(*exp) else {
                return None;
            };
            if n.is_integer() && n.is_positive() {
                n.to_i32()
            } else {
                None
            }
        }
        Expr::Mul(a, b) => match (ctx.get(*a), ctx.get(*b)) {
            (Expr::Number(_), _) => ln_power_display_degree(ctx, *b),
            (_, Expr::Number(_)) => ln_power_display_degree(ctx, *a),
            _ => None,
        },
        _ => None,
    }
}

/// Compare terms for display ordering.
/// Primary for function-polynomial terms: recognizable degree descending, so
/// post-calculus log polynomials keep their mathematical reading order.
/// Primary otherwise: positive terms before negative, preserving established
/// affine orientation such as `1 - x`.
/// Tertiary: degree and compare_expr tie-breaking.
pub fn cmp_term_for_display(ctx: &Context, a: ExprId, b: ExprId) -> std::cmp::Ordering {
    use crate::ordering::compare_expr;
    use std::cmp::Ordering;

    let (a_neg, _, _) = check_negative(ctx, a);
    let (b_neg, _, _) = check_negative(ctx, b);
    let deg_a = poly_degree_fast(ctx, a);
    let deg_b = poly_degree_fast(ctx, b);
    let ln_deg_a = ln_power_display_degree(ctx, a);
    let ln_deg_b = ln_power_display_degree(ctx, b);
    let log_polynomial_pair = matches!((ln_deg_a, ln_deg_b), (Some(a), Some(b)) if a > 0 || b > 0);

    if log_polynomial_pair {
        // Log-polynomial presentation, e.g. ln(x)^5 - 2*ln(x)^4 + ...
        if let (Some(da), Some(db)) = (ln_deg_a, ln_deg_b) {
            if da != db {
                return db.cmp(&da);
            }
        }
    }

    // Positive terms before negative terms when degree did not decide.
    match (a_neg, b_neg) {
        (false, true) => return Ordering::Less, // positive < negative
        (true, false) => return Ordering::Greater, // negative > positive
        _ => {}                                 // Same sign: fall through to secondary
    }

    if !log_polynomial_pair {
        // Original non-function behavior: within the same sign, order by degree.
        match (deg_a, deg_b) {
            (Some(da), Some(db)) => {
                if da != db {
                    return db.cmp(&da);
                }
            }
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (None, None) => {}
        }
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
        let mut worklist = vec![(id, false)];

        while let Some((curr, in_denominator)) = worklist.pop() {
            match ctx.get(curr) {
                Expr::Matrix { .. } => return None, // Matrix blocks fraction display

                Expr::Neg(inner) => {
                    sign *= -1;
                    worklist.push((*inner, in_denominator));
                }

                Expr::Mul(l, r) => {
                    // Check for matrices in either operand
                    if matches!(ctx.get(*l), Expr::Matrix { .. })
                        || matches!(ctx.get(*r), Expr::Matrix { .. })
                    {
                        return None;
                    }
                    worklist.push((*l, in_denominator));
                    worklist.push((*r, in_denominator));
                }

                Expr::Pow(base, exp_id) => {
                    if matches!(ctx.get(*base), Expr::Matrix { .. }) {
                        return None;
                    }

                    // Check if exponent is integer
                    if let Some(exp) = Self::as_int(ctx, *exp_id) {
                        let effective_exp = if in_denominator { -exp } else { exp };
                        if effective_exp < 0 {
                            den.push(DisplayFactor {
                                base: *base,
                                exp: -effective_exp,
                            });
                        } else if effective_exp > 0 {
                            num.push(DisplayFactor {
                                base: *base,
                                exp: effective_exp,
                            });
                        }
                        // exp == 0 means factor is 1, skip
                    } else if in_denominator {
                        den.push(DisplayFactor { base: curr, exp: 1 });
                    } else {
                        // Non-integer exponent: treat as single num factor
                        num.push(DisplayFactor { base: curr, exp: 1 });
                    }
                }

                Expr::Div(n, d) => {
                    // Add numerator factors, denominator factors
                    if matches!(ctx.get(*d), Expr::Matrix { .. }) {
                        return None;
                    }
                    worklist.push((*n, in_denominator));
                    if in_denominator {
                        worklist.push((*d, false));
                    } else {
                        match ctx.get(*d) {
                            Expr::Neg(inner) => {
                                sign *= -1;
                                den.push(DisplayFactor {
                                    base: *inner,
                                    exp: 1,
                                });
                            }
                            Expr::Number(n) if n.is_negative() => {
                                sign *= -1;
                                if !(-n.clone()).is_one() {
                                    den.push(DisplayFactor { base: *d, exp: 1 });
                                }
                            }
                            _ => den.push(DisplayFactor { base: *d, exp: 1 }),
                        }
                    }
                }

                Expr::Number(n) if n.is_negative() => {
                    sign *= -1;
                    if !(-n.clone()).is_one() {
                        if in_denominator {
                            den.push(DisplayFactor { base: curr, exp: 1 });
                        } else {
                            num.push(DisplayFactor { base: curr, exp: 1 });
                        }
                    }
                }

                // Atoms: add as numerator factor
                _ => {
                    if in_denominator {
                        den.push(DisplayFactor { base: curr, exp: 1 });
                    } else {
                        num.push(DisplayFactor { base: curr, exp: 1 });
                    }
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
        // A non-integer rational base renders as a fraction `p/q` and MUST be parenthesized before an
        // exponent (`(3/2)^2`, not `3/2^2` — which re-parses as `3/(2^2)`). `precedence` reports atom
        // for a `Number`, so detect the fraction value explicitly, same as the `Pow` renderer.
        let base_is_fraction_number =
            matches!(ctx.get(factor.base), Expr::Number(n) if !n.is_integer());
        let needs_parens = base_prec < 2 || (factor.exp != 1 && base_is_fraction_number);

        if let Expr::Number(n) = ctx.get(factor.base) {
            if n.is_negative() {
                let magnitude = -n.clone();
                // `|p/q|^e` needs the same parentheses as the positive case.
                if factor.exp != 1 && !magnitude.is_integer() {
                    write!(f, "({magnitude})^{}", factor.exp)?;
                } else {
                    write!(f, "{magnitude}")?;
                    if factor.exp != 1 {
                        write!(f, "^{}", factor.exp)?;
                    }
                }
                continue;
            }
        }

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
