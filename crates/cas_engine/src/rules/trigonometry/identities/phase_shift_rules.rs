//! Phase shift and supplementary angle rules.
//!
//! Contains `SinSupplementaryAngleRule` and the `extract_phase_shift` helper
//! used by other trigonometric rule modules.

use crate::define_rule;
use crate::helpers::{as_add, as_mul, is_pi};
use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Expr, ExprId};
use cas_math::trig_phase_shift_support::{extract_pi_coefficient, extract_pi_half_multiple};
use num_traits::One;

// =============================================================================
// Sin Supplementary Angle Rule
// =============================================================================
// sin(π - x) → sin(x)
// sin(k·π - x) → (-1)^(k+1) · sin(x) for integer k
// cos(π - x) → -cos(x)
//
// This enables simplification of expressions like sin(8π/9) = sin(π - π/9) = sin(π/9)

define_rule!(
    SinSupplementaryAngleRule,
    "Supplementary Angle",
    |ctx, expr| {
        use crate::helpers::extract_rational_pi_multiple;
        use num_rational::BigRational;

        let (fn_id, args) = match ctx.get(expr) {
            Expr::Function(fn_id, args) => (*fn_id, args.clone()),
            _ => return None,
        };
        {
            let builtin = ctx.builtin_of(fn_id);
            if args.len() != 1 {
                return None;
            }

            let is_sin = matches!(builtin, Some(BuiltinFn::Sin));
            let is_cos = matches!(builtin, Some(BuiltinFn::Cos));
            if !is_sin && !is_cos {
                return None;
            }

            let arg = args[0];

            // Try to check if arg is a rational multiple of π
            // where the coefficient is of the form (n - small) for some positive integer n
            // e.g., 8/9 = 1 - 1/9, so sin(8π/9) = sin(π - π/9) = sin(π/9)

            if let Some(k) = extract_rational_pi_multiple(ctx, arg) {
                // k = p/q in lowest terms
                let p = k.numer();
                let q = k.denom();

                // Check if p/q is close enough to an integer that the supplementary form is simpler.
                // For sin(k·π) where k = (n*q - m)/q, we can write it as sin((n*q - m)/q · π) = sin(n·π - m/q·π)
                // This simplifies when m < p (i.e., the remainder is smaller than the original numerator)
                //
                // Example: sin(8/9·π) = sin(1·π - 1/9·π) = sin(π/9) because 1 < 8

                // Only for positive k (p > 0)
                if p > &num_bigint::BigInt::from(0) {
                    let one = num_bigint::BigInt::from(1);
                    // n = ceil(p/q) = floor((p + q - 1) / q)
                    let n_candidate = (p + q - &one) / q;
                    let remainder = &n_candidate * q - p; // m = n*q - p

                    // Apply simplification if:
                    // 1. remainder > 0 (i.e., k is not an integer)
                    // 2. remainder < p (i.e., the new form is simpler)
                    // 3. n >= 1 (always true since p > 0)
                    if remainder > num_bigint::BigInt::from(0) && &remainder < p {
                        // The supplementary angle is m/q * π
                        let new_coeff = BigRational::new(remainder.clone(), q.clone());

                        // Build the new angle: (m/q) * π
                        let new_angle = if new_coeff == BigRational::from_integer(1.into()) {
                            ctx.add(Expr::Constant(cas_ast::Constant::Pi))
                        } else {
                            let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                            let coeff_expr = ctx.add(Expr::Number(new_coeff));
                            ctx.add(Expr::Mul(coeff_expr, pi))
                        };

                        // Determine sign based on parity of n
                        // sin(n·π - x) = (-1)^(n+1) · sin(x)
                        // cos(n·π - x) = (-1)^n · cos(x)
                        let n_parity_odd = &n_candidate % 2 == one;

                        let (result, desc) = if is_sin {
                            // sin(n·π - x) = (-1)^(n+1) · sin(x)
                            // n odd → (-1)^(n+1) = 1, so sin(x)
                            // n even → (-1)^(n+1) = -1, so -sin(x)
                            let new_trig =
                                ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![new_angle]);
                            if n_parity_odd {
                                (new_trig, format!("sin({}π - x) = sin(x)", n_candidate))
                            } else {
                                (
                                    ctx.add(Expr::Neg(new_trig)),
                                    format!("sin({}π - x) = -sin(x)", n_candidate),
                                )
                            }
                        } else {
                            // cos(n·π - x) = (-1)^n · cos(x)
                            // n odd → -cos(x), n even → cos(x)
                            let new_trig =
                                ctx.call_builtin(cas_ast::BuiltinFn::Cos, vec![new_angle]);
                            if n_parity_odd {
                                (
                                    ctx.add(Expr::Neg(new_trig)),
                                    format!("cos({}π - x) = -cos(x)", n_candidate),
                                )
                            } else {
                                (new_trig, format!("cos({}π - x) = cos(x)", n_candidate))
                            }
                        };

                        return Some(Rewrite::new(result).desc(desc));
                    }
                }
            }
        }

        None
    }
);

/// Extract (base_term, k) from arg such that arg = base_term + k*π/2
/// Handles multiple canonical forms:
/// - Add(x, π/2) → (x, 1)
/// - Sub(x, π/2) → (x, -1)  
/// - Div(Add(n*x, k*π), m) → (x, k*2/m) if m divides k*2
pub fn extract_phase_shift(ctx: &mut cas_ast::Context, expr: ExprId) -> Option<(ExprId, i32)> {
    // Form 1: Div((coeff*x + k*pi), denom) - the canonical form!
    // Example: (2*x + pi)/2 means x + pi/2, so k=1
    if let Expr::Div(num, den) = ctx.get(expr) {
        let num = *num;
        let den = *den;

        // Get denominator value
        let denom_val: i32 = if let Expr::Number(n) = ctx.get(den) {
            if n.is_integer() {
                n.to_integer().try_into().ok()?
            } else {
                return None;
            }
        } else {
            return None;
        };

        // Check if numerator is Add/Sub
        if let Some((l, r)) = as_add(ctx, num) {
            // Check both terms for π
            // Form: (base + k*pi)/denom where we want k/denom = m/2 for some integer m

            // Check right term for π (most common: 2*x + pi)
            if is_pi(ctx, r) {
                // pi/denom * 2 = shift in units of pi/2
                // For denom=2, shift = 1 (one pi/2)
                let k = 2 / denom_val; // Only works if denom divides 2
                if 2 % denom_val != 0 {
                    // Not a clean pi/2 multiple
                } else {
                    let base = ctx.add(Expr::Div(l, den));
                    return Some((base, k));
                }
            }

            // Check left term for π (less common: pi + 2*x)
            if is_pi(ctx, l) {
                let k = 2 / denom_val;
                if 2 % denom_val != 0 {
                    // Not a clean pi/2 multiple
                } else {
                    let base = ctx.add(Expr::Div(r, den));
                    return Some((base, k));
                }
            }

            // Also check for k*pi form using extract_pi_coefficient
            if let Some(pi_coeff) = extract_pi_coefficient(ctx, r) {
                let k_times_2 = 2 * pi_coeff;
                if k_times_2 % denom_val == 0 {
                    let k = k_times_2 / denom_val;
                    let base = ctx.add(Expr::Div(l, den));
                    return Some((base, k));
                }
            }

            if let Some(pi_coeff) = extract_pi_coefficient(ctx, l) {
                let k_times_2 = 2 * pi_coeff;
                if k_times_2 % denom_val == 0 {
                    let k = k_times_2 / denom_val;
                    let base = ctx.add(Expr::Div(r, den));
                    return Some((base, k));
                }
            }
        }
    }

    // Form 1b: Mul(1/n, Add(coeff*x, k*pi)) - the canonical form for (a + b)/n!
    // Example: (2*x + pi)/2 becomes Mul(1/2, Add(2*x, pi)); shift = 1
    if let Some((coeff_id, inner)) = as_mul(ctx, expr) {
        // Check if coeff is 1/n (a rational with numerator 1)
        if let Expr::Number(coeff) = ctx.get(coeff_id) {
            if coeff.numer() == &num_bigint::BigInt::from(1) && !coeff.denom().is_one() {
                let denom_val: i32 = coeff.denom().try_into().ok().unwrap_or(0);
                if denom_val > 0 {
                    // Check if inner is Add containing pi
                    if let Some((l, r)) = as_add(ctx, inner) {
                        // Check right term for pi
                        if is_pi(ctx, r) {
                            let k = 2 / denom_val;
                            if 2 % denom_val == 0 {
                                // base = l / denom = l * (1/denom) = coeff * l
                                let base = ctx.add(Expr::Mul(coeff_id, l));
                                return Some((base, k));
                            }
                        }

                        // Check left term for pi
                        if is_pi(ctx, l) {
                            let k = 2 / denom_val;
                            if 2 % denom_val == 0 {
                                let base = ctx.add(Expr::Mul(coeff_id, r));
                                return Some((base, k));
                            }
                        }

                        // Check for k*pi form
                        if let Some(pi_coeff) = extract_pi_coefficient(ctx, r) {
                            let k_times_2 = 2 * pi_coeff;
                            if k_times_2 % denom_val == 0 {
                                let k = k_times_2 / denom_val;
                                let base = ctx.add(Expr::Mul(coeff_id, l));
                                return Some((base, k));
                            }
                        }
                    }
                }
            }
        }
    }

    // Form 2/3: Scan the full n-ary sum for a π/2 multiple anywhere among the terms.
    // Handles Add(a, b, π), Sub(a, π), Add(Add(2u,3), π), etc.
    {
        use crate::nary::{AddView, Sign};

        let view = AddView::from_expr(ctx, expr);
        if view.terms.len() >= 2 {
            // Search for a term that is a π/2 multiple
            for (i, (term, sign)) in view.terms.iter().enumerate() {
                if let Some(mut k) = extract_pi_half_multiple(ctx, *term) {
                    if *sign == Sign::Neg {
                        k = -k;
                    }
                    // Rebuild the remaining terms without the π multiple
                    let remaining: smallvec::SmallVec<[(ExprId, Sign); 8]> = view
                        .terms
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, t)| *t)
                        .collect();
                    let rest_view = AddView {
                        root: expr,
                        terms: remaining,
                    };
                    let base = rest_view.rebuild(ctx);
                    return Some((base, k));
                }
            }
        }
    }

    None
}
