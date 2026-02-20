//! Phase shift and supplementary angle rules.
//!
//! Contains `SinSupplementaryAngleRule`.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Expr};

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
