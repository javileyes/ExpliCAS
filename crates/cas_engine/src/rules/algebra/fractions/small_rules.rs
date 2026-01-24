//! Light rationalization rules for simple surd denominators.

use crate::build::mul2_raw;
use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{count_nodes, Context, DisplayExpr, Expr, ExprId};

// ========== Light Rationalization for Single Numeric Surd Denominators ==========
// Transforms: num / (k * √n) → (num * √n) / (k * n)
// Only applies when:
// - denominator contains exactly one numeric square root
// - base of the root is a positive integer
// - no variables inside the radical

define_rule!(
    RationalizeSingleSurdRule,
    "Rationalize Single Surd",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        use cas_ast::views::as_rational_const;
        use num_rational::BigRational;
        use num_traits::ToPrimitive;

        // Only match Div expressions - use zero-clone helper
        let (num, den) = crate::helpers::as_div(ctx, expr)?;

        // Check denominator for Pow(Number(n), 1/2) patterns
        // We need to find exactly one surd in the denominator factors

        // Helper to check if an expression is a numeric square root
        fn is_numeric_sqrt(ctx: &Context, id: ExprId) -> Option<i64> {
            if let Expr::Pow(base, exp) = ctx.get(id) {
                // Check exponent is 1/2 (using robust detection)
                let exp_val = as_rational_const(ctx, *exp, 8)?;
                let half = BigRational::new(1.into(), 2.into());
                if exp_val != half {
                    return None;
                }
                // Check base is a positive integer
                if let Expr::Number(n) = ctx.get(*base) {
                    if n.is_integer() {
                        return n.numer().to_i64().filter(|&x| x > 0);
                    }
                }
            }
            None
        }

        // Try different denominator patterns
        // CLONE_OK: Multi-branch match extracting sqrt value from Mul product
        let (sqrt_n_value, other_den_factors): (i64, Vec<ExprId>) = match ctx.get(den).clone() {
            // Case 1: Denominator is just √n
            Expr::Pow(_, _) => {
                if let Some(n) = is_numeric_sqrt(ctx, den) {
                    (n, vec![])
                } else {
                    return None;
                }
            }

            // Case 2: Denominator is k * √n or √n * k (one level of Mul)
            Expr::Mul(l, r) => {
                if let Some(n) = is_numeric_sqrt(ctx, l) {
                    // √n * k form
                    (n, vec![r])
                } else if let Some(n) = is_numeric_sqrt(ctx, r) {
                    // k * √n form
                    (n, vec![l])
                } else {
                    // Check if either side is a Mul containing √n (two levels)
                    // For simplicity, we only handle shallow cases
                    return None;
                }
            }

            // Case 3: Function("sqrt", [n])
            Expr::Function(name, ref args) if ctx.sym_name(*name) == "sqrt" && args.len() == 1 => {
                if let Expr::Number(n) = ctx.get(args[0]) {
                    if n.is_integer() {
                        if let Some(n_int) = n.numer().to_i64().filter(|&x| x > 0) {
                            (n_int, vec![])
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                } else {
                    return None; // Variable inside sqrt
                }
            }

            _ => return None,
        };

        // Build the rationalized form: (num * √n) / (other_den * n)
        let n_expr = ctx.num(sqrt_n_value);
        let half = ctx.rational(1, 2);
        let sqrt_n = ctx.add(Expr::Pow(n_expr, half));

        // New numerator: num * √n
        let new_num = mul2_raw(ctx, num, sqrt_n);

        // New denominator: other_den_factors * n
        let n_in_den = ctx.num(sqrt_n_value);
        let new_den = if other_den_factors.is_empty() {
            n_in_den
        } else {
            let mut den_product = other_den_factors[0];
            for &f in &other_den_factors[1..] {
                den_product = mul2_raw(ctx, den_product, f);
            }
            mul2_raw(ctx, den_product, n_in_den)
        };

        let new_expr = ctx.add(Expr::Div(new_num, new_den));

        // Optional: Check node count didn't explode (shouldn't for this simple transform)
        if count_nodes(ctx, new_expr) > count_nodes(ctx, expr) + 10 {
            return None;
        }

        Some(Rewrite::new(new_expr).desc(format!(
            "{} / {} -> {} / {}",
            DisplayExpr {
                context: ctx,
                id: num
            },
            DisplayExpr {
                context: ctx,
                id: den
            },
            DisplayExpr {
                context: ctx,
                id: new_num
            },
            DisplayExpr {
                context: ctx,
                id: new_den
            }
        )))
    }
);
