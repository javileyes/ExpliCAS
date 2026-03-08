//! Expression normalization helpers used by implicit-domain rendering.

use cas_ast::{Context, Expr, ExprId};

/// Normalize an expression for display in domain conditions.
///
/// Strategy:
/// 1. Try polynomial conversion and canonicalization.
/// 2. If leading polynomial coefficient is negative, negate the polynomial.
/// 3. Fallback: unwrap top-level negation (`-E` -> `E`) for cleaner condition display.
pub fn normalize_condition_expr(ctx: &mut Context, expr: ExprId) -> ExprId {
    use crate::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};

    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 20,
        max_pow_exp: 10,
    };

    if let Ok(poly) = multipoly_from_expr(ctx, expr, &budget) {
        let needs_negation = if let Some((coeff, _mono)) = poly.leading_term_lex() {
            coeff < &num_rational::BigRational::from_integer(0.into())
        } else {
            false
        };

        let normalized_poly = if needs_negation { poly.neg() } else { poly };
        return multipoly_to_expr(&normalized_poly, ctx);
    }

    if let Expr::Neg(inner) = ctx.get(expr) {
        return *inner;
    }

    expr
}

/// Extract base when expression is `base^(2k)` with integer `k > 0`.
///
/// Returns `Some(base)` only for positive even integer exponents.
pub fn extract_even_positive_power_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*exp) {
            if n.is_integer() {
                let exp_int = n.to_integer();
                let two: num_bigint::BigInt = 2.into();
                let zero: num_bigint::BigInt = 0.into();
                if &exp_int % &two == zero && exp_int > zero {
                    return Some(*base);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn normalize_condition_expr_polynomial_order() {
        let mut ctx = Context::new();
        let expr = parse("1+x", &mut ctx).expect("parse");
        let norm = normalize_condition_expr(&mut ctx, expr);
        let rendered = cas_formatter::DisplayExpr {
            context: &ctx,
            id: norm,
        }
        .to_string();
        assert_eq!(rendered, "x + 1");
    }

    #[test]
    fn normalize_condition_expr_unwraps_negation() {
        let mut ctx = Context::new();
        let expr = parse("-(x+1)", &mut ctx).expect("parse");
        let norm = normalize_condition_expr(&mut ctx, expr);
        let rendered = cas_formatter::DisplayExpr {
            context: &ctx,
            id: norm,
        }
        .to_string();
        assert_eq!(rendered, "x + 1");
    }

    #[test]
    fn extract_even_positive_power_base_detects_match() {
        let mut ctx = Context::new();
        let expr = parse("x^4", &mut ctx).expect("parse");
        let x = parse("x", &mut ctx).expect("parse");
        assert_eq!(extract_even_positive_power_base(&ctx, expr), Some(x));
    }

    #[test]
    fn extract_even_positive_power_base_rejects_odd_and_zero() {
        let mut ctx = Context::new();
        let odd = parse("x^3", &mut ctx).expect("parse");
        let zero = parse("x^0", &mut ctx).expect("parse");
        assert_eq!(extract_even_positive_power_base(&ctx, odd), None);
        assert_eq!(extract_even_positive_power_base(&ctx, zero), None);
    }
}
