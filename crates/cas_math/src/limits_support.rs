use crate::infinity_support::{mk_infinity, InfSign};
use cas_ast::{Constant, Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;

/// Check if an expression depends on a specific variable id.
///
/// Uses iterative traversal to avoid recursion limits on deep trees.
pub fn depends_on(ctx: &Context, expr: ExprId, var: ExprId) -> bool {
    let mut stack = vec![expr];

    while let Some(current) = stack.pop() {
        if current == var {
            return true;
        }

        match ctx.get(current) {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) => stack.push(*inner),
            Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => {
                for arg in args {
                    stack.push(*arg);
                }
            }
            Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) => {}
            Expr::Matrix { .. } | Expr::SessionRef(_) => {}
        }
    }

    false
}

/// Parse a power expression with integer exponent.
///
/// Returns `(base, n)` if `expr` is `base^n` where `n` is an integer literal.
pub fn parse_pow_int(ctx: &Context, expr: ExprId) -> Option<(ExprId, i64)> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            let n = crate::expr_extract::extract_i64_integer(ctx, *exp)?;
            Some((*base, n))
        }
        _ => None,
    }
}

/// Create a residual limit expression: `limit(expr, var, approach_symbol)`.
pub fn mk_limit(ctx: &mut Context, expr: ExprId, var: ExprId, approach: InfSign) -> ExprId {
    let approach_sym = match approach {
        InfSign::Pos => ctx.add(Expr::Constant(Constant::Infinity)),
        InfSign::Neg => {
            let inf = ctx.add(Expr::Constant(Constant::Infinity));
            ctx.add(Expr::Neg(inf))
        }
    };
    ctx.call("limit", vec![expr, var, approach_sym])
}

/// Determine resulting infinity sign from approach sign and exponent parity.
pub fn limit_sign(approach: InfSign, power: i64) -> InfSign {
    match approach {
        InfSign::Pos => InfSign::Pos,
        InfSign::Neg => {
            if power % 2 == 0 {
                InfSign::Pos // (-∞)^even = +∞
            } else {
                InfSign::Neg // (-∞)^odd = -∞
            }
        }
    }
}

/// Create infinity with appropriate sign.
pub fn mk_inf(ctx: &mut Context, sign: InfSign) -> ExprId {
    mk_infinity(ctx, sign)
}

/// Rational polynomial limit rule for `P(x)/Q(x)` as `x -> ±∞`.
///
/// Compares polynomial degrees in `var`:
/// - `deg(P) < deg(Q) -> 0`
/// - `deg(P) = deg(Q) -> lc(P)/lc(Q)` when both leading coefficients are numeric
/// - `deg(P) > deg(Q) -> ±∞` according to leading coefficient sign and parity
pub fn rational_poly_limit(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    approach: InfSign,
) -> Option<ExprId> {
    use crate::multipoly::{multipoly_from_expr, PolyBudget};
    use num_traits::Signed;

    // Match Div(num, den)
    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };

    // Get variable name
    let Expr::Variable(var_sym_id) = ctx.get(var).clone() else {
        return None;
    };
    let var_name = ctx.sym_name(var_sym_id);

    // Conservative budget for polynomial conversion
    let budget = PolyBudget {
        max_terms: 100,
        max_total_degree: 20,
        max_pow_exp: 4,
    };

    // Convert numerator and denominator to polynomials
    let p_num = multipoly_from_expr(ctx, num, &budget).ok()?;
    let p_den = multipoly_from_expr(ctx, den, &budget).ok()?;

    // Get variable index in polynomial
    // If var not in poly, it's constant wrt var (degree 0)
    let var_idx_num = p_num.var_index(var_name);
    let var_idx_den = p_den.var_index(var_name);

    // If neither contains the variable, constant rule handles it
    if var_idx_num.is_none() && var_idx_den.is_none() {
        return None; // Let constant rule handle it
    }

    // Check for zero denominator polynomial
    if p_den.is_zero() {
        return None; // Division by zero - don't handle here
    }

    // Get degrees
    let deg_p = var_idx_num.map(|idx| p_num.degree_in(idx)).unwrap_or(0);
    let deg_q = var_idx_den.map(|idx| p_den.degree_in(idx)).unwrap_or(0);

    // Get leading coefficients
    let lc_p = var_idx_num
        .map(|idx| p_num.leading_coeff_in(idx))
        .unwrap_or_else(|| p_num.clone());
    let lc_q = var_idx_den
        .map(|idx| p_den.leading_coeff_in(idx))
        .unwrap_or_else(|| p_den.clone());

    // Both leading coefficients must be numeric constants
    let lc_p_val = lc_p.constant_value()?;
    let lc_q_val = lc_q.constant_value()?;

    // Case 1: deg(P) < deg(Q) -> 0
    if deg_p < deg_q {
        return Some(ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(0)))));
    }

    // Case 2: deg(P) = deg(Q) -> lc(P)/lc(Q)
    if deg_p == deg_q {
        let ratio = &lc_p_val / &lc_q_val;
        return Some(ctx.add(Expr::Number(ratio)));
    }

    // Case 3: deg(P) > deg(Q) -> ±∞
    // Sign = sign(lc_p/lc_q) * sign(x^k) where k = deg_p - deg_q
    let k = deg_p - deg_q;
    let ratio = &lc_p_val / &lc_q_val;

    // Determine sign of ratio
    let ratio_positive = ratio.is_positive();

    // Determine sign of x^k for the approach
    let xk_positive = match approach {
        InfSign::Pos => true,
        InfSign::Neg => k % 2 == 0, // x^even is positive, x^odd is negative
    };

    // Combined sign: positive if both same, negative if different
    let result_positive = ratio_positive == xk_positive;
    let sign = if result_positive {
        InfSign::Pos
    } else {
        InfSign::Neg
    };

    Some(mk_infinity(ctx, sign))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    fn parse_expr(ctx: &mut Context, s: &str) -> ExprId {
        parse(s, ctx).expect("parse failed")
    }

    #[test]
    fn depends_on_detects_simple_variable() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x + 1");
        let x = parse_expr(&mut ctx, "x");
        assert!(depends_on(&ctx, expr, x));
    }

    #[test]
    fn depends_on_rejects_constant_expression() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "5 + pi");
        let x = parse_expr(&mut ctx, "x");
        assert!(!depends_on(&ctx, expr, x));
    }

    #[test]
    fn parse_pow_int_extracts_integer_exponent() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x^3");
        let (_, n) = parse_pow_int(&ctx, expr).expect("power");
        assert_eq!(n, 3);
    }

    #[test]
    fn limit_sign_handles_neg_infinity_parity() {
        assert_eq!(limit_sign(InfSign::Pos, 7), InfSign::Pos);
        assert_eq!(limit_sign(InfSign::Neg, 2), InfSign::Pos);
        assert_eq!(limit_sign(InfSign::Neg, 3), InfSign::Neg);
    }

    #[test]
    fn mk_limit_builds_limit_call_with_signed_infinity_symbol() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x^2");
        let var = parse_expr(&mut ctx, "x");
        let lim = mk_limit(&mut ctx, expr, var, InfSign::Neg);

        let Expr::Function(_fn_id, args) = ctx.get(lim) else {
            panic!("expected limit function call");
        };
        assert_eq!(args.len(), 3);
        assert_eq!(args[0], expr);
        assert_eq!(args[1], var);

        let approach = args[2];
        match ctx.get(approach) {
            Expr::Neg(inner) => {
                assert!(matches!(
                    ctx.get(*inner),
                    Expr::Constant(Constant::Infinity)
                ));
            }
            _ => panic!("expected negative infinity argument"),
        }
    }

    #[test]
    fn rational_poly_limit_handles_equal_and_higher_degree_cases() {
        let mut ctx = Context::new();
        let x = parse_expr(&mut ctx, "x");

        let equal = parse_expr(&mut ctx, "(3*x^2 + 1)/(6*x^2 - 5)");
        let higher = parse_expr(&mut ctx, "(2*x^3)/(x^2+1)");

        let equal_out = rational_poly_limit(&mut ctx, equal, x, InfSign::Pos).expect("equal");
        let higher_out = rational_poly_limit(&mut ctx, higher, x, InfSign::Neg).expect("higher");

        assert!(matches!(ctx.get(equal_out), Expr::Number(_)));
        assert!(matches!(ctx.get(higher_out), Expr::Neg(_)));
    }
}
