use crate::infinity_support::{mk_infinity, InfSign};
use cas_ast::{Constant, Context, Expr, ExprId};

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
}
