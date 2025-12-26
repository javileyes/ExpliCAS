//! Helper functions for limit computation.

use crate::rules::infinity::{mk_infinity, InfSign};
use cas_ast::{Context, Expr, ExprId};

use super::types::Approach;

/// Check if an expression depends on a variable.
///
/// Uses iterative traversal (stack-safe).
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
            let n = crate::helpers::get_integer(ctx, *exp)?;
            Some((*base, n))
        }
        _ => None,
    }
}

/// Create a residual limit expression: `limit(expr, var, approach_symbol)`.
pub fn mk_limit(ctx: &mut Context, expr: ExprId, var: ExprId, approach: Approach) -> ExprId {
    let approach_sym = match approach {
        Approach::PosInfinity => ctx.add(Expr::Constant(cas_ast::Constant::Infinity)),
        Approach::NegInfinity => {
            let inf = ctx.add(Expr::Constant(cas_ast::Constant::Infinity));
            ctx.add(Expr::Neg(inf))
        }
    };
    ctx.add(Expr::Function(
        "limit".into(),
        vec![expr, var, approach_sym],
    ))
}

/// Determine the sign of infinity based on approach and power parity.
pub fn limit_sign(approach: Approach, power: i64) -> InfSign {
    match approach {
        Approach::PosInfinity => InfSign::Pos,
        Approach::NegInfinity => {
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
    fn test_depends_on_simple_variable() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x + 1");
        let x = parse_expr(&mut ctx, "x");

        assert!(depends_on(&ctx, expr, x), "x+1 should depend on x");
    }

    #[test]
    fn test_depends_on_constant() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "5 + pi");
        let x = parse_expr(&mut ctx, "x");

        assert!(!depends_on(&ctx, expr, x), "5+pi should not depend on x");
    }

    #[test]
    fn test_parse_pow_int() {
        let mut ctx = Context::new();
        let expr = parse_expr(&mut ctx, "x^3");

        let result = parse_pow_int(&ctx, expr);
        assert!(result.is_some());
        let (_, n) = result.unwrap();
        assert_eq!(n, 3);
    }
}
