//! Structural expression predicates used across algebra rules.

use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::Signed;

/// Check if expression contains any function call.
///
/// Note: for powers, this follows existing engine behavior and only descends
/// into the base expression.
pub fn contains_function(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Function(_, _) => true,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            contains_function(ctx, *l) || contains_function(ctx, *r)
        }
        Expr::Neg(e) | Expr::Hold(e) | Expr::Pow(e, _) => contains_function(ctx, *e),
        _ => false,
    }
}

/// Check if expression is a pure constant expression (no variables/functions).
pub fn is_constant_expr(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(_) | Expr::Constant(_) => true,
        Expr::Neg(inner) => is_constant_expr(ctx, *inner),
        Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Add(l, r) | Expr::Sub(l, r) => {
            is_constant_expr(ctx, *l) && is_constant_expr(ctx, *r)
        }
        Expr::Pow(base, exp) => is_constant_expr(ctx, *base) && is_constant_expr(ctx, *exp),
        _ => false,
    }
}

/// Check if expression structurally contains a root form.
///
/// Recognizes:
/// - `sqrt(...)`
/// - powers with fractional exponent in `(0, 1)` magnitude
/// - powers with explicit `Div(_, _)` exponents
pub fn contains_root_term(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Function(fn_id, _) if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) => true,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            contains_root_term(ctx, *l) || contains_root_term(ctx, *r)
        }
        Expr::Neg(e) | Expr::Hold(e) => contains_root_term(ctx, *e),
        Expr::Pow(_, exp) => match ctx.get(*exp) {
            Expr::Number(n) => !n.is_integer() && n.abs() < BigRational::from_integer(1.into()),
            Expr::Div(_, _) => true,
            _ => false,
        },
        _ => false,
    }
}

/// Check if denominator is trivially equal to 1.
pub fn is_trivial_denom_one(ctx: &Context, d: ExprId) -> bool {
    use num_traits::One;
    matches!(ctx.get(d), Expr::Number(n) if n.is_one())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn contains_function_detects_calls() {
        let mut ctx = Context::new();
        let expr = parse("sin(x)+1", &mut ctx).expect("parse");
        assert!(contains_function(&ctx, expr));
    }

    #[test]
    fn constant_expr_detection() {
        let mut ctx = Context::new();
        let c = parse("2*pi/3", &mut ctx).expect("parse constant");
        let x = parse("2*x", &mut ctx).expect("parse non-constant");
        assert!(is_constant_expr(&ctx, c));
        assert!(!is_constant_expr(&ctx, x));
    }

    #[test]
    fn root_detection() {
        let mut ctx = Context::new();
        let sqrt_expr = parse("sqrt(x)", &mut ctx).expect("parse sqrt");
        let plain = parse("x^2", &mut ctx).expect("parse plain");
        assert!(contains_root_term(&ctx, sqrt_expr));
        assert!(!contains_root_term(&ctx, plain));
    }
}
