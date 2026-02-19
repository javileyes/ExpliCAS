//! Structural expression predicates used across algebra rules.

use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed};

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
        Expr::Pow(base, exp) => {
            let fractional_exp = match ctx.get(*exp) {
                Expr::Number(n) => !n.is_integer() && n.abs() < BigRational::from_integer(1.into()),
                Expr::Div(_, _) => true,
                _ => false,
            };
            fractional_exp || contains_root_term(ctx, *base)
        }
        _ => false,
    }
}

/// Check if denominator is trivially equal to 1.
pub fn is_trivial_denom_one(ctx: &Context, d: ExprId) -> bool {
    matches!(ctx.get(d), Expr::Number(n) if n.is_one())
}

/// Check if expression is numerically equal to `+1`.
pub fn is_one_expr(ctx: &Context, id: ExprId) -> bool {
    matches!(ctx.get(id), Expr::Number(n) if n.is_one())
}

/// Check if expression is numerically equal to `+2`.
pub fn is_two_expr(ctx: &Context, id: ExprId) -> bool {
    matches!(ctx.get(id), Expr::Number(n) if *n == BigRational::from_integer(2.into()))
}

/// Check if expression is numerically equal to `+1/2`.
pub fn is_half_expr(ctx: &Context, id: ExprId) -> bool {
    matches!(ctx.get(id), Expr::Number(n) if *n == BigRational::new(1.into(), 2.into()))
}

/// Check if expression is numerically equal to `-1`.
pub fn is_minus_one_expr(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(n) => n == &BigRational::from_integer((-1).into()),
        Expr::Neg(inner) => is_one_expr(ctx, *inner),
        _ => false,
    }
}

/// Check if expression contains any explicit division node.
pub fn contains_div_term(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Div(_, _) => true,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Pow(l, r) => {
            contains_div_term(ctx, *l) || contains_div_term(ctx, *r)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => contains_div_term(ctx, *inner),
        _ => false,
    }
}

/// Check if expression contains division-like structure.
///
/// Recognizes explicit `Div` plus inverse-power encodings like `x^(-1)`.
pub fn contains_division_like_term(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Div(_, _) => true,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            contains_division_like_term(ctx, *l) || contains_division_like_term(ctx, *r)
        }
        Expr::Pow(base, exp) => {
            exponent_implies_division(ctx, *exp)
                || contains_division_like_term(ctx, *base)
                || contains_division_like_term(ctx, *exp)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => contains_division_like_term(ctx, *inner),
        Expr::Function(_, args) => args.iter().any(|a| contains_division_like_term(ctx, *a)),
        Expr::Matrix { data, .. } => data.iter().any(|e| contains_division_like_term(ctx, *e)),
        _ => false,
    }
}

fn exponent_implies_division(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Neg(_) => true,
        Expr::Number(n) => n.is_negative(),
        _ => false,
    }
}

/// Check if expression contains a function call or any root form.
pub fn contains_function_or_root(ctx: &Context, id: ExprId) -> bool {
    contains_function(ctx, id) || contains_root_term(ctx, id)
}

/// Check if numerator/denominator pair is a constant fraction.
pub fn is_constant_fraction(ctx: &Context, n: ExprId, d: ExprId) -> bool {
    is_constant_expr(ctx, n) && is_constant_expr(ctx, d)
}

/// Check if expression is a small numeric literal in `[-max_abs, max_abs]`.
pub fn is_simple_number_abs_leq(ctx: &Context, id: ExprId, max_abs: i64) -> bool {
    if max_abs < 0 {
        return false;
    }
    match ctx.get(id) {
        Expr::Number(n) => n.abs() <= BigRational::from_integer(max_abs.into()),
        Expr::Neg(inner) => is_simple_number_abs_leq(ctx, *inner, max_abs),
        _ => false,
    }
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
        let root_in_base = parse("(sqrt(x))^2", &mut ctx).expect("parse root in pow base");
        let plain = parse("x^2", &mut ctx).expect("parse plain");
        assert!(contains_root_term(&ctx, sqrt_expr));
        assert!(contains_root_term(&ctx, root_in_base));
        assert!(!contains_root_term(&ctx, plain));
    }

    #[test]
    fn one_minus_one_detection() {
        let mut ctx = Context::new();
        let one = parse("1", &mut ctx).expect("parse one");
        let two = parse("2", &mut ctx).expect("parse two");
        let half = ctx.rational(1, 2);
        let minus_one = parse("-1", &mut ctx).expect("parse minus one");
        assert!(is_one_expr(&ctx, one));
        assert!(is_two_expr(&ctx, two));
        assert!(is_half_expr(&ctx, half));
        assert!(is_minus_one_expr(&ctx, minus_one));
        assert!(!is_minus_one_expr(&ctx, one));
    }

    #[test]
    fn div_and_function_root_detection() {
        let mut ctx = Context::new();
        let with_div = parse("x*(1/y)", &mut ctx).expect("parse div");
        let with_neg_pow = parse("a + b^(-1)", &mut ctx).expect("parse neg pow");
        let with_function_div = parse("sin(1/x)", &mut ctx).expect("parse function div");
        let with_root = parse("a + sqrt(b)", &mut ctx).expect("parse root");
        let plain = parse("x*y", &mut ctx).expect("parse plain");
        assert!(contains_div_term(&ctx, with_div));
        assert!(contains_division_like_term(&ctx, with_neg_pow));
        assert!(contains_division_like_term(&ctx, with_function_div));
        assert!(contains_function_or_root(&ctx, with_root));
        assert!(!contains_division_like_term(&ctx, plain));
        assert!(!contains_div_term(&ctx, plain));
    }

    #[test]
    fn constant_fraction_and_small_number_detection() {
        let mut ctx = Context::new();
        let n = parse("2*pi", &mut ctx).expect("parse n");
        let d = parse("3", &mut ctx).expect("parse d");
        let sym = parse("x", &mut ctx).expect("parse x");
        let minus_two = parse("-2", &mut ctx).expect("parse -2");
        assert!(is_constant_fraction(&ctx, n, d));
        assert!(!is_constant_fraction(&ctx, sym, d));
        assert!(is_simple_number_abs_leq(&ctx, minus_two, 2));
        assert!(!is_simple_number_abs_leq(&ctx, minus_two, 1));
    }
}
