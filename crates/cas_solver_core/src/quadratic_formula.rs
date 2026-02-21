use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;

/// Build `sqrt(radicand)` as `radicand^(1/2)` in AST form.
pub fn sqrt_expr(ctx: &mut Context, radicand: ExprId) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let half = ctx.add(Expr::Div(one, two));
    ctx.add(Expr::Pow(radicand, half))
}

/// Build both quadratic-formula roots from `a`, `b`, and a precomputed `sqrt(delta)`.
///
/// Returns `(x1, x2)` where:
/// - `x1 = (-b - sqrt(delta)) / (2a)`
/// - `x2 = (-b + sqrt(delta)) / (2a)`
pub fn roots_from_a_b_and_sqrt(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    sqrt_delta: ExprId,
) -> (ExprId, ExprId) {
    let neg_b = ctx.add(Expr::Neg(b));
    let two = ctx.num(2);
    let two_a = ctx.add(Expr::Mul(two, a));

    let num1 = ctx.add(Expr::Sub(neg_b, sqrt_delta));
    let x1 = ctx.add(Expr::Div(num1, two_a));

    let num2 = ctx.add(Expr::Add(neg_b, sqrt_delta));
    let x2 = ctx.add(Expr::Div(num2, two_a));

    (x1, x2)
}

/// Build both quadratic-formula roots from `a`, `b`, and `delta`.
pub fn roots_from_a_b_delta(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    delta: ExprId,
) -> (ExprId, ExprId) {
    let sqrt_delta = sqrt_expr(ctx, delta);
    roots_from_a_b_and_sqrt(ctx, a, b, sqrt_delta)
}

/// Compute the quadratic discriminant `b^2 - 4ac`.
pub fn discriminant(a: &BigRational, b: &BigRational, c: &BigRational) -> BigRational {
    b.clone() * b.clone() - BigRational::from_integer(4.into()) * a.clone() * c.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Expr;

    #[test]
    fn test_sqrt_expr_builds_half_power() {
        let mut ctx = Context::new();
        let d = ctx.num(5);
        let s = sqrt_expr(&mut ctx, d);
        match ctx.get(s) {
            Expr::Pow(base, exp) => {
                assert_eq!(*base, d);
                match ctx.get(*exp) {
                    Expr::Div(n, m) => {
                        assert!(matches!(ctx.get(*n), Expr::Number(_)));
                        assert!(matches!(ctx.get(*m), Expr::Number(_)));
                    }
                    other => panic!("Expected Div exponent, got {:?}", other),
                }
            }
            other => panic!("Expected Pow, got {:?}", other),
        }
    }

    #[test]
    fn test_roots_from_a_b_delta_builds_two_divisions() {
        let mut ctx = Context::new();
        let a = ctx.num(2);
        let b = ctx.num(3);
        let d = ctx.num(1);
        let (x1, x2) = roots_from_a_b_delta(&mut ctx, a, b, d);

        assert!(matches!(ctx.get(x1), Expr::Div(_, _)));
        assert!(matches!(ctx.get(x2), Expr::Div(_, _)));
    }

    #[test]
    fn test_discriminant() {
        let a = BigRational::from_integer(1.into());
        let b = BigRational::from_integer(3.into());
        let c = BigRational::from_integer(2.into());
        let d = discriminant(&a, &b, &c);
        assert_eq!(d, BigRational::from_integer(1.into()));
    }
}
