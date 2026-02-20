use cas_ast::{Context, Expr, ExprId};

/// Add two exponent expressions, folding when both are numeric literals.
pub fn add_exp(ctx: &mut Context, e1: ExprId, e2: ExprId) -> ExprId {
    if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(e1), ctx.get(e2)) {
        let sum = n1 + n2;
        ctx.add(Expr::Number(sum))
    } else {
        ctx.add(Expr::Add(e1, e2))
    }
}

/// Multiply two exponent expressions, folding when both are numeric literals.
pub fn mul_exp(ctx: &mut Context, e1: ExprId, e2: ExprId) -> ExprId {
    if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(e1), ctx.get(e2)) {
        let prod = n1 * n2;
        ctx.add(Expr::Number(prod))
    } else {
        crate::build::mul2_raw(ctx, e1, e2)
    }
}

/// Check whether an expression has a numeric factor at top level.
pub fn has_numeric_factor(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(_) => true,
        Expr::Mul(l, r) => {
            matches!(ctx.get(*l), Expr::Number(_)) || matches!(ctx.get(*r), Expr::Number(_))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;
    use num_rational::BigRational;

    #[test]
    fn add_and_mul_fold_numeric_literals() {
        let mut ctx = Context::new();
        let a = ctx.num(2);
        let b = ctx.num(3);

        let add = add_exp(&mut ctx, a, b);
        assert!(
            matches!(ctx.get(add), Expr::Number(n) if *n == BigRational::from_integer(5.into()))
        );

        let mul = mul_exp(&mut ctx, a, b);
        assert!(
            matches!(ctx.get(mul), Expr::Number(n) if *n == BigRational::from_integer(6.into()))
        );
    }

    #[test]
    fn detects_top_level_numeric_factor() {
        let mut ctx = Context::new();
        let with_num = parse("2*x", &mut ctx).expect("parse with_num");
        let without_num = parse("x*y", &mut ctx).expect("parse without_num");

        assert!(has_numeric_factor(&ctx, with_num));
        assert!(!has_numeric_factor(&ctx, without_num));
    }
}
