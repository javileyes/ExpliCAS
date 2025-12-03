#[cfg(test)]
mod latex_tests {
    use super::*;

    #[test]
    fn test_number_times_number() {
        let mut ctx = Context::new();
        let five = ctx.num(5);
        let expr = ctx.add(Expr::Mul(five, five));

        let latex = LaTeXExpr {
            context: &ctx,
            id: expr,
        };
        assert_eq!(latex.to_latex(), "5\\cdot 5");
    }

    #[test]
    fn test_number_times_variable() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let x = ctx.var("x");
        let expr = ctx.add(Expr::Mul(two, x));

        let latex = LaTeXExpr {
            context: &ctx,
            id: expr,
        };
        assert_eq!(latex.to_latex(), "2x");
    }

    #[test]
    fn test_variable_times_number() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Mul(x, two));

        let latex = LaTeXExpr {
            context: &ctx,
            id: expr,
        };
        // x*2 can stay as simple multiplication without confusion
        assert_eq!(latex.to_latex(), "x2");
    }

    #[test]
    fn test_complex_times_number() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_plus_1 = ctx.add(Expr::Add(x, one));
        let two = ctx.num(2);
        let expr = ctx.add(Expr::Mul(x_plus_1, two));

        let latex = LaTeXExpr {
            context: &ctx,
            id: expr,
        };
        // (x+1)*2 needs explicit multiplication
        assert_eq!(latex.to_latex(), "(x + 1)\\cdot 2");
    }
}
