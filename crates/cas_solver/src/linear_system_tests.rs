#[cfg(test)]
mod tests {
    use crate::linear_system::{solve_2x2_linear_system, solve_nxn_linear_system, LinSolveResult};
    use cas_ast::Expr;
    use num_rational::BigRational;

    #[test]
    fn solve_2x2_linear_system_unique_solution() {
        let mut ctx = cas_ast::Context::new();
        let eq1 = match cas_parser::parse_statement("x+y=3", &mut ctx).expect("eq1 parse") {
            cas_parser::Statement::Equation(eq) => eq,
            _ => panic!("expected equation"),
        };
        let eq2 = match cas_parser::parse_statement("x-y=1", &mut ctx).expect("eq2 parse") {
            cas_parser::Statement::Equation(eq) => eq,
            _ => panic!("expected equation"),
        };

        let expr1 = ctx.add(Expr::Sub(eq1.lhs, eq1.rhs));
        let expr2 = ctx.add(Expr::Sub(eq2.lhs, eq2.rhs));

        let (x, y) = solve_2x2_linear_system(&ctx, expr1, expr2, "x", "y")
            .expect("linear solve should succeed");

        assert_eq!(x, BigRational::from_integer(2.into()));
        assert_eq!(y, BigRational::from_integer(1.into()));
    }

    #[test]
    fn solve_nxn_linear_system_detects_inconsistent() {
        let mut ctx = cas_ast::Context::new();
        let eq1 = match cas_parser::parse_statement("x+y=1", &mut ctx).expect("eq1 parse") {
            cas_parser::Statement::Equation(eq) => eq,
            _ => panic!("expected equation"),
        };
        let eq2 = match cas_parser::parse_statement("x+y=2", &mut ctx).expect("eq2 parse") {
            cas_parser::Statement::Equation(eq) => eq,
            _ => panic!("expected equation"),
        };

        let exprs = vec![
            ctx.add(Expr::Sub(eq1.lhs, eq1.rhs)),
            ctx.add(Expr::Sub(eq2.lhs, eq2.rhs)),
        ];

        let result = solve_nxn_linear_system(&ctx, &exprs, &["x", "y"]).expect("solver result");
        match result {
            LinSolveResult::Inconsistent => {}
            _ => panic!("expected inconsistent system"),
        }
    }
}
