use cas_api_models::EvalJsonLimitApproach;
use cas_ast::ExprId;

pub(crate) fn map_limit_approach(approach: EvalJsonLimitApproach) -> cas_solver::Approach {
    match approach {
        EvalJsonLimitApproach::PosInfinity => cas_solver::Approach::PosInfinity,
        EvalJsonLimitApproach::NegInfinity => cas_solver::Approach::NegInfinity,
    }
}

pub(crate) fn parse_solve_input_as_equation_expr(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<ExprId, String> {
    let stmt = crate::input_parse_common::parse_statement_or_session_ref(ctx, input)?;
    let parsed = match stmt {
        cas_parser::Statement::Equation(eq) => ctx.call("Equal", vec![eq.lhs, eq.rhs]),
        cas_parser::Statement::Expression(expr) => {
            let zero = ctx.num(0);
            ctx.call("Equal", vec![expr, zero])
        }
    };
    Ok(parsed)
}
