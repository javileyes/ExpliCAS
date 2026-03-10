use cas_api_models::EvalJsonLimitApproach;
use cas_ast::ExprId;

pub(crate) fn map_limit_approach(approach: EvalJsonLimitApproach) -> crate::Approach {
    match approach {
        EvalJsonLimitApproach::PosInfinity => crate::Approach::PosInfinity,
        EvalJsonLimitApproach::NegInfinity => crate::Approach::NegInfinity,
    }
}

pub(crate) fn parse_solve_input_for_eval_request(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<ExprId, String> {
    let stmt = crate::parse_statement_or_session_ref(ctx, input)?;
    let parsed = match stmt {
        cas_parser::Statement::Equation(eq) => ctx.call("Equal", vec![eq.lhs, eq.rhs]),
        cas_parser::Statement::Expression(expr) => {
            let zero = ctx.num(0);
            ctx.call("Equal", vec![expr, zero])
        }
    };
    Ok(parsed)
}
