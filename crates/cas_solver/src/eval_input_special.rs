use cas_api_models::EvalLimitApproach;
use cas_ast::{Equation, ExprId};
use cas_math::limit_types::Approach;

pub(crate) fn map_limit_approach(approach: EvalLimitApproach) -> Approach {
    match approach {
        EvalLimitApproach::PosInfinity => Approach::PosInfinity,
        EvalLimitApproach::NegInfinity => Approach::NegInfinity,
    }
}

pub(crate) fn parse_solve_input_for_eval_request(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<(ExprId, Option<Equation>), String> {
    let stmt = crate::parse_statement_or_session_ref(ctx, input)?;
    let (parsed, original_equation) = match stmt {
        cas_parser::Statement::Equation(eq) => (ctx.call("Equal", vec![eq.lhs, eq.rhs]), Some(eq)),
        cas_parser::Statement::Expression(expr) => {
            let zero = ctx.num(0);
            (ctx.call("Equal", vec![expr, zero]), None)
        }
    };
    Ok((parsed, original_equation))
}
