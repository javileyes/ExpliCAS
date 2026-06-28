use cas_api_models::EvalLimitApproach;
use cas_ast::{Equation, ExprId};
use cas_math::limit_types::{Approach, FiniteLimitSide};

pub(crate) fn map_limit_approach(
    ctx: &mut cas_ast::Context,
    approach: EvalLimitApproach,
) -> Result<Approach, String> {
    match approach {
        EvalLimitApproach::PosInfinity => Ok(Approach::PosInfinity),
        EvalLimitApproach::NegInfinity => Ok(Approach::NegInfinity),
        EvalLimitApproach::Finite(point) => {
            let parsed = cas_parser::parse(&point, ctx)
                .map_err(|e| format!("Parse error in limit approach: {e}"))?;
            Ok(Approach::Finite(parsed))
        }
        EvalLimitApproach::FiniteFromLeft(point) => {
            let parsed = cas_parser::parse(&point, ctx)
                .map_err(|e| format!("Parse error in limit approach: {e}"))?;
            Ok(Approach::FiniteOneSided(parsed, FiniteLimitSide::Left))
        }
        EvalLimitApproach::FiniteFromRight(point) => {
            let parsed = cas_parser::parse(&point, ctx)
                .map_err(|e| format!("Parse error in limit approach: {e}"))?;
            Ok(Approach::FiniteOneSided(parsed, FiniteLimitSide::Right))
        }
    }
}

pub(crate) fn parse_solve_input_for_eval_request(
    ctx: &mut cas_ast::Context,
    input: &str,
) -> Result<(ExprId, Option<Equation>), String> {
    let stmt = crate::parse_statement_or_session_ref(ctx, input)?;
    let (parsed, original_equation) = match stmt {
        cas_parser::Statement::Equation(eq) => (
            ctx.call(eq.op.builtin_name(), vec![eq.lhs, eq.rhs]),
            Some(eq),
        ),
        cas_parser::Statement::Expression(expr) => {
            let zero = ctx.num(0);
            (ctx.call("Equal", vec![expr, zero]), None)
        }
    };
    Ok((parsed, original_equation))
}
