use cas_ast::Context;

use crate::linear_system_parse::parse_linear_system_spec;
use crate::linear_system_types::{
    LinearSystemCommandEvalError, LinearSystemCommandEvalOutput, LinearSystemSpec,
};

fn solve_linear_system_spec(
    ctx: &Context,
    spec: &LinearSystemSpec,
) -> Result<cas_solver::LinSolveResult, cas_solver::LinearSystemError> {
    let n = spec.vars.len();
    if spec.exprs.len() != n {
        return Err(cas_solver::LinearSystemError::NotLinear(
            "equation/variable count mismatch".to_string(),
        ));
    }

    match n {
        2 => {
            let (x, y) = match cas_solver::solve_2x2_linear_system(
                ctx,
                spec.exprs[0],
                spec.exprs[1],
                &spec.vars[0],
                &spec.vars[1],
            ) {
                Ok(pair) => pair,
                Err(cas_solver::LinearSystemError::InfiniteSolutions) => {
                    return Ok(cas_solver::LinSolveResult::Infinite);
                }
                Err(cas_solver::LinearSystemError::NoSolution) => {
                    return Ok(cas_solver::LinSolveResult::Inconsistent);
                }
                Err(e) => return Err(e),
            };
            Ok(cas_solver::LinSolveResult::Unique(vec![x, y]))
        }
        3 => {
            let (x, y, z) = match cas_solver::solve_3x3_linear_system(
                ctx,
                spec.exprs[0],
                spec.exprs[1],
                spec.exprs[2],
                &spec.vars[0],
                &spec.vars[1],
                &spec.vars[2],
            ) {
                Ok(triple) => triple,
                Err(cas_solver::LinearSystemError::InfiniteSolutions) => {
                    return Ok(cas_solver::LinSolveResult::Infinite);
                }
                Err(cas_solver::LinearSystemError::NoSolution) => {
                    return Ok(cas_solver::LinSolveResult::Inconsistent);
                }
                Err(e) => return Err(e),
            };
            Ok(cas_solver::LinSolveResult::Unique(vec![x, y, z]))
        }
        _ => {
            let var_refs: Vec<&str> = spec.vars.iter().map(String::as_str).collect();
            cas_solver::solve_nxn_linear_system(ctx, &spec.exprs, &var_refs)
        }
    }
}

pub(crate) fn evaluate_linear_system_command_input(
    ctx: &mut Context,
    input: &str,
) -> Result<LinearSystemCommandEvalOutput, LinearSystemCommandEvalError> {
    let spec = parse_linear_system_spec(ctx, input).map_err(LinearSystemCommandEvalError::Parse)?;
    let result =
        solve_linear_system_spec(ctx, &spec).map_err(LinearSystemCommandEvalError::Solve)?;
    Ok(LinearSystemCommandEvalOutput {
        vars: spec.vars,
        result,
    })
}
