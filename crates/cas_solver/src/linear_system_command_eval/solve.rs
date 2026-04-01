use cas_ast::Context;
use cas_ast::ExprId;

use crate::linear_system_command_parse::LinearSystemSpec;

pub(super) fn solve_linear_system_parts(
    ctx: &Context,
    exprs: &[ExprId],
    vars: &[String],
) -> Result<crate::LinSolveResult, crate::LinearSystemError> {
    let n = vars.len();
    if exprs.len() != n {
        return Err(crate::LinearSystemError::NotLinear(
            "equation/variable count mismatch".to_string(),
        ));
    }

    match n {
        2 => {
            let (x, y) =
                match crate::solve_2x2_linear_system(ctx, exprs[0], exprs[1], &vars[0], &vars[1]) {
                    Ok(pair) => pair,
                    Err(crate::LinearSystemError::InfiniteSolutions) => {
                        return Ok(crate::LinSolveResult::Infinite);
                    }
                    Err(crate::LinearSystemError::NoSolution) => {
                        return Ok(crate::LinSolveResult::Inconsistent);
                    }
                    Err(e) => return Err(e),
                };
            Ok(crate::LinSolveResult::Unique(vec![x, y]))
        }
        3 => {
            let (x, y, z) = match crate::solve_3x3_linear_system(
                ctx, exprs[0], exprs[1], exprs[2], &vars[0], &vars[1], &vars[2],
            ) {
                Ok(triple) => triple,
                Err(crate::LinearSystemError::InfiniteSolutions) => {
                    return Ok(crate::LinSolveResult::Infinite);
                }
                Err(crate::LinearSystemError::NoSolution) => {
                    return Ok(crate::LinSolveResult::Inconsistent);
                }
                Err(e) => return Err(e),
            };
            Ok(crate::LinSolveResult::Unique(vec![x, y, z]))
        }
        _ => {
            let var_refs: Vec<&str> = vars.iter().map(String::as_str).collect();
            crate::solve_nxn_linear_system(ctx, exprs, &var_refs)
        }
    }
}

pub(super) fn solve_linear_system_spec(
    ctx: &Context,
    spec: &LinearSystemSpec,
) -> Result<crate::LinSolveResult, crate::LinearSystemError> {
    solve_linear_system_parts(ctx, &spec.exprs, &spec.vars)
}
