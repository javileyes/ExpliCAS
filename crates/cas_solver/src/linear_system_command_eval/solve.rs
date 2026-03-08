use cas_ast::Context;

use crate::linear_system_command_types::LinearSystemSpec;

pub(super) fn solve_linear_system_spec(
    ctx: &Context,
    spec: &LinearSystemSpec,
) -> Result<crate::LinSolveResult, crate::LinearSystemError> {
    let n = spec.vars.len();
    if spec.exprs.len() != n {
        return Err(crate::LinearSystemError::NotLinear(
            "equation/variable count mismatch".to_string(),
        ));
    }

    match n {
        2 => {
            let (x, y) = match crate::solve_2x2_linear_system(
                ctx,
                spec.exprs[0],
                spec.exprs[1],
                &spec.vars[0],
                &spec.vars[1],
            ) {
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
                ctx,
                spec.exprs[0],
                spec.exprs[1],
                spec.exprs[2],
                &spec.vars[0],
                &spec.vars[1],
                &spec.vars[2],
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
            let var_refs: Vec<&str> = spec.vars.iter().map(String::as_str).collect();
            crate::solve_nxn_linear_system(ctx, &spec.exprs, &var_refs)
        }
    }
}
