use cas_ast::Context;
use cas_ast::ExprId;

use crate::linear_system_command_parse::LinearSystemSpec;

pub(super) fn solve_linear_system_parts(
    ctx: &mut Context,
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
                    // The unknowns list drives linearity: `a·x` or `+ a` with a
                    // parameter `a` is LINEAR in {x, y} — retry symbolically
                    // before giving up (rational failures keep their message).
                    Err(rational_error) => {
                        return match crate::solve_2x2_symbolic(
                            ctx, exprs[0], exprs[1], &vars[0], &vars[1],
                        ) {
                            Ok(crate::Symbolic2x2Outcome::Unique {
                                values,
                                det_condition,
                            }) => Ok(crate::LinSolveResult::UniqueExpr {
                                values,
                                nonzero_conditions: det_condition.into_iter().collect(),
                            }),
                            Ok(crate::Symbolic2x2Outcome::DegenerateSymbolic) => {
                                Err(crate::LinearSystemError::NotLinear(
                                    "symbolic coefficients with det = 0: \
                                     rank classification is a future rung"
                                        .to_string(),
                                ))
                            }
                            Err(_) => Err(rational_error),
                        };
                    }
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
                // S6: same doctrine as the 2×2 arm — the unknowns list drives
                // linearity; parameters go to the coefficients, Cramer runs
                // over polynomial determinants.
                Err(rational_error) => {
                    return match crate::solve_nxn_symbolic(ctx, exprs, vars) {
                        Ok(crate::Symbolic2x2Outcome::Unique {
                            values,
                            det_condition,
                        }) => Ok(crate::LinSolveResult::UniqueExpr {
                            values,
                            nonzero_conditions: det_condition.into_iter().collect(),
                        }),
                        Ok(crate::Symbolic2x2Outcome::DegenerateSymbolic) => {
                            Err(crate::LinearSystemError::NotLinear(
                                "symbolic coefficients with det = 0: \
                                 rank classification is a future rung"
                                    .to_string(),
                            ))
                        }
                        Err(_) => Err(rational_error),
                    };
                }
            };
            Ok(crate::LinSolveResult::Unique(vec![x, y, z]))
        }
        _ => {
            let var_refs: Vec<&str> = vars.iter().map(String::as_str).collect();
            match crate::solve_nxn_linear_system(ctx, exprs, &var_refs) {
                Ok(result) => Ok(result),
                Err(
                    e @ (crate::LinearSystemError::InfiniteSolutions
                    | crate::LinearSystemError::NoSolution),
                ) => Err(e),
                // S7: the generic symbolic Cramer covers n ≥ 4 too — the
                // cofactor budget is the deliberate guard against blowup.
                Err(rational_error) => match crate::solve_nxn_symbolic(ctx, exprs, vars) {
                    Ok(crate::Symbolic2x2Outcome::Unique {
                        values,
                        det_condition,
                    }) => Ok(crate::LinSolveResult::UniqueExpr {
                        values,
                        nonzero_conditions: det_condition.into_iter().collect(),
                    }),
                    Ok(crate::Symbolic2x2Outcome::DegenerateSymbolic) => {
                        Err(crate::LinearSystemError::NotLinear(
                            "symbolic coefficients with det = 0: \
                             rank classification is a future rung"
                                .to_string(),
                        ))
                    }
                    Err(_) => Err(rational_error),
                },
            }
        }
    }
}

pub(super) fn solve_linear_system_spec(
    ctx: &mut Context,
    spec: &LinearSystemSpec,
) -> Result<crate::LinSolveResult, crate::LinearSystemError> {
    solve_linear_system_parts(ctx, &spec.exprs, &spec.vars)
}
