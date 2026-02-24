mod arithmetic;
mod functions;
mod power;

use crate::engine::Simplifier;
use crate::solver::{medium_step, SolveStep, SolverOptions, MAX_SOLVE_DEPTH, SOLVE_DEPTH};
use cas_ast::{Expr, ExprId, RelOp, SolutionSet};
use cas_solver_core::solve_outcome::{
    plan_negated_lhs_isolation_step, residual_solution_set, resolve_isolated_variable_outcome,
    solve_term_isolation_rewrite_pipeline_with_item, IsolatedVariableOutcome,
};

use crate::error::CasError;

pub(crate) fn isolate(
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    ctx: &super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    // Check recursion depth
    let current_depth = SOLVE_DEPTH.with(|d| *d.borrow());
    if current_depth > MAX_SOLVE_DEPTH {
        return Err(CasError::SolverError(
            "Maximum solver recursion depth exceeded in isolation.".to_string(),
        ));
    }

    let steps = Vec::new();

    let lhs_expr = simplifier.context.get(lhs).clone();

    match lhs_expr {
        Expr::Variable(sym_id) if simplifier.context.sym_name(sym_id) == var => {
            let sim_rhs = simplifier.simplify(rhs).0;
            let solved = match resolve_isolated_variable_outcome(
                &mut simplifier.context,
                sim_rhs,
                op,
                var,
            ) {
                IsolatedVariableOutcome::Solved(set) => (set, Vec::new()),
                IsolatedVariableOutcome::ContainsTargetVariable => {
                    if let Some((solution_set, steps)) =
                        crate::solver::linear_collect::try_linear_collect(lhs, rhs, var, simplifier)
                    {
                        (solution_set, steps)
                    } else if let Some((solution_set, steps)) =
                        crate::solver::linear_collect::try_linear_collect_v2(
                            lhs, rhs, var, simplifier,
                        )
                    {
                        (solution_set, steps)
                    } else {
                        (
                            residual_solution_set(&mut simplifier.context, lhs, rhs, var),
                            Vec::new(),
                        )
                    }
                }
            };
            prepend_steps(solved, steps)
        }
        Expr::Add(l, r) => {
            arithmetic::isolate_add(lhs, l, r, rhs, op, var, simplifier, opts, steps, ctx)
        }
        Expr::Sub(l, r) => {
            arithmetic::isolate_sub(l, r, rhs, op, var, simplifier, opts, steps, ctx)
        }
        Expr::Mul(l, r) => {
            arithmetic::isolate_mul(lhs, l, r, rhs, op, var, simplifier, opts, steps, ctx)
        }
        Expr::Div(l, r) => {
            arithmetic::isolate_div(lhs, l, r, rhs, op, var, simplifier, opts, steps, ctx)
        }
        Expr::Pow(b, e) => {
            power::isolate_pow(lhs, b, e, rhs, op, var, simplifier, opts, steps, ctx)
        }
        Expr::Function(fn_id, args) => {
            functions::isolate_function(fn_id, args, rhs, op, var, simplifier, opts, steps, ctx)
        }
        Expr::Neg(inner) => {
            let include_item = simplifier.collect_steps();
            let rewrite = plan_negated_lhs_isolation_step(&mut simplifier.context, inner, rhs, op);
            let solved = solve_term_isolation_rewrite_pipeline_with_item(
                rewrite,
                include_item,
                |equation| {
                    isolate(
                        equation.lhs,
                        equation.rhs,
                        equation.op,
                        var,
                        simplifier,
                        opts,
                        ctx,
                    )
                },
                |item| medium_step(item.description, item.equation),
            )?;
            prepend_steps((solved.solution_set, solved.steps), steps)
        }
        _ => Err(CasError::IsolationError(
            var.to_string(),
            format!("Cannot isolate from {:?}", lhs_expr),
        )),
    }
}

// =============================================================================
// Helpers (used by submodules via `super::`)
// =============================================================================

pub(crate) fn prepend_steps(
    (set, mut res_steps): (SolutionSet, Vec<SolveStep>),
    mut steps: Vec<SolveStep>,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    steps.append(&mut res_steps);
    Ok((set, steps))
}
