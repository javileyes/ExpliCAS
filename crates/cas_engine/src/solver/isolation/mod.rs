mod arithmetic;
mod functions;
mod power;

use crate::engine::Simplifier;
use crate::solver::{medium_step, SolveStep, SolverOptions, MAX_SOLVE_DEPTH, SOLVE_DEPTH};
use cas_ast::{Expr, ExprId, RelOp, SolutionSet};
use cas_solver_core::solve_outcome::{
    plan_negated_lhs_isolation_step, residual_solution_set, resolve_isolated_variable_outcome,
    solve_isolated_variable_lhs_with_resolver_with_state,
    solve_negated_lhs_isolation_plan_with_and_merge_with_existing_steps,
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
    let current_depth = SOLVE_DEPTH.with(|d| d.get());
    if current_depth > MAX_SOLVE_DEPTH {
        return Err(CasError::SolverError(
            "Maximum solver recursion depth exceeded in isolation.".to_string(),
        ));
    }

    let steps = Vec::new();

    let lhs_expr = simplifier.context.get(lhs).clone();

    match lhs_expr {
        Expr::Variable(sym_id) if simplifier.context.sym_name(sym_id) == var => {
            let solved = solve_isolated_variable_lhs_with_resolver_with_state(
                simplifier,
                lhs,
                rhs,
                op,
                var,
                |simplifier, sim_rhs, rel_op, solve_var| {
                    resolve_isolated_variable_outcome(
                        &mut simplifier.context,
                        sim_rhs,
                        rel_op,
                        solve_var,
                    )
                },
                |simplifier, expr| simplifier.simplify(expr).0,
                |simplifier, solve_lhs, solve_rhs, solve_var| {
                    crate::solver::linear_collect::try_linear_collect(
                        solve_lhs, solve_rhs, solve_var, simplifier,
                    )
                },
                |simplifier, solve_lhs, solve_rhs, solve_var| {
                    crate::solver::linear_collect::try_linear_collect_v2(
                        solve_lhs, solve_rhs, solve_var, simplifier,
                    )
                },
                |simplifier, solve_lhs, solve_rhs, solve_var| {
                    residual_solution_set(&mut simplifier.context, solve_lhs, solve_rhs, solve_var)
                },
            );
            Ok(solved)
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
            let rewrite =
                plan_negated_lhs_isolation_step(&mut simplifier.context, inner, rhs, op.clone());
            let include_item = simplifier.collect_steps();
            solve_negated_lhs_isolation_plan_with_and_merge_with_existing_steps(
                rewrite,
                var,
                include_item,
                steps,
                |equation, solve_var| {
                    isolate(
                        equation.lhs,
                        equation.rhs,
                        equation.op,
                        solve_var,
                        simplifier,
                        opts,
                        ctx,
                    )
                },
                |item| medium_step(item.description, item.equation),
            )
        }
        _ => Err(CasError::IsolationError(
            var.to_string(),
            format!("Cannot isolate from {:?}", lhs_expr),
        )),
    }
}
