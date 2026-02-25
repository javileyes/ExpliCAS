mod arithmetic;
mod functions;
mod power;

use crate::engine::Simplifier;
use crate::solver::{medium_step, SolveStep, SolverOptions, MAX_SOLVE_DEPTH, SOLVE_DEPTH};
use cas_ast::{Expr, ExprId, RelOp, SolutionSet};
use cas_solver_core::solve_outcome::{
    residual_solution_set, solve_isolated_variable_lhs_with_resolver,
    solve_negated_lhs_isolation_with,
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
            let runtime_cell = std::cell::RefCell::new(simplifier);
            let solved = solve_isolated_variable_lhs_with_resolver(
                lhs,
                rhs,
                op,
                var,
                |sim_rhs, rel_op, solve_var| {
                    let mut simplifier_ref = runtime_cell.borrow_mut();
                    cas_solver_core::solve_outcome::resolve_isolated_variable_outcome(
                        &mut simplifier_ref.context,
                        sim_rhs,
                        rel_op,
                        solve_var,
                    )
                },
                |expr| {
                    let mut simplifier_ref = runtime_cell.borrow_mut();
                    simplifier_ref.simplify(expr).0
                },
                |lhs_expr, rhs_expr, solve_var| {
                    let mut simplifier_ref = runtime_cell.borrow_mut();
                    crate::solver::linear_collect::try_linear_collect(
                        lhs_expr,
                        rhs_expr,
                        solve_var,
                        *simplifier_ref,
                    )
                },
                |lhs_expr, rhs_expr, solve_var| {
                    let mut simplifier_ref = runtime_cell.borrow_mut();
                    crate::solver::linear_collect::try_linear_collect_v2(
                        lhs_expr,
                        rhs_expr,
                        solve_var,
                        *simplifier_ref,
                    )
                },
                |lhs_expr, rhs_expr, solve_var| {
                    let mut simplifier_ref = runtime_cell.borrow_mut();
                    residual_solution_set(
                        &mut simplifier_ref.context,
                        lhs_expr,
                        rhs_expr,
                        solve_var,
                    )
                },
            );
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
            let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
            let solved = solve_negated_lhs_isolation_with(
                || {
                    let mut simplifier_ref = runtime_cell.borrow_mut();
                    cas_solver_core::solve_outcome::plan_negated_lhs_isolation_step(
                        &mut simplifier_ref.context,
                        inner,
                        rhs,
                        op.clone(),
                    )
                },
                var,
                include_item,
                |equation, solve_var| {
                    let mut simplifier_ref = runtime_cell.borrow_mut();
                    isolate(
                        equation.lhs,
                        equation.rhs,
                        equation.op,
                        solve_var,
                        *simplifier_ref,
                        opts,
                        ctx,
                    )
                },
                |item| medium_step(item.description, item.equation),
            )?;
            prepend_steps(solved, steps)
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
