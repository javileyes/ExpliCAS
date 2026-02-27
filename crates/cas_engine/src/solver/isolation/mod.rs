mod arithmetic;
mod functions;
mod power;

use crate::engine::Simplifier;
use crate::solver::{medium_step, SolveStep, SolverOptions, MAX_SOLVE_DEPTH};
use cas_ast::{ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_dispatch::execute_isolation_dispatch_with_default_isolated_and_negated_entries_for_var_with_state;

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
    // Check solve recursion depth tracked in SolveCtx.
    if ctx.depth() > MAX_SOLVE_DEPTH {
        return Err(CasError::SolverError(
            "Maximum solver recursion depth exceeded in isolation.".to_string(),
        ));
    }

    execute_isolation_dispatch_with_default_isolated_and_negated_entries_for_var_with_state(
        simplifier,
        lhs,
        rhs,
        op.clone(),
        var,
        |simplifier| &simplifier.context,
        |simplifier| &mut simplifier.context,
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
        |simplifier, left, right| {
            arithmetic::isolate_add(
                lhs,
                left,
                right,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                Vec::new(),
                ctx,
            )
        },
        |simplifier, left, right| {
            arithmetic::isolate_sub(
                left,
                right,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                Vec::new(),
                ctx,
            )
        },
        |simplifier, left, right| {
            arithmetic::isolate_mul(
                lhs,
                left,
                right,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                Vec::new(),
                ctx,
            )
        },
        |simplifier, left, right| {
            arithmetic::isolate_div(
                lhs,
                left,
                right,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                Vec::new(),
                ctx,
            )
        },
        |simplifier, base, exponent| {
            power::isolate_pow(
                lhs,
                base,
                exponent,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                Vec::new(),
                ctx,
            )
        },
        |simplifier, fn_id, args| {
            functions::isolate_function(
                fn_id,
                args,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                Vec::new(),
                ctx,
            )
        },
        |simplifier| simplifier.collect_steps(),
        |simplifier, equation, solve_var| {
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
        |_simplifier, lhs_expr| {
            CasError::IsolationError(
                var.to_string(),
                format!("Cannot isolate from {:?}", lhs_expr),
            )
        },
    )
}
