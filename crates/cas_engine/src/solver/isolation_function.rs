use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{
    context_render_expr, medium_step, simplifier_context, simplifier_context_mut, SolveCtx,
    SolveStep, SolverOptions,
};
use cas_ast::symbol::SymbolId;
use cas_ast::{ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_functions::execute_function_isolation_with_default_kernels_and_unified_step_mapper_for_var_with_state;

use super::isolation::isolate;

/// Handle isolation for `Function(fn_id, args)`: abs, log, ln, exp, sqrt, trig
#[allow(clippy::too_many_arguments)]
pub(super) fn isolate_function(
    fn_id: SymbolId,
    args: Vec<ExprId>,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let include_items = simplifier.collect_steps();
    execute_function_isolation_with_default_kernels_and_unified_step_mapper_for_var_with_state(
        simplifier,
        fn_id,
        &args,
        rhs,
        op,
        var,
        include_items,
        steps,
        simplifier_context,
        simplifier_context_mut,
        context_render_expr,
        |simplifier, lhs_expr, rhs_expr, inner_op| {
            isolate(lhs_expr, rhs_expr, inner_op, var, simplifier, opts, ctx)
        },
        |simplifier, rhs_expr| {
            let (simplified_rhs, sim_steps) = simplifier.simplify(rhs_expr);
            let entries = sim_steps
                .into_iter()
                .map(|step| (step.description, step.after))
                .collect::<Vec<_>>();
            (simplified_rhs, entries)
        },
        medium_step,
        |_simplifier, missing_var| CasError::VariableNotFound(missing_var.to_string()),
        |simplifier, unsupported_fn_id, arity, unsupported_var| {
            CasError::IsolationError(
                unsupported_var.to_string(),
                format!(
                    "Cannot invert function '{}' with {} arguments",
                    simplifier.context.sym_name(unsupported_fn_id),
                    arity
                ),
            )
        },
        |_simplifier, fn_name| CasError::UnknownFunction(fn_name.to_string()),
    )
}
