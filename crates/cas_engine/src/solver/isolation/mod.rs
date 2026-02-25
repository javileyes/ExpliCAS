mod arithmetic;
mod functions;
mod power;

use crate::engine::Simplifier;
use crate::solver::{medium_step, SolveStep, SolverOptions, MAX_SOLVE_DEPTH, SOLVE_DEPTH};
use cas_ast::{Expr, ExprId, RelOp, SolutionSet};
use cas_solver_core::solve_outcome::{
    residual_solution_set, solve_isolated_variable_lhs_with_runtime,
    solve_negated_lhs_isolation_with_runtime, IsolatedVariableRuntime, NegatedLhsIsolationRuntime,
    TermIsolationRewriteExecutionItem,
};

use crate::error::CasError;

struct NegatedIsolationRuntime<'a, 'ctx> {
    simplifier: &'a mut Simplifier,
    opts: SolverOptions,
    solve_ctx: &'ctx super::SolveCtx,
}

struct IsolatedVariableRuntimeAdapter<'a> {
    simplifier: &'a mut Simplifier,
}

impl cas_solver_core::solve_outcome::CircularIsolatedRuntime<SolveStep>
    for IsolatedVariableRuntimeAdapter<'_>
{
    fn try_linear_collect(
        &mut self,
        lhs: ExprId,
        rhs: ExprId,
        var: &str,
    ) -> Option<(SolutionSet, Vec<SolveStep>)> {
        crate::solver::linear_collect::try_linear_collect(lhs, rhs, var, self.simplifier)
    }

    fn try_linear_collect_v2(
        &mut self,
        lhs: ExprId,
        rhs: ExprId,
        var: &str,
    ) -> Option<(SolutionSet, Vec<SolveStep>)> {
        crate::solver::linear_collect::try_linear_collect_v2(lhs, rhs, var, self.simplifier)
    }

    fn residual_solution(&mut self, lhs: ExprId, rhs: ExprId, var: &str) -> SolutionSet {
        residual_solution_set(&mut self.simplifier.context, lhs, rhs, var)
    }
}

impl IsolatedVariableRuntime<SolveStep> for IsolatedVariableRuntimeAdapter<'_> {
    fn context(&mut self) -> &mut cas_ast::Context {
        &mut self.simplifier.context
    }

    fn simplify_rhs(&mut self, rhs: ExprId) -> ExprId {
        self.simplifier.simplify(rhs).0
    }
}

impl NegatedLhsIsolationRuntime<CasError, SolveStep> for NegatedIsolationRuntime<'_, '_> {
    fn context(&mut self) -> &mut cas_ast::Context {
        &mut self.simplifier.context
    }

    fn solve_rewritten(
        &mut self,
        equation: cas_ast::Equation,
        var: &str,
    ) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
        isolate(
            equation.lhs,
            equation.rhs,
            equation.op,
            var,
            self.simplifier,
            self.opts,
            self.solve_ctx,
        )
    }

    fn map_item_to_step(&mut self, item: TermIsolationRewriteExecutionItem) -> SolveStep {
        medium_step(item.description, item.equation)
    }
}

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
            let mut runtime = IsolatedVariableRuntimeAdapter { simplifier };
            let solved = solve_isolated_variable_lhs_with_runtime(lhs, rhs, op, var, &mut runtime);
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
            let mut runtime = NegatedIsolationRuntime {
                simplifier,
                opts,
                solve_ctx: ctx,
            };
            let solved = solve_negated_lhs_isolation_with_runtime(
                inner,
                rhs,
                op,
                var,
                include_item,
                &mut runtime,
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
