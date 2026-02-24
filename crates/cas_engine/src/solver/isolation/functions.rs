use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{SolveStep, SolverOptions};
use cas_ast::symbol::SymbolId;
use cas_ast::{ExprId, RelOp, SolutionSet};
use cas_solver_core::function_inverse::{
    derive_function_isolation_route, execute_unary_inverse_with_runtime,
    solve_unary_inverse_execution_pipeline_with_items_runtime, FunctionIsolationRoute,
    FunctionIsolationRouteError, UnaryInverseRuntime, UnaryInverseSolveExecutionItem,
    UnaryInverseSolveRuntime,
};
use cas_solver_core::isolation_utils::{contains_var, numeric_sign};
use cas_solver_core::log_isolation::{
    plan_log_isolation_step_with_runtime, solve_log_isolation_rewrite_pipeline_with_item,
    LogIsolationExecutionItem, LogIsolationPlanRuntime, LogIsolationRewriteRuntime,
};
use cas_solver_core::solve_outcome::{
    finalize_abs_split_solution_set, plan_abs_isolation, solve_abs_isolation_plan_with_runtime,
    solve_abs_split_pipeline_with_optional_items_runtime, AbsIsolationPlanRuntime,
    AbsIsolationSolved, AbsSplitExecutionItem, AbsSplitRuntime,
};

use super::{isolate, prepend_steps};

struct EngineUnaryInverseRuntime<'a> {
    simplifier: &'a mut Simplifier,
}

impl UnaryInverseRuntime for EngineUnaryInverseRuntime<'_> {
    fn context(&mut self) -> &mut cas_ast::Context {
        &mut self.simplifier.context
    }

    fn simplify_rhs_with_entries(&mut self, rhs: ExprId) -> (ExprId, Vec<(String, ExprId)>) {
        let (simplified_rhs, sim_steps) = self.simplifier.simplify(rhs);
        let entries = sim_steps
            .into_iter()
            .map(|step| (step.description, step.after))
            .collect::<Vec<_>>();
        (simplified_rhs, entries)
    }
}

struct EngineLogIsolationRuntime<'a, 'b> {
    simplifier: &'a mut Simplifier,
    var: &'b str,
    opts: SolverOptions,
    ctx: &'a super::super::SolveCtx,
}

struct EngineLogIsolationPlanRuntime;

impl LogIsolationPlanRuntime for EngineLogIsolationPlanRuntime {
    fn render_expr(&mut self, core_ctx: &cas_ast::Context, expr: ExprId) -> String {
        format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: core_ctx,
                id: expr
            }
        )
    }
}

impl LogIsolationRewriteRuntime<CasError, SolveStep> for EngineLogIsolationRuntime<'_, '_> {
    fn solve_rewritten(
        &mut self,
        equation: &cas_ast::Equation,
    ) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
        isolate(
            equation.lhs,
            equation.rhs,
            equation.op.clone(),
            self.var,
            self.simplifier,
            self.opts,
            self.ctx,
        )
    }

    fn map_item_to_step(&mut self, item: LogIsolationExecutionItem) -> SolveStep {
        SolveStep {
            description: item.description,
            equation_after: item.equation,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        }
    }
}

struct EngineFunctionIsolationRuntime<'a, 'b> {
    simplifier: &'a mut Simplifier,
    opts: SolverOptions,
    solve_ctx: &'b super::super::SolveCtx,
}

struct EngineAbsIsolationPlanRuntime;

impl AbsIsolationPlanRuntime<CasError, cas_ast::Equation, (cas_ast::Equation, cas_ast::Equation)>
    for EngineAbsIsolationPlanRuntime
{
    fn solve_single(&mut self, equation: cas_ast::Equation) -> Result<cas_ast::Equation, CasError> {
        Ok(equation)
    }

    fn solve_split(
        &mut self,
        positive: cas_ast::Equation,
        negative: cas_ast::Equation,
    ) -> Result<(cas_ast::Equation, cas_ast::Equation), CasError> {
        Ok((positive, negative))
    }
}

impl AbsSplitRuntime<CasError, SolveStep> for EngineFunctionIsolationRuntime<'_, '_> {
    fn render_expr(&mut self, expr: ExprId) -> String {
        format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &self.simplifier.context,
                id: expr
            }
        )
    }

    fn solve_branch(
        &mut self,
        equation: &cas_ast::Equation,
        var: &str,
    ) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
        isolate(
            equation.lhs,
            equation.rhs,
            equation.op.clone(),
            var,
            self.simplifier,
            self.opts,
            self.solve_ctx,
        )
    }

    fn map_item_to_step(&mut self, item: AbsSplitExecutionItem) -> SolveStep {
        SolveStep {
            description: item.description().to_string(),
            equation_after: item.equation,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        }
    }
}

impl UnaryInverseSolveRuntime<CasError, SolveStep> for EngineFunctionIsolationRuntime<'_, '_> {
    fn solve_rewritten(
        &mut self,
        lhs: ExprId,
        rhs: ExprId,
        op: RelOp,
        var: &str,
    ) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
        isolate(
            lhs,
            rhs,
            op,
            var,
            self.simplifier,
            self.opts,
            self.solve_ctx,
        )
    }

    fn map_item_to_step(&mut self, item: UnaryInverseSolveExecutionItem) -> SolveStep {
        SolveStep {
            description: item.description().to_string(),
            equation_after: item.equation,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        }
    }
}

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
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    match derive_function_isolation_route(&simplifier.context, fn_id, &args, var) {
        Ok(FunctionIsolationRoute::AbsUnary { arg }) => {
            isolate_abs(arg, rhs, op, var, simplifier, opts, steps, ctx)
        }
        Ok(FunctionIsolationRoute::LogBinary { base, arg }) => {
            isolate_log(base, arg, rhs, op, var, simplifier, opts, steps, ctx)
        }
        Ok(FunctionIsolationRoute::UnaryInvertible { arg }) => {
            isolate_unary_function(fn_id, arg, rhs, op, var, simplifier, opts, steps, ctx)
        }
        Err(FunctionIsolationRouteError::VariableNotFoundInUnaryArg) => {
            Err(CasError::VariableNotFound(var.to_string()))
        }
        Err(FunctionIsolationRouteError::UnsupportedArity) => Err(CasError::IsolationError(
            var.to_string(),
            format!(
                "Cannot invert function '{}' with {} arguments",
                simplifier.context.sym_name(fn_id),
                args.len()
            ),
        )),
    }
}

/// Handle `|A| = RHS` (absolute value isolation)
///
/// Soundness invariant: `|A| = B` requires `B ≥ 0` (absolute values are
/// non-negative). When `B` is a symbolic expression containing the solve
/// variable, we attach `NonNegative(rhs)` as a condition guard rather than
/// attempting to solve a guard inequality recursively.
#[allow(clippy::too_many_arguments)]
fn isolate_abs(
    arg: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let rhs_sign = numeric_sign(&simplifier.context, rhs);
    let abs_plan = plan_abs_isolation(&mut simplifier.context, arg, rhs, op.clone(), rhs_sign);
    let mut plan_runtime = EngineAbsIsolationPlanRuntime;
    let dispatched_abs = solve_abs_isolation_plan_with_runtime(abs_plan, &mut plan_runtime)?;

    match dispatched_abs {
        AbsIsolationSolved::ReturnedEmptySet => Ok((SolutionSet::Empty, steps)),
        AbsIsolationSolved::IsolatedSingle(equation) => isolate(
            equation.lhs,
            equation.rhs,
            equation.op,
            var,
            simplifier,
            opts,
            ctx,
        ),
        AbsIsolationSolved::Split((positive, negative)) => {
            let include_items = simplifier.collect_steps();
            let mut runtime = EngineFunctionIsolationRuntime {
                simplifier,
                opts,
                solve_ctx: ctx,
            };
            let solved = solve_abs_split_pipeline_with_optional_items_runtime(
                positive,
                negative,
                arg,
                include_items,
                &steps,
                var,
                &mut runtime,
            )?;

            let final_set = finalize_abs_split_solution_set(
                &runtime.simplifier.context,
                op,
                contains_var(&runtime.simplifier.context, rhs, var),
                rhs,
                solved.positive_set,
                solved.negative_set,
            );
            Ok((final_set, solved.steps))
        }
    }
}

/// Handle `log(base, arg) = RHS`
#[allow(clippy::too_many_arguments)]
fn isolate_log(
    base: ExprId,
    arg: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let mut plan_runtime = EngineLogIsolationPlanRuntime;
    let rewrite = plan_log_isolation_step_with_runtime(
        &mut simplifier.context,
        base,
        arg,
        rhs,
        var,
        op.clone(),
        &mut plan_runtime,
    )
    .ok_or_else(|| {
        CasError::IsolationError(
            var.to_string(),
            "Cannot isolate from log function".to_string(),
        )
    })?;
    let include_item = simplifier.collect_steps();
    let mut runtime = EngineLogIsolationRuntime {
        simplifier,
        var,
        opts,
        ctx,
    };
    let solved =
        solve_log_isolation_rewrite_pipeline_with_item(rewrite, include_item, &mut runtime)?;
    prepend_steps(solved.solved, steps)
}

/// Handle single-argument functions: ln, exp, sqrt, sin, cos, tan
#[allow(clippy::too_many_arguments)]
fn isolate_unary_function(
    fn_id: SymbolId,
    arg: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let fn_name = simplifier.context.sym_name(fn_id).to_string();
    let execution = {
        let mut runtime = EngineUnaryInverseRuntime { simplifier };
        execute_unary_inverse_with_runtime(&mut runtime, &fn_name, arg, rhs, op.clone(), true)
            .ok_or_else(|| CasError::UnknownFunction(fn_name.clone()))?
    };
    let include_items = simplifier.collect_steps();
    let mut runtime = EngineFunctionIsolationRuntime {
        simplifier,
        opts,
        solve_ctx: ctx,
    };
    let solved_execution = solve_unary_inverse_execution_pipeline_with_items_runtime(
        execution,
        include_items,
        var,
        &mut runtime,
    )?;
    prepend_steps(
        (solved_execution.solution_set, solved_execution.steps),
        steps,
    )
}
