use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{SolveStep, SolverOptions};
use cas_ast::symbol::SymbolId;
use cas_ast::{BuiltinFn, ExprId, RelOp, SolutionSet};
use cas_solver_core::function_inverse::{
    execute_unary_inverse_with_runtime, solve_unary_inverse_execution_with, UnaryInverseRuntime,
};
use cas_solver_core::isolation_utils::{contains_var, numeric_sign};
use cas_solver_core::log_isolation::{
    collect_log_isolation_execution_items, plan_log_isolation_step_with,
    solve_log_isolation_rewrite_with,
};
use cas_solver_core::solve_outcome::{
    build_abs_split_execution_with, collect_abs_split_execution_items,
    finalize_abs_split_solution_set, materialize_abs_split_execution, plan_abs_isolation,
    solve_abs_isolation_plan_with, solve_abs_split_cases_with, AbsIsolationSolved,
    AbsSplitSolvedCases,
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
    if simplifier.context.is_builtin(fn_id, BuiltinFn::Abs) && args.len() == 1 {
        isolate_abs(args[0], rhs, op, var, simplifier, opts, steps, ctx)
    } else if simplifier.context.is_builtin(fn_id, BuiltinFn::Log) && args.len() == 2 {
        isolate_log(args[0], args[1], rhs, op, var, simplifier, opts, steps, ctx)
    } else if args.len() == 1 {
        let arg = args[0];
        if contains_var(&simplifier.context, arg, var) {
            isolate_unary_function(fn_id, args[0], rhs, op, var, simplifier, opts, steps, ctx)
        } else {
            Err(CasError::VariableNotFound(var.to_string()))
        }
    } else {
        Err(CasError::IsolationError(
            var.to_string(),
            format!(
                "Cannot invert function '{}' with {} arguments",
                simplifier.context.sym_name(fn_id),
                args.len()
            ),
        ))
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
    let dispatched_abs = solve_abs_isolation_plan_with(
        abs_plan,
        |equation| Ok::<_, CasError>(equation),
        |positive, negative| Ok::<_, CasError>((positive, negative)),
    )?;

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
            let split_execution = if simplifier.collect_steps() {
                build_abs_split_execution_with(positive, negative, arg, |id| {
                    format!(
                        "{}",
                        cas_formatter::DisplayExpr {
                            context: &simplifier.context,
                            id
                        }
                    )
                })
            } else {
                materialize_abs_split_execution(positive, negative)
            };
            let split_items = collect_abs_split_execution_items(&split_execution);
            let mut branch_case_idx = 0usize;
            let solved = solve_abs_split_cases_with(&split_execution, |equation| {
                let mut case_steps = steps.clone();
                if let Some(item) = split_items.get(branch_case_idx) {
                    case_steps.push(SolveStep {
                        description: item.description().to_string(),
                        equation_after: item.equation.clone(),
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: vec![],
                    });
                }
                branch_case_idx += 1;
                let results = isolate(
                    equation.lhs,
                    equation.rhs,
                    equation.op.clone(),
                    var,
                    simplifier,
                    opts,
                    ctx,
                )?;
                prepend_steps(results, case_steps)
            })?;

            let AbsSplitSolvedCases {
                positive_branch: (set1, steps1_out),
                negative_branch: (set2, steps2_out),
            } = solved;

            // ── Combine branches + soundness guard rhs≥0 ──────────────────────
            let final_set = finalize_abs_split_solution_set(
                &simplifier.context,
                op,
                contains_var(&simplifier.context, rhs, var),
                rhs,
                set1,
                set2,
            );

            let mut all_steps = steps1_out;
            all_steps.extend(steps2_out);

            Ok((final_set, all_steps))
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
    mut steps: Vec<SolveStep>,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let rewrite = plan_log_isolation_step_with(
        &mut simplifier.context,
        base,
        arg,
        rhs,
        var,
        op.clone(),
        |core_ctx, id| {
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: core_ctx,
                    id
                }
            )
        },
    )
    .ok_or_else(|| {
        CasError::IsolationError(
            var.to_string(),
            "Cannot isolate from log function".to_string(),
        )
    })?;
    let solved = solve_log_isolation_rewrite_with(rewrite, |equation| {
        isolate(
            equation.lhs,
            equation.rhs,
            equation.op.clone(),
            var,
            simplifier,
            opts,
            ctx,
        )
    })?;
    if simplifier.collect_steps() {
        for item in collect_log_isolation_execution_items(&solved.rewrite) {
            steps.push(SolveStep {
                description: item.description().to_string(),
                equation_after: item.equation,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
    }
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
    mut steps: Vec<SolveStep>,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let fn_name = simplifier.context.sym_name(fn_id).to_string();
    let execution = {
        let mut runtime = EngineUnaryInverseRuntime { simplifier };
        execute_unary_inverse_with_runtime(&mut runtime, &fn_name, arg, rhs, op.clone(), true)
            .ok_or_else(|| CasError::UnknownFunction(fn_name.clone()))?
    };
    let solved_execution =
        solve_unary_inverse_execution_with(execution, |lhs, target_rhs, target_op| {
            isolate(lhs, target_rhs, target_op, var, simplifier, opts, ctx)
        })?;

    if simplifier.collect_steps() {
        for item in solved_execution.execution.rewrite_items {
            steps.push(SolveStep {
                description: item.description().to_string(),
                equation_after: item.equation,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        for item in solved_execution.execution.rhs_cleanup_items {
            steps.push(SolveStep {
                description: item.description().to_string(),
                equation_after: item.equation,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
    }
    prepend_steps(solved_execution.solved, steps)
}
