use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{SolveStep, SolverOptions};
use cas_ast::symbol::SymbolId;
use cas_ast::{BuiltinFn, Equation, ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::{combine_abs_branch_sets, contains_var, numeric_sign};
use cas_solver_core::solve_outcome::{
    build_abs_split_step_with, guard_abs_solution_with_nonnegative_rhs, plan_abs_isolation,
    AbsIsolationPlan, AbsSplitCase,
};

use super::{isolate, prepend_steps};

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

    match abs_plan {
        AbsIsolationPlan::ReturnEmptySet => Ok((SolutionSet::Empty, steps)),
        AbsIsolationPlan::IsolateSingleEquation { equation } => isolate(
            equation.lhs,
            equation.rhs,
            equation.op,
            var,
            simplifier,
            opts,
            ctx,
        ),
        AbsIsolationPlan::SplitBranches { positive, negative } => {
            // ── Branch 1: Positive case (A op B) ────────────────────────────────
            let eq1 = positive;
            let eq2 = negative;
            let mut steps1 = steps.clone();
            if simplifier.collect_steps() {
                let split_step = build_abs_split_step_with(
                    AbsSplitCase::Positive,
                    eq1.clone(),
                    arg,
                    rhs,
                    op.clone(),
                    |id| {
                        format!(
                            "{}",
                            cas_formatter::DisplayExpr {
                                context: &simplifier.context,
                                id
                            }
                        )
                    },
                );
                steps1.push(SolveStep {
                    description: split_step.description,
                    equation_after: split_step.equation_after,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            let results1 = isolate(eq1.lhs, eq1.rhs, eq1.op, var, simplifier, opts, ctx)?;
            let (set1, steps1_out) = prepend_steps(results1, steps1)?;

            // ── Branch 2: Negative case ─────────────────────────────────────────
            let neg_rhs = eq2.rhs;
            let op2 = eq2.op.clone();
            let mut steps2 = steps.clone();
            if simplifier.collect_steps() {
                let split_step = build_abs_split_step_with(
                    AbsSplitCase::Negative,
                    eq2.clone(),
                    arg,
                    neg_rhs,
                    op2.clone(),
                    |id| {
                        format!(
                            "{}",
                            cas_formatter::DisplayExpr {
                                context: &simplifier.context,
                                id
                            }
                        )
                    },
                );
                steps2.push(SolveStep {
                    description: split_step.description,
                    equation_after: split_step.equation_after,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            let results2 = isolate(eq2.lhs, eq2.rhs, eq2.op, var, simplifier, opts, ctx)?;
            let (set2, steps2_out) = prepend_steps(results2, steps2)?;

            // ── Combine branches ────────────────────────────────────────────────
            let combined_set = combine_abs_branch_sets(&simplifier.context, op, set1, set2);

            let mut all_steps = steps1_out;
            all_steps.extend(steps2_out);

            // ── Soundness guard: rhs ≥ 0 ───────────────────────────────────────
            // When rhs contains the solve variable, the combined set may be unsound
            // (e.g., |x| = x gives AllReals from branch 1 without domain restriction).
            // Guard: wrap in Conditional with NonNegative(rhs).
            let final_set = guard_abs_solution_with_nonnegative_rhs(
                contains_var(&simplifier.context, rhs, var),
                rhs,
                combined_set,
            );

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
    let plan = cas_solver_core::log_isolation::plan_log_isolation(
        &mut simplifier.context,
        base,
        arg,
        rhs,
        var,
    )
    .ok_or_else(|| {
        CasError::IsolationError(
            var.to_string(),
            "Cannot isolate from log function".to_string(),
        )
    })?;

    let log_step = cas_solver_core::log_isolation::build_log_isolation_step_with(
        plan,
        base,
        op.clone(),
        |id| {
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id
                }
            )
        },
    );
    let new_eq = log_step.equation_after.clone();
    if simplifier.collect_steps() {
        steps.push(SolveStep {
            description: log_step.description,
            equation_after: log_step.equation_after,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }
    let results = isolate(
        new_eq.lhs, new_eq.rhs, new_eq.op, var, simplifier, opts, ctx,
    )?;
    prepend_steps(results, steps)
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
    let plan = cas_solver_core::function_inverse::plan_unary_inverse_rewrite(
        &mut simplifier.context,
        &fn_name,
        arg,
        rhs,
        op.clone(),
        true,
    )
    .ok_or_else(|| CasError::UnknownFunction(fn_name.clone()))?;
    let new_rhs = plan.equation.rhs;

    if simplifier.collect_steps() {
        steps.push(SolveStep {
            description: plan.step_description.to_string(),
            equation_after: plan.equation.clone(),
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }

    let target_rhs = if plan.needs_rhs_cleanup {
        let (simplified_rhs, sim_steps) = simplify_rhs(new_rhs, arg, op.clone(), simplifier);
        steps.extend(sim_steps);
        simplified_rhs
    } else {
        new_rhs
    };

    let results = isolate(arg, target_rhs, op, var, simplifier, opts, ctx)?;
    prepend_steps(results, steps)
}

fn simplify_rhs(
    rhs: ExprId,
    lhs: ExprId,
    op: RelOp,
    simplifier: &mut Simplifier,
) -> (ExprId, Vec<SolveStep>) {
    let (simplified_rhs, sim_steps) = simplifier.simplify(rhs);
    let mut steps = Vec::new();

    if simplifier.collect_steps() {
        for step in sim_steps {
            steps.push(SolveStep {
                description: step.description,
                equation_after: Equation {
                    lhs,
                    rhs: step.after,
                    op: op.clone(),
                },
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
    }
    (simplified_rhs, steps)
}
