mod arithmetic;
mod functions;
mod power;

use crate::engine::Simplifier;
use crate::solver::{SolveStep, SolverOptions, MAX_SOLVE_DEPTH, SOLVE_DEPTH};
use cas_ast::{Expr, ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::contains_var;
use cas_solver_core::solution_set::isolated_var_solution;
use cas_solver_core::solve_outcome::{build_negated_lhs_isolation_step, residual_solution_set};

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

    let mut steps = Vec::new();

    let lhs_expr = simplifier.context.get(lhs).clone();

    match lhs_expr {
        Expr::Variable(sym_id) if simplifier.context.sym_name(sym_id) == var => {
            // Simplify RHS before returning
            let (sim_rhs, _) = simplifier.simplify(rhs);

            // GUARDRAIL: Reject if solution still contains target variable (circular)
            if contains_var(&simplifier.context, sim_rhs, var) {
                // Phase 2: Try linear_collect strategy before giving up
                if let Some((solution_set, linear_steps)) =
                    crate::solver::linear_collect::try_linear_collect(lhs, rhs, var, simplifier)
                {
                    let mut all_steps = steps;
                    all_steps.extend(linear_steps);
                    return Ok((solution_set, all_steps));
                }

                // Phase 2.1: Try structural linear form extractor
                if let Some((solution_set, linear_steps)) =
                    crate::solver::linear_collect::try_linear_collect_v2(lhs, rhs, var, simplifier)
                {
                    let mut all_steps = steps;
                    all_steps.extend(linear_steps);
                    return Ok((solution_set, all_steps));
                }

                // If linear_collect didn't work, return as Residual
                return Ok((
                    residual_solution_set(&mut simplifier.context, lhs, rhs, var),
                    steps,
                ));
            }

            let set = isolated_var_solution(&mut simplifier.context, sim_rhs, op);
            Ok((set, steps))
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
            // -A = RHS -> A = -RHS
            // -A < RHS -> A > -RHS (flip inequality)
            let new_eq = cas_solver_core::equation_rewrite::isolate_negated_lhs(
                &mut simplifier.context,
                inner,
                rhs,
                op,
            );
            let new_rhs = new_eq.rhs;
            let new_op = new_eq.op.clone();

            if simplifier.collect_steps() {
                let didactic_step = build_negated_lhs_isolation_step(new_eq);
                steps.push(SolveStep {
                    description: didactic_step.description,
                    equation_after: didactic_step.equation_after,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }

            let results = isolate(inner, new_rhs, new_op, var, simplifier, opts, ctx)?;
            prepend_steps(results, steps)
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
