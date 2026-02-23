mod arithmetic;
mod functions;
mod power;

use crate::engine::Simplifier;
use crate::solver::{SolveStep, SolverOptions, MAX_SOLVE_DEPTH, SOLVE_DEPTH};
use cas_ast::{Expr, ExprId, RelOp, SolutionSet};
use cas_solver_core::solve_outcome::{
    plan_negated_lhs_isolation_step, residual_solution_set,
    resolve_circular_isolated_outcome_with_runtime, resolve_isolated_variable_outcome,
    solve_term_isolation_rewrite_pipeline_with_item, CircularIsolatedOutcome,
    CircularIsolatedRuntime, IsolatedVariableOutcome,
};

use crate::error::CasError;

struct EngineCircularIsolatedRuntime<'a> {
    simplifier: &'a mut Simplifier,
}

impl CircularIsolatedRuntime<SolveStep> for EngineCircularIsolatedRuntime<'_> {
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
            // Simplify RHS before returning
            let (sim_rhs, _) = simplifier.simplify(rhs);

            match resolve_isolated_variable_outcome(&mut simplifier.context, sim_rhs, op, var) {
                IsolatedVariableOutcome::Solved(set) => Ok((set, steps)),
                IsolatedVariableOutcome::ContainsTargetVariable => {
                    let mut runtime = EngineCircularIsolatedRuntime { simplifier };
                    let circular =
                        resolve_circular_isolated_outcome_with_runtime(lhs, rhs, var, &mut runtime);
                    match circular {
                        CircularIsolatedOutcome::Solved {
                            solution_set,
                            steps: linear_steps,
                        } => {
                            let mut all_steps = steps;
                            all_steps.extend(linear_steps);
                            Ok((solution_set, all_steps))
                        }
                        CircularIsolatedOutcome::Residual(set) => Ok((set, steps)),
                    }
                }
            }
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
            let plan = plan_negated_lhs_isolation_step(&mut simplifier.context, inner, rhs, op);
            let include_item = simplifier.collect_steps();
            let solved = solve_term_isolation_rewrite_pipeline_with_item(
                plan,
                include_item,
                |equation| {
                    isolate(
                        equation.lhs,
                        equation.rhs,
                        equation.op.clone(),
                        var,
                        simplifier,
                        opts,
                        ctx,
                    )
                },
                |item| SolveStep {
                    description: item.description,
                    equation_after: item.equation,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                },
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
