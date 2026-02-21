mod arithmetic;
mod functions;
mod power;

use crate::engine::Simplifier;
use crate::solver::solution_set::{neg_inf, pos_inf};
use crate::solver::{SolveStep, SolverOptions, MAX_SOLVE_DEPTH, SOLVE_DEPTH};
use cas_ast::{BoundType, Equation, Expr, ExprId, Interval, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::contains_var;

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
                let residual = cas_solver_core::isolation_utils::mk_residual_solve(
                    &mut simplifier.context,
                    lhs,
                    rhs,
                    var,
                );
                return Ok((SolutionSet::Residual(residual), steps));
            }

            let set = match op {
                RelOp::Eq => SolutionSet::Discrete(vec![sim_rhs]),
                RelOp::Neq => {
                    let i1 = Interval {
                        min: neg_inf(&mut simplifier.context),
                        min_type: BoundType::Open,
                        max: sim_rhs,
                        max_type: BoundType::Open,
                    };
                    let i2 = Interval {
                        min: sim_rhs,
                        min_type: BoundType::Open,
                        max: pos_inf(&mut simplifier.context),
                        max_type: BoundType::Open,
                    };
                    SolutionSet::Union(vec![i1, i2])
                }
                RelOp::Lt => SolutionSet::Continuous(Interval {
                    min: neg_inf(&mut simplifier.context),
                    min_type: BoundType::Open,
                    max: sim_rhs,
                    max_type: BoundType::Open,
                }),
                RelOp::Gt => SolutionSet::Continuous(Interval {
                    min: sim_rhs,
                    min_type: BoundType::Open,
                    max: pos_inf(&mut simplifier.context),
                    max_type: BoundType::Open,
                }),
                RelOp::Leq => SolutionSet::Continuous(Interval {
                    min: neg_inf(&mut simplifier.context),
                    min_type: BoundType::Open,
                    max: sim_rhs,
                    max_type: BoundType::Closed,
                }),
                RelOp::Geq => SolutionSet::Continuous(Interval {
                    min: sim_rhs,
                    min_type: BoundType::Closed,
                    max: pos_inf(&mut simplifier.context),
                    max_type: BoundType::Open,
                }),
            };
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
            // -A < RHS -> A > -RHS (Flip op)
            let new_rhs = simplifier.context.add(Expr::Neg(rhs));
            let new_op = cas_solver_core::isolation_utils::flip_inequality(op);
            let new_eq = Equation {
                lhs: inner,
                rhs: new_rhs,
                op: new_op.clone(),
            };

            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: "Multiply both sides by -1 (flips inequality)".to_string(),
                    equation_after: new_eq,
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

pub(crate) fn simplify_rhs(
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
