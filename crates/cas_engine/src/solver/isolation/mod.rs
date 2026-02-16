mod arithmetic;
mod functions;
mod power;

use crate::engine::Simplifier;
use crate::solver::solution_set::{neg_inf, pos_inf};
use crate::solver::{SolveStep, SolverOptions, MAX_SOLVE_DEPTH, SOLVE_DEPTH};
use cas_ast::{BoundType, Context, Equation, Expr, ExprId, Interval, RelOp, SolutionSet};

use crate::error::CasError;

/// Create a residual solve expression: solve(__eq__(lhs, rhs), var)
/// Used when solver can't justify a step but wants graceful degradation.
pub(super) fn mk_residual_solve(ctx: &mut Context, lhs: ExprId, rhs: ExprId, var: &str) -> ExprId {
    let eq_expr = cas_ast::eq::wrap_eq(ctx, lhs, rhs);
    let var_expr = ctx.var(var);
    ctx.call("solve", vec![eq_expr, var_expr])
}

pub(crate) fn isolate(
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    env: &super::SolveDomainEnv,
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
                let residual = mk_residual_solve(&mut simplifier.context, lhs, rhs, var);
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
            arithmetic::isolate_add(lhs, l, r, rhs, op, var, simplifier, opts, steps, env)
        }
        Expr::Sub(l, r) => {
            arithmetic::isolate_sub(l, r, rhs, op, var, simplifier, opts, steps, env)
        }
        Expr::Mul(l, r) => {
            arithmetic::isolate_mul(lhs, l, r, rhs, op, var, simplifier, opts, steps, env)
        }
        Expr::Div(l, r) => {
            arithmetic::isolate_div(lhs, l, r, rhs, op, var, simplifier, opts, steps, env)
        }
        Expr::Pow(b, e) => {
            power::isolate_pow(lhs, b, e, rhs, op, var, simplifier, opts, steps, env)
        }
        Expr::Function(fn_id, args) => {
            functions::isolate_function(fn_id, args, rhs, op, var, simplifier, opts, steps, env)
        }
        Expr::Neg(inner) => {
            // -A = RHS -> A = -RHS
            // -A < RHS -> A > -RHS (Flip op)
            let new_rhs = simplifier.context.add(Expr::Neg(rhs));
            let new_op = flip_inequality(op);
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

            let results = isolate(inner, new_rhs, new_op, var, simplifier, opts, env)?;
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

/// Check if an expression is known to be negative (extended version).
///
/// Unlike `helpers::is_negative`, this also recursively analyzes Mul products
/// using XOR logic: (-a) * b = negative, (-a) * (-b) = positive.
///
/// This is specific to solver isolation logic where we need to determine
/// sign to correctly flip inequalities when multiplying/dividing.
pub(crate) fn is_known_negative(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n < num_rational::BigRational::from_integer(0.into()),
        Expr::Neg(_) => true,
        Expr::Mul(l, r) => is_known_negative(ctx, *l) ^ is_known_negative(ctx, *r),
        _ => false,
    }
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

pub fn contains_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    match ctx.get(expr) {
        Expr::Variable(sym_id) => ctx.sym_name(*sym_id) == var,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            contains_var(ctx, *l, var) || contains_var(ctx, *r, var)
        }
        Expr::Neg(e) | Expr::Hold(e) => contains_var(ctx, *e, var),
        Expr::Function(_, args) => args.iter().any(|a| contains_var(ctx, *a, var)),
        Expr::Matrix { data, .. } => data.iter().any(|elem| contains_var(ctx, *elem, var)),
        // Leaves — no children to check
        Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => false,
    }
}

/// Attempt to recompose a^e / b^e -> (a/b)^e when both powers have the same exponent.
///
/// This is used to undo the simplification (a/b)^x -> a^x/b^x when solving exponentials,
/// allowing clean isolation: (a/b)^x = c/d -> x = log(a/b, c/d).
///
/// Uses structural comparison to match exponents that are semantically equal
/// but may have different ExprIds (which happens during simplification).
///
/// Returns Some(recomposed_expr) where recomposed = (a/b)^e, if pattern matches.
/// Returns None if pattern does not match.
pub(crate) fn try_recompose_pow_quotient(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    use crate::ordering::compare_expr;
    use std::cmp::Ordering;

    let expr_data = ctx.get(expr).clone();
    if let Expr::Div(num, den) = expr_data {
        let num_data = ctx.get(num).clone();
        let den_data = ctx.get(den).clone();
        if let (Expr::Pow(a, e1), Expr::Pow(b, e2)) = (num_data, den_data) {
            // Use structural comparison instead of ExprId ==
            if compare_expr(ctx, e1, e2) == Ordering::Equal {
                let new_base = ctx.add(Expr::Div(a, b));
                return Some(ctx.add(Expr::Pow(new_base, e1)));
            }
        }
    }
    None
}

pub(crate) fn flip_inequality(op: RelOp) -> RelOp {
    match op {
        RelOp::Eq => RelOp::Eq,
        RelOp::Neq => RelOp::Neq,
        RelOp::Lt => RelOp::Gt,
        RelOp::Gt => RelOp::Lt,
        RelOp::Leq => RelOp::Geq,
        RelOp::Geq => RelOp::Leq,
    }
}
