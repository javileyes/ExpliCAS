use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{SolveStep, SolverOptions};
use cas_ast::symbol::SymbolId;
use cas_ast::{
    BuiltinFn, Case, ConditionPredicate, ConditionSet, Equation, Expr, ExprId, RelOp, SolutionSet,
};
use cas_solver_core::isolation_utils::{combine_abs_branch_sets, contains_var, flip_inequality};

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
    // ── Pre-check: numeric RHS ──────────────────────────────────────────
    // |A| is always ≥ 0, so |A| = negative is impossible.
    if matches!(op, RelOp::Eq) {
        if let Expr::Number(n) = simplifier.context.get(rhs) {
            use num_traits::{Signed, Zero};
            if n.is_negative() {
                // |A| = (negative) → no solution
                return Ok((SolutionSet::Empty, steps));
            }
            if n.is_zero() {
                // |A| = 0  →  A = 0  (only one branch needed)
                return isolate(arg, rhs, op, var, simplifier, opts, ctx);
            }
            // n > 0: fall through to normal branch split
        }
    }

    // ── Branch 1: Positive case (A op B) ────────────────────────────────
    let eq1 = Equation {
        lhs: arg,
        rhs,
        op: op.clone(),
    };
    let mut steps1 = steps.clone();
    if simplifier.collect_steps() {
        steps1.push(SolveStep {
            description: format!(
                "Split absolute value (Case 1): {} {} {}",
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id: arg
                },
                op,
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id: rhs
                }
            ),
            equation_after: eq1,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }
    let results1 = isolate(arg, rhs, op.clone(), var, simplifier, opts, ctx)?;
    let (set1, steps1_out) = prepend_steps(results1, steps1)?;

    // ── Branch 2: Negative case ─────────────────────────────────────────
    let neg_rhs = simplifier.context.add(Expr::Neg(rhs));
    let op2 = flip_inequality(op.clone());

    let eq2 = Equation {
        lhs: arg,
        rhs: neg_rhs,
        op: op2.clone(),
    };
    let mut steps2 = steps.clone();
    if simplifier.collect_steps() {
        steps2.push(SolveStep {
            description: format!(
                "Split absolute value (Case 2): {} {} {}",
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id: arg
                },
                op2,
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id: neg_rhs
                }
            ),
            equation_after: eq2,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }
    let results2 = isolate(arg, neg_rhs, op2, var, simplifier, opts, ctx)?;
    let (set2, steps2_out) = prepend_steps(results2, steps2)?;

    // ── Combine branches ────────────────────────────────────────────────
    let combined_set = combine_abs_branch_sets(&simplifier.context, op, set1, set2);

    let mut all_steps = steps1_out;
    all_steps.extend(steps2_out);

    // ── Soundness guard: rhs ≥ 0 ───────────────────────────────────────
    // When rhs contains the solve variable, the combined set may be unsound
    // (e.g., |x| = x gives AllReals from branch 1 without domain restriction).
    // Guard: wrap in Conditional with NonNegative(rhs).
    let final_set = if contains_var(&simplifier.context, rhs, var) {
        let guard = ConditionSet::single(ConditionPredicate::NonNegative(rhs));
        SolutionSet::Conditional(vec![Case::new(guard, combined_set)])
    } else {
        combined_set
    };

    Ok((final_set, all_steps))
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
    if contains_var(&simplifier.context, arg, var) && !contains_var(&simplifier.context, base, var)
    {
        // log(b, x) = RHS -> x = b^RHS
        let new_rhs = simplifier.context.add(Expr::Pow(base, rhs));
        let new_eq = Equation {
            lhs: arg,
            rhs: new_rhs,
            op: op.clone(),
        };
        if simplifier.collect_steps() {
            steps.push(SolveStep {
                description: format!(
                    "Exponentiate both sides with base {}",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: base
                    }
                ),
                equation_after: new_eq,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        let results = isolate(arg, new_rhs, op, var, simplifier, opts, ctx)?;
        prepend_steps(results, steps)
    } else if contains_var(&simplifier.context, base, var)
        && !contains_var(&simplifier.context, arg, var)
    {
        let one = simplifier.context.num(1);
        let inv_rhs = simplifier.context.add(Expr::Div(one, rhs));
        let new_rhs = simplifier.context.add(Expr::Pow(arg, inv_rhs));
        let new_eq = Equation {
            lhs: base,
            rhs: new_rhs,
            op: op.clone(),
        };
        if simplifier.collect_steps() {
            steps.push(SolveStep {
                description: "Isolate base of logarithm".to_string(),
                equation_after: new_eq,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        let results = isolate(base, new_rhs, op, var, simplifier, opts, ctx)?;
        prepend_steps(results, steps)
    } else {
        Err(CasError::IsolationError(
            var.to_string(),
            "Cannot isolate from log function".to_string(),
        ))
    }
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
    match simplifier.context.sym_name(fn_id) {
        "ln" => {
            let e = simplifier.context.add(Expr::Constant(cas_ast::Constant::E));
            let new_rhs = simplifier.context.add(Expr::Pow(e, rhs));
            let new_eq = Equation {
                lhs: arg,
                rhs: new_rhs,
                op: op.clone(),
            };
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: "Exponentiate both sides with base e".to_string(),
                    equation_after: new_eq,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            let results = isolate(arg, new_rhs, op, var, simplifier, opts, ctx)?;
            prepend_steps(results, steps)
        }
        "exp" => {
            let new_rhs = simplifier.context.call("ln", vec![rhs]);
            let new_eq = Equation {
                lhs: arg,
                rhs: new_rhs,
                op: op.clone(),
            };
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: "Take natural log of both sides".to_string(),
                    equation_after: new_eq,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            let results = isolate(arg, new_rhs, op, var, simplifier, opts, ctx)?;
            prepend_steps(results, steps)
        }
        "sqrt" => {
            let two = simplifier.context.num(2);
            let new_rhs = simplifier.context.add(Expr::Pow(rhs, two));
            let new_eq = Equation {
                lhs: arg,
                rhs: new_rhs,
                op: op.clone(),
            };
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: "Square both sides".to_string(),
                    equation_after: new_eq,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            let results = isolate(arg, new_rhs, op, var, simplifier, opts, ctx)?;
            prepend_steps(results, steps)
        }
        "sin" => {
            let new_rhs = simplifier.context.call("arcsin", vec![rhs]);
            let new_eq = Equation {
                lhs: arg,
                rhs: new_rhs,
                op: op.clone(),
            };
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: "Take arcsin of both sides".to_string(),
                    equation_after: new_eq,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }

            let (simplified_rhs, sim_steps) = simplify_rhs(new_rhs, arg, op.clone(), simplifier);
            steps.extend(sim_steps);

            let results = isolate(arg, simplified_rhs, op, var, simplifier, opts, ctx)?;
            prepend_steps(results, steps)
        }
        "cos" => {
            let new_rhs = simplifier.context.call("arccos", vec![rhs]);
            let new_eq = Equation {
                lhs: arg,
                rhs: new_rhs,
                op: op.clone(),
            };
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: "Take arccos of both sides".to_string(),
                    equation_after: new_eq,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }

            let (simplified_rhs, sim_steps) = simplify_rhs(new_rhs, arg, op.clone(), simplifier);
            steps.extend(sim_steps);

            let results = isolate(arg, simplified_rhs, op, var, simplifier, opts, ctx)?;
            prepend_steps(results, steps)
        }
        "tan" => {
            let new_rhs = simplifier.context.call("arctan", vec![rhs]);
            let new_eq = Equation {
                lhs: arg,
                rhs: new_rhs,
                op: op.clone(),
            };
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: "Take arctan of both sides".to_string(),
                    equation_after: new_eq,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }

            let (simplified_rhs, sim_steps) = simplify_rhs(new_rhs, arg, op.clone(), simplifier);
            steps.extend(sim_steps);

            let results = isolate(arg, simplified_rhs, op, var, simplifier, opts, ctx)?;
            prepend_steps(results, steps)
        }
        _ => Err(CasError::UnknownFunction(
            simplifier.context.sym_name(fn_id).to_string(),
        )),
    }
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
