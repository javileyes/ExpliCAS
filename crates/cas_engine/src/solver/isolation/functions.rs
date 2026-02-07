use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solution_set::{intersect_solution_sets, union_solution_sets};
use crate::solver::{SolveStep, SolverOptions};
use cas_ast::symbol::SymbolId;
use cas_ast::{BuiltinFn, Equation, Expr, ExprId, RelOp, SolutionSet};

use super::{contains_var, isolate, prepend_steps, simplify_rhs};

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
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    if simplifier.context.is_builtin(fn_id, BuiltinFn::Abs) && args.len() == 1 {
        isolate_abs(args[0], rhs, op, var, simplifier, opts, steps)
    } else if simplifier.context.is_builtin(fn_id, BuiltinFn::Log) && args.len() == 2 {
        isolate_log(args[0], args[1], rhs, op, var, simplifier, opts, steps)
    } else if args.len() == 1 {
        let arg = args[0];
        if contains_var(&simplifier.context, arg, var) {
            isolate_unary_function(fn_id, arg, rhs, op, var, simplifier, opts, steps)
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
fn isolate_abs(
    arg: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    // Branch 1: Positive case (A op B)
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
                cas_ast::DisplayExpr {
                    context: &simplifier.context,
                    id: arg
                },
                op,
                cas_ast::DisplayExpr {
                    context: &simplifier.context,
                    id: rhs
                }
            ),
            equation_after: eq1,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }
    let results1 = isolate(arg, rhs, op.clone(), var, simplifier, opts)?;
    let (set1, steps1_out) = prepend_steps(results1, steps1)?;

    // Branch 2: Negative case
    let neg_rhs = simplifier.context.add(Expr::Neg(rhs));
    let op2 = match op {
        RelOp::Eq => RelOp::Eq,
        RelOp::Neq => RelOp::Neq,
        RelOp::Lt => RelOp::Gt,
        RelOp::Leq => RelOp::Geq,
        RelOp::Gt => RelOp::Lt,
        RelOp::Geq => RelOp::Leq,
    };

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
                cas_ast::DisplayExpr {
                    context: &simplifier.context,
                    id: arg
                },
                op2,
                cas_ast::DisplayExpr {
                    context: &simplifier.context,
                    id: neg_rhs
                }
            ),
            equation_after: eq2,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }
    let results2 = isolate(arg, neg_rhs, op2, var, simplifier, opts)?;
    let (set2, steps2_out) = prepend_steps(results2, steps2)?;

    // Combine sets
    let final_set = match op {
        RelOp::Eq | RelOp::Neq | RelOp::Gt | RelOp::Geq => {
            union_solution_sets(&simplifier.context, set1, set2)
        }
        RelOp::Lt | RelOp::Leq => intersect_solution_sets(&simplifier.context, set1, set2),
    };

    let mut all_steps = steps1_out;
    all_steps.extend(steps2_out);

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
                    cas_ast::DisplayExpr {
                        context: &simplifier.context,
                        id: base
                    }
                ),
                equation_after: new_eq,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        let results = isolate(arg, new_rhs, op, var, simplifier, opts)?;
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
        let results = isolate(base, new_rhs, op, var, simplifier, opts)?;
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
            let results = isolate(arg, new_rhs, op, var, simplifier, opts)?;
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
            let results = isolate(arg, new_rhs, op, var, simplifier, opts)?;
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
            let results = isolate(arg, new_rhs, op, var, simplifier, opts)?;
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

            let results = isolate(arg, simplified_rhs, op, var, simplifier, opts)?;
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

            let results = isolate(arg, simplified_rhs, op, var, simplifier, opts)?;
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

            let results = isolate(arg, simplified_rhs, op, var, simplifier, opts)?;
            prepend_steps(results, steps)
        }
        _ => Err(CasError::UnknownFunction(
            simplifier.context.sym_name(fn_id).to_string(),
        )),
    }
}
