use crate::{EvalOutput, ImplicitCondition, ImportanceLevel};
use cas_api_models::{RequiredConditionJson, SolveStepJson, SolveSubStepJson, WarningJson};
use cas_ast::{Context, Expr, ExprId, RelOp, SolutionSet};
use cas_formatter::{DisplayExpr, LaTeXExpr};

/// Detect the variable to solve for in an equation.
/// Prefers `x` if present, then common names, then alphabetic first.
pub fn detect_solve_variable_eval_json(ctx: &Context, lhs: ExprId, rhs: ExprId) -> String {
    let mut variables = std::collections::HashSet::new();

    fn collect_vars(ctx: &Context, expr: ExprId, vars: &mut std::collections::HashSet<String>) {
        match ctx.get(expr) {
            Expr::Variable(sym_id) => {
                let name = ctx.sym_name(*sym_id);
                if !name.starts_with('#') && name != "e" && name != "i" && name != "pi" {
                    vars.insert(name.to_string());
                }
            }
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => {
                collect_vars(ctx, *a, vars);
                collect_vars(ctx, *b, vars);
            }
            Expr::Neg(e) => collect_vars(ctx, *e, vars),
            Expr::Function(_, args) => {
                for arg in args {
                    collect_vars(ctx, *arg, vars);
                }
            }
            _ => {}
        }
    }

    collect_vars(ctx, lhs, &mut variables);
    collect_vars(ctx, rhs, &mut variables);

    if variables.contains("x") {
        return "x".to_string();
    }

    for preferred in ["y", "z", "t", "n", "a", "b", "c"] {
        if variables.contains(preferred) {
            return preferred.to_string();
        }
    }

    variables
        .into_iter()
        .min()
        .unwrap_or_else(|| "x".to_string())
}

/// Convert solver steps to JSON format (for equation solving in eval-json).
pub fn collect_solve_steps_eval_json(
    output: &EvalOutput,
    ctx: &Context,
    steps_mode: &str,
) -> Vec<SolveStepJson> {
    if steps_mode != "on" {
        return vec![];
    }

    let filtered: Vec<_> = output
        .solve_steps
        .iter()
        .filter(|step| step.importance >= ImportanceLevel::Medium)
        .collect();

    if filtered.is_empty() {
        return vec![];
    }

    filtered
        .iter()
        .enumerate()
        .map(|(i, step)| {
            let lhs_str = format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: step.equation_after.lhs
                }
            );
            let rhs_str = format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: step.equation_after.rhs
                }
            );
            let relop_str = format!("{}", step.equation_after.op);
            let equation_str = format!("{} {} {}", lhs_str, relop_str, rhs_str);

            let lhs_latex = LaTeXExpr {
                context: ctx,
                id: step.equation_after.lhs,
            }
            .to_latex();
            let rhs_latex = LaTeXExpr {
                context: ctx,
                id: step.equation_after.rhs,
            }
            .to_latex();
            let relop_latex = relop_to_latex(&step.equation_after.op);

            let substeps: Vec<SolveSubStepJson> = step
                .substeps
                .iter()
                .enumerate()
                .map(|(j, ss)| {
                    let ss_lhs_str = format!(
                        "{}",
                        DisplayExpr {
                            context: ctx,
                            id: ss.equation_after.lhs
                        }
                    );
                    let ss_rhs_str = format!(
                        "{}",
                        DisplayExpr {
                            context: ctx,
                            id: ss.equation_after.rhs
                        }
                    );
                    let ss_relop_str = format!("{}", ss.equation_after.op);
                    let ss_equation_str = format!("{} {} {}", ss_lhs_str, ss_relop_str, ss_rhs_str);

                    let ss_lhs_latex = LaTeXExpr {
                        context: ctx,
                        id: ss.equation_after.lhs,
                    }
                    .to_latex();
                    let ss_rhs_latex = LaTeXExpr {
                        context: ctx,
                        id: ss.equation_after.rhs,
                    }
                    .to_latex();
                    let ss_relop_latex = relop_to_latex(&ss.equation_after.op);

                    SolveSubStepJson {
                        index: format!("{}.{}", i + 1, j + 1),
                        description: ss.description.clone(),
                        equation: ss_equation_str,
                        lhs_latex: ss_lhs_latex,
                        relop: ss_relop_latex,
                        rhs_latex: ss_rhs_latex,
                    }
                })
                .collect();

            SolveStepJson {
                index: i + 1,
                description: step.description.clone(),
                equation: equation_str,
                lhs_latex,
                relop: relop_latex,
                rhs_latex,
                substeps,
            }
        })
        .collect()
}

/// Engine-level wrapper for collecting solve steps in eval-json format.
pub fn collect_solve_steps_eval_json_with_engine(
    output: &EvalOutput,
    engine: &crate::Engine,
    steps_mode: &str,
) -> Vec<SolveStepJson> {
    collect_solve_steps_eval_json(output, &engine.simplifier.context, steps_mode)
}

/// Format a solution set for eval-json plain-text result.
pub fn format_solution_set_eval_json(ctx: &Context, solution_set: &SolutionSet) -> String {
    match solution_set {
        SolutionSet::Empty => "No solution".to_string(),
        SolutionSet::AllReals => "All real numbers".to_string(),
        SolutionSet::Discrete(exprs) => {
            if exprs.is_empty() {
                "No solution".to_string()
            } else {
                let sols: Vec<String> = exprs
                    .iter()
                    .map(|e| {
                        format!(
                            "{}",
                            DisplayExpr {
                                context: ctx,
                                id: *e
                            }
                        )
                    })
                    .collect();
                format!("{{ {} }}", sols.join(", "))
            }
        }
        SolutionSet::Conditional(cases) => {
            let mut parts = Vec::new();
            for case in cases {
                if crate::is_pure_residual_otherwise(case) {
                    continue;
                }
                let cond_str = cas_formatter::condition_set_to_display(&case.when, ctx);
                let inner_str = format_solution_set_eval_json(ctx, &case.then.solutions);
                if case.when.is_empty() {
                    parts.push(format!("{} otherwise", inner_str));
                } else {
                    parts.push(format!("{} if {}", inner_str, cond_str));
                }
            }

            if parts.is_empty() && !cases.is_empty() {
                for case in cases {
                    if !case.when.is_empty() {
                        let inner_str = format_solution_set_eval_json(ctx, &case.then.solutions);
                        let cond_str = cas_formatter::condition_set_to_display(&case.when, ctx);
                        return format!("{} if {}", inner_str, cond_str);
                    }
                }
            }
            parts.join("; ")
        }
        SolutionSet::Continuous(interval) => {
            format!(
                "[{}, {}]",
                DisplayExpr {
                    context: ctx,
                    id: interval.min
                },
                DisplayExpr {
                    context: ctx,
                    id: interval.max
                }
            )
        }
        SolutionSet::Union(intervals) => {
            let parts: Vec<String> = intervals
                .iter()
                .map(|int| {
                    format!(
                        "[{}, {}]",
                        DisplayExpr {
                            context: ctx,
                            id: int.min
                        },
                        DisplayExpr {
                            context: ctx,
                            id: int.max
                        }
                    )
                })
                .collect();
            parts.join(" ∪ ")
        }
        SolutionSet::Residual(expr) => {
            format!(
                "Solve: {} = 0",
                DisplayExpr {
                    context: ctx,
                    id: *expr
                }
            )
        }
    }
}

/// Format a solution set for eval-json LaTeX result.
pub fn solution_set_to_latex_eval_json(ctx: &Context, solution_set: &SolutionSet) -> String {
    match solution_set {
        SolutionSet::Empty => r"\emptyset".to_string(),
        SolutionSet::AllReals => r"\mathbb{R}".to_string(),
        SolutionSet::Discrete(exprs) => {
            if exprs.is_empty() {
                r"\emptyset".to_string()
            } else {
                let sols: Vec<String> = exprs
                    .iter()
                    .map(|e| {
                        LaTeXExpr {
                            context: ctx,
                            id: *e,
                        }
                        .to_latex()
                    })
                    .collect();
                format!(r"\left\{{ {} \right\}}", sols.join(", "))
            }
        }
        SolutionSet::Conditional(cases) => {
            let mut latex_parts = Vec::new();
            for case in cases {
                if crate::is_pure_residual_otherwise(case) {
                    continue;
                }
                let cond_latex = cas_formatter::condition_set_to_latex(&case.when, ctx);
                let inner_latex = solution_set_to_latex_eval_json(ctx, &case.then.solutions);
                if case.when.is_empty() {
                    latex_parts.push(format!(r"{} & \text{{otherwise}}", inner_latex));
                } else {
                    latex_parts.push(format!(r"{} & \text{{if }} {}", inner_latex, cond_latex));
                }
            }
            if latex_parts.len() == 1 {
                let single = &latex_parts[0];
                if let Some(idx) = single.find(r" & \text{if}") {
                    return single[..idx].to_string();
                }
            }
            format!(
                r"\begin{{cases}} {} \end{{cases}}",
                latex_parts.join(r" \\ ")
            )
        }
        SolutionSet::Continuous(interval) => {
            let min_latex = LaTeXExpr {
                context: ctx,
                id: interval.min,
            }
            .to_latex();
            let max_latex = LaTeXExpr {
                context: ctx,
                id: interval.max,
            }
            .to_latex();
            format!(r"\left[{}, {}\right]", min_latex, max_latex)
        }
        SolutionSet::Union(intervals) => {
            let parts: Vec<String> = intervals
                .iter()
                .map(|int| {
                    let min = LaTeXExpr {
                        context: ctx,
                        id: int.min,
                    }
                    .to_latex();
                    let max = LaTeXExpr {
                        context: ctx,
                        id: int.max,
                    }
                    .to_latex();
                    format!(r"\left[{}, {}\right]", min, max)
                })
                .collect();
            parts.join(r" \cup ")
        }
        SolutionSet::Residual(expr) => {
            let expr_latex = LaTeXExpr {
                context: ctx,
                id: *expr,
            }
            .to_latex();
            format!(r"\text{{Solve: }} {} = 0", expr_latex)
        }
    }
}

fn relop_to_latex(op: &RelOp) -> String {
    match op {
        RelOp::Eq => "=".to_string(),
        RelOp::Lt => "<".to_string(),
        RelOp::Leq => r"\leq".to_string(),
        RelOp::Gt => ">".to_string(),
        RelOp::Geq => r"\geq".to_string(),
        RelOp::Neq => r"\neq".to_string(),
    }
}

/// Collect domain warnings in eval-json schema format.
pub fn collect_warnings_eval_json(output: &EvalOutput) -> Vec<WarningJson> {
    output
        .domain_warnings
        .iter()
        .map(|w| WarningJson {
            rule: w.rule_name.clone(),
            assumption: w.message.clone(),
        })
        .collect()
}

/// Collect required conditions in eval-json schema format.
pub fn collect_required_conditions_eval_json(
    output: &EvalOutput,
    ctx: &Context,
) -> Vec<RequiredConditionJson> {
    output
        .required_conditions
        .iter()
        .map(|cond| {
            let (kind, expr_id) = match cond {
                ImplicitCondition::NonNegative(e) => ("NonNegative", *e),
                ImplicitCondition::Positive(e) => ("Positive", *e),
                ImplicitCondition::NonZero(e) => ("NonZero", *e),
            };
            let expr_str = DisplayExpr {
                context: ctx,
                id: expr_id,
            }
            .to_string();
            RequiredConditionJson {
                kind: kind.to_string(),
                expr_display: expr_str.clone(),
                expr_canonical: expr_str,
            }
        })
        .collect()
}

/// Engine-level wrapper for collecting required conditions in eval-json format.
pub fn collect_required_conditions_eval_json_with_engine(
    output: &EvalOutput,
    engine: &crate::Engine,
) -> Vec<RequiredConditionJson> {
    collect_required_conditions_eval_json(output, &engine.simplifier.context)
}

/// Collect human-readable required condition strings.
pub fn collect_required_display_eval_json(output: &EvalOutput, ctx: &Context) -> Vec<String> {
    output
        .required_conditions
        .iter()
        .map(|cond| cond.display(ctx))
        .collect()
}

/// Engine-level wrapper for collecting required condition display strings.
pub fn collect_required_display_eval_json_with_engine(
    output: &EvalOutput,
    engine: &crate::Engine,
) -> Vec<String> {
    collect_required_display_eval_json(output, &engine.simplifier.context)
}
