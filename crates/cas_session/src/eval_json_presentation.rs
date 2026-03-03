//! Presentation helpers for session-backed `eval-json`.

use cas_api_models::{RequiredConditionJson, SolveStepJson, SolveSubStepJson, WarningJson};
use cas_ast::{Context, ExprId, RelOp, SolutionSet};
use cas_formatter::{DisplayExpr, LaTeXExpr};

pub(crate) fn format_eval_input_latex(ctx: &Context, parsed: ExprId) -> String {
    if let Some((lhs, rhs)) = cas_ast::eq::unwrap_eq(ctx, parsed) {
        let lhs_latex = LaTeXExpr {
            context: ctx,
            id: lhs,
        }
        .to_latex();
        let rhs_latex = LaTeXExpr {
            context: ctx,
            id: rhs,
        }
        .to_latex();
        format!("{lhs_latex} = {rhs_latex}")
    } else {
        LaTeXExpr {
            context: ctx,
            id: parsed,
        }
        .to_latex()
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

pub(crate) fn collect_solve_steps_eval_json(
    solve_steps: &[cas_solver::SolveStep],
    ctx: &Context,
    steps_mode: &str,
) -> Vec<SolveStepJson> {
    if steps_mode != "on" {
        return vec![];
    }

    let filtered: Vec<_> = solve_steps
        .iter()
        .filter(|step| step.importance >= cas_solver::ImportanceLevel::Medium)
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
            let equation_str = format!("{lhs_str} {relop_str} {rhs_str}");

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
                    let ss_equation_str = format!("{ss_lhs_str} {ss_relop_str} {ss_rhs_str}");

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

pub(crate) fn format_solution_set_eval_json(ctx: &Context, solution_set: &SolutionSet) -> String {
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
                if cas_solver::is_pure_residual_otherwise(case) {
                    continue;
                }
                let cond_str = cas_formatter::condition_set_to_display(&case.when, ctx);
                let inner_str = format_solution_set_eval_json(ctx, &case.then.solutions);
                if case.when.is_empty() {
                    parts.push(format!("{inner_str} otherwise"));
                } else {
                    parts.push(format!("{inner_str} if {cond_str}"));
                }
            }

            if parts.is_empty() && !cases.is_empty() {
                for case in cases {
                    if !case.when.is_empty() {
                        let inner_str = format_solution_set_eval_json(ctx, &case.then.solutions);
                        let cond_str = cas_formatter::condition_set_to_display(&case.when, ctx);
                        return format!("{inner_str} if {cond_str}");
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
            parts.join(" U ")
        }
        SolutionSet::Residual(expr) => {
            let expr_str = format!(
                "{}",
                DisplayExpr {
                    context: ctx,
                    id: *expr
                }
            );
            format!("Solve: {expr_str} = 0")
        }
    }
}

pub(crate) fn solution_set_to_latex_eval_json(ctx: &Context, solution_set: &SolutionSet) -> String {
    match solution_set {
        SolutionSet::Empty => r"\emptyset".to_string(),
        SolutionSet::AllReals => r"\mathbb{R}".to_string(),
        SolutionSet::Discrete(exprs) => {
            if exprs.is_empty() {
                r"\emptyset".to_string()
            } else {
                let solutions: Vec<String> = exprs
                    .iter()
                    .map(|e| {
                        LaTeXExpr {
                            context: ctx,
                            id: *e,
                        }
                        .to_latex()
                    })
                    .collect();
                format!(r"\left\{{ {} \right\}}", solutions.join(", "))
            }
        }
        SolutionSet::Conditional(cases) => {
            let mut latex_parts = Vec::new();
            for case in cases {
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

pub(crate) fn collect_warnings_eval_json(
    domain_warnings: &[cas_solver::DomainWarning],
) -> Vec<WarningJson> {
    domain_warnings
        .iter()
        .map(|w| WarningJson {
            rule: w.rule_name.clone(),
            assumption: w.message.clone(),
        })
        .collect()
}

pub(crate) fn collect_required_conditions_eval_json(
    required_conditions: &[cas_solver::ImplicitCondition],
    ctx: &Context,
) -> Vec<RequiredConditionJson> {
    required_conditions
        .iter()
        .map(|cond| {
            let (kind, expr_id) = match cond {
                cas_solver::ImplicitCondition::NonNegative(e) => ("NonNegative", *e),
                cas_solver::ImplicitCondition::Positive(e) => ("Positive", *e),
                cas_solver::ImplicitCondition::NonZero(e) => ("NonZero", *e),
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

pub(crate) fn collect_required_display_eval_json(
    required_conditions: &[cas_solver::ImplicitCondition],
    ctx: &Context,
) -> Vec<String> {
    required_conditions
        .iter()
        .map(|cond| cond.display(ctx))
        .collect()
}
