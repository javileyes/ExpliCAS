use cas_api_models::{SolveStepJson, SolveSubStepJson};
use cas_ast::{Context, RelOp};
use cas_formatter::{DisplayExpr, LaTeXExpr};

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
