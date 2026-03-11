mod equation;

use cas_api_models::{SolveStepWire, SolveSubStepWire};
use cas_ast::Context;

use self::equation::{relop_to_latex, render_equation_strings};

pub(crate) fn collect_output_solve_steps(
    solve_steps: &[crate::SolveStep],
    ctx: &Context,
    steps_mode: &str,
) -> Vec<SolveStepWire> {
    if steps_mode != "on" {
        return vec![];
    }

    let filtered: Vec<_> = solve_steps
        .iter()
        .filter(|step| step.importance >= crate::ImportanceLevel::Medium)
        .collect();

    if filtered.is_empty() {
        return vec![];
    }

    filtered
        .iter()
        .enumerate()
        .map(|(i, step)| {
            let rendered = render_equation_strings(ctx, &step.equation_after);

            let substeps: Vec<SolveSubStepWire> = step
                .substeps
                .iter()
                .enumerate()
                .map(|(j, ss)| {
                    let rendered = render_equation_strings(ctx, &ss.equation_after);

                    SolveSubStepWire {
                        index: format!("{}.{}", i + 1, j + 1),
                        description: ss.description.clone(),
                        equation: rendered.equation,
                        lhs_latex: rendered.lhs_latex,
                        relop: relop_to_latex(&ss.equation_after.op),
                        rhs_latex: rendered.rhs_latex,
                    }
                })
                .collect();

            SolveStepWire {
                index: i + 1,
                description: step.description.clone(),
                equation: rendered.equation,
                lhs_latex: rendered.lhs_latex,
                relop: relop_to_latex(&step.equation_after.op),
                rhs_latex: rendered.rhs_latex,
                substeps,
            }
        })
        .collect()
}
