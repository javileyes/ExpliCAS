mod equation;

use cas_api_models::{SolveStepWire, SolveSubStepWire};
use cas_ast::Context;

use self::equation::{relop_to_latex, render_equation_strings};

pub(crate) fn collect_output_solve_steps(
    solve_steps: &[crate::SolveStep],
    prepended_primary_steps: &[crate::Step],
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

    let prepended_substeps = collect_prepended_primary_substeps(prepended_primary_steps, ctx);

    filtered
        .iter()
        .enumerate()
        .map(|(i, step)| {
            let rendered = render_equation_strings(ctx, &step.equation_after);

            let mut substeps: Vec<SolveSubStepWire> = step
                .substeps
                .iter()
                .map(|ss| {
                    let rendered = render_equation_strings(ctx, &ss.equation_after);

                    SolveSubStepWire {
                        index: String::new(),
                        description: ss.description.clone(),
                        equation: rendered.equation,
                        lhs_latex: rendered.lhs_latex,
                        relop: relop_to_latex(&ss.equation_after.op),
                        rhs_latex: rendered.rhs_latex,
                    }
                })
                .collect();

            if i == 0 && !prepended_substeps.is_empty() {
                let mut merged = prepended_substeps.clone();
                merged.extend(substeps);
                substeps = merged;
            }

            for (j, substep) in substeps.iter_mut().enumerate() {
                substep.index = format!("{}.{}", i + 1, j + 1);
            }

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

fn collect_prepended_primary_substeps(
    primary_steps: &[crate::Step],
    ctx: &Context,
) -> Vec<SolveSubStepWire> {
    primary_steps
        .iter()
        .filter_map(|step| primary_step_to_solve_substep(step, ctx))
        .collect()
}

fn primary_step_to_solve_substep(step: &crate::Step, ctx: &Context) -> Option<SolveSubStepWire> {
    let equation_after_id = step.global_after.or(Some(step.after))?;
    let (lhs, rhs) = cas_ast::eq::unwrap_eq(ctx, equation_after_id)?;
    let equation_after = cas_ast::Equation {
        lhs,
        rhs,
        op: cas_ast::RelOp::Eq,
    };
    let rendered = render_equation_strings(ctx, &equation_after);

    Some(SolveSubStepWire {
        index: String::new(),
        description: step.description.to_string(),
        equation: rendered.equation,
        lhs_latex: rendered.lhs_latex,
        relop: relop_to_latex(&equation_after.op),
        rhs_latex: rendered.rhs_latex,
    })
}

#[cfg(test)]
mod tests {
    use super::collect_output_solve_steps;
    use cas_ast::{eq::wrap_eq, Context, Equation, RelOp};
    use cas_solver_core::step_types::SubStep;

    #[test]
    fn prepends_primary_solve_prep_steps_as_substeps_of_first_solve_step() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let one = ctx.num(1);
        let x_plus_one = ctx.add(cas_ast::Expr::Add(x, one));
        let eq_after = Equation {
            lhs: x_plus_one,
            rhs: zero,
            op: RelOp::Eq,
        };
        let mut primary = crate::Step::new(
            "Factorizar antes de aislar",
            "Factor",
            x,
            x_plus_one,
            Vec::<cas_engine::PathStep>::new(),
            Some(&ctx),
        );
        primary.importance = crate::ImportanceLevel::High;
        primary.global_after = Some(wrap_eq(&mut ctx, eq_after.lhs, eq_after.rhs));

        let solve_step = crate::SolveStep::new(
            "Aislar la variable",
            Equation {
                lhs: x,
                rhs: zero,
                op: RelOp::Eq,
            },
            crate::ImportanceLevel::High,
        )
        .with_substeps(vec![crate::SolveSubStep::new(
            "Mover el término constante",
            Equation {
                lhs: x,
                rhs: one,
                op: RelOp::Eq,
            },
            crate::ImportanceLevel::Medium,
        )]);

        let wires = collect_output_solve_steps(&[solve_step], &[primary], &ctx, "on");

        assert_eq!(wires.len(), 1);
        assert_eq!(wires[0].substeps.len(), 2);
        assert_eq!(wires[0].substeps[0].index, "1.1");
        assert_eq!(
            wires[0].substeps[0].description,
            "Factorizar antes de aislar"
        );
        assert_eq!(wires[0].substeps[0].equation, "x + 1 = 0");
        assert_eq!(wires[0].substeps[1].index, "1.2");
        assert_eq!(
            wires[0].substeps[1].description,
            "Mover el término constante"
        );
    }

    #[test]
    fn skips_primary_steps_without_global_equation_snapshot() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_plus_one = ctx.add(cas_ast::Expr::Add(x, one));
        let mut primary = crate::Step::new(
            "Reescribir el lado izquierdo",
            "Rewrite",
            x,
            x_plus_one,
            Vec::<cas_engine::PathStep>::new(),
            Some(&ctx),
        );
        primary.importance = crate::ImportanceLevel::High;
        primary.meta_mut().substeps.push(SubStep::with_importance(
            "Mostrar intermedio",
            vec!["Agrupar términos".to_string()],
            crate::ImportanceLevel::Medium,
        ));

        let solve_step = crate::SolveStep::new(
            "Aislar la variable",
            Equation {
                lhs: x,
                rhs: one,
                op: RelOp::Eq,
            },
            crate::ImportanceLevel::High,
        );

        let wires = collect_output_solve_steps(&[solve_step], &[primary], &ctx, "on");

        assert_eq!(wires.len(), 1);
        assert!(wires[0].substeps.is_empty());
    }
}
