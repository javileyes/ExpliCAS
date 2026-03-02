use cas_ast::{Context, Equation, RelOp};

#[test]
fn solve_step_lines_render_without_substeps_in_non_verbose_mode() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);

    let steps = vec![cas_solver::SolveStep::new(
        "Isolate x",
        Equation {
            lhs: x,
            rhs: one,
            op: RelOp::Eq,
        },
        cas_solver::ImportanceLevel::Medium,
    )];

    let lines = cas_solver::format_solve_steps_lines(&ctx, &steps, &[], false);
    assert_eq!(lines.len(), 2);
    assert!(lines[0].contains("1. Isolate x"));
    assert!(lines[1].contains("->"));
}

#[test]
fn solve_step_lines_include_substeps_in_verbose_mode() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.num(1);
    let two = ctx.num(2);

    let step = cas_solver::SolveStep::new(
        "Subtract 1 from both sides",
        Equation {
            lhs: x,
            rhs: one,
            op: RelOp::Eq,
        },
        cas_solver::ImportanceLevel::Medium,
    )
    .with_substeps(vec![cas_solver::SolveSubStep::new(
        "Arithmetic simplification",
        Equation {
            lhs: x,
            rhs: two,
            op: RelOp::Eq,
        },
        cas_solver::ImportanceLevel::Low,
    )]);

    let lines = cas_solver::format_solve_steps_lines(&ctx, &[step], &[], true);
    assert!(lines.iter().any(|line| line.contains("1.1.")));
    assert!(lines
        .iter()
        .any(|line| line.contains("Arithmetic simplification")));
}
