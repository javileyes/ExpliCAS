use std::hint::black_box;
use std::time::Duration;

use cas_ast::{Context, SolutionSet};
use cas_didactic::{
    collect_step_payloads, render_simplify_timeline_cli_output, render_simplify_timeline_html,
    render_solve_timeline_cli_output, render_solve_timeline_html, TimelineSimplifyCommandOutput,
    TimelineSolveCommandOutput, VerbosityLevel,
};
use cas_solver::runtime::{to_display_steps, Simplifier, SolverOptions};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(25);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
}

fn simplify_fixture(expr: &str) -> (Context, TimelineSimplifyCommandOutput) {
    let mut simplifier = Simplifier::with_default_rules();
    let parsed_expr = cas_parser::parse(expr, &mut simplifier.context).expect("parse failed");
    let (simplified_expr, raw_steps) = simplifier.simplify(parsed_expr);
    let steps = to_display_steps(raw_steps);
    let ctx = simplifier.context;

    (
        ctx,
        TimelineSimplifyCommandOutput {
            expr_input: expr.to_string(),
            use_aggressive: false,
            parsed_expr,
            simplified_expr,
            steps,
        },
    )
}

fn solve_fixture(input: &str, var: &str) -> (Context, TimelineSolveCommandOutput) {
    let mut simplifier = Simplifier::with_default_rules();
    let (equation, var) = cas_solver::api::prepare_timeline_solve_equation(
        &mut simplifier.context,
        input,
        Some(var.to_string()),
    )
    .expect("prepare timeline solve equation");

    let (solution_set, display_steps, _) = cas_solver::api::solve_with_display_steps(
        &equation,
        &var,
        &mut simplifier,
        SolverOptions::default(),
    )
    .expect("solve with display steps");

    let ctx = simplifier.context;
    (
        ctx,
        TimelineSolveCommandOutput {
            equation,
            var,
            solution_set,
            display_steps,
        },
    )
}

fn sample_solve_output_with_substeps() -> (Context, TimelineSolveCommandOutput) {
    let mut context = Context::new();
    let (equation, var) = cas_solver::api::prepare_timeline_solve_equation(
        &mut context,
        "2*x + 3 = 7",
        Some("x".to_string()),
    )
    .expect("prepare solve equation");
    let reduced_equation = cas_solver::api::prepare_timeline_solve_equation(
        &mut context,
        "2*x = 4",
        Some("x".to_string()),
    )
    .expect("prepare reduced equation")
    .0;
    let solved_equation = cas_solver::api::prepare_timeline_solve_equation(
        &mut context,
        "x = 2",
        Some("x".to_string()),
    )
    .expect("prepare solved equation")
    .0;

    let display_steps: cas_solver::runtime::DisplaySolveSteps =
        cas_solver_core::display_steps::DisplaySteps(vec![
            cas_solver::runtime::SolveStep::new(
                "Subtract 3 from both sides",
                reduced_equation.clone(),
                cas_solver::runtime::ImportanceLevel::Medium,
            )
            .with_substeps(vec![cas_solver::runtime::SolveSubStep::new(
                "Simplify both sides",
                reduced_equation,
                cas_solver::runtime::ImportanceLevel::Low,
            )]),
            cas_solver::runtime::SolveStep::new(
                "Divide both sides by 2",
                solved_equation,
                cas_solver::runtime::ImportanceLevel::High,
            ),
        ]);

    (
        context.clone(),
        TimelineSolveCommandOutput {
            equation,
            var,
            solution_set: SolutionSet::Discrete(vec![context.num(2)]),
            display_steps,
        },
    )
}

fn bench_frontend_didactic(c: &mut Criterion) {
    let mut group = c.benchmark_group("frontend_didactic");
    configure_group(&mut group);

    let simplify_cases = [
        (
            "gcd/scalar_multiple_fraction",
            simplify_fixture("(2*x + 2*y)/(4*x + 4*y)"),
        ),
        (
            "trig/pythagorean_chain",
            simplify_fixture("sin(2*x + 1)^2 + cos(1 + 2*x)^2"),
        ),
        ("heavy/nested_root", simplify_fixture("sqrt(12*x^3)")),
    ];

    for (name, (ctx, out)) in &simplify_cases {
        group.bench_with_input(BenchmarkId::new("step_payloads", name), out, |b, out| {
            b.iter(|| black_box(collect_step_payloads(&out.steps, ctx, "on")))
        });

        group.bench_with_input(
            BenchmarkId::new("simplify_html_normal", name),
            out,
            |b, out| {
                b.iter_batched(
                    || ctx.clone(),
                    |mut ctx| {
                        black_box(render_simplify_timeline_html(
                            &mut ctx,
                            &out.steps,
                            out.parsed_expr,
                            Some(out.simplified_expr),
                            VerbosityLevel::Normal,
                            Some(out.expr_input.as_str()),
                        ))
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simplify_cli_normal", name),
            out,
            |b, out| {
                b.iter_batched(
                    || ctx.clone(),
                    |mut ctx| {
                        black_box(render_simplify_timeline_cli_output(
                            &mut ctx,
                            out,
                            VerbosityLevel::Normal,
                        ))
                    },
                    BatchSize::SmallInput,
                )
            },
        );
    }

    let solve_cases = [
        ("linear/real_solver", solve_fixture("2*x + 3 = 7", "x")),
        (
            "linear/substeps_fixture",
            sample_solve_output_with_substeps(),
        ),
    ];

    for (name, (ctx, out)) in &solve_cases {
        group.bench_with_input(BenchmarkId::new("solve_html", name), out, |b, out| {
            b.iter_batched(
                || ctx.clone(),
                |mut ctx| {
                    black_box(render_solve_timeline_html(
                        &mut ctx,
                        &out.display_steps.0,
                        &out.equation,
                        &out.solution_set,
                        &out.var,
                    ))
                },
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("solve_cli", name), out, |b, out| {
            b.iter_batched(
                || ctx.clone(),
                |mut ctx| black_box(render_solve_timeline_cli_output(&mut ctx, out)),
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, bench_frontend_didactic);
criterion_main!(benches);
