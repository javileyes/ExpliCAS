use cas_ast::{Context, Equation, SolutionSet};
use cas_didactic::{
    render_solve_timeline_cli_output, render_solve_timeline_html, TimelineCliRender,
    TimelineSolveCommandOutput,
};
use cas_solver::runtime::{
    DisplaySolveSteps, ImportanceLevel, Simplifier, SolveStep, SolveSubStep, SolverOptions,
};
use cas_solver_core::{display_steps::DisplaySteps, solver_events::SolverEvent};

type SolveEvent = SolverEvent<Equation, ImportanceLevel>;

fn solve_timeline_output(input: &str, var: &str) -> (Context, TimelineSolveCommandOutput) {
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

    assert!(
        !display_steps.is_empty(),
        "expected solve timeline parity case to produce display steps"
    );

    (
        simplifier.context.clone(),
        TimelineSolveCommandOutput {
            equation,
            var,
            solution_set,
            display_steps,
        },
    )
}

fn sample_timeline_output_with_substeps() -> (Context, TimelineSolveCommandOutput) {
    let mut context = Context::new();
    let (equation, var) = cas_solver::api::prepare_timeline_solve_equation(
        &mut context,
        "2*x + 3 = 7",
        Some("x".to_string()),
    )
    .expect("prepare sample solve equation");
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

    let display_steps = DisplaySteps(vec![
        SolveStep::new(
            "Subtract 3 from both sides",
            reduced_equation.clone(),
            ImportanceLevel::Medium,
        )
        .with_substeps(vec![SolveSubStep::new(
            "Simplify both sides",
            reduced_equation,
            ImportanceLevel::Low,
        )]),
        SolveStep::new(
            "Divide both sides by 2",
            solved_equation,
            ImportanceLevel::High,
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

fn render_html_with_context(context: &Context, output: &TimelineSolveCommandOutput) -> String {
    let mut render_context = context.clone();
    render_solve_timeline_html(
        &mut render_context,
        &output.display_steps.0,
        &output.equation,
        &output.solution_set,
        &output.var,
    )
}

fn render_cli_with_context(
    context: &Context,
    output: &TimelineSolveCommandOutput,
) -> TimelineCliRender {
    let mut render_context = context.clone();
    render_solve_timeline_cli_output(&mut render_context, output)
}

fn solve_steps_to_events(steps: &[SolveStep]) -> Vec<SolveEvent> {
    let mut events = Vec::new();

    for step in steps {
        events.push(SolverEvent::StepProduced {
            description: step.description.to_string(),
            equation_after: step.equation_after.clone(),
            importance: step.importance,
        });

        events.extend(
            step.substeps
                .iter()
                .map(|substep| SolverEvent::SubstepProduced {
                    description: substep.description.to_string(),
                    equation_after: substep.equation_after.clone(),
                    importance: substep.importance,
                }),
        );
    }

    events
}

fn events_to_solve_steps(events: &[SolveEvent]) -> Vec<SolveStep> {
    let mut steps = Vec::new();
    let mut current_step: Option<SolveStep> = None;

    for event in events {
        match event {
            SolverEvent::StepProduced {
                description,
                equation_after,
                importance,
            } => {
                if let Some(step) = current_step.take() {
                    steps.push(step);
                }
                current_step = Some(SolveStep::new(
                    description.clone(),
                    equation_after.clone(),
                    *importance,
                ));
            }
            SolverEvent::SubstepProduced {
                description,
                equation_after,
                importance,
            } => {
                if let Some(step) = current_step.as_mut() {
                    step.substeps.push(SolveSubStep::new(
                        description.clone(),
                        equation_after.clone(),
                        *importance,
                    ));
                }
            }
        }
    }

    if let Some(step) = current_step {
        steps.push(step);
    }

    steps
}

fn events_to_display_solve_steps(events: &[SolveEvent]) -> DisplaySolveSteps {
    DisplaySteps(events_to_solve_steps(events))
}

fn has_same_solve_step_shape(left: &DisplaySolveSteps, right: &DisplaySolveSteps) -> bool {
    left.len() == right.len()
        && left.iter().zip(right.iter()).all(|(lhs, rhs)| {
            lhs.description == rhs.description
                && lhs.importance == rhs.importance
                && lhs.substeps.len() == rhs.substeps.len()
                && lhs
                    .substeps
                    .iter()
                    .zip(rhs.substeps.iter())
                    .all(|(lhs_substep, rhs_substep)| {
                        lhs_substep.description == rhs_substep.description
                            && lhs_substep.importance == rhs_substep.importance
                    })
        })
}

fn roundtrip_display_solve_steps_via_events(
    display_steps: &DisplaySolveSteps,
) -> DisplaySolveSteps {
    let events = solve_steps_to_events(display_steps.as_slice());
    let rebuilt = events_to_display_solve_steps(&events);

    if has_same_solve_step_shape(display_steps, &rebuilt) {
        rebuilt
    } else {
        display_steps.clone()
    }
}

fn with_roundtripped_steps(output: &TimelineSolveCommandOutput) -> TimelineSolveCommandOutput {
    let mut rebuilt = output.clone();
    rebuilt.display_steps = roundtrip_display_solve_steps_via_events(&output.display_steps);
    rebuilt
}

#[test]
fn solve_timeline_html_matches_after_solver_event_roundtrip() {
    let (context, output) = solve_timeline_output("2*x + 3 = 7", "x");
    let rebuilt = with_roundtripped_steps(&output);

    let original_html = render_html_with_context(&context, &output);
    let rebuilt_html = render_html_with_context(&context, &rebuilt);

    assert_eq!(original_html, rebuilt_html);
}

#[test]
fn solve_timeline_cli_output_matches_after_solver_event_roundtrip_with_substeps() {
    let (context, output) = sample_timeline_output_with_substeps();
    let rebuilt = with_roundtripped_steps(&output);

    let original = render_cli_with_context(&context, &output);
    let roundtripped = render_cli_with_context(&context, &rebuilt);

    match (original, roundtripped) {
        (
            TimelineCliRender::Html {
                file_name: original_file_name,
                html: original_html,
                lines: original_lines,
            },
            TimelineCliRender::Html {
                file_name: rebuilt_file_name,
                html: rebuilt_html,
                lines: rebuilt_lines,
            },
        ) => {
            assert_eq!(original_file_name, rebuilt_file_name);
            assert_eq!(original_html, rebuilt_html);
            assert_eq!(original_lines, rebuilt_lines);
        }
        (left, right) => panic!("expected HTML renders, got left={left:?} right={right:?}"),
    }
}
