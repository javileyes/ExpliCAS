use cas_solver_core::solver_events::SolverEvent;

type SolveEvent = SolverEvent<cas_ast::Equation, crate::ImportanceLevel>;

pub(crate) fn solve_steps_to_events(steps: &[crate::SolveStep]) -> Vec<SolveEvent> {
    let mut events = Vec::new();

    for step in steps {
        events.push(SolverEvent::StepProduced {
            description: step.description.clone(),
            equation_after: step.equation_after.clone(),
            importance: step.importance.clone(),
        });

        events.extend(
            step.substeps
                .iter()
                .map(|substep| SolverEvent::SubstepProduced {
                    description: substep.description.clone(),
                    equation_after: substep.equation_after.clone(),
                    importance: substep.importance.clone(),
                }),
        );
    }

    events
}

pub(crate) fn display_solve_steps_to_events(
    display_steps: &crate::DisplaySolveSteps,
) -> Vec<SolveEvent> {
    solve_steps_to_events(display_steps.as_slice())
}

pub(crate) fn events_to_solve_steps(events: &[SolveEvent]) -> Vec<crate::SolveStep> {
    let mut steps = Vec::new();
    let mut current_step: Option<crate::SolveStep> = None;

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
                current_step = Some(crate::SolveStep::new(
                    description.clone(),
                    equation_after.clone(),
                    importance.clone(),
                ));
            }
            SolverEvent::SubstepProduced {
                description,
                equation_after,
                importance,
            } => {
                if let Some(step) = current_step.as_mut() {
                    step.substeps.push(crate::SolveSubStep::new(
                        description.clone(),
                        equation_after.clone(),
                        importance.clone(),
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

pub(crate) fn events_to_display_solve_steps(events: &[SolveEvent]) -> crate::DisplaySolveSteps {
    cas_solver_core::display_steps::DisplaySteps(events_to_solve_steps(events))
}

pub(crate) fn roundtrip_display_solve_steps_via_events(
    display_steps: &crate::DisplaySolveSteps,
) -> crate::DisplaySolveSteps {
    let events = display_solve_steps_to_events(display_steps);
    let rebuilt = events_to_display_solve_steps(&events);

    if has_same_solve_step_shape(display_steps, &rebuilt) {
        rebuilt
    } else {
        display_steps.clone()
    }
}

fn has_same_solve_step_shape(
    left: &crate::DisplaySolveSteps,
    right: &crate::DisplaySolveSteps,
) -> bool {
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

#[cfg(test)]
mod tests {
    use super::{
        display_solve_steps_to_events, events_to_display_solve_steps, events_to_solve_steps,
        roundtrip_display_solve_steps_via_events, solve_steps_to_events,
    };
    use cas_ast::{Equation, RelOp};

    fn eq(lhs: i64, rhs: i64) -> Equation {
        let mut ctx = cas_ast::Context::new();
        Equation {
            lhs: ctx.num(lhs),
            rhs: ctx.num(rhs),
            op: RelOp::Eq,
        }
    }

    fn sample_steps() -> Vec<crate::SolveStep> {
        vec![
            crate::SolveStep::new("isolate x", eq(2, 1), crate::ImportanceLevel::Medium)
                .with_substeps(vec![
                    crate::SolveSubStep::new("subtract 1", eq(1, 0), crate::ImportanceLevel::Low),
                    crate::SolveSubStep::new("simplify", eq(1, 0), crate::ImportanceLevel::Medium),
                ]),
            crate::SolveStep::new("divide by 2", eq(1, 0), crate::ImportanceLevel::High),
        ]
    }

    #[test]
    fn roundtrip_solve_steps_through_solver_events() {
        let steps = sample_steps();
        let events = solve_steps_to_events(&steps);
        let rebuilt = events_to_solve_steps(&events);

        assert_eq!(rebuilt.len(), 2);
        assert_eq!(rebuilt[0].description, "isolate x");
        assert_eq!(rebuilt[0].importance, crate::ImportanceLevel::Medium);
        assert_eq!(rebuilt[0].substeps.len(), 2);
        assert_eq!(rebuilt[0].substeps[0].description, "subtract 1");
        assert_eq!(
            rebuilt[0].substeps[0].importance,
            crate::ImportanceLevel::Low
        );
        assert_eq!(rebuilt[1].description, "divide by 2");
        assert_eq!(rebuilt[1].importance, crate::ImportanceLevel::High);
        assert!(rebuilt[1].substeps.is_empty());
    }

    #[test]
    fn roundtrip_display_solve_steps_through_solver_events() {
        let display_steps = cas_solver_core::display_steps::DisplaySteps(sample_steps());
        let events = display_solve_steps_to_events(&display_steps);
        let rebuilt = events_to_display_solve_steps(&events);

        assert_eq!(rebuilt.len(), 2);
        assert_eq!(rebuilt[0].description, "isolate x");
        assert_eq!(rebuilt[0].substeps.len(), 2);
        assert_eq!(rebuilt[1].description, "divide by 2");
    }

    #[test]
    fn ignores_orphan_substeps_without_panicking() {
        let rebuilt = events_to_solve_steps(&[SolverEvent::SubstepProduced {
            description: "orphan".to_string(),
            equation_after: eq(1, 1),
            importance: crate::ImportanceLevel::Low,
        }]);

        assert!(rebuilt.is_empty());
    }

    #[test]
    fn roundtrip_display_steps_via_events_preserves_shape() {
        let display_steps = cas_solver_core::display_steps::DisplaySteps(sample_steps());
        let rebuilt = roundtrip_display_solve_steps_via_events(&display_steps);

        assert_eq!(rebuilt.len(), 2);
        assert_eq!(rebuilt[0].description, "isolate x");
        assert_eq!(rebuilt[0].substeps.len(), 2);
        assert_eq!(rebuilt[1].description, "divide by 2");
    }
}
