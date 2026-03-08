/// Solve-level events emitted or derived during equation solving.
///
/// These events are equation-aware and intentionally minimal. They model the
/// stable didactic surface of solve flows without exposing low-level runtime
/// details such as failed attempts or branch-internal mechanics.
#[derive(Debug, Clone, PartialEq)]
pub enum SolverEvent<Equation, Importance> {
    /// One primary solve step was produced.
    StepProduced {
        description: String,
        equation_after: Equation,
        importance: Importance,
    },
    /// One subordinate solve substep was produced.
    SubstepProduced {
        description: String,
        equation_after: Equation,
        importance: Importance,
    },
}

/// Observer trait for consumers that want to listen to solve events.
pub trait SolveEventListener<Equation, Importance> {
    /// Receive one event emitted by the solve layer.
    fn on_event(&mut self, event: &SolverEvent<Equation, Importance>);
}

#[cfg(test)]
mod tests {
    use super::SolverEvent;

    #[test]
    fn solver_event_variants_preserve_payload() {
        let step = SolverEvent::StepProduced {
            description: "isolate x".into(),
            equation_after: 11u32,
            importance: 2u8,
        };
        let substep = SolverEvent::SubstepProduced {
            description: "subtract both sides".into(),
            equation_after: 22u32,
            importance: 1u8,
        };

        match step {
            SolverEvent::StepProduced {
                description,
                equation_after,
                importance,
            } => {
                assert_eq!(description, "isolate x");
                assert_eq!(equation_after, 11u32);
                assert_eq!(importance, 2u8);
            }
            _ => panic!("expected step event"),
        }

        match substep {
            SolverEvent::SubstepProduced {
                description,
                equation_after,
                importance,
            } => {
                assert_eq!(description, "subtract both sides");
                assert_eq!(equation_after, 22u32);
                assert_eq!(importance, 1u8);
            }
            _ => panic!("expected substep event"),
        }
    }
}
