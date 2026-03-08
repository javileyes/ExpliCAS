use std::sync::{Arc, Mutex, MutexGuard};

/// Collects solve events emitted or derived during solve runs.
#[derive(Debug, Clone)]
pub struct SolverEventCollector<Equation, Importance> {
    events: Arc<Mutex<Vec<crate::solver_events::SolverEvent<Equation, Importance>>>>,
}

impl<Equation, Importance> Default for SolverEventCollector<Equation, Importance> {
    fn default() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl<Equation, Importance> SolverEventCollector<Equation, Importance> {
    /// Create an empty collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of captured events.
    pub fn len(&self) -> usize {
        self.locked_events().len()
    }

    /// Whether the collector is empty.
    pub fn is_empty(&self) -> bool {
        self.locked_events().is_empty()
    }

    fn locked_events(
        &self,
    ) -> MutexGuard<'_, Vec<crate::solver_events::SolverEvent<Equation, Importance>>> {
        self.events
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

impl<Equation: Clone, Importance: Clone> SolverEventCollector<Equation, Importance> {
    /// Access captured events.
    pub fn events(&self) -> Vec<crate::solver_events::SolverEvent<Equation, Importance>> {
        self.locked_events().clone()
    }

    /// Consume and return captured events.
    pub fn into_events(self) -> Vec<crate::solver_events::SolverEvent<Equation, Importance>> {
        self.events()
    }
}

impl<Equation: Clone, Importance: Clone>
    crate::solver_events::SolveEventListener<Equation, Importance>
    for SolverEventCollector<Equation, Importance>
{
    fn on_event(&mut self, event: &crate::solver_events::SolverEvent<Equation, Importance>) {
        self.locked_events().push(event.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::SolverEventCollector;
    use crate::solver_events::{SolveEventListener, SolverEvent};

    #[test]
    fn collector_captures_solver_events() {
        let mut collector = SolverEventCollector::<u32, u8>::new();
        assert!(collector.is_empty());

        collector.on_event(&SolverEvent::StepProduced {
            description: "step".into(),
            equation_after: 1,
            importance: 2,
        });
        collector.on_event(&SolverEvent::SubstepProduced {
            description: "substep".into(),
            equation_after: 2,
            importance: 1,
        });

        let events = collector.events();
        assert_eq!(collector.len(), 2);
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], SolverEvent::StepProduced { .. }));
        assert!(matches!(events[1], SolverEvent::SubstepProduced { .. }));
    }
}
