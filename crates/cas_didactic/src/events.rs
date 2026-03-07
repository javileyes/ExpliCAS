use std::sync::{Arc, Mutex};

/// Collects engine events emitted during simplification/solve runs.
///
/// This is a minimal bridge for the observer migration: frontends can install
/// this listener in `Simplifier` and later consume the captured events to build
/// timeline or didactic output.
#[derive(Debug, Default, Clone)]
pub struct EngineEventCollector {
    events: Arc<Mutex<Vec<cas_solver_core::engine_events::EngineEvent>>>,
}

impl EngineEventCollector {
    /// Create an empty collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Access captured events.
    pub fn events(&self) -> Vec<cas_solver_core::engine_events::EngineEvent> {
        self.locked_events().clone()
    }

    /// Consume and return captured events.
    pub fn into_events(self) -> Vec<cas_solver_core::engine_events::EngineEvent> {
        self.events()
    }
}

impl cas_solver_core::engine_events::StepListener for EngineEventCollector {
    fn on_event(&mut self, event: &cas_solver_core::engine_events::EngineEvent) {
        self.locked_events().push(event.clone());
    }
}

#[cfg(test)]
mod tests;

impl EngineEventCollector {
    fn locked_events(
        &self,
    ) -> std::sync::MutexGuard<'_, Vec<cas_solver_core::engine_events::EngineEvent>> {
        self.events.lock().expect("engine event collector poisoned")
    }
}
