use std::sync::{Arc, Mutex, MutexGuard};

/// Collects engine events emitted during simplification or solve runs.
#[derive(Debug, Default, Clone)]
pub struct EngineEventCollector {
    events: Arc<Mutex<Vec<crate::engine_events::EngineEvent>>>,
}

impl EngineEventCollector {
    /// Create an empty collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Access captured events.
    pub fn events(&self) -> Vec<crate::engine_events::EngineEvent> {
        self.locked_events().clone()
    }

    /// Consume and return captured events.
    pub fn into_events(self) -> Vec<crate::engine_events::EngineEvent> {
        self.events()
    }

    fn locked_events(&self) -> MutexGuard<'_, Vec<crate::engine_events::EngineEvent>> {
        self.events.lock().expect("engine event collector poisoned")
    }
}

impl crate::engine_events::StepListener for EngineEventCollector {
    fn on_event(&mut self, event: &crate::engine_events::EngineEvent) {
        self.locked_events().push(event.clone());
    }
}
