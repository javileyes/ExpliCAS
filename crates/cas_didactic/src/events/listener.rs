use super::EngineEventCollector;

impl cas_solver_core::engine_events::StepListener for EngineEventCollector {
    fn on_event(&mut self, event: &cas_solver_core::engine_events::EngineEvent) {
        self.locked_events().push(event.clone());
    }
}
