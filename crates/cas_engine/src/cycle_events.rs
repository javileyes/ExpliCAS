//! Compatibility facade for cycle event registry.
//!
//! The implementation lives in `cas_solver_core::cycle_event_registry`.
#![allow(dead_code)]

pub type CycleEvent = cas_solver_core::cycle_models::CycleEvent;
pub type CycleLevel = cas_solver_core::cycle_models::CycleLevel;

#[inline]
pub fn register_cycle_event(event: CycleEvent) {
    cas_solver_core::cycle_event_registry::register_cycle_event(event);
}

#[inline]
pub fn take_cycle_events() -> Vec<CycleEvent> {
    cas_solver_core::cycle_event_registry::take_cycle_events()
}

#[inline]
pub fn clear_cycle_events() {
    cas_solver_core::cycle_event_registry::clear_cycle_events();
}

#[inline]
pub fn truncate_display(s: &str, max_len: usize) -> String {
    cas_solver_core::cycle_event_registry::truncate_display(s, max_len)
}
