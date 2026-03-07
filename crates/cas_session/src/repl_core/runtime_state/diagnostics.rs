use crate::ReplCore;
use cas_solver::PipelineStats;

impl ReplCore {
    /// Return whether debug mode is enabled.
    pub(crate) fn debug_mode(&self) -> bool {
        self.debug_mode
    }

    /// Set debug mode flag.
    pub(crate) fn set_debug_mode(&mut self, value: bool) {
        self.debug_mode = value;
    }

    /// Return whether health tracking is enabled.
    pub(crate) fn health_enabled(&self) -> bool {
        self.health_enabled
    }

    /// Set health tracking flag.
    pub(crate) fn set_health_enabled(&mut self, value: bool) {
        self.health_enabled = value;
    }

    /// Borrow latest pipeline stats.
    pub(crate) fn last_stats(&self) -> Option<&PipelineStats> {
        self.last_stats.as_ref()
    }

    /// Replace latest pipeline stats.
    pub(crate) fn set_last_stats(&mut self, value: Option<PipelineStats>) {
        self.last_stats = value;
    }

    /// Borrow latest health report text.
    pub(crate) fn last_health_report(&self) -> Option<&str> {
        self.last_health_report.as_deref()
    }

    /// Replace latest health report text.
    pub(crate) fn set_last_health_report(&mut self, value: Option<String>) {
        self.last_health_report = value;
    }

    /// Clear latest health report text.
    pub(crate) fn clear_last_health_report(&mut self) {
        self.last_health_report = None;
    }
}
