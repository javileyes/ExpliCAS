pub fn health_enable_message() -> &'static str {
    "Health tracking ENABLED (metrics captured after each simplify)"
}

pub fn health_disable_message() -> &'static str {
    "Health tracking DISABLED"
}

pub fn health_clear_message() -> &'static str {
    "Health statistics cleared."
}

/// Clear health profiling counters for a simplifier.
pub fn clear_health_profiler(simplifier: &mut crate::Simplifier) {
    simplifier.profiler.clear_run();
}

/// Capture current health report only when health tracking is enabled.
pub fn capture_health_report_if_enabled(
    simplifier: &crate::Simplifier,
    health_enabled: bool,
) -> Option<String> {
    if health_enabled {
        Some(simplifier.profiler.health_report())
    } else {
        None
    }
}
