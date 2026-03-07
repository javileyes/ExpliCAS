mod phase;
mod rationalize;

use self::phase::push_phase_stats;
use self::rationalize::push_rationalize_stats;

/// Format pipeline statistics for diagnostics.
pub fn format_pipeline_stats(
    simplifier: &crate::Simplifier,
    stats: &crate::PipelineStats,
) -> String {
    let mut lines: Vec<String> = Vec::new();

    lines.push(String::new());
    lines.push("──── Pipeline Diagnostics ────".to_string());
    push_phase_stats(
        &mut lines,
        simplifier,
        "Core",
        &stats.core,
        crate::SimplifyPhase::Core,
        false,
    );
    push_phase_stats(
        &mut lines,
        simplifier,
        "Transform",
        &stats.transform,
        crate::SimplifyPhase::Transform,
        true,
    );
    push_rationalize_stats(&mut lines, simplifier, stats);
    push_phase_stats(
        &mut lines,
        simplifier,
        "PostCleanup",
        &stats.post_cleanup,
        crate::SimplifyPhase::PostCleanup,
        false,
    );
    lines.push(format!("  Total rewrites: {}", stats.total_rewrites));
    lines.push("───────────────────────────────".to_string());

    lines.join("\n")
}
