use crate::{Simplifier, SimplifyPhase};

pub(super) struct HealthRunStats {
    pub(super) total_rewrites: usize,
    pub(super) core_rewrites: usize,
    pub(super) transform_rewrites: usize,
    pub(super) rationalize_rewrites: usize,
    pub(super) post_rewrites: usize,
    pub(super) growth: i64,
    pub(super) shrink: i64,
    pub(super) cycle_detected: Option<(SimplifyPhase, usize)>,
    pub(super) top_rules: Vec<(String, usize)>,
}

pub(super) fn collect_run_stats(
    simplifier: &Simplifier,
    stats: &crate::PipelineStats,
) -> HealthRunStats {
    HealthRunStats {
        total_rewrites: stats.total_rewrites,
        core_rewrites: stats.core.rewrites_used,
        transform_rewrites: stats.transform.rewrites_used,
        rationalize_rewrites: stats.rationalize.rewrites_used,
        post_rewrites: stats.post_cleanup.rewrites_used,
        growth: simplifier.profiler.total_positive_growth(),
        shrink: simplifier.profiler.total_negative_growth().abs(),
        cycle_detected: super::detect_cycle(stats),
        top_rules: simplifier
            .profiler
            .top_applied_for_phase(SimplifyPhase::Transform, 3),
    }
}
