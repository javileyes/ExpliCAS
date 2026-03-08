pub(super) fn collect_cycle_lines(stats: &crate::PipelineStats) -> Vec<String> {
    [
        (&stats.core.cycle, "Core"),
        (&stats.transform.cycle, "Transform"),
        (&stats.rationalize.cycle, "Rationalize"),
        (&stats.post_cleanup.cycle, "PostCleanup"),
    ]
    .iter()
    .filter_map(|(cycle, phase_name)| {
        cycle.as_ref().map(|info| {
            format!(
                "⚠ Cycle detected in {}: period={} at rewrite={} (stopped early)",
                phase_name, info.period, info.at_step
            )
        })
    })
    .collect()
}
