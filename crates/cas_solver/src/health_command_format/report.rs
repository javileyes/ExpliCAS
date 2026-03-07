pub(super) fn format_health_report_lines(
    last_stats: Option<&crate::PipelineStats>,
    report_text: Option<&str>,
) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();

    if let Some(stats) = last_stats {
        let cycles = collect_cycle_lines(stats);
        let has_cycles = !cycles.is_empty();
        lines.extend(cycles);
        if has_cycles {
            lines.push(String::new());
        }
    }

    if let Some(report) = report_text {
        lines.push(report.to_string());
    } else {
        lines.push("No health report available.".to_string());
        lines.push(
            "Run a simplification first (health is captured when debug mode or health mode is on)."
                .to_string(),
        );
        lines.push("Enable with: health on".to_string());
    }

    lines
}

fn collect_cycle_lines(stats: &crate::PipelineStats) -> Vec<String> {
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
