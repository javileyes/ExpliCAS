mod cycles;
mod fallback;

pub(super) fn format_health_report_lines(
    last_stats: Option<&crate::PipelineStats>,
    report_text: Option<&str>,
) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();

    if let Some(stats) = last_stats {
        let cycles = cycles::collect_cycle_lines(stats);
        let has_cycles = !cycles.is_empty();
        lines.extend(cycles);
        if has_cycles {
            lines.push(String::new());
        }
    }

    if let Some(report) = report_text {
        lines.push(report.to_string());
    } else {
        lines.extend(fallback::fallback_health_report_lines());
    }

    lines
}
