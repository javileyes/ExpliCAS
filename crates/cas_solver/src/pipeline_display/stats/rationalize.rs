pub(super) fn push_rationalize_stats(
    lines: &mut Vec<String>,
    simplifier: &crate::Simplifier,
    stats: &crate::PipelineStats,
) {
    lines.push(format!(
        "  Rationalize: {:?}",
        stats
            .rationalize_level
            .unwrap_or(crate::AutoRationalizeLevel::Off)
    ));

    if let Some(ref outcome) = stats.rationalize_outcome {
        match outcome {
            crate::RationalizeOutcome::Applied => {
                lines.push("              → Applied ✓".to_string());
            }
            crate::RationalizeOutcome::NotApplied(reason) => {
                lines.push(format!("              → NotApplied: {:?}", reason));
            }
        }
    }

    if let Some(ref cycle) = stats.rationalize.cycle {
        lines.push(format!(
            "              ⚠ Cycle detected: period={} at rewrite={} (stopped early)",
            cycle.period, cycle.at_step
        ));
        let top = simplifier
            .profiler
            .top_applied_for_phase(crate::SimplifyPhase::Rationalize, 2);
        if !top.is_empty() {
            let hints: Vec<_> = top.iter().map(|(r, c)| format!("{}={}", r, c)).collect();
            lines.push(format!(
                "              Likely contributors: {}",
                hints.join(", ")
            ));
        }
    }
}
