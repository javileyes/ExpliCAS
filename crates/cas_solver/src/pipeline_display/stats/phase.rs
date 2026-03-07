use cas_solver_core::cycle_models::CycleInfo;

pub(super) fn push_phase_stats(
    lines: &mut Vec<String>,
    simplifier: &crate::Simplifier,
    label: &str,
    stats: &crate::PhaseStats,
    phase: crate::SimplifyPhase,
    include_changed: bool,
) {
    let mut line = format!(
        "  {}: {} iters, {} rewrites",
        pad_label(label),
        stats.iters_used,
        stats.rewrites_used
    );
    if include_changed {
        line.push_str(&format!(", changed={}", stats.changed));
    }
    lines.push(line);

    push_cycle_details(lines, simplifier, &stats.cycle, phase);
}

fn push_cycle_details(
    lines: &mut Vec<String>,
    simplifier: &crate::Simplifier,
    cycle: &Option<CycleInfo>,
    phase: crate::SimplifyPhase,
) {
    if let Some(cycle) = cycle {
        lines.push(format!(
            "              ⚠ Cycle detected: period={} at rewrite={} (stopped early)",
            cycle.period, cycle.at_step
        ));

        let top = simplifier.profiler.top_applied_for_phase(phase, 2);
        if !top.is_empty() {
            let hints: Vec<_> = top.iter().map(|(r, c)| format!("{}={}", r, c)).collect();
            lines.push(format!(
                "              Likely contributors: {}",
                hints.join(", ")
            ));
        }
    }
}

fn pad_label(label: &str) -> String {
    format!("{label:<11}")
}
