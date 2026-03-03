use cas_ast::{Context, ExprId};

/// Display an expression, preferring formatted poly output when available.
pub fn display_expr_or_poly(ctx: &Context, id: ExprId) -> String {
    if let Some(poly_str) = crate::try_render_poly_result(ctx, id) {
        return poly_str;
    }
    cas_formatter::clean_display_string(&format!(
        "{}",
        cas_formatter::DisplayExpr { context: ctx, id }
    ))
}

/// Format pipeline statistics for diagnostics.
pub fn format_pipeline_stats(
    simplifier: &crate::Simplifier,
    stats: &crate::PipelineStats,
) -> String {
    let mut lines: Vec<String> = Vec::new();

    lines.push(String::new());
    lines.push("──── Pipeline Diagnostics ────".to_string());
    lines.push(format!(
        "  Core:       {} iters, {} rewrites",
        stats.core.iters_used, stats.core.rewrites_used
    ));
    if let Some(ref cycle) = stats.core.cycle {
        lines.push(format!(
            "              ⚠ Cycle detected: period={} at rewrite={} (stopped early)",
            cycle.period, cycle.at_step
        ));
        let top = simplifier
            .profiler
            .top_applied_for_phase(crate::SimplifyPhase::Core, 2);
        if !top.is_empty() {
            let hints: Vec<_> = top.iter().map(|(r, c)| format!("{}={}", r, c)).collect();
            lines.push(format!(
                "              Likely contributors: {}",
                hints.join(", ")
            ));
        }
    }
    lines.push(format!(
        "  Transform:  {} iters, {} rewrites, changed={}",
        stats.transform.iters_used, stats.transform.rewrites_used, stats.transform.changed
    ));
    if let Some(ref cycle) = stats.transform.cycle {
        lines.push(format!(
            "              ⚠ Cycle detected: period={} at rewrite={} (stopped early)",
            cycle.period, cycle.at_step
        ));
        let top = simplifier
            .profiler
            .top_applied_for_phase(crate::SimplifyPhase::Transform, 2);
        if !top.is_empty() {
            let hints: Vec<_> = top.iter().map(|(r, c)| format!("{}={}", r, c)).collect();
            lines.push(format!(
                "              Likely contributors: {}",
                hints.join(", ")
            ));
        }
    }
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

    lines.push(format!(
        "  PostCleanup: {} iters, {} rewrites",
        stats.post_cleanup.iters_used, stats.post_cleanup.rewrites_used
    ));
    if let Some(ref cycle) = stats.post_cleanup.cycle {
        lines.push(format!(
            "              ⚠ Cycle detected: period={} at rewrite={} (stopped early)",
            cycle.period, cycle.at_step
        ));
        let top = simplifier
            .profiler
            .top_applied_for_phase(crate::SimplifyPhase::PostCleanup, 2);
        if !top.is_empty() {
            let hints: Vec<_> = top.iter().map(|(r, c)| format!("{}={}", r, c)).collect();
            lines.push(format!(
                "              Likely contributors: {}",
                hints.join(", ")
            ));
        }
    }
    lines.push(format!("  Total rewrites: {}", stats.total_rewrites));
    lines.push("───────────────────────────────".to_string());

    lines.join("\n")
}
