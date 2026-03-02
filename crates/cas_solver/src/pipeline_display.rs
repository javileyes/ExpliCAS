use cas_ast::{Context, ExprId};

/// Normalize the last `Result: ...` line by cleaning display artifacts.
pub fn clean_result_output_line(lines: &mut [String]) {
    let Some(last) = lines.last_mut() else {
        return;
    };
    let Some(raw_value) = last.strip_prefix("Result: ") else {
        return;
    };
    *last = format!("Result: {}", cas_formatter::clean_display_string(raw_value));
}

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

#[cfg(test)]
mod tests {
    use super::{clean_result_output_line, display_expr_or_poly, format_pipeline_stats};

    #[test]
    fn format_pipeline_stats_includes_headers() {
        let simplifier = crate::Simplifier::with_default_rules();
        let stats = crate::PipelineStats::default();
        let text = format_pipeline_stats(&simplifier, &stats);
        assert!(text.contains("Pipeline Diagnostics"));
        assert!(text.contains("Total rewrites:"));
    }

    #[test]
    fn clean_result_output_line_cleans_last_result_line() {
        let mut lines = vec!["info".to_string(), "Result: __hold(x+1)".to_string()];
        clean_result_output_line(&mut lines);
        assert_eq!(lines.last().unwrap(), "Result: x+1");
    }

    #[test]
    fn display_expr_or_poly_falls_back_to_display_expr() {
        let mut ctx = cas_ast::Context::new();
        let expr = ctx.var("x");
        let rendered = display_expr_or_poly(&ctx, expr);
        assert_eq!(rendered, "x");
    }
}
