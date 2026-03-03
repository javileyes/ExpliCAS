#[cfg(test)]
mod tests {
    use crate::pipeline_display::{display_expr_or_poly, format_pipeline_stats};

    #[test]
    fn format_pipeline_stats_includes_headers() {
        let simplifier = crate::Simplifier::with_default_rules();
        let stats = crate::PipelineStats::default();
        let text = format_pipeline_stats(&simplifier, &stats);
        assert!(text.contains("Pipeline Diagnostics"));
        assert!(text.contains("Total rewrites:"));
    }

    #[test]
    fn display_expr_or_poly_falls_back_to_display_expr() {
        let mut ctx = cas_ast::Context::new();
        let expr = ctx.var("x");
        let rendered = display_expr_or_poly(&ctx, expr);
        assert_eq!(rendered, "x");
    }
}
