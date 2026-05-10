#[cfg(test)]
mod tests {
    use crate::pipeline_display::{
        compact_subtracted_difference_display, display_expr_or_poly, format_pipeline_stats,
    };

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

    #[test]
    fn compact_subtracted_difference_display_flattens_single_difference() {
        let rendered = compact_subtracted_difference_display(
            "1/3*x^3*arctan(x) - (x^2 / 6 - 1/6*ln(x^2 + 1))".to_string(),
        );
        assert_eq!(rendered, "1/3*x^3*arctan(x) - x^2 / 6 + 1/6*ln(x^2 + 1)");
    }

    #[test]
    fn compact_subtracted_difference_display_keeps_subtracted_sum_grouped() {
        let rendered = compact_subtracted_difference_display("x + 1 - (3 * x + 2)".to_string());
        assert_eq!(rendered, "x + 1 - (3 * x + 2)");
    }

    #[test]
    fn compact_subtracted_difference_display_handles_latex_result() {
        let rendered = compact_subtracted_difference_display(
            "\\frac{{x}^{3}}{3}\\cdot \\arctan(x) - (\\frac{{x}^{2}}{6} - \\frac{1}{6}\\cdot \\ln({x}^{2} + 1))".to_string(),
        );
        assert_eq!(
            rendered,
            "\\frac{{x}^{3}}{3}\\cdot \\arctan(x) - \\frac{{x}^{2}}{6} + \\frac{1}{6}\\cdot \\ln({x}^{2} + 1)"
        );
    }

    #[test]
    fn compact_subtracted_difference_display_compacts_adjacent_signs() {
        let rendered = compact_subtracted_difference_display("a + -1/2*b - -2/3*c".to_string());
        assert_eq!(rendered, "a - 1/2*b + 2/3*c");
    }

    #[test]
    fn compact_subtracted_difference_display_does_not_rewrite_nested_difference() {
        let rendered = compact_subtracted_difference_display("ln(1 - (1 - 2*x)^2)".to_string());
        assert_eq!(rendered, "ln(1 - (1 - 2*x)^2)");
    }

    #[test]
    fn compact_subtracted_difference_display_does_not_rewrite_powered_difference() {
        let rendered = compact_subtracted_difference_display("a - (b - c)^2".to_string());
        assert_eq!(rendered, "a - (b - c)^2");
    }

    #[test]
    fn compact_subtracted_difference_display_does_not_rewrite_multiplied_difference() {
        for input in ["a - (b - c) * d", "a - (b - c)/d", "a - (b - c)\\cdot d"] {
            let rendered = compact_subtracted_difference_display(input.to_string());
            assert_eq!(rendered, input);
        }
    }

    #[test]
    fn compact_subtracted_difference_display_rewrites_mixed_additive_inner() {
        let left_negative = compact_subtracted_difference_display("a - (b - c + d)".to_string());
        assert_eq!(left_negative, "a - b + c - d");

        let right_negative = compact_subtracted_difference_display("a - (b + c - d)".to_string());
        assert_eq!(right_negative, "a - b - c + d");
    }

    #[test]
    fn compact_subtracted_difference_display_rewrites_additive_suffixes() {
        let plus = compact_subtracted_difference_display("a - (b - c) + d".to_string());
        assert_eq!(plus, "a - b + c + d");

        let minus = compact_subtracted_difference_display("a - (b - c) - d".to_string());
        assert_eq!(minus, "a - b + c - d");
    }
}
