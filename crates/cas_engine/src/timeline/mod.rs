//! Timeline HTML generation module.
//!
//! Provides interactive HTML visualizations for:
//! - Expression simplification steps ([`TimelineHtml`])
//! - Equation solving steps ([`SolveTimelineHtml`])

mod simplify;
mod solve;

// Re-export the public API
pub use cas_formatter::{html_escape, latex_escape};
pub use simplify::{TimelineHtml, VerbosityLevel};
pub use solve::SolveTimelineHtml;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::Step;
    use cas_ast::{Context, Expr};

    #[test]
    fn test_html_generation() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let three = ctx.num(3);
        let add_expr = ctx.add(Expr::Add(two, three));
        let five = ctx.num(5);

        // Create a step for the simplification
        let steps = vec![Step::new(
            "2 + 3 = 5",
            "Combine Constants",
            add_expr,
            five,
            vec![],
            Some(&ctx),
        )];

        let mut timeline = TimelineHtml::new(&mut ctx, &steps, add_expr, VerbosityLevel::Verbose);
        let html = timeline.to_html();

        assert!(html.contains("<!DOCTYPE html"));
        assert!(html.contains("timeline"));
        assert!(html.contains("CAS Simplification"));
        // The HTML should contain our step (Combine Constants has ImportanceLevel::Low, so needs Verbose)
        assert!(html.contains("Combine Constants"));
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("<script>"), "&lt;script&gt;");
        assert_eq!(html_escape("x & y"), "x &amp; y");
    }
}
