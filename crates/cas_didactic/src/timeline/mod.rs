//! Timeline HTML generation module.
//!
//! Provides interactive HTML visualizations for:
//! - Expression simplification steps ([`TimelineHtml`])
//! - Equation solving steps ([`SolveTimelineHtml`])

mod simplify;
mod solve;

// Re-export the public API
use cas_ast::{Context, Equation, ExprId, SolutionSet};
pub use cas_formatter::{html_escape, latex_escape};
use cas_solver::{SolveStep, Step};
pub use simplify::{TimelineHtml, VerbosityLevel};
pub use solve::SolveTimelineHtml;

/// Canonical output file name used by timeline CLI render helpers.
pub const TIMELINE_HTML_FILE: &str = "timeline.html";

/// CLI-facing timeline render artifact.
#[derive(Debug, Clone)]
pub enum TimelineCliRender {
    /// No timeline file should be emitted; return textual lines only.
    NoSteps { lines: Vec<String> },
    /// Timeline file + informational lines.
    Html {
        file_name: &'static str,
        html: String,
        lines: Vec<String>,
    },
}

/// Render full HTML for simplification timeline.
pub fn render_simplify_timeline_html(
    context: &mut Context,
    steps: &[Step],
    original_expr: ExprId,
    simplified_result: Option<ExprId>,
    verbosity: VerbosityLevel,
    input_string: Option<&str>,
) -> String {
    let mut timeline = TimelineHtml::new_with_result_and_style(
        context,
        steps,
        original_expr,
        simplified_result,
        verbosity,
        input_string,
    );
    timeline.to_html()
}

/// Render full HTML for solve timeline.
pub fn render_solve_timeline_html(
    context: &mut Context,
    steps: &[SolveStep],
    original_eq: &Equation,
    solution_set: &SolutionSet,
    var: &str,
) -> String {
    let mut timeline = SolveTimelineHtml::new(context, steps, original_eq, solution_set, var);
    timeline.to_html()
}

/// Build CLI render output for simplify timeline command.
pub fn render_simplify_timeline_cli_output(
    context: &mut Context,
    out: &cas_solver::TimelineSimplifyCommandEvalOutput,
    verbosity: VerbosityLevel,
) -> TimelineCliRender {
    if out.steps.is_empty() {
        return TimelineCliRender::NoSteps {
            lines: vec![cas_solver::timeline_no_steps_message().to_string()],
        };
    }

    let html = render_simplify_timeline_html(
        context,
        &out.steps,
        out.parsed_expr,
        Some(out.simplified_expr),
        verbosity,
        Some(out.expr_input.as_str()),
    );
    let lines = cas_solver::format_timeline_simplify_info_lines(out.use_aggressive);

    TimelineCliRender::Html {
        file_name: TIMELINE_HTML_FILE,
        html,
        lines,
    }
}

/// Build CLI render output for solve timeline command.
pub fn render_solve_timeline_cli_output(
    context: &mut Context,
    out: &cas_solver::TimelineSolveEvalOutput,
) -> TimelineCliRender {
    if out.display_steps.0.is_empty() {
        return TimelineCliRender::NoSteps {
            lines: vec![cas_solver::format_timeline_solve_no_steps_message(
                context,
                &out.solution_set,
            )],
        };
    }

    let html = render_solve_timeline_html(
        context,
        &out.display_steps.0,
        &out.equation,
        &out.solution_set,
        &out.var,
    );
    let lines = vec![
        cas_solver::format_timeline_solve_result_line(context, &out.solution_set),
        cas_solver::timeline_open_hint_message().to_string(),
    ];

    TimelineCliRender::Html {
        file_name: TIMELINE_HTML_FILE,
        html,
        lines,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{Context, Expr};
    use cas_solver::Step;

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

    #[test]
    fn render_simplify_timeline_helper_produces_document() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let y = ctx.var("y");
        let add = ctx.add(cas_ast::Expr::Add(x, y));
        let steps = vec![];
        let html = render_simplify_timeline_html(
            &mut ctx,
            &steps,
            add,
            Some(add),
            VerbosityLevel::Normal,
            Some("x+y"),
        );
        assert!(html.contains("<!DOCTYPE html"));
    }
}
