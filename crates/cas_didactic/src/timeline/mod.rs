//! Timeline HTML generation module.
//!
//! Provides interactive HTML visualizations for:
//! - Expression simplification steps ([`TimelineHtml`])
//! - Equation solving steps ([`SolveTimelineHtml`])

mod simplify;
mod solve;

// Re-export the public API
use cas_ast::{BoundType, Case, Context, Equation, ExprId, Interval, SolutionSet};
use cas_formatter::{condition_set_to_display, DisplayExpr};
pub use cas_formatter::{html_escape, latex_escape};
use cas_solver::{DisplayEvalSteps, DisplaySolveSteps, SolveStep, Step};
pub use simplify::{TimelineHtml, VerbosityLevel};
pub use solve::SolveTimelineHtml;

/// Canonical output file name used by timeline CLI render helpers.
pub const TIMELINE_HTML_FILE: &str = "timeline.html";
const TIMELINE_NO_STEPS_MESSAGE: &str = "No simplification steps to visualize.";
const TIMELINE_OPEN_HINT_MESSAGE: &str = "Open in browser to view interactive visualization.";

/// Simplify branch payload for CLI `timeline` rendering.
#[derive(Debug, Clone)]
pub struct TimelineSimplifyCommandOutput {
    pub expr_input: String,
    pub use_aggressive: bool,
    pub parsed_expr: ExprId,
    pub simplified_expr: ExprId,
    pub steps: DisplayEvalSteps,
}

/// Solve branch payload for CLI `timeline` rendering.
#[derive(Debug, Clone)]
pub struct TimelineSolveCommandOutput {
    pub equation: Equation,
    pub var: String,
    pub solution_set: SolutionSet,
    pub display_steps: DisplaySolveSteps,
}

/// End-to-end output for CLI `timeline` rendering.
#[derive(Debug)]
pub enum TimelineCommandOutput {
    Solve(TimelineSolveCommandOutput),
    Simplify(TimelineSimplifyCommandOutput),
}

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

fn timeline_simplify_info_lines(use_aggressive: bool) -> Vec<String> {
    let mut lines = Vec::new();
    if use_aggressive {
        lines.push("(Aggressive simplification mode)".to_string());
    }
    lines.push(TIMELINE_OPEN_HINT_MESSAGE.to_string());
    lines
}

fn is_pure_residual_otherwise(case: &Case) -> bool {
    case.when.is_empty() && matches!(&case.then.solutions, SolutionSet::Residual(_))
}

fn display_interval(context: &Context, interval: &Interval) -> String {
    let min_bracket = match interval.min_type {
        BoundType::Open => "(",
        BoundType::Closed => "[",
    };
    let max_bracket = match interval.max_type {
        BoundType::Open => ")",
        BoundType::Closed => "]",
    };
    format!(
        "{}{}, {}{}",
        min_bracket,
        DisplayExpr {
            context,
            id: interval.min
        },
        DisplayExpr {
            context,
            id: interval.max
        },
        max_bracket
    )
}

fn display_solution_set(context: &Context, set: &SolutionSet) -> String {
    match set {
        SolutionSet::Empty => "Empty Set".to_string(),
        SolutionSet::AllReals => "All Real Numbers".to_string(),
        SolutionSet::Discrete(exprs) => {
            let rendered: Vec<String> = exprs
                .iter()
                .map(|e| format!("{}", DisplayExpr { context, id: *e }))
                .collect();
            format!("{{ {} }}", rendered.join(", "))
        }
        SolutionSet::Continuous(interval) => display_interval(context, interval),
        SolutionSet::Union(intervals) => intervals
            .iter()
            .map(|i| display_interval(context, i))
            .collect::<Vec<_>>()
            .join(" U "),
        SolutionSet::Residual(expr) => format!("{}", DisplayExpr { context, id: *expr }),
        SolutionSet::Conditional(cases) => {
            let case_strs: Vec<String> = cases
                .iter()
                .filter_map(|case| {
                    if is_pure_residual_otherwise(case) {
                        return None;
                    }
                    let sol_str = display_solution_set(context, &case.then.solutions);
                    if case.when.is_otherwise() {
                        Some(format!("  otherwise: {}", sol_str))
                    } else {
                        let cond_str = condition_set_to_display(&case.when, context);
                        Some(format!("  if {}: {}", cond_str, sol_str))
                    }
                })
                .collect();
            if case_strs.len() == 1 {
                case_strs[0].trim().to_string()
            } else {
                format!("Conditional:\n{}", case_strs.join("\n"))
            }
        }
    }
}

fn format_timeline_solve_result_line(context: &Context, solution_set: &SolutionSet) -> String {
    format!("Result: {}", display_solution_set(context, solution_set))
}

fn format_timeline_solve_no_steps_message(context: &Context, solution_set: &SolutionSet) -> String {
    format!(
        "No solving steps to visualize.\n{}",
        format_timeline_solve_result_line(context, solution_set)
    )
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
    out: &TimelineSimplifyCommandOutput,
    verbosity: VerbosityLevel,
) -> TimelineCliRender {
    if out.steps.is_empty() {
        return TimelineCliRender::NoSteps {
            lines: vec![TIMELINE_NO_STEPS_MESSAGE.to_string()],
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
    let lines = timeline_simplify_info_lines(out.use_aggressive);

    TimelineCliRender::Html {
        file_name: TIMELINE_HTML_FILE,
        html,
        lines,
    }
}

/// Build CLI render output for solve timeline command.
pub fn render_solve_timeline_cli_output(
    context: &mut Context,
    out: &TimelineSolveCommandOutput,
) -> TimelineCliRender {
    if out.display_steps.0.is_empty() {
        return TimelineCliRender::NoSteps {
            lines: vec![format_timeline_solve_no_steps_message(
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
        format_timeline_solve_result_line(context, &out.solution_set),
        TIMELINE_OPEN_HINT_MESSAGE.to_string(),
    ];

    TimelineCliRender::Html {
        file_name: TIMELINE_HTML_FILE,
        html,
        lines,
    }
}

/// Build CLI render output for a full `timeline` command eval output.
pub fn render_timeline_command_cli_output(
    context: &mut Context,
    out: &TimelineCommandOutput,
    verbosity: VerbosityLevel,
) -> TimelineCliRender {
    match out {
        TimelineCommandOutput::Solve(solve_out) => {
            render_solve_timeline_cli_output(context, solve_out)
        }
        TimelineCommandOutput::Simplify(simplify_out) => {
            render_simplify_timeline_cli_output(context, simplify_out, verbosity)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::{Context, Expr};
    use cas_solver::{to_display_steps, Step};

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

    #[test]
    fn render_timeline_command_cli_output_simplify_no_steps() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let out = TimelineCommandOutput::Simplify(TimelineSimplifyCommandOutput {
            expr_input: "x".to_string(),
            use_aggressive: false,
            parsed_expr: x,
            simplified_expr: x,
            steps: to_display_steps(Vec::new()),
        });

        let render = render_timeline_command_cli_output(&mut ctx, &out, VerbosityLevel::Normal);
        match render {
            TimelineCliRender::NoSteps { lines } => assert!(!lines.is_empty()),
            TimelineCliRender::Html { .. } => panic!("expected no-steps render"),
        }
    }
}
