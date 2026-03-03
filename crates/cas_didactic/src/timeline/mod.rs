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

/// Evaluate a `timeline` command and project solver output to didactic payload.
pub fn evaluate_timeline_command_output_with_session<S>(
    engine: &mut cas_solver::Engine,
    session: &mut S,
    input: &str,
    eval_options: &cas_solver::EvalOptions,
) -> Result<TimelineCommandOutput, cas_solver::TimelineCommandEvalError>
where
    S: cas_solver::EvalSession<
        Options = cas_solver::EvalOptions,
        Diagnostics = cas_solver::Diagnostics,
    >,
    S::Store: cas_solver::EvalStore<
        DomainMode = cas_solver::DomainMode,
        RequiredItem = cas_solver::RequiredItem,
        Step = cas_solver::Step,
        Diagnostics = cas_solver::Diagnostics,
    >,
{
    let output =
        cas_solver::evaluate_timeline_command_with_session(engine, session, input, eval_options)?;
    Ok(timeline_command_output_from_solver(output))
}

/// Evaluate and render a `timeline` command to CLI render output in one call.
pub fn evaluate_timeline_command_cli_render_with_session<S>(
    engine: &mut cas_solver::Engine,
    session: &mut S,
    input: &str,
    eval_options: &cas_solver::EvalOptions,
    verbosity: VerbosityLevel,
) -> Result<TimelineCliRender, cas_solver::TimelineCommandEvalError>
where
    S: cas_solver::EvalSession<
        Options = cas_solver::EvalOptions,
        Diagnostics = cas_solver::Diagnostics,
    >,
    S::Store: cas_solver::EvalStore<
        DomainMode = cas_solver::DomainMode,
        RequiredItem = cas_solver::RequiredItem,
        Step = cas_solver::Step,
        Diagnostics = cas_solver::Diagnostics,
    >,
{
    let out = evaluate_timeline_command_output_with_session(engine, session, input, eval_options)?;
    Ok(render_timeline_command_cli_output(
        &mut engine.simplifier.context,
        &out,
        verbosity,
    ))
}

/// Extract timeline input from a full invocation or return trimmed input as-is.
pub fn extract_timeline_invocation_input(line: &str) -> &str {
    line.strip_prefix("timeline")
        .map(str::trim)
        .unwrap_or_else(|| line.trim())
}

/// Evaluate a full `timeline ...` invocation and return normalized CLI actions.
pub fn evaluate_timeline_invocation_cli_actions_with_session<S>(
    engine: &mut cas_solver::Engine,
    session: &mut S,
    line: &str,
    eval_options: &cas_solver::EvalOptions,
    verbosity: VerbosityLevel,
) -> Result<Vec<TimelineCliAction>, cas_solver::TimelineCommandEvalError>
where
    S: cas_solver::EvalSession<
        Options = cas_solver::EvalOptions,
        Diagnostics = cas_solver::Diagnostics,
    >,
    S::Store: cas_solver::EvalStore<
        DomainMode = cas_solver::DomainMode,
        RequiredItem = cas_solver::RequiredItem,
        Step = cas_solver::Step,
        Diagnostics = cas_solver::Diagnostics,
    >,
{
    let input = extract_timeline_invocation_input(line);
    let render = evaluate_timeline_command_cli_render_with_session(
        engine,
        session,
        input,
        eval_options,
        verbosity,
    )?;
    Ok(timeline_cli_actions_from_render(render))
}

/// Convert solver timeline evaluation output into didactic render payload.
pub fn timeline_command_output_from_solver(
    output: cas_solver::TimelineCommandEvalOutput,
) -> TimelineCommandOutput {
    match output {
        cas_solver::TimelineCommandEvalOutput::Solve(out) => {
            TimelineCommandOutput::Solve(TimelineSolveCommandOutput {
                equation: out.equation,
                var: out.var,
                solution_set: out.solution_set,
                display_steps: out.display_steps,
            })
        }
        cas_solver::TimelineCommandEvalOutput::Simplify {
            expr_input,
            aggressive,
            output,
        } => TimelineCommandOutput::Simplify(TimelineSimplifyCommandOutput {
            expr_input,
            use_aggressive: aggressive,
            parsed_expr: output.parsed_expr,
            simplified_expr: output.simplified_expr,
            steps: output.steps,
        }),
    }
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

/// Normalized CLI actions derived from [`TimelineCliRender`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimelineCliAction {
    Output(String),
    WriteFile { path: String, contents: String },
    OpenFile { path: String },
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

/// Convert timeline render output into primitive CLI actions.
pub fn timeline_cli_actions_from_render(render: TimelineCliRender) -> Vec<TimelineCliAction> {
    match render {
        TimelineCliRender::NoSteps { lines } => lines
            .into_iter()
            .map(TimelineCliAction::Output)
            .collect::<Vec<_>>(),
        TimelineCliRender::Html {
            file_name,
            html,
            lines,
        } => {
            let mut actions = vec![
                TimelineCliAction::WriteFile {
                    path: file_name.to_string(),
                    contents: html,
                },
                TimelineCliAction::OpenFile {
                    path: file_name.to_string(),
                },
            ];
            for line in lines {
                actions.push(TimelineCliAction::Output(line));
            }
            actions
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

    #[test]
    fn timeline_cli_actions_from_render_html_emits_io_and_lines() {
        let actions = timeline_cli_actions_from_render(TimelineCliRender::Html {
            file_name: TIMELINE_HTML_FILE,
            html: "<html/>".to_string(),
            lines: vec!["line1".to_string(), "line2".to_string()],
        });
        assert!(matches!(
            actions.first(),
            Some(TimelineCliAction::WriteFile { .. })
        ));
        assert!(matches!(
            actions.get(1),
            Some(TimelineCliAction::OpenFile { .. })
        ));
        assert!(actions
            .iter()
            .any(|action| matches!(action, TimelineCliAction::Output(line) if line == "line1")));
    }

    #[test]
    fn extract_timeline_invocation_input_strips_prefix() {
        assert_eq!(extract_timeline_invocation_input("timeline x+1"), "x+1");
        assert_eq!(
            extract_timeline_invocation_input("timeline solve x+1=2,x"),
            "solve x+1=2,x"
        );
        assert_eq!(extract_timeline_invocation_input("x+1"), "x+1");
    }

    #[test]
    fn evaluate_timeline_invocation_cli_actions_with_session_returns_actions() {
        let mut engine = cas_solver::Engine::new();
        let mut session = cas_session::SessionState::new();
        let options = cas_solver::EvalOptions::default();
        let actions = evaluate_timeline_invocation_cli_actions_with_session(
            &mut engine,
            &mut session,
            "timeline x+1",
            &options,
            VerbosityLevel::Normal,
        )
        .expect("timeline eval");
        assert!(!actions.is_empty());
    }
}
