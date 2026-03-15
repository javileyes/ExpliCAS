//! Timeline command APIs re-exported for session clients.

pub use crate::analysis_command_format_errors::format_timeline_command_error_message;
pub use cas_api_models::{
    TimelineCommandEvalError, TimelineCommandInput, TimelineSimplifyEvalError,
    TimelineSolveEvalError,
};

/// Evaluate REPL `timeline` command (solve/simplify) and return typed output.
pub fn evaluate_timeline_command_with_session<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    input: &str,
    eval_options: &crate::EvalOptions,
) -> Result<TimelineCommandEvalOutput, TimelineCommandEvalError>
where
    S: crate::SolverEvalSession,
{
    match crate::parse_timeline_command_input(input) {
        crate::TimelineCommandInput::Solve(solve_rest) => {
            crate::timeline_solve_eval::evaluate_timeline_solve_with_eval_options(
                &mut engine.simplifier,
                &solve_rest,
                eval_options,
            )
            .map(TimelineCommandEvalOutput::Solve)
            .map_err(TimelineCommandEvalError::Solve)
        }
        crate::TimelineCommandInput::Simplify { expr, aggressive } => {
            crate::timeline_simplify_eval::evaluate_timeline_simplify_with_session(
                engine, session, &expr, aggressive,
            )
            .map(|output| TimelineCommandEvalOutput::Simplify {
                expr_input: expr,
                aggressive,
                output,
            })
            .map_err(TimelineCommandEvalError::Simplify)
        }
    }
}

#[derive(Debug, Clone)]
pub struct TimelineSolveEvalOutput {
    pub equation: cas_ast::Equation,
    pub var: String,
    pub solution_set: cas_ast::SolutionSet,
    pub display_steps: crate::DisplaySolveSteps,
    pub diagnostics: crate::SolveDiagnostics,
}

#[derive(Debug, Clone)]
pub struct TimelineSimplifyEvalOutput {
    pub parsed_expr: cas_ast::ExprId,
    pub simplified_expr: cas_ast::ExprId,
    pub steps: crate::DisplayEvalSteps,
}

#[derive(Debug, Clone)]
pub enum TimelineCommandEvalOutput {
    Solve(TimelineSolveEvalOutput),
    Simplify {
        expr_input: String,
        aggressive: bool,
        output: TimelineSimplifyEvalOutput,
    },
}
