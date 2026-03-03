use crate::timeline_command_simplify::evaluate_timeline_simplify_with_session;
use crate::timeline_command_solve::evaluate_timeline_solve_with_eval_options;

/// Evaluate REPL `timeline` command (solve/simplify) and return typed output.
pub fn evaluate_timeline_command_with_session<S>(
    engine: &mut cas_solver::Engine,
    session: &mut S,
    input: &str,
    eval_options: &cas_solver::EvalOptions,
) -> Result<crate::TimelineCommandEvalOutput, crate::TimelineCommandEvalError>
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
    match crate::solve_input_parse::parse_timeline_command_input(input) {
        crate::TimelineCommandInput::Solve(solve_rest) => {
            evaluate_timeline_solve_with_eval_options(
                &mut engine.simplifier,
                &solve_rest,
                eval_options,
            )
            .map(crate::TimelineCommandEvalOutput::Solve)
            .map_err(crate::TimelineCommandEvalError::Solve)
        }
        crate::TimelineCommandInput::Simplify { expr, aggressive } => {
            evaluate_timeline_simplify_with_session(engine, session, &expr, aggressive)
                .map(|output| crate::TimelineCommandEvalOutput::Simplify {
                    expr_input: expr,
                    aggressive,
                    output,
                })
                .map_err(crate::TimelineCommandEvalError::Simplify)
        }
    }
}
