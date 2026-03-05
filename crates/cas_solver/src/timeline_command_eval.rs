use crate::timeline_simplify_eval::evaluate_timeline_simplify_with_session;
use crate::timeline_solve_eval::evaluate_timeline_solve_with_eval_options;

/// Evaluate REPL `timeline` command (solve/simplify) and return typed output.
pub fn evaluate_timeline_command_with_session<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    input: &str,
    eval_options: &crate::EvalOptions,
) -> Result<crate::TimelineCommandEvalOutput, crate::TimelineCommandEvalError>
where
    S: crate::EvalSession<Options = crate::EvalOptions, Diagnostics = crate::Diagnostics>,
    S::Store: crate::EvalStore<
        DomainMode = crate::DomainMode,
        RequiredItem = crate::RequiredItem,
        Step = crate::Step,
        Diagnostics = crate::Diagnostics,
    >,
{
    match crate::parse_timeline_command_input(input) {
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
