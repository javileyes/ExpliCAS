mod aggressive;
mod standard;

pub(crate) fn evaluate_timeline_simplify_with_session<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    input: &str,
    aggressive: bool,
) -> Result<crate::TimelineSimplifyEvalOutput, crate::TimelineSimplifyEvalError>
where
    S: crate::SolverEvalSession,
{
    if aggressive {
        aggressive::evaluate_timeline_simplify_aggressive(&mut engine.simplifier, input)
    } else {
        standard::evaluate_timeline_simplify_standard(engine, session, input)
    }
}
