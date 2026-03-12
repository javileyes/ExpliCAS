#[cfg(test)]
mod tests {
    use crate::state_core::SessionState;
    #[allow(unused_imports)]
    use cas_solver::session_api::{assumptions::*, runtime::*, simplifier::*, timeline::*};
    use cas_solver_core::eval_options::EvalOptions;

    #[test]
    fn evaluate_timeline_command_with_session_simplify_runs() {
        let mut engine = cas_solver::runtime::Engine::new();
        let mut session = SessionState::new();
        let opts = EvalOptions::default();
        let out = evaluate_timeline_command_with_session(&mut engine, &mut session, "x + x", &opts)
            .expect("timeline command simplify");
        match out {
            TimelineCommandEvalOutput::Simplify { .. } => {}
            _ => panic!("expected simplify"),
        }
    }
}
