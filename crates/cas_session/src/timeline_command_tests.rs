#[cfg(test)]
mod tests {
    use crate::solver_exports::{
        evaluate_timeline_command_with_session, TimelineCommandEvalOutput,
    };
    use crate::SessionState;
    use cas_solver_core::eval_options::EvalOptions;

    #[test]
    fn evaluate_timeline_command_with_session_simplify_runs() {
        let mut engine = cas_solver::Engine::new();
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
