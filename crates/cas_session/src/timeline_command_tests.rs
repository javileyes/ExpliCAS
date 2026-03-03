#[cfg(test)]
mod tests {
    use crate::{evaluate_timeline_command_with_session, SessionState, TimelineCommandEvalOutput};

    #[test]
    fn evaluate_timeline_command_with_session_simplify_runs() {
        let mut engine = cas_solver::Engine::new();
        let mut session = SessionState::new();
        let opts = cas_solver::EvalOptions::default();
        let out = evaluate_timeline_command_with_session(&mut engine, &mut session, "x + x", &opts)
            .expect("timeline command simplify");
        match out {
            TimelineCommandEvalOutput::Simplify { .. } => {}
            _ => panic!("expected simplify"),
        }
    }
}
