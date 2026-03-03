pub use crate::timeline_command_eval::evaluate_timeline_command_with_session;

#[cfg(test)]
mod tests {
    #[test]
    fn evaluate_timeline_command_with_session_simplify_runs() {
        let mut engine = cas_solver::Engine::new();
        let mut session = crate::SessionState::new();
        let opts = cas_solver::EvalOptions::default();
        let out = super::evaluate_timeline_command_with_session(
            &mut engine,
            &mut session,
            "x + x",
            &opts,
        )
        .expect("timeline command simplify");
        match out {
            crate::TimelineCommandEvalOutput::Simplify { .. } => {}
            _ => panic!("expected simplify"),
        }
    }
}
