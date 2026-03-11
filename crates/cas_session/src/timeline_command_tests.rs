#[cfg(test)]
mod tests {
    use crate::SessionState;
    #[allow(unused_imports)]
    use cas_solver::session_api::{
        formatting::*, options::*, runtime::*, session_support::*, symbolic_commands::*, types::*,
    };
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
