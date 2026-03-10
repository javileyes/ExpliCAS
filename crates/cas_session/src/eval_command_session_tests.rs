#[cfg(test)]
mod tests {
    use crate::eval_command::EvalCommandConfig;
    use crate::SessionState;

    #[test]
    fn evaluate_eval_with_session_runs() {
        let mut engine = cas_solver::Engine::new();
        let mut session = SessionState::new();
        let out = cas_solver::evaluate_eval_with_session(
            &mut engine,
            &mut session,
            EvalCommandConfig {
                expr: "x + x",
                auto_store: false,
                max_chars: 2000,
                steps_mode: "off",
                budget_preset: "standard",
                strict: false,
                domain: "generic",
                context_mode: "auto",
                branch_mode: "strict",
                expand_policy: "off",
                complex_mode: "auto",
                const_fold: "off",
                value_domain: "real",
                complex_branch: "principal",
                inv_trig: "strict",
                assume_scope: "real",
            },
            |_steps, _events, _context, _steps_mode| Vec::new(),
        )
        .expect("eval");

        assert!(out.ok);
        assert!(out.result.contains("2 * x"));
    }
}
