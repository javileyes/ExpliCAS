#[cfg(test)]
mod tests {
    use crate::prompt_display::build_prompt_from_eval_options;

    #[test]
    fn build_prompt_from_eval_options_default_is_plain_prompt() {
        let options = cas_solver::EvalOptions::default();
        assert_eq!(build_prompt_from_eval_options(&options), "> ");
    }

    #[test]
    fn build_prompt_from_eval_options_includes_non_default_indicators() {
        let options = cas_solver::EvalOptions {
            steps_mode: cas_solver::StepsMode::Compact,
            complex_mode: cas_solver::ComplexMode::On,
            shared: cas_solver::SharedSemanticConfig {
                context_mode: cas_solver::ContextMode::Solve,
                ..cas_solver::SharedSemanticConfig::default()
            },
            ..cas_solver::EvalOptions::default()
        };
        let prompt = build_prompt_from_eval_options(&options);
        assert!(prompt.contains("[steps:compact]"));
        assert!(prompt.contains("[ctx:solve]"));
        assert!(prompt.contains("[cx:on]"));
    }
}
