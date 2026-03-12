#[cfg(test)]
mod tests {
    use crate::prompt_display::build_prompt_from_eval_options;

    #[test]
    fn build_prompt_from_eval_options_default_is_plain_prompt() {
        let options = crate::EvalOptions::default();
        assert_eq!(build_prompt_from_eval_options(&options), "> ");
    }

    #[test]
    fn build_prompt_from_eval_options_includes_non_default_indicators() {
        let options = crate::EvalOptions {
            steps_mode: crate::StepsMode::Compact,
            complex_mode: crate::ComplexMode::On,
            shared: crate::SharedSemanticConfig {
                context_mode: crate::ContextMode::Solve,
                ..crate::SharedSemanticConfig::default()
            },
            ..crate::EvalOptions::default()
        };
        let prompt = build_prompt_from_eval_options(&options);
        assert!(prompt.contains("[steps:compact]"));
        assert!(prompt.contains("[ctx:solve]"));
        assert!(prompt.contains("[cx:on]"));
    }
}
