/// Build the REPL prompt with mode indicators based on eval options.
pub fn build_prompt_from_eval_options(options: &crate::EvalOptions) -> String {
    let mut indicators = Vec::new();

    // Show steps mode if not On (default)
    match options.steps_mode {
        crate::StepsMode::Off => indicators.push("[steps:off]"),
        crate::StepsMode::Compact => indicators.push("[steps:compact]"),
        crate::StepsMode::On => {}
    }

    // Show context mode if not Auto (default)
    match options.shared.context_mode {
        crate::ContextMode::IntegratePrep => indicators.push("[ctx:integrate]"),
        crate::ContextMode::Solve => indicators.push("[ctx:solve]"),
        crate::ContextMode::Standard => indicators.push("[ctx:standard]"),
        crate::ContextMode::Auto => {}
    }

    // Show branch mode if not Strict (default)
    match options.branch_mode {
        crate::BranchMode::PrincipalBranch => indicators.push("[branch:principal]"),
        crate::BranchMode::Strict => {}
    }

    // Show complex mode if not Auto (default)
    match options.complex_mode {
        crate::ComplexMode::On => indicators.push("[cx:on]"),
        crate::ComplexMode::Off => indicators.push("[cx:off]"),
        crate::ComplexMode::Auto => {}
    }

    // Show expand_policy if Auto (not default Off)
    if options.shared.expand_policy == crate::ExpandPolicy::Auto {
        indicators.push("[autoexp:on]");
    }

    if indicators.is_empty() {
        "> ".to_string()
    } else {
        format!("{} > ", indicators.join(""))
    }
}

#[cfg(test)]
mod tests {
    use super::build_prompt_from_eval_options;

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
