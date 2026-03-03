/// Build REPL prompt indicators from eval options.
///
/// This keeps prompt derivation frontend-agnostic while preserving
/// the existing CLI-visible prompt contract.
pub fn build_prompt_from_eval_options(options: &cas_solver::EvalOptions) -> String {
    let mut indicators = Vec::new();

    match options.steps_mode {
        cas_solver::StepsMode::Off => indicators.push("[steps:off]"),
        cas_solver::StepsMode::Compact => indicators.push("[steps:compact]"),
        cas_solver::StepsMode::On => {}
    }

    match options.shared.context_mode {
        cas_solver::ContextMode::IntegratePrep => indicators.push("[ctx:integrate]"),
        cas_solver::ContextMode::Solve => indicators.push("[ctx:solve]"),
        cas_solver::ContextMode::Standard => indicators.push("[ctx:standard]"),
        cas_solver::ContextMode::Auto => {}
    }

    match options.branch_mode {
        cas_solver::BranchMode::PrincipalBranch => indicators.push("[branch:principal]"),
        cas_solver::BranchMode::Strict => {}
    }

    match options.complex_mode {
        cas_solver::ComplexMode::On => indicators.push("[cx:on]"),
        cas_solver::ComplexMode::Off => indicators.push("[cx:off]"),
        cas_solver::ComplexMode::Auto => {}
    }

    if options.shared.expand_policy == cas_solver::ExpandPolicy::Auto {
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
