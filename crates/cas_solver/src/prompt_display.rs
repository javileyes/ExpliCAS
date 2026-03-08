/// Build REPL prompt indicators from eval options.
///
/// This keeps prompt derivation frontend-agnostic while preserving
/// the existing CLI-visible prompt contract.
pub fn build_prompt_from_eval_options(options: &crate::EvalOptions) -> String {
    let mut indicators = Vec::new();

    match options.steps_mode {
        crate::StepsMode::Off => indicators.push("[steps:off]"),
        crate::StepsMode::Compact => indicators.push("[steps:compact]"),
        crate::StepsMode::On => {}
    }

    match options.shared.context_mode {
        crate::ContextMode::IntegratePrep => indicators.push("[ctx:integrate]"),
        crate::ContextMode::Solve => indicators.push("[ctx:solve]"),
        crate::ContextMode::Standard => indicators.push("[ctx:standard]"),
        crate::ContextMode::Auto => {}
    }

    match options.branch_mode {
        crate::BranchMode::PrincipalBranch => indicators.push("[branch:principal]"),
        crate::BranchMode::Strict => {}
    }

    match options.complex_mode {
        crate::ComplexMode::On => indicators.push("[cx:on]"),
        crate::ComplexMode::Off => indicators.push("[cx:off]"),
        crate::ComplexMode::Auto => {}
    }

    if options.shared.expand_policy == crate::ExpandPolicy::Auto {
        indicators.push("[autoexp:on]");
    }

    if indicators.is_empty() {
        "> ".to_string()
    } else {
        format!("{} > ", indicators.join(""))
    }
}
