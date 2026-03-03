use crate::semantics_preset_types::SemanticsPreset;

const SEMANTICS_PRESETS: [SemanticsPreset; 4] = [
    SemanticsPreset {
        name: "default",
        description: "Reset to engine defaults",
        domain: cas_solver::DomainMode::Generic,
        value: cas_solver::ValueDomain::RealOnly,
        branch: cas_solver::BranchPolicy::Principal,
        inv_trig: cas_solver::InverseTrigPolicy::Strict,
        const_fold: cas_solver::ConstFoldMode::Off,
    },
    SemanticsPreset {
        name: "strict",
        description: "Conservative real + strict domain",
        domain: cas_solver::DomainMode::Strict,
        value: cas_solver::ValueDomain::RealOnly,
        branch: cas_solver::BranchPolicy::Principal,
        inv_trig: cas_solver::InverseTrigPolicy::Strict,
        const_fold: cas_solver::ConstFoldMode::Off,
    },
    SemanticsPreset {
        name: "complex",
        description: "Enable ℂ + safe const_fold (sqrt(-1) → i)",
        domain: cas_solver::DomainMode::Generic,
        value: cas_solver::ValueDomain::ComplexEnabled,
        branch: cas_solver::BranchPolicy::Principal,
        inv_trig: cas_solver::InverseTrigPolicy::Strict,
        const_fold: cas_solver::ConstFoldMode::Safe,
    },
    SemanticsPreset {
        name: "school",
        description: "Real + principal inverse trig (arctan(tan(x)) → x)",
        domain: cas_solver::DomainMode::Generic,
        value: cas_solver::ValueDomain::RealOnly,
        branch: cas_solver::BranchPolicy::Principal,
        inv_trig: cas_solver::InverseTrigPolicy::PrincipalValue,
        const_fold: cas_solver::ConstFoldMode::Off,
    },
];

pub fn semantics_presets() -> &'static [SemanticsPreset] {
    &SEMANTICS_PRESETS
}

pub fn find_semantics_preset(name: &str) -> Option<SemanticsPreset> {
    SEMANTICS_PRESETS
        .iter()
        .copied()
        .find(|preset| preset.name == name)
}
