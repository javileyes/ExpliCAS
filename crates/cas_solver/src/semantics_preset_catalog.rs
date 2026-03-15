use crate::SemanticsPreset;

const SEMANTICS_PRESETS: [SemanticsPreset; 4] = [
    SemanticsPreset {
        name: "default",
        description: "Reset to engine defaults",
        domain: crate::DomainMode::Generic,
        value: crate::ValueDomain::RealOnly,
        branch: crate::BranchPolicy::Principal,
        inv_trig: crate::InverseTrigPolicy::Strict,
        const_fold: crate::ConstFoldMode::Off,
    },
    SemanticsPreset {
        name: "strict",
        description: "Conservative real + strict domain",
        domain: crate::DomainMode::Strict,
        value: crate::ValueDomain::RealOnly,
        branch: crate::BranchPolicy::Principal,
        inv_trig: crate::InverseTrigPolicy::Strict,
        const_fold: crate::ConstFoldMode::Off,
    },
    SemanticsPreset {
        name: "complex",
        description: "Enable ℂ + safe const_fold (sqrt(-1) → i)",
        domain: crate::DomainMode::Generic,
        value: crate::ValueDomain::ComplexEnabled,
        branch: crate::BranchPolicy::Principal,
        inv_trig: crate::InverseTrigPolicy::Strict,
        const_fold: crate::ConstFoldMode::Safe,
    },
    SemanticsPreset {
        name: "school",
        description: "Real + principal inverse trig (arctan(tan(x)) → x)",
        domain: crate::DomainMode::Generic,
        value: crate::ValueDomain::RealOnly,
        branch: crate::BranchPolicy::Principal,
        inv_trig: crate::InverseTrigPolicy::PrincipalValue,
        const_fold: crate::ConstFoldMode::Off,
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
