use crate::log_domain::DomainModeKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionClassKind {
    Definability,
    Analytic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvenanceKind {
    Intrinsic,
    Introduced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveSafetyKind {
    Always,
    IntrinsicCondition(ConditionClassKind),
    NeedsCondition(ConditionClassKind),
    Never,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequirementDescriptorKind {
    pub class: ConditionClassKind,
    pub provenance: ProvenanceKind,
}

pub fn domain_mode_allows_unproven(mode: DomainModeKind, class: ConditionClassKind) -> bool {
    match mode {
        DomainModeKind::Strict => false,
        DomainModeKind::Generic => class == ConditionClassKind::Definability,
        DomainModeKind::Assume => true,
    }
}

pub fn safe_for_prepass(safety: SolveSafetyKind) -> bool {
    matches!(safety, SolveSafetyKind::Always)
}

pub fn safe_for_tactic(safety: SolveSafetyKind, mode: DomainModeKind) -> bool {
    match safety {
        SolveSafetyKind::Always => true,
        SolveSafetyKind::IntrinsicCondition(_) => mode != DomainModeKind::Strict,
        SolveSafetyKind::NeedsCondition(class) => domain_mode_allows_unproven(mode, class),
        SolveSafetyKind::Never => false,
    }
}

pub fn requirement_descriptor(safety: SolveSafetyKind) -> Option<RequirementDescriptorKind> {
    match safety {
        SolveSafetyKind::Always => None,
        SolveSafetyKind::IntrinsicCondition(class) => Some(RequirementDescriptorKind {
            class,
            provenance: ProvenanceKind::Intrinsic,
        }),
        SolveSafetyKind::NeedsCondition(class) => Some(RequirementDescriptorKind {
            class,
            provenance: ProvenanceKind::Introduced,
        }),
        SolveSafetyKind::Never => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        requirement_descriptor, safe_for_prepass, safe_for_tactic, ConditionClassKind,
        ProvenanceKind, RequirementDescriptorKind, SolveSafetyKind,
    };
    use crate::log_domain::DomainModeKind;

    #[test]
    fn always_is_safe_everywhere() {
        let safety = SolveSafetyKind::Always;
        assert!(safe_for_prepass(safety));
        assert!(safe_for_tactic(safety, DomainModeKind::Strict));
        assert!(safe_for_tactic(safety, DomainModeKind::Generic));
        assert!(safe_for_tactic(safety, DomainModeKind::Assume));
    }

    #[test]
    fn intrinsic_blocks_only_in_strict_tactic() {
        let safety = SolveSafetyKind::IntrinsicCondition(ConditionClassKind::Analytic);
        assert!(!safe_for_prepass(safety));
        assert!(!safe_for_tactic(safety, DomainModeKind::Strict));
        assert!(safe_for_tactic(safety, DomainModeKind::Generic));
        assert!(safe_for_tactic(safety, DomainModeKind::Assume));
    }

    #[test]
    fn needs_definability_is_generic_or_assume_only() {
        let safety = SolveSafetyKind::NeedsCondition(ConditionClassKind::Definability);
        assert!(!safe_for_tactic(safety, DomainModeKind::Strict));
        assert!(safe_for_tactic(safety, DomainModeKind::Generic));
        assert!(safe_for_tactic(safety, DomainModeKind::Assume));
    }

    #[test]
    fn needs_analytic_is_assume_only() {
        let safety = SolveSafetyKind::NeedsCondition(ConditionClassKind::Analytic);
        assert!(!safe_for_tactic(safety, DomainModeKind::Strict));
        assert!(!safe_for_tactic(safety, DomainModeKind::Generic));
        assert!(safe_for_tactic(safety, DomainModeKind::Assume));
    }

    #[test]
    fn descriptor_mapping_is_stable() {
        let intrinsic = requirement_descriptor(SolveSafetyKind::IntrinsicCondition(
            ConditionClassKind::Analytic,
        ));
        assert_eq!(
            intrinsic,
            Some(RequirementDescriptorKind {
                class: ConditionClassKind::Analytic,
                provenance: ProvenanceKind::Intrinsic,
            })
        );

        let introduced = requirement_descriptor(SolveSafetyKind::NeedsCondition(
            ConditionClassKind::Definability,
        ));
        assert_eq!(
            introduced,
            Some(RequirementDescriptorKind {
                class: ConditionClassKind::Definability,
                provenance: ProvenanceKind::Introduced,
            })
        );
    }

    #[test]
    fn always_and_never_have_no_descriptor() {
        assert_eq!(requirement_descriptor(SolveSafetyKind::Always), None);
        assert_eq!(requirement_descriptor(SolveSafetyKind::Never), None);
    }
}
