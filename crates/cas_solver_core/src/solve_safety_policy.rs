use crate::log_domain::DomainModeKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionClassKind {
    Definability,
    Analytic,
}

/// Canonical condition-class type shared by engine/solver runtime crates.
///
/// Keep `ConditionClassKind` as the explicit enum name for internal modules and
/// expose `ConditionClass` as the stable cross-crate alias.
pub type ConditionClass = ConditionClassKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvenanceKind {
    Intrinsic,
    Introduced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolveSafetyKind {
    #[default]
    Always,
    IntrinsicCondition(ConditionClassKind),
    NeedsCondition(ConditionClassKind),
    Never,
}

impl SolveSafetyKind {
    #[inline]
    pub fn safe_for_prepass(self) -> bool {
        safe_for_prepass(self)
    }

    #[inline]
    pub fn safe_for_tactic(self, mode: DomainModeKind) -> bool {
        safe_for_tactic(self, mode)
    }

    #[inline]
    pub fn requirement_descriptor(self) -> Option<RequirementDescriptorKind> {
        requirement_descriptor(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequirementDescriptorKind {
    pub class: ConditionClassKind,
    pub provenance: ProvenanceKind,
}

/// Purpose of simplification, controls which rule classes are allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimplifyPurpose {
    /// Standard evaluation - all rules allowed (default).
    #[default]
    Eval,
    /// Solver pre-pass - only `SolveSafetyKind::Always` rules.
    SolvePrepass,
    /// Solver tactic phase - conditional rules can run by domain policy.
    SolveTactic,
}

impl SimplifyPurpose {
    /// Returns true when this phase must suppress assumption generation.
    #[inline]
    pub fn blocks_assumptions(self) -> bool {
        matches!(self, SimplifyPurpose::SolvePrepass)
    }
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

#[inline]
pub fn safe_for_tactic_with_domain_flags(
    safety: SolveSafetyKind,
    assume_mode: bool,
    strict_mode: bool,
) -> bool {
    safe_for_tactic(
        safety,
        crate::strategy_options::solver_domain_mode_kind(assume_mode, strict_mode),
    )
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

#[inline]
pub fn requires_introduced_analytic_condition(safety: SolveSafetyKind) -> bool {
    requirement_descriptor(safety).is_some_and(|req| {
        req.class == ConditionClassKind::Analytic && req.provenance == ProvenanceKind::Introduced
    })
}

#[cfg(test)]
mod tests {
    use super::{
        requirement_descriptor, requires_introduced_analytic_condition, safe_for_prepass,
        safe_for_tactic, safe_for_tactic_with_domain_flags, ConditionClassKind, ProvenanceKind,
        RequirementDescriptorKind, SimplifyPurpose, SolveSafetyKind,
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
    fn safe_for_tactic_flags_matches_mode_dispatch() {
        let safety = SolveSafetyKind::NeedsCondition(ConditionClassKind::Definability);
        assert_eq!(
            safe_for_tactic_with_domain_flags(safety, false, true),
            safe_for_tactic(safety, DomainModeKind::Strict)
        );
        assert_eq!(
            safe_for_tactic_with_domain_flags(safety, false, false),
            safe_for_tactic(safety, DomainModeKind::Generic)
        );
        assert_eq!(
            safe_for_tactic_with_domain_flags(safety, true, false),
            safe_for_tactic(safety, DomainModeKind::Assume)
        );
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

    #[test]
    fn introduced_analytic_requirement_detector_matches_policy() {
        assert!(!requires_introduced_analytic_condition(
            SolveSafetyKind::Always
        ));
        assert!(!requires_introduced_analytic_condition(
            SolveSafetyKind::IntrinsicCondition(ConditionClassKind::Analytic)
        ));
        assert!(!requires_introduced_analytic_condition(
            SolveSafetyKind::NeedsCondition(ConditionClassKind::Definability)
        ));
        assert!(requires_introduced_analytic_condition(
            SolveSafetyKind::NeedsCondition(ConditionClassKind::Analytic)
        ));
    }

    #[test]
    fn simplify_purpose_blocks_assumptions_only_for_prepass() {
        assert!(!SimplifyPurpose::Eval.blocks_assumptions());
        assert!(SimplifyPurpose::SolvePrepass.blocks_assumptions());
        assert!(!SimplifyPurpose::SolveTactic.blocks_assumptions());
    }
}
