use cas_solver_core::solve_safety_policy as core_safety;

/// Safety classification for rules when used during equation solving.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolveSafety {
    /// Global equivalence, always safe in solver pre-pass.
    #[default]
    Always,
    /// Condition is already guaranteed by the input expression.
    IntrinsicCondition(crate::ConditionClass),
    /// Valid only if additional conditions are introduced.
    NeedsCondition(crate::ConditionClass),
    /// Never safe for solver context.
    Never,
}

/// Bridge between static rule safety metadata and domain vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequirementDescriptor {
    /// What kind of condition the rule needs.
    pub class: crate::ConditionClass,
    /// Where the condition comes from.
    pub provenance: crate::Provenance,
}

/// Extension trait to read rule safety using `cas_solver` local types.
pub trait RuleSolveSafetyExt {
    /// Returns solve-safety mapped to `cas_solver::SolveSafety`.
    fn solve_safety_model(&self) -> SolveSafety;
}

impl<T: crate::Rule + ?Sized> RuleSolveSafetyExt for T {
    fn solve_safety_model(&self) -> SolveSafety {
        SolveSafety::from(crate::Rule::solve_safety(self))
    }
}

impl SolveSafety {
    /// Returns true if this rule is safe in solver pre-pass.
    #[inline]
    pub fn safe_for_prepass(&self) -> bool {
        core_safety::safe_for_prepass(to_core_solve_safety(*self))
    }

    /// Returns true if this rule is safe for solver tactic with the given domain mode.
    #[inline]
    pub fn safe_for_tactic(&self, domain_mode: crate::DomainMode) -> bool {
        core_safety::safe_for_tactic(
            to_core_solve_safety(*self),
            to_core_domain_mode(domain_mode),
        )
    }

    /// Maps this safety classification to a requirement descriptor.
    #[inline]
    pub fn requirement_descriptor(&self) -> Option<RequirementDescriptor> {
        core_safety::requirement_descriptor(to_core_solve_safety(*self)).map(|desc| {
            RequirementDescriptor {
                class: from_core_condition_class(desc.class),
                provenance: from_core_provenance(desc.provenance),
            }
        })
    }

    /// Converts to engine solve-safety.
    #[inline]
    pub fn into_engine(self) -> cas_engine::SolveSafety {
        self.into()
    }
}

impl From<cas_engine::SolveSafety> for SolveSafety {
    fn from(value: cas_engine::SolveSafety) -> Self {
        match value {
            cas_engine::SolveSafety::Always => Self::Always,
            cas_engine::SolveSafety::IntrinsicCondition(class) => {
                Self::IntrinsicCondition(class.into())
            }
            cas_engine::SolveSafety::NeedsCondition(class) => Self::NeedsCondition(class.into()),
            cas_engine::SolveSafety::Never => Self::Never,
        }
    }
}

impl From<SolveSafety> for cas_engine::SolveSafety {
    fn from(value: SolveSafety) -> Self {
        match value {
            SolveSafety::Always => cas_engine::SolveSafety::Always,
            SolveSafety::IntrinsicCondition(class) => {
                cas_engine::SolveSafety::IntrinsicCondition(class.into())
            }
            SolveSafety::NeedsCondition(class) => {
                cas_engine::SolveSafety::NeedsCondition(class.into())
            }
            SolveSafety::Never => cas_engine::SolveSafety::Never,
        }
    }
}

impl From<cas_engine::RequirementDescriptor> for RequirementDescriptor {
    fn from(value: cas_engine::RequirementDescriptor) -> Self {
        Self {
            class: value.class.into(),
            provenance: value.provenance.into(),
        }
    }
}

impl From<RequirementDescriptor> for cas_engine::RequirementDescriptor {
    fn from(value: RequirementDescriptor) -> Self {
        Self {
            class: value.class.into(),
            provenance: value.provenance.into(),
        }
    }
}

fn to_core_condition_class(class: crate::ConditionClass) -> core_safety::ConditionClassKind {
    match class {
        crate::ConditionClass::Definability => core_safety::ConditionClassKind::Definability,
        crate::ConditionClass::Analytic => core_safety::ConditionClassKind::Analytic,
    }
}

fn from_core_condition_class(class: core_safety::ConditionClassKind) -> crate::ConditionClass {
    match class {
        core_safety::ConditionClassKind::Definability => crate::ConditionClass::Definability,
        core_safety::ConditionClassKind::Analytic => crate::ConditionClass::Analytic,
    }
}

fn from_core_provenance(provenance: core_safety::ProvenanceKind) -> crate::Provenance {
    match provenance {
        core_safety::ProvenanceKind::Intrinsic => crate::Provenance::Intrinsic,
        core_safety::ProvenanceKind::Introduced => crate::Provenance::Introduced,
    }
}

fn to_core_solve_safety(safety: SolveSafety) -> core_safety::SolveSafetyKind {
    match safety {
        SolveSafety::Always => core_safety::SolveSafetyKind::Always,
        SolveSafety::IntrinsicCondition(class) => {
            core_safety::SolveSafetyKind::IntrinsicCondition(to_core_condition_class(class))
        }
        SolveSafety::NeedsCondition(class) => {
            core_safety::SolveSafetyKind::NeedsCondition(to_core_condition_class(class))
        }
        SolveSafety::Never => core_safety::SolveSafetyKind::Never,
    }
}

fn to_core_domain_mode(
    domain_mode: crate::DomainMode,
) -> cas_solver_core::log_domain::DomainModeKind {
    cas_solver_core::log_domain::domain_mode_kind_from_flags(
        matches!(domain_mode, crate::DomainMode::Assume),
        matches!(domain_mode, crate::DomainMode::Strict),
    )
}
