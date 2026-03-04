use crate::{ConditionClass, DomainMode};

/// Extension trait to read rule safety using `cas_solver` public types.
pub trait RuleSolveSafetyExt {
    /// Returns solve-safety mapped to `cas_solver::SolveSafety`.
    fn solve_safety_model(&self) -> SolveSafety;
}

impl<T: crate::Rule + ?Sized> RuleSolveSafetyExt for T {
    fn solve_safety_model(&self) -> SolveSafety {
        SolveSafety::from(crate::Rule::solve_safety(self))
    }
}

/// Solver-facing solve-safety model with stable `cas_solver` domain types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolveSafety {
    #[default]
    Always,
    IntrinsicCondition(ConditionClass),
    NeedsCondition(ConditionClass),
    Never,
}

/// Solver-facing requirement descriptor.
pub type RequirementDescriptor = cas_solver_core::solve_safety_policy::RequirementDescriptorKind;

impl SolveSafety {
    #[inline]
    pub fn safe_for_prepass(&self) -> bool {
        let safety: cas_solver_core::solve_safety_policy::SolveSafetyKind = (*self).into();
        cas_solver_core::solve_safety_policy::safe_for_prepass(safety)
    }

    #[inline]
    pub fn safe_for_tactic(&self, domain_mode: DomainMode) -> bool {
        let safety: cas_solver_core::solve_safety_policy::SolveSafetyKind = (*self).into();
        cas_solver_core::solve_safety_policy::safe_for_tactic_with_domain_flags(
            safety,
            matches!(domain_mode, DomainMode::Assume),
            matches!(domain_mode, DomainMode::Strict),
        )
    }

    #[inline]
    pub fn requirement_descriptor(&self) -> Option<RequirementDescriptor> {
        let safety: cas_solver_core::solve_safety_policy::SolveSafetyKind = (*self).into();
        cas_solver_core::solve_safety_policy::requirement_descriptor(safety)
    }
}

impl From<cas_solver_core::solve_safety_policy::SolveSafetyKind> for SolveSafety {
    fn from(value: cas_solver_core::solve_safety_policy::SolveSafetyKind) -> Self {
        match value {
            cas_solver_core::solve_safety_policy::SolveSafetyKind::Always => SolveSafety::Always,
            cas_solver_core::solve_safety_policy::SolveSafetyKind::IntrinsicCondition(class) => {
                SolveSafety::IntrinsicCondition(class)
            }
            cas_solver_core::solve_safety_policy::SolveSafetyKind::NeedsCondition(class) => {
                SolveSafety::NeedsCondition(class)
            }
            cas_solver_core::solve_safety_policy::SolveSafetyKind::Never => SolveSafety::Never,
        }
    }
}

impl From<SolveSafety> for cas_solver_core::solve_safety_policy::SolveSafetyKind {
    fn from(value: SolveSafety) -> Self {
        match value {
            SolveSafety::Always => cas_solver_core::solve_safety_policy::SolveSafetyKind::Always,
            SolveSafety::IntrinsicCondition(class) => {
                cas_solver_core::solve_safety_policy::SolveSafetyKind::IntrinsicCondition(class)
            }
            SolveSafety::NeedsCondition(class) => {
                cas_solver_core::solve_safety_policy::SolveSafetyKind::NeedsCondition(class)
            }
            SolveSafety::Never => cas_solver_core::solve_safety_policy::SolveSafetyKind::Never,
        }
    }
}
