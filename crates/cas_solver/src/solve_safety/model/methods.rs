use super::{DomainMode, RequirementDescriptor, SolveSafety};

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
