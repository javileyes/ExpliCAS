#[derive(Debug, Clone, Copy)]
pub enum VerbosityLevel {
    Low,     // Only high-importance steps (Factor, Expand, Integrate, etc.)
    Normal,  // Medium+ importance steps (most transformations)
    Verbose, // All steps including trivial ones
}

impl VerbosityLevel {
    pub(super) fn step_visibility(&self) -> crate::didactic::StepVisibility {
        match self {
            VerbosityLevel::Verbose => crate::didactic::StepVisibility::All,
            VerbosityLevel::Low => crate::didactic::StepVisibility::HighOrHigher,
            VerbosityLevel::Normal => crate::didactic::StepVisibility::MediumOrHigher,
        }
    }
}
