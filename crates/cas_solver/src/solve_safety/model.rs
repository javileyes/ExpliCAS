mod convert;
mod methods;

use crate::{ConditionClass, DomainMode};

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
