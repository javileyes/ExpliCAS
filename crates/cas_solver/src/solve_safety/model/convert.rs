use super::SolveSafety;

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
