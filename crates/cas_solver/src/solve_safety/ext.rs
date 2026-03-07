use super::model::SolveSafety;

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
