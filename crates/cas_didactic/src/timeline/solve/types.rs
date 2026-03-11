use crate::cas_solver::SolveStep;
use cas_ast::{Context, Equation, SolutionSet};

/// Timeline HTML generator for equation solving steps
pub struct SolveTimelineHtml<'a> {
    pub(super) context: &'a mut Context,
    pub(super) steps: &'a [SolveStep],
    pub(super) original_eq: &'a Equation,
    pub(super) solution_set: &'a SolutionSet,
    pub(super) var: String,
    pub(super) title: String,
}
