use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{SolveCtx, SolveStep, SolverOptions};
use cas_ast::SolutionSet;

use super::isolation::isolate;

pub(super) fn isolate_equation(
    equation: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    isolate(
        equation.lhs,
        equation.rhs,
        equation.op.clone(),
        var,
        simplifier,
        opts,
        ctx,
    )
}

pub(super) fn isolate_equation_solutions(
    equation: &cas_ast::Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    ctx: &SolveCtx,
) -> Option<SolutionSet> {
    isolate_equation(equation, var, simplifier, opts, ctx)
        .ok()
        .map(|(solutions, _)| solutions)
}
