use super::strategy_apply::apply_rational_exponent_strategy;
use super::{SolveCtx, SolveStep, SolverOptions};
use crate::engine::Simplifier;
use crate::error::CasError;
use cas_ast::{Equation, RelOp, SolutionSet};

/// Run the early rational-exponent precheck (`x^(p/q) = rhs`) before generic
/// simplification to avoid fractional-power rewrite loops.
pub(super) fn try_rational_exponent_precheck(
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    ctx: &SolveCtx,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    if equation.op != RelOp::Eq {
        return None;
    }

    apply_rational_exponent_strategy(equation, var, simplifier, opts, ctx)
}
