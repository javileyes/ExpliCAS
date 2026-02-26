pub(crate) mod isolation_strategy;
pub(crate) mod quadratic;
pub(crate) mod rational_roots;
pub(crate) mod substitution;

use super::strategy::SolverStrategy;
use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{SolveCtx, SolveStep, SolverOptions};
use cas_ast::{Equation, RelOp, SolutionSet};
use isolation_strategy::{
    CollectTermsStrategy, IsolationStrategy, RationalExponentStrategy, UnwrapStrategy,
};
use quadratic::QuadraticStrategy;
use rational_roots::RationalRootsStrategy;
use substitution::SubstitutionStrategy;

/// Default solver strategy sequence used by `solve_core`.
///
/// Ordering matters for correctness and loop avoidance.
pub(crate) fn default_strategies() -> Vec<Box<dyn SolverStrategy>> {
    vec![
        Box::new(RationalExponentStrategy), // Must run BEFORE UnwrapStrategy to avoid loops
        Box::new(SubstitutionStrategy),
        Box::new(UnwrapStrategy),
        Box::new(QuadraticStrategy),
        Box::new(RationalRootsStrategy), // Degree ≥ 3 with numeric coefficients
        Box::new(CollectTermsStrategy),  // Must run before IsolationStrategy
        Box::new(IsolationStrategy),
    ]
}

/// Run the early rational-exponent precheck (`x^(p/q) = rhs`) before generic
/// simplification to avoid fractional-power rewrite loops.
pub(crate) fn try_rational_exponent_precheck(
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    ctx: &SolveCtx,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    if equation.op != RelOp::Eq {
        return None;
    }

    RationalExponentStrategy.apply(equation, var, simplifier, opts, ctx)
}
