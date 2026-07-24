//! Core solve dispatch pipeline.
//!
//! Contains the internal solve pipeline (`solve_inner`).

use crate::solve_backend_contract::CoreSolverOptions;
use crate::{CasError, Simplifier};
use cas_ast::{Equation, SolutionSet};

/// Core solver implementation.
///
/// All public entry points delegate here. `parent_ctx` carries the shared
/// accumulator so that conditions from recursive calls are visible to the
/// top-level caller.
pub(crate) fn solve_inner(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: CoreSolverOptions,
    parent_ctx: &crate::SolveCtx,
) -> Result<(SolutionSet, Vec<crate::SolveStep>), CasError> {
    // A bare trig equation `sin/cos/tan(x)=c` has an INFINITE periodic family of roots. The top-level
    // entry (`solve_local_core`) already checks this, but the RECURSIVE route (this function, used to
    // solve each `factor = 0` of a zero-product) bypassed it and fell to the unary-inverse path, which
    // returns only the PRINCIPAL root. That dropped periodicity for products of trig factors
    // (`solve(sin(x)*cos(x)=0) -> {0, π/2}` instead of the periodic union). Run the periodic solver on
    // every recursive sub-solve so each trig factor yields its full `SolutionSet::Periodic` family.
    if let Some((set, steps)) =
        crate::solve_backend_local::try_solve_periodic_trig_equation_with_steps(eq, var, simplifier)
    {
        return Ok((set, steps));
    }
    // Same recursive-bypass shape for a linear INEQUALITY with the variable on both sides and a
    // symbolic-constant coefficient: the log-linearization of `e^x {op} 2^x` recurses here with
    // `x·ln(e) {op} ln(2^x)` (= `x {op} x·ln2`), and the runtime isolation's equation-only
    // linear-collect would drop the operator (spurious boundary point `{0}` instead of the ray).
    // Collect to `c1·x + c0`, decide sign(c1) exactly, and recurse — or decline honestly.
    if let Some(result) =
        crate::solve_backend_local::try_symbolic_linear_coeff_inequality(simplifier, eq, var)
    {
        return result;
    }
    cas_solver_core::solve_runtime_recursive_bound_runtime::solve_inner_with_runtime_state_and_default_recursive_routes_and_errors(
        eq,
        var,
        simplifier,
        opts,
        parent_ctx,
        solve_inner,
        crate::register_blocked_hint,
        || {
            CasError::SolverError(
                "Maximum solver recursion depth exceeded. The equation may be too complex."
                    .to_string(),
            )
        },
        || CasError::VariableNotFound(var.to_string()),
    )
}
