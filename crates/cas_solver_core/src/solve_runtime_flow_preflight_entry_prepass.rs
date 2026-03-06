use cas_ast::{Equation, ExprId, RelOp, SolutionSet};

/// Result of running solve preflight plus optional early rational-exponent prepass.
pub enum PreflightOrSolved<Ctx, S, E> {
    Continue {
        domain_exclusions: Vec<ExprId>,
        ctx: Ctx,
    },
    Solved(Result<(SolutionSet, Vec<S>), E>),
}

/// Try the default early rational-exponent prepass:
/// - only for equality equations (`RelOp::Eq`)
/// - if a solve result appears, guard it against domain exclusions
pub fn try_apply_rational_exponent_prepass_with_default_eq_guard_and_exclusion_policy_with_state<
    SState,
    S,
    E,
    FApplyRationalExponent,
    FGuardSolved,
>(
    state: &mut SState,
    equation: &Equation,
    var: &str,
    domain_exclusions: &[ExprId],
    mut apply_rational_exponent: FApplyRationalExponent,
    mut guard_solved_result: FGuardSolved,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    FApplyRationalExponent:
        FnMut(&mut SState, &Equation, &str) -> Option<Result<(SolutionSet, Vec<S>), E>>,
    FGuardSolved:
        FnMut(Result<(SolutionSet, Vec<S>), E>, &[ExprId]) -> Result<(SolutionSet, Vec<S>), E>,
{
    if equation.op != RelOp::Eq {
        return None;
    }
    let result = apply_rational_exponent(state, equation, var)?;
    Some(guard_solved_result(result, domain_exclusions))
}

/// Build preflight context and run the default early rational-exponent prepass.
///
/// Returns:
/// - [`PreflightOrSolved::Solved`] when the prepass solved the equation, or
/// - [`PreflightOrSolved::Continue`] with `(domain_exclusions, solve_ctx)` for
///   the regular strategy pipeline.
pub fn run_preflight_with_default_rational_exponent_prepass_with_state<
    SState,
    Ctx,
    S,
    E,
    FBuildPreflight,
    FApplyRationalExponent,
    FGuardSolved,
>(
    state: &mut SState,
    equation: &Equation,
    var: &str,
    build_preflight: FBuildPreflight,
    mut apply_rational_exponent: FApplyRationalExponent,
    guard_solved_result: FGuardSolved,
) -> PreflightOrSolved<Ctx, S, E>
where
    FBuildPreflight: FnOnce(&mut SState) -> crate::solve_analysis::PreflightContext<Ctx>,
    FApplyRationalExponent:
        FnMut(&mut SState, &Equation, &str, &Ctx) -> Option<Result<(SolutionSet, Vec<S>), E>>,
    FGuardSolved:
        FnMut(Result<(SolutionSet, Vec<S>), E>, &[ExprId]) -> Result<(SolutionSet, Vec<S>), E>,
{
    let preflight = build_preflight(state);
    let domain_exclusions = preflight.domain_exclusions;
    let ctx = preflight.ctx;

    if let Some(result) =
        try_apply_rational_exponent_prepass_with_default_eq_guard_and_exclusion_policy_with_state(
            state,
            equation,
            var,
            &domain_exclusions,
            |state, equation, solve_var| apply_rational_exponent(state, equation, solve_var, &ctx),
            guard_solved_result,
        )
    {
        return PreflightOrSolved::Solved(result);
    }

    PreflightOrSolved::Continue {
        domain_exclusions,
        ctx,
    }
}
