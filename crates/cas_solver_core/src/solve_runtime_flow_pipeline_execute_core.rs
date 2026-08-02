use cas_ast::{Equation, ExprId, SolutionSet};

/// Execute solve strategy dispatch with the default strategy order and
/// per-strategy verification policy.
///
/// Callers provide:
/// - variable-presence check and residual fast-path resolver,
/// - cycle-guard entry,
/// - strategy application for each [`crate::strategy_order::SolveStrategyKind`],
/// - soft-error classifier,
/// - discrete-candidate resolution against `(equation, var)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_default_strategy_order_pipeline_with_state<
    SState,
    S,
    E,
    Guard,
    FContainsVar,
    FResolveVarEliminated,
    FEnterCycle,
    FApplyStrategy,
    FSoftError,
    FResolveDiscreteAgainstEquation,
>(
    state: &mut SState,
    original_equation: &Equation,
    normalized_equation: &Equation,
    residual: ExprId,
    var: &str,
    neq_discrete_backstop: bool,
    contains_var: FContainsVar,
    resolve_var_eliminated: FResolveVarEliminated,
    enter_cycle: FEnterCycle,
    mut apply_strategy: FApplyStrategy,
    is_soft_error: FSoftError,
    mut resolve_discrete_against_equation: FResolveDiscreteAgainstEquation,
    no_solution_error: E,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FContainsVar: FnMut(&mut SState, ExprId, &str) -> bool,
    FResolveVarEliminated: FnMut(&mut SState, ExprId, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FEnterCycle: FnMut(&mut SState, &Equation, &str) -> Result<Guard, E>,
    FApplyStrategy: FnMut(
        &mut SState,
        crate::strategy_order::SolveStrategyKind,
    ) -> Option<Result<(SolutionSet, Vec<S>), E>>,
    FSoftError: FnMut(&E) -> bool,
    FResolveDiscreteAgainstEquation:
        FnMut(&mut SState, &Equation, &str, Vec<ExprId>, Vec<S>) -> (SolutionSet, Vec<S>),
{
    let strategy_order = crate::strategy_order::default_solve_strategy_order();
    crate::solve_analysis::execute_prepared_equation_strategy_pipeline_with_state(
        state,
        normalized_equation,
        residual,
        var,
        strategy_order,
        contains_var,
        resolve_var_eliminated,
        enter_cycle,
        |state, strategy_kind| {
            let should_verify = crate::strategy_order::strategy_should_verify(*strategy_kind);
            let mut attempt = apply_strategy(state, *strategy_kind);
            // `!=` DISCRETE BACKSTOP: when the caller certified both equation
            // sides as TOTAL continuous functions, the `!=` solution set is
            // the preimage of an open set — OPEN — so a non-empty discrete
            // answer is mathematically impossible (it is the associated
            // `= 0` root set leaking through a terminal that lost the op:
            // Cardano/casus-irreducibilis cubics published their roots).
            // Reject it as a strategy decline so honest owners or an honest
            // exhaustion take over; every other shape passes through.
            if neq_discrete_backstop {
                if let Some(Ok((SolutionSet::Discrete(points), _))) = &attempt {
                    if !points.is_empty() {
                        attempt = None;
                    }
                }
            }
            (attempt, should_verify)
        },
        is_soft_error,
        |state, solutions, steps| {
            resolve_discrete_against_equation(state, original_equation, var, solutions, steps)
        },
        no_solution_error,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::RelOp;

    fn run_pipeline_with_discrete_strategy(
        backstop: bool,
    ) -> Result<(SolutionSet, Vec<()>), &'static str> {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let point = ctx.num(7);
        let equation = Equation {
            lhs: x,
            rhs: zero,
            op: RelOp::Neq,
        };
        execute_default_strategy_order_pipeline_with_state(
            &mut (),
            &equation,
            &equation,
            x,
            "x",
            backstop,
            |_, _, _| true,
            |_, _, _| Err("var-eliminated path unused"),
            |_, _, _| Ok::<(), &'static str>(()),
            |_, _| Some(Ok((SolutionSet::Discrete(vec![point]), Vec::new()))),
            |_| false,
            |_, _, _, solutions, steps| (SolutionSet::Discrete(solutions), steps),
            "no strategy",
        )
    }

    #[test]
    fn neq_discrete_backstop_rejects_impossible_discrete_answers() {
        // Armed (caller certified total-continuous `!=`): every strategy's
        // non-empty Discrete is formally impossible and demotes to a decline
        // — the pipeline exhausts honestly instead of publishing the
        // associated equation's roots.
        assert!(run_pipeline_with_discrete_strategy(true).is_err());
        // Disarmed: the same Discrete flows through unchanged.
        let (set, _) = run_pipeline_with_discrete_strategy(false).expect("solved");
        assert!(matches!(set, SolutionSet::Discrete(points) if points.len() == 1));
    }
}
