use super::SolveSessionExecution;

pub(super) fn solve_parsed_with_session<S>(
    simplifier: &mut crate::Simplifier,
    session: &mut S,
    raw_input: String,
    parsed_expr: cas_ast::ExprId,
    original_equation: Option<&cas_ast::Equation>,
    var: &str,
    auto_store: bool,
) -> Result<SolveSessionExecution, String>
where
    S: crate::SolverEvalSession,
{
    let resolved_input = cas_session_core::eval::resolve_eval_input(
        session,
        &mut simplifier.context,
        parsed_expr,
        None,
    )
    .map_err(|error| error.to_string())?;

    let stored_id = cas_session_core::eval::apply_pre_dispatch_store_updates(
        session.store_mut(),
        &simplifier.context,
        parsed_expr,
        raw_input,
        auto_store,
        &resolved_input.cache_hits,
    );

    let mut eq_to_solve = cas_solver_core::solve_entry::equation_from_expr_or_zero(
        &mut simplifier.context,
        resolved_input.resolved,
    );
    if let Some(original_equation) = original_equation {
        eq_to_solve.op = original_equation.op.clone();
    }
    let eval_options = session.options().clone();
    let solver_opts = crate::SolverOptions::from_eval_options(&eval_options);

    let (solution_set, display_steps, solve_diagnostics) =
        crate::solve_with_display_steps(&eq_to_solve, var, simplifier, solver_opts)
            .map_err(|error| format!("Solver error: {error}"))?;

    Ok(SolveSessionExecution {
        stored_id,
        parsed_expr,
        resolved: resolved_input.resolved,
        solution_set,
        display_steps,
        solve_diagnostics,
        inherited_diagnostics: resolved_input.inherited_diagnostics,
        eval_options,
    })
}
