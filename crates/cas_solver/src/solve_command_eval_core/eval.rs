use super::{
    diagnostics::build_solve_command_diagnostics, SolveCommandEvalError, SolveCommandEvalOutput,
};

pub(crate) fn evaluate_solve_parsed_with_session<S>(
    simplifier: &mut crate::Simplifier,
    session: &mut S,
    raw_input: String,
    parsed_expr: cas_ast::ExprId,
    var: &str,
    auto_store: bool,
) -> Result<crate::EvalOutputView, String>
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

    let eq_to_solve = cas_solver_core::solve_entry::equation_from_expr_or_zero(
        &mut simplifier.context,
        resolved_input.resolved,
    );
    let eval_options = session.options().clone();
    let solver_opts = crate::SolverOptions::from_eval_options(&eval_options);

    let (solution_set, display_steps, solve_diagnostics) =
        crate::solve_with_display_steps(&eq_to_solve, var, simplifier, solver_opts)
            .map_err(|error| format!("Solver error: {error}"))?;

    let diagnostics = build_solve_command_diagnostics(
        &simplifier.context,
        resolved_input.resolved,
        &solution_set,
        &solve_diagnostics,
        &resolved_input.inherited_diagnostics,
    );
    let required_conditions = diagnostics.required_conditions();

    let no_simplified_cache: Option<
        cas_session_core::eval::SimplifiedUpdate<
            crate::DomainMode,
            crate::RequiredItem,
            crate::Step,
        >,
    > = None;
    cas_session_core::eval::apply_post_dispatch_store_updates(
        session.store_mut(),
        stored_id,
        diagnostics.clone(),
        no_simplified_cache,
    );

    let solver_assumptions =
        if eval_options.shared.assumption_reporting == crate::AssumptionReporting::Off {
            vec![]
        } else {
            solve_diagnostics.assumed_records.clone()
        };

    Ok(crate::EvalOutputView {
        stored_id,
        parsed: parsed_expr,
        resolved: resolved_input.resolved,
        result: crate::EvalResult::SolutionSet(solution_set),
        steps: crate::DisplayEvalSteps::default(),
        solve_steps: display_steps.into_inner(),
        output_scopes: solve_diagnostics.output_scopes.clone(),
        diagnostics,
        required_conditions,
        domain_warnings: Vec::new(),
        blocked_hints: Vec::new(),
        solver_assumptions,
    })
}

pub fn evaluate_solve_command_with_session<S>(
    simplifier: &mut crate::Simplifier,
    session: &mut S,
    parsed_input: crate::SolveCommandInput,
    auto_store: bool,
) -> Result<SolveCommandEvalOutput, SolveCommandEvalError>
where
    S: crate::SolverEvalSession,
{
    let prepared = super::prepare_solve_eval_request(
        &mut simplifier.context,
        parsed_input.equation.trim(),
        parsed_input.variable,
        auto_store,
    )
    .map_err(SolveCommandEvalError::Prepare)?;

    let output = evaluate_solve_parsed_with_session(
        simplifier,
        session,
        prepared.raw_input.clone(),
        prepared.parsed_expr,
        &prepared.var,
        prepared.auto_store,
    )
    .map_err(SolveCommandEvalError::Eval)?;

    Ok(SolveCommandEvalOutput {
        var: prepared.var,
        original_equation: prepared.original_equation,
        output,
    })
}
