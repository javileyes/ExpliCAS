use super::{super::diagnostics::build_solve_command_diagnostics, SolveSessionExecution};

fn normalize_solver_assumption_records(
    simplifier: &mut crate::Simplifier,
    records: Vec<crate::AssumptionRecord>,
) -> Vec<crate::AssumptionRecord> {
    records
        .into_iter()
        .map(|mut record| {
            if let Ok(expr_id) = cas_parser::parse(&record.expr, &mut simplifier.context) {
                let (normalized_id, _) = simplifier.simplify(expr_id);
                record.expr = cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id: normalized_id,
                }
                .to_string();
            }
            record
        })
        .collect()
}

pub(super) fn finalize_solve_eval_output<S>(
    session: &mut S,
    simplifier: &mut crate::Simplifier,
    execution: SolveSessionExecution,
) -> crate::EvalOutputView
where
    S: crate::SolverEvalSession,
{
    let diagnostics = build_solve_command_diagnostics(
        &simplifier.context,
        execution.resolved,
        &execution.solution_set,
        &execution.solve_diagnostics,
        &execution.inherited_diagnostics,
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
        execution.stored_id,
        diagnostics.clone(),
        no_simplified_cache,
    );

    let solver_assumptions =
        if execution.eval_options.shared.assumption_reporting == crate::AssumptionReporting::Off {
            vec![]
        } else {
            normalize_solver_assumption_records(
                simplifier,
                execution.solve_diagnostics.assumed_records.clone(),
            )
        };

    crate::EvalOutputView {
        stored_id: execution.stored_id,
        parsed: execution.parsed_expr,
        resolved: execution.resolved,
        result: crate::EvalResult::SolutionSet(execution.solution_set),
        steps: crate::DisplayEvalSteps::default(),
        solve_steps: execution.display_steps.into_inner(),
        output_scopes: execution.solve_diagnostics.output_scopes.clone(),
        diagnostics,
        required_conditions,
        domain_warnings: Vec::new(),
        blocked_hints: Vec::new(),
        solver_assumptions,
    }
}
