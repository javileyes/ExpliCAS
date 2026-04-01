use std::collections::BTreeSet;

pub(crate) fn evaluate_linear_system_eval_request_with_session<S>(
    engine: &mut crate::Engine,
    session: &mut S,
    parsed_anchor: cas_ast::ExprId,
    exprs: Vec<cas_ast::ExprId>,
    vars: Vec<String>,
) -> Result<crate::EvalOutputView, String>
where
    S: crate::SolverEvalSession,
{
    let mut resolved_exprs = Vec::with_capacity(exprs.len());
    let mut diagnostics = crate::Diagnostics::new();
    let mut cache_hits = BTreeSet::new();
    let value_domain = session.options().shared.semantics.value_domain;

    for expr in exprs {
        let (resolved, inherited, hits) = session
            .resolve_all_with_diagnostics(&mut engine.simplifier.context, expr)
            .map_err(|e| format!("Resolution error: {e}"))?;

        if let Some(name) = cas_session_core::eval::first_unknown_function_name(
            session,
            &engine.simplifier.context,
            resolved,
        ) {
            return Err(format!("Error: {}", crate::CasError::UnknownFunction(name)));
        }

        diagnostics.extend_required(
            crate::infer_implicit_domain(&engine.simplifier.context, resolved, value_domain)
                .conditions()
                .iter()
                .cloned(),
            crate::RequireOrigin::InputImplicit,
        );
        diagnostics.inherit_requires_from(&inherited);
        cache_hits.extend(hits);
        resolved_exprs.push(resolved);
    }

    cas_session_core::eval::touch_cache_hits(
        session.store_mut(),
        &cache_hits.into_iter().collect::<Vec<_>>(),
    );

    let result =
        super::solve::solve_linear_system_parts(&engine.simplifier.context, &resolved_exprs, &vars)
            .map_err(|error| {
                crate::linear_system_command_format::format_linear_system_command_error_message(
                    &crate::linear_system_command_eval::LinearSystemCommandEvalError::Solve(error),
                )
            })?;

    let command_output =
        crate::linear_system_command_eval::LinearSystemCommandEvalOutput { vars, result };
    let (plain, latex) = crate::linear_system_command_format::render_linear_system_result(
        &mut engine.simplifier.context,
        &command_output,
    );

    diagnostics.dedup_and_sort(&engine.simplifier.context);
    let required_conditions = diagnostics.required_conditions();

    Ok(crate::EvalOutputView {
        stored_id: None,
        parsed: parsed_anchor,
        resolved: resolved_exprs.first().copied().unwrap_or(parsed_anchor),
        result: crate::EvalResult::Text { plain, latex },
        steps: crate::DisplayEvalSteps::default(),
        solve_steps: Vec::new(),
        output_scopes: Vec::new(),
        diagnostics,
        required_conditions,
        domain_warnings: Vec::new(),
        blocked_hints: Vec::new(),
        solver_assumptions: Vec::new(),
    })
}
