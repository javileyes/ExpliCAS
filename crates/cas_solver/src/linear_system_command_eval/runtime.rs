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

    let mut nonlinear_narration: Option<super::nonlinear::NonlinearNarration> = None;
    // A well-formed system that this solver cannot handle is a MATH decline,
    // not an internal fault: it must reach the wire as an honest ok-result
    // (the REPL route already behaves this way — this is parity, and it is
    // what keeps `solve([a*x+y=1, x-y=0], [x, y])`-class inputs from ever
    // surfacing as E_INTERNAL again).
    let result =
        match super::solve::solve_linear_system_parts(
            &mut engine.simplifier.context,
            &resolved_exprs,
            &vars,
        ) {
            Ok(result) => result,
            // S2: a true nonlinearity in the unknowns gets one composition shot —
            // isolate-substitute-solve-verify — before the honest decline.
            Err(error) => match super::nonlinear::try_solve_nonlinear_2x2(
                &mut engine.simplifier,
                &resolved_exprs,
                &vars,
            ) {
                Some((result, narr)) => {
                    nonlinear_narration = narr;
                    result
                }
                None => {
                    let message =
                crate::linear_system_command_format::format_linear_system_command_error_message(
                    &crate::linear_system_command_eval::LinearSystemCommandEvalError::Solve(error),
                );
                    let latex = format!(
                        "\\text{{{}}}",
                        cas_formatter::latex_escape(&message.replace('\n', " "))
                    );
                    diagnostics.dedup_and_sort(&engine.simplifier.context);
                    let required_conditions = diagnostics.required_conditions();
                    return Ok(crate::EvalOutputView {
                        stored_id: None,
                        parsed: parsed_anchor,
                        resolved: resolved_exprs.first().copied().unwrap_or(parsed_anchor),
                        result: crate::EvalResult::Text {
                            plain: message,
                            latex: Some(latex),
                        },
                        strategy: None,
                        steps: crate::DisplayEvalSteps::default(),
                        solve_steps: Vec::new(),
                        output_scopes: Vec::new(),
                        diagnostics,
                        required_conditions,
                        domain_warnings: Vec::new(),
                        blocked_hints: Vec::new(),
                        solver_assumptions: Vec::new(),
                    });
                }
            },
        };

    // Symbolic solutions carry their validity requirements (det ≠ 0) — those
    // ride the canonical required-conditions channel, same as any division.
    if let crate::LinSolveResult::UniqueExpr {
        nonzero_conditions, ..
    } = &result
    {
        diagnostics.extend_required(
            nonzero_conditions
                .iter()
                .map(|&e| cas_solver_core::domain_condition::ImplicitCondition::NonZero(e)),
            crate::RequireOrigin::EquationDerived,
        );
    }

    // S3: the educational half — every outcome narrates (localized at the
    // presentation boundary via the SolveDesc table).
    let solve_steps = super::steps::build_system_solve_steps(
        &mut engine.simplifier.context,
        &resolved_exprs,
        &vars,
        &result,
        nonlinear_narration.as_ref(),
    );

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
        strategy: None,
        steps: crate::DisplayEvalSteps::default(),
        solve_steps,
        output_scopes: Vec::new(),
        diagnostics,
        required_conditions,
        domain_warnings: Vec::new(),
        blocked_hints: Vec::new(),
        solver_assumptions: Vec::new(),
    })
}
