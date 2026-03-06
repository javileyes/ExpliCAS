#[derive(Debug, Clone)]
pub(crate) struct PreparedSolveEvalRequest {
    pub raw_input: String,
    pub parsed_expr: cas_ast::ExprId,
    pub auto_store: bool,
    pub var: String,
    pub original_equation: Option<cas_ast::Equation>,
}

#[derive(Debug, Clone)]
pub struct SolveCommandEvalOutput {
    pub var: String,
    pub original_equation: Option<cas_ast::Equation>,
    pub output: crate::EvalOutputView,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveCommandEvalError {
    Prepare(crate::SolvePrepareError),
    Eval(String),
}

fn first_solution_expr(solution_set: &cas_ast::SolutionSet) -> Option<cas_ast::ExprId> {
    match solution_set {
        cas_ast::SolutionSet::Discrete(values) => values.first().copied(),
        _ => None,
    }
}

fn build_solve_command_diagnostics(
    ctx: &cas_ast::Context,
    resolved: cas_ast::ExprId,
    solution_set: &cas_ast::SolutionSet,
    solve_diagnostics: &crate::SolveDiagnostics,
    inherited: &crate::Diagnostics,
) -> crate::Diagnostics {
    let mut diagnostics = crate::Diagnostics::new();

    for cond in &solve_diagnostics.required {
        diagnostics.push_required(cond.clone(), crate::RequireOrigin::EquationDerived);
    }
    for event in &solve_diagnostics.assumed {
        diagnostics.push_assumed(event.clone());
    }

    let input_domain = crate::infer_implicit_domain(ctx, resolved, crate::ValueDomain::RealOnly);
    for cond in input_domain.conditions() {
        diagnostics.push_required(cond.clone(), crate::RequireOrigin::InputImplicit);
    }

    if let Some(result_expr) = first_solution_expr(solution_set) {
        let output_domain =
            crate::infer_implicit_domain(ctx, result_expr, crate::ValueDomain::RealOnly);
        for cond in output_domain.conditions() {
            diagnostics.push_required(cond.clone(), crate::RequireOrigin::OutputImplicit);
        }
    }

    diagnostics.inherit_requires_from(inherited);
    diagnostics.dedup_and_sort(ctx);
    diagnostics
}

pub(crate) fn prepare_solve_eval_request(
    ctx: &mut cas_ast::Context,
    input: &str,
    explicit_var: Option<String>,
    auto_store: bool,
) -> Result<PreparedSolveEvalRequest, crate::SolvePrepareError> {
    let (parsed_expr, original_equation, var) =
        crate::prepare_solve_expr_and_var(ctx, input, explicit_var)?;

    Ok(PreparedSolveEvalRequest {
        raw_input: input.to_string(),
        parsed_expr,
        auto_store,
        var,
        original_equation,
    })
}

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
    .map_err(|e| e.to_string())?;

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
            .map_err(|e| format!("Solver error: {e}"))?;

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
    let prepared = prepare_solve_eval_request(
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
