fn first_solution_expr(solution_set: &cas_ast::SolutionSet) -> Option<cas_ast::ExprId> {
    match solution_set {
        cas_ast::SolutionSet::Discrete(values) => values.first().copied(),
        _ => None,
    }
}

pub(crate) fn build_solve_command_diagnostics(
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
