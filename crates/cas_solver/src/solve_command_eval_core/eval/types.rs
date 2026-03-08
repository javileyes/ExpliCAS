pub(super) struct SolveSessionExecution {
    pub(super) stored_id: Option<u64>,
    pub(super) parsed_expr: cas_ast::ExprId,
    pub(super) resolved: cas_ast::ExprId,
    pub(super) solution_set: cas_ast::SolutionSet,
    pub(super) display_steps: crate::DisplaySolveSteps,
    pub(super) solve_diagnostics: crate::SolveDiagnostics,
    pub(super) inherited_diagnostics: crate::Diagnostics,
    pub(super) eval_options: crate::EvalOptions,
}
