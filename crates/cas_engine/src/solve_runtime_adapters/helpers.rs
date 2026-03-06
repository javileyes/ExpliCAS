use super::*;
use crate::engine::Simplifier;
use crate::error::CasError;
use cas_ast::symbol::SymbolId;
use cas_ast::{Equation, ExprId, SolutionSet};
use cas_solver_core::external_proof::map_external_nonzero_status_with;

pub(crate) fn simplifier_context(simplifier: &mut Simplifier) -> &cas_ast::Context {
    &simplifier.context
}

pub(crate) fn simplifier_context_mut(simplifier: &mut Simplifier) -> &mut cas_ast::Context {
    &mut simplifier.context
}

pub(crate) fn simplifier_contains_var(simplifier: &mut Simplifier, expr: ExprId, var: &str) -> bool {
    cas_solver_core::isolation_utils::contains_var(&simplifier.context, expr, var)
}

pub(crate) fn simplifier_simplify_expr(simplifier: &mut Simplifier, expr: ExprId) -> ExprId {
    simplifier.simplify(expr).0
}

pub(crate) fn simplifier_expand_expr(simplifier: &mut Simplifier, expr: ExprId) -> ExprId {
    crate::expand(&mut simplifier.context, expr)
}

pub(crate) fn simplifier_render_expr(simplifier: &mut Simplifier, expr: ExprId) -> String {
    cas_formatter::render_expr(&simplifier.context, expr)
}

pub(crate) fn context_render_expr(ctx: &cas_ast::Context, expr: ExprId) -> String {
    cas_formatter::render_expr(ctx, expr)
}

pub(crate) fn simplifier_zero_expr(simplifier: &mut Simplifier) -> ExprId {
    simplifier.context.num(0)
}

pub(crate) fn simplifier_collect_steps(simplifier: &mut Simplifier) -> bool {
    simplifier.collect_steps()
}

pub(crate) fn simplifier_prove_nonzero_status(
    simplifier: &mut Simplifier,
    expr: ExprId,
) -> cas_solver_core::linear_solution::NonZeroStatus {
    map_external_nonzero_status_with(
        crate::prove_nonzero(&simplifier.context, expr),
        |proof| matches!(proof, crate::Proof::Proven | crate::Proof::ProvenImplicit),
        |proof| matches!(proof, crate::Proof::Disproven),
    )
}

pub(crate) fn simplifier_is_known_negative(simplifier: &mut Simplifier, expr: ExprId) -> bool {
    cas_solver_core::isolation_utils::is_known_negative(&simplifier.context, expr)
}

pub(crate) fn medium_step(description: String, equation_after: Equation) -> SolveStep {
    SolveStep::new(description, equation_after, crate::ImportanceLevel::Medium)
}

pub(crate) fn low_substep(description: String, equation_after: Equation) -> SolveSubStep {
    SolveSubStep {
        description,
        equation_after,
        importance: crate::ImportanceLevel::Low,
    }
}

pub(crate) fn attach_substeps(step: SolveStep, substeps: Vec<SolveSubStep>) -> SolveStep {
    step.with_substeps(substeps)
}

pub(crate) fn map_symbolic_inequalities_not_supported_error(
    _plan_error: cas_solver_core::quadratic_formula::QuadraticCoefficientSolvePlanError,
) -> CasError {
    CasError::SolverError("Inequalities with symbolic coefficients not yet supported".to_string())
}

pub(crate) fn map_variable_not_found_solver_error(missing_var: &str) -> CasError {
    CasError::VariableNotFound(missing_var.to_string())
}

pub(crate) fn emit_quadratic_formula_scope(solve_ctx: &SolveCtx) {
    solve_ctx.emit_scope(cas_formatter::display_transforms::ScopeTag::Rule(
        "QuadraticFormula",
    ));
}

pub(crate) fn solver_cycle_detected_error() -> CasError {
    CasError::SolverError(
        "Cycle detected: equation revisited after rewriting (equivalent form loop)".to_string(),
    )
}

pub(crate) fn map_no_strategy_solved_error() -> CasError {
    CasError::SolverError("No strategy could solve this equation.".to_string())
}

pub(crate) fn map_isolation_error(var: &str, message: impl AsRef<str>) -> CasError {
    CasError::IsolationError(var.to_string(), message.as_ref().to_string())
}

pub(crate) fn map_unsupported_in_real_domain_error(message: &str) -> CasError {
    CasError::UnsupportedInRealDomain(message.to_string())
}

pub(crate) fn map_unknown_function_error(fn_name: &str) -> CasError {
    CasError::UnknownFunction(fn_name.to_string())
}

pub(crate) fn map_isolation_cannot_isolate_error<T: std::fmt::Debug>(
    var: &str,
    lhs_expr: T,
) -> CasError {
    CasError::IsolationError(var.to_string(), format!("Cannot isolate from {:?}", lhs_expr))
}

pub(crate) fn simplify_rhs_with_step_pairs(
    simplifier: &mut Simplifier,
    rhs_expr: ExprId,
) -> (ExprId, Vec<(String, ExprId)>) {
    let (simplified_rhs, sim_steps) = simplifier.simplify(rhs_expr);
    let entries = sim_steps
        .into_iter()
        .map(|step| (step.description, step.after))
        .collect::<Vec<_>>();
    (simplified_rhs, entries)
}

pub(crate) fn sym_name_as_string(simplifier: &mut Simplifier, fn_symbol: SymbolId) -> String {
    simplifier.context.sym_name(fn_symbol).to_string()
}

pub(crate) fn solve_equation_with_solver_ctx(
    simplifier: &mut Simplifier,
    equation: &Equation,
    solve_var: &str,
    opts: SolverOptions,
    solve_ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    solve_with_ctx_and_options(equation, solve_var, simplifier, opts, solve_ctx)
}

pub(crate) fn isolate_equation_with_solver_ctx(
    simplifier: &mut Simplifier,
    equation: &Equation,
    solve_var: &str,
    opts: SolverOptions,
    solve_ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    crate::solve_isolation_runtime::isolate(
        equation.lhs,
        equation.rhs,
        equation.op.clone(),
        solve_var,
        simplifier,
        opts,
        solve_ctx,
    )
}

pub(crate) fn classify_log_solve_with_solver_ctx(
    ctx: &cas_ast::Context,
    base: ExprId,
    other_side: ExprId,
    opts: &SolverOptions,
    solve_ctx: &SolveCtx,
) -> cas_solver_core::log_domain::LogSolveDecision {
    cas_solver_core::solve_runtime_flow::classify_log_solve_with_domain_env_and_runtime_positive_prover(
        ctx,
        base,
        other_side,
        opts.value_domain,
        opts.core_domain_mode(),
        &solve_ctx.domain_env,
        crate::helpers::prove_positive,
    )
}

pub(crate) fn note_log_assumption_with_solver_ctx(
    ctx: &cas_ast::Context,
    base: ExprId,
    other_side: ExprId,
    assumption: cas_solver_core::log_domain::LogAssumption,
    solve_ctx: &SolveCtx,
) {
    cas_solver_core::solve_runtime_flow::note_log_assumption_with_runtime_sink(
        ctx,
        base,
        other_side,
        assumption,
        |event| solve_ctx.note_assumption(event),
    );
}

pub(crate) fn note_log_blocked_hint_with_default_sink(
    ctx: &cas_ast::Context,
    hint: cas_solver_core::solve_outcome::LogBlockedHintRecord,
) {
    cas_solver_core::solve_runtime_flow::note_log_blocked_hint_with_runtime_sink(
        ctx,
        hint,
        crate::register_blocked_hint,
    );
}
