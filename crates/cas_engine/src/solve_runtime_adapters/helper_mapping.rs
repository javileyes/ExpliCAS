use crate::error::CasError;
use crate::solve_runtime::{SolveCtx, SolveStep, SolveSubStep};
use cas_ast::Equation;

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
    CasError::IsolationError(
        var.to_string(),
        format!("Cannot isolate from {:?}", lhs_expr),
    )
}
