//! Shared runtime mapping helpers for solver adapters.
//!
//! This module centralizes small step-construction and error-mapping helpers
//! used by runtime crates (`cas_engine`, `cas_solver`) so both can share one
//! canonical implementation.

use cas_ast::Equation;

pub type DefaultSolveSubStep =
    crate::solve_aliases::SolveSubStep<Equation, crate::step_types::ImportanceLevel>;
pub type DefaultSolveStep = crate::solve_aliases::SolveStep<
    Equation,
    crate::step_types::ImportanceLevel,
    DefaultSolveSubStep,
>;

pub fn medium_step(description: String, equation_after: Equation) -> DefaultSolveStep {
    DefaultSolveStep::new(
        description,
        equation_after,
        crate::step_types::ImportanceLevel::Medium,
    )
}

pub fn low_substep(description: String, equation_after: Equation) -> DefaultSolveSubStep {
    DefaultSolveSubStep {
        description,
        equation_after,
        importance: crate::step_types::ImportanceLevel::Low,
    }
}

pub fn attach_substeps(
    step: DefaultSolveStep,
    substeps: Vec<DefaultSolveSubStep>,
) -> DefaultSolveStep {
    step.with_substeps(substeps)
}

pub fn map_symbolic_inequalities_not_supported_error(
    _plan_error: crate::quadratic_formula::QuadraticCoefficientSolvePlanError,
) -> crate::error_model::CasError {
    crate::error_model::CasError::SolverError(
        "Inequalities with symbolic coefficients not yet supported".to_string(),
    )
}

pub fn map_variable_not_found_solver_error(missing_var: &str) -> crate::error_model::CasError {
    crate::error_model::CasError::VariableNotFound(missing_var.to_string())
}

pub fn emit_quadratic_formula_scope<DomainEnv, ImplicitCondition, AssumptionEvent>(
    solve_ctx: &crate::solve_aliases::SolveCtx<
        DomainEnv,
        ImplicitCondition,
        AssumptionEvent,
        cas_formatter::display_transforms::ScopeTag,
    >,
) where
    ImplicitCondition: Eq + std::hash::Hash + Clone,
    AssumptionEvent: Clone,
{
    solve_ctx.emit_scope(cas_formatter::display_transforms::ScopeTag::Rule(
        "QuadraticFormula",
    ));
}

pub fn solver_cycle_detected_error() -> crate::error_model::CasError {
    crate::error_model::CasError::SolverError(
        "Cycle detected: equation revisited after rewriting (equivalent form loop)".to_string(),
    )
}

pub fn map_no_strategy_solved_error() -> crate::error_model::CasError {
    crate::error_model::CasError::SolverError("No strategy could solve this equation.".to_string())
}

pub fn map_isolation_error(var: &str, message: impl AsRef<str>) -> crate::error_model::CasError {
    crate::error_model::CasError::IsolationError(var.to_string(), message.as_ref().to_string())
}

pub fn map_unsupported_in_real_domain_error(message: &str) -> crate::error_model::CasError {
    crate::error_model::CasError::UnsupportedInRealDomain(message.to_string())
}

pub fn map_unknown_function_error(fn_name: &str) -> crate::error_model::CasError {
    crate::error_model::CasError::UnknownFunction(fn_name.to_string())
}

pub fn map_isolation_cannot_isolate_error<T: std::fmt::Debug>(
    var: &str,
    lhs_expr: T,
) -> crate::error_model::CasError {
    crate::error_model::CasError::IsolationError(
        var.to_string(),
        format!("Cannot isolate from {:?}", lhs_expr),
    )
}
