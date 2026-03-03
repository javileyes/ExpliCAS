//! Solver API facade.
//!
//! Keeps `cas_solver` public API stable while centralizing solver entry points
//! in this crate.

use cas_ast::{Equation, ExprId, SolutionSet};
use cas_math::tri_proof::TriProof;
use cas_solver_core::external_proof::map_external_nonzero_status_with;
pub use cas_solver_core::isolation_utils::contains_var;
pub use cas_solver_core::verify_stats;

use crate::Simplifier;

pub use crate::types::{
    DisplaySolveSteps, SolveCtx, SolveDiagnostics, SolveDomainEnv, SolveStep, SolveSubStep,
    SolverOptions,
};

/// Infer the variable to solve for when user doesn't specify one explicitly.
///
/// Returns:
/// - `Ok(Some(var))` if exactly one free variable found
/// - `Ok(None)` if no variables found
/// - `Err(vars)` if multiple variables found (ambiguous)
///
/// Filters out known constants (`pi`, `π`, `e`, `i`) and internal symbols
/// (`_*`, `#*`).
pub fn infer_solve_variable(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> Result<Option<String>, Vec<String>> {
    let all_vars = cas_ast::collect_variables(ctx, expr);

    let free_vars: Vec<String> = all_vars
        .into_iter()
        .filter(|v| {
            let is_constant = matches!(v.as_str(), "pi" | "π" | "e" | "i");
            let is_internal = v.starts_with('_') || v.starts_with('#');
            !is_constant && !is_internal
        })
        .collect();

    match free_vars.len() {
        0 => Ok(None),
        1 => Ok(free_vars.into_iter().next()),
        _ => {
            let mut sorted = free_vars;
            sorted.sort();
            Err(sorted)
        }
    }
}

/// Solve an equation for a variable.
pub fn solve(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
) -> Result<(SolutionSet, Vec<SolveStep>), crate::CasError> {
    crate::solve_core::solve(eq, var, simplifier)
}

/// Solve with display-ready steps and diagnostics.
pub fn solve_with_display_steps(
    eq: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> Result<(SolutionSet, DisplaySolveSteps, SolveDiagnostics), crate::CasError> {
    crate::solve_core::solve_with_display_steps(eq, var, simplifier, opts)
}

pub(crate) fn simplifier_context(simplifier: &mut Simplifier) -> &cas_ast::Context {
    &simplifier.context
}

pub(crate) fn simplifier_context_mut(simplifier: &mut Simplifier) -> &mut cas_ast::Context {
    &mut simplifier.context
}

pub(crate) fn simplifier_contains_var(
    simplifier: &mut Simplifier,
    expr: ExprId,
    var: &str,
) -> bool {
    contains_var(&simplifier.context, expr, var)
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

pub(crate) fn prove_positive_core(
    ctx: &cas_ast::Context,
    expr: ExprId,
    value_domain: crate::ValueDomain,
) -> TriProof {
    match crate::prove_positive(ctx, expr, value_domain) {
        crate::Proof::Proven | crate::Proof::ProvenImplicit => TriProof::Proven,
        crate::Proof::Disproven => TriProof::Disproven,
        crate::Proof::Unknown => TriProof::Unknown,
    }
}

pub(crate) fn simplifier_is_known_negative(simplifier: &mut Simplifier, expr: ExprId) -> bool {
    cas_solver_core::isolation_utils::is_known_negative(&simplifier.context, expr)
}

pub(crate) fn medium_step(description: String, equation_after: Equation) -> SolveStep {
    SolveStep::new(description, equation_after, crate::ImportanceLevel::Medium)
}
