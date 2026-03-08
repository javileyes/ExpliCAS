//! Domain delta checks, implicit domain inference, and related helpers.

use super::{ImplicitCondition, ImplicitDomain};
use crate::semantics::ValueDomain;
use cas_ast::{Context, ExprId};

pub use cas_solver_core::domain_inference::{AnalyticExpansionResult, DomainDelta};

/// Check if a rewrite would expand the domain by removing implicit constraints.
pub fn domain_delta_check(
    ctx: &Context,
    input: ExprId,
    output: ExprId,
    vd: ValueDomain,
) -> DomainDelta {
    cas_solver_core::domain_inference::domain_delta_check(ctx, input, output, |ctx, expr| {
        infer_implicit_domain(ctx, expr, vd)
    })
}

/// Quick check: does this rewrite expand analytic domain?
pub fn expands_analytic_domain(
    ctx: &Context,
    input: ExprId,
    output: ExprId,
    vd: ValueDomain,
) -> bool {
    cas_solver_core::domain_inference::expands_analytic_domain(ctx, input, output, |ctx, expr| {
        infer_implicit_domain(ctx, expr, vd)
    })
}

/// Context-aware check: does this rewrite expand analytic domain considering the full tree?
pub fn check_analytic_expansion(
    ctx: &Context,
    root: ExprId,
    rewritten_node: ExprId,
    replacement: ExprId,
    vd: ValueDomain,
) -> AnalyticExpansionResult {
    cas_solver_core::domain_inference::check_analytic_expansion(
        ctx,
        root,
        rewritten_node,
        replacement,
        |ctx, expr| infer_implicit_domain(ctx, expr, vd),
    )
}

/// Quick check: does this rewrite expand analytic domain?
pub fn expands_analytic_in_context(
    ctx: &Context,
    root: ExprId,
    rewritten_node: ExprId,
    replacement: ExprId,
    vd: ValueDomain,
) -> bool {
    cas_solver_core::domain_inference::expands_analytic_in_context(
        ctx,
        root,
        rewritten_node,
        replacement,
        |ctx, expr| infer_implicit_domain(ctx, expr, vd),
    )
}

/// Infer implicit domain constraints from expression structure.
pub fn infer_implicit_domain(ctx: &Context, root: ExprId, vd: ValueDomain) -> ImplicitDomain {
    // Track call count for regression testing
    cas_solver_core::domain_inference_counter::inc();

    cas_solver_core::domain_inference::infer_implicit_domain(ctx, root, vd == ValueDomain::RealOnly)
}

/// Derive additional required conditions from equation equality.
pub fn derive_requires_from_equation(
    ctx: &Context,
    lhs: ExprId,
    rhs: ExprId,
    existing: &ImplicitDomain,
    vd: ValueDomain,
) -> Vec<ImplicitCondition> {
    cas_solver_core::domain_inference::derive_requires_from_equation(
        ctx,
        lhs,
        rhs,
        existing,
        vd == ValueDomain::RealOnly,
        |ctx, expr| crate::helpers::prove_positive(ctx, expr, vd),
    )
}
