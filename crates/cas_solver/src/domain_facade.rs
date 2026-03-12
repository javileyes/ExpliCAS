//! Domain-inference facade for solver consumers.
//!
//! This module keeps domain helpers grouped so `lib.rs` stays focused on
//! high-level API exports during migration.

/// Infer implicit domain constraints from expression structure.
pub fn infer_implicit_domain(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
    vd: crate::ValueDomain,
) -> crate::ImplicitDomain {
    cas_solver_core::domain_inference_counter::inc();
    cas_solver_core::domain_inference::infer_implicit_domain(
        ctx,
        root,
        vd == crate::ValueDomain::RealOnly,
    )
}

/// Derive additional required conditions from equation equality.
pub fn derive_requires_from_equation(
    ctx: &cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
    existing: &crate::ImplicitDomain,
    vd: crate::ValueDomain,
) -> Vec<crate::ImplicitCondition> {
    cas_solver_core::domain_inference::derive_requires_from_equation(
        ctx,
        lhs,
        rhs,
        existing,
        vd == crate::ValueDomain::RealOnly,
        |ctx, expr| crate::prove_positive(ctx, expr, vd),
    )
}

/// Check if a rewrite would expand the domain by removing implicit constraints.
pub fn domain_delta_check(
    ctx: &cas_ast::Context,
    input: cas_ast::ExprId,
    output: cas_ast::ExprId,
    vd: crate::ValueDomain,
) -> crate::DomainDelta {
    cas_solver_core::domain_inference::domain_delta_check(ctx, input, output, |ctx, expr| {
        infer_implicit_domain(ctx, expr, vd)
    })
}

/// Convert solver path steps to a compact AST expression path.
#[allow(dead_code)]
pub fn pathsteps_to_expr_path(steps: &[crate::PathStep]) -> cas_ast::ExprPath {
    steps.iter().map(crate::PathStep::to_child_index).collect()
}
