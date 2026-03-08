//! Telescoping strategy facade for engine compatibility.

use cas_ast::{Context, ExprId};

pub use cas_solver_core::telescoping_runtime::{TelescopingResult, TelescopingStep};

/// Attempt to prove an identity using telescoping strategy.
pub fn telescope(ctx: &mut Context, expr: ExprId) -> TelescopingResult {
    cas_solver_core::telescoping_runtime::telescope_with_runtime_simplify(
        ctx,
        expr,
        |source_ctx, source_expr| {
            let mut simplifier = crate::Simplifier::with_default_rules();
            simplifier.context = source_ctx.clone();
            let (result, _) = simplifier.simplify(source_expr);
            *source_ctx = simplifier.context;
            result
        },
    )
}
