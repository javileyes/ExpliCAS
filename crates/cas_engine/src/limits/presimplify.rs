//! Compatibility shim for limit pre-simplification.
//!
//! The allowlist-only pre-simplification core now lives in
//! `cas_math::limits_support::presimplify_safe_for_limit`.

use crate::{Budget, CasError};
use cas_ast::{Context, ExprId};

/// Apply safe pre-simplification to an expression before limit evaluation.
///
/// This wrapper keeps the engine-facing signature while delegating the
/// transformation logic to `cas_math`.
pub fn presimplify_safe(
    ctx: &mut Context,
    expr: ExprId,
    _budget: &mut Budget,
) -> Result<ExprId, CasError> {
    Ok(cas_math::limits_support::presimplify_safe_for_limit(
        ctx, expr,
    ))
}
