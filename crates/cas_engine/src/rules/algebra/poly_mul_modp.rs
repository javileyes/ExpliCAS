//! poly_mul_modp: Fast polynomial multiplication in mod-p space.
//!
//! Returns an opaque `poly_ref(id)` instead of materializing AST.
//! Use `poly_stats(poly_ref(id))` to inspect, or `poly_to_expr(poly_ref(id), max_terms)`
//! to materialize up to a limit.

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::poly_modp_calls::rewrite_poly_mul_modp_stats_call_with_limit_policy;
use cas_math::poly_modp_conv::DEFAULT_PRIME;
use cas_math::poly_store::POLY_MAX_STORE_TERMS;

// =============================================================================
// poly_mul_modp(a, b [, p]) -> poly_ref(id)
// =============================================================================

define_rule!(
    PolyMulModpRule,
    "Polynomial Multiplication (mod p)",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    PhaseMask::TRANSFORM,
    |ctx, expr| {
        let rewritten = rewrite_poly_mul_modp_stats_call_with_limit_policy(
            ctx,
            expr,
            DEFAULT_PRIME,
            POLY_MAX_STORE_TERMS,
            |estimated_terms, limit| {
                tracing::warn!(
                    estimated_terms = %estimated_terms,
                    limit = limit,
                    "poly_mul_modp aborted: estimated {} terms exceeds limit {}",
                    estimated_terms,
                    limit
                );
            },
        )?;
        Some(Rewrite::new(rewritten.0).desc(rewritten.1))
    }
);

/// Register polynomial arithmetic rules
pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(PolyMulModpRule));
}
