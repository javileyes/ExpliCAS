//! poly_mul_modp: Fast polynomial multiplication in mod-p space.
//!
//! Returns an opaque `poly_ref(id)` instead of materializing AST.
//! Use `poly_stats(poly_ref(id))` to inspect, or `poly_to_expr(poly_ref(id), max_terms)`
//! to materialize up to a limit.

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::poly_modp_calls::try_eval_poly_mul_modp_stats_call_with_limit_policy;
use cas_math::poly_modp_conv::DEFAULT_PRIME;
use cas_math::poly_store::{PolyMeta, POLY_MAX_STORE_TERMS};

fn format_poly_mul_modp_stats_desc(meta: &PolyMeta, modulus: u64) -> String {
    format!(
        "poly_mul_modp: {} terms, degree {}, {} vars (mod {})",
        meta.n_terms, meta.max_total_degree, meta.n_vars, modulus
    )
}

// =============================================================================
// poly_mul_modp(a, b [, p]) -> poly_ref(id)
// =============================================================================

define_rule!(
    PolyMulModpRule,
    "Polynomial Multiplication (mod p)",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    PhaseMask::TRANSFORM,
    |ctx, expr| {
        let call = try_eval_poly_mul_modp_stats_call_with_limit_policy(
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
        let desc = format_poly_mul_modp_stats_desc(&call.meta, call.modulus);
        Some(Rewrite::new(call.stats_expr).desc(desc))
    }
);

/// Register polynomial arithmetic rules
pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(PolyMulModpRule));
}
