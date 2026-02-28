//! poly_mul_modp: Fast polynomial multiplication in mod-p space.
//!
//! Returns an opaque `poly_ref(id)` instead of materializing AST.
//! Use `poly_stats(poly_ref(id))` to inspect, or `poly_to_expr(poly_ref(id), max_terms)`
//! to materialize up to a limit.

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::poly_modp_calls::{
    format_poly_mul_modp_stats_desc, try_eval_poly_mul_modp_stats_call_with_limit_policy,
    try_rewrite_poly_stats_poly_result_arg,
};
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

        // NOTE: We cannot insert into poly_store here because rules don't have
        // mutable access to SessionState. This will be handled by the eager evaluator
        // or a separate builtin mechanism.
        //
        // For now, return stats as a function call that can be displayed.
        // NOTE: This uses poly_mul_stats (NOT poly_result) to distinguish from
        // the id-based poly_result(id) format used elsewhere.
        let result = call.stats_expr;

        Some(
            Rewrite::new(result)
                .desc_lazy(|| format_poly_mul_modp_stats_desc(&call.meta, call.modulus)),
        )
    }
);

// =============================================================================
// poly_stats(poly_ref(id)) -> {terms, degree, vars, modulus}
// =============================================================================

define_rule!(
    PolyStatsRule,
    "Polynomial Statistics",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    PhaseMask::TRANSFORM,
    |ctx, expr| {
        let poly_result_arg = try_rewrite_poly_stats_poly_result_arg(ctx, expr)?;
        Some(Rewrite::new(poly_result_arg).desc("poly_stats: already computed"))
    }
);

/// Register polynomial arithmetic rules
pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(PolyMulModpRule));
    simplifier.add_rule(Box::new(PolyStatsRule));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parent_context::ParentContext;
    use crate::rule::Rule;
    use cas_ast::{Context, Expr};
    use cas_parser::parse;

    #[test]
    fn test_poly_mul_modp_basic() {
        let mut ctx = Context::new();
        let expr = parse("poly_mul_modp((x+1)^2, (x-1)^2)", &mut ctx).unwrap();

        let parent = ParentContext::root();
        let result = PolyMulModpRule.apply(&mut ctx, expr, &parent);

        assert!(result.is_some(), "Rule should fire for poly_mul_modp");
        let rewrite = result.unwrap();

        // Should be poly_mul_stats(terms, degree, vars, modulus)
        if let Expr::Function(fn_id, args) = ctx.get(rewrite.new_expr) {
            let name = ctx.sym_name(*fn_id);
            assert_eq!(name, "poly_mul_stats");
            assert_eq!(args.len(), 4);
        } else {
            panic!("Expected poly_result function");
        }
    }
}
