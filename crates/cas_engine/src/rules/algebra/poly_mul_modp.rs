//! poly_mul_modp: Fast polynomial multiplication in mod-p space.
//!
//! Returns an opaque `poly_ref(id)` instead of materializing AST.
//! Use `poly_stats(poly_ref(id))` to inspect, or `poly_to_expr(poly_ref(id), max_terms)`
//! to materialize up to a limit.

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::poly_modp_calls::rewrite_poly_mul_modp_stats_call_with_defaults;

// =============================================================================
// poly_mul_modp(a, b [, p]) -> poly_ref(id)
// =============================================================================

define_rule!(
    PolyMulModpRule,
    "Polynomial Multiplication (mod p)",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    PhaseMask::TRANSFORM,
    |ctx, expr| {
        let rewritten =
            rewrite_poly_mul_modp_stats_call_with_defaults(ctx, expr, |estimated_terms, limit| {
                tracing::warn!(
                    estimated_terms = %estimated_terms,
                    limit = limit,
                    "poly_mul_modp aborted: estimated {} terms exceeds limit {}",
                    estimated_terms,
                    limit
                );
            })?;
        Some(Rewrite::new(rewritten.0).desc(rewritten.1))
    }
);

/// Register polynomial arithmetic rules
pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(PolyMulModpRule));
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
