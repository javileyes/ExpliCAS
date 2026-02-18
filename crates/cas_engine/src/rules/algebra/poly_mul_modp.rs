//! poly_mul_modp: Fast polynomial multiplication in mod-p space.
//!
//! Returns an opaque `poly_ref(id)` instead of materializing AST.
//! Use `poly_stats(poly_ref(id))` to inspect, or `poly_to_expr(poly_ref(id), max_terms)`
//! to materialize up to a limit.

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Expr};
use cas_math::poly_modp_conv::{
    expr_to_poly_modp_with_store as expr_to_poly_modp, PolyModpBudget, VarTable,
};
use cas_math::poly_store::{PolyMeta, POLY_MAX_STORE_TERMS};

/// Default prime for mod-p operations
pub const DEFAULT_PRIME: u64 = 4503599627370449;

// =============================================================================
// poly_mul_modp(a, b [, p]) -> poly_ref(id)
// =============================================================================

define_rule!(
    PolyMulModpRule,
    "Polynomial Multiplication (mod p)",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    PhaseMask::TRANSFORM,
    |ctx, expr| {
        let (name, args) = match ctx.get(expr) {
            Expr::Function(fn_id, a) => (ctx.sym_name(*fn_id).to_string(), a.clone()),
            _ => return None,
        };

        if name != "poly_mul_modp" {
            return None;
        }

        if args.len() < 2 || args.len() > 3 {
            return None;
        }

        let a_expr = args[0];
        let b_expr = args[1];
        let p = if args.len() == 3 {
            match ctx.get(args[2]) {
                Expr::Number(n) => n.to_integer().try_into().ok()?,
                _ => return None,
            }
        } else {
            DEFAULT_PRIME
        };

        // Convert both to MultiPolyModP
        let budget = PolyModpBudget::default();
        let mut vars = VarTable::new();

        let pa = match expr_to_poly_modp(ctx, a_expr, p, &budget, &mut vars) {
            Ok(poly) => poly,
            Err(_) => return None,
        };

        let pb = match expr_to_poly_modp(ctx, b_expr, p, &budget, &mut vars) {
            Ok(poly) => poly,
            Err(_) => return None,
        };

        // Guard: estimate product size and abort if too large
        let est_terms = (pa.num_terms() as u128) * (pb.num_terms() as u128);
        if est_terms > POLY_MAX_STORE_TERMS as u128 {
            tracing::warn!(
                estimated_terms = %est_terms,
                limit = POLY_MAX_STORE_TERMS,
                "poly_mul_modp aborted: estimated {} terms exceeds limit {}",
                est_terms, POLY_MAX_STORE_TERMS
            );
            return None;
        }

        // Multiply in mod-p space (fast)
        let product = pa.mul(&pb);

        // Build metadata
        let meta = PolyMeta {
            modulus: p,
            n_terms: product.num_terms(),
            n_vars: vars.len(),
            max_total_degree: product.total_degree(),
            var_names: vars.names().iter().map(|s| s.to_string()).collect(),
        };

        // NOTE: We cannot insert into poly_store here because rules don't have
        // mutable access to SessionState. This will be handled by the eager evaluator
        // or a separate builtin mechanism.
        //
        // For now, return stats as a function call that can be displayed.
        // NOTE: This uses poly_mul_stats (NOT poly_result) to distinguish from
        // the id-based poly_result(id) format used elsewhere.
        let terms = ctx.num(meta.n_terms as i64);
        let degree = ctx.num(meta.max_total_degree as i64);
        let nvars = ctx.num(meta.n_vars as i64);
        let modulus = ctx.num(meta.modulus as i64);

        let result = ctx.call("poly_mul_stats", vec![terms, degree, nvars, modulus]);

        Some(Rewrite::new(result).desc_lazy(|| {
            format!(
                "poly_mul_modp: {} terms, degree {}, {} vars (mod {})",
                meta.n_terms, meta.max_total_degree, meta.n_vars, meta.modulus
            )
        }))
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
        let (name, args) = match ctx.get(expr) {
            Expr::Function(fn_id, a) => (ctx.sym_name(*fn_id).to_string(), a.clone()),
            _ => return None,
        };

        if name != "poly_stats" || args.len() != 1 {
            return None;
        }

        // Check if arg is poly_result(terms, degree, vars, modulus)
        if let Expr::Function(inner_name, inner_args) = ctx.get(args[0]) {
            if ctx.is_builtin(*inner_name, BuiltinFn::PolyResult) && inner_args.len() == 4 {
                // Already has stats, just format nicely
                return Some(Rewrite::new(args[0]).desc("poly_stats: already computed"));
            }
        }

        None
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
    use cas_ast::Context;
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
