//! Shared operations for lowering expressions over `poly_result(...)` references.
//!
//! This module centralizes binary/unary polynomial reference combination logic so
//! rule-orchestration layers can stay thin.

use crate::poly_result::{parse_poly_result_id, wrap_poly_result};
use crate::poly_store::{
    thread_local_add, thread_local_mul, thread_local_neg, thread_local_pow,
    thread_local_promote_expr_with_base, thread_local_sub, PolyId,
};
use cas_ast::{Context, ExprId};

/// Binary polynomial operation in the thread-local PolyStore.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolyBinaryOp {
    Add,
    Sub,
    Mul,
}

/// How a binary combination was achieved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolyCombineKind {
    /// Both operands were already `poly_result(...)`.
    Direct,
    /// One side required auto-promotion before combining.
    Promoted,
}

/// Result of combining two operands in poly space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolyCombineResult {
    pub expr: ExprId,
    pub kind: PolyCombineKind,
}

fn apply_binary(op: PolyBinaryOp, left: PolyId, right: PolyId) -> Option<PolyId> {
    match op {
        PolyBinaryOp::Add => thread_local_add(left, right),
        PolyBinaryOp::Sub => thread_local_sub(left, right),
        PolyBinaryOp::Mul => thread_local_mul(left, right),
    }
}

/// Try combining two operands in poly space.
///
/// Supports:
/// - direct `poly_result` + `poly_result` combinations
/// - auto-promotion of one non-poly side using the other side as base context
pub fn try_combine_binary_poly_with_promotion(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    op: PolyBinaryOp,
    promote_max_nodes: usize,
    promote_max_terms: usize,
) -> Option<PolyCombineResult> {
    let id_l = parse_poly_result_id(ctx, left);
    let id_r = parse_poly_result_id(ctx, right);

    match (id_l, id_r) {
        (Some(id_l), Some(id_r)) => {
            let new_id = apply_binary(op, id_l, id_r)?;
            Some(PolyCombineResult {
                expr: wrap_poly_result(ctx, new_id),
                kind: PolyCombineKind::Direct,
            })
        }
        (Some(id_l), None) => {
            let id_r_promoted = thread_local_promote_expr_with_base(
                ctx,
                right,
                id_l,
                promote_max_nodes,
                promote_max_terms,
            )?;
            let new_id = apply_binary(op, id_l, id_r_promoted)?;
            Some(PolyCombineResult {
                expr: wrap_poly_result(ctx, new_id),
                kind: PolyCombineKind::Promoted,
            })
        }
        (None, Some(id_r)) => {
            let id_l_promoted = thread_local_promote_expr_with_base(
                ctx,
                left,
                id_r,
                promote_max_nodes,
                promote_max_terms,
            )?;
            let new_id = apply_binary(op, id_l_promoted, id_r)?;
            Some(PolyCombineResult {
                expr: wrap_poly_result(ctx, new_id),
                kind: PolyCombineKind::Promoted,
            })
        }
        (None, None) => None,
    }
}

/// Try negating a `poly_result(...)` expression.
pub fn try_negate_poly_ref(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let id = parse_poly_result_id(ctx, expr)?;
    let new_id = thread_local_neg(id)?;
    Some(wrap_poly_result(ctx, new_id))
}

/// Try computing `(poly_result(...))^exp` for non-negative integer exponent.
pub fn try_pow_poly_ref(ctx: &mut Context, base: ExprId, exp: u32) -> Option<ExprId> {
    let id = parse_poly_result_id(ctx, base)?;
    let new_id = thread_local_pow(id, exp)?;
    Some(wrap_poly_result(ctx, new_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multipoly_modp::MultiPolyModP;
    use crate::poly_modp_conv::DEFAULT_PRIME;
    use crate::poly_store::{
        clear_thread_local_store, thread_local_insert, with_thread_local_store, PolyMeta,
    };
    use cas_parser::parse;

    fn insert_x_plus_const(constant: u64) -> PolyId {
        let x = MultiPolyModP::var(0, DEFAULT_PRIME, 1);
        let c = MultiPolyModP::constant(constant, DEFAULT_PRIME, 1);
        let poly = x.add(&c);
        let meta = PolyMeta {
            modulus: DEFAULT_PRIME,
            n_terms: poly.num_terms(),
            n_vars: 1,
            max_total_degree: 1,
            var_names: vec!["x".to_string()],
        };
        thread_local_insert(meta, poly)
    }

    #[test]
    fn combine_direct_poly_results_add() {
        let mut ctx = Context::new();
        clear_thread_local_store();
        let id_a = insert_x_plus_const(1);
        let id_b = insert_x_plus_const(2);
        let pr_a = wrap_poly_result(&mut ctx, id_a);
        let pr_b = wrap_poly_result(&mut ctx, id_b);

        let combined = try_combine_binary_poly_with_promotion(
            &mut ctx,
            pr_a,
            pr_b,
            PolyBinaryOp::Add,
            200,
            10_000,
        )
        .expect("combine");
        assert_eq!(combined.kind, PolyCombineKind::Direct);
        assert!(parse_poly_result_id(&ctx, combined.expr).is_some());
    }

    #[test]
    fn combine_promotes_rhs_when_needed() {
        let mut ctx = Context::new();
        clear_thread_local_store();
        let b = parse("2", &mut ctx).expect("parse b");
        let id_a = insert_x_plus_const(1);
        let pr_a = wrap_poly_result(&mut ctx, id_a);

        let combined = try_combine_binary_poly_with_promotion(
            &mut ctx,
            pr_a,
            b,
            PolyBinaryOp::Sub,
            200,
            10_000,
        )
        .expect("combine with promotion");

        assert_eq!(combined.kind, PolyCombineKind::Promoted);
        let new_id = parse_poly_result_id(&ctx, combined.expr).expect("poly_result id");

        // Ensure resulting id is materialized in thread-local store.
        with_thread_local_store(|store| {
            assert!(store.get(new_id).is_some());
        });
    }
}
