//! Build helpers for expression construction.
//!
//! This module provides unified builders for constructing expressions
//! with consistent behavior regarding order preservation and simplification.

use cas_ast::{Context, Expr, ExprId};

/// Raw 2-factor multiplication. Preserves operand order using `add_raw`.
///
/// Use this inside recursive rules where you want exact structure control.
/// For simplified output (1*x → x), use `mul2_simpl` from `crate::rules::algebra`.
#[inline]
pub fn mul2_raw(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    ctx.add_raw(Expr::Mul(a, b))
}
