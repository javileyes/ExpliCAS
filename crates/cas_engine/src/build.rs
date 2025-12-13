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

/// Raw multiplication of multiple factors. Right-folds without simplification.
///
/// `mul_many_raw(ctx, [a, b, c])` → `a * (b * c)` (right-associative)
///
/// Returns `None` if factors is empty.
pub fn mul_many_raw(ctx: &mut Context, factors: &[ExprId]) -> Option<ExprId> {
    if factors.is_empty() {
        return None;
    }
    let mut result = *factors.last().unwrap();
    for &factor in factors.iter().rev().skip(1) {
        result = ctx.add_raw(Expr::Mul(factor, result));
    }
    Some(result)
}

/// Raw 2-factor addition. Preserves operand order using `add_raw`.
#[inline]
pub fn add2_raw(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    ctx.add_raw(Expr::Add(a, b))
}
