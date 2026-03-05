//! Compatibility facade for cycle detection.
//!
//! The implementation lives in `cas_solver_core::cycle_detection`.
#![allow(dead_code)]

pub type FingerprintMemo = cas_solver_core::cycle_detection::FingerprintMemo;
pub type CycleDetector = cas_solver_core::cycle_detection::CycleDetector;
pub type CycleInfo = cas_solver_core::cycle_models::CycleInfo;

#[inline]
pub fn expr_fingerprint(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
    memo: &mut FingerprintMemo,
) -> u64 {
    cas_solver_core::cycle_detection::expr_fingerprint(ctx, root, memo)
}
