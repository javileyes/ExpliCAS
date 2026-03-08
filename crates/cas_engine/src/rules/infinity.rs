//! Infinity arithmetic rules for the extended real line ℝ ∪ {+∞, −∞}.
//!
//! Implements safe, conservative rules for infinity operations.
//! Only collapses when all non-infinity terms are "finite literal" (numbers, known constants).
//!
//! # Covered operations
//! - `finite + ∞ → ∞` (absorption)
//! - `finite / ∞ → 0`
//! - `∞ + (-∞) → Undefined` (indeterminate)
//! - `0 · ∞ → Undefined` (indeterminate)

use crate::rule::Rewrite;
use cas_ast::{Context, ExprId};
use cas_math::infinity_support::{
    try_rewrite_add_infinity_absorption_expr, try_rewrite_div_by_infinity_expr,
    try_rewrite_inf_div_finite_expr, try_rewrite_mul_finite_infinity_expr,
    try_rewrite_mul_zero_infinity_expr,
};

// ============================================================
// RULES
// ============================================================

/// Rule: Infinity absorption in addition.
///
/// - `finite + ∞ → ∞`
/// - `finite + (-∞) → -∞`
/// - `∞ + (-∞) → Undefined` (indeterminate)
///
/// Only applies when ALL non-infinity terms are finite literals (conservative).
pub fn add_infinity_absorption(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let plan = try_rewrite_add_infinity_absorption_expr(ctx, expr)?;
    Some(Rewrite::new(plan.rewritten).desc(plan.description))
}

/// Rule: Division by infinity.
///
/// `finite / ∞ → 0`
/// `finite / (-∞) → 0`
///
/// Only applies when numerator is a finite literal.
pub fn div_by_infinity(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let plan = try_rewrite_div_by_infinity_expr(ctx, expr)?;
    Some(Rewrite::new(plan.rewritten).desc(plan.description))
}

/// Rule: Zero times infinity is indeterminate.
///
/// `0 · ∞ → Undefined`
/// `∞ · 0 → Undefined`
pub fn mul_zero_infinity(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let plan = try_rewrite_mul_zero_infinity_expr(ctx, expr)?;
    Some(Rewrite::new(plan.rewritten).desc(plan.description))
}

/// Rule: Finite (non-zero) times infinity.
///
/// `finite * ∞ → ±∞` (sign depends on finite's sign)
/// - `3 * infinity → infinity`
/// - `(-2) * infinity → -infinity`
/// - `x * infinity → no simplification` (conservative)
pub fn mul_finite_infinity(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let plan = try_rewrite_mul_finite_infinity_expr(ctx, expr)?;
    Some(Rewrite::new(plan.rewritten).desc(plan.description))
}

/// Rule: Infinity divided by finite (non-zero).
///
/// `∞ / finite → ±∞` (sign depends on finite's sign)
/// - `infinity / 2 → infinity`
/// - `infinity / (-3) → -infinity`
/// - `-infinity / 2 → -infinity`
pub fn inf_div_finite(ctx: &mut Context, expr: ExprId) -> Option<Rewrite> {
    let plan = try_rewrite_inf_div_finite_expr(ctx, expr)?;
    Some(Rewrite::new(plan.rewritten).desc(plan.description))
}

// ============================================================
// RULE STRUCTS (for pipeline registration)
// ============================================================

use crate::define_rule;

define_rule!(
    AddInfinityRule,
    "Infinity Absorption in Addition",
    Some(crate::target_kind::TargetKindSet::ADD),
    |ctx, expr| { add_infinity_absorption(ctx, expr) }
);

define_rule!(
    DivByInfinityRule,
    "Division by Infinity",
    Some(crate::target_kind::TargetKindSet::DIV),
    |ctx, expr| { div_by_infinity(ctx, expr) }
);

define_rule!(
    MulZeroInfinityRule,
    "Zero Times Infinity Indeterminate",
    Some(crate::target_kind::TargetKindSet::MUL),
    |ctx, expr| { mul_zero_infinity(ctx, expr) }
);

define_rule!(
    MulInfinityRule,
    "Finite Times Infinity",
    Some(crate::target_kind::TargetKindSet::MUL),
    |ctx, expr| { mul_finite_infinity(ctx, expr) }
);

define_rule!(
    InfDivFiniteRule,
    "Infinity Divided by Finite",
    Some(crate::target_kind::TargetKindSet::DIV),
    |ctx, expr| { inf_div_finite(ctx, expr) }
);

/// Register infinity arithmetic rules with the simplifier.
///
/// These rules should be registered early in the pipeline (with CORE rules)
/// to handle infinity operations before other simplifications.
pub fn register(simplifier: &mut crate::Simplifier) {
    // Indeterminate forms first (highest priority)
    simplifier.add_rule(Box::new(MulZeroInfinityRule));
    // Then absorption/computation rules
    simplifier.add_rule(Box::new(MulInfinityRule));
    simplifier.add_rule(Box::new(AddInfinityRule));
    simplifier.add_rule(Box::new(DivByInfinityRule));
    simplifier.add_rule(Box::new(InfDivFiniteRule));
}
