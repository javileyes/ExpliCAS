//! Generalized n-angle inverse-trig composition rules.
//!
//! These rules replace the hardcoded n=1,2,3 rules in `trig_inverse_compositions.rs`
//! with recurrence-based generators supporting n=1..MAX_N (10) for sin/cos/tan of
//! `n·arctan(t)`, and sin/cos of `n·arccos(t)` and `n·arcsin(t)`.
//!
//! # Architecture
//!
//! ```text
//! trig(n·invtrig(t))
//!   │
//!   ├── extract outer trig function (sin/cos/tan)
//!   ├── extract_int_multiple(arg, 1..MAX_N) → (sign, n, inner)
//!   ├── match inner = arctan/arccos/arcsin(t)
//!   ├── guard: count_nodes(t) ≤ MAX_INNER_NODES
//!   ├── generate via recurrence (no intermediate simplification)
//!   ├── guard: count_nodes(result) ≤ MAX_OUTPUT_NODES
//!   └── return Rewrite with appropriate SolveSafety/assumptions
//! ```
//!
//! # Recurrences
//!
//! ## Arctan (Weierstrass)
//!
//! From `(1+it)^n = Aₙ(t) + i·Bₙ(t)`:
//! - `sin(n·arctan(t)) = Bₙ(t) / (1+t²)^(n/2)`
//! - `cos(n·arctan(t)) = Aₙ(t) / (1+t²)^(n/2)`
//! - `tan(n·arctan(t)) = Bₙ(t) / Aₙ(t)`
//!
//! ## Arccos (Chebyshev)
//!
//! - `cos(n·arccos(t)) = Tₙ(t)`  (Chebyshev first kind)
//! - `sin(n·arccos(t)) = √(1-t²)·Uₙ₋₁(t)` (Chebyshev second kind)
//!
//! ## Arcsin
//!
//! - `sin(n·arcsin(t)) = Sₙ` via `S_{k+1} = Sₖ·√(1-t²) + Cₖ·t`
//! - `cos(n·arcsin(t)) = Cₙ` via `C_{k+1} = Cₖ·√(1-t²) - Sₖ·t`

use cas_math::inv_trig_n_angle_support::{
    try_plan_n_angle_acos_expr, try_plan_n_angle_asin_expr, try_plan_n_angle_atan_expr,
};

use crate::define_rule;
use crate::rule::Rewrite;
use crate::target_kind::TargetKindSet;

/// Maximum n for n·invtrig(t) expansion.
const MAX_N: i64 = 10;

/// Maximum AST node count for the inner argument `t`.
const MAX_INNER_NODES: usize = 20;

/// Maximum AST node count for the generated output expression.
/// Prevents recurrence from producing unreasonably large ASTs.
const MAX_OUTPUT_NODES: usize = 300;

// =============================================================================
// Arctan: Weierstrass recurrence  (sin/cos/tan)
// =============================================================================
//
// (1+it)^n = Aₙ + i·Bₙ
// Recurrence: A₀=1, B₀=0, A_{k+1} = Aₖ - t·Bₖ, B_{k+1} = Bₖ + t·Aₖ
//
// sin(n·atan(t)) = Bₙ / (1+t²)^(n/2)
// cos(n·atan(t)) = Aₙ / (1+t²)^(n/2)
// tan(n·atan(t)) = Bₙ / Aₙ

define_rule!(
    NAngleAtanRule,
    "N-Angle Inverse Atan Composition",
    Some(TargetKindSet::FUNCTION),
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Definability
    ),
    |ctx, expr, _parent_ctx| {
        let plan =
            try_plan_n_angle_atan_expr(ctx, expr, MAX_N, MAX_INNER_NODES, MAX_OUTPUT_NODES)?;
        Some(
            Rewrite::new(plan.rewritten)
                .desc(plan.desc)
                .budget_exempt()
                .assume(crate::AssumptionEvent::nonzero(
                    ctx,
                    plan.assume_nonzero_expr,
                )),
        )
    }
);

// =============================================================================
// Arccos: Chebyshev recurrence  (sin/cos/tan)
// =============================================================================
//
// cos(n·arccos(t)) = Tₙ(t) — Chebyshev polynomial of the first kind
// sin(n·arccos(t)) = √(1-t²)·Uₙ₋₁(t) — sin factor × Chebyshev second kind
// tan(n·arccos(t)) = √(1-t²)·Uₙ₋₁(t) / Tₙ(t) — ratio (NonZero(Tₙ))
//
// T₀=1, T₁=t, T_{k+1} = 2t·Tₖ - T_{k-1}
// U₀=1, U₁=2t, U_{k+1} = 2t·Uₖ - U_{k-1}

define_rule!(
    NAngleAcosRule,
    "N-Angle Inverse Acos Composition",
    Some(TargetKindSet::FUNCTION),
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Definability
    ),
    |ctx, expr, _parent_ctx| {
        let plan =
            try_plan_n_angle_acos_expr(ctx, expr, MAX_N, MAX_INNER_NODES, MAX_OUTPUT_NODES)?;
        Some(
            Rewrite::new(plan.rewritten)
                .desc(plan.desc)
                .budget_exempt()
                .assume(crate::AssumptionEvent::nonzero(
                    ctx,
                    plan.assume_nonzero_expr,
                )),
        )
    }
);

// =============================================================================
// Arcsin: sin/cos recurrence  (sin/cos/tan)
// =============================================================================
//
// With θ=arcsin(t), sinθ=t, cosθ=√(1-t²):
// S₀=0,  C₀=1
// S₁=t,  C₁=√(1-t²)
// S_{k+1} = Sₖ·cosθ + Cₖ·sinθ = Sₖ·√(1-t²) + Cₖ·t
// C_{k+1} = Cₖ·cosθ - Sₖ·sinθ = Cₖ·√(1-t²) - Sₖ·t
//
// sin(n·arcsin(t)) = Sₙ
// cos(n·arcsin(t)) = Cₙ
// tan(n·arcsin(t)) = Sₙ / Cₙ  (NonZero(Cₙ))

define_rule!(
    NAngleAsinRule,
    "N-Angle Inverse Asin Composition",
    Some(TargetKindSet::FUNCTION),
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Definability
    ),
    |ctx, expr, _parent_ctx| {
        let plan =
            try_plan_n_angle_asin_expr(ctx, expr, MAX_N, MAX_INNER_NODES, MAX_OUTPUT_NODES)?;
        Some(
            Rewrite::new(plan.rewritten)
                .desc(plan.desc)
                .budget_exempt()
                .assume(crate::AssumptionEvent::nonzero(
                    ctx,
                    plan.assume_nonzero_expr,
                )),
        )
    }
);

// =============================================================================
// Registration
// =============================================================================

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(NAngleAtanRule));
    simplifier.add_rule(Box::new(NAngleAcosRule));
    simplifier.add_rule(Box::new(NAngleAsinRule));
}
