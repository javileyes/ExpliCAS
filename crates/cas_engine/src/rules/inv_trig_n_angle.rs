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

use cas_ast::{BuiltinFn, Expr};
use cas_math::inv_trig_n_angle_support::{
    arcsin_recurrence, build_one_minus_t_sq, build_one_plus_t_sq, build_sqrt, chebyshev_t,
    chebyshev_u_nm1, count_nodes_dedup, weierstrass_recurrence,
};
use cas_math::trig_roots_flatten::extract_int_multiple;
use num_bigint::BigInt;
use num_rational::BigRational;

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
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, _parent_ctx| {
        // Match sin|cos|tan(arg0)
        let (fn_id, arg0) = match ctx.get(expr) {
            Expr::Function(fn_id, args) if args.len() == 1 => (*fn_id, args[0]),
            _ => return None,
        };

        let trig = match ctx.builtin_of(fn_id) {
            Some(b @ (BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan)) => b,
            _ => return None,
        };

        // Try n=1 (no multiplier) and n=2..MAX_N via extract_int_multiple
        let (is_positive, n, inner) = if let Expr::Function(_, _) = ctx.get(arg0) {
            // Could be n=1: trig(arctan(t))
            (true, 1i64, arg0)
        } else {
            // Try all multiples 2..=MAX_N
            let mut found = None;
            for k in 2..=MAX_N {
                if let Some((sign, inner)) = extract_int_multiple(ctx, arg0, k) {
                    found = Some((sign, k, inner));
                    break;
                }
            }
            found?
        };

        // Match inner = arctan(t)
        let t = match ctx.get(inner) {
            Expr::Function(inv_id, inv_args) if inv_args.len() == 1 => {
                match ctx.builtin_of(*inv_id) {
                    Some(BuiltinFn::Atan | BuiltinFn::Arctan) => inv_args[0],
                    _ => return None,
                }
            }
            _ => return None,
        };

        // Guard: inner arg size
        if count_nodes_dedup(ctx, t) > MAX_INNER_NODES {
            return None;
        }

        let n_usize = n as usize;

        // Build Weierstrass polynomials
        let (a_n, b_n) = weierstrass_recurrence(ctx, t, n_usize);

        // Build result based on trig function
        let (result, den_for_assumption, desc) = match trig {
            BuiltinFn::Tan => {
                // tan(n·atan(t)) = Bₙ / Aₙ
                let result = ctx.add(Expr::Div(b_n, a_n));
                let desc = format!("tan({n}·atan(t)) = Bₙ/Aₙ");
                (result, a_n, desc)
            }
            BuiltinFn::Sin | BuiltinFn::Cos => {
                // denominator = (1+t²)^(n/2)
                let one_plus_t_sq = build_one_plus_t_sq(ctx, t);
                let exp = ctx.add(Expr::Number(BigRational::new(
                    BigInt::from(n),
                    BigInt::from(2),
                )));
                let denom = ctx.add(Expr::Pow(one_plus_t_sq, exp));

                let (numerator, desc) = match trig {
                    BuiltinFn::Sin => (b_n, format!("sin({n}·atan(t)) = Bₙ/(1+t²)^({n}/2)")),
                    _ => (a_n, format!("cos({n}·atan(t)) = Aₙ/(1+t²)^({n}/2)")),
                };
                let result = ctx.add(Expr::Div(numerator, denom));
                (result, denom, desc)
            }
            _ => return None,
        };

        // Apply sign parity: sin(-nθ) = -sin(nθ), cos(-nθ) = cos(nθ), tan(-nθ) = -tan(nθ)
        let result = if !is_positive {
            match trig {
                BuiltinFn::Cos => result, // cos is even
                _ => ctx.add(Expr::Neg(result)), // sin, tan are odd
            }
        } else {
            result
        };

        // Guard: output size
        if count_nodes_dedup(ctx, result) > MAX_OUTPUT_NODES {
            return None;
        }

        Some(
            Rewrite::new(result)
                .desc(desc)
                .budget_exempt()
                .assume(crate::assumptions::AssumptionEvent::nonzero(ctx, den_for_assumption)),
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
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, _parent_ctx| {
        // Match sin|cos|tan(arg0)
        let (fn_id, arg0) = match ctx.get(expr) {
            Expr::Function(fn_id, args) if args.len() == 1 => (*fn_id, args[0]),
            _ => return None,
        };

        let trig = match ctx.builtin_of(fn_id) {
            Some(b @ (BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan)) => b,
            _ => return None,
        };

        // Try n=1 (no multiplier) and n=2..MAX_N
        let (is_positive, n, inner) = if let Expr::Function(_, _) = ctx.get(arg0) {
            (true, 1i64, arg0)
        } else {
            let mut found = None;
            for k in 2..=MAX_N {
                if let Some((sign, inner)) = extract_int_multiple(ctx, arg0, k) {
                    found = Some((sign, k, inner));
                    break;
                }
            }
            found?
        };

        // Match inner = arccos(t)
        let t = match ctx.get(inner) {
            Expr::Function(inv_id, inv_args) if inv_args.len() == 1 => {
                match ctx.builtin_of(*inv_id) {
                    Some(BuiltinFn::Acos | BuiltinFn::Arccos) => inv_args[0],
                    _ => return None,
                }
            }
            _ => return None,
        };

        // Guard: inner arg size
        if count_nodes_dedup(ctx, t) > MAX_INNER_NODES {
            return None;
        }

        let n_usize = n as usize;

        let (result, den_for_assumption, desc) = match trig {
            BuiltinFn::Cos => {
                // cos(n·arccos(t)) = Tₙ(t)
                let tn = chebyshev_t(ctx, t, n_usize);
                // No denominator — no NonZero condition needed
                let one = ctx.num(1);
                (tn, one, format!("cos({n}·arccos(t)) = T_{n}(t)"))
            }
            BuiltinFn::Sin => {
                if n_usize == 0 {
                    return None;
                }
                // sin(n·arccos(t)) = √(1-t²)·Uₙ₋₁(t)
                let one_minus = build_one_minus_t_sq(ctx, t);
                let sqrt_part = build_sqrt(ctx, one_minus);
                let u = chebyshev_u_nm1(ctx, t, n_usize);
                let result = ctx.add(Expr::Mul(sqrt_part, u));
                let one = ctx.num(1);
                (
                    result,
                    one,
                    format!("sin({n}·arccos(t)) = √(1-t²)·U_{{{n}-1}}(t)"),
                )
            }
            BuiltinFn::Tan => {
                if n_usize == 0 {
                    return None;
                }
                // tan(n·arccos(t)) = √(1-t²)·Uₙ₋₁(t) / Tₙ(t)
                let tn = chebyshev_t(ctx, t, n_usize);
                let one_minus = build_one_minus_t_sq(ctx, t);
                let sqrt_part = build_sqrt(ctx, one_minus);
                let u = chebyshev_u_nm1(ctx, t, n_usize);
                let numerator = ctx.add(Expr::Mul(sqrt_part, u));
                let result = ctx.add(Expr::Div(numerator, tn));
                (
                    result,
                    tn,
                    format!("tan({n}·arccos(t)) = √(1-t²)·U_{{{n}-1}}(t)/T_{n}(t)"),
                )
            }
            _ => return None,
        };

        // Apply sign parity: cos is even, sin/tan are odd
        let result = if !is_positive {
            match trig {
                BuiltinFn::Cos => result,
                _ => ctx.add(Expr::Neg(result)),
            }
        } else {
            result
        };

        // Guard: output size
        if count_nodes_dedup(ctx, result) > MAX_OUTPUT_NODES {
            return None;
        }

        Some(
            Rewrite::new(result)
                .desc(desc)
                .budget_exempt()
                .assume(crate::assumptions::AssumptionEvent::nonzero(ctx, den_for_assumption)),
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
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, _parent_ctx| {
        // Match sin|cos|tan(arg0)
        let (fn_id, arg0) = match ctx.get(expr) {
            Expr::Function(fn_id, args) if args.len() == 1 => (*fn_id, args[0]),
            _ => return None,
        };

        let trig = match ctx.builtin_of(fn_id) {
            Some(b @ (BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan)) => b,
            _ => return None,
        };

        // Try n=1 (no multiplier) and n=2..MAX_N
        let (is_positive, n, inner) = if let Expr::Function(_, _) = ctx.get(arg0) {
            (true, 1i64, arg0)
        } else {
            let mut found = None;
            for k in 2..=MAX_N {
                if let Some((sign, inner)) = extract_int_multiple(ctx, arg0, k) {
                    found = Some((sign, k, inner));
                    break;
                }
            }
            found?
        };

        // Match inner = arcsin(t)
        let t = match ctx.get(inner) {
            Expr::Function(inv_id, inv_args) if inv_args.len() == 1 => {
                match ctx.builtin_of(*inv_id) {
                    Some(BuiltinFn::Asin | BuiltinFn::Arcsin) => inv_args[0],
                    _ => return None,
                }
            }
            _ => return None,
        };

        // Guard: inner arg size
        if count_nodes_dedup(ctx, t) > MAX_INNER_NODES {
            return None;
        }

        let n_usize = n as usize;

        // Pre-build cosθ = √(1-t²)
        let one_minus = build_one_minus_t_sq(ctx, t);
        let cos_theta = build_sqrt(ctx, one_minus);

        let (s_n, c_n) = arcsin_recurrence(ctx, t, cos_theta, n_usize);

        let (result, den_for_assumption, desc) = match trig {
            BuiltinFn::Sin => {
                let one = ctx.num(1);
                (s_n, one, format!("sin({n}·arcsin(t)) via recurrence"))
            }
            BuiltinFn::Cos => {
                let one = ctx.num(1);
                (c_n, one, format!("cos({n}·arcsin(t)) via recurrence"))
            }
            BuiltinFn::Tan => {
                // tan(n·arcsin(t)) = Sₙ / Cₙ
                let result = ctx.add(Expr::Div(s_n, c_n));
                (result, c_n, format!("tan({n}·arcsin(t)) = Sₙ/Cₙ"))
            }
            _ => return None,
        };

        // Apply sign parity: cos is even, sin/tan are odd
        let result = if !is_positive {
            match trig {
                BuiltinFn::Cos => result,
                _ => ctx.add(Expr::Neg(result)),
            }
        } else {
            result
        };

        // Guard: output size
        if count_nodes_dedup(ctx, result) > MAX_OUTPUT_NODES {
            return None;
        }

        Some(
            Rewrite::new(result)
                .desc(desc)
                .budget_exempt()
                .assume(crate::assumptions::AssumptionEvent::nonzero(ctx, den_for_assumption)),
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
