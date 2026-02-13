//! Targeted simplifications for Weierstrass / inverse-trig bridge identities:
//!
//! **n=1 (Basic inverse):**
//!   sin(arctan(t)) → t / √(1+t²)
//!   cos(arctan(t)) → 1 / √(1+t²)
//!   tan(arctan(t)) → t
//!
//! **n=2 (Weierstrass / arctan):**
//!   sin(2·arctan(t)) → 2t / (1+t²)
//!   cos(2·arctan(t)) → (1−t²) / (1+t²)
//!   tan(2·arctan(t)) → 2t / (1−t²)
//!
//! **n=2 (Chebyshev / arcsin):**
//!   sin(2·arcsin(t)) → 2t·√(1−t²)
//!   cos(2·arcsin(t)) → 1 − 2t²
//!
//! **n=3 (Triple angle / arctan):**
//!   sin(3·arctan(t)) → (3t − t³) / (1+t²)^(3/2)
//!   cos(3·arctan(t)) → (1 − 3t²) / (1+t²)^(3/2)
//!   tan(3·arctan(t)) → (3t − t³) / (1 − 3t²)
//!
//! **n=3 (Chebyshev / arcsin):**
//!   sin(3·arcsin(t)) → 3t − 4t³
//!   cos(3·arcsin(t)) → 4(1−t²)^(3/2) − 3√(1−t²)
//!
//! These bridge rules connect the trig and inverse-trig sub-worlds directly,
//! reducing compositions to algebraic expressions without needing expand_mode.
//! They bypass the `TrigInverseExpansionRule` (gated behind `Analytic`) so they
//! fire even in `DomainMode::Generic`.
//!
//! **Domain safety**: Arctan identities introduce explicit divisions, so
//! those rules are classified as `NeedsCondition(Definability)` and emit
//! `AssumptionEvent::nonzero(denominator)`. Arcsin identities are purely
//! algebraic (no divisions) and are safe unconditionally.

use cas_ast::{BuiltinFn, Expr};

use crate::define_rule;
use crate::helpers::{extract_double_angle_arg, extract_triple_angle_arg};
use crate::rule::Rewrite;
use crate::target_kind::TargetKindSet;

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(BasicInverseAtanRule));
    simplifier.add_rule(Box::new(WeierstrassInverseAtanRule));
    simplifier.add_rule(Box::new(TripleAngleInverseAtanRule));
    simplifier.add_rule(Box::new(DoubleAngleInverseAsinRule));
    simplifier.add_rule(Box::new(TripleAngleInverseAsinRule));
}

// =============================================================================
// n=2: sin/cos/tan(2·arctan(t))
// =============================================================================

define_rule!(
    WeierstrassInverseAtanRule,
    "Weierstrass Inverse Atan Composition",
    Some(TargetKindSet::FUNCTION),
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, _parent_ctx| {
        // Match sin|cos|tan(arg0) with a single argument
        let (fn_id, arg0) = match ctx.get(expr) {
            Expr::Function(fn_id, args) if args.len() == 1 => (*fn_id, args[0]),
            _ => return None,
        };

        let trig = match ctx.builtin_of(fn_id) {
            Some(b @ (BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan)) => b,
            _ => return None,
        };

        // Match arg0 = 2 * inner  (double-angle form)
        let inner = extract_double_angle_arg(ctx, arg0)?;

        // Match inner = atan(t) or arctan(t)
        let t = match ctx.get(inner) {
            Expr::Function(inv_id, inv_args) if inv_args.len() == 1 => {
                match ctx.builtin_of(*inv_id) {
                    Some(BuiltinFn::Atan | BuiltinFn::Arctan) => inv_args[0],
                    _ => return None,
                }
            }
            _ => return None,
        };

        // Build common sub-expressions: t², 1+t², 1-t², 2t
        let two = ctx.num(2);
        let one = ctx.num(1);
        let t_sq = ctx.add(Expr::Pow(t, two));

        let (new_expr, den, desc) = match trig {
            BuiltinFn::Sin => {
                // sin(2·atan(t)) = 2t / (1+t²)
                let num = ctx.add(Expr::Mul(two, t));
                let den = ctx.add(Expr::Add(one, t_sq));
                let result = ctx.add(Expr::Div(num, den));
                (result, den, "sin(2·atan(t)) = 2t/(1+t²)")
            }
            BuiltinFn::Cos => {
                // cos(2·atan(t)) = (1-t²) / (1+t²)
                let num = ctx.add(Expr::Sub(one, t_sq));
                let den = ctx.add(Expr::Add(one, t_sq));
                let result = ctx.add(Expr::Div(num, den));
                (result, den, "cos(2·atan(t)) = (1−t²)/(1+t²)")
            }
            BuiltinFn::Tan => {
                // tan(2·atan(t)) = 2t / (1-t²)
                let num = ctx.add(Expr::Mul(two, t));
                let den = ctx.add(Expr::Sub(one, t_sq));
                let result = ctx.add(Expr::Div(num, den));
                (result, den, "tan(2·atan(t)) = 2t/(1−t²)")
            }
            _ => return None,
        };

        Some(
            Rewrite::new(new_expr)
                .desc(desc)
                .assume(crate::assumptions::AssumptionEvent::nonzero(ctx, den)),
        )
    }
);

// =============================================================================
// n=3: sin/cos/tan(3·arctan(t))
// =============================================================================
//
// From (1+it)³ = (1−3t²) + i(3t−t³),  |1+it|³ = (1+t²)^(3/2):
//
//   sin(3·arctan(t)) = (3t − t³) / (1+t²)^(3/2)
//   cos(3·arctan(t)) = (1 − 3t²) / (1+t²)^(3/2)
//   tan(3·arctan(t)) = (3t − t³) / (1 − 3t²)
//
// These are valid for all real t. The denominators:
//   - (1+t²)^(3/2) > 0 for all real t (emit nonzero as conservative Definability)
//   - (1 − 3t²) can be zero (genuine singularity for tan)

define_rule!(
    TripleAngleInverseAtanRule,
    "Triple Angle Inverse Atan Composition",
    Some(TargetKindSet::FUNCTION),
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, _parent_ctx| {
        // Match sin|cos|tan(arg0) with a single argument
        let (fn_id, arg0) = match ctx.get(expr) {
            Expr::Function(fn_id, args) if args.len() == 1 => (*fn_id, args[0]),
            _ => return None,
        };

        let trig = match ctx.builtin_of(fn_id) {
            Some(b @ (BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan)) => b,
            _ => return None,
        };

        // Match arg0 = 3 * inner  (triple-angle form)
        let inner = extract_triple_angle_arg(ctx, arg0)?;

        // Match inner = atan(t) or arctan(t)
        let t = match ctx.get(inner) {
            Expr::Function(inv_id, inv_args) if inv_args.len() == 1 => {
                match ctx.builtin_of(*inv_id) {
                    Some(BuiltinFn::Atan | BuiltinFn::Arctan) => inv_args[0],
                    _ => return None,
                }
            }
            _ => return None,
        };

        // Build common sub-expressions
        let one = ctx.num(1);
        let two = ctx.num(2);
        let three = ctx.num(3);
        let t_sq = ctx.add(Expr::Pow(t, two));              // t²
        let t_cubed = ctx.add(Expr::Pow(t, three));          // t³
        let three_t = ctx.add(Expr::Mul(three, t));          // 3t
        let three_t_sq = ctx.add(Expr::Mul(three, t_sq));    // 3t²
        let one_plus_t_sq = ctx.add(Expr::Add(one, t_sq));   // 1 + t²

        let (new_expr, den, desc) = match trig {
            BuiltinFn::Sin => {
                // sin(3·atan(t)) = (3t − t³) / (1+t²)^(3/2)
                let num = ctx.add(Expr::Sub(three_t, t_cubed));
                let three_half = ctx.add(Expr::Number(
                    num_rational::BigRational::new(3.into(), 2.into()),
                ));
                let den = ctx.add(Expr::Pow(one_plus_t_sq, three_half));
                let result = ctx.add(Expr::Div(num, den));
                (result, den, "sin(3·atan(t)) = (3t−t³)/(1+t²)^(3/2)")
            }
            BuiltinFn::Cos => {
                // cos(3·atan(t)) = (1 − 3t²) / (1+t²)^(3/2)
                let num = ctx.add(Expr::Sub(one, three_t_sq));
                let three_half = ctx.add(Expr::Number(
                    num_rational::BigRational::new(3.into(), 2.into()),
                ));
                let den = ctx.add(Expr::Pow(one_plus_t_sq, three_half));
                let result = ctx.add(Expr::Div(num, den));
                (result, den, "cos(3·atan(t)) = (1−3t²)/(1+t²)^(3/2)")
            }
            BuiltinFn::Tan => {
                // tan(3·atan(t)) = (3t − t³) / (1 − 3t²)
                let num = ctx.add(Expr::Sub(three_t, t_cubed));
                let den = ctx.add(Expr::Sub(one, three_t_sq));
                let result = ctx.add(Expr::Div(num, den));
                (result, den, "tan(3·atan(t)) = (3t−t³)/(1−3t²)")
            }
            _ => return None,
        };

        Some(
            Rewrite::new(new_expr)
                .desc(desc)
                .assume(crate::assumptions::AssumptionEvent::nonzero(ctx, den)),
        )
    }
);

// =============================================================================
// n=1: sin/cos/tan(arctan(t))
// =============================================================================
//
// Basic identities from the right triangle with opposite=t, adjacent=1:
//   sin(arctan(t)) = t / √(1+t²)
//   cos(arctan(t)) = 1 / √(1+t²)
//   tan(arctan(t)) = t
//
// These bypass `TrigInverseExpansionRule` (Analytic) and fire in Generic mode.
// For sin/cos, the denominator √(1+t²) > 0 for all real t.

define_rule!(
    BasicInverseAtanRule,
    "Basic Inverse Atan Composition",
    Some(TargetKindSet::FUNCTION),
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, _parent_ctx| {
        // Match sin|cos|tan(arg0) with a single argument
        let (fn_id, arg0) = match ctx.get(expr) {
            Expr::Function(fn_id, args) if args.len() == 1 => (*fn_id, args[0]),
            _ => return None,
        };

        let trig = match ctx.builtin_of(fn_id) {
            Some(b @ (BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan)) => b,
            _ => return None,
        };

        // Match arg0 = atan(t) or arctan(t) directly (n=1, no multiplier)
        let t = match ctx.get(arg0) {
            Expr::Function(inv_id, inv_args) if inv_args.len() == 1 => {
                match ctx.builtin_of(*inv_id) {
                    Some(BuiltinFn::Atan | BuiltinFn::Arctan) => inv_args[0],
                    _ => return None,
                }
            }
            _ => return None,
        };

        // Build common sub-expressions
        let one = ctx.num(1);
        let two = ctx.num(2);
        let t_sq = ctx.add(Expr::Pow(t, two));                // t²
        let one_plus_t_sq = ctx.add(Expr::Add(one, t_sq));    // 1 + t²
        let half = ctx.add(Expr::Number(
            num_rational::BigRational::new(1.into(), 2.into()),
        ));
        let sqrt_denom = ctx.add(Expr::Pow(one_plus_t_sq, half)); // √(1+t²)

        let (new_expr, den, desc) = match trig {
            BuiltinFn::Sin => {
                // sin(arctan(t)) = t / √(1+t²)
                let result = ctx.add(Expr::Div(t, sqrt_denom));
                (result, sqrt_denom, "sin(atan(t)) = t/√(1+t²)")
            }
            BuiltinFn::Cos => {
                // cos(arctan(t)) = 1 / √(1+t²)
                let result = ctx.add(Expr::Div(one, sqrt_denom));
                (result, sqrt_denom, "cos(atan(t)) = 1/√(1+t²)")
            }
            BuiltinFn::Tan => {
                // tan(arctan(t)) = t (trivial inverse)
                return Some(
                    Rewrite::new(t)
                        .desc("tan(atan(t)) = t")
                );
            }
            _ => return None,
        };

        Some(
            Rewrite::new(new_expr)
                .desc(desc)
                .assume(crate::assumptions::AssumptionEvent::nonzero(ctx, den)),
        )
    }
);

// =============================================================================
// n=2: sin/cos(2·arcsin(t))  — Chebyshev reduction
// =============================================================================
//
// From sin(2θ) = 2sinθcosθ and cos(2θ) = 1−2sin²θ, with θ=arcsin(t):
//
//   sin(2·arcsin(t)) = 2t·√(1−t²)
//   cos(2·arcsin(t)) = 1 − 2t²
//
// These are purely algebraic: no denominators, so no Definability condition.

define_rule!(
    DoubleAngleInverseAsinRule,
    "Double Angle Inverse Asin Composition",
    Some(TargetKindSet::FUNCTION),
    |ctx, expr| {
        // Match sin|cos(arg0) with a single argument
        let (fn_id, arg0) = match ctx.get(expr) {
            Expr::Function(fn_id, args) if args.len() == 1 => (*fn_id, args[0]),
            _ => return None,
        };

        let trig = match ctx.builtin_of(fn_id) {
            Some(b @ (BuiltinFn::Sin | BuiltinFn::Cos)) => b,
            _ => return None,
        };

        // Match arg0 = 2 * inner  (double-angle form)
        let inner = extract_double_angle_arg(ctx, arg0)?;

        // Match inner = arcsin(t) or asin(t)
        let t = match ctx.get(inner) {
            Expr::Function(inv_id, inv_args) if inv_args.len() == 1 => {
                match ctx.builtin_of(*inv_id) {
                    Some(BuiltinFn::Asin | BuiltinFn::Arcsin) => inv_args[0],
                    _ => return None,
                }
            }
            _ => return None,
        };

        let (new_expr, desc) = match trig {
            BuiltinFn::Sin => {
                // sin(2·arcsin(t)) = 2t·√(1−t²)
                let two = ctx.num(2);
                let one = ctx.num(1);
                let t_sq = ctx.add(Expr::Pow(t, two));
                let one_minus_t_sq = ctx.add(Expr::Sub(one, t_sq));
                let half = ctx.add(Expr::Number(num_rational::BigRational::new(
                    1.into(),
                    2.into(),
                )));
                let sqrt_part = ctx.add(Expr::Pow(one_minus_t_sq, half));
                let two_t = ctx.add(Expr::Mul(two, t));
                let result = ctx.add(Expr::Mul(two_t, sqrt_part));
                (result, "sin(2·arcsin(t)) = 2t·√(1−t²)")
            }
            BuiltinFn::Cos => {
                // cos(2·arcsin(t)) = 1 − 2t²
                let two = ctx.num(2);
                let one = ctx.num(1);
                let t_sq = ctx.add(Expr::Pow(t, two));
                let two_t_sq = ctx.add(Expr::Mul(two, t_sq));
                let result = ctx.add(Expr::Sub(one, two_t_sq));
                (result, "cos(2·arcsin(t)) = 1 − 2t²")
            }
            _ => return None,
        };

        Some(Rewrite::new(new_expr).desc(desc))
    }
);

// =============================================================================
// n=3: sin/cos(3·arcsin(t))  — Chebyshev reduction
// =============================================================================
//
// From sin(3θ) = 3sinθ − 4sin³θ and cos(3θ) = 4cos³θ − 3cosθ,
// with θ=arcsin(t), sinθ=t, cosθ=√(1−t²):
//
//   sin(3·arcsin(t)) = 3t − 4t³
//   cos(3·arcsin(t)) = 4(1−t²)^(3/2) − 3√(1−t²)
//
// No denominators are introduced.

define_rule!(
    TripleAngleInverseAsinRule,
    "Triple Angle Inverse Asin Composition",
    Some(TargetKindSet::FUNCTION),
    |ctx, expr| {
        // Match sin|cos(arg0) with a single argument
        let (fn_id, arg0) = match ctx.get(expr) {
            Expr::Function(fn_id, args) if args.len() == 1 => (*fn_id, args[0]),
            _ => return None,
        };

        let trig = match ctx.builtin_of(fn_id) {
            Some(b @ (BuiltinFn::Sin | BuiltinFn::Cos)) => b,
            _ => return None,
        };

        // Match arg0 = 3 * inner  (triple-angle form)
        let inner = extract_triple_angle_arg(ctx, arg0)?;

        // Match inner = arcsin(t) or asin(t)
        let t = match ctx.get(inner) {
            Expr::Function(inv_id, inv_args) if inv_args.len() == 1 => {
                match ctx.builtin_of(*inv_id) {
                    Some(BuiltinFn::Asin | BuiltinFn::Arcsin) => inv_args[0],
                    _ => return None,
                }
            }
            _ => return None,
        };

        let (new_expr, desc) = match trig {
            BuiltinFn::Sin => {
                // sin(3·arcsin(t)) = 3t − 4t³
                let three = ctx.num(3);
                let four = ctx.num(4);
                let three_t = ctx.add(Expr::Mul(three, t));
                let t_cubed = ctx.add(Expr::Pow(t, three));
                let four_t_cubed = ctx.add(Expr::Mul(four, t_cubed));
                let result = ctx.add(Expr::Sub(three_t, four_t_cubed));
                (result, "sin(3·arcsin(t)) = 3t − 4t³")
            }
            BuiltinFn::Cos => {
                // cos(3·arcsin(t)) = 4(1−t²)^(3/2) − 3√(1−t²)
                let one = ctx.num(1);
                let two = ctx.num(2);
                let three = ctx.num(3);
                let four = ctx.num(4);
                let t_sq = ctx.add(Expr::Pow(t, two));
                let one_minus_t_sq = ctx.add(Expr::Sub(one, t_sq));
                let half = ctx.add(Expr::Number(num_rational::BigRational::new(
                    1.into(),
                    2.into(),
                )));
                let three_half = ctx.add(Expr::Number(num_rational::BigRational::new(
                    3.into(),
                    2.into(),
                )));
                let sqrt_part = ctx.add(Expr::Pow(one_minus_t_sq, half));
                let pow_3_2 = ctx.add(Expr::Pow(one_minus_t_sq, three_half));
                let four_pow = ctx.add(Expr::Mul(four, pow_3_2));
                let three_sqrt = ctx.add(Expr::Mul(three, sqrt_part));
                let result = ctx.add(Expr::Sub(four_pow, three_sqrt));
                (result, "cos(3·arcsin(t)) = 4(1−t²)^(3/2) − 3√(1−t²)")
            }
            _ => return None,
        };

        Some(Rewrite::new(new_expr).desc(desc))
    }
);
