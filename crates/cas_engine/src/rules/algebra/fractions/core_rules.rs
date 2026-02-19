//! Core fraction rules and helpers.
//!
//! This module contains the main simplification and cancellation rules,
//! along with helper functions for polynomial comparison and factor collection.

use cas_ast::{Context, ExprId};

// =============================================================================
// Context-aware helpers for AddFractionsRule gating
// =============================================================================

/// Check if a function name is trigonometric (sin, cos, tan and inverses/hyperbolics)
pub(super) fn is_trig_function_name(name: &str) -> bool {
    matches!(
        name,
        "sin"
            | "cos"
            | "tan"
            | "csc"
            | "sec"
            | "cot"
            | "asin"
            | "acos"
            | "atan"
            | "sinh"
            | "cosh"
            | "tanh"
            | "asinh"
            | "acosh"
            | "atanh"
    )
}

/// Check if a function (by SymbolId) is trigonometric
pub(super) fn is_trig_function(ctx: &Context, fn_id: usize) -> bool {
    ctx.builtin_of(fn_id)
        .is_some_and(|b| is_trig_function_name(b.name()))
}

/// Check if expression is a constant involving π (e.g., pi, pi/9, 2*pi/3)
pub(super) fn is_pi_constant(ctx: &Context, id: ExprId) -> bool {
    crate::helpers::extract_rational_pi_multiple(ctx, id).is_some()
}

// =============================================================================
// Polynomial equality helper (for canonical comparison ignoring AST order)
// =============================================================================

pub(super) use cas_math::poly_compare::{poly_eq, poly_relation, SignRelation};

// =============================================================================
// A1: Structural Factor Cancellation (without polynomial expansion)
// =============================================================================

pub(super) use cas_math::fraction_factors::build_mul_from_factors_int_pow as build_mul_from_factors_a1;
pub(super) use cas_math::fraction_factors::collect_mul_factors_int_pow;

// =============================================================================
// Multivariate GCD (Layers 1 + 2 + 2.5)
// =============================================================================

pub(super) use cas_math::fraction_multivar_gcd::try_multivar_gcd;

// ========== Helper to extract fraction parts from both Div and Mul(1/n,x) ==========
// This is needed because canonicalization may convert Div(x,n) to Mul(1/n,x)

pub(super) use cas_math::fraction_forms::check_divisible_denominators;
pub(super) use cas_math::fraction_forms::extract_as_fraction;
