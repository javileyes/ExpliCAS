//! Core fraction rules and helpers.
//!
//! This module contains the main simplification and cancellation rules,
//! along with helper functions for polynomial comparison and factor collection.

// =============================================================================
// Context-aware helpers for AddFractionsRule gating
// =============================================================================

pub(super) use cas_math::expr_classify::{is_pi_constant, is_trig_function};

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
