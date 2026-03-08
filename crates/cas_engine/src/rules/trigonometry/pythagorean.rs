//! # Pythagorean Identity Simplification
//!
//! This module provides the `TrigPythagoreanSimplifyRule` which applies the
//! Pythagorean identity to simplify expressions of the form:
//!
//! - `k - k*sin²(x) → k*cos²(x)`
//! - `k - k*cos²(x) → k*sin²(x)`
//!
//! This rule was extracted from `CancelCommonFactorsRule` for better
//! step-by-step transparency and pedagogical clarity.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::trig_power_identity_support::{
    try_rewrite_pythagorean_chain_add_expr, try_rewrite_pythagorean_factor_form_add_expr,
    try_rewrite_pythagorean_generic_coefficient_add_expr,
    try_rewrite_pythagorean_high_power_add_expr, try_rewrite_pythagorean_linear_fold_add_expr,
    try_rewrite_pythagorean_local_collect_fold_add_expr,
    try_rewrite_recognize_csc_squared_add_expr, try_rewrite_recognize_sec_squared_add_expr,
};

define_rule!(
    TrigPythagoreanSimplifyRule,
    "Pythagorean Factor Form",
    |ctx, expr| {
        let rewrite = try_rewrite_pythagorean_factor_form_add_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// =============================================================================
// TrigPythagoreanChainRule: sin²(t) + cos²(t) → 1 (n-ary)
// =============================================================================
// Searches for sin²(t) and cos²(t) pairs with matching arguments in an additive
// chain of ANY length and replaces the pair with 1.
// This enables simplifications like: cos²(x/2) + sin²(x/2) - 1 → 0

define_rule!(
    TrigPythagoreanChainRule,
    "Pythagorean Chain Identity",
    |ctx, expr| {
        let rewrite = try_rewrite_pythagorean_chain_add_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// =============================================================================
// TrigPythagoreanGenericCoefficientRule: A*sin²(t) + A*cos²(t) → A
// =============================================================================
// Extends the Pythagorean identity to work when the coefficient is any expression,
// not just a numeric constant. This enables simplifications like:
//   cos(u)²*sin(x)² + cos(u)²*cos(x)² → cos(u)²
// Which is needed to prove equivalences in combined identities.
//
// Key insight: when a term like cos(u)²*sin(x)² contains multiple trig², we
// extract ALL possible candidates and match across terms.

define_rule!(
    TrigPythagoreanGenericCoefficientRule,
    "Pythagorean with Generic Coefficient",
    |ctx, expr| {
        let rewrite = try_rewrite_pythagorean_generic_coefficient_add_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// =============================================================================
// TrigPythagoreanLinearFoldRule: a·sin²(t) + b·cos²(t) + c → (a-b)·sin²(t) + (b+c)
// =============================================================================
// Uses the identity sin²(t) + cos²(t) = 1 to reduce linear combinations.
// Example: cos²(u) + 2·sin²(u) - 1 → sin²(u)
// This handles cases where we have both sin² and cos² of the same argument
// with numeric coefficients.

define_rule!(
    TrigPythagoreanLinearFoldRule,
    "Pythagorean Linear Fold",
    |ctx, expr| {
        let rewrite = try_rewrite_pythagorean_linear_fold_add_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// =============================================================================
// TrigPythagoreanLocalCollectFoldRule: k·R·sin²(t) + R·cos²(t) - R → (k-1)·R·sin²(t)
// =============================================================================
// Finds triplets in Add n-ary where:
// - Two terms share a common residual factor R multiplied by sin²(t) and cos²(t)
// - A third term is -R (or c·R where c allows folding)
// Example: 2·cos(x)²·sin(u)² + cos(u)²·cos(x)² - cos(x)² → cos(x)²·sin(u)²

define_rule!(
    TrigPythagoreanLocalCollectFoldRule,
    "Pythagorean Local Collect Fold",
    |ctx, expr| {
        let rewrite = try_rewrite_pythagorean_local_collect_fold_add_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// =============================================================================
// RecognizeSecSquaredRule: 1 + tan²(x) → sec²(x) (contraction to canonical form)
// =============================================================================
// This is the CANONICAL direction - contracting to sec² is "simpler" (fewer nodes).
// The reverse (expansion) should NOT be done in generic mode to avoid worsen.

define_rule!(
    RecognizeSecSquaredRule,
    "Recognize Secant Squared",
    |ctx, expr| {
        let rewrite = try_rewrite_recognize_sec_squared_add_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// =============================================================================
// RecognizeCscSquaredRule: 1 + cot²(x) → csc²(x) (contraction to canonical form)
// =============================================================================

define_rule!(
    RecognizeCscSquaredRule,
    "Recognize Cosecant Squared",
    |ctx, expr| {
        let rewrite = try_rewrite_recognize_csc_squared_add_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// =============================================================================
// TrigPythagoreanHighPowerRule: R − R·trig²(x) → R·other²(x)
// =============================================================================
// Handles cases where trig² is embedded in a higher trig power, e.g.:
//   4·sin(x) − 4·sin³(x) → 4·cos²(x)·sin(x)
//   sin²(x) − sin²(x)·cos²(x) → sin⁴(x)
//
// Strategy: flatten both Add terms into multiplicative factor lists, decompose
// any trig^n (n≥2) into trig^(n-2)·trig², and check if the "bigger" term has
// exactly one extra trig² factor compared to the "smaller" one.

define_rule!(
    TrigPythagoreanHighPowerRule,
    "Pythagorean High-Power Factor",
    |ctx, expr| {
        let rewrite = try_rewrite_pythagorean_high_power_add_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);
