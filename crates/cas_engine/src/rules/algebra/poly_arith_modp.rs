//! Polynomial Arithmetic on __hold operands (mod p).
//!
//! This module provides rules for arithmetic operations between __hold-wrapped
//! polynomial expressions. When both sides of a Sub/Add are __hold(polynomial),
//! we convert to MultiPolyModP and compute in that domain.
//!
//! Key use case: `poly_gcd(expand(a*g), expand(b*g), modp) - expand(g)` → 0
//!
//! The problem: expand() returns __hold(giant_polynomial) to prevent simplifier
//! explosion, but Sub doesn't "enter" __hold by design. This rule handles it.

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::poly_modp_calls::try_rewrite_hold_poly_sub_to_zero;
use cas_math::poly_modp_conv::DEFAULT_PRIME;

// PolySubModpRule: handle __hold(P) - __hold(Q) in polynomial domain
define_rule!(
    PolySubModpRule,
    "Polynomial Subtraction (mod p)",
    Some(crate::target_kind::TargetKindSet::SUB),
    PhaseMask::CORE | PhaseMask::POST,
    |ctx, expr| {
        let zero = try_rewrite_hold_poly_sub_to_zero(ctx, expr, DEFAULT_PRIME)?;
        Some(
            Rewrite::new(zero)
                .desc("__hold(P) - __hold(Q) = 0 (equal polynomials mod p, up to scalar)"),
        )
    }
);
