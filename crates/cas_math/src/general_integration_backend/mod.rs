//! General algorithmic integration backend (block 12).
//!
//! This module defines the result contract that a broader integration backend
//! must satisfy before any candidate can be consumed by public integration
//! routes, plus the bounded probe/verification machinery behind it.
//!
//! Ownership map:
//! - [`probe_runner`]: backend modes, budgets, configuration, and probe
//!   accounting.
//! - [`model`]: the result contract (candidate, method tags, verification
//!   statuses, residual reasons, and domain/constant/trace policies).
//! - [`verification`]: the antiderivative verification service
//!   (`diff(candidate, x) ~ integrand`) with structured outcomes.
//! - [`verification_normalization`]: bounded structural normalization used by
//!   verification to match generated derivatives against integrands. New
//!   `normalize_backend_*` cases belong here only for shapes the algebraic
//!   zero test cannot decide.
//! - [`verification_algebraic`]: multipoly zero-test for rational residuals
//!   (Phase 4 workstream) — the decision procedure that replaces structural
//!   normalization growth for rational shapes, including square-root atoms
//!   of variable-free radicands via the quotient relation `t^2 = radicand`.
//! - [`methods`]: method probes (rational, Hermite, heurisch) and the public
//!   entry point [`try_algorithmic_integration_backend`].
//!
//! Known boundary note: `verification` reuses affine-denominator part
//! extraction owned by `methods` (`affine_denominator_linear_numerator_parts`),
//! and both lean on a shared expression toolkit (numeric helpers and
//! `build_backend_*` builders). That toolkit is a consolidation candidate once
//! another family confirms the pattern; do not widen it ad hoc.

const BACKEND_VERIFICATION_NORMALIZE_DEPTH: usize = 32;
const BACKEND_VERIFICATION_NORMALIZE_PASSES: usize = 4;
const BACKEND_RESIDUAL_SIGNATURE_DEPTH: usize = 48;
const BACKEND_EXTERNAL_COEFFICIENT_DEPTH: usize = 16;

mod methods;
mod model;
mod probe_runner;
mod verification;
mod verification_algebraic;
mod verification_normalization;

// Re-export the public API so `cas_math::general_integration_backend::X`
// continues to work unchanged.
pub use methods::{
    backend_positive_quadratic_denominator_radius,
    general_rational_partial_fraction_narration_parts,
    multi_quadratic_partial_fraction_decomposition_expr, try_algorithmic_integration_backend,
    GeneralRationalNarrationParts,
};
pub use model::*;
pub use probe_runner::*;
pub use verification::{
    antiderivative_verification_report, verify_antiderivative_by_differentiation,
};

#[cfg(test)]
mod tests;
