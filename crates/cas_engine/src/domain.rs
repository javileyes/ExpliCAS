//! Domain assumption modes for simplification.
//!
//! This module defines the `DomainMode` enum that controls how the engine
//! handles operations that depend on domain assumptions (like `x/x → 1`).
//!
//! # Modes
//!
//! - **Strict**: Only perform operations that are valid for ALL values.
//!   `x/x` stays as `x/x` because it requires `x ≠ 0`.
//!
//! - **Assume**: Use user-provided assumptions. If the user declares
//!   `assume(x ≠ 0)`, then `x/x → 1` is allowed with a domain_assumption step.
//!
//! - **Generic**: Classic CAS behavior - work "almost everywhere".
//!   `x/x → 1` is allowed because it's valid for all x except 0.
//!   This is the default mode for backward compatibility.
//!
//! # Example
//!
//! ```ignore
//! use cas_engine::DomainMode;
//!
//! let opts = SimplifyOptions {
//!     domain: DomainMode::Strict,
//!     ..Default::default()
//! };
//! ```

/// Domain assumption mode for simplification.
///
/// Controls how the engine handles operations that require domain assumptions
/// like `x/x → 1` (requires `x ≠ 0`) or `√(x²) → x` (requires `x ≥ 0`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DomainMode {
    /// No domain assumptions - only proven-safe simplifications.
    ///
    /// `x/x` stays as `x/x` because we cannot prove `x ≠ 0`.
    /// `2/2 → 1` is allowed because 2 is provably non-zero.
    Strict,

    /// Use user-provided assumptions.
    ///
    /// If `assumptions.implies_nonzero(x)` returns true, then `x/x → 1`
    /// is allowed with an explicit `domain_assumption` in the step.
    Assume,

    /// "Almost everywhere" algebra (default, classic CAS behavior).
    ///
    /// Operations are allowed if they're valid for "generic" values.
    /// `x/x → 1` is allowed because it's valid for all x ≠ 0.
    /// This is the default for backward compatibility.
    #[default]
    Generic,
}

impl DomainMode {
    /// Returns true if this mode is strict (no assumptions).
    pub fn is_strict(self) -> bool {
        matches!(self, DomainMode::Strict)
    }

    /// Returns true if this mode allows generic "almost everywhere" algebra.
    pub fn is_generic(self) -> bool {
        matches!(self, DomainMode::Generic)
    }

    /// Returns true if this mode uses explicit assumptions.
    pub fn is_assume(self) -> bool {
        matches!(self, DomainMode::Assume)
    }
}

/// Result of attempting to prove a property about an expression.
///
/// Used by domain-aware simplification to decide whether operations
/// like `x/x → 1` are safe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Proof {
    /// Property is provably true (e.g., 2 ≠ 0 is proven)
    Proven,
    /// Property status is unknown (e.g., we don't know if x ≠ 0)
    Unknown,
    /// Property is provably false (e.g., 0 ≠ 0 is disproven)
    Disproven,
}

impl Proof {
    /// Returns true if this is a proven property.
    pub fn is_proven(self) -> bool {
        matches!(self, Proof::Proven)
    }

    /// Returns true if this is an unknown property.
    pub fn is_unknown(self) -> bool {
        matches!(self, Proof::Unknown)
    }

    /// Returns true if this is a disproven property.
    pub fn is_disproven(self) -> bool {
        matches!(self, Proof::Disproven)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_mode_default_is_generic() {
        assert_eq!(DomainMode::default(), DomainMode::Generic);
    }

    #[test]
    fn test_domain_mode_predicates() {
        assert!(DomainMode::Strict.is_strict());
        assert!(!DomainMode::Strict.is_generic());

        assert!(DomainMode::Generic.is_generic());
        assert!(!DomainMode::Generic.is_strict());

        assert!(DomainMode::Assume.is_assume());
        assert!(!DomainMode::Assume.is_strict());
    }
}
