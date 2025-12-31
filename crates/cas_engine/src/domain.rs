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

    /// Check if this mode allows an unproven condition of the given class.
    ///
    /// This is the **central gate** for the Strict/Generic/Assume contract:
    /// - **Strict**: Never allows unproven conditions (returns false always)
    /// - **Generic**: Allows Definability (≠0), rejects Analytic (>0, ranges)
    /// - **Assume**: Allows all condition classes
    ///
    /// # Example
    /// ```ignore
    /// use cas_engine::assumptions::ConditionClass;
    ///
    /// // NonZero is Definability
    /// assert!(!DomainMode::Strict.allows_unproven(ConditionClass::Definability));
    /// assert!(DomainMode::Generic.allows_unproven(ConditionClass::Definability));
    /// assert!(DomainMode::Assume.allows_unproven(ConditionClass::Definability));
    ///
    /// // Positive is Analytic
    /// assert!(!DomainMode::Strict.allows_unproven(ConditionClass::Analytic));
    /// assert!(!DomainMode::Generic.allows_unproven(ConditionClass::Analytic));
    /// assert!(DomainMode::Assume.allows_unproven(ConditionClass::Analytic));
    /// ```
    pub fn allows_unproven(self, class: crate::assumptions::ConditionClass) -> bool {
        use crate::assumptions::ConditionClass;
        match self {
            DomainMode::Strict => false, // Never accept unproven
            DomainMode::Generic => class == ConditionClass::Definability,
            DomainMode::Assume => true, // Accept all
        }
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

// =============================================================================
// Canonical Domain Gate Helper
// =============================================================================

/// Result of a domain-aware cancellation decision.
///
/// Used by cancellation rules to determine whether a factor can be cancelled
/// and whether to record a domain assumption.
#[derive(Debug, Clone)]
pub struct CancelDecision {
    /// Whether the cancellation is allowed.
    pub allow: bool,
    /// Optional domain assumption message for steps/warnings.
    /// Set when cancellation is allowed but based on unproven assumption.
    pub assumption: Option<&'static str>,
}

impl CancelDecision {
    /// Create a decision that allows cancellation with no assumption.
    pub fn allow() -> Self {
        Self {
            allow: true,
            assumption: None,
        }
    }

    /// Create a decision that blocks cancellation.
    pub fn deny() -> Self {
        Self {
            allow: false,
            assumption: None,
        }
    }

    /// Create a decision that allows with a domain assumption warning.
    pub fn allow_with_assumption(msg: &'static str) -> Self {
        Self {
            allow: true,
            assumption: Some(msg),
        }
    }
}

/// Canonical helper: Decide whether to cancel a factor based on domain mode.
///
/// This is the single point of truth for all cancellation rules (NonZero condition).
/// Call this before any `a/a → 1` or similar cancellation.
///
/// Uses the ConditionClass taxonomy:
/// - **Strict**: Only allows if `proof == Proven`
/// - **Generic**: Allows Definability (NonZero) even if Unknown
/// - **Assume**: Allows all conditions if Unknown
///
/// # Returns
///
/// - `allow: true` with no assumption for proven factors (e.g., `2/2 → 1`)
/// - `allow: false` in Strict mode for unproven factors (e.g., `x/x` stays)
/// - `allow: true` with assumption in Generic/Assume for unproven factors
///
/// # Example
///
/// ```ignore
/// let proof = prove_nonzero(ctx, factor);
/// let decision = can_cancel_factor(mode, proof);
/// if !decision.allow {
///     return None; // Don't cancel in Strict mode
/// }
/// // Apply rewrite, use decision.assumption if present
/// ```
pub fn can_cancel_factor(mode: DomainMode, proof: Proof) -> CancelDecision {
    use crate::assumptions::ConditionClass;

    match proof {
        // Always allow if proven
        Proof::Proven => CancelDecision::allow(),

        // Never allow if disproven (division by 0)
        Proof::Disproven => CancelDecision::deny(),

        // Unknown: use ConditionClass gate
        Proof::Unknown => {
            // NonZero is Definability class
            if mode.allows_unproven(ConditionClass::Definability) {
                CancelDecision::allow_with_assumption("cancelled factor assumed nonzero")
            } else {
                CancelDecision::deny()
            }
        }
    }
}

/// Canonical helper: Decide whether to apply an Analytic condition (Positive, NonNegative).
///
/// This is the single point of truth for rules requiring x > 0 or x ≥ 0.
/// Uses for: `exp(ln(x)) → x`, `ln(x*y) → ln(x) + ln(y)`, etc.
///
/// Uses the ConditionClass taxonomy:
/// - **Strict**: Only allows if `proof == Proven`
/// - **Generic**: BLOCKS (Analytic is not Definability)
/// - **Assume**: Allows and records assumption
///
/// # Returns
///
/// - `allow: true` with no assumption for proven conditions
/// - `allow: false` in Strict and Generic modes for unproven
/// - `allow: true` with assumption only in Assume mode
///
/// # Example
///
/// ```ignore
/// let proof = prove_positive(ctx, arg, vd);
/// let decision = can_apply_analytic(mode, proof);
/// if !decision.allow {
///     return None; // Strict/Generic: don't apply
/// }
/// // Apply rewrite, use decision.assumption if present
/// ```
pub fn can_apply_analytic(mode: DomainMode, proof: Proof) -> CancelDecision {
    use crate::assumptions::ConditionClass;

    match proof {
        // Always allow if proven
        Proof::Proven => CancelDecision::allow(),

        // Never allow if disproven (definitely ≤ 0)
        Proof::Disproven => CancelDecision::deny(),

        // Unknown: use Analytic ConditionClass gate
        Proof::Unknown => {
            // Positive/NonNegative is Analytic class (only Assume allows)
            if mode.allows_unproven(ConditionClass::Analytic) {
                CancelDecision::allow_with_assumption("assumed positive")
            } else {
                CancelDecision::deny() // Strict and Generic block this
            }
        }
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
