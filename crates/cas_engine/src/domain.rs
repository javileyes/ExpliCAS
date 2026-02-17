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

use std::cell::RefCell;

// =============================================================================
// Thread-Local Blocked Hint Collector
// =============================================================================

thread_local! {
    /// Thread-local storage for blocked hints during simplification.
    /// This allows rules to emit hints without modifying the ParentContext signature.
    static BLOCKED_HINTS: RefCell<Vec<BlockedHint>> = const { RefCell::new(Vec::new()) };
}

/// Register a blocked hint (called by can_apply_analytic_with_hint).
/// Hints are deduplicated by (rule, key).
pub fn register_blocked_hint(hint: BlockedHint) {
    BLOCKED_HINTS.with(|hints| {
        let mut hints = hints.borrow_mut();
        // Dedup: check if already exists
        let exists = hints
            .iter()
            .any(|h| h.rule == hint.rule && h.key == hint.key);
        if !exists {
            hints.push(hint);
        }
    });
}

/// Take all blocked hints, clearing the thread-local storage.
/// Called at end of simplification to retrieve hints.
pub fn take_blocked_hints() -> Vec<BlockedHint> {
    BLOCKED_HINTS.with(|hints| std::mem::take(&mut *hints.borrow_mut()))
}

/// Clear blocked hints without returning them.
/// Called at start of simplification to reset state.
pub fn clear_blocked_hints() {
    BLOCKED_HINTS.with(|hints| hints.borrow_mut().clear());
}

/// Domain assumption mode for simplification.
///
/// Controls how the engine handles operations that require domain assumptions
/// like `x/x → 1` (requires `x ≠ 0`) or `√(x²) → x` (requires `x ≥ 0`).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
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
    /// Property is implied by expression structure (e.g., sqrt(x) implies x ≥ 0).
    /// Only valid when witness survives in output - no assumption event generated.
    ProvenImplicit,
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

/// Hint emitted when an Analytic condition blocks transformation in Generic mode.
///
/// This enables pedagogical warnings like:
/// "Cannot simplify: requires x > 0. Use `semantics set domain assume` to allow."
#[derive(Debug, Clone)]
pub struct BlockedHint {
    /// The assumption key (e.g., Positive{expr}) - used for deduplication
    pub key: crate::assumptions::AssumptionKey,
    /// The original expression ID for pretty-printing (display only)
    pub expr_id: cas_ast::ExprId,
    /// Name of the rule that was blocked
    pub rule: String,
    /// Suggestion for the user
    pub suggestion: &'static str,
}

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
    /// Hint when blocked due to Analytic condition in Generic mode.
    /// Only set for Generic mode blocks, not for Strict.
    pub blocked_hint: Option<BlockedHint>,
    /// Keys of assumptions made when allowing (for timeline tracking).
    /// Populated when Unknown proof is allowed by mode.
    pub assumed_keys: smallvec::SmallVec<[crate::assumptions::AssumptionKey; 2]>,
}

impl CancelDecision {
    /// Create a decision that allows cancellation with no assumption (proven).
    pub fn allow() -> Self {
        Self {
            allow: true,
            assumption: None,
            blocked_hint: None,
            assumed_keys: smallvec::SmallVec::new(),
        }
    }

    /// Create a decision that blocks cancellation (no pedagogical hint).
    pub fn deny() -> Self {
        Self {
            allow: false,
            assumption: None,
            blocked_hint: None,
            assumed_keys: smallvec::SmallVec::new(),
        }
    }

    /// Create a decision that blocks cancellation with a pedagogical hint.
    /// Used when Generic mode blocks an Analytic condition.
    pub fn deny_with_hint(
        key: crate::assumptions::AssumptionKey,
        expr_id: cas_ast::ExprId,
        rule: &'static str,
    ) -> Self {
        Self {
            allow: false,
            assumption: None,
            blocked_hint: Some(BlockedHint {
                key,
                expr_id,
                rule: rule.to_string(),
                suggestion: "use `semantics set domain assume` to allow analytic assumptions",
            }),
            assumed_keys: smallvec::SmallVec::new(),
        }
    }

    /// Create a decision that allows with a domain assumption warning.
    pub fn allow_with_assumption(msg: &'static str) -> Self {
        Self {
            allow: true,
            assumption: Some(msg),
            blocked_hint: None,
            assumed_keys: smallvec::SmallVec::new(),
        }
    }

    /// Create a decision that allows with tracked assumed keys for timeline.
    /// Use this when Unknown proof is allowed by mode - populates assumption info.
    pub fn allow_with_keys(
        msg: &'static str,
        keys: smallvec::SmallVec<[crate::assumptions::AssumptionKey; 2]>,
    ) -> Self {
        Self {
            allow: true,
            assumption: Some(msg),
            blocked_hint: None,
            assumed_keys: keys,
        }
    }

    /// Convert assumed_keys to AssumptionEvents for Rewrite propagation.
    /// Call this when constructing Rewrite to populate assumption_events.
    pub fn assumption_events(
        &self,
        ctx: &cas_ast::Context,
        expr_id: cas_ast::ExprId,
    ) -> smallvec::SmallVec<[crate::assumptions::AssumptionEvent; 1]> {
        self.assumed_keys
            .iter()
            .map(|key| {
                let expr_display = cas_formatter::DisplayExpr {
                    context: ctx,
                    id: expr_id,
                }
                .to_string();
                // Format: "Assumed x ≠ 0" instead of "Assumed ≠ 0 (NonZero)"
                let message = match key {
                    crate::assumptions::AssumptionKey::NonZero { .. } => {
                        format!("Assumed {} ≠ 0", expr_display)
                    }
                    crate::assumptions::AssumptionKey::Positive { .. } => {
                        format!("Assumed {} > 0", expr_display)
                    }
                    crate::assumptions::AssumptionKey::NonNegative { .. } => {
                        format!("Assumed {} ≥ 0", expr_display)
                    }
                    crate::assumptions::AssumptionKey::Defined { .. } => {
                        format!("Assumed {} is defined", expr_display)
                    }
                    crate::assumptions::AssumptionKey::InvTrigPrincipalRange { func, .. } => {
                        format!("Assumed {} in {} principal range", expr_display, func)
                    }
                    crate::assumptions::AssumptionKey::ComplexPrincipalBranch { func, .. } => {
                        format!("Assumed {}({}) principal branch", func, expr_display)
                    }
                };
                crate::assumptions::AssumptionEvent {
                    key: key.clone(),
                    expr_display,
                    message,
                    kind: crate::assumptions::AssumptionKind::DerivedFromRequires,
                    expr_id: Some(expr_id),
                }
            })
            .collect()
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
    // Delegate to unified domain_facts::decide_by_class().
    // NonZero cancellation uses Definability condition class.
    let strength = crate::domain_facts::proof_to_strength(proof);
    crate::domain_facts::decide_by_class(
        mode,
        crate::assumptions::ConditionClass::Definability,
        strength,
        "cancelled factor assumed nonzero",
    )
}

/// Rich version of `can_cancel_factor` that emits pedagogical hints when Strict blocks.
///
/// This enables hints like:
/// "Blocked in Strict: requires x ≠ 0 [SimplifyFraction]. Use `domain generic` to allow."
///
/// # Arguments
///
/// * `mode` - Current DomainMode
/// * `proof` - Proof status for nonzero
/// * `key` - AssumptionKey for the condition (e.g., NonZero{expr})
/// * `expr_id` - ExprId for pretty-printing the expression
/// * `rule` - Name of the rule being blocked
pub fn can_cancel_factor_with_hint(
    mode: DomainMode,
    proof: Proof,
    key: crate::assumptions::AssumptionKey,
    expr_id: cas_ast::ExprId,
    rule: &'static str,
) -> CancelDecision {
    use crate::assumptions::ConditionClass;

    match proof {
        // Always allow if proven (explicit or implicit)
        Proof::Proven | Proof::ProvenImplicit => CancelDecision::allow(),

        // Never allow if disproven (division by 0)
        Proof::Disproven => CancelDecision::deny(),

        // Unknown: use ConditionClass gate
        Proof::Unknown => {
            if mode.allows_unproven(ConditionClass::Definability) {
                // Generic/Assume mode: allow with tracked assumption for timeline
                let keys = smallvec::smallvec![key.clone()];
                CancelDecision::allow_with_keys("cancelled factor assumed nonzero", keys)
            } else if mode == DomainMode::Strict {
                // Strict mode: block WITH pedagogical hint
                let hint = BlockedHint {
                    key: key.clone(),
                    expr_id,
                    rule: rule.to_string(),
                    suggestion: "use `domain generic` to allow definability assumptions",
                };
                register_blocked_hint(hint);
                CancelDecision::deny_with_hint(key, expr_id, rule)
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
/// - **Generic**: BLOCKS with pedagogical hint (Analytic is not Definability)
/// - **Assume**: Allows and records assumption
///
/// For full hint information (rule name, assumption key), use `can_apply_analytic_with_hint`.
///
/// # Returns
///
/// - `allow: true` with no assumption for proven conditions
/// - `allow: false` in Strict and Generic modes for unproven
/// - `allow: true` with assumption only in Assume mode
pub fn can_apply_analytic(mode: DomainMode, proof: Proof) -> CancelDecision {
    // Delegate to unified domain_facts::decide_by_class().
    // Analytic conditions (Positive, NonNegative) use Analytic condition class.
    let strength = crate::domain_facts::proof_to_strength(proof);
    crate::domain_facts::decide_by_class(
        mode,
        crate::assumptions::ConditionClass::Analytic,
        strength,
        "assumed positive",
    )
}

/// Rich version of `can_apply_analytic` that includes pedagogical hint for Generic blocks.
///
/// Use this when you have the `AssumptionKey` available to provide better error messages.
/// The hint is only emitted for Generic mode (not Strict, where blocking is expected).
///
/// # Arguments
///
/// * `mode` - Current DomainMode
/// * `proof` - Proof status for the condition
/// * `key` - AssumptionKey for the condition (e.g., Positive{expr})
/// * `rule` - Name of the rule being blocked
///
/// # Example
///
/// ```ignore
/// let key = AssumptionKey::positive(ctx, arg);
/// let decision = can_apply_analytic_with_hint(mode, proof, key, "Exponential-Log Inverse");
/// if !decision.allow {
///     if let Some(hint) = &decision.blocked_hint {
///         parent_ctx.register_blocked_hint(hint.clone());
///     }
///     return None;
/// }
/// ```
pub fn can_apply_analytic_with_hint(
    mode: DomainMode,
    proof: Proof,
    key: crate::assumptions::AssumptionKey,
    expr_id: cas_ast::ExprId,
    rule: &'static str,
) -> CancelDecision {
    use crate::assumptions::ConditionClass;

    match proof {
        // Always allow if proven (explicit or implicit)
        Proof::Proven | Proof::ProvenImplicit => CancelDecision::allow(),

        // Never allow if disproven (definitely ≤ 0)
        Proof::Disproven => CancelDecision::deny(),

        // Unknown: use Analytic ConditionClass gate
        Proof::Unknown => {
            if mode.allows_unproven(ConditionClass::Analytic) {
                // Assume mode: allow with tracked assumption for timeline
                let keys = smallvec::smallvec![key.clone()];
                CancelDecision::allow_with_keys("assumed positive", keys)
            } else if mode == DomainMode::Generic {
                // Generic mode: block WITH pedagogical hint
                // Auto-register to thread-local for REPL to retrieve
                let hint = BlockedHint {
                    key: key.clone(),
                    expr_id,
                    rule: rule.to_string(),
                    suggestion: "use `semantics set domain assume` to allow analytic assumptions",
                };
                register_blocked_hint(hint);
                CancelDecision::deny_with_hint(key, expr_id, rule)
            } else {
                // Strict mode: block without hint (expected behavior)
                CancelDecision::deny()
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
