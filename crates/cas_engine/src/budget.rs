//! Unified Anti-Explosion Budget System
//!
//! This module provides a coherent budget system to prevent computational explosion
//! across all CAS operations. It replaces the fragmented budget types (ExpandBudget,
//! PolyBudget, ZippelBudget, etc.) with a single, auditable policy.
//!
//! # Architecture
//!
//! - **`Operation`**: What is being done (Expand, Simplify, GCD, etc.)
//! - **`Metric`**: What is being measured (nodes, terms, rewrites, etc.)
//! - **`Budget`**: Configuration + runtime state, array-indexed for O(1) charge
//! - **`BudgetScope`**: RAII guard for tracking current operation
//!
//! # Usage
//!
//! ```ignore
//! let mut budget = Budget::default();
//!
//! // Set limits
//! budget.set_limit(Operation::Expand, Metric::TermsMaterialized, 1000);
//!
//! // Charge during expansion
//! budget.charge(Operation::Expand, Metric::TermsMaterialized, 50)?;
//!
//! // Use scope for automatic operation tracking
//! {
//!     let _scope = budget.scope(Operation::SimplifyCore);
//!     budget.charge_current(Metric::RewriteSteps, 1)?;
//! }
//! ```

use std::fmt;

// =============================================================================
// Operation enum
// =============================================================================

/// Operations that consume budget.
///
/// Each operation represents a distinct computational phase that can explode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Operation {
    /// Core simplification (rule applications in main loop)
    #[default]
    SimplifyCore = 0,
    /// Transform phase (distribution, expansion)
    SimplifyTransform = 1,
    /// Auto-expand of Pow(Add, n)
    Expand = 2,
    /// Multinomial expansion
    MultinomialExpand = 3,
    /// Expression to polynomial conversion
    ExprToPoly = 4,
    /// Expensive polynomial operations (mul, div, gcd)
    PolyOps = 5,
    /// Mod-p polynomial conversion
    PolyModpConv = 6,
    /// Zippel GCD algorithm
    GcdZippel = 7,
    /// Rationalization
    Rationalize = 8,
}

impl Operation {
    /// Number of operation variants (for array sizing)
    pub const COUNT: usize = 9;

    /// Get index for array access
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SimplifyCore => write!(f, "SimplifyCore"),
            Self::SimplifyTransform => write!(f, "SimplifyTransform"),
            Self::Expand => write!(f, "Expand"),
            Self::MultinomialExpand => write!(f, "MultinomialExpand"),
            Self::ExprToPoly => write!(f, "ExprToPoly"),
            Self::PolyOps => write!(f, "PolyOps"),
            Self::PolyModpConv => write!(f, "PolyModpConv"),
            Self::GcdZippel => write!(f, "GcdZippel"),
            Self::Rationalize => write!(f, "Rationalize"),
        }
    }
}

// =============================================================================
// Metric enum
// =============================================================================

/// Metrics being measured for budget tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Metric {
    /// Nodes created in Context (proxy for memory)
    /// Tracked via `ctx.stats().nodes_created`
    NodesCreated = 0,
    /// Rule applications (proxy for simplify time)
    RewriteSteps = 1,
    /// Terms materialized during expansion (proxy for combinatorial explosion)
    TermsMaterialized = 2,
    /// Expensive polynomial operations (div, gcd, eval, interpolation)
    PolyOps = 3,
}

impl Metric {
    /// Number of metric variants (for array sizing)
    pub const COUNT: usize = 4;

    /// Get index for array access
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodesCreated => write!(f, "NodesCreated"),
            Self::RewriteSteps => write!(f, "RewriteSteps"),
            Self::TermsMaterialized => write!(f, "TermsMaterialized"),
            Self::PolyOps => write!(f, "PolyOps"),
        }
    }
}

// =============================================================================
// BudgetExceeded error
// =============================================================================

/// Error returned when a budget limit is exceeded.
#[derive(Debug, Clone)]
pub struct BudgetExceeded {
    /// Which operation exceeded the budget
    pub op: Operation,
    /// Which metric exceeded
    pub metric: Metric,
    /// How much was used
    pub used: u64,
    /// What the limit was
    pub limit: u64,
}

impl fmt::Display for BudgetExceeded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "budget exceeded during {}: {} used {} (limit {})",
            self.op, self.metric, self.used, self.limit
        )
    }
}

impl std::error::Error for BudgetExceeded {}

// =============================================================================
// PassStats: budget tracking data from a simplify pass
// =============================================================================

/// Statistics collected from a single simplify pass for budget charging.
///
/// Returned by `apply_rules_loop_*` functions so the caller can charge
/// the unified Budget at the end of each pass.
#[derive(Debug, Clone, Default)]
pub struct PassStats {
    /// Number of rewrites applied in this pass
    pub rewrite_count: u64,
    /// Delta in nodes created during this pass
    pub nodes_delta: u64,
    /// The operation type for this pass (Core or Transform)
    pub op: Operation,
    /// If set, the pass hit an internal limit and stopped early
    pub stop_reason: Option<BudgetExceeded>,
}

// =============================================================================
// Budget struct (array-indexed for O(1) performance)
// =============================================================================

/// Unified budget configuration and runtime state.
///
/// Uses fixed-size arrays indexed by `Operation` and `Metric` for O(1) access
/// with zero allocation overhead.
#[derive(Debug, Clone)]
pub struct Budget {
    /// Limits per (operation, metric). 0 = unlimited.
    limits: [[u64; Metric::COUNT]; Operation::COUNT],
    /// Usage counters per (operation, metric).
    used: [[u64; Metric::COUNT]; Operation::COUNT],
    /// Current operation (for `charge_current`)
    current_op: Operation,
    /// Whether to enforce limits strictly (Err on exceed) or best-effort (silent cap)
    strict: bool,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            limits: [[0; Metric::COUNT]; Operation::COUNT],
            used: [[0; Metric::COUNT]; Operation::COUNT],
            current_op: Operation::SimplifyCore,
            strict: true,
        }
    }
}

impl Budget {
    /// Create a new budget with no limits (all zeros = unlimited).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a budget in strict mode (default) or best-effort mode.
    pub fn with_strict(strict: bool) -> Self {
        Self {
            strict,
            ..Self::default()
        }
    }

    /// Set a limit for a specific (operation, metric) pair.
    ///
    /// A limit of 0 means unlimited.
    #[inline]
    pub fn set_limit(&mut self, op: Operation, metric: Metric, limit: u64) {
        self.limits[op.index()][metric.index()] = limit;
    }

    /// Get the limit for a specific (operation, metric) pair.
    #[inline]
    pub fn limit(&self, op: Operation, metric: Metric) -> u64 {
        self.limits[op.index()][metric.index()]
    }

    /// Get current usage for a specific (operation, metric) pair.
    #[inline]
    pub fn used(&self, op: Operation, metric: Metric) -> u64 {
        self.used[op.index()][metric.index()]
    }

    /// Reset all usage counters to zero.
    pub fn reset(&mut self) {
        self.used = [[0; Metric::COUNT]; Operation::COUNT];
    }

    /// Charge a specific amount to an (operation, metric) pair.
    ///
    /// Returns `Err(BudgetExceeded)` if strict mode and limit exceeded.
    /// In best-effort mode, returns `Ok(())` but stops incrementing at limit.
    #[inline]
    pub fn charge(
        &mut self,
        op: Operation,
        metric: Metric,
        amount: u64,
    ) -> Result<(), BudgetExceeded> {
        let limit = self.limits[op.index()][metric.index()];
        let used = &mut self.used[op.index()][metric.index()];

        *used = used.saturating_add(amount);

        if limit > 0 && *used > limit {
            if self.strict {
                return Err(BudgetExceeded {
                    op,
                    metric,
                    used: *used,
                    limit,
                });
            }
            // Best-effort: cap at limit, don't error
            *used = limit;
        }

        Ok(())
    }

    /// Charge to the current operation (set by scope).
    #[inline]
    pub fn charge_current(&mut self, metric: Metric, amount: u64) -> Result<(), BudgetExceeded> {
        self.charge(self.current_op, metric, amount)
    }

    /// Check if a charge would exceed the limit (without actually charging).
    #[inline]
    pub fn would_exceed(&self, op: Operation, metric: Metric, amount: u64) -> bool {
        let limit = self.limits[op.index()][metric.index()];
        if limit == 0 {
            return false; // Unlimited
        }
        let used = self.used[op.index()][metric.index()];
        used.saturating_add(amount) > limit
    }

    /// Create a scope for an operation. Restores previous operation on drop.
    pub fn scope(&mut self, op: Operation) -> BudgetScope<'_> {
        let prev_op = self.current_op;
        self.current_op = op;
        BudgetScope {
            budget: self,
            prev_op,
        }
    }

    /// Get the current operation.
    #[inline]
    pub fn current_op(&self) -> Operation {
        self.current_op
    }

    /// Check if in strict mode.
    #[inline]
    pub fn is_strict(&self) -> bool {
        self.strict
    }
}

// =============================================================================
// BudgetScope (RAII guard)
// =============================================================================

/// RAII guard that tracks the current operation and restores it on drop.
pub struct BudgetScope<'a> {
    budget: &'a mut Budget,
    prev_op: Operation,
}

impl<'a> BudgetScope<'a> {
    /// Charge to this scope's operation.
    #[inline]
    pub fn charge(&mut self, metric: Metric, amount: u64) -> Result<(), BudgetExceeded> {
        self.budget.charge_current(metric, amount)
    }
}

impl<'a> Drop for BudgetScope<'a> {
    fn drop(&mut self) {
        self.budget.current_op = self.prev_op;
    }
}

// =============================================================================
// Default limits (production-ready values)
// =============================================================================

impl Budget {
    /// Create a budget with sensible production defaults.
    pub fn with_defaults() -> Self {
        let mut b = Self::new();

        // Simplify limits
        b.set_limit(Operation::SimplifyCore, Metric::RewriteSteps, 500);
        b.set_limit(Operation::SimplifyTransform, Metric::RewriteSteps, 200);

        // Expand limits
        b.set_limit(Operation::Expand, Metric::TermsMaterialized, 300);
        b.set_limit(Operation::MultinomialExpand, Metric::TermsMaterialized, 500);

        // Poly limits
        b.set_limit(Operation::ExprToPoly, Metric::TermsMaterialized, 200);
        b.set_limit(Operation::PolyOps, Metric::PolyOps, 1000);

        // GCD limits
        b.set_limit(Operation::GcdZippel, Metric::PolyOps, 500);

        b
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_charge_under_limit() {
        let mut budget = Budget::new();
        budget.set_limit(Operation::Expand, Metric::TermsMaterialized, 100);

        assert!(budget
            .charge(Operation::Expand, Metric::TermsMaterialized, 50)
            .is_ok());
        assert_eq!(
            budget.used(Operation::Expand, Metric::TermsMaterialized),
            50
        );
    }

    #[test]
    fn test_charge_exceeds_limit_strict() {
        let mut budget = Budget::new();
        budget.set_limit(Operation::Expand, Metric::TermsMaterialized, 100);

        let result = budget.charge(Operation::Expand, Metric::TermsMaterialized, 150);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert_eq!(err.op, Operation::Expand);
        assert_eq!(err.metric, Metric::TermsMaterialized);
        assert_eq!(err.used, 150);
        assert_eq!(err.limit, 100);
    }

    #[test]
    fn test_charge_exceeds_limit_best_effort() {
        let mut budget = Budget::with_strict(false);
        budget.set_limit(Operation::Expand, Metric::TermsMaterialized, 100);

        // Should not error in best-effort mode
        assert!(budget
            .charge(Operation::Expand, Metric::TermsMaterialized, 150)
            .is_ok());
        // But usage is capped at limit
        assert_eq!(
            budget.used(Operation::Expand, Metric::TermsMaterialized),
            100
        );
    }

    #[test]
    fn test_unlimited_when_zero() {
        let mut budget = Budget::new();
        // No limit set (0 = unlimited)

        assert!(budget
            .charge(Operation::Expand, Metric::TermsMaterialized, 1_000_000)
            .is_ok());
        assert_eq!(
            budget.used(Operation::Expand, Metric::TermsMaterialized),
            1_000_000
        );
    }

    #[test]
    fn test_scope_tracks_operation() {
        let mut budget = Budget::new();
        budget.set_limit(Operation::SimplifyCore, Metric::RewriteSteps, 10);

        assert_eq!(budget.current_op(), Operation::SimplifyCore);

        {
            let scope = budget.scope(Operation::Expand);
            // Can't read budget directly while scope holds mutable borrow
            // But we can verify via scope's internal reference
            assert_eq!(scope.budget.current_op(), Operation::Expand);
        }

        // Should restore after scope drops
        assert_eq!(budget.current_op(), Operation::SimplifyCore);
    }

    #[test]
    fn test_scope_charge_current() {
        let mut budget = Budget::new();
        budget.set_limit(Operation::Expand, Metric::TermsMaterialized, 100);

        {
            let mut scope = budget.scope(Operation::Expand);
            assert!(scope.charge(Metric::TermsMaterialized, 25).is_ok());
        }

        assert_eq!(
            budget.used(Operation::Expand, Metric::TermsMaterialized),
            25
        );
    }

    #[test]
    fn test_would_exceed() {
        let mut budget = Budget::new();
        budget.set_limit(Operation::PolyOps, Metric::PolyOps, 100);
        budget
            .charge(Operation::PolyOps, Metric::PolyOps, 80)
            .unwrap();

        assert!(!budget.would_exceed(Operation::PolyOps, Metric::PolyOps, 10));
        assert!(budget.would_exceed(Operation::PolyOps, Metric::PolyOps, 30));
    }

    #[test]
    fn test_reset() {
        let mut budget = Budget::new();
        budget
            .charge(Operation::Expand, Metric::TermsMaterialized, 100)
            .unwrap();
        assert_eq!(
            budget.used(Operation::Expand, Metric::TermsMaterialized),
            100
        );

        budget.reset();
        assert_eq!(budget.used(Operation::Expand, Metric::TermsMaterialized), 0);
    }

    #[test]
    fn test_accumulative_charge() {
        let mut budget = Budget::new();
        budget.set_limit(Operation::SimplifyCore, Metric::RewriteSteps, 100);

        budget
            .charge(Operation::SimplifyCore, Metric::RewriteSteps, 30)
            .unwrap();
        budget
            .charge(Operation::SimplifyCore, Metric::RewriteSteps, 30)
            .unwrap();
        budget
            .charge(Operation::SimplifyCore, Metric::RewriteSteps, 30)
            .unwrap();

        assert_eq!(
            budget.used(Operation::SimplifyCore, Metric::RewriteSteps),
            90
        );

        // This should exceed
        let result = budget.charge(Operation::SimplifyCore, Metric::RewriteSteps, 20);
        assert!(result.is_err());
    }

    #[test]
    fn test_default_limits() {
        let budget = Budget::with_defaults();

        assert_eq!(
            budget.limit(Operation::SimplifyCore, Metric::RewriteSteps),
            500
        );
        assert_eq!(
            budget.limit(Operation::Expand, Metric::TermsMaterialized),
            300
        );
        assert_eq!(budget.limit(Operation::GcdZippel, Metric::PolyOps), 500);
    }

    #[test]
    fn test_error_display() {
        let err = BudgetExceeded {
            op: Operation::Expand,
            metric: Metric::TermsMaterialized,
            used: 150,
            limit: 100,
        };

        let msg = format!("{}", err);
        assert!(msg.contains("Expand"));
        assert!(msg.contains("TermsMaterialized"));
        assert!(msg.contains("150"));
        assert!(msg.contains("100"));
    }
}
