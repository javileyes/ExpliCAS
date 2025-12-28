//! Simplification phases for the phase-based pipeline.
//!
//! The simplifier executes rules in a fixed phase order:
//! 1. Core - Safe local simplifications
//! 2. Transform - Distribution, expansion, collection
//! 3. Rationalize - Automatic rationalization per policy
//! 4. PostCleanup - Final cleanup without expansion
//!
//! Key invariant: Transform never runs after Rationalize.

use bitflags::bitflags;

bitflags! {
    /// Bitmask indicating which phases a rule is allowed to run in.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PhaseMask: u8 {
        /// Core phase: safe, non-growing simplifications
        const CORE = 1 << 0;
        /// Transform phase: distribution, expansion, collection
        const TRANSFORM = 1 << 1;
        /// Rationalize phase: rationalization rules
        const RATIONALIZE = 1 << 2;
        /// PostCleanup phase: final cleanup
        const POST = 1 << 3;
        /// All phases
        const ALL = Self::CORE.bits() | Self::TRANSFORM.bits()
                  | Self::RATIONALIZE.bits() | Self::POST.bits();
    }
}

/// Phase of the simplification pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimplifyPhase {
    /// Core simplifications: identities, constant folding, basic rewrites.
    /// Safe to run at any time, never causes expression growth.
    Core,

    /// Transform phase: distribution, expansion, collection, polynomial strategies.
    /// May cause expression growth but enables further simplification.
    Transform,

    /// Rationalization phase: auto-rationalize denominators per policy.
    /// Runs after Transform to avoid re-expansion of rationalized results.
    Rationalize,

    /// Post-cleanup phase: final safe simplifications.
    /// Same rules as Core, but runs after Rationalize to clean up.
    PostCleanup,
}

impl SimplifyPhase {
    /// Convert phase to its corresponding mask bit.
    pub fn mask(&self) -> PhaseMask {
        match self {
            Self::Core => PhaseMask::CORE,
            Self::Transform => PhaseMask::TRANSFORM,
            Self::Rationalize => PhaseMask::RATIONALIZE,
            Self::PostCleanup => PhaseMask::POST,
        }
    }

    /// Returns true if this phase allows distribution/expansion rules.
    pub fn allows_distribution(&self) -> bool {
        matches!(self, Self::Transform)
    }

    /// Returns true if this phase allows rationalization rules.
    pub fn allows_rationalization(&self) -> bool {
        matches!(self, Self::Rationalize)
    }

    /// Returns true if this phase runs core (safe, non-growing) rules.
    pub fn is_core_phase(&self) -> bool {
        matches!(self, Self::Core | Self::PostCleanup)
    }

    /// Returns the next phase in the pipeline, or None if this is the last.
    pub fn next(&self) -> Option<Self> {
        match self {
            Self::Core => Some(Self::Transform),
            Self::Transform => Some(Self::Rationalize),
            Self::Rationalize => Some(Self::PostCleanup),
            Self::PostCleanup => None,
        }
    }

    /// Returns all phases in pipeline order.
    pub fn all() -> &'static [SimplifyPhase] {
        &[
            SimplifyPhase::Core,
            SimplifyPhase::Transform,
            SimplifyPhase::Rationalize,
            SimplifyPhase::PostCleanup,
        ]
    }
}

impl std::fmt::Display for SimplifyPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Core => write!(f, "Core"),
            Self::Transform => write!(f, "Transform"),
            Self::Rationalize => write!(f, "Rationalize"),
            Self::PostCleanup => write!(f, "PostCleanup"),
        }
    }
}

/// Statistics from a single phase execution.
#[derive(Debug, Clone, Default)]
pub struct PhaseStats {
    /// Which phase was executed
    pub phase: Option<SimplifyPhase>,
    /// Number of iterations used (out of budget)
    pub iters_used: usize,
    /// Number of rewrites (steps) applied
    pub rewrites_used: usize,
    /// Whether the expression changed
    pub changed: bool,
    /// Cycle detected (ping-pong) - if Some, phase was stopped early
    pub cycle: Option<crate::cycle_detector::CycleInfo>,
}

impl PhaseStats {
    pub fn new(phase: SimplifyPhase) -> Self {
        Self {
            phase: Some(phase),
            iters_used: 0,
            rewrites_used: 0,
            changed: false,
            cycle: None,
        }
    }
}

/// Statistics from the entire pipeline execution.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    pub core: PhaseStats,
    pub transform: PhaseStats,
    pub rationalize: PhaseStats,
    pub post_cleanup: PhaseStats,
    pub total_rewrites: usize,
    /// Outcome of rationalization attempt (if phase was run)
    pub rationalize_outcome: Option<crate::rationalize_policy::RationalizeOutcome>,
    /// The level that was attempted
    pub rationalize_level: Option<crate::rationalize_policy::AutoRationalizeLevel>,
    /// Collected assumptions (deduplicated, with counts)
    pub assumptions: Vec<crate::assumptions::AssumptionRecord>,
}

/// Iteration budgets for each phase of the pipeline.
///
/// These control how many fixed-point iterations each phase can run
/// before moving to the next phase.
#[derive(Debug, Clone, Copy)]
pub struct PhaseBudgets {
    /// Max iterations for Core phase (default: 8)
    pub core_iters: usize,
    /// Max iterations for Transform phase (default: 6)
    pub transform_iters: usize,
    /// Max iterations for Rationalize phase (default: 3)
    pub rationalize_iters: usize,
    /// Max iterations for PostCleanup phase (default: 4)
    pub post_iters: usize,
    /// Global safety limit on total rewrites (default: 200)
    pub max_total_rewrites: usize,
}

impl Default for PhaseBudgets {
    fn default() -> Self {
        Self {
            core_iters: 8,
            transform_iters: 6,
            rationalize_iters: 3,
            post_iters: 4,
            max_total_rewrites: 200,
        }
    }
}

impl PhaseBudgets {
    /// Get the iteration budget for a specific phase.
    pub fn for_phase(&self, phase: SimplifyPhase) -> usize {
        match phase {
            SimplifyPhase::Core => self.core_iters,
            SimplifyPhase::Transform => self.transform_iters,
            SimplifyPhase::Rationalize => self.rationalize_iters,
            SimplifyPhase::PostCleanup => self.post_iters,
        }
    }
}

// =============================================================================
// Auto-expand policy (opt-in expansion for cheap cases)
// =============================================================================

/// Policy controlling automatic expansion during simplification.
///
/// This is part of the three-tier expansion system:
/// - `simplify()` with `Off`: Preserves factored forms like `(x+1)^n`
/// - `simplify()` with `Auto`: Expands cheap cases within budget limits
/// - `expand()`: Aggressively expands all polynomial powers
///
/// Default is `Off` (structure-preserving) to maintain solver-friendly forms.
///
/// # Example
/// ```text
/// # With Off (default)
/// (1+i)^2  →  (1+i)^2  (preserved)
///
/// # With Auto
/// (1+i)^2  →  2*i  (expanded, within budget)
/// ```
///
/// See also: [`ExpandBudget`] for budget configuration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExpandPolicy {
    /// Standard mode: never auto-expand. Preserves `(x+1)^n` forms.
    /// Use explicit `expand()` for expansion.
    #[default]
    Off,
    /// Automatically expand cheap polynomial powers within budget limits.
    /// Useful for identity tests and normalization.
    Auto,
}

/// Budget limits for auto-expansion to prevent explosion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpandBudget {
    /// Maximum exponent to auto-expand (e.g., 4 means (x+1)^4 is ok, ^5 is not)
    pub max_pow_exp: u32,
    /// Maximum terms in base (e.g., 4 = tetra-nomial max)
    pub max_base_terms: u32,
    /// Maximum terms in expanded result (prevents explosion)
    pub max_generated_terms: u32,
    /// Maximum number of variables in base expression
    pub max_vars: u32,
}

impl Default for ExpandBudget {
    fn default() -> Self {
        Self {
            max_pow_exp: 4,
            max_base_terms: 4,
            max_generated_terms: 300,
            max_vars: 4,
        }
    }
}

/// Options controlling the simplification pipeline.
#[derive(Debug, Clone)]
pub struct SimplifyOptions {
    /// Whether to run the Transform phase (distribution, expansion).
    /// Set to false for `simplify --no-transform`.
    pub enable_transform: bool,

    /// Rationalization policy (includes auto_level and budgets).
    /// Set auto_level to Off for `simplify --no-rationalize`.
    pub rationalize: crate::rationalize_policy::RationalizePolicy,

    /// Per-phase iteration budgets.
    pub budgets: PhaseBudgets,

    /// Whether to collect steps for timeline display.
    pub collect_steps: bool,

    /// Whether we're in "expand mode" - forces aggressive distribution.
    /// When true, bypasses educational guards that preserve factored forms.
    pub expand_mode: bool,

    /// Auto-expand policy: Off (default) or Auto (expand cheap cases).
    pub expand_policy: ExpandPolicy,

    /// Budget limits for auto-expansion (only used when expand_policy=Auto).
    pub expand_budget: ExpandBudget,

    /// Context mode (Standard, Solve, etc.) - Solve mode blocks auto-expand.
    pub context_mode: crate::options::ContextMode,

    /// Domain assumption mode: Strict, Assume, or Generic (default).
    ///
    /// - Strict: No domain assumptions, only proven-safe simplifications
    /// - Assume: Use user-provided assumptions (future)
    /// - Generic: Classic CAS behavior, "almost everywhere" algebra (default)
    pub domain: crate::domain::DomainMode,

    /// Inverse trig composition policy (arctan(tan(x)), etc.).
    ///
    /// - Strict: Do not simplify inverse compositions
    /// - PrincipalValue: Simplify with principal domain warning
    pub inv_trig: crate::semantics::InverseTrigPolicy,

    /// Value domain for constant evaluation (ℝ vs ℂ).
    pub value_domain: crate::semantics::ValueDomain,

    /// Branch policy for multi-valued functions (only if ComplexEnabled).
    pub branch: crate::semantics::BranchPolicy,

    /// Assumption reporting level (Off, Summary, Trace).
    pub assumption_reporting: crate::assumptions::AssumptionReporting,
}

impl Default for SimplifyOptions {
    fn default() -> Self {
        Self {
            enable_transform: true,
            rationalize: crate::rationalize_policy::RationalizePolicy::default(),
            budgets: PhaseBudgets::default(),
            collect_steps: true,
            expand_mode: false,
            expand_policy: ExpandPolicy::default(),
            expand_budget: ExpandBudget::default(),
            context_mode: crate::options::ContextMode::default(),
            domain: crate::domain::DomainMode::default(), // Generic
            inv_trig: crate::semantics::InverseTrigPolicy::default(), // Strict
            value_domain: crate::semantics::ValueDomain::default(), // RealOnly
            branch: crate::semantics::BranchPolicy::default(), // Principal
            assumption_reporting: crate::assumptions::AssumptionReporting::Off, // Default to Off (conservador)
        }
    }
}

impl SimplifyOptions {
    /// Options for `expand()` command: Core → Transform → PostCleanup (no Rationalize)
    /// Forces aggressive distribution by setting expand_mode = true
    pub fn for_expand() -> Self {
        let mut opt = Self::default();
        opt.rationalize.auto_level = crate::rationalize_policy::AutoRationalizeLevel::Off;
        opt.expand_mode = true;
        opt
    }

    /// Options for `simplify --no-transform`
    pub fn no_transform() -> Self {
        Self {
            enable_transform: false,
            ..Default::default()
        }
    }

    /// Options for `simplify --no-rationalize`
    pub fn no_rationalize() -> Self {
        let mut opt = Self::default();
        opt.rationalize.auto_level = crate::rationalize_policy::AutoRationalizeLevel::Off;
        opt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_order() {
        assert_eq!(SimplifyPhase::Core.next(), Some(SimplifyPhase::Transform));
        assert_eq!(
            SimplifyPhase::Transform.next(),
            Some(SimplifyPhase::Rationalize)
        );
        assert_eq!(
            SimplifyPhase::Rationalize.next(),
            Some(SimplifyPhase::PostCleanup)
        );
        assert_eq!(SimplifyPhase::PostCleanup.next(), None);
    }

    #[test]
    fn test_phase_properties() {
        assert!(SimplifyPhase::Core.is_core_phase());
        assert!(!SimplifyPhase::Core.allows_distribution());
        assert!(!SimplifyPhase::Core.allows_rationalization());

        assert!(!SimplifyPhase::Transform.is_core_phase());
        assert!(SimplifyPhase::Transform.allows_distribution());
        assert!(!SimplifyPhase::Transform.allows_rationalization());

        assert!(!SimplifyPhase::Rationalize.is_core_phase());
        assert!(!SimplifyPhase::Rationalize.allows_distribution());
        assert!(SimplifyPhase::Rationalize.allows_rationalization());

        assert!(SimplifyPhase::PostCleanup.is_core_phase());
        assert!(!SimplifyPhase::PostCleanup.allows_distribution());
    }

    #[test]
    fn test_all_phases() {
        let phases = SimplifyPhase::all();
        assert_eq!(phases.len(), 4);
        assert_eq!(phases[0], SimplifyPhase::Core);
        assert_eq!(phases[3], SimplifyPhase::PostCleanup);
    }
}
