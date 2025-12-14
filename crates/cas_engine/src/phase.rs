//! Simplification phases for the phase-based pipeline.
//!
//! The simplifier executes rules in a fixed phase order:
//! 1. Core - Safe local simplifications
//! 2. Transform - Distribution, expansion, collection
//! 3. Rationalize - Automatic rationalization per policy
//! 4. PostCleanup - Final cleanup without expansion
//!
//! Key invariant: Transform never runs after Rationalize.

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
}

impl PhaseStats {
    pub fn new(phase: SimplifyPhase) -> Self {
        Self {
            phase: Some(phase),
            iters_used: 0,
            rewrites_used: 0,
            changed: false,
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
}

impl Default for SimplifyOptions {
    fn default() -> Self {
        Self {
            enable_transform: true,
            rationalize: crate::rationalize_policy::RationalizePolicy::default(),
            budgets: PhaseBudgets::default(),
            collect_steps: true,
        }
    }
}

impl SimplifyOptions {
    /// Options for `expand()` command: Core → Transform → PostCleanup (no Rationalize)
    pub fn for_expand() -> Self {
        let mut opt = Self::default();
        opt.rationalize.auto_level = crate::rationalize_policy::AutoRationalizeLevel::Off;
        opt
    }

    /// Options for `simplify --no-transform`
    pub fn no_transform() -> Self {
        let mut opt = Self::default();
        opt.enable_transform = false;
        opt
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
