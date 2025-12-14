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
