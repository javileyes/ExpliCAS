//! Shared simplification phase model.
//!
//! Runtime crates can consume this module to avoid duplicating phase ordering
//! and phase-mask semantics.

use bitflags::bitflags;

bitflags! {
    /// Bitmask indicating which phases a rule is allowed to run in.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PhaseMask: u8 {
        /// Core phase: safe, non-growing simplifications.
        const CORE = 1 << 0;
        /// Transform phase: distribution, expansion, collection.
        const TRANSFORM = 1 << 1;
        /// Rationalize phase: rationalization rules.
        const RATIONALIZE = 1 << 2;
        /// PostCleanup phase: final cleanup.
        const POST = 1 << 3;
        /// All phases.
        const ALL = Self::CORE.bits() | Self::TRANSFORM.bits()
                  | Self::RATIONALIZE.bits() | Self::POST.bits();
    }
}

/// Phase of the simplification pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimplifyPhase {
    /// Core simplifications: safe, non-growth rewrites.
    Core,
    /// Transform phase: may increase size but exposes simplifications.
    Transform,
    /// Rationalization phase: denominator rationalization.
    Rationalize,
    /// Post-cleanup phase: final safe simplifications.
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

#[cfg(test)]
mod tests {
    use super::{PhaseMask, SimplifyPhase};

    #[test]
    fn phase_order_is_stable() {
        let all = SimplifyPhase::all();
        assert_eq!(
            all,
            &[
                SimplifyPhase::Core,
                SimplifyPhase::Transform,
                SimplifyPhase::Rationalize,
                SimplifyPhase::PostCleanup
            ]
        );
    }

    #[test]
    fn next_phase_transitions_are_correct() {
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
    fn mask_mapping_is_correct() {
        assert_eq!(SimplifyPhase::Core.mask(), PhaseMask::CORE);
        assert_eq!(SimplifyPhase::Transform.mask(), PhaseMask::TRANSFORM);
        assert_eq!(SimplifyPhase::Rationalize.mask(), PhaseMask::RATIONALIZE);
        assert_eq!(SimplifyPhase::PostCleanup.mask(), PhaseMask::POST);
    }
}
