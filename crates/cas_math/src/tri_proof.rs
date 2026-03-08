//! Three-valued proof status used by generic predicate helpers.

/// Conservative proof classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriProof {
    Proven,
    Disproven,
    Unknown,
}

impl TriProof {
    pub fn is_proven(self) -> bool {
        matches!(self, Self::Proven)
    }

    pub fn is_disproven(self) -> bool {
        matches!(self, Self::Disproven)
    }
}
