//! Mathematical soundness classification for transformations.

/// Mathematical soundness classification for rule transformations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SoundnessLabel {
    /// Equivalence for all values in the implicit domain of the input.
    #[default]
    Equivalence,
    /// Equivalence, but requires additional introduced conditions.
    EquivalenceUnderIntroducedRequires,
    /// The rule chooses one branch of a multi-valued expression.
    BranchChoice,
    /// Extends the value domain (for example R -> C).
    DomainExtension,
    /// Heuristic transform without global equivalence guarantee.
    Heuristic,
}

#[cfg(test)]
mod tests {
    use super::SoundnessLabel;

    #[test]
    fn default_label_is_equivalence() {
        assert_eq!(SoundnessLabel::default(), SoundnessLabel::Equivalence);
    }
}
