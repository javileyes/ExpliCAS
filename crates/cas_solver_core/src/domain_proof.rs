/// Result of attempting to prove a property about an expression.
///
/// Used by domain-aware simplification to decide whether operations
/// like `x/x -> 1` are safe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Proof {
    /// Property is provably true (e.g., `2 != 0` is proven)
    Proven,
    /// Property is implied by expression structure (e.g., `sqrt(x)` implies `x >= 0`).
    /// Only valid when witness survives in output.
    ProvenImplicit,
    /// Property status is unknown (e.g., we don't know if `x != 0`)
    Unknown,
    /// Property is provably false (e.g., `0 != 0` is disproven)
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
    use super::Proof;

    #[test]
    fn proof_helpers_match_variants() {
        assert!(Proof::Proven.is_proven());
        assert!(!Proof::Proven.is_unknown());
        assert!(!Proof::Proven.is_disproven());

        assert!(!Proof::Unknown.is_proven());
        assert!(Proof::Unknown.is_unknown());
        assert!(!Proof::Unknown.is_disproven());

        assert!(!Proof::Disproven.is_proven());
        assert!(!Proof::Disproven.is_unknown());
        assert!(Proof::Disproven.is_disproven());
    }
}
