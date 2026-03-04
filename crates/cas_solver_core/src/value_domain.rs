/// Value domain for constant evaluation.
///
/// Controls the universe of values (R vs C).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum ValueDomain {
    /// Real numbers extended with +-infinity and undefined.
    /// `sqrt(-1)` -> `undefined`
    #[default]
    RealOnly,

    /// Complex numbers with principal branch.
    /// `sqrt(-1)` -> `i`
    ComplexEnabled,
}

impl ValueDomain {
    /// Returns true when domain is real-only.
    #[inline]
    pub fn is_real_only(self) -> bool {
        matches!(self, Self::RealOnly)
    }

    /// Returns true when complex values are enabled.
    #[inline]
    pub fn is_complex_enabled(self) -> bool {
        matches!(self, Self::ComplexEnabled)
    }
}

#[cfg(test)]
mod tests {
    use super::ValueDomain;

    #[test]
    fn default_is_real_only() {
        assert_eq!(ValueDomain::default(), ValueDomain::RealOnly);
    }

    #[test]
    fn helper_flags_match_variants() {
        assert!(ValueDomain::RealOnly.is_real_only());
        assert!(!ValueDomain::RealOnly.is_complex_enabled());
        assert!(ValueDomain::ComplexEnabled.is_complex_enabled());
        assert!(!ValueDomain::ComplexEnabled.is_real_only());
    }
}
