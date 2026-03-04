/// Domain assumption mode for simplification.
///
/// Controls how transforms that require side-conditions behave:
/// - `Strict`: only proven-safe rewrites
/// - `Generic`: allow definability assumptions (classic CAS "almost everywhere")
/// - `Assume`: allow all side-conditions as assumptions
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DomainMode {
    /// No domain assumptions: only proven-safe simplifications.
    Strict,
    /// Use user-provided assumptions.
    Assume,
    /// "Almost everywhere" algebra (default).
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
    pub fn allows_unproven(self, class: crate::solve_safety_policy::ConditionClass) -> bool {
        use crate::solve_safety_policy::ConditionClass;
        match self {
            DomainMode::Strict => false,
            DomainMode::Generic => class == ConditionClass::Definability,
            DomainMode::Assume => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DomainMode;
    use crate::solve_safety_policy::ConditionClass;

    #[test]
    fn default_is_generic() {
        assert_eq!(DomainMode::default(), DomainMode::Generic);
    }

    #[test]
    fn mode_helper_flags_match_variants() {
        assert!(DomainMode::Strict.is_strict());
        assert!(DomainMode::Generic.is_generic());
        assert!(DomainMode::Assume.is_assume());
    }

    #[test]
    fn allows_unproven_contract_matches_expected_matrix() {
        assert!(!DomainMode::Strict.allows_unproven(ConditionClass::Definability));
        assert!(DomainMode::Generic.allows_unproven(ConditionClass::Definability));
        assert!(DomainMode::Assume.allows_unproven(ConditionClass::Definability));

        assert!(!DomainMode::Strict.allows_unproven(ConditionClass::Analytic));
        assert!(!DomainMode::Generic.allows_unproven(ConditionClass::Analytic));
        assert!(DomainMode::Assume.allows_unproven(ConditionClass::Analytic));
    }
}
