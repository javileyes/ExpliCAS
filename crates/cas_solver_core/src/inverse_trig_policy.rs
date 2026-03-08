/// Policy for inverse-function compositions like `arctan(tan(x))`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum InverseTrigPolicy {
    /// Do not simplify inverse compositions.
    #[default]
    Strict,

    /// Simplify assuming principal-value domain with warning.
    PrincipalValue,
}

impl InverseTrigPolicy {
    /// Returns true for strict mode.
    #[inline]
    pub fn is_strict(self) -> bool {
        matches!(self, Self::Strict)
    }
}

#[cfg(test)]
mod tests {
    use super::InverseTrigPolicy;

    #[test]
    fn default_is_strict() {
        assert_eq!(InverseTrigPolicy::default(), InverseTrigPolicy::Strict);
    }

    #[test]
    fn strict_helper_matches_variant() {
        assert!(InverseTrigPolicy::Strict.is_strict());
        assert!(!InverseTrigPolicy::PrincipalValue.is_strict());
    }
}
