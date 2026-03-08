/// Scope for assumptions when `DomainMode = Assume`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum AssumeScope {
    /// Assume positivity/nonzero for real-domain operations.
    /// Error if operation requires promotion to complex.
    #[default]
    Real,

    /// Like `Real`, but if complex is needed, return residual + warning
    /// instead of hard error. Never implicitly promotes.
    Wildcard,
}

impl AssumeScope {
    /// Returns true when wildcard assumptions are enabled.
    #[inline]
    pub fn is_wildcard(self) -> bool {
        matches!(self, Self::Wildcard)
    }
}

#[cfg(test)]
mod tests {
    use super::AssumeScope;

    #[test]
    fn default_is_real() {
        assert_eq!(AssumeScope::default(), AssumeScope::Real);
    }

    #[test]
    fn wildcard_helper_matches_variant() {
        assert!(!AssumeScope::Real.is_wildcard());
        assert!(AssumeScope::Wildcard.is_wildcard());
    }
}
