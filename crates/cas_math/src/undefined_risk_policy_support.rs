/// Domain-policy decision for cancellation rewrites that may hide undefined
/// subexpressions (e.g. `a-a -> 0`, `a+(-a) -> 0`).
///
/// The policy is intentionally conservative:
/// - Strict: block if undefined risk exists.
/// - Generic/Assume: allow.
pub fn allow_cancellation_with_undefined_risk_mode_flags(
    assume_mode: bool,
    strict_mode: bool,
    has_undefined_risk: bool,
) -> bool {
    // Assume takes precedence when both flags are accidentally true.
    if assume_mode {
        return true;
    }
    if strict_mode {
        return !has_undefined_risk;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::allow_cancellation_with_undefined_risk_mode_flags;

    #[test]
    fn strict_blocks_when_risky() {
        assert!(!allow_cancellation_with_undefined_risk_mode_flags(
            false, true, true
        ));
    }

    #[test]
    fn strict_allows_when_not_risky() {
        assert!(allow_cancellation_with_undefined_risk_mode_flags(
            false, true, false
        ));
    }

    #[test]
    fn generic_allows_even_when_risky() {
        assert!(allow_cancellation_with_undefined_risk_mode_flags(
            false, false, true
        ));
    }

    #[test]
    fn assume_allows_even_when_risky() {
        assert!(allow_cancellation_with_undefined_risk_mode_flags(
            true, false, true
        ));
    }

    #[test]
    fn assume_has_priority_over_strict() {
        assert!(allow_cancellation_with_undefined_risk_mode_flags(
            true, true, true
        ));
    }
}
