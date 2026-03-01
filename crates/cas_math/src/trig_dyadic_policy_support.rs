#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DyadicSinNonzeroPolicyDecision {
    /// Do not apply rewrite when `sin(theta) != 0` is not proven.
    Block,
    /// Apply rewrite; may require an assumption event.
    Apply { assume_sin_nonzero: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DyadicSinNonzeroMode {
    Assume,
    Strict,
    Generic,
}

fn dyadic_sin_nonzero_mode_from_flags(
    assume_mode: bool,
    strict_mode: bool,
) -> DyadicSinNonzeroMode {
    if assume_mode {
        DyadicSinNonzeroMode::Assume
    } else if strict_mode {
        DyadicSinNonzeroMode::Strict
    } else {
        DyadicSinNonzeroMode::Generic
    }
}

/// Decide domain policy for `2^n * ∏cos(2^k·θ) -> sin(2^n·θ)/sin(θ)`.
///
/// If `sin(theta) != 0` is already proven, all modes apply without assumptions.
/// Otherwise:
/// - Strict/Generic: block.
/// - Assume: apply and record assumption.
pub fn decide_dyadic_sin_nonzero_policy(
    assume_mode: bool,
    strict_mode: bool,
    sin_nonzero_proven: bool,
) -> DyadicSinNonzeroPolicyDecision {
    if sin_nonzero_proven {
        return DyadicSinNonzeroPolicyDecision::Apply {
            assume_sin_nonzero: false,
        };
    }

    let mode = dyadic_sin_nonzero_mode_from_flags(assume_mode, strict_mode);
    match mode {
        DyadicSinNonzeroMode::Assume => DyadicSinNonzeroPolicyDecision::Apply {
            assume_sin_nonzero: true,
        },
        DyadicSinNonzeroMode::Strict | DyadicSinNonzeroMode::Generic => {
            DyadicSinNonzeroPolicyDecision::Block
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{decide_dyadic_sin_nonzero_policy, DyadicSinNonzeroPolicyDecision};

    #[test]
    fn proven_nonzero_always_applies() {
        let out = decide_dyadic_sin_nonzero_policy(false, true, true);
        assert_eq!(
            out,
            DyadicSinNonzeroPolicyDecision::Apply {
                assume_sin_nonzero: false
            }
        );
    }

    #[test]
    fn strict_blocks_unproven() {
        let out = decide_dyadic_sin_nonzero_policy(false, true, false);
        assert_eq!(out, DyadicSinNonzeroPolicyDecision::Block);
    }

    #[test]
    fn generic_blocks_unproven() {
        let out = decide_dyadic_sin_nonzero_policy(false, false, false);
        assert_eq!(out, DyadicSinNonzeroPolicyDecision::Block);
    }

    #[test]
    fn assume_applies_unproven_with_assumption() {
        let out = decide_dyadic_sin_nonzero_policy(true, false, false);
        assert_eq!(
            out,
            DyadicSinNonzeroPolicyDecision::Apply {
                assume_sin_nonzero: true
            }
        );
    }

    #[test]
    fn assume_priority_over_strict() {
        let out = decide_dyadic_sin_nonzero_policy(true, true, false);
        assert_eq!(
            out,
            DyadicSinNonzeroPolicyDecision::Apply {
                assume_sin_nonzero: true
            }
        );
    }
}
