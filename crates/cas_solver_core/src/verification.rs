use cas_ast::{ExprId, SolutionSet};

/// Result of verifying a single solution.
#[derive(Debug, Clone)]
pub enum VerifyStatus {
    /// Solution verified: equation simplifies to 0 after substitution.
    Verified,
    /// Solution could not be verified (residual remains).
    Unverifiable {
        /// The residual expression that didn't simplify to 0.
        residual: ExprId,
        /// Human-readable reason.
        reason: String,
    },
    /// Solution type not checkable (intervals, AllReals, residual).
    NotCheckable {
        /// Reason why verification is not possible.
        reason: &'static str,
    },
}

/// Summary of verification for a solution set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifySummary {
    /// All solutions verified.
    AllVerified,
    /// Some solutions verified, some not.
    PartiallyVerified,
    /// No solutions verified.
    NoneVerified,
    /// Solution type not checkable.
    NotCheckable,
    /// Empty solution set (trivially verified).
    Empty,
}

/// Result of verifying an entire solution set.
#[derive(Debug, Clone)]
pub struct VerifyResult {
    /// Status for each discrete solution (if applicable).
    pub solutions: Vec<(ExprId, VerifyStatus)>,
    /// Overall summary.
    pub summary: VerifySummary,
    /// Guard under which verification was performed (for Conditional).
    pub guard_description: Option<String>,
}

/// Verify a solution set using a callback for each discrete candidate.
///
/// The callback is invoked only for `SolutionSet::Discrete` entries.
/// Non-discrete sets are mapped to `NotCheckable` summaries.
pub fn verify_solution_set_with<F>(solutions: &SolutionSet, verify_discrete: &mut F) -> VerifyResult
where
    F: FnMut(ExprId) -> VerifyStatus,
{
    match solutions {
        SolutionSet::Empty => VerifyResult {
            solutions: vec![],
            summary: VerifySummary::Empty,
            guard_description: None,
        },

        SolutionSet::Discrete(sols) => {
            let mut results = Vec::with_capacity(sols.len());
            let mut verified_count = 0;

            for &sol in sols {
                let status = verify_discrete(sol);
                if matches!(status, VerifyStatus::Verified) {
                    verified_count += 1;
                }
                results.push((sol, status));
            }

            VerifyResult {
                summary: discrete_summary(results.len(), verified_count),
                solutions: results,
                guard_description: None,
            }
        }

        SolutionSet::AllReals => VerifyResult {
            solutions: vec![],
            summary: VerifySummary::NotCheckable,
            guard_description: Some("not checkable (infinite set: all reals)".to_string()),
        },

        SolutionSet::Continuous(_interval) => VerifyResult {
            solutions: vec![],
            summary: VerifySummary::NotCheckable,
            guard_description: Some("not checkable (continuous interval)".to_string()),
        },

        SolutionSet::Union(_intervals) => VerifyResult {
            solutions: vec![],
            summary: VerifySummary::NotCheckable,
            guard_description: Some("not checkable (union of intervals)".to_string()),
        },

        SolutionSet::Residual(_expr) => VerifyResult {
            solutions: vec![],
            summary: VerifySummary::NotCheckable,
            guard_description: Some("unverifiable (residual expression)".to_string()),
        },

        SolutionSet::Conditional(cases) => {
            let mut all_results = Vec::new();
            let mut has_verified = false;
            let mut has_not_checkable = false;

            for case in cases {
                let case_result = verify_solution_set_with(&case.then.solutions, verify_discrete);

                match case_result.summary {
                    VerifySummary::AllVerified | VerifySummary::PartiallyVerified => {
                        has_verified = true;
                    }
                    VerifySummary::NotCheckable => {
                        has_not_checkable = true;
                    }
                    _ => {}
                }

                all_results.extend(case_result.solutions);
            }

            VerifyResult {
                solutions: all_results,
                summary: conditional_summary(has_verified, has_not_checkable),
                guard_description: None,
            }
        }
    }
}

/// Compute summary for discrete verification outcomes.
pub fn discrete_summary(total: usize, verified_count: usize) -> VerifySummary {
    if total == 0 {
        VerifySummary::Empty
    } else if verified_count == total {
        VerifySummary::AllVerified
    } else if verified_count > 0 {
        VerifySummary::PartiallyVerified
    } else {
        VerifySummary::NoneVerified
    }
}

/// Compute summary for aggregated conditional verification outcomes.
pub fn conditional_summary(has_verified: bool, has_not_checkable: bool) -> VerifySummary {
    if has_verified && !has_not_checkable {
        VerifySummary::AllVerified
    } else if has_verified {
        VerifySummary::PartiallyVerified
    } else if has_not_checkable {
        VerifySummary::NotCheckable
    } else {
        VerifySummary::NoneVerified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;

    #[test]
    fn test_discrete_summary() {
        assert_eq!(discrete_summary(0, 0), VerifySummary::Empty);
        assert_eq!(discrete_summary(3, 3), VerifySummary::AllVerified);
        assert_eq!(discrete_summary(3, 1), VerifySummary::PartiallyVerified);
        assert_eq!(discrete_summary(3, 0), VerifySummary::NoneVerified);
    }

    #[test]
    fn test_conditional_summary() {
        assert_eq!(conditional_summary(true, false), VerifySummary::AllVerified);
        assert_eq!(
            conditional_summary(true, true),
            VerifySummary::PartiallyVerified
        );
        assert_eq!(
            conditional_summary(false, true),
            VerifySummary::NotCheckable
        );
        assert_eq!(
            conditional_summary(false, false),
            VerifySummary::NoneVerified
        );
    }

    #[test]
    fn test_verify_solution_set_with_discrete() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let two = ctx.num(2);
        let set = SolutionSet::Discrete(vec![one, two]);
        let mut calls = 0usize;
        let mut verify = |_id: ExprId| {
            calls += 1;
            VerifyStatus::Verified
        };

        let result = verify_solution_set_with(&set, &mut verify);
        assert_eq!(calls, 2);
        assert_eq!(result.summary, VerifySummary::AllVerified);
        assert_eq!(result.solutions.len(), 2);
    }
}
