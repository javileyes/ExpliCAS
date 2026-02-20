use cas_ast::ExprId;

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
