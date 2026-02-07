pub mod equivalence;
pub mod evaluator;

mod hold;
mod orchestration;
mod simplifier;
mod transform;

pub use evaluator::{eval_f64, eval_f64_checked, EvalCheckedError, EvalCheckedOptions};
pub use hold::{strip_all_holds, substitute_expr_by_id};
pub use orchestration::LoopConfig;
pub use simplifier::Simplifier;

// =============================================================================
// EquivalenceResult: Tri-state result for equiv command
// =============================================================================
//
// V2.14.45: Proper equivalence checking with domain awareness.
// Returns conditional true when rules with EquivalenceUnderIntroducedRequires
// are used or when domain conditions are introduced.
// =============================================================================

/// Result of equivalence checking between two expressions.
#[derive(Debug, Clone)]
pub enum EquivalenceResult {
    /// A ≡ B unconditionally (no domain assumptions needed)
    True,
    /// A ≡ B under specified conditions (domain restrictions)
    ConditionalTrue {
        /// Requires conditions introduced during simplification
        requires: Vec<String>,
    },
    /// A ≢ B (found counterexample or proved non-equivalent)
    False,
    /// Cannot determine (no proof either way)
    Unknown,
}

impl EquivalenceResult {
    /// Returns true if the result indicates equivalence (True or ConditionalTrue)
    pub fn is_equivalent(&self) -> bool {
        matches!(
            self,
            EquivalenceResult::True | EquivalenceResult::ConditionalTrue { .. }
        )
    }
}
