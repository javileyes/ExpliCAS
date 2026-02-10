//! # Fractions Module
//!
//! This module provides rules for fraction manipulation including:
//! - Simplification (SimplifyFractionRule, NestedFractionRule)
//! - Cancellation (CancelCommonFactorsRule, etc.)
//! - Combining (AddFractionsRule, FoldAddIntoFractionRule)
//! - Rationalization (RationalizeDenominatorRule, etc.)

// Core fraction helpers module
mod core_rules;

// core_rules helpers are imported directly by sibling modules via `super::core_rules::{...}`

// GCD-based cancellation rules module
mod gcd_cancel;
mod gcd_cancel_didactic;

// Re-export rules from gcd_cancel
pub use gcd_cancel::{
    CancelIdenticalFractionRule, CancelPowerFractionRule, CancelPowersDivisionRule,
    NestedFractionRule, SimplifyFractionRule, SimplifyMulDivRule,
};

// Addition rules module (fraction addition)
mod addition_rules;

// Re-export rules from addition_rules
pub use addition_rules::{
    AddFractionsRule, FoldAddIntoFractionRule, SubFractionsRule, SubTermMatchesDenomRule,
};

// Cancel rules module (rationalization and cancellation)
mod cancel_rules;

// Re-export rules from cancel_rules
pub use cancel_rules::{
    CancelNthRootBinomialFactorRule, GeneralizedRationalizationRule, RationalizeDenominatorRule,
    RationalizeNthRootBinomialRule, SqrtConjugateCollapseRule,
};

// Factor-based cancellation rules (extracted from cancel_rules)
mod cancel_rules_factor;

pub use cancel_rules_factor::{CancelCommonFactorsRule, RationalizeProductDenominatorRule};

// Properly modularized submodules (Phases 2-4)
mod more_rules;
mod rationalize;
mod small_rules;
mod tail;

// Re-export rules from modularized submodules
pub use more_rules::{
    AbsorbNegationIntoDifferenceRule, CanonicalDifferenceProductRule, RationalizeBinomialSurdRule,
};
pub use rationalize::{
    DivAddCommonFactorFromDenRule, DivAddSymmetricFactorRule, FactorBasedLCDRule,
    PullConstantFromFractionRule, QuotientOfPowersRule,
};
pub use small_rules::DivScalarIntoAddRule;
pub use small_rules::RationalizeSingleSurdRule;
pub use tail::CombineSameDenominatorFractionsRule;
