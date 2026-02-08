//! # Fractions Module
//!
//! This module provides rules for fraction manipulation including:
//! - Simplification (SimplifyFractionRule, NestedFractionRule)
//! - Cancellation (CancelCommonFactorsRule, etc.)
//! - Combining (AddFractionsRule, FoldAddIntoFractionRule)
//! - Rationalization (RationalizeDenominatorRule, etc.)

// Core fraction helpers module
mod core_rules;

// Re-export helpers from core_rules (used by sibling modules)
pub use core_rules::{
    build_mul_from_factors_a1, check_divisible_denominators, collect_mul_factors_int_pow,
    extract_as_fraction, is_pi_constant, is_trig_function_name,
};

// GCD-based cancellation rules module
mod gcd_cancel;

// Re-export rules from gcd_cancel
pub use gcd_cancel::{
    CancelIdenticalFractionRule, CancelPowerFractionRule, CancelPowersDivisionRule,
    NestedFractionRule, SimplifyFractionRule, SimplifyMulDivRule,
};

// Addition rules module (fraction addition)
mod addition_rules;

// Re-export rules from addition_rules
pub use addition_rules::{AddFractionsRule, FoldAddIntoFractionRule};

// Cancel rules module (rationalization and cancellation)
mod cancel_rules;

// Re-export rules from cancel_rules
pub use cancel_rules::{
    CancelCommonFactorsRule, CancelNthRootBinomialFactorRule, GeneralizedRationalizationRule,
    RationalizeDenominatorRule, RationalizeNthRootBinomialRule, RationalizeProductDenominatorRule,
    SqrtConjugateCollapseRule,
};

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
pub use small_rules::RationalizeSingleSurdRule;
pub use tail::CombineSameDenominatorFractionsRule;
