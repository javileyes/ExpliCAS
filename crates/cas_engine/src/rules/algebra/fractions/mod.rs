//! # Fractions Module
//!
//! This module provides rules for fraction manipulation including:
//! - Simplification (SimplifyFractionRule, NestedFractionRule)
//! - Cancellation (CancelCommonFactorsRule, etc.)
//! - Combining (AddFractionsRule, FoldAddIntoFractionRule)
//! - Rationalization (RationalizeDenominatorRule, etc.)

// Fractions-specific helpers (Phase 1 of incremental migration)
// These are not imported yet - they will be used when individual files
// are converted to proper modules in Phase 2+
mod helpers;

// Core fraction rules module (Phase 5.1)
mod core_rules;

// Re-export rules and helpers from core_rules
pub use core_rules::{
    // Helpers for sibling modules
    build_mul_from_factors_a1,
    check_divisible_denominators,
    collect_mul_factors_int_pow,
    extract_as_fraction,
    is_pi_constant,
    is_trig_function_name,
    // Rules
    CancelIdenticalFractionRule,
    CancelPowerFractionRule,
    CancelPowersDivisionRule,
    NestedFractionRule,
    SimplifyFractionRule,
    SimplifyMulDivRule,
};

// Cancel rules module (Phase 5.2)
mod cancel_rules;

// Re-export rules from cancel_rules
pub use cancel_rules::{
    AddFractionsRule, CancelCommonFactorsRule, CancelNthRootBinomialFactorRule,
    FoldAddIntoFractionRule, GeneralizedRationalizationRule, RationalizeDenominatorRule,
    RationalizeNthRootBinomialRule, RationalizeProductDenominatorRule, SqrtConjugateCollapseRule,
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
