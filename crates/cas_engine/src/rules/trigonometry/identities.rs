use crate::define_rule;
use crate::helpers::{extract_double_angle_arg, extract_triple_angle_arg, is_pi, is_pi_over_n};
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{Expr, ExprId};
use num_traits::{One, Zero};
use std::cmp::Ordering;

// Phase 6: Migrated to proper module (must be declared before core.rs include)
mod values_rules;
pub use values_rules::{
    CscCotPythagoreanRule, SecTanPythagoreanRule, TanToSinCosRule, TanTripleProductRule,
    TrigQuotientRule,
};
// Re-export helpers for use by other modules (core_rules, expansions.rs)
pub use values_rules::{has_large_coefficient, is_multiple_angle};

// Phase 7: Migrated to proper module
mod core_rules;
pub use core_rules::{
    AngleIdentityRule, EvaluateTrigRule, PythagoreanIdentityRule, SinCosIntegerPiRule,
    TrigOddEvenParityRule,
};

// Phase 5: Migrated to proper module (must be declared before expansion_rules)
mod power_products_rules;
pub use power_products_rules::{SinCosSumQuotientRule, TrigHiddenCubicIdentityRule};
// Re-export helpers for use by other modules (expansion_rules)
pub use power_products_rules::{
    build_avg, build_half_diff, extract_trig_arg, normalize_for_even_fn,
};

// Phase 8: Migrated to proper module
mod expansion_rules;
pub use expansion_rules::{
    CanonicalizeTrigSquareRule, DoubleAngleContractionRule, DoubleAngleRule, HalfAngleTangentRule,
    QuintupleAngleRule, RecursiveTrigExpansionRule, TrigSumToProductRule, TripleAngleRule,
};

// Phase 4: Migrated to proper module
mod sum_to_product_rules;
pub use sum_to_product_rules::{register, AngleConsistencyRule, DyadicCosProductToSinRule};

// Phase 1: Migrated to proper module
mod angle_expansion_rules;
pub use angle_expansion_rules::{ProductToSumRule, TrigPhaseShiftRule};
include!("identities/half_angle_phase.rs");

// Phase 2: Migrated to proper module
mod tan_half_angle_rules;
pub use tan_half_angle_rules::{
    GeneralizedSinCosContractionRule, HyperbolicHalfAngleSquaresRule,
    TanDoubleAngleContractionRule, TrigQuotientToNamedRule,
};

// Phase 3: Migrated to proper module
mod misc_rules;
pub use misc_rules::TrigSumToProductContractionRule;
