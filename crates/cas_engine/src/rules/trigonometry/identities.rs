//! Trigonometric identity rules for simplification and transformation.
//!
//! This module re-exports all identity rules from specialized submodules.

// --- Core evaluation and fundamental identities ---
mod core_rules;
pub use core_rules::{
    AngleIdentityRule, EvaluateTrigRule, PythagoreanIdentityRule, SinCosIntegerPiRule,
    TrigOddEvenParityRule,
};

mod values_rules;
pub use values_rules::{
    CscCotPythagoreanRule, SecTanPythagoreanRule, TanToSinCosRule, TanTripleProductRule,
    TrigQuotientRule,
};

// --- Angle expansion and contraction ---
mod expansion_rules;
pub use expansion_rules::{DoubleAngleRule, TrigSumToProductRule};

mod contraction_rules;
pub use contraction_rules::{
    AngleSumFractionToTanRule, Cos2xAdditiveContractionRule, DoubleAngleContractionRule,
    HalfAngleTangentRule,
};

mod multi_angle_rules;
pub use multi_angle_rules::{
    CanonicalizeTrigSquareRule, QuintupleAngleRule, RecursiveTrigExpansionRule,
    TripleAngleContractionRule, TripleAngleRule,
};

mod angle_expansion_rules;
pub use angle_expansion_rules::{ProductToSumRule, TrigPhaseShiftRule};

// --- Sum-to-product and product-to-sum ---
mod sum_to_product_rules;
pub use sum_to_product_rules::{register, AngleConsistencyRule, DyadicCosProductToSinRule};

mod power_products_rules;
pub use power_products_rules::{
    SinCosQuarticSumRule, SinCosSumQuotientRule, TrigHiddenCubicIdentityRule,
};

// --- Phase shift and supplementary angle ---
mod phase_shift_rules;
pub use phase_shift_rules::SinSupplementaryAngleRule;

// --- Half-angle and Weierstrass substitution ---
mod half_angle_phase_rules;
pub use half_angle_phase_rules::{
    CotHalfAngleDifferenceRule, HyperbolicTanhPythRule, TanDifferenceRule,
    WeierstrassContractionRule,
};

mod identity_zero_rules;
pub use identity_zero_rules::{
    Sin4xIdentityZeroRule, TanDifferenceIdentityZeroRule, WeierstrassCosIdentityZeroRule,
    WeierstrassSinIdentityZeroRule,
};

mod tan_half_angle_rules;
pub use tan_half_angle_rules::{
    GeneralizedSinCosContractionRule, HyperbolicHalfAngleSquaresRule,
    TanDoubleAngleContractionRule, TrigHalfAngleSquaresRule, TrigQuotientToNamedRule,
};

// --- Miscellaneous identity rules ---
mod misc_rules;
pub use misc_rules::TrigSumToProductContractionRule;
