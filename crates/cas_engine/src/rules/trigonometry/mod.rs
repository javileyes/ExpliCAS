//! Trigonometry rules submodule
//!
//! This module contains data-driven trigonometric simplification rules.
//! - `values.rs`: Static lookup tables for special angle values
//! - `evaluation.rs`: Table-driven evaluation rule
//! - `identities.rs`: Pythagorean, angle sum/diff, double-angle, and other identity rules
//! - `pythagorean.rs`: Standalone Pythagorean identity simplification rule
//! - `pythagorean_secondary.rs`: Reciprocal conversions and even-power difference rules
//! - `weierstrass.rs`: Weierstrass substitution (t = tan(x/2))

pub mod evaluation;
pub mod identities;
pub mod pythagorean;
pub mod pythagorean_secondary;
pub(crate) mod trig_matchers;
pub mod values;
pub mod weierstrass;

pub use evaluation::EvaluateTrigTableRule;
pub use identities::*;
pub use pythagorean::{
    RecognizeCscSquaredRule, RecognizeSecSquaredRule, TrigPythagoreanChainRule,
    TrigPythagoreanGenericCoefficientRule, TrigPythagoreanHighPowerRule,
    TrigPythagoreanLinearFoldRule, TrigPythagoreanLocalCollectFoldRule,
    TrigPythagoreanSimplifyRule,
};
pub use pythagorean_secondary::{
    CotToCosSinRule, CscToRecipSinRule, SecToRecipCosRule, TrigEvenPowerDifferenceRule,
};
pub use values::{detect_special_angle, lookup_trig_value, SpecialAngle, TrigValue};
pub use weierstrass::{ReverseWeierstrassRule, WeierstrassSubstitutionRule};
