//! Trigonometry rules submodule
//!
//! This module contains data-driven trigonometric simplification rules.
//! - `values.rs`: Static lookup tables for special angle values
//! - `evaluation.rs`: Table-driven evaluation rule
//! - `identities.rs`: Pythagorean, angle sum/diff, double-angle, and other identity rules
//! - `pythagorean.rs`: Standalone Pythagorean identity simplification rule
//! - `weierstrass.rs`: Weierstrass substitution (t = tan(x/2))

pub mod evaluation;
pub mod identities;
pub mod pythagorean;
pub(crate) mod trig_matchers;
pub mod values;
pub mod weierstrass;

pub use evaluation::EvaluateTrigTableRule;
pub use identities::*;
pub use pythagorean::{
    CotToCosSinRule, CscToRecipSinRule, RecognizeCscSquaredRule, RecognizeSecSquaredRule,
    SecToRecipCosRule, TrigEvenPowerDifferenceRule, TrigPythagoreanChainRule,
    TrigPythagoreanGenericCoefficientRule, TrigPythagoreanLinearFoldRule,
    TrigPythagoreanLocalCollectFoldRule, TrigPythagoreanSimplifyRule,
};
pub use values::{detect_special_angle, lookup_trig_value, SpecialAngle, TrigValue};
pub use weierstrass::{ReverseWeierstrassRule, WeierstrassSubstitutionRule};
