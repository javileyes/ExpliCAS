//! Trigonometry rules submodule
//!
//! This module contains data-driven trigonometric simplification rules.
//! - `values.rs`: Static lookup tables for special angle values
//! - `evaluation.rs`: Table-driven evaluation rule
//! - `identities.rs`: Pythagorean, angle sum/diff, double-angle, and other identity rules

pub mod evaluation;
pub mod identities;
pub mod values;

pub use evaluation::EvaluateTrigTableRule;
pub use identities::*;
pub use values::{detect_special_angle, lookup_trig_value, SpecialAngle, TrigValue};
