use crate::define_rule;
use crate::helpers::{extract_double_angle_arg, extract_triple_angle_arg, is_pi, is_pi_over_n};
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{Expr, ExprId};
use num_traits::{One, Zero};
use std::cmp::Ordering;

include!("identities/core.rs");
include!("identities/values.rs");
include!("identities/power_products.rs");
include!("identities/expansions.rs");

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
