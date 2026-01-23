//! # Fractions Module
//!
//! This module provides rules for fraction manipulation including:
//! - Simplification (SimplifyFractionRule, NestedFractionRule)
//! - Cancellation (CancelCommonFactorsRule, etc.)
//! - Combining (AddFractionsRule, FoldAddIntoFractionRule)
//! - Rationalization (RationalizeDenominatorRule, etc.)

use crate::build::mul2_raw;
use crate::define_rule;
use crate::multipoly::{
    gcd_multivar_layer2, gcd_multivar_layer25, multipoly_from_expr, multipoly_to_expr, GcdBudget,
    GcdLayer, Layer25Budget, MultiPoly, PolyBudget,
};
use crate::phase::PhaseMask;
use crate::polynomial::Polynomial;
use crate::rule::{ChainedRewrite, Rewrite};
use cas_ast::count_nodes;
use cas_ast::{Context, DisplayExpr, Expr, ExprId};
use num_traits::{One, Signed, Zero};
use std::cmp::Ordering;

// Import shared algebra-level helpers
use super::helpers::*;

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

// Cancel rules - still using include!() until core_rules migration is verified
include!("cancel.rs");

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
