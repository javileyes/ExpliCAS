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

use super::helpers::*;

// Submodules - temporarily using include!() for incremental migration
include!("core.rs");
include!("cancel.rs");
include!("rationalize.rs");
include!("small_rules.rs");
include!("more_rules.rs");
include!("tail.rs");
