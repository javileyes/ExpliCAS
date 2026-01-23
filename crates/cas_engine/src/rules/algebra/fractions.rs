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

include!("fractions/01_core.rs");
include!("fractions/02_cancel.rs");
include!("fractions/03_rationalize.rs");
include!("fractions/04_small_rules.rs");
include!("fractions/05_more_rules.rs");
include!("fractions/06_tail.rs");
