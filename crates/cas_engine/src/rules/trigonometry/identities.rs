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
include!("identities/sum_to_product.rs");
include!("identities/angle_expansion.rs");
include!("identities/half_angle_phase.rs");
include!("identities/tan_half_angle.rs");
include!("identities/misc.rs");
