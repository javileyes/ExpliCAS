use crate::define_rule;
use crate::helpers::{extract_double_angle_arg, extract_triple_angle_arg, is_pi, is_pi_over_n};
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{Expr, ExprId};
use num_traits::{One, Zero};
use std::cmp::Ordering;

include!("identities/01_core.rs");
include!("identities/02_values.rs");
include!("identities/03_power_products.rs");
include!("identities/04_expansions_and_tests.rs");
include!("identities/05_sum_to_product.rs");
include!("identities/06_angle_expansion.rs");
include!("identities/07_half_angle_phase.rs");
include!("identities/08_tan_half_angle.rs");
include!("identities/09_misc_and_tests.rs");
