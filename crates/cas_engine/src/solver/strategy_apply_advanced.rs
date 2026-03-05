//! Advanced strategy facade.
//!
//! Keeps call sites stable while implementation is split by strategy family.

pub(super) use super::strategy_apply_roots::apply_rational_roots_strategy;
pub(super) use super::strategy_apply_subst_quad::{
    apply_quadratic_strategy, apply_substitution_strategy,
};
