//! Tests de las reglas aritméticas, extraídos del módulo (P2).
//!
//! Vivían como `mod tests` inline dentro de `rules/arithmetic.rs`, donde
//! eran 8.524 de sus 39.346 líneas.
//!
//! No confundir con `rules/arithmetic_tests.rs`, que es un fichero aparte y
//! mucho más pequeño (145 líneas) que ya existía.

use super::{
    canonicalize_nested_integer_powers, exprs_equal_up_to_add_term_order,
    extract_scaled_double_sine_product_for_cancellation, term_has_matrix_product_factor,
    try_build_direct_sum_diff_cubes_quotient_equivalence_rewrite,
    try_rewrite_hyperbolic_angle_sum_diff_for_cancellation, CollapseExactOneShiftedQuotientRule,
    CollapseExactZeroCommonScaledDifferenceRule, CollapseExactZeroProductFactorRule,
    CollapseExactZeroThreeTermSubsetRule, ExpandHyperbolicAngleSumDiffToEnableCancellationRule,
    ExpandHyperbolicPythagoreanFactorToEnableCancellationRule,
    ExpandLogAbsMulDivToEnableCancellationRule, ExpandLogProductPowerToEnableCancellationRule,
    ExpandOddHalfPowerToEnableCancellationRule, ExpandTrigPhaseShiftToEnableCancellationRule,
    ExpandTrigSineProductTripleAngleToEnableCancellationRule,
    ExpandTrigSquareIdentityToEnableCancellationRule,
    ExpandTrigSumToProductToEnableCancellationRule, SubSelfToZeroRule,
    SubtractExpandedSumDiffCubesQuotientRule,
};
use crate::parent_context::ParentContext;
use crate::rule::Rule;
use crate::DomainMode;
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr};
use cas_formatter::DisplayExpr;
use cas_parser::parse;
use std::cmp::Ordering;

fn assert_empty_or_legacy_description(actual: &str, expected: &str) {
    assert!(
        actual.is_empty() || actual == expected,
        "expected direct collapse or legacy label {:?}, got {:?}",
        expected,
        actual
    );
}

mod fractions;
mod general;
mod hyperbolic;
mod logarithms;
mod pairing;
mod phase_shift;
mod powers;
mod solve_prep;
mod trig;
mod trig_angles;
mod zero_collapse;
