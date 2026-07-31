//! Symbolic integration helpers shared by integration-facing rule layers.

use crate::build::mul2_raw;
use crate::calculus_domain_support::real_domain_is_empty_or_nonfinite_over_reals;
use crate::cancel_support::collect_additive_terms_signed;
use crate::expr_extract::extract_abs_argument_view;
use crate::expr_nary::{build_balanced_add, build_balanced_mul, mul_leaves, AddView, Sign};
use crate::expr_predicates::{contains_named_var, contains_variable};
use crate::factor::factor;
use crate::general_integration_backend::{
    verify_antiderivative_by_differentiation, AlgorithmicIntegrationCandidate,
};
use crate::polynomial::Polynomial;
use crate::root_forms::{try_rewrite_simplify_square_root_expr, SimplifySquareRootRewriteKind};
use crate::symbolic_integration_derivative_cofactor_support::{
    additive_cofactor_from_term_cofactors, factor_product_excluding_index, factor_product_or_one,
    factors_excluding_index, factors_excluding_two_indices, indexed_signed_matching_unary_factor,
    product_cofactor_excluding_unary_builtin_arg, signed_factor_product_excluding_index,
    unique_product_cofactor_excluding_unary_builtin_arg,
};
use crate::symbolic_integration_hyperbolic_policy::signed_hyperbolic_like_factor;
use crate::symbolic_integration_hyperbolic_reciprocal_policy::{
    build_hyperbolic_denominator_nonzero_condition,
    build_hyperbolic_reciprocal_derivative_integral, build_hyperbolic_reciprocal_table_integral,
    hyperbolic_reciprocal_derivative_policy, hyperbolic_reciprocal_table_policy,
    hyperbolic_tangent_arg, indexed_hyperbolic_reciprocal_derivative_numerator_factor,
    indexed_hyperbolic_tangent_factor_arg, indexed_reciprocal_hyperbolic_square_parts,
    reciprocal_hyperbolic_power_arg, reciprocal_hyperbolic_square_parts,
    HyperbolicReciprocalPrimitiveScaleOps,
};
use crate::symbolic_integration_log_support::{
    affine_constant_base_log_antiderivative_from_slope, constant_base_log_derivative_correction,
    ln_abs, positive_integer_constant_log_base_derivative_correction,
    positive_integer_constant_log_base_ln, valid_constant_log_base_ln_from_rational_value,
};
use crate::symbolic_integration_polynomial_support::{
    constant_polynomial_ratio, elementary_polynomial_substitution_kernel_antiderivative,
    polynomial_substitution_kernel, PolynomialSubstitutionKernel,
};
use crate::symbolic_integration_reciprocal_trig_policy::{
    build_reciprocal_trig_denominator_nonzero_condition, build_reciprocal_trig_derivative_integral,
    build_reciprocal_trig_log_argument, build_trig_pole_nonzero_condition,
    has_trig_pole_builtin_factor_except, indexed_reciprocal_trig_denominator_call,
    indexed_reciprocal_trig_square_parts, indexed_trig_log_derivative_numerator_factor,
    indexed_trig_log_derivative_raw_numerator_factor, indexed_trig_pole_builtin_factor,
    is_reciprocal_trig_call, reciprocal_trig_denominator_call,
    reciprocal_trig_derivative_base_antiderivative, reciprocal_trig_derivative_policy,
    reciprocal_trig_derivative_policy_from_reciprocal,
    reciprocal_trig_reciprocal_parts_from_denominator, reciprocal_trig_square_parts,
    trig_log_derivative_numerator_builtin, trig_pole_nonzero_builtin,
};
use crate::symbolic_integration_trig_policy::{signed_trig_like_factor, trig_like_factor};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::cmp::Ordering;

type LinearPartialFractionTerms = Vec<(BigRational, Polynomial, usize)>;

const MAX_EXP_POLYNOMIAL_BY_PARTS_DEGREE: usize = 8;
const MAX_TRIG_POLYNOMIAL_BY_PARTS_DEGREE: usize = 8;
const MAX_HYPERBOLIC_POLYNOMIAL_BY_PARTS_DEGREE: usize = 8;
const SYMBOLIC_INTEGRATION_DOMAIN_PROOF_DEPTH: usize = 8;
const SYMBOLIC_INTEGRATION_DOMAIN_SCAN_DEPTH: usize = 24;

#[derive(Clone)]
struct SqrtLinearDenominator {
    scale: BigRational,
    slope: BigRational,
    offset: BigRational,
}

#[derive(Clone, Copy)]
enum SymbolicSquareShiftArgument {
    DivideByParameter,
    MultiplyByParameter,
}

struct SymbolicSquareShiftDenominator {
    scale: BigRational,
    parameter: ExprId,
    argument: SymbolicSquareShiftArgument,
    argument_scale: BigRational,
}

struct PositiveSquareShiftDenominator {
    scale: BigRational,
    offset: BigRational,
    offset_root: BigRational,
}

struct ArctanSqrtAffineDerivativeParts {
    radicand: ExprId,
    scale: BigRational,
    argument_scale: BigRational,
}

struct ArctanSqrtAffineGapParts {
    argument_scale: BigRational,
    denominator_root: BigRational,
    kernel_scale_factor: BigRational,
}

#[derive(Clone, Copy)]
enum InverseHyperbolicSqrtReciprocalKind {
    Asinh,
    Atanh,
}

struct InverseHyperbolicSqrtReciprocalParts {
    base: Polynomial,
    constant: BigRational,
    scale: BigRational,
    scale_sqrt_factor: Option<BigRational>,
}

struct TrigRatioSquareParts {
    builtin: BuiltinFn,
    arg: ExprId,
    a: BigRational,
}

struct ReciprocalTrigPowerAffineParts {
    arg: ExprId,
    a: BigRational,
}

struct TrigPowerQuotientParts {
    arg: ExprId,
    a: BigRational,
}

struct ReciprocalTrigPowerQuotientParts {
    arg: ExprId,
    a: BigRational,
}

struct HyperbolicReciprocalDerivativeParts {
    denominator_builtin: BuiltinFn,
    arg: ExprId,
    scale: ExprId,
}

struct SqrtTrigReciprocalDerivativeParts {
    denominator_builtin: BuiltinFn,
    arg: ExprId,
    radicand: ExprId,
    scale: ExprId,
}

struct SqrtTrigLogDerivativeParts {
    denominator_builtin: BuiltinFn,
    arg: ExprId,
    radicand: ExprId,
    scale: BigRational,
}

struct SqrtReciprocalTrigLogDerivativeParts {
    denominator_builtin: BuiltinFn,
    arg: ExprId,
    radicand: ExprId,
    scale: BigRational,
}

struct SqrtHyperbolicLogDerivativeParts {
    log_builtin: BuiltinFn,
    arg: ExprId,
    radicand: ExprId,
    scale: BigRational,
}

struct SqrtHyperbolicReciprocalParts {
    denominator_builtin: BuiltinFn,
    arg: ExprId,
    radicand: ExprId,
    scale: ExprId,
}

type SqrtHyperbolicReciprocalSquareParts = SqrtHyperbolicReciprocalParts;
type SqrtHyperbolicReciprocalDerivativeParts = SqrtHyperbolicReciprocalParts;

pub struct PositiveConstantRadiusQuadraticParts {
    pub linear_arg: ExprId,
    pub slope: ExprId,
    pub arctan_arg: ExprId,
    pub arctan_scale: ExprId,
}

const MAX_ARCTAN_BY_PARTS_POLY_DEGREE: usize = 6;

struct LinearRadicalAtom {
    radicand: ExprId,
    slope: BigRational,
    offset: BigRational,
    /// Exponent numerator k: the atom is radicand^(k/2) with k odd.
    half_power: i64,
}

const POSITIVE_QUADRATIC_LN_BY_PARTS_MAX_COFACTOR_DEGREE: usize = 8;
const AFFINE_LN_BY_PARTS_MAX_COFACTOR_DEGREE: usize = 3;

enum ExpByPartsCofactorFailure {
    Stop,
    Skip,
}

// Exp-by-parts route map:
// source constructors stop on malformed cofactors to preserve the historic
// antiderivative path, while target probes skip them so a later factor can
// still match. Keep route priority visible in `integrate_symbolic_expr`.
struct LinearExpPolynomialProductParts {
    exp_factor: ExprId,
    exp_arg: ExprId,
    arg_slope: BigRational,
    cofactor_poly: Polynomial,
}

struct SignedLinearFunctionFactorParts {
    builtin: BuiltinFn,
    arg: ExprId,
    sign: Sign,
    factor: ExprId,
    arg_slope: BigRational,
}

#[derive(Clone)]
struct PartialFractionLinearFactor {
    factor: Polynomial,
    multiplicity: usize,
}

struct LinearPositiveQuadraticPartialFraction {
    quotient: Polynomial,
    linear_factor: Polynomial,
    linear_terms: Vec<(BigRational, usize)>,
    quadratic_factor: Polynomial,
    quadratic_numerator: Polynomial,
}

struct MultiLinearPositiveQuadraticPartialFraction {
    quotient: Polynomial,
    linear_factors: Vec<Polynomial>,
    linear_terms: Vec<(Polynomial, BigRational, usize)>,
    quadratic_factor: Polynomial,
    quadratic_numerator: Polynomial,
}

type RequiredConditionFn = fn(&mut Context, ExprId, &str) -> Option<ExprId>;
type RequiredConditionsFn = fn(&mut Context, ExprId, &str) -> Vec<ExprId>;

enum RequiredConditionCollector {
    Optional(RequiredConditionFn),
    Multi(RequiredConditionsFn),
}

const REQUIRED_NONZERO_CONDITION_COLLECTORS_BEFORE_RESIDUAL_SCAN: &[RequiredConditionCollector] =
    &[RequiredConditionCollector::Optional(
        trig_log_required_nonzero,
    )];

const REQUIRED_NONZERO_CONDITION_COLLECTORS_AFTER_RESIDUAL_SCAN: &[RequiredConditionCollector] = &[
    RequiredConditionCollector::Optional(arctan_symbolic_scaled_variable_required_nonzero),
    RequiredConditionCollector::Optional(
        arctan_sqrt_var_symbolic_square_shift_required_nonzero_parameter,
    ),
    RequiredConditionCollector::Optional(polynomial_trig_log_required_nonzero),
    RequiredConditionCollector::Optional(reciprocal_trig_log_required_nonzero),
    RequiredConditionCollector::Optional(trig_log_derivative_ratio_required_nonzero),
    RequiredConditionCollector::Optional(trig_reciprocal_derivative_required_nonzero),
    RequiredConditionCollector::Optional(polynomial_trig_reciprocal_derivative_required_nonzero),
    RequiredConditionCollector::Optional(sqrt_trig_reciprocal_derivative_required_nonzero),
    RequiredConditionCollector::Optional(sqrt_trig_log_derivative_required_nonzero),
    RequiredConditionCollector::Optional(sqrt_reciprocal_trig_log_derivative_required_nonzero),
    RequiredConditionCollector::Optional(sqrt_hyperbolic_log_derivative_required_nonzero),
    RequiredConditionCollector::Optional(sqrt_hyperbolic_reciprocal_square_required_nonzero),
    RequiredConditionCollector::Optional(sqrt_hyperbolic_reciprocal_derivative_required_nonzero),
    RequiredConditionCollector::Optional(polynomial_trig_reciprocal_factor_required_nonzero),
    RequiredConditionCollector::Optional(
        constant_scaled_trig_reciprocal_derivative_required_nonzero,
    ),
    RequiredConditionCollector::Optional(trig_ratio_power_reciprocal_square_required_nonzero),
    RequiredConditionCollector::Optional(polynomial_reciprocal_trig_square_required_nonzero),
    RequiredConditionCollector::Optional(polynomial_sec_csc_square_required_nonzero),
    RequiredConditionCollector::Optional(sec_fourth_required_nonzero),
    RequiredConditionCollector::Optional(csc_fourth_required_nonzero),
    RequiredConditionCollector::Optional(sec_sixth_required_nonzero),
    RequiredConditionCollector::Optional(csc_sixth_required_nonzero),
    RequiredConditionCollector::Optional(sec_eighth_required_nonzero),
    RequiredConditionCollector::Optional(csc_eighth_required_nonzero),
    RequiredConditionCollector::Optional(trig_ratio_square_required_nonzero),
    RequiredConditionCollector::Optional(reciprocal_trig_square_required_nonzero),
    RequiredConditionCollector::Optional(
        polynomial_denominator_power_substitution_required_nonzero,
    ),
    RequiredConditionCollector::Optional(
        polynomial_negative_denominator_power_substitution_required_nonzero,
    ),
    RequiredConditionCollector::Optional(
        polynomial_reciprocal_quotient_denominator_power_substitution_required_nonzero,
    ),
    RequiredConditionCollector::Multi(rational_linear_partial_fraction_required_nonzero),
    RequiredConditionCollector::Multi(rational_linear_positive_quadratic_required_nonzero),
];

const REQUIRED_POSITIVE_CONDITION_COLLECTORS: &[RequiredConditionCollector] = &[
    RequiredConditionCollector::Optional(bounded_inverse_trig_linear_radicand),
    RequiredConditionCollector::Optional(arctan_sqrt_var_reciprocal_required_positive_radicand),
    RequiredConditionCollector::Optional(
        arctan_sqrt_affine_derivative_required_positive_radicand_from_mut_context,
    ),
    RequiredConditionCollector::Optional(asinh_sqrt_reciprocal_positive_condition),
    RequiredConditionCollector::Optional(atanh_sqrt_reciprocal_positive_condition),
    RequiredConditionCollector::Optional(arcsin_polynomial_substitution_radicand),
    RequiredConditionCollector::Multi(acosh_polynomial_substitution_positive_conditions),
    RequiredConditionCollector::Optional(sqrt_derivative_substitution_radicand),
    RequiredConditionCollector::Optional(affine_sqrt_product_derivative_radicand_from_mut_context),
    RequiredConditionCollector::Multi(shifted_sqrt_arcsin_inverse_product_positive_conditions),
    RequiredConditionCollector::Optional(sqrt_trig_reciprocal_derivative_radicand),
    RequiredConditionCollector::Optional(sqrt_trig_log_derivative_radicand),
    RequiredConditionCollector::Optional(sqrt_reciprocal_trig_log_derivative_radicand),
    RequiredConditionCollector::Optional(sqrt_hyperbolic_log_derivative_radicand),
    RequiredConditionCollector::Optional(sqrt_hyperbolic_reciprocal_square_radicand),
    RequiredConditionCollector::Optional(sqrt_hyperbolic_reciprocal_derivative_radicand),
    RequiredConditionCollector::Optional(
        polynomial_fractional_denominator_power_substitution_required_positive,
    ),
    RequiredConditionCollector::Multi(polynomial_log_power_product_required_positive_condition),
    RequiredConditionCollector::Optional(atanh_polynomial_substitution_denominator),
    RequiredConditionCollector::Multi(acosh_affine_radicands),
];

#[cfg(test)]
mod tests;

mod by_parts;
mod general;
mod hyperbolic;
mod inverse_trig;
mod logs_exp;
mod polynomial;
mod radicals;
mod rational;
mod substitution;
mod support;
mod trigonometric;

pub use by_parts::*;
pub use general::*;
pub use hyperbolic::*;
pub use inverse_trig::*;
pub use logs_exp::*;
pub use polynomial::*;
pub use radicals::*;
pub use rational::*;
pub use substitution::*;
pub use support::*;
pub use trigonometric::*;
