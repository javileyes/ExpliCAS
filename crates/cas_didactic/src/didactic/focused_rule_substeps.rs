use super::nested_fraction_analysis::NestedFractionPattern;
use super::SubStep;
use crate::didactic::try_as_fraction;
use crate::runtime::Step;
use cas_ast::ordering::compare_expr;
use cas_ast::views::{as_rational_const, square_free_decompose};
use cas_ast::{substitute_expr_by_id, BuiltinFn, Constant, Context, Expr, ExprId};
use cas_math::cancel_support::try_cancel_common_additive_terms_expr;
use cas_math::expr_destructure::{as_div, as_mul, as_pow};
use cas_math::expr_extract::extract_i64_multiplier_and_base_factors;
use cas_math::expr_extract::{
    extract_exp_argument, extract_log_base_argument_view, log10_base_sentinel,
};
use cas_math::expr_nary::build_balanced_mul;
use cas_math::expr_nary::{self, AddView, MulView, Sign};
use cas_math::limit_types::Approach;
use cas_math::poly_compare::poly_eq;
use cas_math::polynomial::Polynomial;
use cas_math::summation_support::{
    detect_factorized_telescoping_square_base, extract_linear_offset, extract_unit_shifted_base,
    try_extract_finite_aggregate_call, FiniteAggregateCall,
};
use cas_solver_core::domain_condition::ImplicitCondition;
use cas_solver_core::quadratic_coeffs::{
    extract_quadratic_coefficients, extract_simplified_nonzero_quadratic_coefficients_with_state,
};
use cas_solver_core::rule_names::{
    RULE_CANCEL_EXACT_ADDITIVE_PAIRS, RULE_CONSERVAR_INTEGRAL_RESIDUAL, RULE_EVALUATE_NUMERIC_POWER,
};
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::cmp::Ordering;
use std::collections::BTreeMap;

/// Upper bound on repeated integration-by-parts reductions to narrate. The
/// engine integrates `p(x) * elementary` up to degree 8, so a degree-8 cofactor
/// needs 8 applications; the loop is a defensive backstop (degree strictly
/// decreases each level, so it always terminates first).
const MAX_REPEATED_BY_PARTS_NARRATION_LEVELS: usize = 8;

/// Defensive cap for narrators that re-enter the dispatch chain on a synthetic
/// child `Step` (linearity over a sum, component-wise over a vector).
/// Termination is already STRUCTURAL — the terms of an `AddView` are never
/// themselves an `Add`, the rows of a `Matrix` are never the matrix — so this is
/// a backstop, not the argument.
const MAX_NARRATION_RECURSION_DEPTH: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinomialSquareKind {
    Sum,
    Difference,
}

type SignedTerms = Vec<(ExprId, Sign)>;
type ConcreteLogExpansion = (ExprId, ExprId, SignedTerms);

struct FactoredDivisionExpandedTerms {
    plain: String,
    latex: String,
}

#[derive(Debug, Clone, Copy)]
struct SignedAddTerm {
    term: ExprId,
    negative: bool,
}

struct CompleteSquareSubstepPlan {
    work: Context,
    substeps: Vec<CompleteSquareSubstepExpr>,
}

struct CompleteSquareSubstepExpr {
    title: &'static str,
    before: ExprId,
    after: ExprId,
}

#[derive(Clone, Copy)]
enum LogInversePowerBaseKind {
    Natural,
    Decimal,
    Explicit,
}

struct LogInversePowerPlan {
    source_base: ExprId,
    log_expr: ExprId,
    explicit_target_base: Option<ExprId>,
    base_kind: LogInversePowerBaseKind,
}

struct ExponentialLogPowerInversePlan {
    source_base: ExprId,
    log_expr: ExprId,
    log_arg: ExprId,
    outer_factors: Vec<ExprId>,
    base_kind: LogInversePowerBaseKind,
}

#[derive(Debug, Clone, Copy)]
struct OddHalfPowerSimplifyPlan {
    base: ExprId,
    radicand_power: i64,
    outside_power: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TelescopingFractionSubstepDirection {
    Split,
    Combine,
}

/// Upper bound on inputs that get a number-theory substep: above this the explanation would be
/// unreadable (and the trial loops slow), so we emit no substep and the result still stands.
const NUMBER_THEORY_SUBSTEP_MAX: i64 = 100_000;

#[derive(Clone, Copy)]
enum TrigPowerReductionKind {
    SinEvenPower,
    CosEvenPower,
    SinCosSquares,
}

#[derive(Clone, Copy)]
enum TrigTripleAngleKind {
    Sin,
    Cos,
    Tan,
}

#[derive(Clone, Copy)]
enum TrigQuadrupleAngleKind {
    Sin,
    Cos,
}

#[derive(Clone, Copy)]
enum TrigQuintupleAngleKind {
    Sin,
    Cos,
}

#[derive(Clone, Copy)]
enum HyperbolicTripleAngleKind {
    Sinh,
    Cosh,
    Tanh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HalfAngleTangentVariant {
    OneMinusCosOverSin,
    SinOverOnePlusCos,
}

#[derive(Clone, Copy)]
enum LogDidacticFamily {
    Ln,
    Log10,
    LogBase(ExprId),
}

struct ScaledLogDidacticTerm {
    family: LogDidacticFamily,
    coeff: num_bigint::BigInt,
}

const MAX_FULL_POLY_PRODUCT_SUBSTEP_TERMS: usize = 14;

#[derive(Debug, Clone)]
struct PolyContribution {
    coeff: BigRational,
    degree: usize,
}

#[derive(Debug, Clone)]
struct PolyProductDidacticPlan {
    expanded_display: String,
    expanded_latex: String,
    grouped_display: Option<String>,
    grouped_latex: Option<String>,
    expanded_terms: usize,
    repeated_degree_groups: usize,
    cancelled_degree_groups: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrigLogTableKind {
    Tangent,
    Cotangent,
    Secant,
    Cosecant,
}

struct TrigLogTableMatch {
    kind: TrigLogTableKind,
    arg: ExprId,
    trace: TrigLogTableTrace,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TrigLogTableTrace {
    AffineArgument,
    ConstantMultipleAffineArgument {
        cofactor_display: String,
        cofactor_latex: String,
        coefficient: BigRational,
        slope: BigRational,
    },
    PolynomialCofactor(Box<TrigLogPolynomialCofactorTrace>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TrigLogPolynomialCofactorTrace {
    cofactor_display: String,
    cofactor_latex: String,
    derivative_display: String,
    derivative_latex: String,
    scale: BigRational,
    symbolic_scale_display: Option<String>,
    symbolic_scale_latex: Option<String>,
}

struct TrigPolynomialCofactor {
    builtin: BuiltinFn,
    arg: ExprId,
    expr: ExprId,
    polynomial: Polynomial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HyperbolicLogTableKind {
    Tanh,
    ReciprocalTanh,
    CoshOverSinh,
}

struct HyperbolicLogTableMatch {
    kind: HyperbolicLogTableKind,
    arg: ExprId,
    cofactor_display: String,
    cofactor_latex: String,
    derivative_display: String,
    derivative_latex: String,
    scale: BigRational,
    symbolic_scale_display: Option<String>,
    symbolic_scale_latex: Option<String>,
}

struct IntegrationConstantFactorAdjustment<'a> {
    cofactor_display: &'a str,
    cofactor_latex: &'a str,
    derivative_display: &'a str,
    derivative_latex: &'a str,
    scale: &'a BigRational,
    symbolic_scale_display: Option<&'a str>,
    symbolic_scale_latex: Option<&'a str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
enum HyperbolicReciprocalTableKind {
    CoshSquare,
    CoshFourth,
    SinhSquare,
    SinhFourth,
    SinhOverCoshSquare,
    CoshOverSinhSquare,
}

struct HyperbolicReciprocalTableMatch {
    kind: HyperbolicReciprocalTableKind,
    arg: ExprId,
    cofactor_display: String,
    cofactor_latex: String,
    derivative_display: String,
    derivative_latex: String,
    scale: BigRational,
    symbolic_scale_display: Option<String>,
    symbolic_scale_latex: Option<String>,
}

struct PolynomialDerivativeTableMatch {
    builtin: BuiltinFn,
    arg: ExprId,
    cofactor_display: String,
    cofactor_latex: String,
    derivative_display: String,
    derivative_latex: String,
    scale: BigRational,
    symbolic_scale_display: Option<String>,
    symbolic_scale_latex: Option<String>,
}

struct PolynomialDerivativeCofactorTrace {
    derivative_display: String,
    derivative_latex: String,
    scale: BigRational,
    cofactor_display: String,
    cofactor_latex: String,
    symbolic_scale_display: Option<String>,
    symbolic_scale_latex: Option<String>,
}

struct LinearDerivativeCofactorTrace {
    derivative_display: String,
    derivative_latex: String,
    scale: BigRational,
    cofactor_display: String,
    cofactor_latex: String,
}

struct LogPowerProductTableMatch {
    power: u32,
    natural_log: bool,
    base: ExprId,
    cofactor_display: String,
    cofactor_latex: String,
    derivative_display: String,
    derivative_latex: String,
    scale: BigRational,
}

struct PolynomialBaseTableMatch {
    title: &'static str,
    base: ExprId,
    cofactor_display: String,
    cofactor_latex: String,
    derivative_display: String,
    derivative_latex: String,
    scale: BigRational,
}

struct NestedInversePolynomialTableMatch {
    builtin: BuiltinFn,
    arg: ExprId,
    derivative_display: String,
    derivative_latex: String,
}

struct ArctanSqrtVarTableMatch {
    arg: ExprId,
    derivative_display: String,
    derivative_latex: String,
}

struct ArctanSqrtArgMatch {
    scale: BigRational,
    radicand: ExprId,
    symbolic_denominator: Option<ExprId>,
    symbolic_multiplier: Option<ExprId>,
}

#[derive(Clone, Copy)]
enum TrigQuotientTableKind {
    Tangent,
    Cotangent,
    SecantSquare,
    CosecantSquare,
}

struct TrigQuotientTableMatch {
    kind: TrigQuotientTableKind,
    arg: ExprId,
    cofactor_display: String,
    cofactor_latex: String,
    derivative_display: String,
    derivative_latex: String,
    scale: BigRational,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReciprocalTrigDerivativeProductKind {
    SecantTangent,
    CosecantCotangent,
}

struct ReciprocalTrigDerivativeProductMatch {
    kind: ReciprocalTrigDerivativeProductKind,
    arg: ExprId,
    cofactor_display: String,
    cofactor_latex: String,
    derivative_display: String,
    derivative_latex: String,
    scale: BigRational,
    symbolic_scale_display: Option<String>,
    symbolic_scale_latex: Option<String>,
}

#[derive(Clone, Copy)]
struct SignedAffineCoeff {
    coeff: ExprId,
    is_negative: bool,
}

struct ReciprocalTrigNumeratorCofactor {
    builtin: BuiltinFn,
    arg: ExprId,
    display: String,
    latex: String,
    polynomial: Polynomial,
}

struct SqrtChainCofactorTrace {
    derivative_display: String,
    derivative_latex: String,
    scale: BigRational,
    cofactor_display: String,
    cofactor_latex: String,
    symbolic_scale_display: Option<String>,
    symbolic_scale_latex: Option<String>,
}

struct SqrtChainSymbolicScaleTraceInput<'a> {
    numerator_negative: bool,
    numerator_factors: &'a [ExprId],
    denominator_factors: &'a [ExprId],
    radicand: ExprId,
    radicand_derivative: &'a Polynomial,
    derivative_simplified: ExprId,
    var_name: &'a str,
}

struct NestedTrigLogDerivativeSubstitutionPlan {
    log_expr: ExprId,
    derivative_display: String,
    derivative_latex: String,
}

/// Narrate the NAMED notable limit behind a resolved finite-limit step (the audit gap:
/// `sin(x)/x = 1` is computed but the rule is never named). Sound by the result check:
/// the substep is emitted ONLY when the result equals the notable value, so
/// `limit(sin(x)/x, x, 5) = sin(5)/5` is correctly NOT narrated as `sin(u)/u → 1`.
/// Title of the factor-and-cancel technique, shared between the one-line
/// recognizer (`notable_limit_name`) and the deepened multi-substep builder so
/// the two never drift.
const LIMIT_FACTOR_CANCEL_TITLE: &str =
    "Factorizar numerador y denominador y cancelar el factor común antes de evaluar";

/// Title of the direct-substitution (continuity) technique, shared between the
/// one-line recognizer and the deepened builder.
const LIMIT_DIRECT_SUBSTITUTION_TITLE: &str =
    "Sustitución directa: el límite de un polinomio es su valor en el punto (continuidad)";

/// Prefix every "notable limit" technique title shares, used both to format the
/// titles and to dispatch the 0/0 deepening.
const LIMIT_NOTABLE_PREFIX: &str = "Aplicar el límite notable: ";

/// Prefix the generic 0/0 (L'Hôpital / Taylor) technique title shares, used to
/// dispatch the iterated-L'Hôpital deepening.
const LIMIT_LHOPITAL_DESC_PREFIX: &str = "Indeterminación 0/0 en";

/// Title of the squeeze-theorem technique, shared between the recognizer and the
/// deepened bounding builder.
const LIMIT_SQUEEZE_TITLE: &str =
    "Aplicar el teorema del sándwich: factor acotado × infinitésimo → 0";

/// Prefix every infinity-dominance technique title shares, used to dispatch the
/// ∞/∞ deepening.
const LIMIT_DOMINANCE_PREFIX: &str = "Dominancia:";

/// Title of the conjugate-rationalization technique for an `√(quadratic) − linear`
/// (or `linear − √(quadratic)`) `∞−∞` limit at infinity, shared between the
/// recognizer and the deepened builder.
const LIMIT_CONJUGATE_TITLE: &str =
    "Racionaliza multiplicando y dividiendo por el conjugado (indeterminación ∞−∞)";

/// Title of the common-denominator technique for a reciprocal-difference `c/f − d/g`
/// `∞−∞` limit at a finite point, shared between the recognizer and the builder.
const LIMIT_COMMON_DENOM_TITLE: &str =
    "Combina las fracciones sobre un común denominador (indeterminación ∞−∞)";

/// Growth-rate class of an expression at `var → ∞`, ordered `Log < Power < Exp` (the standard
/// dominance hierarchy `ln(x) ≪ x^a ≪ e^x`). `None` for shapes outside the hierarchy.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
enum LimitGrowthClass {
    Log,
    Power,
    Exp,
}

impl LimitGrowthClass {
    fn name(self) -> &'static str {
        match self {
            LimitGrowthClass::Log => "el logaritmo",
            LimitGrowthClass::Power => "la potencia",
            LimitGrowthClass::Exp => "la exponencial",
        }
    }
}

#[derive(Clone, Copy)]
enum CubeIdentityKind {
    Sum,
    Difference,
}

struct CubeIdentityPlan {
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
}

#[derive(Clone, Copy)]
enum SixthPowerIdentityKind {
    Sum,
    Difference,
}

struct SixthPowerIdentityPlan {
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
}

struct ReciprocalExponentPlan;

#[cfg(test)]
mod number_theory_substep_tests;

#[cfg(test)]
mod limit_notable_tests;

#[cfg(test)]
mod named_identity_matcher_tests;

#[cfg(test)]
mod common_denominator_numerator_tests;

#[cfg(test)]
mod telescoping_render_tests;

mod calculus;
mod factoring;
mod fractions;
mod general;
mod hyperbolic;
mod limits;
mod logs_exp;
mod number_theory;
mod polynomial;
mod powers;
mod radicals;
mod rendering;
mod support;
mod trigonometric;

use calculus::*;
use factoring::*;
use fractions::*;
pub(crate) use general::*;
use hyperbolic::*;
pub(crate) use limits::*;
use logs_exp::*;
use number_theory::*;
use polynomial::*;
use powers::*;
use radicals::*;
use rendering::*;
use support::*;
use trigonometric::*;
