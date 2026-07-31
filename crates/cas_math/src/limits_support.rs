use crate::calculus_domain_support::{
    known_positive_constant_exceeds_one, real_domain_is_empty_for_static_expr,
};
use crate::infinity_support::{mk_infinity, InfSign};
use crate::limit_types::{
    Approach, FiniteLimitSide, LimitEvalOutcome, LimitOptions, PreSimplifyMode,
};
use crate::perfect_square_support::rational_sqrt;
use crate::pi_helpers::extract_rational_pi_multiple;
use crate::polynomial::Polynomial;
use crate::root_forms::{extract_square_root_base, rational_cbrt_exact, rational_nth_root};
use crate::trig_eval_table_support::lookup_trig_or_inverse;
use crate::trig_values::TrigValue;
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

const LIMIT_STATIC_DOMAIN_PROOF_DEPTH: usize = 8;
const LIMIT_STATIC_DOMAIN_SCAN_DEPTH: usize = 24;

struct FiniteTrigZeroTailLocal<'a> {
    var: ExprId,
    point: ExprId,
    point_value: &'a BigRational,
    side: FiniteLimitSide,
    var_name: &'a str,
}

#[derive(Clone)]
enum UnitLogBase {
    Natural,
    Fixed(BigRational),
    UnitBoundary(InfSign),
}

#[derive(Clone, Copy)]
enum InverseTrigEndpoint {
    Lower,
    Upper,
}

const FINITE_INTEGER_POWER_EXACT_FOLD_LIMIT: u64 = 32;
const FINITE_LOG_EXACT_RATIONAL_NUMERATOR_LIMIT: i64 = 32;
const FINITE_LOG_EXACT_RATIONAL_DENOMINATOR_LIMIT: i64 = 8;

enum SqueezeFactorClass {
    /// Resolves to 0.
    Infinitesimal,
    /// Resolves to a finite nonzero value (bounded cofactor).
    FiniteLimit,
    /// No limit, but globally bounded near the point (e.g. sin(1/x)).
    BoundedOscillator,
}

/// Highest Taylor order tracked by the higher-order 0/0 quotient rule.
const TAYLOR_QUOTIENT_MAX_ORDER: usize = 12;

thread_local! {
    /// Re-entry depth of the L'Hôpital rule. The rule evaluates the limit of the
    /// numerator and denominator by recursing into the finite cascade, which can
    /// land back here; this caps that re-entry.
    static LHOPITAL_REENTRY_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Maximum L'Hôpital re-entries AND successive differentiations. Four covers the
/// university repertoire (a quartic-order 0/0 needs four passes) while bounding
/// the cost and any pathological recursion.
const MAX_LHOPITAL_DEPTH: usize = 4;

/// Variable-count cap for the multivariate Taylor verb (precedent: the
/// engine's `VERB_MAX_VARS` for the vectorial verbs).
const TAYLOR_MULTIVAR_MAX_VARS: usize = 8;
/// Cap on the number of multi-indices `C(order+d, d)` the multivariate
/// expansion may enumerate (F2, Fase 3): beyond it the command declines to an
/// honest residual instead of assembling a combinatorial tree.
const TAYLOR_MULTIVAR_MAX_TERMS: u128 = 64;

#[derive(Debug, Clone)]
struct PolynomialGrowthInfo {
    degree: u32,
    leading_coeff: BigRational,
}

#[derive(Debug, Clone)]
struct ScaledPolynomialExpTailInfo {
    coeff: BigRational,
    tail: InfSign,
}

#[derive(Debug, Clone)]
struct ScaledSubpolynomialTailInfo {
    coeff: BigRational,
}

/// Emitted by the entry kill-switch when the ambient value domain is complex:
/// the rules below reason with the real order, which does not decide complex
/// limits (`e^(-1/z²)` has no limit at 0 in ℂ although every real-order rule
/// concludes 0).
pub const COMPLEX_DOMAIN_LIMIT_UNSUPPORTED_WARNING: &str =
    "Limits under the complex value domain are not supported safely yet";
/// Emitted when the approach point contains the imaginary unit while the value
/// domain is real: no rule's real-neighbourhood reasoning applies at such a
/// point, so substituting it would fabricate a value (e.g. `tanh` at `iπ/2` is
/// a pole).
pub const IMAGINARY_POINT_LIMIT_UNSUPPORTED_WARNING: &str =
    "Limit points containing the imaginary unit are not supported in the real value domain";
const FINITE_POINT_LIMIT_UNSUPPORTED_WARNING: &str =
    "Finite point limits are not supported safely yet";
const FINITE_EMPTY_PUNCTURED_REAL_NEIGHBORHOOD_WARNING_DETAIL: &str =
    "real-domain condition holds only at the approach point; no punctured real neighbourhood is available";

struct FiniteResidualPoint {
    var_name: String,
    point_value: BigRational,
}

/// Classification of a computed one-sided limit for the bilateral combiner.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LateralLimitClass {
    PosInfinity,
    NegInfinity,
    /// A finite expression (no ∞/undefined/limit-residual inside).
    Finite(ExprId),
}

/// One path witness for a multivariate DNE verdict (F8, Fase 3).
pub struct MultivarPathWitness {
    /// e.g. `y = x^2` — built structurally minimal at the origin.
    pub path_display: String,
    /// Exact value along the path, or a proven-DNE note.
    pub value_display: String,
}

/// Verdict of the path battery: either two witnesses with DIFFERENT exact
/// values, or a single path whose univariate limit provably does not exist.
pub struct MultivarDneByPaths {
    pub witness_a: MultivarPathWitness,
    pub witness_b: Option<MultivarPathWitness>,
}

/// Cell cap for componentwise matrix limits (precedent: the engine's
/// `COMPONENTWISE_MAX_CELLS` for matrix diff/integrate).
const MATRIX_LIMIT_MAX_CELLS: usize = 64;

const PRESIMPLIFY_MAX_DEPTH: usize = 500;

#[cfg(test)]
mod tests;

mod general;
mod infinity;
mod lhopital;
mod logs_exp;
mod polynomial;
mod rational;
mod sign_zero;
mod support;
mod tail_growth;
mod trigonometric;

pub use general::*;
pub use infinity::*;
use lhopital::*;
use logs_exp::*;
pub(crate) use polynomial::*;
pub(crate) use rational::*;
pub use sign_zero::*;
pub(crate) use support::*;
use tail_growth::*;
use trigonometric::*;
