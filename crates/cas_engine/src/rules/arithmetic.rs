use crate::define_rule;
use crate::parent_context::ParentContext;
use crate::rule::{Rewrite, Rule};
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Expr};
use cas_math::arithmetic_cancel_support::{
    try_rewrite_add_inverse_zero_expr, try_rewrite_sub_self_zero_expr,
};
use cas_math::arithmetic_rule_support::{
    try_rewrite_add_zero_expr, try_rewrite_combine_constants_expr, try_rewrite_mul_one_expr,
    try_rewrite_normalize_mul_neg_expr, try_rewrite_simplify_numeric_exponents_expr,
};
use cas_math::arithmetic_zero_support::{match_div_zero_numerator_pattern, match_mul_zero_pattern};
use cas_math::expansion_rule_support::{try_expand_small_pow_sum_expr, SmallPowExpandPolicy};
use cas_math::expr_destructure::{as_div, as_mul, as_pow};
use cas_math::expr_extract::{extract_exp_argument, extract_i64_integer};
use cas_math::expr_nary::{build_balanced_add, build_balanced_mul, AddView, MulView, Sign};
use cas_math::expr_predicates::{
    contains_division_like_term, contains_named_var, is_minus_one_expr, is_one_expr, is_zero_expr,
};
use cas_math::expr_rewrite::smart_mul;
use cas_math::fold_add_build_support::try_build_fold_add_fraction_rewrite;
use cas_math::fold_add_fraction_support::extract_fold_add_operands;
use cas_math::fraction_add_rewrite_support::{
    plan_add_fraction_rewrite_with, AddFractionRewriteInput,
};
use cas_math::fraction_combine_policy_support::try_plan_same_denominator_combination_with;
use cas_math::fraction_pair_support::extract_fraction_pair;
use cas_math::fraction_sub_rewrite_support::plan_sub_fraction_rewrite_with;
use cas_math::hyperbolic_identity_support::{
    try_rewrite_hyperbolic_pythagorean_sub_expr, try_rewrite_hyperbolic_triple_angle,
};
use cas_math::logarithm_inverse_support::{
    make_log_expr, try_extract_log_parts, try_rewrite_log_chain_product_expr,
};
use cas_math::nested_fraction_support::try_rewrite_simplify_nested_fraction_expr;
use cas_math::perfect_square_support::rational_sqrt;
use cas_math::poly_compare::poly_eq;
use cas_math::root_forms::extract_square_root_base;
use cas_math::summation_support::{
    try_plan_finite_product_evaluation, try_plan_finite_sum_evaluation, ProductEvaluationKind,
    SumEvaluationKind, SumEvaluationPlan,
};
use cas_math::symbolic_integration_support::get_linear_coeffs;
use cas_math::telescoping_dirichlet::try_dirichlet_kernel_identity;
use cas_math::trig_canonicalization_support::{
    try_rewrite_csc_cot_pythagorean_identity_expr, try_rewrite_sec_tan_pythagorean_identity_expr,
};
use cas_math::trig_contraction_support::try_rewrite_double_angle_contraction_expr;
use cas_math::trig_core_identity_support::{
    try_rewrite_angle_sum_diff_identity_expr, try_rewrite_pythagorean_identity_add_expr,
};
use cas_math::trig_half_angle_support::{
    try_rewrite_hyperbolic_half_angle_squares_expr, try_rewrite_trig_half_angle_squares_expr,
};
use cas_math::trig_linear_support::extract_coef_and_base;
use cas_math::trig_multi_angle_support::{
    try_rewrite_double_angle_function_expr, try_rewrite_quintuple_angle_expr,
    try_rewrite_recursive_trig_expansion_expr, try_rewrite_triple_angle_expr,
    TrigMultiAngleRewriteKind,
};
use cas_math::trig_power_identity_support::{
    try_rewrite_pythagorean_factor_form_add_expr, try_rewrite_recognize_csc_squared_add_expr,
    try_rewrite_recognize_sec_squared_add_expr,
};
use cas_math::trig_roots_flatten::{
    extract_double_angle_arg_relaxed, extract_triple_angle_arg_relaxed, flatten_mul_chain,
};
use cas_math::trig_sum_product_support::{
    args_match_as_multiset, build_avg_with_simplifier, build_half_diff_with_simplifier,
    extract_trig_two_term_diff, extract_trig_two_term_sum, normalize_for_even_fn,
    try_rewrite_product_to_sum_expr, try_rewrite_sum_to_product_contraction_expr,
    TrigProductToSumRewriteKind, TrigSumToProductContractionRewriteKind,
};
use cas_solver_core::quadratic_coeffs::extract_quadratic_coefficients;
use cas_solver_core::rule_names::{
    RULE_CANCEL_EXACT_ADDITIVE_PAIRS, RULE_EXPAND_LOG_ABS_MUL_DIV, RULE_SUM_EXPONENTS,
};
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::Signed;
use num_traits::{One, ToPrimitive, Zero};
use std::cmp::Ordering;
use web_time::Instant;

type BinaryProductWithSumFactor = (
    cas_ast::ExprId,
    (cas_ast::ExprId, Sign),
    (cas_ast::ExprId, Sign),
);

#[derive(Clone, Copy)]
enum SolvePrepCoeffShape {
    Atom,
    NegAtom,
    NegWithDiv,
    NegOther,
    AddSubWithDiv,
    AddSubNoDiv,
    Div,
    MulWithDiv,
    MulNoDiv,
    Pow,
    Function,
    Hold,
    Matrix,
    SessionRef,
}

impl SolvePrepCoeffShape {
    fn simplify_a_profile_label(self) -> &'static str {
        match self {
            Self::Atom => "rule.solve_prep.extract.simplify_a.shape.atom",
            Self::NegAtom => "rule.solve_prep.extract.simplify_a.shape.neg_atom",
            Self::NegWithDiv => "rule.solve_prep.extract.simplify_a.shape.neg_with_div",
            Self::NegOther => "rule.solve_prep.extract.simplify_a.shape.neg_other",
            Self::AddSubWithDiv => "rule.solve_prep.extract.simplify_a.shape.addsub_with_div",
            Self::AddSubNoDiv => "rule.solve_prep.extract.simplify_a.shape.addsub_no_div",
            Self::Div => "rule.solve_prep.extract.simplify_a.shape.div",
            Self::MulWithDiv => "rule.solve_prep.extract.simplify_a.shape.mul_with_div",
            Self::MulNoDiv => "rule.solve_prep.extract.simplify_a.shape.mul_no_div",
            Self::Pow => "rule.solve_prep.extract.simplify_a.shape.pow",
            Self::Function => "rule.solve_prep.extract.simplify_a.shape.function",
            Self::Hold => "rule.solve_prep.extract.simplify_a.shape.hold",
            Self::Matrix => "rule.solve_prep.extract.simplify_a.shape.matrix",
            Self::SessionRef => "rule.solve_prep.extract.simplify_a.shape.session_ref",
        }
    }

    fn simplify_b_profile_label(self) -> &'static str {
        match self {
            Self::Atom => "rule.solve_prep.extract.simplify_b.shape.atom",
            Self::NegAtom => "rule.solve_prep.extract.simplify_b.shape.neg_atom",
            Self::NegWithDiv => "rule.solve_prep.extract.simplify_b.shape.neg_with_div",
            Self::NegOther => "rule.solve_prep.extract.simplify_b.shape.neg_other",
            Self::AddSubWithDiv => "rule.solve_prep.extract.simplify_b.shape.addsub_with_div",
            Self::AddSubNoDiv => "rule.solve_prep.extract.simplify_b.shape.addsub_no_div",
            Self::Div => "rule.solve_prep.extract.simplify_b.shape.div",
            Self::MulWithDiv => "rule.solve_prep.extract.simplify_b.shape.mul_with_div",
            Self::MulNoDiv => "rule.solve_prep.extract.simplify_b.shape.mul_no_div",
            Self::Pow => "rule.solve_prep.extract.simplify_b.shape.pow",
            Self::Function => "rule.solve_prep.extract.simplify_b.shape.function",
            Self::Hold => "rule.solve_prep.extract.simplify_b.shape.hold",
            Self::Matrix => "rule.solve_prep.extract.simplify_b.shape.matrix",
            Self::SessionRef => "rule.solve_prep.extract.simplify_b.shape.session_ref",
        }
    }

    fn simplify_c_profile_label(self) -> &'static str {
        match self {
            Self::Atom => "rule.solve_prep.extract.simplify_c.shape.atom",
            Self::NegAtom => "rule.solve_prep.extract.simplify_c.shape.neg_atom",
            Self::NegWithDiv => "rule.solve_prep.extract.simplify_c.shape.neg_with_div",
            Self::NegOther => "rule.solve_prep.extract.simplify_c.shape.neg_other",
            Self::AddSubWithDiv => "rule.solve_prep.extract.simplify_c.shape.addsub_with_div",
            Self::AddSubNoDiv => "rule.solve_prep.extract.simplify_c.shape.addsub_no_div",
            Self::Div => "rule.solve_prep.extract.simplify_c.shape.div",
            Self::MulWithDiv => "rule.solve_prep.extract.simplify_c.shape.mul_with_div",
            Self::MulNoDiv => "rule.solve_prep.extract.simplify_c.shape.mul_no_div",
            Self::Pow => "rule.solve_prep.extract.simplify_c.shape.pow",
            Self::Function => "rule.solve_prep.extract.simplify_c.shape.function",
            Self::Hold => "rule.solve_prep.extract.simplify_c.shape.hold",
            Self::Matrix => "rule.solve_prep.extract.simplify_c.shape.matrix",
            Self::SessionRef => "rule.solve_prep.extract.simplify_c.shape.session_ref",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OddHalfPowerProductForm {
    base: cas_ast::ExprId,
    outside_power: i64,
}

thread_local! {
    static DEFAULT_SIMPLIFY_NESTING: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

// Deterministic breadth cap for speculative exact-zero probes: each
// top-level pipeline may launch at most this many nested default
// simplifies. The subset-enumeration probes otherwise multiply (terms
// x candidates x 8 comparisons), each one a FULL pipeline, hanging
// sums like sin^2 cos^2 - sin^4 indefinitely. A counter (not a
// deadline) keeps scorecard fingerprints machine-independent. The
// budget is ACTIVE only inside a top-level pipeline scope: direct
// unit-test calls to the probe helpers stay ungated (tests share
// threads, so a consumable TLS would drain across unrelated tests).
const DEFAULT_SIMPLIFY_PROBE_BUDGET: u32 = 48;

// The first few probes of a pipeline may run a FULL nested simplify
// (some equivalences - the phase-shift quotient pair - only prove
// through root shortcuts that the local passes skip); the rest run
// the cheap local passes. Successful matches happen in the first
// probes; only runaway enumerations reach the tail.
const DEFAULT_SIMPLIFY_FULL_PROBE_BUDGET: u32 = 24;

thread_local! {
    static DEFAULT_SIMPLIFY_PROBES_LEFT: std::cell::Cell<Option<u32>> =
        const { std::cell::Cell::new(None) };
    // Probe results memo, keyed by (Context::instance_tag, ExprId). The
    // speculative probes ask for the same handful of subtrees hundreds of
    // times per solve (`solve(e^x+e^(-x)=4,x)` measured 522 calls over 13
    // distinct inputs); a hit replays the earlier result without consuming
    // budget or nesting. The instance tag makes cross-Context replay
    // impossible (pipelines over FRESH or CLONED arenas share this thread
    // local; a bare-ExprId key served foreign ids and crashed `Context::get`).
    // Within one arena the Context is append-only, so a hit never goes stale;
    // refusals (budget exhausted / nesting cap) are NOT cached. Entries stay
    // across pipelines ON PURPOSE (the solver re-probes the same subtrees from
    // dozens of pipelines); memory is bounded by a size cap at arming time.
    static DEFAULT_SIMPLIFY_PROBE_MEMO: std::cell::RefCell<rustc_hash::FxHashMap<(u64, cas_ast::ExprId, crate::semantics::ValueDomain), cas_ast::ExprId>> =
        std::cell::RefCell::new(rustc_hash::FxHashMap::default());
    // Ambient VALUE DOMAIN for the speculative probe pipelines, armed by the
    // top-level pipeline alongside the probe budget. Without it every probe
    // ran with `SimplifyOptions::default()` (= RealOnly), so real-only
    // identities (`√(x²) ≡ |x|`) proved "equivalences" that the OUTER
    // complex-mode session then adopted — `sqrt(9x²) − 3|x| → 0` under
    // `--value-domain complex` (audit 2026-07-30, ficha S4-001). RealOnly
    // default keeps every real-mode caller byte-identical (the sticky
    // value-domain precedent of the solve backend).
    static DEFAULT_SIMPLIFY_PROBE_VALUE_DOMAIN: std::cell::Cell<crate::semantics::ValueDomain> =
        const { std::cell::Cell::new(crate::semantics::ValueDomain::RealOnly) };
}

pub(crate) struct DefaultSimplifyProbeBudgetScope {
    saved: Option<Option<u32>>,
    saved_value_domain: Option<crate::semantics::ValueDomain>,
}

impl Drop for DefaultSimplifyProbeBudgetScope {
    fn drop(&mut self) {
        // Restore (not clear): internal residual pipelines also run at
        // nesting 0 inside an outer pipeline, and clearing here would
        // strip the OUTER pipeline's remaining budget mid-flight.
        if let Some(saved) = self.saved {
            DEFAULT_SIMPLIFY_PROBES_LEFT.with(|left| left.set(saved));
        }
        if let Some(saved) = self.saved_value_domain {
            DEFAULT_SIMPLIFY_PROBE_VALUE_DOMAIN.with(|vd| vd.set(saved));
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OddHalfPowerCancellationMatch {
    focus_before: cas_ast::ExprId,
    focus_after: cas_ast::ExprId,
    rewritten_expr: cas_ast::ExprId,
    base: Option<cas_ast::ExprId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LogAbsMulDivCancellationMatch {
    focus_after: cas_ast::ExprId,
    components: [(cas_ast::ExprId, Sign); 2],
}

#[derive(Debug, Clone)]
struct LogPowerProductCancellationMatch {
    raw_focus_after: cas_ast::ExprId,
    focus_after: cas_ast::ExprId,
    components: Vec<(cas_ast::ExprId, Sign)>,
    needs_power_split: bool,
}

#[derive(Debug, Clone)]
struct LogPowerProductCancellationComponentsMatch {
    focus_expr: cas_ast::ExprId,
    components: Vec<(cas_ast::ExprId, Sign)>,
    changed_by_power: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HyperbolicPythagoreanFactorCancellationMatch {
    local_before: cas_ast::ExprId,
    local_after: cas_ast::ExprId,
    mode: HyperbolicPythagoreanFactorCancellationMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HyperbolicPythagoreanFactorCancellationMode {
    FactorThenRewrite {
        factorized: cas_ast::ExprId,
        rewritten: cas_ast::ExprId,
    },
    AlreadyFactored {
        identity_desc: &'static str,
    },
}

thread_local! {
    /// Per-pipeline memo for the pairwise cancellation matcher (pure over
    /// interned nodes; cleared with the other gate memos).
    static CANCELLATION_MATCH_MEMO: std::cell::RefCell<rustc_hash::FxHashMap<(cas_ast::ExprId, cas_ast::ExprId), bool>> =
        std::cell::RefCell::new(rustc_hash::FxHashMap::default());
}

enum ScaledHyperbolicProductPatternForCancellation {
    SinhCosh(cas_ast::ExprId, cas_ast::ExprId),
    CoshCosh(cas_ast::ExprId, cas_ast::ExprId),
    SinhSinh(cas_ast::ExprId, cas_ast::ExprId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SolvePrepExactEquivalenceRewrite {
    rewritten: cas_ast::ExprId,
    local_before: cas_ast::ExprId,
    local_after: cas_ast::ExprId,
    nonzero_expr: cas_ast::ExprId,
    build_route: SolvePrepBuildRoute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SolvePrepBuildRoute {
    NegSymbolic,
    NegGeneric,
    PosHalf,
    PosSymbolic,
    PosGeneric,
}

impl SolvePrepBuildRoute {
    fn defer_simplify_c_profile_label(self) -> &'static str {
        match self {
            Self::NegSymbolic => "rule.solve_prep.extract.defer_simplify_c.neg_symbolic_scale",
            Self::PosSymbolic => "rule.solve_prep.extract.defer_simplify_c.pos_symbolic_scale",
            Self::NegGeneric => "rule.solve_prep.extract.defer_simplify_c.neg_generic_scale",
            Self::PosHalf => "rule.solve_prep.extract.defer_simplify_c.pos_half_scale",
            Self::PosGeneric => "rule.solve_prep.extract.defer_simplify_c.pos_generic_scale",
        }
    }

    fn simplify_c_profile_label(self) -> &'static str {
        match self {
            Self::NegSymbolic => "rule.solve_prep.extract.simplify_c.build.neg_symbolic_scale",
            Self::NegGeneric => "rule.solve_prep.extract.simplify_c.build.neg_generic_scale",
            Self::PosHalf => "rule.solve_prep.extract.simplify_c.build.pos_half_scale",
            Self::PosSymbolic => "rule.solve_prep.extract.simplify_c.build.pos_symbolic_scale",
            Self::PosGeneric => "rule.solve_prep.extract.simplify_c.build.pos_generic_scale",
        }
    }

    fn default_simplify_profile_label(self) -> &'static str {
        match self {
            Self::NegSymbolic => {
                "rule.fast_solve_prep.route.default_simplify_match.build.neg_symbolic_scale"
            }
            Self::NegGeneric => {
                "rule.fast_solve_prep.route.default_simplify_match.build.neg_generic_scale"
            }
            Self::PosHalf => {
                "rule.fast_solve_prep.route.default_simplify_match.build.pos_half_scale"
            }
            Self::PosSymbolic => {
                "rule.fast_solve_prep.route.default_simplify_match.build.pos_symbolic_scale"
            }
            Self::PosGeneric => {
                "rule.fast_solve_prep.route.default_simplify_match.build.pos_generic_scale"
            }
        }
    }

    fn candidate_total_zero_profile_label(self) -> &'static str {
        match self {
            Self::NegSymbolic => {
                "rule.fast_solve_prep.route.candidate_total_zero.build.neg_symbolic_scale"
            }
            Self::NegGeneric => {
                "rule.fast_solve_prep.route.candidate_total_zero.build.neg_generic_scale"
            }
            Self::PosHalf => "rule.fast_solve_prep.route.candidate_total_zero.build.pos_half_scale",
            Self::PosSymbolic => {
                "rule.fast_solve_prep.route.candidate_total_zero.build.pos_symbolic_scale"
            }
            Self::PosGeneric => {
                "rule.fast_solve_prep.route.candidate_total_zero.build.pos_generic_scale"
            }
        }
    }
}

thread_local! {
    /// Per-pipeline memos for the hot per-node solve-prep gates. The gates
    /// are pure functions of interned (append-only) Context nodes, so a
    /// cached bool never goes stale for a given Context; the memos are
    /// cleared on `Simplifier::new` and at every `simplify_with_stats`
    /// entry, which bounds them and prevents ExprId collisions across
    /// different Contexts on the same thread. On recurrence-shaped DAGs
    /// (tan(10*arcsin(t))) these gates were re-walking overlapping subDAGs
    /// from every Add/Sub node visit.
    static VARIABLE_SQUARE_GATE_MEMO: std::cell::RefCell<rustc_hash::FxHashMap<cas_ast::ExprId, bool>> =
        std::cell::RefCell::new(rustc_hash::FxHashMap::default());
    static SHIFTED_SQUARE_GATE_MEMO: std::cell::RefCell<rustc_hash::FxHashMap<cas_ast::ExprId, bool>> =
        std::cell::RefCell::new(rustc_hash::FxHashMap::default());
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrigPhaseShiftCancellationMode {
    LinearToShifted,
    ShiftedToLinear,
    ShiftedToShifted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TrigPhaseShiftCancellationMatch {
    local_before: cas_ast::ExprId,
    local_after: cas_ast::ExprId,
    mode: TrigPhaseShiftCancellationMode,
}

enum LinearFocusPhaseShiftMatchOutcome {
    NotLinear,
    NeedsGeneralRoute,
    LinearNoMatch,
    Matched(TrigPhaseShiftCancellationMatch),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhaseShiftKindForCancellation {
    Quarter,
    Sixth,
    Third,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExactPhaseShiftLinearCombinationData {
    arg: cas_ast::ExprId,
    coeff: cas_ast::ExprId,
    kind: PhaseShiftKindForCancellation,
    sin_sign: i8,
    cos_sign: i8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExactPhaseShiftLinearCombinationExtraction {
    NotLinear,
    LinearButNotExact,
    Exact(ExactPhaseShiftLinearCombinationData),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GeneralPhaseShiftTermData {
    coeff: cas_ast::ExprId,
    trig_fn: BuiltinFn,
    base_arg: cas_ast::ExprId,
    ratio: cas_ast::ExprId,
    subtract_shift: bool,
    global_sign: i8,
}

type GeneralPhaseShiftLinearSignature = (cas_ast::ExprId, cas_ast::ExprId, cas_ast::ExprId, i8, i8);

type ExactPhaseShiftTermData = (
    cas_ast::ExprId,
    cas_ast::ExprId,
    PhaseShiftKindForCancellation,
    i8,
    i8,
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExactSinSinProductToSumCosKind {
    Sum,
    Diff,
}

define_rule!(
    ExpandTrigSumToProductToEnableCancellationRule,
    "Sum-to-Product Identity Cancellation Bridge",
    Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB)),
    crate::phase::PhaseMask::POST,
    priority: 510,
    |ctx, expr| {
        if !maybe_trig_sum_to_product_zero_candidate(ctx, expr) {
            return None;
        }

        if let Some(rewrite) = try_build_exact_trig_sum_to_product_zero_scope_rewrite(ctx, expr) {
            return Some(rewrite);
        }

        match ctx.get(expr).clone() {
            Expr::Sub(lhs, rhs) => {
                if let Some((rewritten, description)) =
                    try_rewrite_trig_sum_to_product_for_cancellation(ctx, lhs)
                {
                    if exprs_match_after_default_simplify(ctx, rewritten, rhs) {
                        return Some(Rewrite::with_local(
                            ctx.add(Expr::Sub(rhs, rhs)),
                            description,
                            lhs,
                            rhs,
                        ));
                    }
                }

                if let Some((rewritten, description)) =
                    try_rewrite_trig_sum_to_product_for_cancellation(ctx, rhs)
                {
                    if exprs_match_after_default_simplify(ctx, rewritten, lhs) {
                        return Some(Rewrite::with_local(
                            ctx.add(Expr::Sub(lhs, lhs)),
                            description,
                            rhs,
                            lhs,
                        ));
                    }
                }

                None
            }
            Expr::Add(lhs, rhs) => {
                if let Some((rewritten, description)) =
                    try_rewrite_trig_sum_to_product_for_cancellation(ctx, lhs)
                {
                    if expr_matches_negation_after_default_simplify(ctx, rewritten, rhs) {
                        return Some(Rewrite::with_local(
                            ctx.add(Expr::Add(rewritten, rhs)),
                            description,
                            lhs,
                            rewritten,
                        ));
                    }
                }

                if let Some((rewritten, description)) =
                    try_rewrite_trig_sum_to_product_for_cancellation(ctx, rhs)
                {
                    if expr_matches_negation_after_default_simplify(ctx, rewritten, lhs) {
                        return Some(Rewrite::with_local(
                            ctx.add(Expr::Add(lhs, rewritten)),
                            description,
                            rhs,
                            rewritten,
                        ));
                    }
                }

                None
            }
            _ => None,
        }
    }
);

define_rule!(
    ExpandTrigSineProductTripleAngleToEnableCancellationRule,
    "Product-to-Sum and Triple-Angle Identity",
    Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB)),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    priority: 510,
    |ctx, expr| {
        if !maybe_trig_square_zero_candidate(ctx, expr) {
            return None;
        }

        try_build_exact_trig_sine_product_triple_angle_zero_scope_rewrite(ctx, expr)
    }
);

define_rule!(
    CancelExactAdditivePairsRule,
    RULE_CANCEL_EXACT_ADDITIVE_PAIRS,
    Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB)),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    priority: 511,
    |ctx, expr, parent_ctx| {
        let rewritten = try_rewrite_exact_additive_term_cancellation_expr(ctx, expr)?;
        let allow =
            cas_solver_core::undefined_risk_policy_support::allow_cancellation_with_undefined_risk_mode_flags(
                matches!(parent_ctx.domain_mode(), crate::DomainMode::Assume),
                matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict),
                crate::collect::has_undefined_risk(ctx, expr),
            );
        if !allow {
            return None;
        }

        Some(Rewrite::new(rewritten).desc("Cancel exact additive pairs"))
    }
);

define_rule!(
    CollapseExactZeroTrigDoubleAngleCosVariantRule,
    "Double Angle Expansion",
    Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB)),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    priority: 510,
    |ctx, expr| {
        if !maybe_trig_square_zero_candidate(ctx, expr) {
            return None;
        }

        try_build_exact_zero_trig_double_angle_cos_variant_zero_scope_rewrite(ctx, expr)
    }
);

define_rule!(
    ExpandTrigSquareIdentityToEnableCancellationRule,
    "Trig Square Identity",
    Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB)),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    priority: 510,
    |ctx, expr| {
        if !maybe_trig_square_zero_candidate(ctx, expr) {
            return None;
        }

        try_build_exact_trig_square_zero_scope_rewrite(ctx, expr)
    }
);

define_rule!(
    ExpandTrigPhaseShiftToEnableCancellationRule,
    "Phase Shift Identity",
    Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB)),
    crate::phase::PhaseMask::POST,
    priority: 510,
    |ctx, expr| {
        if !maybe_trig_phase_shift_zero_candidate(ctx, expr) {
            return None;
        }
        let profiling =
            crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();

        let exact_scope = if profiling {
            run_profiled_orchestrator_option_section(
                "rule.phase_shift.exact_scope_rewrite",
                Some(render_expr_for_orchestrator_profile(ctx, expr)),
                || try_build_exact_trig_phase_shift_zero_scope_rewrite(ctx, expr),
            )
        } else {
            try_build_exact_trig_phase_shift_zero_scope_rewrite(ctx, expr)
        };
        if let Some(rewrite) = exact_scope {
            return Some(rewrite);
        }

        match ctx.get(expr).clone() {
            Expr::Sub(lhs, rhs) => {
                let rewrite_match = if profiling {
                    run_profiled_orchestrator_option_section(
                        "rule.phase_shift.binary_sub_match",
                        Some(format!(
                            "{}  ||  {}",
                            render_expr_for_orchestrator_profile(ctx, lhs),
                            render_expr_for_orchestrator_profile(ctx, rhs)
                        )),
                        || {
                            try_find_trig_phase_shift_cancellation_match(ctx, lhs, rhs, false)
                                .or_else(|| {
                                    try_find_trig_phase_shift_cancellation_match(
                                        ctx, rhs, lhs, false,
                                    )
                                })
                        },
                    )?
                } else {
                    try_find_trig_phase_shift_cancellation_match(ctx, lhs, rhs, false)
                        .or_else(|| {
                            try_find_trig_phase_shift_cancellation_match(ctx, rhs, lhs, false)
                        })?
                };
                Some(build_trig_phase_shift_zero_rewrite(ctx, rewrite_match))
            }
            Expr::Add(lhs, rhs) => {
                let rewrite_match = if profiling {
                    let pair_sample = Some(format!(
                        "{}  ||  {}",
                        render_expr_for_orchestrator_profile(ctx, lhs),
                        render_expr_for_orchestrator_profile(ctx, rhs)
                    ));
                    let trig_pair_supported = run_profiled_orchestrator_option_section(
                        "rule.phase_shift.binary_add_match.trig_pair_gate",
                        pair_sample.clone(),
                        || {
                            binary_add_pair_has_trig_phase_shift_shape_for_cancellation(
                                ctx,
                                lhs,
                                rhs,
                                pair_sample.clone(),
                            )
                            .then_some(())
                        },
                    )
                    .is_some();
                    if !trig_pair_supported {
                        return None;
                    }
                    let _ = run_profiled_orchestrator_option_section(
                        "rule.phase_shift.binary_add_match.surface_plain_algebraic_gate",
                        pair_sample.clone(),
                        || {
                            profile_binary_add_surface_pair_shape_for_phase_shift(
                                ctx,
                                lhs,
                                rhs,
                                pair_sample.clone(),
                            );
                            Some(())
                        },
                    )
                    .is_some();
                    let _ = run_profiled_orchestrator_option_section(
                        "rule.phase_shift.binary_add_match.term_family_gate",
                        pair_sample.clone(),
                        || {
                            profile_binary_add_term_family_pair_for_phase_shift(
                                ctx,
                                lhs,
                                rhs,
                                pair_sample.clone(),
                            );
                            Some(())
                        },
                    )
                    .is_some();
                    let productive_term_family = run_profiled_orchestrator_option_section(
                        "rule.phase_shift.binary_add_match.productive_term_family_gate",
                        pair_sample.clone(),
                        || {
                            profile_binary_add_productive_term_family_gate_for_phase_shift(
                                ctx,
                                lhs,
                                rhs,
                                pair_sample.clone(),
                            );
                            binary_add_pair_has_productive_phase_shift_term_family_for_cancellation(
                                ctx, lhs, rhs,
                            )
                            .then_some(())
                        },
                    )
                    .is_some();
                    if !productive_term_family {
                        return None;
                    }
                    run_profiled_orchestrator_option_section(
                        "rule.phase_shift.binary_add_match",
                        pair_sample.clone(),
                        || {
                            run_profiled_orchestrator_option_section(
                                "rule.phase_shift.binary_add_match.forward_try",
                                pair_sample.clone(),
                                || {
                                    try_find_trig_phase_shift_cancellation_match(
                                        ctx, lhs, rhs, true,
                                    )
                                },
                            )
                            .or_else(|| {
                                run_profiled_orchestrator_option_section(
                                    "rule.phase_shift.binary_add_match.reverse_try",
                                    pair_sample,
                                    || {
                                        try_find_trig_phase_shift_cancellation_match(
                                            ctx, rhs, lhs, true,
                                        )
                                    },
                                )
                            })
                        },
                    )?
                } else {
                    if !binary_add_pair_has_trig_phase_shift_shape_for_cancellation(
                        ctx, lhs, rhs, None,
                    ) {
                        return None;
                    }
                    if !binary_add_pair_has_productive_phase_shift_term_family_for_cancellation(
                        ctx, lhs, rhs,
                    ) {
                        return None;
                    }
                    try_find_trig_phase_shift_cancellation_match(ctx, lhs, rhs, true).or_else(
                        || try_find_trig_phase_shift_cancellation_match(ctx, rhs, lhs, true),
                    )?
                };
                Some(build_trig_phase_shift_zero_rewrite(ctx, rewrite_match))
            }
            _ => None,
        }
    }
);

define_rule!(
    ExpandHyperbolicPythagoreanFactorToEnableCancellationRule,
    "Hyperbolic Pythagorean Identity Cancellation Bridge",
    Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB)),
    crate::phase::PhaseMask::CORE
        | crate::phase::PhaseMask::TRANSFORM
        | crate::phase::PhaseMask::POST,
    priority: 510,
    |ctx, expr, parent_ctx| {
        if !maybe_hyperbolic_pythagorean_factor_zero_candidate(ctx, expr) {
            return None;
        }

        if let Some(rewrite) =
            try_build_exact_hyperbolic_pythagorean_factor_zero_scope_rewrite(ctx, expr)
        {
            return Some(rewrite);
        }

        if parent_ctx.depth() == 0 {
            if let Some(rewrite_match) =
                try_rewrite_hyperbolic_pythagorean_factor_for_cancellation(ctx, expr)
            {
                // Only `AlreadyFactored` — one factored term cancelling its
                // companion (`sinh·(cosh²−1) − sinh³`) — genuinely collapses to 0.
                //
                // `FactorThenRewrite` at the root is a STANDALONE difference
                // `k·cosh³ − k·cosh`, which equals `k·cosh·(cosh²−1) = k·cosh·sinh²`
                // and is NEVER identically 0. Collapsing it to 0 was a wrong-answer
                // (`cosh(3x) − cosh(x)` → 0). Decline instead of forcing 0: the
                // correct expanded form `k·cosh³ − k·cosh` is left as-is (just like a
                // plain polynomial `y³ − y` is not eagerly factored), so the value is
                // right and multi-term cancellations still flow through the scope
                // rewrite above.
                if let HyperbolicPythagoreanFactorCancellationMode::AlreadyFactored { .. } =
                    rewrite_match.mode
                {
                    return Some(build_hyperbolic_pythagorean_factor_root_zero_rewrite(
                        ctx,
                        expr,
                        rewrite_match,
                    ));
                }
            }
        }

        None
    }
);

define_rule!(
    ExpandHyperbolicAngleSumDiffToEnableCancellationRule,
    "Hyperbolic Angle Sum/Difference Identity",
    Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB)),
    crate::phase::PhaseMask::POST,
    priority: 510,
    |ctx, expr| {
        if !maybe_hyperbolic_angle_sum_diff_zero_candidate(ctx, expr) {
            return None;
        }

        if let Some(rewrite) =
            try_build_exact_hyperbolic_angle_sum_diff_zero_scope_rewrite(ctx, expr)
        {
            return Some(rewrite);
        }

        match ctx.get(expr).clone() {
            Expr::Sub(lhs, rhs) => {
                if let Some(rewritten) =
                    try_rewrite_hyperbolic_angle_sum_diff_for_cancellation(ctx, lhs)
                {
                    if exprs_match_after_default_simplify(ctx, rewritten, rhs) {
                        return Some(Rewrite::with_local(
                            ctx.add(Expr::Sub(rhs, rhs)),
                            "Expand hyperbolic angle sum/difference",
                            lhs,
                            rhs,
                        ));
                    }
                }

                if let Some(rewritten) =
                    try_rewrite_hyperbolic_angle_sum_diff_for_cancellation(ctx, rhs)
                {
                    if exprs_match_after_default_simplify(ctx, rewritten, lhs) {
                        return Some(Rewrite::with_local(
                            ctx.add(Expr::Sub(lhs, lhs)),
                            "Expand hyperbolic angle sum/difference",
                            rhs,
                            lhs,
                        ));
                    }
                }

                None
            }
            Expr::Add(lhs, rhs) => {
                if let Some(rewritten) =
                    try_rewrite_hyperbolic_angle_sum_diff_for_cancellation(ctx, lhs)
                {
                    if expr_matches_negation_for_cancellation(ctx, rewritten, rhs) {
                        return Some(Rewrite::with_local(
                            ctx.add(Expr::Add(rewritten, rhs)),
                            "Expand hyperbolic angle sum/difference",
                            lhs,
                            rewritten,
                        ));
                    }
                }

                if let Some(rewritten) =
                    try_rewrite_hyperbolic_angle_sum_diff_for_cancellation(ctx, rhs)
                {
                    if expr_matches_negation_for_cancellation(ctx, rewritten, lhs) {
                        return Some(Rewrite::with_local(
                            ctx.add(Expr::Add(lhs, rewritten)),
                            "Expand hyperbolic angle sum/difference",
                            rhs,
                            rewritten,
                        ));
                    }
                }

                None
            }
            _ => None,
        }
    }
);

define_rule!(
    AddZeroRule,
    "Identity Property of Addition",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_add_zero_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.description))
    }
);

define_rule!(
    MulOneRule,
    "Identity Property of Multiplication",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_mul_one_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.description))
    }
);

// MulZeroRule: 0*e → 0
// Domain Mode Policy: 0*e → 0 changes the domain of definition if e can be undefined.
// Uses ConditionClass taxonomy:
// - Strict: only apply if other factor has no undefined risk
// - Generic: apply with Defined(e) assumption (Definability class)
// - Assume: apply with Defined(e) assumption
define_rule!(
    MulZeroRule,
    "Zero Property of Multiplication",
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        let pattern = match_mul_zero_pattern(ctx, expr)?;
        let other = pattern.other;
        let has_risk = crate::collect::has_undefined_risk(ctx, other);
        let allowed = cas_solver_core::undefined_risk_policy_support::allow_cancellation_with_undefined_risk_mode_flags(
            matches!(parent_ctx.domain_mode(), crate::DomainMode::Assume),
            matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict),
            has_risk,
        );

        if !allowed {
            return None; // Strict mode: don't simplify if has risk
        }

        // Build assumption events if has risk and allowed
        let assumption_events: smallvec::SmallVec<[crate::AssumptionEvent; 1]> = if has_risk {
            smallvec::smallvec![crate::AssumptionEvent::defined(ctx, other)]
        } else {
            smallvec::SmallVec::new()
        };

        let description = if pattern.zero_on_lhs {
            "0 * x = 0".to_string()
        } else {
            "x * 0 = 0".to_string()
        };

        let zero = ctx.num(0);
        Some(Rewrite::new(zero).desc(description).assume_all(assumption_events))
    }
);

define_rule!(
    DivZeroRule,
    "Zero Property of Division",
    solve_safety: crate::SolveSafety::NeedsCondition(
        crate::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::Proof;
        use crate::Predicate;

        if let Some(rewrite) = try_build_exact_zero_radical_numerator_const_division_rewrite(ctx, expr) {
            return Some(rewrite);
        }

        let pattern = match_div_zero_numerator_pattern(ctx, expr)?;
        let den = pattern.denominator;

        // Special case: 0/0 → undefined (all modes)
        if pattern.denominator_is_literal_zero {
            let undef = ctx.add(Expr::Constant(cas_ast::Constant::Undefined));
            return Some(Rewrite::new(undef).desc("0/0 is undefined"));
        }

        // Use unified oracle for NonZero condition (Definability class)
        let decision = crate::oracle_allows_with_hint(
            ctx,
            parent_ctx.domain_mode(),
            parent_ctx.value_domain(),
            &Predicate::NonZero(den),
            "Zero Property of Division",
        );

        if !decision.allow {
            return None; // Strict mode: don't simplify if not proven
        }

        // Build assumption events if needed
        let den_proof = crate::helpers::prove_nonzero(ctx, den);
        let assumption_events: smallvec::SmallVec<[crate::AssumptionEvent; 1]> = if decision.assumption.is_some() && den_proof != Proof::Proven {
            smallvec::smallvec![crate::AssumptionEvent::nonzero(ctx, den)]
        } else {
            smallvec::SmallVec::new()
        };

        let zero = ctx.num(0);
        Some(Rewrite::new(zero).desc("0 / d = 0").assume_all(assumption_events))
    }
);

define_rule!(
    CombineConstantsRule,
    "Combine Constants",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_combine_constants_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.description))
    }
);

// =============================================================================
// SubSelfToZeroRule: a - a = 0 (Short-circuit)
// =============================================================================
//
// V2.14.45: This rule MUST fire before expansion rules like TanToSinCosRule.
// Without this, tan(3x) - tan(3x) would expand both tans and fail to cancel.
// Uses priority 500 to ensure it runs first.
//
// Domain Policy: Same as AddInverseRule - check for undefined subexpressions.
// Uses compare_expr for structural equality (handles tan(3x) == tan(3·x)).
// =============================================================================
define_rule!(
    ExpandOddHalfPowerToEnableCancellationRule,
    "Expand Odd Half Power",
    Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB)),
    crate::phase::PhaseMask::POST,
    priority: 510,
    |ctx, expr| {
        if !maybe_odd_half_power_zero_candidate(ctx, expr) {
            return None;
        }

        let (lhs, rhs, was_add_with_neg) = match ctx.get(expr) {
            Expr::Sub(lhs, rhs) => (*lhs, *rhs, false),
            Expr::Add(lhs, rhs) => match ctx.get(*rhs) {
                Expr::Neg(inner) => (*lhs, *inner, true),
                _ => return None,
            },
            _ => return None,
        };

        let candidate = if let Some(matched) =
            try_match_odd_half_power_cancellation_side(ctx, lhs, rhs)
        {
            let new_expr = rebuild_subtractive_expr(ctx, matched.rewritten_expr, rhs, was_add_with_neg);
            let mut rewrite = Rewrite::with_local(
                new_expr,
                "Rewrite an odd half-integer power using a square root",
                matched.focus_before,
                matched.focus_after,
            );
            if let Some(base) = matched.base {
                rewrite = rewrite.requires(crate::ImplicitCondition::NonNegative(base));
            }
            Some(rewrite)
        } else if let Some(matched) =
            try_match_odd_half_power_cancellation_side(ctx, rhs, lhs)
        {
            let new_expr = rebuild_subtractive_expr(ctx, lhs, matched.rewritten_expr, was_add_with_neg);
            let mut rewrite = Rewrite::with_local(
                new_expr,
                "Rewrite an odd half-integer power using a square root",
                matched.focus_before,
                matched.focus_after,
            );
            if let Some(base) = matched.base {
                rewrite = rewrite.requires(crate::ImplicitCondition::NonNegative(base));
            }
            Some(rewrite)
        } else {
            None
        }?;

        Some(candidate)
    }
);

define_rule!(
    ExpandLogProductPowerToEnableCancellationRule,
    "Expand Log Product Power",
    Some(crate::target_kind::TargetKindSet::ADD_SUB),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    priority: 511,
    |ctx, expr| {
        if !maybe_log_product_power_zero_candidate(ctx, expr) {
            return None;
        }
        let view = AddView::from_expr(ctx, expr);
        let prepared_terms: Vec<_> = view
            .terms
            .iter()
            .copied()
            .map(|(term_expr, term_sign)| {
                let (normalized_term_expr, normalized_term_sign) =
                    normalize_signed_add_term(ctx, term_expr, term_sign);
                let fast_log_term =
                    try_normalize_log_term_for_fast_match(ctx, normalized_term_expr);
                (normalized_term_expr, normalized_term_sign, fast_log_term)
            })
            .collect();

        for (focus_index, (raw_focus_expr, _raw_focus_sign)) in
            view.terms.iter().copied().enumerate()
        {
            let (focus_expr, focus_sign, _) = prepared_terms[focus_index];
            let Some(matched) =
                try_match_log_product_power_cancellation_components_side(ctx, focus_expr)
            else {
                continue;
            };

            let mut used_indices = Vec::new();
            let mut all_components_found = true;
            for (component_expr, component_sign) in &matched.components {
                let expected_sign = if focus_sign == Sign::Pos {
                    component_sign.negate()
                } else {
                    *component_sign
                };

                let mut found_index = None;
                for (term_index, (normalized_term_expr, normalized_term_sign, fast_log_term)) in
                    prepared_terms.iter().copied().enumerate()
                {
                    if term_index == focus_index || used_indices.contains(&term_index) {
                        continue;
                    }
                    if normalized_term_sign != expected_sign {
                        continue;
                    }
                    if log_cancellation_component_matches(
                        ctx,
                        normalized_term_expr,
                        fast_log_term,
                        *component_expr,
                    ) {
                        found_index = Some(term_index);
                        break;
                    }
                }

                if let Some(term_index) = found_index {
                    used_indices.push(term_index);
                } else {
                    all_components_found = false;
                    break;
                }
            }

            if !all_components_found || used_indices.len() + 1 != view.terms.len() {
                continue;
            }

            let zero = ctx.num(0);
            let focus_after = build_signed_add_expr(ctx, &matched.components);
            let focus_before_display = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: raw_focus_expr
                }
            );
            let focus_after_display = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: focus_after
                }
            );

            let mut rewrite = Rewrite::with_local(
                zero,
                "Log expansion followed by exact cancellation",
                expr,
                zero,
            )
            .substep(
                "Expandir el logaritmo del producto o del cociente",
                vec![format!("Reescribir {focus_before_display} como {focus_after_display}.")],
            );

            if matched.changed_by_power {
                rewrite = rewrite.substep(
                    "Sacar exponentes fuera del logaritmo cuando sea necesario",
                    vec![format!("Así se obtiene {focus_after_display}.")],
                );
            }

            return Some(rewrite.substep(
                "Cancelar términos iguales",
                vec![
                    "Tras la expansión, los términos opuestos se anulan y el resultado es 0."
                        .to_string(),
                ],
            ));
        }

        None
    }
);

define_rule!(
    ExpandLogAbsMulDivToEnableCancellationRule,
    RULE_EXPAND_LOG_ABS_MUL_DIV,
    Some(crate::target_kind::TargetKindSet::ADD_SUB),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    priority: 510,
    |ctx, expr| {
        if !maybe_log_abs_mul_div_zero_candidate(ctx, expr) {
            return None;
        }
        let view = AddView::from_expr(ctx, expr);
        let prepared_terms: Vec<_> = view
            .terms
            .iter()
            .copied()
            .map(|(term_expr, term_sign)| {
                let (normalized_term_expr, normalized_term_sign) =
                    normalize_signed_add_term(ctx, term_expr, term_sign);
                let fast_log_term =
                    try_normalize_log_term_for_fast_match(ctx, normalized_term_expr);
                (normalized_term_expr, normalized_term_sign, fast_log_term)
            })
            .collect();

        for (focus_index, (raw_focus_expr, _raw_focus_sign)) in
            view.terms.iter().copied().enumerate()
        {
            let (focus_expr, focus_sign, _) = prepared_terms[focus_index];
            let Some(matched) = try_match_log_abs_mul_div_cancellation_side(ctx, focus_expr) else {
                continue;
            };

            let mut used_indices = Vec::new();
            let mut all_components_found = true;
            for (component_expr, component_sign) in matched.components {
                let expected_sign = if focus_sign == Sign::Pos {
                    component_sign.negate()
                } else {
                    component_sign
                };

                let mut found_index = None;
                for (term_index, (normalized_term_expr, normalized_term_sign, fast_log_term)) in
                    prepared_terms.iter().copied().enumerate()
                {
                    if term_index == focus_index || used_indices.contains(&term_index) {
                        continue;
                    }
                    if normalized_term_sign != expected_sign {
                        continue;
                    }
                    if log_cancellation_component_matches(
                        ctx,
                        normalized_term_expr,
                        fast_log_term,
                        component_expr,
                    ) {
                        found_index = Some(term_index);
                        break;
                    }
                }

                if let Some(term_index) = found_index {
                    used_indices.push(term_index);
                } else {
                    all_components_found = false;
                    break;
                }
            }

            if !all_components_found {
                continue;
            }

            let exact_identity_scope = used_indices.len() + 1 == view.terms.len();

            if exact_identity_scope {
                let zero = ctx.num(0);
                let focus_before_display = format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: ctx,
                        id: raw_focus_expr
                    }
                );
                let focus_after_display = format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: ctx,
                        id: matched.focus_after
                    }
                );

                return Some(
                    Rewrite::with_local(
                        zero,
                        "Log expansion followed by exact cancellation",
                        expr,
                        zero,
                    )
                    .substep(
                        "Expandir el logaritmo del producto o del cociente",
                        vec![format!(
                            "Reescribir {focus_before_display} como {focus_after_display}."
                        )],
                    )
                    .substep(
                        "Cancelar términos iguales",
                        vec![
                            "Tras la expansión, los términos opuestos se anulan y el resultado es 0."
                                .to_string(),
                        ],
                    ),
                );
            }

            let mut rebuilt_terms = smallvec::SmallVec::<[(cas_ast::ExprId, Sign); 8]>::new();
            for (term_index, (term_expr, term_sign)) in view.terms.iter().copied().enumerate() {
                if term_index == focus_index {
                    for (component_expr, component_sign) in matched.components {
                        let global_sign = if focus_sign == Sign::Pos {
                            component_sign
                        } else {
                            component_sign.negate()
                        };
                        rebuilt_terms.push((component_expr, global_sign));
                    }
                    continue;
                }
                rebuilt_terms.push((term_expr, term_sign));
            }

            let new_expr = AddView {
                root: expr,
                terms: rebuilt_terms,
            }
            .rebuild(ctx);
            let focus_after_local = if focus_sign == Sign::Pos {
                matched.focus_after
            } else {
                ctx.add(Expr::Neg(matched.focus_after))
            };

            return Some(Rewrite::with_local(
                new_expr,
                "Log expansion",
                raw_focus_expr,
                focus_after_local,
            ));
        }

        None
    }
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExactZeroCommonScaleRouteProfileFamily {
    SameDenominator,
    ResidualDirect,
    TailFastTrigRaw,
    TailFastTrigNormalized,
    TailTwoTermCoreEquivalence,
    Other,
}

#[derive(Clone, Copy)]
enum PositiveOnePassthroughProfileKind {
    Strippable,
    SinglePositiveOne,
    SingleOther,
    AddNoPositiveOne,
    AddOnlyPositiveOne,
}

#[derive(Clone, Copy)]
enum AddNoPositiveOneProfileDetail {
    NegativeOne,
    OtherNumeric,
    NonNumeric,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SingleHyperbolicTermRejectProfile {
    builtin: BuiltinFn,
    arg: cas_ast::ExprId,
}

#[derive(Clone, Copy)]
enum DirectCoreEquivalenceProfileFamily {
    DirectMatch,
    SymbolicScaleSumLhs,
    SymbolicScaleSumRhs,
    LogExpansion,
    LogChainProduct,
    TrigReciprocal,
    CosDiffSinDiffQuotient,
    SumDiffCubesQuotient,
    PhaseShiftIdentity,
    CosProductTelescoping,
    FiniteSum,
    FiniteProduct,
    TrigPowerReduction,
    DoubleAngleContraction,
    DefaultSimplify,
    Other,
}

#[derive(Clone, Copy)]
enum QuotientCancelProfileShape {
    Radical,
    Polynomial,
    Monomial,
    Other,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LogProfileKind {
    Ln,
    GeneralBase,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LogArgProfileShape {
    Atom,
    AddSub,
    Mul,
    Div,
    Pow,
    Abs,
    Function,
    Other,
}

#[derive(Clone, Copy)]
struct LogProfileMember {
    negated: bool,
    kind: LogProfileKind,
    arg_shape: LogArgProfileShape,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PowerMergeExponentProfileKind {
    Integer,
    Fractional,
    Symbolic,
}

#[derive(Clone, Copy)]
enum DirectCoreDefaultSimplifyProfileScope {
    SameDenominatorTail,
    ShiftedQuotientExactOne,
}

define_rule!(
    CollapseExactZeroThreeTermSubsetRule,
    "Collapse Exact Zero Additive Subexpression",
    Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB)),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    priority: 509,
    |ctx, expr| {
        if default_simplify_nesting_depth() > 0 {
            return None;
        }

        let add_view = AddView::from_expr(ctx, expr);
        let term_count = add_view.terms.len();
        if same_arg_sin_cos_additive_pair(ctx, expr) {
            return None;
        }
        let normalized_terms: Vec<_> = add_view
            .terms
            .iter()
            .copied()
            .map(|(term_expr, term_sign)| normalize_signed_add_term(ctx, term_expr, term_sign))
            .collect();
        let has_negative_contribution = normalized_terms
            .iter()
            .any(|(_, sign)| *sign == Sign::Neg);

        if !has_negative_contribution {
            return None;
        }
        if has_atanh_common_log_mismatch_with_plain_passthrough(ctx, expr) {
            return None;
        }
        if let Some((lhs_core, rhs_core)) = extract_two_term_core_difference(ctx, expr) {
            if is_atanh_common_log_definition_mismatch_pair(ctx, lhs_core, rhs_core) {
                return None;
            }
            if let Some(rewrite) =
                try_build_unit_fraction_trig_denominator_equivalence_zero_core_rewrite(ctx, expr)
            {
                return Some(rewrite);
            }
            if let Some(rewrite) =
                try_build_direct_trig_power_reduction_equivalence_rewrite(ctx, lhs_core, rhs_core)
            {
                return Some(rewrite);
            }
            if let Some(rewrite) =
                try_build_exact_zero_squared_shared_passthrough_difference_rewrite(ctx, expr)
            {
                return Some(rewrite);
            }
        }
        if let Some(rewrite) =
            try_build_exact_trig_product_to_sum_sin_sin_three_term_zero_rewrite(ctx, expr)
        {
            return Some(rewrite);
        }
        let has_same_denominator_residual =
            term_count == 2 && extract_same_denominator_residual_cores(ctx, expr).is_some();
        let has_nontrivial_common_scale = term_count == 2
            && (extract_common_multiplicative_residual_sum(ctx, expr).is_some()
                || has_same_denominator_residual);
        if term_count >= 4 && !has_nontrivial_common_scale {
            if let Some(rewrite) =
                try_build_exact_zero_shared_passthrough_difference_rewrite(ctx, expr)
            {
                return Some(rewrite);
            }
        }
        if let Some(rewrite) = try_build_direct_small_zero_additive_combination_rewrite(ctx, expr)
        {
            return Some(rewrite);
        }
        if let Some(rewrite) = try_build_structural_three_term_poly_zero_rewrite(ctx, expr) {
            return Some(rewrite);
        }
        let maybe_two_term_direct_trig_or_hyperbolic_identity = term_count == 2
            && !has_nontrivial_common_scale
            && expr_contains_any_builtin(
                ctx,
                expr,
                &[
                    BuiltinFn::Sin,
                    BuiltinFn::Cos,
                    BuiltinFn::Tan,
                    BuiltinFn::Cot,
                    BuiltinFn::Sec,
                    BuiltinFn::Csc,
                    BuiltinFn::Sinh,
                    BuiltinFn::Cosh,
                    BuiltinFn::Tanh,
                ],
            );
        let maybe_repeated_phase_shift_with_passthrough = term_count == 8
            && expr_contains_any_builtin(ctx, expr, &[BuiltinFn::Sin, BuiltinFn::Cos]);
        let maybe_two_term_direct_half_angle_square = term_count == 2
            && !has_nontrivial_common_scale
            && expr_contains_any_builtin(ctx, expr, &[BuiltinFn::Sin, BuiltinFn::Cos]);
        let maybe_two_term_direct_symbolic_root_denesting_identity = term_count == 2
            && !has_nontrivial_common_scale
            && expr_contains_sqrt_or_half_power(ctx, expr)
            && extract_two_term_core_difference(ctx, expr).is_some_and(|(lhs_core, rhs_core)| {
                try_match_symbolic_root_denesting_pair(ctx, lhs_core).is_some()
                    || try_match_symbolic_root_denesting_pair(ctx, rhs_core).is_some()
            });
        if term_count == 2 {
            if let Some((lhs_core, rhs_core)) = extract_two_term_core_difference(ctx, expr) {
                if let Some(rewrite) =
                    try_build_direct_negative_even_root_power_reciprocal_rewrite(
                        ctx, lhs_core, rhs_core,
                    )
                {
                    return Some(rewrite);
                }
                if let Some(rewrite) =
                    try_build_direct_reciprocal_half_power_product_rewrite(ctx, lhs_core, rhs_core)
                {
                    return Some(rewrite);
                }
                if let Some(rewrite) = try_build_direct_common_sqrt_denominator_fraction_rewrite(
                    ctx, lhs_core, rhs_core,
                ) {
                    return Some(rewrite);
                }
                if let Some(rewrite) =
                    try_build_direct_sqrt_over_base_fraction_rewrite(ctx, lhs_core, rhs_core)
                {
                    return Some(rewrite);
                }
            }
        }
        let maybe_small_trig_direct_identity = (2..=3).contains(&term_count)
            && !has_nontrivial_common_scale
            && expr_contains_any_builtin(
                ctx,
                expr,
                &[
                    BuiltinFn::Sin,
                    BuiltinFn::Cos,
                    BuiltinFn::Tan,
                    BuiltinFn::Cot,
                    BuiltinFn::Sec,
                    BuiltinFn::Csc,
                ],
        );
        if maybe_small_trig_direct_identity {
            if additive_has_variable_scaled_direct_trig_or_hyperbolic_term(ctx, expr, 2)
                && !maybe_trig_power_reduction_zero_candidate(ctx, expr)
                && !maybe_trig_double_angle_cos_variant_zero_scope_candidate(ctx, expr)
            {
                return None;
            }
            if let Some(rewrite) = try_build_exact_zero_identity_rewrite_direct(ctx, expr) {
                return Some(rewrite);
            }
        }

        if let Some(rewrite) =
            try_build_exact_zero_hyperbolic_sinh_cubic_polynomial_zero_scope_rewrite(ctx, expr)
        {
            return Some(rewrite);
        }

        if has_nontrivial_common_scale {
            if let Some((common_factor, residual_expr)) =
                extract_common_multiplicative_residual_sum(ctx, expr)
            {
                if let Some((lhs_core, rhs_core)) =
                    extract_two_term_core_difference(ctx, residual_expr)
                {
                    if let Some(child_rewrite) =
                        try_build_direct_negative_even_root_power_reciprocal_rewrite(
                            ctx, lhs_core, rhs_core,
                        )
                    {
                        return Some(build_common_scale_exact_zero_rewrite(
                            ctx,
                            expr,
                            common_factor,
                            residual_expr,
                            child_rewrite,
                        ));
                    }
                }
            }
            return None;
        }

        if let Some(rewrite) = try_build_exact_zero_shared_passthrough_difference_rewrite(ctx, expr)
        {
            return Some(rewrite);
        }

        if maybe_repeated_phase_shift_with_passthrough {
            if let Some(rewrite) =
                try_build_repeated_trig_phase_shift_pair_with_canceling_passthrough_rewrite(
                    ctx, expr,
                )
            {
                return Some(rewrite);
            }
            if let Some(rewrite) =
                try_build_structural_cancel_then_exact_zero_subset_rewrite(ctx, expr)
            {
                return Some(rewrite);
            }
        }

        if maybe_two_term_direct_half_angle_square {
            if let Some(rewrite) = try_build_two_term_direct_half_angle_square_rewrite(ctx, expr) {
                return Some(rewrite);
            }
        }
        if maybe_two_term_direct_symbolic_root_denesting_identity {
            if let Some(rewrite) = try_build_exact_zero_identity_rewrite_direct(ctx, expr) {
                return Some(rewrite);
            }
        }

        if maybe_two_term_direct_trig_or_hyperbolic_identity {
            if let Some(rewrite) = try_build_exact_zero_identity_rewrite_direct(ctx, expr) {
                return Some(rewrite);
            }
        }

        if let Some(rewrite) = try_build_difference_of_equivalent_square_bases_zero_rewrite(ctx, expr)
        {
            return Some(rewrite);
        }

        if !maybe_exact_zero_additive_candidate(ctx, expr) {
            return None;
        }

        if let Some(rewrite) = try_build_repeated_trig_phase_shift_pair_zero_rewrite(ctx, expr) {
            return Some(rewrite);
        }

        if let Some(rewrite) = try_build_structural_cancel_then_exact_zero_subset_rewrite(ctx, expr)
        {
            return Some(rewrite);
        }

        if let Some(rewrite) = try_build_structural_cancel_subset_passthrough_rewrite(ctx, expr) {
            return Some(rewrite);
        }

        if let Some(rewrite) = try_build_exact_zero_identity_rewrite(ctx, expr) {
            return Some(rewrite);
        }

        try_build_exact_zero_three_term_subset_passthrough_rewrite(ctx, expr)
    }
);

define_rule!(
    CollapseExactZeroCommonScaledDifferenceRule,
    "Collapse Common-Scale Equivalent Difference",
    Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB)),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    priority: 509,
    |ctx, expr, parent_ctx| {
        if !maybe_exact_zero_common_scaled_difference_candidate(ctx, expr) {
            return None;
        }
        if is_risky_plain_trig_angle_pair_for_common_scale(ctx, expr) {
            return None;
        }
        let allow = cas_solver_core::undefined_risk_policy_support::allow_cancellation_with_undefined_risk_mode_flags(
            matches!(parent_ctx.domain_mode(), crate::DomainMode::Assume),
            matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict),
            crate::collect::has_undefined_risk(ctx, expr),
        );
        if !allow {
            return None;
        }

        try_build_exact_zero_common_scaled_difference_rewrite_with_context(ctx, expr, parent_ctx)
    }
);

define_rule!(
    CollapseExactZeroProductFactorRule,
    "Collapse Zero Product via Exact Residual",
    Some(crate::target_kind::TargetKindSet::MUL),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    priority: 509,
    |ctx, expr, parent_ctx| {
        try_build_exact_zero_product_factor_rewrite(ctx, expr, parent_ctx)
    }
);

define_rule!(
    CollapseExactOneShiftedQuotientRule,
    "Collapse Shifted Quotient of Equivalent Expressions",
    Some(crate::target_kind::TargetKindSet::DIV),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    priority: 509,
    |ctx, expr| {
        try_build_shifted_quotient_exact_one_rewrite(ctx, expr)
    }
);

define_rule!(
    SubSelfToZeroRule,
    "Subtraction Self-Cancel",
    priority: 500, // High priority: before any expansion rules
    |ctx, expr, parent_ctx| {
        let rewrite = try_rewrite_sub_self_zero_expr(ctx, expr)?;
        let allow =
            cas_solver_core::undefined_risk_policy_support::allow_cancellation_with_undefined_risk_mode_flags(
                matches!(parent_ctx.domain_mode(), crate::DomainMode::Assume),
                matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict),
                crate::collect::has_undefined_risk(ctx, rewrite.inner),
            );
        if !allow {
            return None;
        }

        Some(Rewrite::new(rewrite.rewritten).desc("a - a = 0"))
    }
);

define_rule!(
    SubtractExpandedSumDiffCubesQuotientRule,
    "Subtract Expanded Sum/Difference of Cubes Quotient",
    priority: 500,
    |ctx, expr, parent_ctx| {
        use crate::{ImplicitCondition, Predicate};

        let (lhs, rhs) = match ctx.get(expr) {
            Expr::Sub(lhs, rhs) => (*lhs, *rhs),
            Expr::Add(lhs, rhs) => match (ctx.get(*lhs), ctx.get(*rhs)) {
                (_, Expr::Neg(inner)) => (*lhs, *inner),
                (Expr::Neg(inner), _) => (*rhs, *inner),
                _ => return None,
            },
            _ => return None,
        };

        let (num, den) = match ctx.get(lhs) {
            Expr::Div(num, den) => (*num, *den),
            _ => return None,
        };

        let decision = crate::oracle_allows_with_hint(
            ctx,
            parent_ctx.domain_mode(),
            parent_ctx.value_domain(),
            &Predicate::NonZero(den),
            "Subtract Expanded Sum/Difference of Cubes Quotient",
        );
        if !decision.allow {
            return None;
        }

        let plan = crate::rules::algebra::fractions::try_plan_sum_diff_of_cubes_in_num(
            ctx, num, den, false,
        )?;

        let cancelled = canonicalize_nested_integer_powers(ctx, plan.cancelled_result);
        let rhs = canonicalize_nested_integer_powers(ctx, rhs);
        if !(cas_math::expr_domain::exprs_equivalent(ctx, cancelled, rhs)
            || exprs_equal_up_to_add_term_order(ctx, cancelled, rhs))
        {
            return None;
        }

        Some(
            Rewrite::new(ctx.num(0))
                .desc("((a^3 ± b^3)/(a ± b)) - expanded quotient = 0")
                .requires(ImplicitCondition::NonZero(den)),
        )
    }
);

// AddInverseRule: a + (-a) = 0
// Domain Mode Policy: Like other cancellation rules, we must respect domain_mode
// because if `a` can be undefined (e.g., x/(x+1) when x=-1), then a + (-a)
// is undefined, not 0.
// - Strict: only if `a` contains no potentially-undefined subexpressions (no variable denominator)
// - Assume: always apply (educational mode assumption: all expressions are defined)
// - Generic: same as Assume
//
// V2.12.13: REMOVED redundant "is defined" assumption event.
// The individual Div operations already emit NonZero(denominator) as Requires.
// Showing "a is defined" here is redundant and confusing.
define_rule!(AddInverseRule, "Add Inverse", |ctx, expr, parent_ctx| {
    let rewrite = try_rewrite_add_inverse_zero_expr(ctx, expr)?;
    let allow =
        cas_solver_core::undefined_risk_policy_support::allow_cancellation_with_undefined_risk_mode_flags(
            matches!(parent_ctx.domain_mode(), crate::DomainMode::Assume),
            matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict),
            crate::collect::has_undefined_risk(ctx, rewrite.inner),
        );
    if !allow {
        return None;
    }

    // V2.12.13: No assumption events - the division conditions are already
    // tracked as Requires from the original Div operations.
    // Adding "a is defined" here is redundant and clutters the output.
    Some(Rewrite::new(rewrite.rewritten).desc("a + (-a) = 0"))
});

#[cfg(test)]
mod tests;

// Simplify sums of fractions in exponents: x^(1/2 + 1/3) → x^(5/6)
// This makes the fraction sum visible as a step in the timeline.
define_rule!(
    SimplifyNumericExponentsRule,
    RULE_SUM_EXPONENTS,
    |ctx, expr| {
        let rewrite = try_rewrite_simplify_numeric_exponents_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.description))
    }
);

// =============================================================================
// NormalizeMulNegRule: Lift Neg out of Mul for canonical form
// =============================================================================
//
// Canonical form: Neg should be at the TOP of Mul, not buried inside.
// This unlocks cancellations in Add like: a*(-b) + (-a)*b → Neg(a*b) + Neg(a*b) → -2*a*b
//
// Rewrites:
// - Mul(Neg(a), b) → Neg(Mul(a, b))
// - Mul(a, Neg(b)) → Neg(Mul(a, b))
// - Mul(Neg(a), Neg(b)) → Mul(a, b)  (double negation cancels)
//
// This is idempotent and always reduces complexity.
// =============================================================================
define_rule!(
    NormalizeMulNegRule,
    "Normalize Negation in Product",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_normalize_mul_neg_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.description))
    }
);

mod fractions;
mod general;
mod hyperbolic;
mod logarithms;
mod phase_shift;
mod powers;
mod profiling;
mod solve_prep;
mod support;
mod trig;
mod trig_angles;
mod zero_collapse;

pub(crate) use fractions::*;
pub(crate) use general::*;
pub(crate) use hyperbolic::*;
use logarithms::*;
pub(crate) use phase_shift::*;
pub(crate) use powers::*;
pub(crate) use profiling::*;
pub(crate) use solve_prep::*;
pub(crate) use support::*;
pub(crate) use trig::*;
pub(crate) use trig_angles::*;
pub(crate) use zero_collapse::*;
