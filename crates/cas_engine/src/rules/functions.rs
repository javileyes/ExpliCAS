use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::abs_support::{
    abs_domain_mode_from_flags, abs_needs_implicit_domain_check, is_ln_or_log_call,
    try_plan_abs_nonnegative_rewrite, try_plan_abs_positive_rewrite,
    try_plan_symbolic_root_cancel_rewrite, try_rewrite_abs_even_power_expr,
    try_rewrite_abs_exp_identity_expr, try_rewrite_abs_idempotent_expr,
    try_rewrite_abs_numeric_factor_expr, try_rewrite_abs_odd_power_expr,
    try_rewrite_abs_power_even_expr, try_rewrite_abs_power_odd_magnitude_expr,
    try_rewrite_abs_product_identity_expr, try_rewrite_abs_quotient_identity_expr,
    try_rewrite_abs_sqrt_identity_expr, try_rewrite_abs_sub_normalize_expr,
    try_rewrite_abs_sum_nonnegative_expr, try_rewrite_evaluate_abs_expr,
    try_rewrite_sqrt_square_expr, try_unwrap_abs_arg, value_domain_mode_from_flag,
    AbsAssumptionKind, AbsDomainRewriteKind, AbsFixedRewriteKind, SymbolicRootCancelRewriteKind,
};
use cas_math::root_forms::try_rewrite_odd_half_power_expr;

fn format_abs_domain_rewrite_desc(kind: AbsDomainRewriteKind) -> &'static str {
    match kind {
        AbsDomainRewriteKind::Positive => "|x| = x for x > 0",
        AbsDomainRewriteKind::PositiveAssume => "|x| = x (assuming x > 0)",
        AbsDomainRewriteKind::NonNegative => "|x| = x for x >= 0",
        AbsDomainRewriteKind::NonNegativeAssume => "|x| = x (assuming x >= 0)",
    }
}

fn format_symbolic_root_cancel_desc(kind: SymbolicRootCancelRewriteKind) -> &'static str {
    match kind {
        SymbolicRootCancelRewriteKind::AssumeNonNegative => "sqrt(x^n, n) = x (assuming x >= 0)",
    }
}

fn format_abs_fixed_rewrite_desc(kind: AbsFixedRewriteKind) -> &'static str {
    match kind {
        AbsFixedRewriteKind::Idempotent => "||x|| = |x|",
        AbsFixedRewriteKind::SqrtSquare => "sqrt(x^2) = |x|",
        AbsFixedRewriteKind::SumNonnegative => "|x² + ...| = x² + ...",
        AbsFixedRewriteKind::SubNormalize => "|a−b| = |b−a|",
        AbsFixedRewriteKind::ProductIdentity => "|x|·|y| = |x·y|",
        AbsFixedRewriteKind::QuotientIdentity => "|x| / |y| = |x / y|",
        AbsFixedRewriteKind::SqrtIdentity => "|√x| = √x",
        AbsFixedRewriteKind::ExpIdentity => "|e^x| = e^x",
    }
}

define_rule!(EvaluateAbsRule, "Evaluate Absolute Value", |ctx, expr| {
    let rewrite = try_rewrite_evaluate_abs_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
});

/// V2.14.20: Simplify absolute value under positivity
/// |x| → x when x > 0 is proven or assumed (depending on DomainMode)
pub struct AbsPositiveSimplifyRule;

impl crate::rule::Rule for AbsPositiveSimplifyRule {
    fn name(&self) -> &str {
        "Abs Under Positivity"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::helpers::prove_positive;
        use crate::{DomainMode, Proof};

        let vd = parent_ctx.value_domain();
        let dm = parent_ctx.domain_mode();
        let inner = try_unwrap_abs_arg(ctx, expr)?;
        let pos = prove_positive(ctx, inner, vd);
        let proven = pos == Proof::Proven;
        let mode = abs_domain_mode_from_flags(
            matches!(dm, DomainMode::Assume),
            matches!(dm, DomainMode::Strict),
        );

        // Only needed in Strict/Generic to accept inherited positivity from sticky implicit domain.
        let implied = if abs_needs_implicit_domain_check(mode, proven) {
            if let Some(id) = parent_ctx.implicit_domain() {
                let dc = crate::DomainContext::new(id.conditions().iter().cloned().collect());
                let cond = crate::ImplicitCondition::Positive(inner);
                dc.is_condition_implied(ctx, &cond)
            } else {
                false
            }
        } else {
            false
        };

        let plan = try_plan_abs_positive_rewrite(ctx, expr, mode, proven, implied)?;
        let mut rewrite = Rewrite::new(plan.rewritten)
            .desc(format_abs_domain_rewrite_desc(plan.kind))
            .local(expr, plan.rewritten);

        if matches!(plan.assumption, Some(AbsAssumptionKind::Positive)) {
            rewrite = rewrite.assume(crate::AssumptionEvent::positive_assumed(ctx, inner));
        }
        Some(rewrite)
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    // V2.14.20: Run in POST phase only so |a| created by LogPowerRule exists first
    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        crate::phase::PhaseMask::POST
    }

    // V2.14.21: Ensure step is visible - domain simplification is didactically important
    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

/// Simplify absolute value under non-negativity
/// |x| → x when x ≥ 0 is proven or implied (e.g., from sqrt(x) requirements)
/// This complements AbsPositiveSimplifyRule for the non-strict case
pub struct AbsNonNegativeSimplifyRule;

impl crate::rule::Rule for AbsNonNegativeSimplifyRule {
    fn name(&self) -> &str {
        "Abs Under Non-Negativity"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::helpers::prove_nonnegative;
        use crate::{DomainMode, Proof};

        let vd = parent_ctx.value_domain();
        let dm = parent_ctx.domain_mode();
        let inner = try_unwrap_abs_arg(ctx, expr)?;
        let nonneg = prove_nonnegative(ctx, inner, vd);
        let proven = nonneg == Proof::Proven;
        let mode = abs_domain_mode_from_flags(
            matches!(dm, DomainMode::Assume),
            matches!(dm, DomainMode::Strict),
        );

        // Only needed in Strict/Generic to accept inherited non-negativity from sticky implicit domain.
        let implied = if abs_needs_implicit_domain_check(mode, proven) {
            if let Some(id) = parent_ctx.implicit_domain() {
                let dc = crate::DomainContext::new(id.conditions().iter().cloned().collect());
                let cond = crate::ImplicitCondition::NonNegative(inner);
                dc.is_condition_implied(ctx, &cond)
            } else {
                false
            }
        } else {
            false
        };

        let plan = try_plan_abs_nonnegative_rewrite(ctx, expr, mode, proven, implied)?;
        let mut rewrite = Rewrite::new(plan.rewritten)
            .desc(format_abs_domain_rewrite_desc(plan.kind))
            .local(expr, plan.rewritten);

        if matches!(plan.assumption, Some(AbsAssumptionKind::NonNegative)) {
            rewrite = rewrite.assume(crate::AssumptionEvent::nonnegative(ctx, inner));
        }
        Some(rewrite)
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    // Run in POST phase only, after abs values from other rules exist
    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        crate::phase::PhaseMask::POST
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

/// AbsSquaredRule: |x|^(2k) → x^(2k) for even integer k
///
/// This rule simplifies absolute values raised to even powers since the result
/// is always non-negative. However, we SKIP this transformation when the parent
/// is a logarithm (ln, log) because it would prevent the more educational
/// transformation ln(|x|^n) → n·ln(|x|).
///
/// V2.15.9: Converted from define_rule! to structured Rule to access parent_ctx.
pub struct AbsSquaredRule;

impl crate::rule::Rule for AbsSquaredRule {
    fn name(&self) -> &str {
        "Abs Squared Identity"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        // V2.15.9: Skip if parent is ln or log to allow LogAbsPowerRule to apply first
        if parent_ctx
            .immediate_parent()
            .is_some_and(|parent_id| is_ln_or_log_call(ctx, parent_id))
        {
            return None;
        }

        let rewrite = try_rewrite_abs_power_even_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::POW)
    }

    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::CORE | PhaseMask::TRANSFORM | PhaseMask::RATIONALIZE
    }
}

/// AbsPowerOddMagnitudeRule: |x|^(2k+1) → x^(2k)·|x|
///
/// This canonical form exposes the even-power part as a plain polynomial
/// factor while preserving one magnitude atom. As with AbsSquaredRule, skip
/// under ln/log so logarithm-specific rules can keep the didactic |x|^n form.
pub struct AbsPowerOddMagnitudeRule;

impl crate::rule::Rule for AbsPowerOddMagnitudeRule {
    fn name(&self) -> &str {
        "Abs Odd Power Canonicalize"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        if parent_ctx
            .immediate_parent()
            .is_some_and(|parent_id| is_ln_or_log_call(ctx, parent_id))
        {
            return None;
        }

        let rewrite = try_rewrite_abs_power_odd_magnitude_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::POW)
    }

    fn allowed_phases(&self) -> PhaseMask {
        PhaseMask::CORE | PhaseMask::TRANSFORM | PhaseMask::RATIONALIZE
    }
}

define_rule!(
    SimplifySqrtSquareRule,
    "Simplify Square Root of Square",
    |ctx, expr| {
        let rewrite = try_rewrite_sqrt_square_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_abs_fixed_rewrite_desc(rewrite.kind)))
    }
);

// SimplifySqrtOddPowerRule: x^(n/2) -> |x|^k * sqrt(x) where n = 2k+1 (odd >= 3)
// Works on canonicalized form: sqrt(x^3) becomes x^(3/2) before reaching this rule
// Examples:
//   x^(3/2) -> |x| * sqrt(x)     (n=3, k=1)
//   x^(5/2) -> |x|^2 * sqrt(x)   (n=5, k=2)
//   x^(7/2) -> |x|^3 * sqrt(x)   (n=7, k=3)
define_rule!(
    SimplifySqrtOddPowerRule,
    "Simplify Odd Half-Integer Power",
    Some(crate::target_kind::TargetKindSet::POW), // Only match Pow expressions
    PhaseMask::POST, // Run in POST phase after canonicalization is done
    |ctx, expr| {
        let rewrite = try_rewrite_odd_half_power_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc_lazy(|| {
            format!(
                "x^({}/2) = |x|^{} * √x",
                rewrite.numerator, rewrite.abs_power
            )
        }))
    }
);

// ============================================================================
// SymbolicRootCancelRule: sqrt(x^n, n) → x when n is symbolic (Assume mode only)
// ============================================================================
//
// V2.14.45: When the index is symbolic (not a numeric literal), we can't
// determine parity to decide between x and |x|. In Assume mode, we simplify
// to x with the assumption x ≥ 0 (which makes both even and odd cases equivalent).
//
// CONTRACT: sqrt(x, n) / root(x, n) semantics assume n is a POSITIVE INTEGER.
// This is the standard mathematical definition of n-th root where n ∈ ℤ⁺.
// We do NOT emit requires for n ≠ 0 or n > 0 because this is implicit in the
// root function's domain definition.
//
// - Generic/Strict: block (handled by keeping sqrt form in CanonicalizeRootRule)
// - Assume: sqrt(x^n, n) → x with Requires: x ≥ 0
// ============================================================================
pub struct SymbolicRootCancelRule;

impl crate::rule::Rule for SymbolicRootCancelRule {
    fn name(&self) -> &str {
        "Symbolic Root Cancel"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        let domain_mode = parent_ctx.domain_mode();
        let mode = abs_domain_mode_from_flags(
            matches!(domain_mode, crate::DomainMode::Assume),
            matches!(domain_mode, crate::DomainMode::Strict),
        );
        let value_domain = value_domain_mode_from_flag(matches!(
            parent_ctx.value_domain(),
            crate::semantics::ValueDomain::RealOnly
        ));

        let plan = try_plan_symbolic_root_cancel_rewrite(ctx, expr, mode, value_domain)?;

        use crate::ImplicitCondition;
        let mut rewrite = crate::rule::Rewrite::new(plan.rewritten)
            .desc(format_symbolic_root_cancel_desc(plan.kind));
        if plan.requires_nonnegative {
            rewrite = rewrite.requires(ImplicitCondition::NonNegative(plan.rewritten));
        }
        if matches!(plan.assumption, Some(AbsAssumptionKind::NonNegative)) {
            rewrite = rewrite.assume(crate::AssumptionEvent::nonnegative(ctx, plan.rewritten));
        }
        Some(rewrite)
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

// EvaluateMetaFunctionsRule: Handles meta functions that operate on expressions
// - simplify(expr) → expr (already simplified by bottom-up processing)
// - factor(expr) → expr (factoring is done by other rules during simplification)
// - expand(expr) → expanded version (calls actual expand logic)
define_rule!(
    EvaluateMetaFunctionsRule,
    "Evaluate Meta Functions",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = crate::meta_functions_support::try_rewrite_meta_function_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// =============================================================================
// Abs Idempotent Rule: ||x|| → |x|
// Absolute value of absolute value is just absolute value
// =============================================================================
define_rule!(AbsIdempotentRule, "Abs Idempotent", |ctx, expr| {
    let rewrite = try_rewrite_abs_idempotent_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(format_abs_fixed_rewrite_desc(rewrite.kind)))
});

// =============================================================================
// Abs Of Even Power Rule: |x^(2k)| → x^(2k)
// Absolute value of even power is just the even power (always non-negative)
// =============================================================================
define_rule!(AbsOfEvenPowerRule, "Abs Of Even Power", |ctx, expr| {
    let rewrite = try_rewrite_abs_even_power_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
});

// =============================================================================
// Abs Pow Odd Integer Rule: |x^n| → |x|^n for positive odd integer n
// This canonicalizes the abs-power form so that `abs(x^5)` and `abs(x)^5`
// converge to the same AST, enabling structural cancellation in the solver.
// Even powers are handled by AbsOfEvenPowerRule (|x^2k| → x^2k).
// =============================================================================
define_rule!(
    AbsPowOddIntegerRule,
    "Abs Distribute Over Odd Power",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    |ctx, expr| {
        let rewrite = try_rewrite_abs_odd_power_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// =============================================================================
// Abs Product Rule: |x| * |y| → |x * y|
// Multiplicative property of absolute value
// =============================================================================
define_rule!(
    AbsProductRule,
    "Abs Product",
    Some(crate::target_kind::TargetKindSet::MUL),
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    |ctx, expr| {
        let rewrite = try_rewrite_abs_product_identity_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_abs_fixed_rewrite_desc(rewrite.kind)))
    }
);

// =============================================================================
// Abs Quotient Rule: |x| / |y| → |x / y|
// Quotient property of absolute value
// =============================================================================
define_rule!(
    AbsQuotientRule,
    "Abs Quotient",
    Some(crate::target_kind::TargetKindSet::DIV),
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    |ctx, expr| {
        let rewrite = try_rewrite_abs_quotient_identity_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_abs_fixed_rewrite_desc(rewrite.kind)))
    }
);

// =============================================================================
// Abs Sqrt Rule: |sqrt(x)| → sqrt(x)
// Square root is always non-negative (when it exists in reals)
// =============================================================================
define_rule!(AbsSqrtRule, "Abs Of Sqrt", |ctx, expr| {
    let rewrite = try_rewrite_abs_sqrt_identity_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(format_abs_fixed_rewrite_desc(rewrite.kind)))
});

// =============================================================================
// Abs Exp Rule: |e^x| → e^x
// Exponential is always positive
// =============================================================================
define_rule!(AbsExpRule, "Abs Of Exp", |ctx, expr| {
    let rewrite = try_rewrite_abs_exp_identity_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(format_abs_fixed_rewrite_desc(rewrite.kind)))
});

// =============================================================================
// Abs Sum Of Squares Rule: |x² + y²| → x² + y²
// Sum of squares is always non-negative
// =============================================================================
define_rule!(AbsSumOfSquaresRule, "Abs Of Sum Of Squares", |ctx, expr| {
    let rewrite = try_rewrite_abs_sum_nonnegative_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(format_abs_fixed_rewrite_desc(rewrite.kind)))
});

// =============================================================================
// Abs Sub Normalize Rule: |a - b| → |b - a|
// Canonicalize the argument of abs(Sub(..)) so that |a-b| and |b-a| produce
// the same normal form, enabling cancellation of |u-1| - |1-u| → 0.
// Uses compare_expr ordering: if a > b in canonical order, swap to |b-a|.
//
// V2.16: Relaxed from atoms-only to compound expressions with dedup node cap.
// This enables convergence for cases like |sin(u)-1| vs |1-sin(u)|.
// Guards: per-operand ≤ 20 dedup nodes, total abs expr ≤ 60 dedup nodes.
// =============================================================================
define_rule!(
    AbsSubNormalizeRule,
    "Abs Sub Normalize",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_abs_sub_normalize_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_abs_fixed_rewrite_desc(rewrite.kind)))
    }
);

// =============================================================================
// Abs Numeric Factor Rule: |k·x| → |k|·|x| for any nonzero numeric k
// Extracts numeric factors from absolute value (both positive and negative).
// Examples: |2u| → 2|u|,  |(-3)·sin(x)| → 3·|sin(x)|
// =============================================================================
fn format_abs_numeric_factor_desc(
    kind: cas_math::abs_support::AbsNumericFactorRewriteKind,
) -> &'static str {
    match kind {
        cas_math::abs_support::AbsNumericFactorRewriteKind::Positive => "|k·x| = k·|x| for k > 0",
        cas_math::abs_support::AbsNumericFactorRewriteKind::Negative => {
            "|(-k)·x| = |k|·|x| for k < 0"
        }
    }
}

define_rule!(
    AbsPositiveFactorRule,
    "Abs Positive Factor",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_abs_numeric_factor_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_abs_numeric_factor_desc(rewrite.kind)))
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(SimplifySqrtSquareRule)); // Must go BEFORE EvaluateAbsRule to catch sqrt(x^2) early
                                                           // V2.14.45: SimplifySqrtOddPowerRule DISABLED - causes split/merge cycle with ProductPowerRule
                                                           // x^(5/2) → |x|²*√x is a "worsening" transformation (increases AST nodes).
                                                           // The canonical form for odd half-integer powers is Pow(x, n/2), NOT the product form.
                                                           // If visual "extracted square" form is desired, it belongs in a renderer or explain-mode.
                                                           // simplifier.add_rule(Box::new(SimplifySqrtOddPowerRule)); // sqrt(x^3) -> |x| * sqrt(x)
    simplifier.add_rule(Box::new(SymbolicRootCancelRule)); // V2.14.45: sqrt(x^n, n) -> x in Assume mode
    simplifier.add_rule(Box::new(EvaluateAbsRule));
    simplifier.add_rule(Box::new(AbsPositiveSimplifyRule)); // V2.14.20: |x| -> x when x > 0
    simplifier.add_rule(Box::new(AbsNonNegativeSimplifyRule)); // |x| -> x when x >= 0 (from sqrt requirements)
    simplifier.add_rule(Box::new(AbsSquaredRule));
    simplifier.add_rule(Box::new(AbsPowerOddMagnitudeRule)); // |x|^(2k+1) -> x^(2k)*|x|
    simplifier.add_rule(Box::new(AbsIdempotentRule)); // ||x|| → |x|
    simplifier.add_rule(Box::new(AbsOfEvenPowerRule)); // |x^2k| → x^2k
    simplifier.add_rule(Box::new(AbsPowOddIntegerRule)); // |x^n| → |x|^n (odd n)
    simplifier.add_rule(Box::new(AbsProductRule)); // |x|*|y| → |xy|
    simplifier.add_rule(Box::new(AbsQuotientRule)); // |x|/|y| → |x/y|
    simplifier.add_rule(Box::new(AbsSqrtRule)); // |sqrt(x)| → sqrt(x)
    simplifier.add_rule(Box::new(AbsExpRule)); // |e^x| → e^x
    simplifier.add_rule(Box::new(AbsSumOfSquaresRule)); // |x² + y²| → x² + y²
    simplifier.add_rule(Box::new(AbsSubNormalizeRule)); // |a-b| → |b-a| (canonical)
    simplifier.add_rule(Box::new(AbsPositiveFactorRule)); // |k·x| → k·|x| for k > 0
    simplifier.add_rule(Box::new(EvaluateMetaFunctionsRule)); // Make simplify/factor/expand transparent
}
