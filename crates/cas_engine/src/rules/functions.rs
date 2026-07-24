use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use cas_math::abs_support::{
    abs_domain_mode_from_flags, abs_needs_implicit_domain_check, is_ln_or_log_call,
    try_plan_abs_negative_rewrite, try_plan_abs_nonnegative_rewrite, try_plan_abs_positive_rewrite,
    try_plan_symbolic_root_cancel_rewrite, try_rewrite_abs_even_power_expr,
    try_rewrite_abs_exp_identity_expr, try_rewrite_abs_idempotent_expr,
    try_rewrite_abs_numeric_factor_expr, try_rewrite_abs_odd_power_expr,
    try_rewrite_abs_power_even_expr, try_rewrite_abs_power_odd_magnitude_expr,
    try_rewrite_abs_product_identity_expr, try_rewrite_abs_quotient_identity_expr,
    try_rewrite_abs_quotient_sub_normalize_expr, try_rewrite_abs_sqrt_identity_expr,
    try_rewrite_abs_sub_normalize_expr, try_rewrite_abs_sum_nonnegative_expr,
    try_rewrite_evaluate_abs_expr, try_rewrite_evaluate_sign_expr,
    try_rewrite_even_power_over_abs_expr, try_rewrite_sqrt_square_expr, try_unwrap_abs_arg,
    value_domain_mode_from_flag, AbsAssumptionKind, AbsDomainRewriteKind, AbsFixedRewriteKind,
    SymbolicRootCancelRewriteKind,
};
use cas_math::difference_of_cubes_support::try_rewrite_cbrt_perfect_cube_expr;
use cas_math::expr_nary::{AddView, Sign};
use cas_math::root_forms::try_rewrite_odd_half_power_expr;
use cas_solver_core::rule_names::{RULE_ABS_UNDER_NON_NEGATIVITY, RULE_ABS_UNDER_POSITIVITY};

fn format_abs_domain_rewrite_desc(kind: AbsDomainRewriteKind) -> &'static str {
    match kind {
        AbsDomainRewriteKind::Positive => "|x| = x for x > 0",
        AbsDomainRewriteKind::PositiveAssume => "|x| = x (assuming x > 0)",
        AbsDomainRewriteKind::Negative => "|x| = -x for x <= 0",
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
        AbsFixedRewriteKind::QuotientSubNormalize => "|(a−b)/c| = |(b−a)/c|",
        AbsFixedRewriteKind::ProductIdentity => "|x|·|y| = |x·y|",
        AbsFixedRewriteKind::QuotientIdentity => "|x| / |y| = |x / y|",
        AbsFixedRewriteKind::SqrtIdentity => "|√x| = √x",
        AbsFixedRewriteKind::ExpIdentity => "|e^x| = e^x",
    }
}

fn expr_contains_structural(ctx: &Context, haystack: ExprId, needle: ExprId) -> bool {
    if haystack == needle || cas_ast::ordering::compare_expr(ctx, haystack, needle).is_eq() {
        return true;
    }

    match ctx.get(haystack) {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            expr_contains_structural(ctx, *l, needle) || expr_contains_structural(ctx, *r, needle)
        }
        Expr::Pow(base, exp) => {
            expr_contains_structural(ctx, *base, needle)
                || expr_contains_structural(ctx, *exp, needle)
        }
        Expr::Neg(inner) => expr_contains_structural(ctx, *inner, needle),
        Expr::Function(_, args) => args
            .iter()
            .any(|arg| expr_contains_structural(ctx, *arg, needle)),
        _ => false,
    }
}

fn expr_structurally_equal(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    left == right || cas_ast::ordering::compare_expr(ctx, left, right).is_eq()
}

fn try_unwrap_neg_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Neg(inner) => Some(*inner),
        _ => None,
    }
}

fn merged_implicit_domain_conditions(
    ctx: &mut Context,
    parent_ctx: &crate::parent_context::ParentContext,
) -> Vec<crate::ImplicitCondition> {
    parent_ctx
        .implicit_domain()
        .cloned()
        .or_else(|| {
            parent_ctx
                .root_expr()
                .map(|root| crate::infer_implicit_domain(ctx, root, parent_ctx.value_domain()))
        })
        .into_iter()
        .flat_map(|id| id.conditions().iter().cloned().collect::<Vec<_>>())
        .collect()
}

fn remaining_add_terms_cancel_structurally(
    ctx: &Context,
    terms: &[(ExprId, Sign)],
    first_skip: usize,
    second_skip: usize,
) -> bool {
    let mut used = vec![false; terms.len()];
    used[first_skip] = true;
    used[second_skip] = true;

    for i in 0..terms.len() {
        if used[i] {
            continue;
        }
        let (left_expr, left_sign) = terms[i];
        let mut matched = false;
        for j in (i + 1)..terms.len() {
            if used[j] {
                continue;
            }
            let (right_expr, right_sign) = terms[j];
            if left_sign != right_sign && expr_structurally_equal(ctx, left_expr, right_expr) {
                used[i] = true;
                used[j] = true;
                matched = true;
                break;
            }
        }
        if !matched {
            return false;
        }
    }

    true
}

fn abs_add_term_pair_is_domain_zero(
    ctx: &mut Context,
    dc: &crate::DomainContext,
    inner: ExprId,
    paired_sign: Sign,
) -> bool {
    match paired_sign {
        Sign::Neg => {
            dc.is_condition_implied(ctx, &crate::ImplicitCondition::NonNegative(inner))
                || dc.is_condition_implied(ctx, &crate::ImplicitCondition::Positive(inner))
        }
        Sign::Pos => {
            let neg_inner = ctx.add(Expr::Neg(inner));
            dc.is_condition_implied(ctx, &crate::ImplicitCondition::NonNegative(neg_inner))
                || dc.is_condition_implied(ctx, &crate::ImplicitCondition::Positive(neg_inner))
        }
    }
}

fn try_cancel_abs_add_view_under_domain(
    ctx: &mut Context,
    expr: ExprId,
    dc: &crate::DomainContext,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }
    let terms: Vec<_> = view.terms.iter().copied().collect();

    for (abs_index, (abs_expr, abs_sign)) in terms.iter().copied().enumerate() {
        if abs_sign != Sign::Pos {
            continue;
        }
        let Some(inner) = try_unwrap_abs_arg(ctx, abs_expr) else {
            continue;
        };
        for (other_index, (other_expr, other_sign)) in terms.iter().copied().enumerate() {
            if other_index == abs_index || !expr_structurally_equal(ctx, inner, other_expr) {
                continue;
            }
            if !abs_add_term_pair_is_domain_zero(ctx, dc, inner, other_sign) {
                continue;
            }
            if !remaining_add_terms_cancel_structurally(ctx, &terms, abs_index, other_index) {
                continue;
            }

            let zero = ctx.num(0);
            let desc = match other_sign {
                Sign::Neg => "|x| - x = 0 for x >= 0",
                Sign::Pos => "|x| + x = 0 for x <= 0",
            };
            return Some(Rewrite::new(zero).desc(desc).local(expr, zero));
        }
    }

    None
}

define_rule!(
    EvaluateAbsRule,
    "Evaluate Absolute Value",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_evaluate_abs_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(
    EvaluateSignRule,
    "Evaluate Sign",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let (rewritten, desc) = try_rewrite_evaluate_sign_expr(ctx, expr)?;
        Some(Rewrite::new(rewritten).desc(desc))
    }
);

define_rule!(
    EvaluateFloorCeilRoundRule,
    "Evaluate Floor/Ceil/Round",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        // floor/ceil/round of a foldable RATIONAL constant -> its exact integer
        // (`floor(7/2) -> 3`, `ceil(7/2) -> 4`, `round(5/2) -> 3`, `floor(5) -> 5`).
        let rewritten = cas_math::const_eval::try_eval_floor_ceil_round(ctx, expr)?;
        Some(Rewrite::new(rewritten).desc("Evaluate floor/ceil/round of a constant"))
    }
);

define_rule!(
    EvenPowerOverAbsRule,
    "Even Power Over Absolute Value",
    Some(crate::target_kind::TargetKindSet::DIV),
    |ctx, expr, parent_ctx| {
        // x^(2k)/|x| -> |x|^(2k-1), a removable-singularity cancellation that
        // keeps the x != 0 condition (the same treatment as x^2/x -> x).
        // REAL-ONLY: over ℂ, z^(2k) ≠ |z|^(2k), so the cancellation is unsound.
        if parent_ctx.value_domain() != crate::semantics::ValueDomain::RealOnly {
            return None;
        }
        let (rewritten, base) = try_rewrite_even_power_over_abs_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewritten)
                .requires(crate::ImplicitCondition::NonZero(base))
                .desc("x^(2k)/|x| = |x|^(2k-1)"),
        )
    }
);

/// V2.14.20: Simplify absolute value under positivity
/// |x| → x when x > 0 is proven or assumed (depending on DomainMode)
pub struct AbsPositiveSimplifyRule;

impl crate::rule::Rule for AbsPositiveSimplifyRule {
    fn name(&self) -> &str {
        RULE_ABS_UNDER_POSITIVITY
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
            let conditions = merged_implicit_domain_conditions(ctx, parent_ctx);
            if !conditions.is_empty() {
                let dc = crate::DomainContext::new(conditions);
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

/// Simplify absolute value under inherited non-positivity.
///
/// This consumes existing domain facts such as `-x >= 0`; it never introduces a
/// new negative assumption.
pub struct AbsNegativeSimplifyRule;

impl crate::rule::Rule for AbsNegativeSimplifyRule {
    fn name(&self) -> &str {
        "Abs Under Negativity"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        let inner = try_unwrap_abs_arg(ctx, expr)?;
        let implicit_domain: Option<crate::ImplicitDomain> =
            parent_ctx.implicit_domain().cloned().or_else(|| {
                parent_ctx
                    .root_expr()
                    .map(|root| crate::infer_implicit_domain(ctx, root, parent_ctx.value_domain()))
            });
        let neg_inner = ctx.add(Expr::Neg(inner));
        // Exact-prover arm, mirroring `AbsPositiveSimplifyRule`: `x < 0` is
        // `−x > 0` through the same const/surd sign layer (π, e, φ, surds).
        // Without it the negated sibling never resolved (`|3 − π|` survived
        // while `|π − 3|` folded) and the result depended on whether the
        // canonical-negation flip ran before or after the positivity rule.
        let proven_negative =
            crate::helpers::prove_positive(ctx, neg_inner, parent_ctx.value_domain())
                == crate::Proof::Proven;
        let negative_implied = proven_negative
            || implicit_domain.as_ref().is_some_and(|id| {
                let dc = crate::DomainContext::new(id.conditions().iter().cloned().collect());
                dc.is_condition_implied(ctx, &crate::ImplicitCondition::NonNegative(neg_inner))
            });

        let plan = try_plan_abs_negative_rewrite(ctx, expr, negative_implied)?;
        Some(
            Rewrite::new(plan.rewritten)
                .desc(format_abs_domain_rewrite_desc(plan.kind))
                .local(expr, plan.rewritten),
        )
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST
    }

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
        RULE_ABS_UNDER_NON_NEGATIVITY
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
            let conditions = merged_implicit_domain_conditions(ctx, parent_ctx);
            if !conditions.is_empty() {
                let dc = crate::DomainContext::new(conditions);
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

/// Cancel |u| - u or |u| + u when the root implicit domain proves u has the
/// required sign. This is intentionally narrower than general arithmetic
/// cancellation: it only consumes an exact abs atom paired with the same inner.
pub struct AbsDomainAddSubCancellationRule;

impl crate::rule::Rule for AbsDomainAddSubCancellationRule {
    fn name(&self) -> &str {
        "Abs Domain Add/Sub Cancellation"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        let conditions = merged_implicit_domain_conditions(ctx, parent_ctx);
        if conditions.is_empty() {
            return None;
        }
        let dc = crate::DomainContext::new(conditions);

        if let Some(rewrite) = try_cancel_abs_add_view_under_domain(ctx, expr, &dc) {
            return Some(rewrite);
        }

        match ctx.get(expr).clone() {
            Expr::Sub(left, right) => {
                let inner = try_unwrap_abs_arg(ctx, left)?;
                if !expr_structurally_equal(ctx, inner, right) {
                    return None;
                }
                let nonnegative = dc
                    .is_condition_implied(ctx, &crate::ImplicitCondition::NonNegative(inner))
                    || dc.is_condition_implied(ctx, &crate::ImplicitCondition::Positive(inner));
                if !nonnegative {
                    return None;
                }
                let zero = ctx.num(0);
                Some(
                    Rewrite::new(zero)
                        .desc("|x| - x = 0 for x >= 0")
                        .local(expr, zero),
                )
            }
            Expr::Add(left, right) => {
                let (abs_expr, other) = if try_unwrap_abs_arg(ctx, left).is_some() {
                    (left, right)
                } else if try_unwrap_abs_arg(ctx, right).is_some() {
                    (right, left)
                } else {
                    return None;
                };
                let inner = try_unwrap_abs_arg(ctx, abs_expr)?;

                if let Some(negated_other) = try_unwrap_neg_arg(ctx, other) {
                    if !expr_structurally_equal(ctx, inner, negated_other) {
                        return None;
                    }
                    let nonnegative = dc
                        .is_condition_implied(ctx, &crate::ImplicitCondition::NonNegative(inner))
                        || dc.is_condition_implied(ctx, &crate::ImplicitCondition::Positive(inner));
                    if !nonnegative {
                        return None;
                    }
                    let zero = ctx.num(0);
                    return Some(
                        Rewrite::new(zero)
                            .desc("|x| - x = 0 for x >= 0")
                            .local(expr, zero),
                    );
                }

                if !expr_structurally_equal(ctx, inner, other) {
                    return None;
                }
                let neg_inner = ctx.add(Expr::Neg(inner));
                let nonpositive = dc
                    .is_condition_implied(ctx, &crate::ImplicitCondition::NonNegative(neg_inner))
                    || dc.is_condition_implied(ctx, &crate::ImplicitCondition::Positive(neg_inner));
                if !nonpositive {
                    return None;
                }
                let zero = ctx.num(0);
                Some(
                    Rewrite::new(zero)
                        .desc("|x| + x = 0 for x <= 0")
                        .local(expr, zero),
                )
            }
            _ => None,
        }
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD_SUB)
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn priority(&self) -> i32 {
        512
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
        // REAL-ONLY: over ℂ, |z|^(2k) = (z·z̄)^k ≠ z^(2k) — stripping the abs
        // is unsound for potentially-complex bases.
        if parent_ctx.value_domain() != crate::semantics::ValueDomain::RealOnly {
            return None;
        }
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
        // REAL-ONLY: |z|^(2k+1) → z^(2k)·|z| assumes |z|² = z², false over ℂ.
        if parent_ctx.value_domain() != crate::semantics::ValueDomain::RealOnly {
            return None;
        }
        if parent_ctx
            .immediate_parent()
            .is_some_and(|parent_id| is_ln_or_log_call(ctx, parent_id))
        {
            return None;
        }

        let Expr::Pow(abs_base, _) = ctx.get(expr) else {
            return None;
        };
        let abs_base = *abs_base;
        let _ = try_unwrap_abs_arg(ctx, abs_base)?;

        for &ancestor in parent_ctx.all_ancestors() {
            let Expr::Div(num, den) = ctx.get(ancestor) else {
                continue;
            };
            if !expr_contains_structural(ctx, *num, expr) {
                continue;
            }
            let den_has_shared_abs = cas_math::expr_sub_like::extract_sub_like_pair(ctx, *den)
                .map(|(a, b)| {
                    expr_contains_structural(ctx, a, abs_base)
                        || expr_contains_structural(ctx, b, abs_base)
                })
                .unwrap_or(false);
            if den_has_shared_abs {
                return None;
            }
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
    Some(crate::target_kind::TargetKindSet::FUNCTION | crate::target_kind::TargetKindSet::POW),
    |ctx, expr, parent_ctx| {
        // REAL-ONLY: `√(z²) = |z|` is a real-domain identity — over ℂ,
        // `√(i²) = √(-1) = i` while `|i| = 1`.
        if parent_ctx.value_domain() != crate::semantics::ValueDomain::RealOnly {
            return None;
        }
        let rewrite = try_rewrite_sqrt_square_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_abs_fixed_rewrite_desc(rewrite.kind)))
    }
);

// EvaluateCbrtPerfectCubeRule: cbrt(8) → 2, cbrt(-8) → -2, cbrt(8/27) → 2/3.
// Mirrors the perfect-square folding of sqrt; a non-cube argument (cbrt(2), cbrt(16)) is
// left symbolic, so only exact rational cube roots collapse.
define_rule!(
    EvaluateCbrtPerfectCubeRule,
    "Evaluate Cube Root of Perfect Cube",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewritten = try_rewrite_cbrt_perfect_cube_expr(ctx, expr)?;
        Some(Rewrite::new(rewritten).desc("∛ of a perfect cube"))
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

/// True when `expr` still contains an UNEVALUATED calculus call — the order
/// guard of `SubsRule` (see below): substitution must wait for those to
/// evaluate first, or `subs(diff(f,x), x, 1)` would bind the differentiation
/// variable before differentiating (the REPL let-flow's order trap).
pub(crate) fn contains_unevaluated_calculus_call(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    use cas_ast::Expr;
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Function(f, args) => {
                let name = ctx.sym_name(*f);
                if matches!(
                    name,
                    "diff"
                        | "integrate"
                        | "limit"
                        | "sum"
                        | "product"
                        | "root_sum"
                        | "taylor"
                        | "series"
                        | "subs"
                ) {
                    return true;
                }
                stack.extend(args.iter().copied());
            }
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

// SubsRule: inline evaluation-at-a-point `subs(expr, x, value)` (cierre del
// frente vectorial, pregunta abierta #2 — decisión del usuario 2026-07-18).
// ORDER-SAFE BY CONSTRUCTION: declines while the target still contains an
// unevaluated calculus call — the cascade evaluates those first, so
// `subs(diff(x^2*y, x), x, 1)` differentiates BEFORE binding (the let-flow's
// order trap cannot happen here); a residual calculus call keeps `subs` an
// honest residual. Multi-variable evaluation nests: `subs(subs(f,x,1),y,2)`
// (the inner guard also waits for inner subs). Purely syntactic node
// substitution — no ValueDomain gate (guardrail #1: no ceremony).
define_rule!(
    SubsRule,
    "Substitute Value",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr, parent_ctx| {
        if !ctx.is_call_named(expr, "subs") {
            return None;
        }
        // F3 (Fase 3): a NESTED subs chain collapses at the OUTERMOST call in
        // ONE rewrite. Resolving child-first (the bottom-up default) makes the
        // parent's result re-observe the child's fingerprint, and the
        // phase-global CycleDetector reads that convergence as a period-1
        // cycle — blocked-hint noise ("requires cos(t) (defined)") on every
        // per-component circulation / Green composition. An inner subs on the
        // TARGET SPINE of an ancestor subs therefore declines and lets the
        // outermost collapse the whole chain.
        if parent_ctx.has_ancestor_matching(ctx, |ctx, ancestor| {
            subs_target_spine_contains(ctx, ancestor, expr)
        }) {
            return None;
        }
        let (base, bindings) = peel_subs_target_spine(ctx, expr)?;
        if contains_unevaluated_calculus_call(ctx, base) {
            return None;
        }
        // Apply the bindings innermost-first; a binding whose variable is not
        // mentioned is a NO-OP and is skipped (substituting anyway would
        // rebuild a content-equal copy — the other half of the false-cycle
        // noise).
        let mut result = base;
        for (var_expr, var_sym, value) in bindings.into_iter().rev() {
            if expr_mentions_variable(ctx, result, var_sym) {
                result = cas_ast::traversal::substitute_expr_by_id(ctx, result, var_expr, value);
            }
        }
        Some(Rewrite::new(result).desc("Sustituir la variable por el valor en la expresión"))
    }
);

/// Peel the nested-subs TARGET SPINE of `expr`: `subs(subs(T, x, v1), y, v2)`
/// yields base `T` and bindings `[(y, v2), (x, v1)]` (outermost first). Every
/// spine node must be a 3-argument `subs` with a pure `Variable` in second
/// position; the outermost call itself must be one too (else `None`).
#[allow(clippy::type_complexity)]
fn peel_subs_target_spine(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<(ExprId, cas_ast::symbol::SymbolId, ExprId)>)> {
    let mut bindings = Vec::new();
    let mut current = expr;
    loop {
        let cas_ast::Expr::Function(fn_id, args) = ctx.get(current) else {
            break;
        };
        let name = ctx.sym_name(*fn_id);
        if name != "subs" || args.len() != 3 {
            break;
        }
        let (target, var, value) = (args[0], args[1], args[2]);
        let cas_ast::Expr::Variable(var_sym) = ctx.get(var) else {
            break;
        };
        bindings.push((var, *var_sym, value));
        current = target;
    }
    if bindings.is_empty() {
        return None;
    }
    Some((current, bindings))
}

/// True when `needle` sits on the nested-subs TARGET SPINE of `ancestor`
/// (i.e. `ancestor` is a 3-argument `subs` whose first argument — possibly
/// through further nested `subs` targets — is `needle`).
fn subs_target_spine_contains(ctx: &Context, ancestor: ExprId, needle: ExprId) -> bool {
    let mut current = ancestor;
    loop {
        let cas_ast::Expr::Function(fn_id, args) = ctx.get(current) else {
            return false;
        };
        let name = ctx.sym_name(*fn_id);
        if name != "subs" || args.len() != 3 {
            return false;
        }
        if args[0] == needle {
            return true;
        }
        current = args[0];
    }
}

/// True when `expr` mentions the variable symbol anywhere.
fn expr_mentions_variable(ctx: &Context, expr: ExprId, var_sym: cas_ast::symbol::SymbolId) -> bool {
    use cas_ast::Expr;
    let mut stack = vec![expr];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Variable(sym) if *sym == var_sym => return true,
            Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => {
                stack.push(*a);
                stack.push(*b);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
        }
    }
    false
}

// EvaluateMetaFunctionsRule: Handles meta functions that operate on expressions
// - simplify(expr) → expr (already simplified by bottom-up processing)
// - factor(expr) → expr (factoring is done by other rules during simplification)
// - expand(expr) → expanded version (calls actual expand logic)
define_rule!(
    EvaluateMetaFunctionsRule,
    "Evaluate Meta Functions",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr, parent_ctx| {
        let complex_enabled =
            parent_ctx.value_domain() == crate::semantics::ValueDomain::ComplexEnabled;
        let rewrite = crate::meta_functions_support::try_rewrite_meta_function_expr_in_domain(
            ctx,
            expr,
            complex_enabled,
        )?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// =============================================================================
// Abs Idempotent Rule: ||x|| → |x|
// Absolute value of absolute value is just absolute value
// =============================================================================
define_rule!(
    AbsIdempotentRule,
    "Abs Idempotent",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr| {
        let rewrite = try_rewrite_abs_idempotent_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_abs_fixed_rewrite_desc(rewrite.kind)))
    }
);

// =============================================================================
// Abs Of Even Power Rule: |x^(2k)| → x^(2k)
// Absolute value of even power is just the even power (always non-negative)
// REAL-ONLY: over ℂ, |z^(2k)| = |z|^(2k) ≠ z^(2k) (e.g. |(1+i)²| = 2, not 2i).
// In complex mode the abs stays (honest); closed Gaussian args fold via
// GaussianAbsRule instead.
// =============================================================================
define_rule!(
    AbsOfEvenPowerRule,
    "Abs Of Even Power",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr, parent_ctx| {
        if parent_ctx.value_domain() != crate::semantics::ValueDomain::RealOnly {
            return None;
        }
        let rewrite = try_rewrite_abs_even_power_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

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
define_rule!(
    AbsSqrtRule,
    "Abs Of Sqrt",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr, parent_ctx| {
        // REAL-ONLY: over ℂ the principal √ of a negative/complex argument is
        // complex (`√(-4) = 2i`, `|2i| = 2 ≠ 2i`).
        if parent_ctx.value_domain() != crate::semantics::ValueDomain::RealOnly {
            return None;
        }
        let rewrite = try_rewrite_abs_sqrt_identity_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_abs_fixed_rewrite_desc(rewrite.kind)))
    }
);

// =============================================================================
// Abs Exp Rule: |e^x| → e^x
// Exponential is always positive
// =============================================================================
define_rule!(
    AbsExpRule,
    "Abs Of Exp",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr, parent_ctx| {
        // REAL-ONLY: over ℂ, `|e^z| = e^Re(z)` (`|e^(ix)| = 1 ≠ e^(ix)`).
        if parent_ctx.value_domain() != crate::semantics::ValueDomain::RealOnly {
            return None;
        }
        let rewrite = try_rewrite_abs_exp_identity_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_abs_fixed_rewrite_desc(rewrite.kind)))
    }
);

// =============================================================================
// Abs Sum Of Squares Rule: |x² + y²| → x² + y²
// Sum of squares is always non-negative
// =============================================================================
define_rule!(
    AbsSumOfSquaresRule,
    "Abs Of Sum Of Squares",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    |ctx, expr, parent_ctx| {
        // REAL-ONLY: "squares are nonnegative" is a real-domain fact — over ℂ,
        // `|z² (+ c)| ≠ z² + c` in general (`(1+i)² = 2i`).
        if parent_ctx.value_domain() != crate::semantics::ValueDomain::RealOnly {
            return None;
        }
        let rewrite = try_rewrite_abs_sum_nonnegative_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_abs_fixed_rewrite_desc(rewrite.kind)))
    }
);

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
    PhaseMask::POST,
    |ctx, expr, parent_ctx| {
        // A CONSTANT argument with a provable sign is owned by the
        // positivity/negativity rules — normalizing it here (single-pass
        // POST, after those rules already ran) flips it into a form nobody
        // revisits, so `|π − 3|` survived while `|3 − π|` folded and the
        // result depended on steps mode. Undecidable or variable-bearing
        // arguments keep the canonical flip (the dedup/cancellation use).
        if let Some(inner) = try_unwrap_abs_arg(ctx, expr) {
            if !cas_math::expr_predicates::contains_variable(ctx, inner) {
                let vd = parent_ctx.value_domain();
                if crate::helpers::prove_positive(ctx, inner, vd) == crate::Proof::Proven {
                    return None;
                }
                let neg_inner = ctx.add(Expr::Neg(inner));
                if crate::helpers::prove_positive(ctx, neg_inner, vd) == crate::Proof::Proven {
                    return None;
                }
            }
        }
        let rewrite = try_rewrite_abs_sub_normalize_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_abs_fixed_rewrite_desc(rewrite.kind)))
    }
);

define_rule!(
    AbsQuotientSubNormalizeRule,
    "Abs Quotient Sub Normalize",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    PhaseMask::POST,
    |ctx, expr| {
        let rewrite = try_rewrite_abs_quotient_sub_normalize_expr(ctx, expr)?;
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
    simplifier.add_rule(Box::new(SubsRule));
    simplifier.add_rule(Box::new(SimplifySqrtSquareRule)); // Must go BEFORE EvaluateAbsRule to catch sqrt(x^2) early
    simplifier.add_rule(Box::new(EvaluateCbrtPerfectCubeRule)); // cbrt(8) → 2 (perfect cubes only)
                                                                // V2.14.45: SimplifySqrtOddPowerRule DISABLED - causes split/merge cycle with ProductPowerRule
                                                                // x^(5/2) → |x|²*√x is a "worsening" transformation (increases AST nodes).
                                                                // The canonical form for odd half-integer powers is Pow(x, n/2), NOT the product form.
                                                                // If visual "extracted square" form is desired, it belongs in a renderer or explain-mode.
                                                                // simplifier.add_rule(Box::new(SimplifySqrtOddPowerRule)); // sqrt(x^3) -> |x| * sqrt(x)
    simplifier.add_rule(Box::new(SymbolicRootCancelRule)); // V2.14.45: sqrt(x^n, n) -> x in Assume mode
    simplifier.add_rule(Box::new(EvaluateAbsRule));
    simplifier.add_rule(Box::new(EvaluateSignRule));
    simplifier.add_rule(Box::new(EvaluateFloorCeilRoundRule));
    simplifier.add_rule(Box::new(EvenPowerOverAbsRule));
    simplifier.add_rule(Box::new(AbsNegativeSimplifyRule)); // |x| -> -x when inherited domain proves x < 0
    simplifier.add_rule(Box::new(AbsPositiveSimplifyRule)); // V2.14.20: |x| -> x when x > 0
    simplifier.add_rule(Box::new(AbsNonNegativeSimplifyRule)); // |x| -> x when x >= 0 (from sqrt requirements)
    simplifier.add_rule(Box::new(AbsDomainAddSubCancellationRule)); // |x| ± x cancellation under root domain
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
    simplifier.add_rule(Box::new(AbsQuotientSubNormalizeRule)); // |(a-b)/c| → |(b-a)/c|
    simplifier.add_rule(Box::new(AbsPositiveFactorRule)); // |k·x| → k·|x| for k > 0
    simplifier.add_rule(Box::new(EvaluateMetaFunctionsRule)); // Make simplify/factor/expand transparent
}
