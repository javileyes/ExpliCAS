use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_math::exponents_support::try_rewrite_exp_to_epow_expr;
use cas_math::expr_nary::{
    try_rewrite_canonicalize_add_expr, try_rewrite_canonicalize_mul_expr,
    CanonicalizeAddRewriteKind, CanonicalizeMulRewriteKind,
};
use cas_math::expr_sub_like::{
    try_rewrite_add_negative_constant_to_sub_expr, try_rewrite_cancel_fraction_signs_expr,
    try_rewrite_canonicalize_div_expr, try_rewrite_canonicalize_negation_expr,
    try_rewrite_neg_coeff_flip_binomial_expr, try_rewrite_neg_sub_flip_expr,
    try_rewrite_normalize_binomial_order_expr,
};
use cas_math::root_forms::{try_rewrite_canonical_root_expr, CanonicalRootRewriteKind};
use std::cmp::Ordering;

define_rule!(CanonicalizeAddRule, "Canonicalize Addition", importance: crate::step::ImportanceLevel::Low, |ctx, expr| {
    let rewrite = try_rewrite_canonicalize_add_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(format_canonicalize_add_desc(rewrite.kind)))
});

define_rule!(
    CanonicalizeMulRule,
    "Canonicalize Multiplication",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_canonicalize_mul_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_canonicalize_mul_desc(rewrite.kind)))
    }
);

fn format_canonicalize_add_desc(kind: CanonicalizeAddRewriteKind) -> &'static str {
    match kind {
        CanonicalizeAddRewriteKind::SortTerms => "Sort addition terms",
        CanonicalizeAddRewriteKind::RightAssociate => "Fix associativity (a+b)+c -> a+(b+c)",
    }
}

fn format_canonicalize_mul_desc(kind: CanonicalizeMulRewriteKind) -> &'static str {
    match kind {
        CanonicalizeMulRewriteKind::SortFactors => "Sort multiplication factors",
        CanonicalizeMulRewriteKind::RightAssociate => "Fix associativity (a*b)*c -> a*(b*c)",
    }
}

fn format_normalize_binomial_order_desc() -> &'static str {
    "(y-x) -> -(x-y) for canonical order"
}

fn format_neg_sub_flip_desc() -> &'static str {
    "-(a - b) → (b - a) (canonical orientation)"
}

fn format_cancel_fraction_signs_desc() -> &'static str {
    "(-A)/(-B) = A/B (cancel double sign)"
}

fn format_neg_coeff_flip_binomial_desc() -> &'static str {
    "(-k) * (...) * (a-b) → k * (...) * (b-a)"
}

define_rule!(CanonicalizeDivRule, "Canonicalize Division", importance: crate::step::ImportanceLevel::Low, |ctx, expr| {
    let rewrite = try_rewrite_canonicalize_div_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
});

define_rule!(CanonicalizeRootRule, "Canonicalize Roots", importance: crate::step::ImportanceLevel::Low, |ctx, expr| {
    let rewrite = try_rewrite_canonical_root_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(format_canonical_root_desc(rewrite.kind)))
});

fn format_canonical_root_desc(kind: CanonicalRootRewriteKind) -> &'static str {
    match kind {
        CanonicalRootRewriteKind::SqrtEvenPower => "sqrt(x^2k) -> |x|^k",
        CanonicalRootRewriteKind::SqrtUnary => "sqrt(x) = x^(1/2)",
        CanonicalRootRewriteKind::SqrtWithIndex => "sqrt(x, n) = x^(1/n)",
        CanonicalRootRewriteKind::RootWithIndex => "root(x, n) = x^(1/n)",
    }
}

define_rule!(NormalizeSignsRule, "Normalize Signs", |ctx, expr| {
    let rewrite = try_rewrite_add_negative_constant_to_sub_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
});

// Normalize binomial order: (b-a) -> -(a-b) when a < b alphabetically
// This ensures consistent representation of binomials like (y-x) vs (x-y)
// so they can be recognized as opposites in fraction simplification.
define_rule!(
    NormalizeBinomialOrderRule,
    "Normalize Binomial Order",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_normalize_binomial_order_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_normalize_binomial_order_desc()))
    }
);

// Rule: -(a - b) → (b - a) ONLY when inner is non-canonical (a > b)
// This prevents the 2-cycle with NormalizeBinomialOrderRule:
// - Normalize: (a-b) → -(b-a) when a > b (produces canonical inner with b < a)
// - Flip: -(a-b) → (b-a) ONLY when a > b (inner is non-canonical)
// Since a > b and b < a are mutually exclusive, no cycle can occur.
define_rule!(
    NegSubFlipRule,
    "Flip Negative Subtraction",
    None,
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_neg_sub_flip_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc(format_neg_sub_flip_desc())
                .local(rewrite.inner, rewrite.rewritten),
        )
    }
);

fn is_explicit_negation_like(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    match ctx.get(expr) {
        cas_ast::Expr::Neg(_) => true,
        cas_ast::Expr::Mul(l, r) => {
            let minus_one = num_rational::BigRational::from_integer((-1).into());
            matches!(ctx.get(*l), cas_ast::Expr::Number(n) if *n == minus_one)
                || matches!(ctx.get(*r), cas_ast::Expr::Number(n) if *n == minus_one)
        }
        _ => false,
    }
}

fn is_implicit_negated_sub_like(ctx: &mut cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    if is_explicit_negation_like(ctx, expr) {
        return false;
    }

    cas_math::expr_sub_like::extract_sub_like_pair(ctx, expr)
        .is_some_and(|(a, b)| cas_ast::ordering::compare_expr(ctx, a, b) == Ordering::Less)
}

fn cosmetic_fraction_sign_cancel_gated_mode(
    parent_ctx: &crate::parent_context::ParentContext,
) -> bool {
    match parent_ctx.simplify_purpose() {
        crate::SimplifyPurpose::Eval => false,
        crate::SimplifyPurpose::SolvePrepass => true,
        crate::SimplifyPurpose::SolveTactic => {
            matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict)
        }
    }
}

fn is_cosmetic_double_implicit_neg_fraction(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let cas_ast::Expr::Div(num, den) = ctx.get(expr) else {
        return false;
    };
    let (num, den) = (*num, *den);

    is_implicit_negated_sub_like(ctx, num) && is_implicit_negated_sub_like(ctx, den)
}

fn should_skip_cosmetic_fraction_sign_cancel(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    parent_ctx: &crate::parent_context::ParentContext,
) -> bool {
    cosmetic_fraction_sign_cancel_gated_mode(parent_ctx)
        && is_cosmetic_double_implicit_neg_fraction(ctx, expr)
}

fn should_skip_cosmetic_sub_negation_canonicalization(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
    parent_ctx: &crate::parent_context::ParentContext,
) -> bool {
    if !cosmetic_fraction_sign_cancel_gated_mode(parent_ctx) {
        return false;
    }
    if !matches!(ctx.get(expr), cas_ast::Expr::Sub(_, _)) {
        return false;
    }

    let Some(parent_id) = parent_ctx.immediate_parent() else {
        return false;
    };
    let cas_ast::Expr::Div(num, den) = ctx.get(parent_id) else {
        return false;
    };
    if *num != expr && *den != expr {
        return false;
    }

    is_cosmetic_double_implicit_neg_fraction(ctx, parent_id)
}

// Rule: (-A)/(-B) → A/B - Cancel double negation in fractions
// This handles cases like (1-√x)/(1-x) → (√x-1)/(x-1)
// by recognizing that (1-√x) is the negation of (√x-1), etc.
//
// No loop risk: produces canonical order which won't match again.
pub struct CancelFractionSignsRule;

impl crate::rule::SimpleRule for CancelFractionSignsRule {
    fn name(&self) -> &str {
        "Cancel Fraction Signs"
    }

    fn apply_simple(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
    ) -> Option<crate::rule::Rewrite> {
        let rewrite = try_rewrite_cancel_fraction_signs_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_cancel_fraction_signs_desc()))
    }

    fn apply_with_context(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        if should_skip_cosmetic_fraction_sign_cancel(ctx, expr, parent_ctx) {
            return None;
        }
        self.apply_simple(ctx, expr)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Low
    }
}

pub struct CanonicalizeNegationRule;

impl crate::rule::SimpleRule for CanonicalizeNegationRule {
    fn name(&self) -> &str {
        "Canonicalize Negation"
    }

    fn apply_simple(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
    ) -> Option<crate::rule::Rewrite> {
        let rewrite = try_rewrite_canonicalize_negation_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }

    fn apply_with_context(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        if should_skip_cosmetic_sub_negation_canonicalization(ctx, expr, parent_ctx) {
            return None;
        }
        self.apply_simple(ctx, expr)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Low
    }
}

// Rule: (-k) * (...) * (a - b) → k * (...) * (b - a) when k > 0
// This produces cleaner output like "1/2 * x * (√2 - 1)" instead of "-1/2 * x * (1 - √2)"
// No loop risk: produces positive coefficient which won't match again
define_rule!(
    NegCoeffFlipBinomialRule,
    "Flip binomial under negative coefficient",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_neg_coeff_flip_binomial_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_neg_coeff_flip_binomial_desc()))
    }
);
/// ExpToEPowRule: Convert exp(x) → e^x
///
/// GATE: Only applies in RealOnly mode.
/// In ComplexEnabled, exp(z) is univalued while e^z (via pow) could imply
/// multivalued logarithm semantics. Keeping exp() as a function preserves
/// the intended univalued semantics in complex domain.
///
/// This allows ExponentialLogRule to match e^(ln(x)) → x patterns.
pub struct ExpToEPowRule;

impl crate::rule::Rule for ExpToEPowRule {
    fn name(&self) -> &str {
        "Convert exp to Power"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::semantics::ValueDomain;

        // GATE: Only in RealOnly (exp is univalued; ComplexEnabled needs special handling)
        if parent_ctx.value_domain() != ValueDomain::RealOnly {
            return None;
        }

        let rewrite = try_rewrite_exp_to_epow_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc("exp(x) = e^x"))
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    // RE-ENABLED: Needed for -0 → 0 normalization
    // The non-determinism issue with Sub→Add(Neg) is now handled by canonical ordering
    simplifier.add_rule(Box::new(CanonicalizeNegationRule));

    simplifier.add_rule(Box::new(CanonicalizeAddRule));
    simplifier.add_rule(Box::new(CanonicalizeMulRule));
    simplifier.add_rule(Box::new(CanonicalizeDivRule));
    simplifier.add_rule(Box::new(CancelFractionSignsRule)); // (-A)/(-B) → A/B
    simplifier.add_rule(Box::new(CanonicalizeRootRule));
    simplifier.add_rule(Box::new(NormalizeSignsRule));
    // NormalizeBinomialOrderRule DISABLED - causes stack overflow in asin_acos tests
    // even with guarded NegSubFlipRule. The cycle likely involves other rules.
    // EvenPowSubSwapRule handles the specific (x-y)^2 - (y-x)^2 = 0 case safely.
    // simplifier.add_rule(Box::new(NormalizeBinomialOrderRule));
    simplifier.add_rule(Box::new(NegSubFlipRule)); // -(a-b) → (b-a) only when a > b
    simplifier.add_rule(Box::new(NegCoeffFlipBinomialRule)); // (-k)*(a-b) → k*(b-a)

    // exp(x) → e^x (RealOnly only - preserves complex semantics)
    simplifier.add_rule(Box::new(ExpToEPowRule));
}
