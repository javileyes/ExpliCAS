use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use cas_math::expr_destructure::as_div;
use cas_math::polynomial::Polynomial;
use cas_math::root_den_rationalize_support::RootDenRationalizeRewriteKind;
use num_rational::BigRational;
use num_traits::{One, Zero};

// =============================================================================
// RationalizeLinearSqrtDenRule: 1/(sqrt(t)+c) → (sqrt(t)-c)/(t-c²)
// =============================================================================
// Rationalizes denominators with linear sqrt terms by multiplying by conjugate.
// This is a canonical transformation that eliminates radicals from denominators.
//
// Examples:
//   1/(sqrt(2)+1) → (sqrt(2)-1)/1 = sqrt(2)-1
//   1/(sqrt(3)+1) → (sqrt(3)-1)/2
//   1/(sqrt(u)+1) → (sqrt(u)-1)/(u-1)
//   2/(sqrt(3)-1) → 2*(sqrt(3)+1)/2 = sqrt(3)+1
//
// Guard: Only apply when result is simpler (no radicals in denominator)
// =============================================================================
define_rule!(
    RationalizeLinearSqrtDenRule,
    "Rationalize Linear Sqrt Denominator",
    |ctx, expr| {
        fn format_div_expand_cancel_desc(
            kind: cas_math::div_expand_cancel_support::DivExpandToCancelKind,
        ) -> &'static str {
            match kind {
                cas_math::div_expand_cancel_support::DivExpandToCancelKind::OpaqueSubstitution => {
                    "Polynomial division with opaque substitution"
                }
                cas_math::div_expand_cancel_support::DivExpandToCancelKind::ExpandedEquality => {
                    "Expanded numerator equals denominator"
                }
            }
        }

        if let Some(exact_quotient) = cas_math::div_expand_cancel_support::try_rewrite_div_expand_to_cancel_expr_with_thread_guards(
            ctx,
            expr,
            |base_ctx, sub_frac| {
                let mut simplifier = crate::Simplifier::with_default_rules();
                simplifier.context = base_ctx.clone();
                let (simplified, _) = simplifier.simplify(sub_frac);
                Some((simplifier.context, simplified))
            },
            crate::expand::expand,
            |expanded_ctx, expanded_num, expanded_den| {
                let mut simplifier = crate::Simplifier::with_default_rules();
                simplifier.context = expanded_ctx;
                let (simplified_num, _) = simplifier.simplify(expanded_num);
                let (simplified_den, _) = simplifier.simplify(expanded_den);
                Some((simplifier.context, simplified_num, simplified_den))
            },
        ) {
            let mut rewrite = Rewrite::new(exact_quotient.rewritten)
                .desc(format_div_expand_cancel_desc(exact_quotient.kind));
            if let Some((_, den)) = as_div(ctx, expr) {
                rewrite = rewrite.requires(crate::ImplicitCondition::NonZero(den));
            }
            return Some(rewrite);
        }

        let rewrite = cas_math::root_den_rationalize_support::try_rewrite_shifted_unit_square_over_linear_sqrt_den_expr(
            ctx, expr,
        )
        .or_else(|| {
            cas_math::root_den_rationalize_support::try_rewrite_rationalize_linear_sqrt_den_expr(
                ctx, expr,
            )
        })?;
        let mut out =
            Rewrite::new(rewrite.rewritten).desc(format_root_den_rationalize_desc(rewrite.kind));
        if matches!(
            rewrite.kind,
            RootDenRationalizeRewriteKind::ShiftedUnitSquareExactQuotient
        ) {
            if let Some((_, den)) = as_div(ctx, expr) {
                out = out.requires(crate::ImplicitCondition::NonZero(den));
            }
        }
        Some(out)
    }
);

// =============================================================================
// RationalizeSumOfSqrtsDenRule: k/(sqrt(p)+sqrt(q)) → k*(sqrt(p)-sqrt(q))/(p-q)
// =============================================================================
// Rationalizes denominators with sum of two square roots.
//
// Examples:
//   3/(sqrt(2)+sqrt(3)) → 3*(sqrt(2)-sqrt(3))/(2-3) = -3*(sqrt(2)-sqrt(3))
//   1/(sqrt(5)+sqrt(2)) → (sqrt(5)-sqrt(2))/3
// =============================================================================
define_rule!(
    RationalizeSumOfSqrtsDenRule,
    "Rationalize Sum of Sqrts Denominator",
    |ctx, expr| {
        let rewrite =
            cas_math::root_den_rationalize_support::try_rewrite_rationalize_sum_of_sqrts_den_expr(
                ctx, expr,
            )?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_root_den_rationalize_desc(rewrite.kind)))
    }
);

// =============================================================================
// CubeRootDenRationalizeRule: k/(1+u^(1/3)) → k*(1-u^(1/3)+u^(2/3))/(1+u)
// =============================================================================
// Uses the sum of cubes identity: 1 + r³ = (1 + r)(1 - r + r²)
// So: 1/(1+r) = (1-r+r²)/(1+r³)
// With r = u^(1/3), r³ = u
//
// Similarly for difference: 1 - r³ = (1 - r)(1 + r + r²)
// So: 1/(1-r) = (1+r+r²)/(1-r³)
//
// Examples:
//   1/(1+u^(1/3)) → (1-u^(1/3)+u^(2/3))/(1+u)
//   1/(1-u^(1/3)) → (1+u^(1/3)+u^(2/3))/(1-u)
// =============================================================================
define_rule!(
    CubeRootDenRationalizeRule,
    "Rationalize Cube Root Denominator",
    |ctx, expr| {
        let rewrite =
            cas_math::root_den_rationalize_support::try_rewrite_rationalize_cube_root_den_expr(
                ctx, expr,
            )?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_root_den_rationalize_desc(rewrite.kind)))
    }
);

fn format_root_den_rationalize_desc(kind: RootDenRationalizeRewriteKind) -> &'static str {
    match kind {
        RootDenRationalizeRewriteKind::ShiftedUnitSquareExactQuotient => {
            "Cancel exact shifted square before rationalizing"
        }
        RootDenRationalizeRewriteKind::LinearSqrtDen => "Rationalize: multiply by conjugate",
        RootDenRationalizeRewriteKind::SumOfSqrtsDen => {
            "Rationalize: (sqrt(p)±sqrt(q)) multiply by conjugate"
        }
        RootDenRationalizeRewriteKind::CubeRootDen => {
            "Rationalize: cube root denominator via sum of cubes"
        }
    }
}

// =============================================================================
// RootMergeMulRule: sqrt(a) * sqrt(b) → sqrt(a*b)
// =============================================================================
// Merges products of square roots into a single root.
// This is valid for non-negative real a and b.
//
// Examples:
//   sqrt(u) * sqrt(b) → sqrt(u*b)
//   u^(1/2) * b^(1/2) → (u*b)^(1/2)
//
// Requires: a ≥ 0 and b ≥ 0 (or they are squared terms)
// =============================================================================
pub struct RootMergeMulRule;

impl crate::rule::Rule for RootMergeMulRule {
    fn name(&self) -> &str {
        "Merge Sqrt Product"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        let strict_mode = matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict);
        let vd = parent_ctx.value_domain();
        let rewrite = cas_math::root_power_canonical_support::try_rewrite_root_merge_mul_expr_with(
            ctx,
            expr,
            strict_mode,
            |core_ctx, inner| {
                cas_solver_core::predicate_proofs::prove_nonnegative_core_with(
                    core_ctx,
                    inner,
                    vd,
                    crate::helpers::prove_nonnegative,
                )
            },
        )?;

        let mut out = Rewrite::new(rewrite.rewritten).desc("√a · √b = √(a·b)");
        if rewrite.assume_left_nonnegative {
            out = out.assume(crate::AssumptionEvent::nonnegative(ctx, rewrite.left_base));
        }
        if rewrite.assume_right_nonnegative {
            out = out.assume(crate::AssumptionEvent::nonnegative(ctx, rewrite.right_base));
        }
        Some(out)
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::MUL)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Medium
    }
}

// =============================================================================
// RootMergeDivRule: sqrt(a) / sqrt(b) → sqrt(a/b)
// =============================================================================
// Merges quotients of square roots into a single root.
// This is valid for non-negative real a and positive b.
//
// Examples:
//   sqrt(u) / sqrt(b) → sqrt(u/b)
//   u^(1/2) / b^(1/2) → (u/b)^(1/2)
//
// Requires: a ≥ 0 and b > 0
// =============================================================================
pub struct RootMergeDivRule;

impl crate::rule::Rule for RootMergeDivRule {
    fn name(&self) -> &str {
        "Merge Sqrt Quotient"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        let strict_mode = matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict);
        let vd = parent_ctx.value_domain();
        let rewrite = cas_math::root_power_canonical_support::try_rewrite_root_merge_div_expr_with(
            ctx,
            expr,
            strict_mode,
            |core_ctx, inner| {
                cas_solver_core::predicate_proofs::prove_nonnegative_core_with(
                    core_ctx,
                    inner,
                    vd,
                    crate::helpers::prove_nonnegative,
                )
            },
            |core_ctx, inner| {
                cas_solver_core::predicate_proofs::prove_positive_core_with(
                    core_ctx,
                    inner,
                    vd,
                    crate::helpers::prove_positive,
                )
            },
        )?;

        let mut out = Rewrite::new(rewrite.rewritten).desc("√a / √b = √(a/b)");
        if rewrite.assume_num_nonnegative {
            out = out.assume(crate::AssumptionEvent::nonnegative(ctx, rewrite.num_base));
        }
        if rewrite.assume_den_positive {
            out = out.assume(crate::AssumptionEvent::positive(ctx, rewrite.den_base));
        }
        Some(out)
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::DIV)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Medium
    }
}

// =============================================================================
// ReciprocalSqrtProductMergeRule: a^(-1/2) * b^(-1/2) -> (a*b)^(-1/2)
// =============================================================================
// Merges products of reciprocal square-root powers into a single reciprocal root.
// This is valid in the real domain only when both bases are positive. The rule
// keeps those individual requirements explicit instead of weakening them to
// only a*b > 0.
//
// Guarded to composed symbolic bases so simple independent factors such as
// x^(-1/2)*y^(-1/2) do not change broad canonical traffic.
// =============================================================================
pub struct ReciprocalSqrtProductMergeRule;

impl crate::rule::Rule for ReciprocalSqrtProductMergeRule {
    fn name(&self) -> &str {
        "Merge Reciprocal Sqrt Product"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        if !parent_ctx.has_ancestor_matching(ctx, |ancestor_ctx, ancestor| {
            matches!(
                ancestor_ctx.get(ancestor),
                Expr::Add(_, _) | Expr::Sub(_, _)
            )
        }) {
            return None;
        }

        let Expr::Mul(left, right) = ctx.get(expr) else {
            return None;
        };
        let left = *left;
        let right = *right;
        let left = cas_ast::hold::strip_all_holds(ctx, left);
        let right = cas_ast::hold::strip_all_holds(ctx, right);
        let left_base = reciprocal_sqrt_power_base(ctx, left)?;
        let right_base = reciprocal_sqrt_power_base(ctx, right)?;

        if !reciprocal_sqrt_product_base_is_eligible(ctx, left_base)
            || !reciprocal_sqrt_product_base_is_eligible(ctx, right_base)
            || !reciprocal_sqrt_product_base_is_composed(ctx, left_base)
            || !reciprocal_sqrt_product_base_is_composed(ctx, right_base)
        {
            return None;
        }

        let product = ctx.add(Expr::Mul(left_base, right_base));
        let exp = ctx.rational(-1, 2);
        let rewritten = ctx.add(Expr::Pow(product, exp));
        Some(
            Rewrite::new(rewritten)
                .desc("a^(-1/2)·b^(-1/2) = (a·b)^(-1/2)")
                .requires(crate::ImplicitCondition::Positive(left_base))
                .requires(crate::ImplicitCondition::Positive(right_base)),
        )
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::MUL)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Low
    }
}

// =============================================================================
// ReciprocalSqrtProductFactorCancelRule: b*(a*b)^(-1/2) -> (a/b)^(-1/2)
// =============================================================================
// Cancels one positive product factor against a reciprocal square-root product.
// This is valid over the reals only when both product factors are positive:
//
//   b / sqrt(a*b) = sqrt(b/a) = (a/b)^(-1/2)
//
// The rule is intentionally narrow: it only fires in additive residual contexts
// and only when at least one product factor is composed. That keeps broad
// independent-symbol traffic such as y*(x*y)^(-1/2) unchanged, while still
// covering interval factors like x*(1-x).
// =============================================================================
pub struct ReciprocalSqrtProductFactorCancelRule;

impl crate::rule::Rule for ReciprocalSqrtProductFactorCancelRule {
    fn name(&self) -> &str {
        "Cancel Reciprocal Sqrt Product Factor"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        if !parent_ctx.has_ancestor_matching(ctx, |ancestor_ctx, ancestor| {
            matches!(
                ancestor_ctx.get(ancestor),
                Expr::Add(_, _) | Expr::Sub(_, _)
            )
        }) {
            return None;
        }

        let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
        if factors.len() < 2 {
            return None;
        }

        for (sqrt_index, factor) in factors.iter().copied().enumerate() {
            let Some(product_base) = reciprocal_sqrt_power_base(ctx, factor) else {
                continue;
            };
            let Some((left_base, right_base)) =
                reciprocal_sqrt_binary_product_base(ctx, product_base)
            else {
                continue;
            };

            for (remaining_base, cancel_base) in [(left_base, right_base), (right_base, left_base)]
            {
                if !reciprocal_sqrt_product_base_is_composed(ctx, cancel_base)
                    && !reciprocal_sqrt_product_base_is_composed(ctx, remaining_base)
                {
                    continue;
                }
                let scale_expr =
                    multiply_all(ctx, factors_without_index(&factors, sqrt_index).into_iter());
                let Some(scale) = scaled_polynomial_factor(ctx, scale_expr, cancel_base) else {
                    continue;
                };
                let quotient_base = ctx.add(Expr::Div(remaining_base, cancel_base));
                let exp = ctx.rational(-1, 2);
                let reciprocal = ctx.add(Expr::Pow(quotient_base, exp));
                let rewritten = multiply_scale(ctx, scale, reciprocal);
                return Some(
                    Rewrite::new(rewritten)
                        .desc("b·(a·b)^(-1/2) = (a/b)^(-1/2)")
                        .requires(crate::ImplicitCondition::Positive(remaining_base))
                        .requires(crate::ImplicitCondition::Positive(cancel_base)),
                );
            }
        }

        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::MUL)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Low
    }
}

// =============================================================================
// ReciprocalSqrtQuotientDenCancelRule: (a/b)^(-1/2)/b -> (a*b)^(-1/2)
// =============================================================================
// Bridges compact calculus presentations with embedded residual simplification:
//
//   1 / (b*sqrt(a/b)) = 1 / sqrt(a*b)
//
// The rule is intentionally residual-scoped and requires a composed quotient
// side, so plain independent-symbol traffic such as (x/y)^(-1/2)/y stays
// untouched.
// =============================================================================
pub struct ReciprocalSqrtQuotientDenCancelRule;

impl crate::rule::Rule for ReciprocalSqrtQuotientDenCancelRule {
    fn name(&self) -> &str {
        "Cancel Reciprocal Sqrt Quotient Denominator"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        if !parent_ctx.has_ancestor_matching(ctx, |ancestor_ctx, ancestor| {
            matches!(
                ancestor_ctx.get(ancestor),
                Expr::Add(_, _) | Expr::Sub(_, _)
            )
        }) {
            return None;
        }

        let (quotient_power, actual_den) = match ctx.get(expr) {
            Expr::Div(num, den) => (*num, *den),
            _ => return None,
        };
        let quotient_base = reciprocal_sqrt_power_base(ctx, quotient_power)?;
        let quotient_base = cas_ast::hold::strip_all_holds(ctx, quotient_base);
        let (quotient_num_raw, quotient_den_raw) = match ctx.get(quotient_base) {
            Expr::Div(num, den) => (*num, *den),
            _ => return None,
        };
        let quotient_num = cas_ast::hold::strip_all_holds(ctx, quotient_num_raw);
        let quotient_den = cas_ast::hold::strip_all_holds(ctx, quotient_den_raw);

        if !reciprocal_sqrt_product_base_is_composed(ctx, quotient_num)
            && !reciprocal_sqrt_product_base_is_composed(ctx, quotient_den)
        {
            return None;
        }

        let scale = scaled_polynomial_factor(ctx, actual_den, quotient_den)?;
        let product_base = ctx.add(Expr::Mul(quotient_num, quotient_den));
        let exp = ctx.rational(-1, 2);
        let reciprocal = ctx.add(Expr::Pow(product_base, exp));
        let rewritten = multiply_scale(ctx, BigRational::one() / scale, reciprocal);
        Some(
            Rewrite::new(rewritten)
                .desc("(a/b)^(-1/2)/b = (a*b)^(-1/2)")
                .requires(crate::ImplicitCondition::Positive(quotient_num))
                .requires(crate::ImplicitCondition::Positive(quotient_den)),
        )
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::DIV)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Low
    }
}

fn reciprocal_sqrt_power_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if reciprocal_sqrt_product_is_negative_half(ctx, *exp) {
        Some(*base)
    } else {
        None
    }
}

fn reciprocal_sqrt_binary_product_base(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Mul(left, right) => (*left, *right),
        _ => return None,
    };
    Some((
        cas_ast::hold::strip_all_holds(ctx, left),
        cas_ast::hold::strip_all_holds(ctx, right),
    ))
}

fn factors_without_index(factors: &[ExprId], skip_index: usize) -> Vec<ExprId> {
    factors
        .iter()
        .enumerate()
        .filter_map(|(index, factor)| (index != skip_index).then_some(*factor))
        .collect()
}

fn multiply_all(ctx: &mut Context, factors: impl Iterator<Item = ExprId>) -> ExprId {
    factors
        .reduce(|left, right| ctx.add(Expr::Mul(left, right)))
        .unwrap_or_else(|| ctx.num(1))
}

fn multiply_scale(ctx: &mut Context, scale: BigRational, expr: ExprId) -> ExprId {
    if scale == BigRational::one() {
        return expr;
    }
    if scale == -BigRational::one() {
        return ctx.add(Expr::Neg(expr));
    }
    let scale_expr = ctx.add(Expr::Number(scale));
    ctx.add(Expr::Mul(scale_expr, expr))
}

fn scaled_polynomial_factor(
    ctx: &Context,
    actual_expr: ExprId,
    expected_expr: ExprId,
) -> Option<BigRational> {
    let actual_vars = cas_ast::collect_variables(ctx, actual_expr);
    let expected_vars = cas_ast::collect_variables(ctx, expected_expr);
    let vars = actual_vars.union(&expected_vars).collect::<Vec<_>>();
    if vars.len() != 1 {
        return None;
    }
    let var_name = vars[0].as_str();
    let actual = Polynomial::from_expr(ctx, actual_expr, var_name).ok()?;
    let expected = Polynomial::from_expr(ctx, expected_expr, var_name).ok()?;
    polynomial_scale_factor(&expected, &actual)
}

fn polynomial_scale_factor(expected: &Polynomial, actual: &Polynomial) -> Option<BigRational> {
    if expected.is_zero() {
        return actual.is_zero().then_some(BigRational::one());
    }
    if expected.degree() != actual.degree() {
        return None;
    }

    let mut scale = None;
    for (expected_coeff, actual_coeff) in expected.coeffs.iter().zip(actual.coeffs.iter()) {
        if expected_coeff.is_zero() {
            if !actual_coeff.is_zero() {
                return None;
            }
            continue;
        }
        let candidate = actual_coeff / expected_coeff;
        if scale.as_ref().is_some_and(|scale| scale != &candidate) {
            return None;
        }
        scale = Some(candidate);
    }

    scale.filter(|scale| !scale.is_zero())
}

fn reciprocal_sqrt_product_is_negative_half(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(value) => value == &num_rational::BigRational::new((-1).into(), 2.into()),
        Expr::Div(num, den) => {
            matches!(
                ctx.get(*num),
                Expr::Number(value) if value == &num_rational::BigRational::from_integer((-1).into())
            ) && matches!(
                ctx.get(*den),
                Expr::Number(value) if value == &num_rational::BigRational::from_integer(2.into())
            )
        }
        Expr::Neg(inner) => {
            matches!(
                ctx.get(*inner),
                Expr::Number(value)
                    if value == &num_rational::BigRational::new(1.into(), 2.into())
            ) || matches!(
                ctx.get(*inner),
                Expr::Div(num, den)
                    if matches!(
                        ctx.get(*num),
                        Expr::Number(value) if value.is_one()
                    ) && matches!(
                        ctx.get(*den),
                        Expr::Number(value) if value == &num_rational::BigRational::from_integer(2.into())
                    )
            )
        }
        _ => false,
    }
}

fn reciprocal_sqrt_product_base_is_eligible(ctx: &Context, expr: ExprId) -> bool {
    reciprocal_sqrt_product_contains_symbol(ctx, expr)
        && !matches!(ctx.get(expr), Expr::Number(_) | Expr::Constant(_))
}

fn reciprocal_sqrt_product_base_is_composed(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Mul(_, _) | Expr::Div(_, _) | Expr::Pow(_, _)
    )
}

fn reciprocal_sqrt_product_contains_symbol(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Variable(_) => true,
        Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => false,
        Expr::Neg(inner) | Expr::Hold(inner) => {
            reciprocal_sqrt_product_contains_symbol(ctx, *inner)
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            reciprocal_sqrt_product_contains_symbol(ctx, *left)
                || reciprocal_sqrt_product_contains_symbol(ctx, *right)
        }
        Expr::Function(_, args) => args
            .iter()
            .any(|arg| reciprocal_sqrt_product_contains_symbol(ctx, *arg)),
        Expr::Matrix { data, .. } => data
            .iter()
            .any(|arg| reciprocal_sqrt_product_contains_symbol(ctx, *arg)),
    }
}

// =============================================================================
// PowPowCancelReciprocalRule: (u^y)^(1/y) → u
// =============================================================================
// Cancels reciprocal exponents in nested powers.
// This is valid for u > 0 and y ≠ 0 in real domain.
//
// Examples:
//   (u^y)^(1/y) → u
//   (x^n)^(1/n) → x
//
// Requires: u > 0 (base), y ≠ 0 (exponent)
// =============================================================================
pub struct PowPowCancelReciprocalRule;

impl crate::rule::Rule for PowPowCancelReciprocalRule {
    fn name(&self) -> &str {
        "Cancel Reciprocal Exponents"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // REAL-ONLY: every arm is a real-domain identity — the even-numerator
        // arm emits `|u|` (`(x²)^(1/2) = |x|`, false over ℂ: `(i²)^(1/2) = i`),
        // and the fallback cancels under a REAL positivity assumption.
        if parent_ctx.value_domain() != crate::semantics::ValueDomain::RealOnly {
            return None;
        }
        let strict_mode = matches!(parent_ctx.domain_mode(), crate::DomainMode::Strict);
        let vd = parent_ctx.value_domain();
        let rewrite =
            cas_math::root_power_canonical_support::try_rewrite_powpow_cancel_reciprocal_expr_with(
                ctx,
                expr,
                strict_mode,
                |core_ctx, inner| {
                    cas_solver_core::predicate_proofs::prove_positive_core_with(
                        core_ctx,
                        inner,
                        vd,
                        crate::helpers::prove_positive,
                    )
                },
                |core_ctx, inner| {
                    cas_solver_core::predicate_proofs::prove_nonzero_core_with(
                        core_ctx,
                        inner,
                        crate::helpers::prove_nonzero,
                    )
                },
            )?;

        let mut out = Rewrite::new(rewrite.rewritten).desc("(u^y)^(1/y) = u");
        if rewrite.assume_base_positive {
            out = out.assume(crate::AssumptionEvent::positive(ctx, rewrite.base));
        }
        if rewrite.assume_exp_nonzero {
            out = out.assume(crate::AssumptionEvent::nonzero(ctx, rewrite.inner_exp));
        }
        Some(out)
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::POW)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Medium
    }
}

// =============================================================================
// ReciprocalSqrtCanonRule: Canonicalize reciprocal sqrt forms to Pow(x, -1/2)
// =============================================================================
// Ensures all representations of "1/√x" converge to a single canonical AST:
//
//   Pattern 1: 1/√x      = Div(1, Pow(x, 1/2))     → Pow(x, -1/2)
//   Pattern 2: f/√x      = Div(f, Pow(x, 1/2))     → f*Pow(x, -1/2)
//   Pattern 3: √x/x      = Div(Pow(x, 1/2), x)     → Pow(x, -1/2)
//   Pattern 4: √(x^(-1)) = Pow(Pow(x,-1), 1/2)     → already handled by PowerPowerRule
//
// GUARD: Only applied when the base contains symbols (variables).
// Pure numeric bases (e.g., 1/√2) are left as-is to avoid creating Pow(2, -1/2)
// forms that Strict-mode verification cannot fold back to √2/2.
//
// This is sound in RealOnly: all forms require x > 0, same definability domain.
// No cycle risk: NegativeExponentNormalizationRule only fires on INTEGER negative
// exponents, and -1/2 is not integer.
// =============================================================================

pub struct ReciprocalSqrtCanonRule;

impl crate::rule::Rule for ReciprocalSqrtCanonRule {
    fn name(&self) -> &str {
        "Canonicalize Reciprocal Sqrt"
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        let rewrite =
            cas_math::reciprocal_sqrt_canon_support::try_rewrite_reciprocal_sqrt_canon_expr(
                ctx, expr,
            )?;
        let desc = match ctx.get(expr) {
            Expr::Div(num, _) => {
                if matches!(ctx.get(*num), Expr::Number(n) if n.is_one()) {
                    "1/√x = x^(-1/2)"
                } else if let Expr::Div(num, den) = ctx.get(expr) {
                    if cas_math::root_forms::extract_square_root_base(ctx, *num).is_some_and(
                        |base| cas_ast::ordering::compare_expr(ctx, base, *den).is_eq(),
                    ) {
                        "√x/x = x^(-1/2)"
                    } else {
                        "f/√x = f·x^(-1/2)"
                    }
                } else {
                    "Canonicalize reciprocal sqrt"
                }
            }
            _ => "Canonicalize reciprocal sqrt",
        };
        Some(crate::rule::Rewrite::new(rewrite.rewritten).desc(desc))
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::DIV)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Low
    }
}

#[cfg(test)]
mod tests {
    use super::{
        RationalizeLinearSqrtDenRule, ReciprocalSqrtProductFactorCancelRule,
        ReciprocalSqrtProductMergeRule, ReciprocalSqrtQuotientDenCancelRule,
    };
    use crate::parent_context::ParentContext;
    use crate::rule::Rule;
    use crate::DomainMode;
    use cas_ast::ordering::compare_expr;
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    #[test]
    fn rationalize_linear_sqrt_den_prefers_exact_quotient_plus_one() {
        let mut ctx = Context::new();
        let expr = parse(
            "(u^2 + 2*(u^2 + 1)^(1/2) + 2)/((u^2 + 1)^(1/2) + 1)",
            &mut ctx,
        )
        .unwrap_or_else(|err| panic!("parse expr: {err}"));
        let rule = RationalizeLinearSqrtDenRule;
        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        assert!(
            rendered == "(u^2 + 1)^(1/2) + 1"
                || rendered == "1 + (u^2 + 1)^(1/2)"
                || rendered == "sqrt(u^2 + 1) + 1"
                || rendered == "1 + sqrt(u^2 + 1)",
            "unexpected exact-quotient rewrite: {rendered}"
        );
    }

    #[test]
    fn reciprocal_sqrt_product_merge_combines_composed_positive_bases() {
        let mut ctx = Context::new();
        let expr = parse("(2-2*x)^(-1/2)*(2*x-1)^(-1/2)", &mut ctx)
            .unwrap_or_else(|err| panic!("parse expr: {err}"));
        let parent = parse("(2-2*x)^(-1/2)*(2*x-1)^(-1/2) - z", &mut ctx)
            .unwrap_or_else(|err| panic!("parse parent: {err}"));
        let rule = ReciprocalSqrtProductMergeRule;
        let parent_ctx = ParentContext::with_parent(parent).with_domain_mode(DomainMode::Generic);
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        assert!(
            rendered.contains("2 - 2 * x")
                && rendered.contains("2 * x - 1")
                && rendered.contains("^(-1/2)"),
            "unexpected merged reciprocal root: {rendered}"
        );
        assert_eq!(rewrite.required_conditions.len(), 2);
    }

    #[test]
    fn reciprocal_sqrt_product_merge_skips_plain_independent_symbols() {
        let mut ctx = Context::new();
        let expr =
            parse("x^(-1/2)*y^(-1/2)", &mut ctx).unwrap_or_else(|err| panic!("parse expr: {err}"));
        let parent = parse("x^(-1/2)*y^(-1/2) - z", &mut ctx)
            .unwrap_or_else(|err| panic!("parse parent: {err}"));
        let rule = ReciprocalSqrtProductMergeRule;
        let parent_ctx = ParentContext::with_parent(parent).with_domain_mode(DomainMode::Generic);

        assert!(rule.apply(&mut ctx, expr, &parent_ctx).is_none());
    }

    #[test]
    fn reciprocal_sqrt_product_factor_cancel_handles_affine_partition_factor() {
        let mut ctx = Context::new();
        let expr = parse("(x*(1-x))^(-1/2)*(2-2*x)", &mut ctx)
            .unwrap_or_else(|err| panic!("parse expr: {err}"));
        let parent = parse("2*(x/(1-x))^(-1/2) - (x*(1-x))^(-1/2)*(2-2*x)", &mut ctx)
            .unwrap_or_else(|err| panic!("parse parent: {err}"));
        let rule = ReciprocalSqrtProductFactorCancelRule;
        let parent_ctx = ParentContext::with_parent(parent).with_domain_mode(DomainMode::Generic);
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        assert!(
            rendered.contains("2")
                && rendered.contains("x / (1 - x)")
                && rendered.contains("^(-1/2)"),
            "unexpected factor-cancel rewrite: {rendered}"
        );
        assert_eq!(rewrite.required_conditions.len(), 2);
    }

    #[test]
    fn reciprocal_sqrt_product_factor_cancel_handles_simple_interval_factor() {
        let mut ctx = Context::new();
        let expr =
            parse("x*(x*(1-x))^(-1/2)", &mut ctx).unwrap_or_else(|err| panic!("parse expr: {err}"));
        let parent = parse("x*(x*(1-x))^(-1/2) - ((1-x)/x)^(-1/2)", &mut ctx)
            .unwrap_or_else(|err| panic!("parse parent: {err}"));
        let rule = ReciprocalSqrtProductFactorCancelRule;
        let parent_ctx = ParentContext::with_parent(parent).with_domain_mode(DomainMode::Generic);
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));

        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        assert!(
            rendered.contains("(1 - x) / x") && rendered.contains("^(-1/2)"),
            "unexpected simple-factor rewrite: {rendered}"
        );
        assert_eq!(rewrite.required_conditions.len(), 2);
    }

    #[test]
    fn reciprocal_sqrt_product_factor_cancel_skips_plain_independent_symbols() {
        let mut ctx = Context::new();
        let expr =
            parse("y*(x*y)^(-1/2)", &mut ctx).unwrap_or_else(|err| panic!("parse expr: {err}"));
        let parent = parse("y*(x*y)^(-1/2) - z", &mut ctx)
            .unwrap_or_else(|err| panic!("parse parent: {err}"));
        let rule = ReciprocalSqrtProductFactorCancelRule;
        let parent_ctx = ParentContext::with_parent(parent).with_domain_mode(DomainMode::Generic);

        assert!(rule.apply(&mut ctx, expr, &parent_ctx).is_none());
    }

    #[test]
    fn reciprocal_sqrt_quotient_den_cancel_handles_composed_affine_denominator() {
        let mut ctx = Context::new();
        let expr = parse("((2*x+1)/(3-2*x))^(-1/2)/(3-2*x)", &mut ctx)
            .unwrap_or_else(|err| panic!("parse expr: {err}"));
        let parent = parse(
            "((2*x+1)/(3-2*x))^(-1/2)/(3-2*x) - ((2*x+1)*(3-2*x))^(-1/2)",
            &mut ctx,
        )
        .unwrap_or_else(|err| panic!("parse parent: {err}"));
        let rule = ReciprocalSqrtQuotientDenCancelRule;
        let parent_ctx = ParentContext::with_parent(parent).with_domain_mode(DomainMode::Generic);
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );
        assert!(
            rendered.contains("(2 * x + 1)")
                && rendered.contains("(3 - 2 * x)")
                && rendered.contains("^(-1/2)"),
            "unexpected quotient-denominator cancel rewrite: {rendered}"
        );
        assert_eq!(rewrite.required_conditions.len(), 2);
    }

    #[test]
    fn reciprocal_sqrt_quotient_den_cancel_skips_plain_independent_symbols() {
        let mut ctx = Context::new();
        let expr =
            parse("(x/y)^(-1/2)/y", &mut ctx).unwrap_or_else(|err| panic!("parse expr: {err}"));
        let parent = parse("(x/y)^(-1/2)/y - z", &mut ctx)
            .unwrap_or_else(|err| panic!("parse parent: {err}"));
        let rule = ReciprocalSqrtQuotientDenCancelRule;
        let parent_ctx = ParentContext::with_parent(parent).with_domain_mode(DomainMode::Generic);

        assert!(rule.apply(&mut ctx, expr, &parent_ctx).is_none());
    }

    #[test]
    fn rationalize_linear_sqrt_den_prefers_root_ctx_exact_quotient() {
        let mut ctx = Context::new();
        let expr = parse("(2*sqrt(u) + u + 1)/(sqrt(u) + u)", &mut ctx)
            .unwrap_or_else(|err| panic!("parse expr: {err}"));
        let rule = RationalizeLinearSqrtDenRule;
        let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
        let rewrite = rule
            .apply(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("rewrite"));
        let expected =
            parse("1/sqrt(u) + 1", &mut ctx).unwrap_or_else(|err| panic!("expected: {err}"));
        assert_eq!(
            compare_expr(&ctx, rewrite.new_expr, expected),
            std::cmp::Ordering::Equal
        );
    }
}
