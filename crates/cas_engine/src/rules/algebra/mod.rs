#[cfg(test)]
mod root_denesting_tests;
#[cfg(test)]
mod tests;

pub mod fractions;
pub use fractions::*;

pub mod distribution;
pub use distribution::*;

pub mod factoring;
pub use factoring::*;

pub mod roots;
pub use roots::*;

pub mod root_denesting;
pub use root_denesting::*;

pub mod root_denesting_advanced;
pub use root_denesting_advanced::*;

pub mod poly_gcd;
pub use poly_gcd::PolyGcdRule;

pub mod gcd_exact;
pub use gcd_exact::PolyGcdExactRule;

pub mod gcd_modp;

pub mod poly_arith_modp;
pub use poly_arith_modp::PolySubModpRule;

pub mod difference_of_cubes;
pub use difference_of_cubes::*;

pub mod poly_mul_modp;

pub mod poly_stats;

use num_traits::Zero;

fn extract_cube_base(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> Option<cas_ast::ExprId> {
    let cas_ast::Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let cas_ast::Expr::Number(n) = ctx.get(*exp) else {
        return None;
    };
    if n.is_integer() && *n == num_rational::BigRational::from_integer(3.into()) {
        Some(*base)
    } else {
        None
    }
}

fn expr_eq_fast(ctx: &cas_ast::Context, left: cas_ast::ExprId, right: cas_ast::ExprId) -> bool {
    if left == right {
        return true;
    }

    match (ctx.get(left), ctx.get(right)) {
        (cas_ast::Expr::Variable(a), cas_ast::Expr::Variable(b)) => a == b,
        (cas_ast::Expr::Constant(a), cas_ast::Expr::Constant(b)) => a == b,
        (cas_ast::Expr::Number(a), cas_ast::Expr::Number(b)) => a == b,
        _ => cas_ast::ordering::compare_expr(ctx, left, right) == std::cmp::Ordering::Equal,
    }
}

fn exact_common_mul_factor(
    ctx: &cas_ast::Context,
    num: cas_ast::ExprId,
    den: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, cas_ast::ExprId)> {
    let cas_ast::Expr::Mul(num_left, num_right) = ctx.get(num) else {
        return None;
    };
    let cas_ast::Expr::Mul(den_left, den_right) = ctx.get(den) else {
        return None;
    };

    let pairs = [
        (*num_left, *num_right, *den_left, *den_right),
        (*num_left, *num_right, *den_right, *den_left),
        (*num_right, *num_left, *den_left, *den_right),
        (*num_right, *num_left, *den_right, *den_left),
    ];

    for (common_num, other_num, common_den, other_den) in pairs {
        if expr_eq_fast(ctx, common_num, common_den)
            && !matches!(ctx.get(common_num), cas_ast::Expr::Number(n) if n.is_zero())
        {
            return Some((common_num, other_num, other_den));
        }
    }

    None
}

/// Exact-shape pre-order detection of `(F*A)/(F*B)` or `(A*F)/(B*F)`.
///
/// This keeps the guard intentionally narrow: only raw top-level `Mul/Mul`
/// forms with one exact common factor. It avoids paying the broader factor
/// cancellation pipeline on the common REPL hotspot where the shared factor is
/// already explicit.
#[allow(clippy::too_many_arguments)]
pub fn try_exact_common_factor_mul_fraction_preorder(
    ctx: &mut cas_ast::Context,
    expr_id: cas_ast::ExprId,
    num: cas_ast::ExprId,
    den: cas_ast::ExprId,
    collect_steps: bool,
    steps: &mut Vec<crate::step::Step>,
    current_path: &[crate::step::PathStep],
) -> Option<cas_ast::ExprId> {
    let (_common, other_num, other_den) = exact_common_mul_factor(ctx, num, den)?;
    let rewritten = ctx.add_raw(cas_ast::Expr::Div(other_num, other_den));

    if collect_steps {
        let mut cancel_step = crate::step::Step::new(
            "Cancel common factor",
            "Pre-order Common Factor Cancel",
            expr_id,
            rewritten,
            current_path.to_vec(),
            Some(ctx),
        );
        cancel_step.before = expr_id;
        cancel_step.after = rewritten;
        cancel_step.global_before = Some(expr_id);
        cancel_step.global_after = Some(rewritten);
        cancel_step.importance = crate::step::ImportanceLevel::High;
        steps.push(cancel_step);
    }

    Some(rewritten)
}

pub fn register(simplifier: &mut crate::Simplifier) {
    // Pre-order: Cancel cube root difference pattern BEFORE GCD/fraction rules
    // (x - b³) / (x^(2/3) + b·x^(1/3) + b²) → x^(1/3) - b
    simplifier.add_rule(Box::new(CancelCubeRootDifferenceRule));
    simplifier.add_rule(Box::new(CancelSumDiffCubesFractionRule));

    // V2.14.35: Ultra-light P^m/P^n → P^(m-n) rule (shallow ExprId comparison)
    // Must fire BEFORE heavier fraction rules to avoid stack overflow on large powers
    simplifier.add_rule(Box::new(CancelPowersDivisionRule));
    // Step 2 of didactic expansion: Cancel P/P → 1 (must fire after expansion rules)
    simplifier.add_rule(Box::new(CancelIdenticalFractionRule));
    // Step 2 variant: Cancel P^n/P → P^(n-1) (for perfect squares and similar)
    simplifier.add_rule(Box::new(CancelPowerFractionRule));
    // Reciprocal cleanup: 1 / P^(-a) → P^a, preserving domain via nonzero(P)
    simplifier.add_rule(Box::new(CollapseReciprocalNegativePowerRule));
    simplifier.add_rule(Box::new(SimplifyFractionRule));
    simplifier.add_rule(Box::new(NestedFractionRule));
    simplifier.add_rule(Box::new(SimplifyMulDivRule));
    simplifier.add_rule(Box::new(FoldAddIntoFractionRule));
    simplifier.add_rule(Box::new(SubTermMatchesDenomRule)); // a - b/a → (a²-b)/a
    simplifier.add_rule(Box::new(SymmetricReciprocalSumRule));
    simplifier.add_rule(Box::new(AddFractionsRule));
    simplifier.add_rule(Box::new(SubFractionsRule));
    simplifier.add_rule(Box::new(CombineSameDenominatorFractionsRule));
    // Distribute numeric denominator into sums: (2x+4)/2 → x+2
    simplifier.add_rule(Box::new(DivScalarIntoAddRule));
    // Try opaque/polynomial quotient cancellation before rationalization destroys
    // simple shared-root quotients like (t^2+2t)/(t+2).
    simplifier.add_rule(Box::new(DivExpandToCancelRule));
    // Mirror cancellation for (A±B)/(A^2-B^2) before denominator rationalization/factoring.
    simplifier.add_rule(Box::new(ReciprocalDifferenceOfSquaresRule));
    // Compact rationalization rules (Level 0, 1) - should apply first
    simplifier.add_rule(Box::new(RationalizeSingleSurdRule));
    simplifier.add_rule(Box::new(RationalizeBinomialSurdRule));
    // General rationalization rules (Level 2) - fallback for complex cases
    simplifier.add_rule(Box::new(RationalizeDenominatorRule)); // sqrt only (diff squares)
    simplifier.add_rule(Box::new(RationalizeNthRootBinomialRule)); // cube root and higher (geometric sum)
    simplifier.add_rule(Box::new(GeneralizedRationalizationRule));
    simplifier.add_rule(Box::new(RationalizeProductDenominatorRule));
    simplifier.add_rule(Box::new(FactorRule));
    simplifier.add_rule(Box::new(CancelCommonFactorsRule));
    simplifier.add_rule(Box::new(DivAddCommonFactorFromDenRule)); // Factor out common from Add in Div to enable cancel
    simplifier.add_rule(Box::new(DivDenFactorOutRule)); // Factor out common from Add denominator to enable cancel
    simplifier.add_rule(Box::new(DivExpandNumForCancelRule)); // Expand Mul(X, Add(...)) in Div numerator when shared factors exist
    simplifier.add_rule(Box::new(DivAddSymmetricFactorRule)); // Cancel common factor from Add/Add fraction
    simplifier.add_rule(Box::new(QuotientOfPowersRule));
    simplifier.add_rule(Box::new(CancelNthRootBinomialFactorRule)); // (x+1)/(x^(1/3)+1) → x^(2/3)-x^(1/3)+1
    simplifier.add_rule(Box::new(SqrtConjugateCollapseRule)); // sqrt(A)*B → sqrt(B) when A*B=1
    simplifier.add_rule(Box::new(RootDenestingRule));
    simplifier.add_rule(Box::new(CubicConjugateTrapRule));
    simplifier.add_rule(Box::new(DenestSqrtAddSqrtRule));
    simplifier.add_rule(Box::new(DenestPerfectCubeInQuadraticFieldRule));
    simplifier.add_rule(Box::new(SimplifySquareRootRule));
    simplifier.add_rule(Box::new(ExtractPerfectSquareFromRadicandRule));
    simplifier.add_rule(Box::new(PullConstantFromFractionRule));
    simplifier.add_rule(Box::new(ExpandRule));
    simplifier.add_rule(Box::new(FactorBasedLCDRule));
    // P2: DifferenceOfSquaresRule for (a-b)(a+b) → a² - b²
    simplifier.add_rule(Box::new(DifferenceOfSquaresRule));
    // R1, R2: Fraction difference canonicalization for cyclic sums
    simplifier.add_rule(Box::new(AbsorbNegationIntoDifferenceRule));
    simplifier.add_rule(Box::new(CanonicalDifferenceProductRule));
    // Factor common integer from sums (POST phase): 2*√2 - 2 → 2*(√2 - 1)
    // Safe because DistributeRule now has PhaseMask excluding POST
    simplifier.add_rule(Box::new(FactorCommonIntegerFromAdd));
    // Polynomial GCD: poly_gcd(a*g, b*g) → g (structural)
    simplifier.add_rule(Box::new(PolyGcdRule));
    // Polynomial GCD exact: poly_gcd_exact(a, b) → algebraic GCD over ℚ
    simplifier.add_rule(Box::new(PolyGcdExactRule));
    // Polynomial GCD mod p: poly_gcd_modp(a, b) → fast Zippel GCD
    simplifier.add_rule(Box::new(gcd_modp::PolyGcdModpRule));
    // Polynomial equality mod p: poly_eq_modp(a, b) → 1 or 0
    simplifier.add_rule(Box::new(gcd_modp::PolyEqModpRule));
    // Polynomial arithmetic on __hold: __hold(P) - __hold(Q) = 0 if equal mod p
    simplifier.add_rule(Box::new(PolySubModpRule));
    // Sum of three cubes identity: x³ + y³ + z³ = 3xyz when x + y + z = 0
    // Handles cyclic differences: (a-b)³ + (b-c)³ + (c-a)³ = 3(a-b)(b-c)(c-a)
    simplifier.add_rule(Box::new(SumThreeCubesZeroRule));
    // Polynomial multiplication mod p: poly_mul_modp(a, b) → poly_result(...)
    poly_mul_modp::register(simplifier);
    // Polynomial stats: poly_stats(poly_result(id)) → metadata
    poly_stats::register(simplifier);
    // simplifier.add_rule(Box::new(FactorDifferenceSquaresRule)); // Too aggressive for default, causes loops with DistributeRule
}

/// Pre-order detection of (A² - B²) / (A ± B) pattern
/// Called BEFORE recursing into children to prevent auto-expand from destroying the pattern.
/// Pre-order rule to recognize and simplify (A² - B²) / (A ± B) → (A ∓ B)
///
/// This function intercepts the pattern BEFORE children are simplified,
/// preventing auto-expand from destroying the algebraic structure.
///
/// # Pre-order Didactic Step Pattern (Reference Implementation)
///
/// This function demonstrates the correct pattern for creating coherent
/// didactic steps in pre-order rules. Key techniques:
///
/// 1. **Create intermediate expressions** for each transformation state:
///    - `intermediate_with_orig_den`: factored numerator, original denominator
///    - `intermediate`: factored numerator, simplified denominator
///
/// 2. **Use `before_local`/`after_local`** to focus the "Rule:" line on
///    the specific sub-expression being transformed, not the whole expression.
///
/// 3. **Chain steps correctly**: Each step's `after` equals next step's `before`.
///
/// 4. **Add conditional steps** (like "Combine like terms") only when the
///    sub-expression actually changes.
///
/// See `crate::step` module documentation for the full pattern description.
///
/// Returns Some(result) if the pattern matches, None otherwise.
#[allow(clippy::too_many_arguments)]
pub fn try_difference_of_squares_preorder(
    ctx: &mut cas_ast::Context,
    expr_id: cas_ast::ExprId,
    num: cas_ast::ExprId,
    den: cas_ast::ExprId,
    allow_abs_square_equiv: bool,
    collect_steps: bool,
    steps: &mut Vec<crate::step::Step>,
    current_path: &[crate::step::PathStep],
) -> Option<cas_ast::ExprId> {
    let plan =
        cas_math::difference_of_squares_support::try_plan_difference_of_squares_division_expr(
            ctx,
            num,
            den,
            cas_math::difference_of_squares_support::DifferenceOfSquaresDivisionPolicy {
                allow_abs_square_equiv,
                ..cas_math::difference_of_squares_support::DifferenceOfSquaresDivisionPolicy::default()
            },
        )?;
    let factored_num = plan.factored_numerator;
    let intermediate_with_orig_den = plan.intermediate_with_orig_den;
    let den_simplified = plan.den_simplified;
    let intermediate = plan.intermediate;
    let final_result = plan.final_result;

    // STEP 4: Record steps if needed (Factor step + optional Simplify Denominator + Cancel step)
    if collect_steps {
        // Step 1: Factor - focus on numerator only
        // Before: original expression, After: factored num with ORIGINAL denominator
        let mut factor_step = crate::step::Step::new(
            "Factor: A² - B² = (A-B)(A+B)",
            "Pre-order Difference of Squares",
            num,
            factored_num,
            current_path.to_vec(),
            Some(ctx),
        );
        factor_step.before = expr_id;
        factor_step.after = intermediate_with_orig_den;
        factor_step.global_before = Some(expr_id);
        factor_step.global_after = Some(intermediate_with_orig_den);
        // Focus Rule line on numerator transformation only
        {
            let meta = factor_step.meta_mut();
            meta.before_local = Some(num);
            meta.after_local = Some(factored_num);
            meta.required_conditions
                .push(crate::ImplicitCondition::NonZero(den));
        }
        factor_step.importance = crate::step::ImportanceLevel::High;
        steps.push(factor_step);

        // Step 1.5 (conditional): Show denominator simplification if it changed
        // Before: factored num with orig den, After: factored num with simplified den
        if den != den_simplified {
            let mut simplify_den_step = crate::step::Step::new(
                "Combine like terms",
                "Combine Like Terms",
                den,
                den_simplified,
                current_path.to_vec(),
                Some(ctx),
            );
            simplify_den_step.before = intermediate_with_orig_den;
            simplify_den_step.after = intermediate;
            simplify_den_step.global_before = Some(intermediate_with_orig_den);
            simplify_den_step.global_after = Some(intermediate);
            {
                let meta = simplify_den_step.meta_mut();
                meta.before_local = Some(den);
                meta.after_local = Some(den_simplified);
            }
            simplify_den_step.importance = crate::step::ImportanceLevel::Medium;
            steps.push(simplify_den_step);
        }

        // Step 2: Cancel
        let mut cancel_step = crate::step::Step::new(
            "Cancel common factor",
            "Pre-order Difference of Squares Cancel",
            intermediate,
            final_result,
            current_path.to_vec(),
            Some(ctx),
        );
        cancel_step.before = intermediate;
        cancel_step.after = final_result;
        cancel_step.global_before = Some(intermediate);
        cancel_step.global_after = Some(final_result);
        cancel_step.importance = crate::step::ImportanceLevel::High;
        steps.push(cancel_step);
    }

    // Return FINAL result to prevent GCD from reprocessing
    Some(final_result)
}

/// Pre-order detection of `(A² - 2AB + B²) / (A - B)` pattern.
/// Called before child recursion to preserve the numerator structure and skip
/// the later didactic fraction pipeline in the plain hot path.
#[allow(clippy::too_many_arguments)]
pub fn try_perfect_square_minus_preorder(
    ctx: &mut cas_ast::Context,
    expr_id: cas_ast::ExprId,
    num: cas_ast::ExprId,
    den: cas_ast::ExprId,
    collect_steps: bool,
    steps: &mut Vec<crate::step::Step>,
    current_path: &[crate::step::PathStep],
) -> Option<cas_ast::ExprId> {
    let plan = crate::rules::algebra::fractions::try_plan_perfect_square_minus_in_num(
        ctx,
        num,
        den,
        collect_steps,
    )?;
    let factored_num = plan.factored_numerator;
    let intermediate = plan.rewritten;
    let final_result = plan.cancelled_result;

    if collect_steps {
        let mut factor_step = crate::step::Step::new(
            "Recognize: A² - 2AB + B² = (A-B)²",
            "Pre-order Perfect Square Minus",
            num,
            factored_num,
            current_path.to_vec(),
            Some(ctx),
        );
        factor_step.before = expr_id;
        factor_step.after = intermediate;
        factor_step.global_before = Some(expr_id);
        factor_step.global_after = Some(intermediate);
        {
            let meta = factor_step.meta_mut();
            meta.before_local = Some(num);
            meta.after_local = Some(factored_num);
            meta.required_conditions
                .push(crate::ImplicitCondition::NonZero(den));
        }
        factor_step.importance = crate::step::ImportanceLevel::High;
        steps.push(factor_step);

        let mut cancel_step = crate::step::Step::new(
            "Cancel common factor",
            "Pre-order Perfect Square Minus Cancel",
            intermediate,
            final_result,
            current_path.to_vec(),
            Some(ctx),
        );
        cancel_step.before = intermediate;
        cancel_step.after = final_result;
        cancel_step.global_before = Some(intermediate);
        cancel_step.global_after = Some(final_result);
        cancel_step.importance = crate::step::ImportanceLevel::High;
        steps.push(cancel_step);
    }

    Some(final_result)
}

/// Pre-order detection of `(A^3 - B^3)/(A-B)` or `(A^3 + B^3)/(A+B)`.
///
/// This runs before recursive child simplification so denominator rewrites like
/// `sin(x)^2 - 1 -> -cos(x)^2` do not destroy the visible common factor first.
#[allow(clippy::too_many_arguments)]
pub fn try_sum_diff_of_cubes_preorder(
    ctx: &mut cas_ast::Context,
    expr_id: cas_ast::ExprId,
    num: cas_ast::ExprId,
    den: cas_ast::ExprId,
    collect_steps: bool,
    steps: &mut Vec<crate::step::Step>,
    current_path: &[crate::step::PathStep],
) -> Option<cas_ast::ExprId> {
    let plan = crate::rules::algebra::fractions::try_plan_sum_diff_of_cubes_in_num(
        ctx,
        num,
        den,
        collect_steps,
    )?;
    let factored_num = plan.factored_numerator;
    let intermediate = plan.rewritten;
    let final_result = plan.cancelled_result;

    if collect_steps {
        let mut factor_step = crate::step::Step::new(
            plan.desc,
            "Pre-order Sum/Difference of Cubes",
            num,
            factored_num,
            current_path.to_vec(),
            Some(ctx),
        );
        factor_step.before = expr_id;
        factor_step.after = intermediate;
        factor_step.global_before = Some(expr_id);
        factor_step.global_after = Some(intermediate);
        {
            let meta = factor_step.meta_mut();
            meta.before_local = Some(num);
            meta.after_local = Some(factored_num);
            meta.required_conditions
                .push(crate::ImplicitCondition::NonZero(den));
        }
        factor_step.importance = crate::step::ImportanceLevel::High;
        steps.push(factor_step);

        let mut cancel_step = crate::step::Step::new(
            "Cancel common factor",
            "Pre-order Sum/Difference of Cubes Cancel",
            intermediate,
            final_result,
            current_path.to_vec(),
            Some(ctx),
        );
        cancel_step.before = intermediate;
        cancel_step.after = final_result;
        cancel_step.global_before = Some(intermediate);
        cancel_step.global_after = Some(final_result);
        cancel_step.importance = crate::step::ImportanceLevel::High;
        steps.push(cancel_step);
    }

    Some(final_result)
}

/// Pre-order detection of exact additive scalar-multiple fractions.
///
/// This is intentionally restricted to the plain hidden path (`steps off`,
/// no listener) from the transformer, so it can return the fully normalized
/// scalar result directly and skip the recursive traversal of both children.
pub fn try_structural_scalar_multiple_preorder(
    ctx: &mut cas_ast::Context,
    num: cas_ast::ExprId,
    den: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let plan =
        cas_math::fraction_gcd_plan_support::try_plan_structural_scalar_multiple_fraction_rewrite(
            ctx, num, den, false,
        )?;
    Some(plan.forms.result_norm)
}

fn collapse_numeric_fraction_result(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let cas_ast::Expr::Div(num, den) = ctx.get(expr) else {
        return expr;
    };
    let (cas_ast::Expr::Number(num), cas_ast::Expr::Number(den)) = (ctx.get(*num), ctx.get(*den))
    else {
        return expr;
    };
    if den.is_zero() {
        return expr;
    }
    ctx.add(cas_ast::Expr::Number(num.clone() / den.clone()))
}

/// Exact pre-order detection of additive scalar-multiple fractions.
///
/// This variant can preserve the visible standard-path contract when
/// `collect_steps` is enabled by synthesizing the same two-step
/// `Simplify Nested Fraction` sequence the rule would otherwise emit.
#[allow(clippy::too_many_arguments)]
pub fn try_exact_scalar_multiple_fraction_preorder(
    ctx: &mut cas_ast::Context,
    expr_id: cas_ast::ExprId,
    num: cas_ast::ExprId,
    den: cas_ast::ExprId,
    collect_steps: bool,
    steps: &mut Vec<crate::step::Step>,
    current_path: &[crate::step::PathStep],
) -> Option<cas_ast::ExprId> {
    let plan =
        cas_math::fraction_gcd_plan_support::try_plan_structural_scalar_multiple_fraction_rewrite(
            ctx,
            num,
            den,
            collect_steps,
        )?;
    let final_result = collapse_numeric_fraction_result(ctx, plan.forms.result);

    if collect_steps {
        if let Some(factored_form_norm) = plan.forms.factored_form_norm {
            let mut factor_step = if current_path.is_empty() {
                crate::step::Step::new_compact(
                    &format!(
                        "Factor by GCD: {}",
                        cas_formatter::render_expr(ctx, plan.gcd_expr)
                    ),
                    "Simplify Nested Fraction",
                    expr_id,
                    factored_form_norm,
                )
            } else {
                crate::step::Step::new(
                    &format!(
                        "Factor by GCD: {}",
                        cas_formatter::render_expr(ctx, plan.gcd_expr)
                    ),
                    "Simplify Nested Fraction",
                    expr_id,
                    factored_form_norm,
                    current_path.to_vec(),
                    Some(ctx),
                )
            };
            factor_step.global_before = Some(expr_id);
            factor_step.global_after = Some(factored_form_norm);
            factor_step.importance = crate::step::ImportanceLevel::High;
            steps.push(factor_step);

            let mut cancel_step = if current_path.is_empty() {
                crate::step::Step::new_compact(
                    "Cancel common factor",
                    "Simplify Nested Fraction",
                    factored_form_norm,
                    plan.forms.result_norm,
                )
            } else {
                crate::step::Step::new(
                    "Cancel common factor",
                    "Simplify Nested Fraction",
                    factored_form_norm,
                    plan.forms.result_norm,
                    current_path.to_vec(),
                    Some(ctx),
                )
            };
            cancel_step.global_before = Some(factored_form_norm);
            cancel_step.global_after = Some(final_result);
            cancel_step.importance = crate::step::ImportanceLevel::High;
            steps.push(cancel_step);
        } else {
            let mut cancel_step = if current_path.is_empty() {
                crate::step::Step::new_compact(
                    "Cancel common factor",
                    "Simplify Nested Fraction",
                    expr_id,
                    plan.forms.result_norm,
                )
            } else {
                crate::step::Step::new(
                    "Cancel common factor",
                    "Simplify Nested Fraction",
                    expr_id,
                    plan.forms.result_norm,
                    current_path.to_vec(),
                    Some(ctx),
                )
            };
            cancel_step.global_before = Some(expr_id);
            cancel_step.global_after = Some(final_result);
            cancel_step.importance = crate::step::ImportanceLevel::High;
            steps.push(cancel_step);
        }
    }

    Some(final_result)
}

/// Exact-shape pre-order detection of `(a^3 - b^3)/(a-b)` and `(a^3 + b^3)/(a+b)`.
///
/// This intentionally avoids algebraic equivalence checks and only handles the
/// raw input shape, which keeps the guard cheap enough for the hidden hot path.
pub fn try_exact_sum_diff_of_cubes_preorder(
    ctx: &mut cas_ast::Context,
    num: cas_ast::ExprId,
    den: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let (a, b, is_difference) = match ctx.get(num) {
        cas_ast::Expr::Sub(left, right) => (
            extract_cube_base(ctx, *left)?,
            extract_cube_base(ctx, *right)?,
            true,
        ),
        cas_ast::Expr::Add(left, right) => (
            extract_cube_base(ctx, *left)?,
            extract_cube_base(ctx, *right)?,
            false,
        ),
        _ => return None,
    };

    match (is_difference, ctx.get(den)) {
        (true, cas_ast::Expr::Sub(dl, dr))
            if expr_eq_fast(ctx, *dl, a) && expr_eq_fast(ctx, *dr, b) => {}
        (false, cas_ast::Expr::Add(dl, dr))
            if expr_eq_fast(ctx, *dl, a) && expr_eq_fast(ctx, *dr, b) => {}
        _ => return None,
    }

    let two = ctx.num(2);
    let a_sq = ctx.add_raw(cas_ast::Expr::Pow(a, two));
    let b_sq = ctx.add_raw(cas_ast::Expr::Pow(b, two));
    let ab = ctx.add_raw(cas_ast::Expr::Mul(a, b));

    if is_difference {
        let sum_ab = ctx.add_raw(cas_ast::Expr::Add(a_sq, b_sq));
        Some(ctx.add_raw(cas_ast::Expr::Add(sum_ab, ab)))
    } else {
        let sum_ab = ctx.add_raw(cas_ast::Expr::Add(a_sq, b_sq));
        Some(ctx.add_raw(cas_ast::Expr::Sub(sum_ab, ab)))
    }
}
