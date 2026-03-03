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

pub fn register(simplifier: &mut crate::Simplifier) {
    // Pre-order: Cancel cube root difference pattern BEFORE GCD/fraction rules
    // (x - b³) / (x^(2/3) + b·x^(1/3) + b²) → x^(1/3) - b
    simplifier.add_rule(Box::new(CancelCubeRootDifferenceRule));

    // V2.14.35: Ultra-light P^m/P^n → P^(m-n) rule (shallow ExprId comparison)
    // Must fire BEFORE heavier fraction rules to avoid stack overflow on large powers
    simplifier.add_rule(Box::new(CancelPowersDivisionRule));
    // Step 2 of didactic expansion: Cancel P/P → 1 (must fire after expansion rules)
    simplifier.add_rule(Box::new(CancelIdenticalFractionRule));
    // Step 2 variant: Cancel P^n/P → P^(n-1) (for perfect squares and similar)
    simplifier.add_rule(Box::new(CancelPowerFractionRule));
    simplifier.add_rule(Box::new(SimplifyFractionRule));
    simplifier.add_rule(Box::new(NestedFractionRule));
    simplifier.add_rule(Box::new(SimplifyMulDivRule));
    simplifier.add_rule(Box::new(FoldAddIntoFractionRule));
    simplifier.add_rule(Box::new(SubTermMatchesDenomRule)); // a - b/a → (a²-b)/a
    simplifier.add_rule(Box::new(AddFractionsRule));
    simplifier.add_rule(Box::new(SubFractionsRule));
    simplifier.add_rule(Box::new(CombineSameDenominatorFractionsRule));
    // Distribute numeric denominator into sums: (2x+4)/2 → x+2
    simplifier.add_rule(Box::new(DivScalarIntoAddRule));
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
    simplifier.add_rule(Box::new(DivExpandToCancelRule)); // Expand Mul(Add,Add) in Div to enable cancellation
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
    collect_steps: bool,
    steps: &mut Vec<crate::step::Step>,
    current_path: &[crate::step::PathStep],
) -> Option<cas_ast::ExprId> {
    let plan =
        cas_math::difference_of_squares_support::try_plan_difference_of_squares_division_expr(
            ctx,
            num,
            den,
            cas_math::difference_of_squares_support::DifferenceOfSquaresDivisionPolicy::default(),
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
                .push(crate::implicit_domain::ImplicitCondition::NonZero(den));
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
