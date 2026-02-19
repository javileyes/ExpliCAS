#[cfg(test)]
mod tests;

pub mod fractions;
pub use fractions::*;

pub mod distribution;
pub use distribution::*;

pub mod factoring;
mod factoring_helpers;
pub use factoring::*;

pub mod roots;
pub use roots::*;

pub mod root_denesting;
pub use root_denesting::*;

pub mod root_denesting_advanced;
pub use root_denesting_advanced::*;

pub mod poly_gcd;
pub use poly_gcd::*;

pub mod gcd_exact;
pub use gcd_exact::*;

pub mod gcd_modp;

pub mod poly_arith_modp;
pub use poly_arith_modp::*;

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
    use cas_ast::Expr;

    // STEP 1: Check if numerator is A² - B²
    // Can be Sub(Pow(A,2), Pow(B,2)) or Add(Pow(A,2), Neg(Pow(B,2)))
    let (a, b) = match ctx.get(num) {
        Expr::Sub(left, right) => {
            // Check if left = A² and right = B²
            let a_opt = extract_squared_base(ctx, *left)?;
            let b_opt = extract_squared_base(ctx, *right)?;
            (a_opt, b_opt)
        }
        Expr::Add(left, right) => {
            // Check for Add(A², Neg(B²))
            let a_opt = extract_squared_base(ctx, *left)?;
            if let Expr::Neg(inner) = ctx.get(*right) {
                let b_opt = extract_squared_base(ctx, *inner)?;
                (a_opt, b_opt)
            } else {
                return None;
            }
        }
        _ => return None,
    };

    #[allow(dead_code)]
    enum DenMatch {
        AMinusB,
        APlusB,
        BMinusA,
        BPlusA,
    }

    // STEP 2: Quick structural pre-filter to avoid expensive poly conversion
    // The denominator must be Sub/Add to match A-B or A+B, so skip poly conversion
    // for simple cases like x, x*y, etc.
    let den_is_candidate = matches!(ctx.get(den), Expr::Sub(_, _) | Expr::Add(_, _));
    if !den_is_candidate {
        return None; // Denominator can't possibly be A-B or A+B
    }

    // STEP 3: Check if denominator matches A-B, A+B using POLYNOMIAL comparison
    // This handles cases like den = (x+2y) - (3x-y) being compared to A-B
    // where A and B come from different AST nodes but are polynomially equal
    use cas_math::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};

    let budget = PolyBudget {
        max_terms: 50,
        max_total_degree: 6,
        max_pow_exp: 4,
    };

    // Build A-B and A+B as raw expressions
    let a_minus_b_raw = ctx.add(Expr::Sub(a, b));
    let a_plus_b_raw = ctx.add(Expr::Add(a, b));

    // Convert to polynomials for comparison
    let den_poly = multipoly_from_expr(ctx, den, &budget).ok()?;
    let a_minus_b_poly = multipoly_from_expr(ctx, a_minus_b_raw, &budget).ok()?;
    let a_plus_b_poly = multipoly_from_expr(ctx, a_plus_b_raw, &budget).ok()?;

    // Check which form the denominator matches
    let _den_match = if den_poly == a_minus_b_poly {
        DenMatch::AMinusB
    } else if den_poly == a_minus_b_poly.neg() {
        DenMatch::BMinusA // den = -(A-B) = B-A
    } else if den_poly == a_plus_b_poly {
        DenMatch::APlusB
    } else if den_poly == a_plus_b_poly.neg() {
        DenMatch::BPlusA // den = -(A+B) = -A-B (same as B+A up to sign)
    } else {
        return None; // No match
    };

    // STEP 3: Build FINAL result and intermediate forms
    // A² - B² = (A-B)(A+B)
    // If den = A-B: result = A+B
    // If den = B-A = -(A-B): result = -(A+B)
    // If den = A+B: result = A-B
    // If den = B+A = -(A+B): result = -(A-B) = B-A

    // Simplify factors to canonical polynomial form
    let a_minus_b = if let Ok(p) = multipoly_from_expr(ctx, a_minus_b_raw, &budget) {
        multipoly_to_expr(&p, ctx)
    } else {
        a_minus_b_raw
    };

    let a_plus_b = if let Ok(p) = multipoly_from_expr(ctx, a_plus_b_raw, &budget) {
        multipoly_to_expr(&p, ctx)
    } else {
        a_plus_b_raw
    };

    // Simplify denominator to canonical form
    let den_simplified = if let Ok(p) = multipoly_from_expr(ctx, den, &budget) {
        multipoly_to_expr(&p, ctx)
    } else {
        den
    };

    // Build factored numerator and compute final result based on den match
    let factored_num = ctx.add(Expr::Mul(a_minus_b, a_plus_b));
    // Two intermediate states for proper didactic step sequence:
    // 1. Factored numerator with ORIGINAL denominator (for Factor step)
    // 2. Factored numerator with SIMPLIFIED denominator (for Combine step → Cancel step)
    let intermediate_with_orig_den = ctx.add(Expr::Div(factored_num, den));
    let intermediate = ctx.add(Expr::Div(factored_num, den_simplified));

    let final_result = match _den_match {
        DenMatch::AMinusB => a_plus_b, // A² - B² = (A-B)(A+B), cancel (A-B) -> A+B
        DenMatch::BMinusA => {
            // den = -(A-B), cancel (A-B) -> -(A+B)
            ctx.add(Expr::Neg(a_plus_b))
        }
        DenMatch::APlusB | DenMatch::BPlusA => a_minus_b, // cancel (A+B) -> A-B
    };

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

/// Helper: Extract base from Pow(base, 2)
fn extract_squared_base(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> Option<cas_ast::ExprId> {
    use cas_ast::Expr;
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*exp) {
            if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into()) {
                return Some(*base);
            }
        }
    }
    None
}
