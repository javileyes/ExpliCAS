use crate::build::mul2_raw;
use crate::define_rule;
use crate::multipoly::{
    gcd_multivar_layer2, gcd_multivar_layer25, multipoly_from_expr, multipoly_to_expr, GcdBudget,
    GcdLayer, Layer25Budget, MultiPoly, PolyBudget,
};
use crate::phase::PhaseMask;
use crate::polynomial::Polynomial;
use crate::rule::Rewrite;
use cas_ast::count_nodes;
use cas_ast::{Context, DisplayExpr, Expr, ExprId};
use num_traits::{One, Signed, Zero};
use std::cmp::Ordering;

use super::helpers::*;

// =============================================================================
// Polynomial equality helper (for canonical comparison ignoring AST order)
// =============================================================================

/// Compare two expressions as polynomials (ignoring AST structure/order).
/// Returns true if both convert to the same canonical polynomial form.
/// Falls back to false if either is not a polynomial (budget exceeded, non-poly, etc).
fn poly_eq(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    // Use a tight budget for recognizer comparisons
    let budget = PolyBudget {
        max_terms: 100,
        max_total_degree: 10,
        max_pow_exp: 5,
    };

    let pa = match multipoly_from_expr(ctx, a, &budget) {
        Ok(p) => p,
        Err(_) => return false,
    };
    let pb = match multipoly_from_expr(ctx, b, &budget) {
        Ok(p) => p,
        Err(_) => return false,
    };

    pa == pb
}

// =============================================================================
// A1: Structural Factor Cancellation (without polynomial expansion)
// =============================================================================

/// Collect multiplicative factors with integer exponents from an expression.
/// - Mul(...) is flattened
/// - Pow(base, k) with integer k becomes (base, k)
/// - Everything else becomes (expr, 1)
#[allow(dead_code)]
fn collect_mul_factors_int_pow(ctx: &Context, expr: ExprId) -> Vec<(ExprId, i64)> {
    let mut factors = Vec::new();
    collect_mul_factors_recursive(ctx, expr, 1, &mut factors);
    factors
}

fn collect_mul_factors_recursive(
    ctx: &Context,
    expr: ExprId,
    mult: i64,
    factors: &mut Vec<(ExprId, i64)>,
) {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            // Binary Mul: recursively collect both sides
            collect_mul_factors_recursive(ctx, *left, mult, factors);
            collect_mul_factors_recursive(ctx, *right, mult, factors);
        }
        Expr::Pow(base, exp) => {
            // Check if exponent is an integer
            if let Some(k) = get_integer_exponent_a1(ctx, *exp) {
                factors.push((*base, mult * k));
            } else {
                factors.push((expr, mult));
            }
        }
        _ => {
            factors.push((expr, mult));
        }
    }
}

/// Extract integer from exponent expression (Number or Neg(Number))
fn get_integer_exponent_a1(ctx: &Context, exp: ExprId) -> Option<i64> {
    match ctx.get(exp) {
        Expr::Number(n) => {
            if n.is_integer() {
                n.to_integer().try_into().ok()
            } else {
                None
            }
        }
        Expr::Neg(inner) => get_integer_exponent_a1(ctx, *inner).map(|k| -k),
        _ => None,
    }
}

/// Cancel common factors between numerator and denominator.
/// Returns (new_num, new_den, cancelled_factors) if any cancellation happened.
#[allow(dead_code)]
fn cancel_common_factors_structural(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> Option<(ExprId, ExprId, Vec<ExprId>)> {
    let mut num_factors = collect_mul_factors_int_pow(ctx, num);
    let mut den_factors = collect_mul_factors_int_pow(ctx, den);

    // Sort both by canonical ordering for merge
    num_factors.sort_by(|(a, _), (b, _)| compare_expr_for_sort_a1(ctx, *a, *b));
    den_factors.sort_by(|(a, _), (b, _)| compare_expr_for_sort_a1(ctx, *a, *b));

    let mut cancelled: Vec<ExprId> = Vec::new();
    let mut new_num_factors: Vec<(ExprId, i64)> = Vec::new();
    let mut new_den_factors: Vec<(ExprId, i64)> = Vec::new();

    let mut i = 0;
    let mut j = 0;

    while i < num_factors.len() && j < den_factors.len() {
        let (num_base, num_exp) = num_factors[i];
        let (den_base, den_exp) = den_factors[j];

        match compare_expr_for_sort_a1(ctx, num_base, den_base) {
            Ordering::Less => {
                new_num_factors.push((num_base, num_exp));
                i += 1;
            }
            Ordering::Greater => {
                new_den_factors.push((den_base, den_exp));
                j += 1;
            }
            Ordering::Equal => {
                // Same base - cancel exponents
                let diff = num_exp - den_exp;
                if diff > 0 {
                    new_num_factors.push((num_base, diff));
                } else if diff < 0 {
                    new_den_factors.push((den_base, -diff));
                }
                // Track cancelled factor
                cancelled.push(num_base);
                i += 1;
                j += 1;
            }
        }
    }

    // Add remaining factors
    while i < num_factors.len() {
        new_num_factors.push(num_factors[i]);
        i += 1;
    }
    while j < den_factors.len() {
        new_den_factors.push(den_factors[j]);
        j += 1;
    }

    // If nothing was cancelled, return None
    if cancelled.is_empty() {
        return None;
    }

    // Build new numerator and denominator
    let new_num = build_mul_from_factors_a1(ctx, &new_num_factors);
    let new_den = build_mul_from_factors_a1(ctx, &new_den_factors);

    Some((new_num, new_den, cancelled))
}

/// Compare expressions for sorting (wrapper for canonical comparison)
#[allow(dead_code)]
fn compare_expr_for_sort_a1(ctx: &Context, a: ExprId, b: ExprId) -> Ordering {
    // Use DisplayExpr for string representation
    let a_str = format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: a
        }
    );
    let b_str = format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: b
        }
    );
    a_str.cmp(&b_str)
}

/// Build a product from factors with integer exponents.
///
/// Uses canonical `MulBuilder` (right-fold with exponents).
/// (See ARCHITECTURE.md "Canonical Utilities Registry")
#[allow(dead_code)]
fn build_mul_from_factors_a1(ctx: &mut Context, factors: &[(ExprId, i64)]) -> ExprId {
    use cas_ast::views::MulBuilder;

    let mut builder = MulBuilder::new_simple();
    for &(base, exp) in factors {
        if exp > 0 {
            builder.push_pow(base, exp);
        }
        // Negative exponents shouldn't appear in numerator/denominator factors
    }
    builder.build(ctx)
}

/// Build domain assumption for cancelled factors.
/// Returns Some(...) if non-numeric factors were cancelled.
#[allow(dead_code)]
fn build_cancel_domain_assumption(ctx: &Context, cancelled: &[ExprId]) -> Option<&'static str> {
    if cancelled.is_empty() {
        return None;
    }
    // Check if any cancelled factor is non-numeric
    for &factor in cancelled {
        match ctx.get(factor) {
            Expr::Number(_) => continue, // Numbers are always ≠ 0 if we cancelled them
            _ => return Some("Assuming cancelled factor ≠ 0"),
        }
    }
    None
}

// NOTE: A1 structural factor cancellation helper functions above can be integrated
// into the existing CancelCommonFactorsRule (line ~1460) to add power support.
// The existing rule already handles basic factor cancellation.

// =============================================================================
// Multivariate GCD (Layers 1 + 2 + 2.5)
// =============================================================================

/// Align MultiPoly to a target variable set by embedding it into the larger space.
/// For example, a polynomial in [h] can be embedded into [h, x] by adding zero exponents for x.
fn align_multipoly_vars(p: &MultiPoly, target_vars: &[String]) -> MultiPoly {
    use std::collections::BTreeMap;

    if p.vars == target_vars {
        return p.clone();
    }

    // Map old var indices to new indices
    let mapping: Vec<Option<usize>> = p
        .vars
        .iter()
        .map(|v| target_vars.iter().position(|tv| tv == v))
        .collect();

    let mut new_terms: Vec<(num_rational::BigRational, Vec<u32>)> = Vec::new();
    let mut term_map: BTreeMap<Vec<u32>, num_rational::BigRational> = BTreeMap::new();

    for (coeff, mono) in &p.terms {
        let mut new_mono = vec![0u32; target_vars.len()];
        for (old_idx, &exp) in mono.iter().enumerate() {
            if let Some(Some(new_idx)) = mapping.get(old_idx) {
                new_mono[*new_idx] = exp;
            }
        }
        let entry = term_map
            .entry(new_mono)
            .or_insert_with(num_rational::BigRational::zero);
        *entry = entry.clone() + coeff.clone();
    }

    for (mono, coeff) in term_map {
        if !coeff.is_zero() {
            new_terms.push((coeff, mono));
        }
    }

    MultiPoly {
        vars: target_vars.to_vec(),
        terms: new_terms,
    }
}

/// Try to compute GCD of two expressions using multivariate polynomial representation.
/// Returns None if expressions can't be converted to polynomials or if GCD is trivial (1).
/// Returns Some((quotient_num, quotient_den, gcd_expr, layer)) if non-trivial GCD found.
fn try_multivar_gcd(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
) -> Option<(ExprId, ExprId, ExprId, GcdLayer)> {
    let budget = PolyBudget::default();

    // Try to convert both to MultiPoly
    let p_num = multipoly_from_expr(ctx, num, &budget).ok()?;
    let p_den = multipoly_from_expr(ctx, den, &budget).ok()?;

    // Skip if not multivariate (let univariate path handle it)
    if p_num.vars.len() <= 1 {
        return None;
    }

    // Align variables if needed: embed both into union of variables
    let (p_num, p_den) = if p_num.vars != p_den.vars {
        // Compute union of variables (sorted for consistency)
        let mut all_vars: Vec<String> = p_num
            .vars
            .iter()
            .chain(p_den.vars.iter())
            .cloned()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        all_vars.sort();

        // Embed both polynomials into the shared variable space
        let p_num_aligned = align_multipoly_vars(&p_num, &all_vars);
        let p_den_aligned = align_multipoly_vars(&p_den, &all_vars);
        (p_num_aligned, p_den_aligned)
    } else {
        (p_num, p_den)
    };

    // Layer 1: Monomial GCD
    let mono_gcd = p_num.monomial_gcd_with(&p_den).ok()?;
    let has_mono_gcd = mono_gcd.iter().any(|&e| e > 0);

    // Layer 1: Content GCD
    let content_num = p_num.content();
    let content_den = p_den.content();
    let content_gcd = gcd_rational(content_num.clone(), content_den.clone());
    let has_content_gcd = !content_gcd.is_one();

    // If no GCD found at Layer 1, try Layer 2 and Layer 2.5
    if !has_mono_gcd && !has_content_gcd {
        if p_num.vars.len() >= 2 {
            // Try Layer 2 first (bivar with seeds)
            let gcd_budget = GcdBudget::default();
            if let Some(gcd_poly) = gcd_multivar_layer2(&p_num, &p_den, &gcd_budget) {
                if !gcd_poly.is_one() && !gcd_poly.is_constant() {
                    let q_num = p_num.div_exact(&gcd_poly)?;
                    let q_den = p_den.div_exact(&gcd_poly)?;
                    let new_num = multipoly_to_expr(&q_num, ctx);
                    let new_den = multipoly_to_expr(&q_den, ctx);
                    let gcd_expr = multipoly_to_expr(&gcd_poly, ctx);
                    return Some((new_num, new_den, gcd_expr, GcdLayer::Layer2HeuristicSeeds));
                }
            }

            // Try Layer 2.5 (tensor grid for multi-param factors)
            let layer25_budget = Layer25Budget::default();
            if let Some(gcd_poly) = gcd_multivar_layer25(&p_num, &p_den, &layer25_budget) {
                if !gcd_poly.is_one() && !gcd_poly.is_constant() {
                    let q_num = p_num.div_exact(&gcd_poly)?;
                    let q_den = p_den.div_exact(&gcd_poly)?;
                    let new_num = multipoly_to_expr(&q_num, ctx);
                    let new_den = multipoly_to_expr(&q_den, ctx);
                    let gcd_expr = multipoly_to_expr(&gcd_poly, ctx);
                    return Some((new_num, new_den, gcd_expr, GcdLayer::Layer25TensorGrid));
                }
            }
        }
        return None;
    }

    // Divide by monomial GCD
    let (p_num, p_den) = if has_mono_gcd {
        (
            p_num.div_monomial_exact(&mono_gcd)?,
            p_den.div_monomial_exact(&mono_gcd)?,
        )
    } else {
        (p_num, p_den)
    };

    // Divide by content GCD
    let (p_num, p_den) = if has_content_gcd {
        (
            p_num.div_scalar_exact(&content_gcd)?,
            p_den.div_scalar_exact(&content_gcd)?,
        )
    } else {
        (p_num, p_den)
    };

    // Build GCD expression (monomial * content)
    let mut gcd_poly = MultiPoly::one(p_num.vars.clone());

    if has_mono_gcd {
        gcd_poly = gcd_poly.mul_monomial(&mono_gcd).ok()?;
    }

    if has_content_gcd {
        gcd_poly = gcd_poly.mul_scalar(&content_gcd);
    }

    // Convert back to expressions
    let new_num = multipoly_to_expr(&p_num, ctx);
    let new_den = multipoly_to_expr(&p_den, ctx);
    let gcd_expr = multipoly_to_expr(&gcd_poly, ctx);

    Some((new_num, new_den, gcd_expr, GcdLayer::Layer1MonomialContent))
}

// ========== Helper to extract fraction parts from both Div and Mul(1/n,x) ==========
// This is needed because canonicalization may convert Div(x,n) to Mul(1/n,x)

/// Extract (numerator, denominator, is_fraction) from an expression.
/// Recognizes:
/// - Div(num, den) → (num, den, true)
/// - Mul(Number(1/n), x) or Mul(x, Number(1/n)) → (x, n, true) where numerator of coeff is ±1
/// - Mul(Div(1,den), x) or Mul(x, Div(1,den)) → (x, den, true) for symbolic denominators
/// - anything else → (expr, 1, false)
fn extract_as_fraction(ctx: &mut Context, expr: ExprId) -> (ExprId, ExprId, bool) {
    use num_bigint::BigInt;
    use num_rational::BigRational;
    use num_traits::Signed;

    // Case 1: Direct Div - use zero-clone destructuring
    if let Some((num, den)) = crate::helpers::as_div(ctx, expr) {
        return (num, den, true);
    }

    // Case 2 & 3: Mul with fractional coefficient
    if let Some((l, r)) = crate::helpers::as_mul(ctx, expr) {
        // Helper to check if a Number is ±1/n and extract denominator
        let check_unit_fraction = |n: &BigRational| -> Option<(BigInt, bool)> {
            if n.is_integer() {
                return None;
            }
            let numer = n.numer();
            let abs_numer: BigInt = if numer < &BigInt::from(0) {
                -numer.clone()
            } else {
                numer.clone()
            };
            if abs_numer == BigInt::from(1) {
                let is_negative = numer.is_negative();
                return Some((n.denom().clone(), is_negative));
            }
            None
        };

        // Helper to check if expression is Div(1, den) or Div(-1, den)
        let check_unit_div = |factor: ExprId| -> Option<(ExprId, bool)> {
            let (num, den) = crate::helpers::as_div(ctx, factor)?;
            let n = crate::helpers::as_number(ctx, num)?;
            if n.is_integer() {
                let n_val = n.numer();
                if *n_val == BigInt::from(1) {
                    return Some((den, false));
                } else if *n_val == BigInt::from(-1) {
                    return Some((den, true));
                }
            }
            None
        };

        // Case 2: Check for Number(±1/n)
        if let Some(n) = crate::helpers::as_number(ctx, l) {
            if let Some((denom, is_neg)) = check_unit_fraction(n) {
                let denom_expr = ctx.add(Expr::Number(BigRational::from_integer(denom)));
                if is_neg {
                    let neg_r = ctx.add(Expr::Neg(r));
                    return (neg_r, denom_expr, true);
                }
                return (r, denom_expr, true);
            }
        }
        if let Some(n) = crate::helpers::as_number(ctx, r) {
            if let Some((denom, is_neg)) = check_unit_fraction(n) {
                let denom_expr = ctx.add(Expr::Number(BigRational::from_integer(denom)));
                if is_neg {
                    let neg_l = ctx.add(Expr::Neg(l));
                    return (neg_l, denom_expr, true);
                }
                return (l, denom_expr, true);
            }
        }

        // Case 3: Check for Div(1, den) or Div(-1, den) - symbolic denominator
        if let Some((den, is_neg)) = check_unit_div(l) {
            if is_neg {
                let neg_r = ctx.add(Expr::Neg(r));
                return (neg_r, den, true);
            }
            return (r, den, true);
        }
        if let Some((den, is_neg)) = check_unit_div(r) {
            if is_neg {
                let neg_l = ctx.add(Expr::Neg(l));
                return (neg_l, den, true);
            }
            return (l, den, true);
        }
    }

    // Not a recognized fraction form
    let one = ctx.num(1);
    (expr, one, false)
}

// ========== Micro-API for safe Mul construction ==========
// Use this instead of ctx.add(Expr::Mul(...)) in this file.

// =============================================================================
// STEP 2: Cancel identical numerator/denominator (P/P → 1)
// =============================================================================

// Cancels P/P → 1 when numerator equals denominator structurally.
// This is Step 2 after didactic expansion rules (e.g., (a+b)² → a² + 2ab + b²).
define_rule!(
    CancelIdenticalFractionRule,
    "Cancel Identical Numerator/Denominator",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::helpers::prove_nonzero;
        use crate::implicit_domain::ImplicitCondition;

        // Match Div(num, den)
        let (num, den) = crate::helpers::as_div(ctx, expr)?;

        // Check if num == den structurally
        if crate::ordering::compare_expr(ctx, num, den) != std::cmp::Ordering::Equal {
            return None;
        }

        // DOMAIN GATE: In Strict mode, only cancel if den is provably non-zero
        let domain_mode = parent_ctx.domain_mode();
        let proof = prove_nonzero(ctx, den);
        let key = crate::assumptions::AssumptionKey::nonzero_key(ctx, den);
        let decision = crate::domain::can_cancel_factor_with_hint(
            domain_mode,
            proof,
            key,
            den,
            "Cancel Identical Numerator/Denominator",
        );

        if !decision.allow {
            // Strict mode + Unknown proof: don't simplify (e.g., x/x stays)
            return None;
        }

        // Match! P/P → 1
        let one = ctx.num(1);

        Some(Rewrite::new(one)
            .desc("Cancel: P/P → 1")
            .local(expr, one)
            .requires(ImplicitCondition::NonZero(den))
            .assume_all(decision.assumption_events(ctx, den)))
    }
);

// =============================================================================
// EARLY RETURN: Didactic expansion of perfect-square denominators for cancellation
// =============================================================================

/// Try to detect and cancel `(a^2 + 2ab + b^2) / (a+b)^2 → 1` with visible expansion step.
///
/// This avoids the "magic" GCD path by showing the user that (a+b)^2 = a^2 + 2ab + b^2.
/// Returns Some(Rewrite) if the pattern matches, None otherwise to fall through to GCD.
fn try_expand_binomial_square_in_den_for_cancel(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    _domain_mode: crate::domain::DomainMode,
    _parent_ctx: &crate::parent_context::ParentContext,
) -> Option<Rewrite> {
    use crate::implicit_domain::ImplicitCondition;
    use crate::rules::algebra::helpers::smart_mul;

    // STEP 1: Check if den = Pow(base, 2) where base = Add(a, b)
    let (base, exp) = crate::helpers::as_pow(ctx, den)?;

    // Exponent must be exactly 2 (integer)
    let exp_val = crate::helpers::as_i64(ctx, exp)?;
    if exp_val != 2 {
        return None;
    }

    // Base must be a binomial (a + b)
    let (a, b) = crate::helpers::as_add(ctx, base)?;

    // STEP 2: Build expanded form: a^2 + 2*a*b + b^2
    // Using right-associative Add: Add(a^2, Add(2*a*b, b^2)) to match typical parser output
    let two = ctx.num(2);
    let exp_two = ctx.num(2);

    let a_sq = ctx.add(Expr::Pow(a, exp_two));
    let exp_two_b = ctx.num(2);
    let b_sq = ctx.add(Expr::Pow(b, exp_two_b));
    // Split smart_mul calls to avoid nested mutable borrows
    let a_times_b = smart_mul(ctx, a, b);
    let two_ab = smart_mul(ctx, two, a_times_b);

    // Build: a^2 + 2*a*b + b^2  (right-associative)
    let middle_sum = ctx.add(Expr::Add(two_ab, b_sq));
    let expanded = ctx.add(Expr::Add(a_sq, middle_sum));

    // STEP 3: Check if num equals expanded using polynomial comparison
    // This handles reordering by parser
    if !poly_eq(ctx, num, expanded) {
        return None;
    }

    // STEP 4: Match! Create didactic rewrite
    // Return Div(num, expanded_den) - NOT `1` directly!
    // This allows a separate CancelIdenticalFractionRule to fire and show P/P → 1
    let after = ctx.add(Expr::Div(num, expanded));

    // Create the rewrite with:
    // - before_local = den (the (a+b)^2)
    // - after_local = expanded (a^2 + 2ab + b^2)
    // - requires = den ≠ 0 (not assumption_events!)
    let rewrite = Rewrite::new(after)
        .desc("Expand: (a+b)² → a² + 2ab + b²")
        .local(den, expanded)
        .requires(ImplicitCondition::NonZero(den));

    Some(rewrite)
}

// =============================================================================
// EARLY RETURN: Difference of squares factorization for cancellation
// =============================================================================

// Detects (a² - b²) / (a+b) or (a² - b²) / (a-b) and factors the numerator.
// Returns Div((a-b)(a+b), den) to allow visible cancellation step.
fn try_difference_of_squares_in_num(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    _domain_mode: crate::domain::DomainMode,
    _parent_ctx: &crate::parent_context::ParentContext,
) -> Option<Rewrite> {
    use crate::implicit_domain::ImplicitCondition;
    use crate::rules::algebra::helpers::smart_mul;

    // STEP 1: Check if num is a² - b² form
    // Try Sub(Pow(a,2), Pow(b,2)) first
    let (a, b) = if let Some((left, right)) = crate::helpers::as_sub(ctx, num) {
        // left - right, check if both are squares
        let (a, exp_a) = crate::helpers::as_pow(ctx, left)?;
        let exp_a_val = crate::helpers::as_i64(ctx, exp_a)?;
        if exp_a_val != 2 {
            return None;
        }

        let (b, exp_b) = crate::helpers::as_pow(ctx, right)?;
        let exp_b_val = crate::helpers::as_i64(ctx, exp_b)?;
        if exp_b_val != 2 {
            return None;
        }

        (a, b)
    } else if let Some((left, right)) = crate::helpers::as_add(ctx, num) {
        // Try Add(Pow(a,2), Neg(Pow(b,2))) which is how parser represents a² - b²
        let (a, exp_a) = crate::helpers::as_pow(ctx, left)?;
        let exp_a_val = crate::helpers::as_i64(ctx, exp_a)?;
        if exp_a_val != 2 {
            return None;
        }

        // right must be Neg(Pow(b,2))
        let neg_inner = crate::helpers::as_neg(ctx, right)?;
        let (b, exp_b) = crate::helpers::as_pow(ctx, neg_inner)?;
        let exp_b_val = crate::helpers::as_i64(ctx, exp_b)?;
        if exp_b_val != 2 {
            return None;
        }

        (a, b)
    } else {
        return None; // Not a subtraction
    };

    // STEP 2: Check if den matches (a+b) or (a-b)
    let den_matches_a_plus_b = if let Some((da, db)) = crate::helpers::as_add(ctx, den) {
        // Check if {da, db} == {a, b} in some order
        (crate::ordering::compare_expr(ctx, da, a) == std::cmp::Ordering::Equal
            && crate::ordering::compare_expr(ctx, db, b) == std::cmp::Ordering::Equal)
            || (crate::ordering::compare_expr(ctx, da, b) == std::cmp::Ordering::Equal
                && crate::ordering::compare_expr(ctx, db, a) == std::cmp::Ordering::Equal)
    } else {
        false
    };

    let den_matches_a_minus_b = if let Some((da, db)) = crate::helpers::as_sub(ctx, den) {
        // Check if den = a - b
        crate::ordering::compare_expr(ctx, da, a) == std::cmp::Ordering::Equal
            && crate::ordering::compare_expr(ctx, db, b) == std::cmp::Ordering::Equal
    } else {
        false
    };

    if !den_matches_a_plus_b && !den_matches_a_minus_b {
        return None;
    }

    // STEP 3: Build factored form (a-b)(a+b) for display
    let a_minus_b = ctx.add(Expr::Sub(a, b));
    let a_plus_b = ctx.add(Expr::Add(a, b));
    let factored_num = smart_mul(ctx, a_minus_b, a_plus_b);

    // STEP 4: Determine the result after cancellation
    // If den matches (a+b), result is (a-b)
    // If den matches (a-b), result is (a+b)
    let result = if den_matches_a_plus_b {
        a_minus_b
    } else {
        a_plus_b
    };

    // Create the rewrite showing factorization AND cancellation in one step
    // local shows: a² - b² → (a-b)(a+b)
    let rewrite = Rewrite::new(result)
        .desc("Factor and cancel: a² - b² = (a-b)(a+b)")
        .local(num, factored_num)
        .requires(ImplicitCondition::NonZero(den));

    Some(rewrite)
}

// =============================================================================
// EARLY RETURN: Perfect square minus factorization (a-b)²
// =============================================================================

// Detects (a² - 2ab + b²) / (a-b) and recognizes it as (a-b)²/(a-b) = a-b
fn try_perfect_square_minus_in_num(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    _domain_mode: crate::domain::DomainMode,
    _parent_ctx: &crate::parent_context::ParentContext,
) -> Option<Rewrite> {
    use crate::implicit_domain::ImplicitCondition;
    use crate::rules::algebra::helpers::smart_mul;

    // STEP 1: Check if den is (a - b) - try Sub first, then Add(a, Neg(b))
    let (a, b) = if let Some((a, b)) = crate::helpers::as_sub(ctx, den) {
        (a, b)
    } else if let Some((left, right)) = crate::helpers::as_add(ctx, den) {
        // Try Add(a, Neg(b)) which is how parser represents a - b
        if let Some(neg_inner) = crate::helpers::as_neg(ctx, right) {
            (left, neg_inner)
        } else if let Some(neg_inner) = crate::helpers::as_neg(ctx, left) {
            // Try Add(Neg(b), a) = a - b = -(b - a)
            (right, neg_inner) // This gives (a, b) = (right, inner) which is b - (inner)
        } else {
            return None;
        }
    } else {
        return None;
    };

    // STEP 2: Build expected numerator: a² - 2ab + b²
    let two = ctx.num(2);
    let exp_two = ctx.num(2);
    let a_sq = ctx.add(Expr::Pow(a, exp_two));
    let exp_two_b = ctx.num(2);
    let b_sq = ctx.add(Expr::Pow(b, exp_two_b));
    let a_b = smart_mul(ctx, a, b);
    let two_ab = smart_mul(ctx, two, a_b);
    let neg_two_ab = ctx.add(Expr::Neg(two_ab));

    // Build a² + (-2ab) + b² (canonical form for a² - 2ab + b²)
    // Right-associative: Add(a², Add(Neg(2ab), b²))
    let inner_sum = ctx.add(Expr::Add(neg_two_ab, b_sq));
    let expected_num = ctx.add(Expr::Add(a_sq, inner_sum));

    // STEP 3: Check if num equals expected using polynomial comparison
    // This handles reordering by parser (e.g., y² + x² - 2xy vs x² - 2xy + y²)
    if !poly_eq(ctx, num, expected_num) {
        return None;
    }

    // STEP 4: Build factored form (a-b)² for display
    let a_minus_b = ctx.add(Expr::Sub(a, b));
    let exp_for_square = ctx.num(2);
    let factored_num = ctx.add(Expr::Pow(a_minus_b, exp_for_square));

    // Result is (a-b) after cancellation
    let result = a_minus_b;

    let rewrite = Rewrite::new(result)
        .desc("Recognize and cancel: a² - 2ab + b² = (a-b)²")
        .local(num, factored_num)
        .requires(ImplicitCondition::NonZero(den));

    Some(rewrite)
}

// =============================================================================
// EARLY RETURN: Sum/Difference of cubes factorization
// =============================================================================

// Detects (a³ - b³) / (a-b) and recognizes it as (a-b)(a²+ab+b²)/(a-b) = a²+ab+b²
// Also handles (a³ + b³) / (a+b) = (a+b)(a²-ab+b²)/(a+b) = a²-ab+b²
fn try_sum_diff_of_cubes_in_num(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    _domain_mode: crate::domain::DomainMode,
    _parent_ctx: &crate::parent_context::ParentContext,
) -> Option<Rewrite> {
    use crate::implicit_domain::ImplicitCondition;
    use crate::rules::algebra::helpers::smart_mul;

    // Check if num is a³ - b³ (as Sub or Add with Neg)
    let (a, b, is_difference) = if let Some((left, right)) = crate::helpers::as_sub(ctx, num) {
        // a³ - b³
        let (a, exp_a) = crate::helpers::as_pow(ctx, left)?;
        if crate::helpers::as_i64(ctx, exp_a)? != 3 {
            return None;
        }

        let (b, exp_b) = crate::helpers::as_pow(ctx, right)?;
        if crate::helpers::as_i64(ctx, exp_b)? != 3 {
            return None;
        }

        (a, b, true)
    } else if let Some((left, right)) = crate::helpers::as_add(ctx, num) {
        // Try a³ + (-b³) for difference, or a³ + b³ for sum
        let (a, exp_a) = crate::helpers::as_pow(ctx, left)?;
        if crate::helpers::as_i64(ctx, exp_a)? != 3 {
            return None;
        }

        if let Some(neg_inner) = crate::helpers::as_neg(ctx, right) {
            // a³ + (-b³) = a³ - b³
            let (b, exp_b) = crate::helpers::as_pow(ctx, neg_inner)?;
            if crate::helpers::as_i64(ctx, exp_b)? != 3 {
                return None;
            }
            (a, b, true)
        } else {
            // a³ + b³
            let (b, exp_b) = crate::helpers::as_pow(ctx, right)?;
            if crate::helpers::as_i64(ctx, exp_b)? != 3 {
                return None;
            }
            (a, b, false)
        }
    } else {
        return None;
    };

    // Check if den matches the expected factor
    // Try Sub(a,b) first, then Add(a, Neg(b))
    let den_is_a_minus_b = if let Some((da, db)) = crate::helpers::as_sub(ctx, den) {
        poly_eq(ctx, da, a) && poly_eq(ctx, db, b)
    } else if let Some((left, right)) = crate::helpers::as_add(ctx, den) {
        // Try Add(a, Neg(b)) = a - b
        if let Some(neg_inner) = crate::helpers::as_neg(ctx, right) {
            poly_eq(ctx, left, a) && poly_eq(ctx, neg_inner, b)
        } else if let Some(neg_inner) = crate::helpers::as_neg(ctx, left) {
            poly_eq(ctx, right, a) && poly_eq(ctx, neg_inner, b)
        } else {
            false
        }
    } else {
        false
    };

    let den_is_a_plus_b = if let Some((da, db)) = crate::helpers::as_add(ctx, den) {
        // Skip if it's actually a subtraction (Add with Neg)
        if crate::helpers::as_neg(ctx, da).is_some() || crate::helpers::as_neg(ctx, db).is_some() {
            false // This is actually a subtraction, not a+b
        } else {
            (poly_eq(ctx, da, a) && poly_eq(ctx, db, b))
                || (poly_eq(ctx, da, b) && poly_eq(ctx, db, a))
        }
    } else {
        false
    };

    // For a³ - b³, factor is (a-b); for a³ + b³, factor is (a+b)
    if is_difference && !den_is_a_minus_b {
        return None;
    }
    if !is_difference && !den_is_a_plus_b {
        return None;
    }

    // Build the result: a² ± ab + b²
    let exp_two = ctx.num(2);
    let a_sq = ctx.add(Expr::Pow(a, exp_two));
    let exp_two_b = ctx.num(2);
    let b_sq = ctx.add(Expr::Pow(b, exp_two_b));
    let ab = smart_mul(ctx, a, b);

    let result = if is_difference {
        // a³ - b³ = (a-b)(a² + ab + b²), so result is a² + ab + b²
        let inner = ctx.add(Expr::Add(ab, b_sq));
        ctx.add(Expr::Add(a_sq, inner))
    } else {
        // a³ + b³ = (a+b)(a² - ab + b²), so result is a² - ab + b²
        let neg_ab = ctx.add(Expr::Neg(ab));
        let inner = ctx.add(Expr::Add(neg_ab, b_sq));
        ctx.add(Expr::Add(a_sq, inner))
    };

    // Build factored form for display
    let linear_factor = if is_difference {
        ctx.add(Expr::Sub(a, b))
    } else {
        ctx.add(Expr::Add(a, b))
    };
    let factored_num = smart_mul(ctx, linear_factor, result);

    let desc = if is_difference {
        "Factor and cancel: a³ - b³ = (a-b)(a² + ab + b²)"
    } else {
        "Factor and cancel: a³ + b³ = (a+b)(a² - ab + b²)"
    };

    let rewrite = Rewrite::new(result)
        .desc(desc)
        .local(num, factored_num)
        .requires(ImplicitCondition::NonZero(den));

    Some(rewrite)
}

define_rule!(
    SimplifyFractionRule,
    "Simplify Nested Fraction",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::helpers::prove_nonzero;
        use cas_ast::views::RationalFnView;

        // Capture domain mode for cancellation decisions
        let domain_mode = parent_ctx.domain_mode();

        // Use RationalFnView to detect any fraction form while preserving structure
        let view = RationalFnView::from(ctx, expr)?;
        let (num, den) = (view.num, view.den);

        // EARLY RETURN: Check for didactic perfect-square cancellation
        // (a^2 + 2ab + b^2) / (a+b)^2 → 1 with visible expansion step
        if let Some(rewrite) = try_expand_binomial_square_in_den_for_cancel(ctx, num, den, domain_mode, parent_ctx) {
            return Some(rewrite);
        }

        // EARLY RETURN: Check for difference of squares factorization
        // (a² - b²) / (a+b) → (a-b)(a+b) / (a+b) with visible factorization step
        if let Some(rewrite) = try_difference_of_squares_in_num(ctx, num, den, domain_mode, parent_ctx) {
            return Some(rewrite);
        }

        // EARLY RETURN: Check for perfect square minus recognition
        // (a² - 2ab + b²) / (a-b) → (a-b)²/(a-b) = a-b
        if let Some(rewrite) = try_perfect_square_minus_in_num(ctx, num, den, domain_mode, parent_ctx) {
            return Some(rewrite);
        }

        // EARLY RETURN: Check for sum/difference of cubes factorization
        // (a³ - b³) / (a-b) → (a-b)(a²+ab+b²) / (a-b) = a²+ab+b²
        if let Some(rewrite) = try_sum_diff_of_cubes_in_num(ctx, num, den, domain_mode, parent_ctx) {
            return Some(rewrite);
        }

        // 0. Try multivariate GCD first (Layer 1: monomial + content)
        let vars = collect_variables(ctx, expr);
        if vars.len() > 1 {
            if let Some((new_num, new_den, gcd_expr, layer)) = try_multivar_gcd(ctx, num, den) {
                // DOMAIN GATE: Check if we can cancel by this GCD
                // In Strict mode, only allow if GCD is provably non-zero
                let gcd_proof = prove_nonzero(ctx, gcd_expr);

                // Use can_cancel_factor_with_hint for pedagogical hints in Strict mode
                let key = crate::assumptions::AssumptionKey::nonzero_key(ctx, gcd_expr);
                let decision = crate::domain::can_cancel_factor_with_hint(
                    domain_mode,
                    gcd_proof,
                    key,
                    gcd_expr,
                    "Simplify Nested Fraction",
                );
                if !decision.allow {
                    // Strict mode + Unknown proof: don't simplify (e.g., x/x stays)
                    return None;
                }

                // Build factored form for display
                let factored_num = mul2_raw(ctx, new_num, gcd_expr);
                let factored_den = if let Expr::Number(n) = ctx.get(new_den) {
                    if n.is_one() {
                        gcd_expr
                    } else {
                        mul2_raw(ctx, new_den, gcd_expr)
                    }
                } else {
                    mul2_raw(ctx, new_den, gcd_expr)
                };
                let factored_form = ctx.add(Expr::Div(factored_num, factored_den));

                // Layer tag for description
                let layer_tag = match layer {
                    GcdLayer::Layer1MonomialContent => "Layer 1: monomial+content",
                    GcdLayer::Layer2HeuristicSeeds => "Layer 2: heuristic seeds",
                    GcdLayer::Layer25TensorGrid => "Layer 2.5: tensor grid",
                };

                // If denominator is 1, return just numerator
                if let Expr::Number(n) = ctx.get(new_den) {
                    if n.is_one() {
                        return Some(Rewrite::new(new_num)
                            .desc(format!(
                                "Simplified fraction by GCD: {} [{}]",
                                DisplayExpr { context: ctx, id: gcd_expr },
                                layer_tag
                            ))
                            .local(factored_form, new_num)
                            .assume_all(decision.assumption_events(ctx, gcd_expr)));
                    }
                }

                let result = ctx.add(Expr::Div(new_num, new_den));
                return Some(Rewrite::new(result)
                    .desc(format!(
                        "Simplified fraction by GCD: {} [{}]",
                        DisplayExpr { context: ctx, id: gcd_expr },
                        layer_tag
                    ))
                    .local(factored_form, result)
                    .assume_all(decision.assumption_events(ctx, gcd_expr)));
            }
        }

        // 1. Univariate path: require single variable
        if vars.len() != 1 {
            return None;
        }
        let var = vars.iter().next().unwrap();

        // 2. Convert to Polynomials
        let p_num = Polynomial::from_expr(ctx, num, var).ok()?;
        let p_den = Polynomial::from_expr(ctx, den, var).ok()?;

        if p_den.is_zero() {
            return None;
        }

        // 3. Compute Polynomial GCD (monic)
        let poly_gcd = p_num.gcd(&p_den);

        // 4. Compute Numeric Content GCD
        // Polynomial GCD is monic, so it misses numeric factors like 27x^3 / 9 -> gcd=9
        let content_num = p_num.content();
        let content_den = p_den.content();

        // Helper to compute GCD of two rationals (assuming integers for now)
        let numeric_gcd = gcd_rational(content_num, content_den);

        // 5. Combine
        // full_gcd = poly_gcd * numeric_gcd
        let scalar = Polynomial::new(vec![numeric_gcd.clone()], var.to_string());
        let full_gcd = poly_gcd.mul(&scalar);

        // 6. Check if GCD is non-trivial
        // If degree is 0 and constant is 1, it's trivial.
        if full_gcd.degree() == 0 && full_gcd.leading_coeff().is_one() {
            return None;
        }

        // 7. Divide by GCD (full_gcd is non-zero since we checked it's non-trivial above)
        let (new_num_poly, rem_num) = p_num
            .div_rem(&full_gcd)
            .expect("div_rem should not fail: full_gcd is non-zero");
        let (new_den_poly, rem_den) = p_den
            .div_rem(&full_gcd)
            .expect("div_rem should not fail: full_gcd is non-zero");

        if !rem_num.is_zero() || !rem_den.is_zero() {
            return None;
        }

        let new_num = new_num_poly.to_expr(ctx);
        let new_den = new_den_poly.to_expr(ctx);
        let gcd_expr = full_gcd.to_expr(ctx);

        // DOMAIN GATE: Check if we can cancel by this GCD
        // In Strict mode, only allow if GCD is provably non-zero
        let gcd_proof = prove_nonzero(ctx, gcd_expr);
        let key = crate::assumptions::AssumptionKey::nonzero_key(ctx, gcd_expr);
        let decision = crate::domain::can_cancel_factor_with_hint(
            domain_mode,
            gcd_proof,
            key,
            gcd_expr,
            "Simplify Nested Fraction",
        );
        if !decision.allow {
            // STRICT PARTIAL CANCEL: Try to cancel only numeric content
            // The numeric_gcd is always provably nonzero (it's a rational ≠ 0)
            if !numeric_gcd.is_one() && !numeric_gcd.is_zero() {
                // Divide both polys by numeric content only (safe in Strict)
                let new_num_partial = p_num.div_scalar(&numeric_gcd);
                let new_den_partial = p_den.div_scalar(&numeric_gcd);

                let new_num_expr = new_num_partial.to_expr(ctx);
                let new_den_expr = new_den_partial.to_expr(ctx);
                let result = ctx.add(Expr::Div(new_num_expr, new_den_expr));

                return Some(Rewrite::new(result)
                    .desc(format!("Reduced numeric content by gcd {} (strict-safe)", numeric_gcd))
                    .local(expr, result));
            }
            // No numeric content to cancel, don't simplify
            return None;
        }

        // Build factored form for "Rule:" display: (new_num * gcd) / (new_den * gcd)
        // This shows the factorization step more clearly
        let factored_num = mul2_raw(ctx, new_num, gcd_expr);
        let factored_den = if let Expr::Number(n) = ctx.get(new_den) {
            if n.is_one() {
                gcd_expr // denominator is just the GCD
            } else {
                mul2_raw(ctx, new_den, gcd_expr)
            }
        } else {
            mul2_raw(ctx, new_den, gcd_expr)
        };
        let factored_form = ctx.add(Expr::Div(factored_num, factored_den));

        // If denominator is 1, return numerator
        if let Expr::Number(n) = ctx.get(new_den) {
            if n.is_one() {
                return Some(Rewrite::new(new_num)
                    .desc(format!("Simplified fraction by GCD: {}", DisplayExpr { context: ctx, id: gcd_expr }))
                    .local(factored_form, new_num)
                    .assume_all(decision.assumption_events(ctx, gcd_expr)));
            }
        }

        let result = ctx.add(Expr::Div(new_num, new_den));
        return Some(Rewrite::new(result)
            .desc(format!("Simplified fraction by GCD: {}", DisplayExpr { context: ctx, id: gcd_expr }))
            .local(factored_form, result)
            .assume_all(decision.assumption_events(ctx, gcd_expr)));
    }
);

define_rule!(
    NestedFractionRule,
    "Simplify Complex Fraction",
    |ctx, expr| {
        use cas_ast::views::RationalFnView;

        // Use RationalFnView to detect any fraction form
        let view = RationalFnView::from(ctx, expr)?;
        let (num, den) = (view.num, view.den);

        let num_denoms = collect_denominators(ctx, num);
        let den_denoms = collect_denominators(ctx, den);

        if num_denoms.is_empty() && den_denoms.is_empty() {
            return None;
        }

        // Collect all unique denominators
        let mut all_denoms = Vec::new();
        all_denoms.extend(num_denoms);
        all_denoms.extend(den_denoms);

        if all_denoms.is_empty() {
            return None;
        }

        // Construct the common multiplier (product of all unique denominators)
        // Ideally LCM, but product is safer for now.
        // We need to deduplicate.
        let mut unique_denoms: Vec<ExprId> = Vec::new();
        for d in all_denoms {
            if !unique_denoms.contains(&d) {
                unique_denoms.push(d);
            }
        }

        if unique_denoms.is_empty() {
            return None;
        }

        let (&first, rest) = unique_denoms.split_first().unwrap();
        let multiplier = rest
            .iter()
            .copied()
            .fold(first, |acc, d| mul2_raw(ctx, acc, d));

        // Multiply num and den by multiplier
        let new_num = distribute(ctx, num, multiplier);
        let new_den = distribute(ctx, den, multiplier);

        let new_expr = ctx.add(Expr::Div(new_num, new_den));

        if new_expr == expr {
            return None;
        }

        // Complexity Check: Ensure we actually reduced the number of divisions or total nodes
        // Counting Div nodes is a good heuristic for "nested fraction simplified"
        let count_divs = |id| count_nodes_of_type(ctx, id, "Div");
        let old_divs = count_divs(expr);
        let new_divs = count_divs(new_expr);

        if new_divs >= old_divs {
            return None;
        }

        return Some(
            Rewrite::new(new_expr)
                .desc("Simplify nested fraction")
                .local(expr, new_expr),
        );
    }
);

define_rule!(
    SimplifyMulDivRule,
    "Simplify Multiplication with Division",
    |ctx, expr| {
        use cas_ast::views::FractionParts;

        // Use zero-clone destructuring
        let (l, r) = crate::helpers::as_mul(ctx, expr)?;

        // Use FractionParts to detect any fraction-like structure
        let fp_l = FractionParts::from(&*ctx, l);
        let fp_r = FractionParts::from(&*ctx, r);

        // If neither side has denominators, nothing to do
        if !fp_l.is_fraction() && !fp_r.is_fraction() {
            return None;
        }

        // Check for simple cancellation: (a/b) * b -> a
        // Only for simple cases to avoid over-simplification
        if fp_l.is_fraction() && fp_l.den.len() == 1 && fp_l.den[0].exp == 1 {
            let den_base = fp_l.den[0].base;
            // Check if r equals the denominator
            if crate::ordering::compare_expr(ctx, den_base, r) == std::cmp::Ordering::Equal {
                // Cancel: (a/b) * b -> a
                let result = if fp_l.num.is_empty() {
                    ctx.num(fp_l.sign as i64)
                } else {
                    let num_prod = FractionParts::build_product_static(ctx, &fp_l.num);
                    if fp_l.sign < 0 {
                        ctx.add(Expr::Neg(num_prod))
                    } else {
                        num_prod
                    }
                };
                return Some(Rewrite::new(result).desc("Cancel division: (a/b)*b -> a"));
            }
        }

        // Check for simple cancellation: a * (b/a) -> b
        if fp_r.is_fraction() && fp_r.den.len() == 1 && fp_r.den[0].exp == 1 {
            let den_base = fp_r.den[0].base;
            if crate::ordering::compare_expr(ctx, den_base, l) == std::cmp::Ordering::Equal {
                let result = if fp_r.num.is_empty() {
                    ctx.num(fp_r.sign as i64)
                } else {
                    let num_prod = FractionParts::build_product_static(ctx, &fp_r.num);
                    if fp_r.sign < 0 {
                        ctx.add(Expr::Neg(num_prod))
                    } else {
                        num_prod
                    }
                };
                return Some(Rewrite::new(result).desc("Cancel division: a*(b/a) -> b"));
            }
        }

        // Avoid combining if either side is just a constant (prefer k * (a/b) for CombineLikeTerms)
        if matches!(ctx.get(l), Expr::Number(_) | Expr::Constant(_))
            || matches!(ctx.get(r), Expr::Number(_) | Expr::Constant(_))
        {
            return None;
        }

        // Combine into single fraction: (n1/d1) * (n2/d2) -> (n1*n2)/(d1*d2)
        // Only do this if at least one side is an actual fraction
        if fp_l.is_fraction() || fp_r.is_fraction() {
            // Build combined numerator: products of all num factors
            let mut combined_num = Vec::new();
            combined_num.extend(fp_l.num.iter().cloned());
            combined_num.extend(fp_r.num.iter().cloned());

            // Build combined denominator
            let mut combined_den = Vec::new();
            combined_den.extend(fp_l.den.iter().cloned());
            combined_den.extend(fp_r.den.iter().cloned());

            let combined_sign = (fp_l.sign as i16 * fp_r.sign as i16) as i8;

            let result_fp = FractionParts {
                sign: combined_sign,
                num: combined_num,
                den: combined_den,
            };

            // Build as division for didactic output
            let new_expr = result_fp.build_as_div(ctx);

            // Avoid no-op rewrites
            if new_expr == expr {
                return None;
            }

            return Some(Rewrite::new(new_expr).desc("Combine fractions in multiplication"));
        }
        None
    }
);

/// Check if one denominator divides the other
/// Returns (new_n1, new_n2, common_den, is_divisible)
///
/// For example:
/// - d1=2, d2=2n → d2 = n·d1, so multiply n1 by n: (n1·n, n2, 2n, true)
/// - d1=2n, d2=2 → d1 = n·d2, so multiply n2 by n: (n1, n2·n, 2n, true)
fn check_divisible_denominators(
    ctx: &mut Context,
    n1: ExprId,
    n2: ExprId,
    d1: ExprId,
    d2: ExprId,
) -> (ExprId, ExprId, ExprId, bool) {
    // Try to find if d2 = k * d1 (d1 divides d2)
    if let Some(k) = try_extract_factor(ctx, d2, d1) {
        // d2 = k * d1, so use d2 as common denominator
        // n1/d1 = n1*k/d2
        let new_n1 = mul2_raw(ctx, n1, k);
        return (new_n1, n2, d2, true);
    }

    // Try to find if d1 = k * d2 (d2 divides d1)
    if let Some(k) = try_extract_factor(ctx, d1, d2) {
        // d1 = k * d2, so use d1 as common denominator
        // n2/d2 = n2*k/d1
        let new_n2 = mul2_raw(ctx, n2, k);
        return (n1, new_n2, d1, true);
    }

    // Not divisible
    (n1, n2, d1, false)
}

/// Returns Some(k) if expr = k * factor, None otherwise
fn try_extract_factor(ctx: &Context, expr: ExprId, factor: ExprId) -> Option<ExprId> {
    // Direct equality check (same ExprId)
    if expr == factor {
        return None; // k would be 1, not useful
    }

    // Check if expr is a Mul containing factor
    if let Expr::Mul(l, r) = ctx.get(expr) {
        // Check l * r where one equals factor (using ExprId equality or structural)
        if *l == factor || exprs_equal(ctx, *l, factor) {
            return Some(*r); // expr = factor * r, so k = r
        }
        if *r == factor || exprs_equal(ctx, *r, factor) {
            return Some(*l); // expr = l * factor, so k = l
        }

        // For nested Mul, we'd need a more sophisticated approach
        // For now, only handle simple a*b case where one of them is the factor
    }

    None
}

/// Check if two expressions are structurally equal
fn exprs_equal(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    if a == b {
        return true;
    }
    match (ctx.get(a), ctx.get(b)) {
        (Expr::Number(n1), Expr::Number(n2)) => n1 == n2,
        (Expr::Variable(v1), Expr::Variable(v2)) => v1 == v2,
        (Expr::Constant(c1), Expr::Constant(c2)) => c1 == c2,
        (Expr::Add(l1, r1), Expr::Add(l2, r2)) => {
            exprs_equal(ctx, *l1, *l2) && exprs_equal(ctx, *r1, *r2)
        }
        (Expr::Sub(l1, r1), Expr::Sub(l2, r2)) => {
            exprs_equal(ctx, *l1, *l2) && exprs_equal(ctx, *r1, *r2)
        }
        (Expr::Mul(l1, r1), Expr::Mul(l2, r2)) => {
            exprs_equal(ctx, *l1, *l2) && exprs_equal(ctx, *r1, *r2)
        }
        (Expr::Div(l1, r1), Expr::Div(l2, r2)) => {
            exprs_equal(ctx, *l1, *l2) && exprs_equal(ctx, *r1, *r2)
        }
        (Expr::Pow(l1, r1), Expr::Pow(l2, r2)) => {
            exprs_equal(ctx, *l1, *l2) && exprs_equal(ctx, *r1, *r2)
        }
        (Expr::Neg(e1), Expr::Neg(e2)) => exprs_equal(ctx, *e1, *e2),
        _ => false,
    }
}

define_rule!(AddFractionsRule, "Add Fractions", |ctx, expr| {
    use cas_ast::views::FractionParts;

    // Use zero-clone destructuring
    let (l, r) = crate::helpers::as_add(ctx, expr)?;

    // First try FractionParts (handles direct Div and complex multiplicative patterns)
    let fp_l = FractionParts::from(&*ctx, l);
    let fp_r = FractionParts::from(&*ctx, r);

    let (n1, d1, is_frac1) = fp_l.to_num_den(ctx);
    let (n2, d2, is_frac2) = fp_r.to_num_den(ctx);

    // If FractionParts didn't detect fractions, try extract_as_fraction as fallback
    // This handles Mul(1/n, x) pattern that FractionParts misses
    let (n1, d1, is_frac1) = if !is_frac1 {
        extract_as_fraction(ctx, l)
    } else {
        (n1, d1, is_frac1)
    };
    let (n2, d2, is_frac2) = if !is_frac2 {
        extract_as_fraction(ctx, r)
    } else {
        (n2, d2, is_frac2)
    };

    if !is_frac1 && !is_frac2 {
        return None;
    }

    // Check if d2 = -d1 or d2 == d1 (semantic comparison for cross-tree equality)
    let (n2, d2, opposite_denom, same_denom) = {
        // Use semantic comparison: denominators from different subexpressions may have same value but different ExprIds
        let cmp = crate::ordering::compare_expr(ctx, d1, d2);
        if d1 == d2 || cmp == Ordering::Equal {
            (n2, d2, false, true)
        } else if are_denominators_opposite(ctx, d1, d2) {
            // Convert d2 -> d1, n2 -> -n2
            let minus_n2 = ctx.add(Expr::Neg(n2));
            (minus_n2, d1, true, false)
        } else {
            (n2, d2, false, false)
        }
    };

    // Check if one denominator divides the other (d2 = k * d1 or d1 = k * d2)
    // This allows combining 1/2 + 1/(2n) = n/(2n) + 1/(2n) = (n+1)/(2n)
    let (n1, n2, common_den, divisible_denom) = check_divisible_denominators(ctx, n1, n2, d1, d2);
    let same_denom = same_denom || divisible_denom;

    // Complexity heuristic
    let old_complexity = count_nodes(ctx, expr);

    // a/b + c/d = (ad + bc) / bd
    let ad = mul2_raw(ctx, n1, d2);
    let bc = mul2_raw(ctx, n2, d1);

    let new_num = if opposite_denom || same_denom {
        ctx.add(Expr::Add(n1, n2))
    } else {
        ctx.add(Expr::Add(ad, bc))
    };

    let new_den = if opposite_denom || same_denom {
        common_den
    } else {
        mul2_raw(ctx, d1, d2)
    };

    // Try to simplify common den
    let common_den = if same_denom || opposite_denom {
        common_den
    } else {
        new_den
    };

    let new_expr = ctx.add(Expr::Div(new_num, common_den));
    let new_complexity = count_nodes(ctx, new_expr);

    // If complexity explodes, avoid adding fractions unless denominators are related
    // Exception: if denominators are numbers, always combine: 1/2 + 1/3 = 5/6
    let is_numeric = |e: ExprId| matches!(ctx.get(e), Expr::Number(_));
    if is_numeric(d1) && is_numeric(d2) {
        return Some(Rewrite::new(new_expr).desc("Add numeric fractions"));
    }

    let simplifies = |ctx: &Context, num: ExprId, den: ExprId| -> (bool, bool) {
        // Heuristics to see if new fraction simplifies
        // e.g. cancellation of factors
        // or algebraic simplification
        // Just checking if we reduced node count isn't enough,
        // because un-added fractions might be smaller locally but harder to work with.
        // But we don't want to create massive expressions.

        // Factor cancellation check would be good.
        // Is there a factor F in num and den?

        // Check if num is 0
        if let Expr::Number(n) = ctx.get(num) {
            if n.is_zero() {
                return (true, true);
            }
        }

        // Check negation
        let is_negation = |ctx: &Context, a: ExprId, b: ExprId| -> bool {
            if let Expr::Neg(n) = ctx.get(a) {
                *n == b
            } else if let Expr::Neg(n) = ctx.get(b) {
                *n == a
            } else if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(a), ctx.get(b)) {
                n1 == &-n2
            } else {
                false
            }
        };

        if is_negation(ctx, num, den) {
            return (true, false);
        }

        // Try Polynomial GCD Check
        let vars = collect_variables(ctx, new_num);
        if vars.len() == 1 {
            if let Some(var) = vars.iter().next() {
                if let Ok(p_num) = Polynomial::from_expr(ctx, new_num, var) {
                    if let Ok(p_den) = Polynomial::from_expr(ctx, common_den, var) {
                        if !p_den.is_zero() {
                            let gcd = p_num.gcd(&p_den);
                            if gcd.degree() > 0 || !gcd.leading_coeff().is_one() {
                                // println!(
                                //     "  -> Simplifies via GCD! deg={} lc={}",
                                //     gcd.degree(),
                                //     gcd.leading_coeff()
                                // );
                                let is_proper = p_num.degree() < p_den.degree();
                                return (true, is_proper);
                            }
                        }
                    }
                }
            }
        }

        (false, false)
    };

    let (does_simplify, is_proper) = simplifies(ctx, new_num, common_den);

    // println!(
    //     "AddFractions check: old={} new={} simplify={} limit={}",
    //     old_complexity,
    //     new_complexity,
    //     does_simplify,
    //     (old_complexity * 3) / 2
    // );

    // Allow complexity growth if we found a simplification (GCD)
    // BUT strict check against improper fractions to prevent loops with polynomial division
    // (DividePolynomialsRule splits improper fractions, AddFractions combines them -> loop)
    if opposite_denom
        || same_denom
        || new_complexity <= old_complexity
        || (does_simplify && is_proper && new_complexity < (old_complexity * 2))
    {
        // println!("AddFractions APPLIED: old={} new={} simplify={}", old_complexity, new_complexity, does_simplify);
        return Some(Rewrite::new(new_expr).desc("Add fractions: a/b + c/d -> (ad+bc)/bd"));
    }
    None
});

/// Recognizes ±1 in various AST forms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SignOne {
    PlusOne,
    MinusOne,
}

/// Check if expr is +1 or -1 (in any AST form)
fn sign_one(ctx: &Context, id: ExprId) -> Option<SignOne> {
    use num_rational::BigRational;
    match ctx.get(id) {
        Expr::Number(n) => {
            if n == &BigRational::from_integer((-1).into()) {
                Some(SignOne::MinusOne)
            } else if n.is_one() {
                Some(SignOne::PlusOne)
            } else {
                None
            }
        }
        Expr::Neg(inner) => match ctx.get(*inner) {
            Expr::Number(n) if n.is_one() => Some(SignOne::MinusOne),
            _ => None,
        },
        _ => None,
    }
}

/// Normalize binomial denominator: canonicalize Add(l, Neg(1)) to conceptual Sub(l, 1)
/// Returns (left_term, right_term_normalized, is_add_normalized, right_is_abs_one)
fn split_binomial_den(ctx: &mut Context, den: ExprId) -> Option<(ExprId, ExprId, bool, bool)> {
    let one = ctx.num(1);

    // Use zero-clone helpers
    if let Some((l, r)) = crate::helpers::as_add(ctx, den) {
        return match sign_one(ctx, r) {
            Some(SignOne::PlusOne) => Some((l, one, true, true)), // l + 1
            Some(SignOne::MinusOne) => Some((l, one, false, true)), // l + (-1) → l - 1
            None => Some((l, r, true, false)),                    // l + r
        };
    }
    if let Some((l, r)) = crate::helpers::as_sub(ctx, den) {
        return match sign_one(ctx, r) {
            Some(SignOne::PlusOne) => Some((l, one, false, true)), // l - 1
            Some(SignOne::MinusOne) => Some((l, one, true, true)), // l - (-1) → l + 1
            None => Some((l, r, false, false)),                    // l - r
        };
    }
    None
}

define_rule!(
    RationalizeDenominatorRule,
    "Rationalize Denominator",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        use cas_ast::views::FractionParts;

        // Use FractionParts to detect any fraction structure
        let fp = FractionParts::from(&*ctx, expr);
        if !fp.is_fraction() {
            return None;
        }

        let (num, den, _) = fp.to_num_den(ctx);

        // Use split_binomial_den to normalize the denominator
        // This canonicalizes Add(√x, Neg(1)) to conceptual Sub(√x, 1)
        let (l, r, is_add, r_is_abs_one) = split_binomial_den(ctx, den)?;

        // Check for sqrt roots (degree 2 only - diff squares only works for sqrt)
        // For nth roots (n >= 3), use RationalizeNthRootBinomialRule instead
        let is_sqrt_root = |e: ExprId| -> bool {
            match ctx.get(e) {
                Expr::Pow(_, exp) => {
                    if let Expr::Number(n) = ctx.get(*exp) {
                        // Must be 1/2 for diff squares to work
                        if !n.is_integer() && n.denom() == &num_bigint::BigInt::from(2) {
                            return true;
                        }
                    }
                    false
                }
                Expr::Function(name, _) => name == "sqrt",
                _ => false,
            }
        };

        let l_sqrt = is_sqrt_root(l);
        let r_sqrt = is_sqrt_root(r);

        // Only apply if at least one term is a sqrt (degree 2)
        // For cube roots and higher, skip - they need geometric sum, not conjugate
        if !l_sqrt && !r_sqrt {
            return None;
        }

        // Construct conjugate using normalized terms
        let conjugate = if is_add {
            ctx.add(Expr::Sub(l, r))
        } else {
            ctx.add(Expr::Add(l, r))
        };

        // Multiply num by conjugate
        let new_num = mul2_raw(ctx, num, conjugate);

        // Compute new den = l^2 - r^2
        // Key fix: if r is ±1, use literal 1 instead of Pow(-1, 2)
        let two = ctx.num(2);
        let one = ctx.num(1);
        let l_sq = ctx.add(Expr::Pow(l, two));
        let r_sq = if r_is_abs_one {
            one // 1² = 1, avoid (-1)²
        } else {
            ctx.add(Expr::Pow(r, two))
        };
        let new_den = ctx.add(Expr::Sub(l_sq, r_sq));

        let new_expr = ctx.add(Expr::Div(new_num, new_den));
        return Some(Rewrite::new(new_expr).desc("Rationalize denominator (diff squares)"));
    }
);

// Rationalize binomial denominators with nth roots (n >= 3) using geometric sum.
// For a^(1/n) - r, multiply by sum_{k=0}^{n-1} a^((n-1-k)/n) * r^k
// This gives denominator a - r^n
define_rule!(
    RationalizeNthRootBinomialRule,
    "Rationalize Nth Root Binomial",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        use cas_ast::views::FractionParts;
        use num_traits::ToPrimitive;

        // Use FractionParts to detect fraction
        let fp = FractionParts::from(&*ctx, expr);
        if !fp.is_fraction() {
            return None;
        }

        let (num, den, _) = fp.to_num_den(ctx);

        // Match den = t ± r where t = base^(1/n) with n >= 3
        // Use zero-clone destructuring
        let (t, r, is_sub) = if let Some((l, r)) = crate::helpers::as_add(ctx, den) {
            (l, r, false)
        } else if let Some((l, r)) = crate::helpers::as_sub(ctx, den) {
            (l, r, true)
        } else {
            return None;
        };

        // Check if t is a^(1/n) with n >= 3
        let (base, n) = match ctx.get(t) {
            Expr::Pow(b, exp) => {
                if let Expr::Number(e) = ctx.get(*exp) {
                    // e must be 1/n with n >= 3
                    if e.numer() == &num_bigint::BigInt::from(1) {
                        if let Some(denom) = e.denom().to_u32() {
                            if denom >= 3 {
                                (*b, denom)
                            } else {
                                return None; // degree 2 handled by diff squares
                            }
                        } else {
                            return None;
                        }
                    } else {
                        return None; // numerator must be 1
                    }
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        // Limit n to prevent explosion (max 8 terms)
        if n > 8 {
            return None;
        }

        // Build multiplier M = sum_{k=0}^{n-1} t^(n-1-k) * r^k
        // For t - r: M = t^(n-1) + t^(n-2)*r + ... + r^(n-1)
        // For t + r: need alternating signs for sum formula to work
        //   (t + r)(t^(n-1) - t^(n-2)*r + ... + (-1)^(n-1)*r^(n-1)) = t^n - (-r)^n = t^n - (-1)^n * r^n

        let mut m_terms: Vec<ExprId> = Vec::new();

        for k in 0..n {
            let exp_t = n - 1 - k; // exponent for t
            let exp_r = k; // exponent for r

            // Build t^exp_t = base^((n-1-k)/n)
            let t_part = if exp_t == 0 {
                ctx.num(1)
            } else if exp_t == 1 {
                t
            } else {
                let exp_val = num_rational::BigRational::new(
                    num_bigint::BigInt::from(exp_t),
                    num_bigint::BigInt::from(n),
                );
                let exp_node = ctx.add(Expr::Number(exp_val));
                ctx.add(Expr::Pow(base, exp_node))
            };

            // Build r^exp_r
            let r_part = if exp_r == 0 {
                ctx.num(1)
            } else if exp_r == 1 {
                r
            } else {
                let exp_node = ctx.num(exp_r as i64);
                ctx.add(Expr::Pow(r, exp_node))
            };

            // Combine t_part * r_part
            let mut term = mul2_raw(ctx, t_part, r_part);

            // For t + r case, alternate signs: (-1)^k
            if !is_sub && k % 2 == 1 {
                term = ctx.add(Expr::Neg(term));
            }

            m_terms.push(term);
        }

        // Build M as sum of terms
        let multiplier = build_sum(ctx, &m_terms);

        // New numerator: num * M
        let new_num = mul2_raw(ctx, num, multiplier);

        // New denominator: base - r^n (for t - r) or base - (-1)^n * r^n (for t + r)
        let r_to_n = {
            let exp_node = ctx.num(n as i64);
            ctx.add(Expr::Pow(r, exp_node))
        };

        let new_den = if is_sub {
            // (t - r) * M = t^n - r^n = base - r^n
            ctx.add(Expr::Sub(base, r_to_n))
        } else {
            // (t + r) * M = t^n - (-r)^n = base - (-1)^n * r^n
            if n % 2 == 0 {
                // Even n: base - r^n
                ctx.add(Expr::Sub(base, r_to_n))
            } else {
                // Odd n: base + r^n (since (-r)^n = -r^n)
                ctx.add(Expr::Add(base, r_to_n))
            }
        };

        let new_expr = ctx.add(Expr::Div(new_num, new_den));

        Some(Rewrite::new(new_expr).desc(format!(
            "Rationalize {} root binomial (geometric sum)",
            ordinal(n)
        )))
    }
);

/// Helper to get ordinal string for small numbers
fn ordinal(n: u32) -> &'static str {
    match n {
        3 => "cube",
        4 => "4th",
        5 => "5th",
        6 => "6th",
        7 => "7th",
        8 => "8th",
        _ => "nth",
    }
}

// Cancel nth root binomial factors: (u ± r^n) / (u^(1/n) ± r) = geometric series
// Example: (x + 1) / (x^(1/3) + 1) = x^(2/3) - x^(1/3) + 1
// Uses identity: a^n - b^n = (a-b)(a^(n-1) + a^(n-2)b + ... + b^(n-1))
//            and: a^n + b^n = (a+b)(a^(n-1) - a^(n-2)b + ... ± b^(n-1)) for odd n
define_rule!(
    CancelNthRootBinomialFactorRule,
    "Cancel Nth Root Binomial Factor",
    None,
    PhaseMask::TRANSFORM | PhaseMask::POST,
    |ctx, expr| {
        use cas_ast::views::FractionParts;
        use num_traits::ToPrimitive;

        // Use FractionParts to detect fraction
        let fp = FractionParts::from(&*ctx, expr);
        if !fp.is_fraction() {
            return None;
        }

        let (num, den, _) = fp.to_num_den(ctx);

        // Match den = t ± r where t = u^(1/n)
        // Use zero-clone destructuring
        let (left, right, den_is_add) = if let Some((l, r)) = crate::helpers::as_add(ctx, den) {
            (l, r, true)
        } else if let Some((l, r)) = crate::helpers::as_sub(ctx, den) {
            (l, r, false)
        } else {
            return None;
        };

        // Helper to extract (base, n) from u^(1/n)
        let extract_nth_root = |e: ExprId| -> Option<(ExprId, u32)> {
            if let Expr::Pow(base, exp) = ctx.get(e) {
                if let Expr::Number(ev) = ctx.get(*exp) {
                    if ev.numer() == &num_bigint::BigInt::from(1) {
                        if let Some(denom) = ev.denom().to_u32() {
                            if denom >= 2 {
                                return Some((*base, denom));
                            }
                        }
                    }
                }
            }
            None
        };

        // Try both orderings: left is Pow or right is Pow
        let (t, r, u, n) = if let Some((base, denom)) = extract_nth_root(left) {
            // left = u^(1/n), right = r
            (left, right, base, denom)
        } else if let Some((base, denom)) = extract_nth_root(right) {
            // right = u^(1/n), left = r
            (right, left, base, denom)
        } else {
            return None;
        };

        // r must be a number (start with integer support)
        let r_val = match ctx.get(r) {
            Expr::Number(rv) => rv.clone(),
            _ => return None,
        };

        // Limit n to prevent explosion
        if n > 8 {
            return None;
        }

        // Compute r^n
        let r_to_n = r_val.pow(n as i32);

        // Determine expected numerator based on sign pattern
        // For den = t + r (t = u^(1/n)):
        //   If n is odd: num should be u + r^n (sum of odd powers)
        //   If n is even: num should be u - r^n (?)
        // For den = t - r:
        //   num should be u - r^n (diff of powers)

        let (expected_num_is_add, expected_r_val) = if den_is_add {
            // t + r: for sum pattern a^n + b^n with odd n
            if n % 2 == 1 {
                (true, r_to_n.clone()) // expect u + r^n
            } else {
                return None; // Even n: a^n + b^n doesn't factor nicely over reals
            }
        } else {
            // t - r: for diff pattern a^n - b^n
            (false, r_to_n.clone()) // expect u - r^n
        };

        // Check if numerator matches expected pattern
        // Use zero-clone destructuring
        let (num_left, num_right, num_is_add) =
            if let Some((l, rr)) = crate::helpers::as_add(ctx, num) {
                (l, rr, true)
            } else if let Some((l, rr)) = crate::helpers::as_sub(ctx, num) {
                (l, rr, false)
            } else {
                return None;
            };

        if num_is_add != expected_num_is_add {
            return None;
        }

        // Check if num_left = u (structurally equal)
        // or num_right = u (commutative)
        let (actual_u, actual_r_n) = if crate::ordering::compare_expr(ctx, num_left, u)
            == std::cmp::Ordering::Equal
        {
            (num_left, num_right)
        } else if crate::ordering::compare_expr(ctx, num_right, u) == std::cmp::Ordering::Equal {
            (num_right, num_left)
        } else {
            return None;
        };

        let _ = actual_u; // used for verification above

        // Check if actual_r_n = expected_r_val (as number)
        let actual_r_n_val = match ctx.get(actual_r_n) {
            Expr::Number(v) => v.clone(),
            _ => return None,
        };

        if actual_r_n_val != expected_r_val {
            return None;
        }

        // Match confirmed! Build the quotient as geometric series
        // For t - r: Q = t^(n-1) + t^(n-2)*r + ... + r^(n-1)
        // For t + r: Q = t^(n-1) - t^(n-2)*r + t^(n-3)*r^2 - ... (alternating)

        let mut terms: Vec<ExprId> = Vec::new();

        for k in 0..n {
            let exp_t = n - 1 - k;
            let exp_r = k;

            // Build t^exp_t = u^((n-1-k)/n)
            let t_part = if exp_t == 0 {
                ctx.num(1)
            } else if exp_t == 1 {
                t // u^(1/n)
            } else {
                let exp_val = num_rational::BigRational::new(
                    num_bigint::BigInt::from(exp_t),
                    num_bigint::BigInt::from(n),
                );
                let exp_node = ctx.add(Expr::Number(exp_val));
                ctx.add(Expr::Pow(u, exp_node))
            };

            // Build r^exp_r
            let r_part = if exp_r == 0 {
                ctx.num(1)
            } else {
                let r_pow_k = r_val.pow(exp_r as i32);
                ctx.add(Expr::Number(r_pow_k))
            };

            // Combine t_part * r_part
            let mut term = mul2_raw(ctx, t_part, r_part);

            // For t + r case, alternate signs
            if den_is_add && k % 2 == 1 {
                term = ctx.add(Expr::Neg(term));
            }

            terms.push(term);
        }

        // Build result as sum
        let result = build_sum(ctx, &terms);

        Some(Rewrite::new(result).desc(format!("Cancel {} root binomial factor", ordinal(n))))
    }
);

// Collapse sqrt(A) * B → sqrt(B) when A and B are conjugates with A*B = 1
// Example: sqrt(x + sqrt(x²-1)) * (x - sqrt(x²-1)) → sqrt(x - sqrt(x²-1))
// This works because (p + s)(p - s) = p² - s² = 1 when s = sqrt(p² - 1)
//
// IMPORTANT: This transformation requires `other` (the conjugate being lifted into sqrt)
// to be non-negative (≥ 0), which is an ANALYTIC condition. In Generic mode, this rule
// should be blocked with a hint. In Assume mode, it proceeds with "Assumed: other ≥ 0".
define_rule!(
    SqrtConjugateCollapseRule,
    "Collapse Sqrt Conjugate Product",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Analytic
    ),
    |ctx, expr, parent_ctx| {
        use cas_ast::views::MulChainView;
        use num_rational::BigRational;
        use crate::domain::{can_apply_analytic_with_hint, Proof};
        use crate::semantics::ValueDomain;

        // Guard: Only apply in RealOnly domain (in Complex, sqrt has branch cuts)
        if parent_ctx.value_domain() != ValueDomain::RealOnly {
            return None;
        }

        // Only match Mul expressions
        if !matches!(ctx.get(expr), Expr::Mul(_, _)) {
            return None;
        }

        // Use MulChainView to get all factors
        let mv = MulChainView::from(&*ctx, expr);
        if mv.factors.len() != 2 {
            return None; // Only handle exactly 2 factors for now
        }

        // Helper to check if expr is sqrt(A) and return A
        let unwrap_sqrt = |e: ExprId| -> Option<ExprId> {
            match ctx.get(e) {
                Expr::Pow(base, exp) => {
                    if let Expr::Number(n) = ctx.get(*exp) {
                        let half = BigRational::new(1.into(), 2.into());
                        if n == &half {
                            return Some(*base);
                        }
                    }
                    None
                }
                Expr::Function(name, args) if name == "sqrt" && args.len() == 1 => Some(args[0]),
                _ => None,
            }
        };

        // Try both orderings: factor[0]=sqrt, factor[1]=other or vice versa
        let (sqrt_arg, other) = if let Some(a) = unwrap_sqrt(mv.factors[0]) {
            (a, mv.factors[1])
        } else if let Some(a) = unwrap_sqrt(mv.factors[1]) {
            (a, mv.factors[0])
        } else {
            return None;
        };

        // Extract binomial terms from A (sqrt_arg) and B (other)
        // Handle both Add(p, s) and Add(p, Neg(s)) and Sub(p, s)
        struct SignedBinomial {
            p: ExprId,
            s: ExprId,
            s_positive: bool, // true if p + s, false if p - s
        }

        let parse_signed_binomial = |e: ExprId| -> Option<SignedBinomial> {
            match ctx.get(e) {
                Expr::Add(l, r) => {
                    // Check if r is Neg(something)
                    if let Expr::Neg(inner) = ctx.get(*r) {
                        Some(SignedBinomial {
                            p: *l,
                            s: *inner,
                            s_positive: false,
                        })
                    } else {
                        Some(SignedBinomial {
                            p: *l,
                            s: *r,
                            s_positive: true,
                        })
                    }
                }
                Expr::Sub(l, r) => Some(SignedBinomial {
                    p: *l,
                    s: *r,
                    s_positive: false,
                }),
                _ => None,
            }
        };

        let a_bin = parse_signed_binomial(sqrt_arg)?;
        let b_bin = parse_signed_binomial(other)?;

        // Check if they're conjugates: same p and s, opposite sign for s
        let p_matches =
            crate::ordering::compare_expr(ctx, a_bin.p, b_bin.p) == std::cmp::Ordering::Equal;
        let s_matches =
            crate::ordering::compare_expr(ctx, a_bin.s, b_bin.s) == std::cmp::Ordering::Equal;
        let signs_opposite = a_bin.s_positive != b_bin.s_positive;

        if !p_matches || !s_matches || !signs_opposite {
            return None;
        }

        // Additional guard: s must be a sqrt (so p² - s² = p² - t for some t)
        unwrap_sqrt(a_bin.s)?;

        // ================================================================
        // Analytic Gate: sqrt(other) requires other ≥ 0 (NonNegative)
        // This is an Analytic condition, blocked in Generic, allowed in Assume
        // ================================================================
        let mode = parent_ctx.domain_mode();
        let key = crate::assumptions::AssumptionKey::nonnegative_key(ctx, other);

        // We don't have a proof for this - it's positivity from structure
        // The conjugate product could be positive or negative depending on x
        let proof = Proof::Unknown;

        let decision = can_apply_analytic_with_hint(
            mode,
            proof,
            key,
            other,
            "Collapse Sqrt Conjugate Product",
        );

        if !decision.allow {
            // Blocked: Generic/Strict mode with unproven NonNegative condition
            return None;
        }

        // All checks passed! Return sqrt(B) = sqrt(other)
        let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
        let result = ctx.add(Expr::Pow(other, half));

        // Build assumption event if we assumed NonNegative
        let assumption_events: smallvec::SmallVec<[crate::assumptions::AssumptionEvent; 1]> = if decision.assumption.is_some() {
            smallvec::smallvec![crate::assumptions::AssumptionEvent::nonnegative(ctx, other)]
        } else {
            smallvec::SmallVec::new()
        };

        Some(Rewrite::new(result).desc("Lift conjugate into sqrt").assume_all(assumption_events))
    }
);

/// Collect all additive terms from an expression
/// For a + b + c, returns vec![a, b, c]
fn collect_additive_terms(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut terms = Vec::new();
    collect_terms_recursive(ctx, expr, &mut terms);
    terms
}

fn collect_terms_recursive(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            collect_terms_recursive(ctx, *l, terms);
            collect_terms_recursive(ctx, *r, terms);
        }
        _ => {
            // It's a leaf term (including Sub which we treat as single term)
            terms.push(expr);
        }
    }
}

/// Check if an expression contains an irrational (root)
fn contains_irrational(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(_, exp) => {
            if let Expr::Number(n) = ctx.get(*exp) {
                !n.is_integer() // Fractional exponent = root
            } else {
                false
            }
        }
        Expr::Function(name, _) => name == "sqrt",
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            contains_irrational(ctx, *l) || contains_irrational(ctx, *r)
        }
        Expr::Neg(e) => contains_irrational(ctx, *e),
        _ => false,
    }
}

/// Build a sum from a list of terms
fn build_sum(ctx: &mut Context, terms: &[ExprId]) -> ExprId {
    if terms.is_empty() {
        return ctx.num(0);
    }
    let mut result = terms[0];
    for &term in terms.iter().skip(1) {
        result = ctx.add(Expr::Add(result, term));
    }
    result
}

define_rule!(
    GeneralizedRationalizationRule,
    "Generalized Rationalization",
    |ctx, expr| {
        use cas_ast::views::FractionParts;

        // Use FractionParts to detect any fraction structure
        let fp = FractionParts::from(&*ctx, expr);
        if !fp.is_fraction() {
            return None;
        }

        let (num, den, _) = fp.to_num_den(ctx);

        let terms = collect_additive_terms(ctx, den);

        // Only apply to 3+ terms (binary case handled by RationalizeDenominatorRule)
        if terms.len() < 3 {
            return None;
        }

        // Check if any term contains a root
        let has_roots = terms.iter().any(|&t| contains_irrational(ctx, t));
        if !has_roots {
            return None;
        }

        // Strategy: Group as (first n-1 terms) + last_term
        // Then apply conjugate: multiply by (group - last) / (group - last)
        let last_term = terms[terms.len() - 1];
        let group_terms = &terms[..terms.len() - 1];
        let group = build_sum(ctx, group_terms);

        // Conjugate: (group - last_term)
        let conjugate = ctx.add(Expr::Sub(group, last_term));

        // New numerator: num * conjugate
        let new_num = mul2_raw(ctx, num, conjugate);

        // New denominator: group^2 - last_term^2 (difference of squares)
        let two = ctx.num(2);
        let group_sq = ctx.add(Expr::Pow(group, two));
        let last_sq = ctx.add(Expr::Pow(last_term, two));
        let new_den = ctx.add(Expr::Sub(group_sq, last_sq));

        // Post-pass: expand the denominator to simplify (1+√2)² → 3+2√2
        // This ensures rationalization results don't leave unexpanded pow-sums
        let new_den_expanded = crate::expand::expand(ctx, new_den);

        let new_expr = ctx.add(Expr::Div(new_num, new_den_expanded));

        Some(Rewrite::new(new_expr).desc(format!(
            "Rationalize: group {} terms and multiply by conjugate",
            terms.len()
        )))
    }
);

/// Collect all multiplicative factors from an expression
fn collect_mul_factors(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut factors = Vec::new();
    collect_factors_recursive(ctx, expr, &mut factors);
    factors
}

fn collect_factors_recursive(ctx: &Context, expr: ExprId, factors: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            collect_factors_recursive(ctx, *l, factors);
            collect_factors_recursive(ctx, *r, factors);
        }
        _ => {
            factors.push(expr);
        }
    }
}

/// Extract root from expression: sqrt(n) or n^(1/k)
/// Returns (radicand, index) where expr = radicand^(1/index)
fn extract_root_base(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    // Check for sqrt(n) function - use zero-clone helper
    if let Some(arg) = crate::helpers::as_fn1(ctx, expr, "sqrt") {
        // sqrt(n) = n^(1/2), return (n, 2)
        let two = ctx.num(2);
        return Some((arg, two));
    }

    // Check for Pow(base, exp) - use zero-clone helper
    if let Some((base, exp)) = crate::helpers::as_pow(ctx, expr) {
        // Check if exp is a Number like 1/k
        if let Some(n) = crate::helpers::as_number(ctx, exp) {
            if !n.is_integer() && n.numer().is_one() {
                // n^(1/k) - return (n, k)
                let k_expr = ctx.add(Expr::Number(num_rational::BigRational::from_integer(
                    n.denom().clone(),
                )));
                return Some((base, k_expr));
            }
        }
        // Check if exp is Div(1, k)
        if let Some((num_exp, den_exp)) = crate::helpers::as_div(ctx, exp) {
            if let Some(n) = crate::helpers::as_number(ctx, num_exp) {
                if n.is_one() {
                    return Some((base, den_exp));
                }
            }
        }
    }
    None
}

define_rule!(
    RationalizeProductDenominatorRule,
    "Rationalize Product Denominator",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        use cas_ast::views::FractionParts;

        // Handle fractions with product denominators containing roots
        let fp = FractionParts::from(&*ctx, expr);
        if !fp.is_fraction() {
            return None;
        }

        let (num, den, _) = fp.to_num_den(ctx);

        let factors = collect_mul_factors(ctx, den);

        // Find a root factor
        let mut root_factor = None;
        let mut non_root_factors = Vec::new();

        for &factor in &factors {
            if extract_root_base(ctx, factor).is_some() && root_factor.is_none() {
                root_factor = Some(factor);
            } else {
                non_root_factors.push(factor);
            }
        }

        let root = root_factor?;

        // Don't apply if denominator is ONLY a root (handled elsewhere or simpler)
        if non_root_factors.is_empty() {
            // Just sqrt(n) in denominator - still rationalize
            if let Some((radicand, _index)) = extract_root_base(ctx, root) {
                // Check if radicand is a binomial (Add or Sub) - these can cause infinite loops
                // when both numerator and denominator have binomial radicals like sqrt(x+y)/sqrt(x-y)
                let is_binomial_radical =
                    matches!(ctx.get(radicand), Expr::Add(_, _) | Expr::Sub(_, _));
                if is_binomial_radical && contains_irrational(ctx, num) {
                    return None;
                }

                // Don't rationalize if radicand is a simple number - power rules handle these better
                // e.g., sqrt(2) / 2^(1/3) should simplify via power combination to 2^(1/6)
                if matches!(ctx.get(radicand), Expr::Number(_)) {
                    return None;
                }

                // 1/sqrt(n) -> sqrt(n)/n
                let new_num = mul2_raw(ctx, num, root);
                let new_den = radicand;
                let new_expr = ctx.add(Expr::Div(new_num, new_den));
                return Some(Rewrite::new(new_expr).desc("Rationalize: multiply by √n/√n"));
            }
            return None;
        }

        // Don't apply if radicand is a simple number - power rules can handle these better
        // e.g., 2*sqrt(2) / (2*2^(1/3)) should simplify via power combination, not rationalization
        if let Some((radicand, _index)) = extract_root_base(ctx, root) {
            if matches!(ctx.get(radicand), Expr::Number(_)) {
                return None;
            }
        }

        // We have: num / (other_factors * root) where root = radicand^(1/index)
        // To rationalize, we need to multiply by radicand^((index-1)/index) / radicand^((index-1)/index)
        // This gives: root * radicand^((index-1)/index) = radicand^(1/index + (index-1)/index) = radicand^1 = radicand
        //
        // For sqrt (index=2): multiply by radicand^(1/2) to get radicand^(1/2 + 1/2) = radicand
        // For cbrt (index=3): multiply by radicand^(2/3) to get radicand^(1/3 + 2/3) = radicand

        if let Some((radicand, index)) = extract_root_base(ctx, root) {
            // Compute the conjugate exponent: (index - 1) / index
            // For square root (index=2): conjugate = 1/2, so conjugate_power = radicand^(1/2) = sqrt(radicand)
            // For cube root (index=3): conjugate = 2/3, so conjugate_power = radicand^(2/3)

            // Get index as integer if possible
            let index_val = if let Expr::Number(n) = ctx.get(index) {
                if n.is_integer() {
                    Some(n.to_integer())
                } else {
                    None
                }
            } else {
                None
            };

            // Only handle integer indices for now
            let index_int = index_val?;
            if index_int <= num_bigint::BigInt::from(1) {
                return None; // Not a valid root index
            }

            // Build conjugate exponent (index - 1) / index
            let one = num_bigint::BigInt::from(1);
            let conjugate_num = &index_int - &one;
            let conjugate_exp = num_rational::BigRational::new(conjugate_num, index_int);
            let conjugate_exp_id = ctx.add(Expr::Number(conjugate_exp));

            // conjugate_power = radicand^((index-1)/index)
            let conjugate_power = ctx.add(Expr::Pow(radicand, conjugate_exp_id));

            // New numerator: num * conjugate_power
            let new_num = mul2_raw(ctx, num, conjugate_power);

            // Build new denominator: other_factors * radicand (since root * conjugate_power = radicand)
            let mut new_den = radicand;
            for &factor in &non_root_factors {
                new_den = mul2_raw(ctx, new_den, factor);
            }

            let new_expr = ctx.add(Expr::Div(new_num, new_den));
            return Some(Rewrite::new(new_expr).desc("Rationalize product denominator"));
        }

        None
    }
);

define_rule!(
    CancelCommonFactorsRule,
    "Cancel Common Factors",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::helpers::prove_nonzero;

        // Capture domain mode once at start
        let domain_mode = parent_ctx.domain_mode();

        // CLONE_OK: Multi-branch match on Div/Pow/Mul requires owned Expr
        let expr_data = ctx.get(expr).clone();

        // Helper to collect factors
        fn collect_factors(ctx: &Context, e: ExprId) -> Vec<ExprId> {
            let mut factors = Vec::new();
            let mut stack = vec![e];
            while let Some(curr) = stack.pop() {
                if let Expr::Mul(l, r) = ctx.get(curr) {
                    stack.push(*r);
                    stack.push(*l);
                } else {
                    factors.push(curr);
                }
            }
            factors
        }

        let (mut num_factors, mut den_factors) = match expr_data {
            Expr::Div(n, d) => (collect_factors(ctx, n), collect_factors(ctx, d)),
            Expr::Pow(b, e) => {
                if let Expr::Number(n) = ctx.get(e) {
                    if n.is_integer() && *n == num_rational::BigRational::from_integer((-1).into())
                    {
                        (vec![ctx.num(1)], collect_factors(ctx, b))
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            Expr::Mul(_, _) => {
                let factors = collect_factors(ctx, expr);
                let mut nf = Vec::new();
                let mut df = Vec::new();
                for f in factors {
                    if let Expr::Pow(b, e) = ctx.get(f) {
                        if let Expr::Number(n) = ctx.get(*e) {
                            if n.is_integer()
                                && *n == num_rational::BigRational::from_integer((-1).into())
                            {
                                df.extend(collect_factors(ctx, *b));
                                continue;
                            }
                        }
                    }
                    nf.push(f);
                }
                if df.is_empty() {
                    return None;
                }
                (nf, df)
            }
            _ => return None,
        };
        // NOTE: Pythagorean identity simplification (k - k*sin² → k*cos²) has been
        // extracted to TrigPythagoreanSimplifyRule for pedagogical clarity.
        // CancelCommonFactorsRule now does pure factor cancellation.

        let mut changed = false;
        let mut assumption_events: smallvec::SmallVec<[crate::assumptions::AssumptionEvent; 1]> = Default::default();
        let mut i = 0;
        while i < num_factors.len() {
            let nf = num_factors[i];
            // println!("Processing num factor: {:?}", ctx.get(nf));
            let mut found = false;
            for j in 0..den_factors.len() {
                let df = den_factors[j];

                // Check exact match
                if crate::ordering::compare_expr(ctx, nf, df) == std::cmp::Ordering::Equal {
                    // DOMAIN GATE: use canonical helper
                    let proof = prove_nonzero(ctx, nf);
                    let key = crate::assumptions::AssumptionKey::nonzero_key(ctx, nf);
                    let decision = crate::domain::can_cancel_factor_with_hint(
                        domain_mode,
                        proof,
                        key,
                        nf,
                        "Cancel Common Factors",
                    );
                    if !decision.allow {
                        continue; // Skip this pair in strict mode
                    }
                    // Record assumption if made
                    if decision.assumption.is_some() {
                        assumption_events.push(
                            crate::assumptions::AssumptionEvent::nonzero(ctx, nf)
                        );
                    }
                    den_factors.remove(j);
                    found = true;
                    changed = true;
                    break;
                }

                // Check power cancellation: nf = x^n, df = x^m
                // Case 1: nf = base^n, df = base. (integer n only to preserve rationalized forms)
                let nf_pow = if let Expr::Pow(b, e) = ctx.get(nf) {
                    Some((*b, *e))
                } else {
                    None
                };
                if let Some((b, e)) = nf_pow {
                    if crate::ordering::compare_expr(ctx, b, df) == std::cmp::Ordering::Equal {
                        if let Expr::Number(n) = ctx.get(e) {
                            // Guard: only integer exponents - skip fractional to preserve rationalized forms
                            // E.g., sqrt(x)/x should NOT become x^(-1/2) as this undoes rationalization
                            if !n.is_integer() {
                                // Skip this pair, continue to next
                            } else {
                                let new_exp = n - num_rational::BigRational::one();
                                if new_exp.is_zero() {
                                    // x^1 / x = 1, remove both factors
                                    // DOMAIN GATE: check base is provably non-zero
                                    let proof = prove_nonzero(ctx, b);
                                    let key =
                                        crate::assumptions::AssumptionKey::nonzero_key(ctx, b);
                                    let decision = crate::domain::can_cancel_factor_with_hint(
                                        domain_mode,
                                        proof,
                                        key,
                                        b,
                                        "Cancel Common Factors",
                                    );
                                    if !decision.allow {
                                        continue; // Skip in strict mode
                                    }
                                    // Record assumption if made
                                    if decision.assumption.is_some() {
                                        assumption_events.push(
                                            crate::assumptions::AssumptionEvent::nonzero(ctx, b)
                                        );
                                    }
                                    den_factors.remove(j);
                                    found = true; // Remove num factor too
                                    changed = true;
                                    break;
                                }
                                let new_term = if new_exp.is_one() {
                                    b
                                } else {
                                    let exp_node = ctx.add(Expr::Number(new_exp));
                                    ctx.add(Expr::Pow(b, exp_node))
                                };
                                num_factors[i] = new_term;
                                den_factors.remove(j);
                                found = false; // Modified num factor
                                changed = true;
                                break;
                            }
                        }
                    }
                }

                // Case 2: nf = base, df = base^m. (integer m only to preserve rationalized forms)
                let df_pow = if let Expr::Pow(b, e) = ctx.get(df) {
                    Some((*b, *e))
                } else {
                    None
                };
                if let Some((b, e)) = df_pow {
                    if crate::ordering::compare_expr(ctx, nf, b) == std::cmp::Ordering::Equal {
                        if let Expr::Number(m) = ctx.get(e) {
                            // Guard: only integer exponents - skip fractional to preserve rationalized forms
                            // E.g., x/sqrt(x) with fractional exp handled by QuotientOfPowersRule
                            if !m.is_integer() {
                                // Skip this pair, continue to next
                            } else {
                                let new_exp = m - num_rational::BigRational::one();
                                if new_exp.is_zero() {
                                    // x / x^1 = 1, remove both factors
                                    // DOMAIN GATE: check base is provably non-zero
                                    let proof = prove_nonzero(ctx, b);
                                    let key =
                                        crate::assumptions::AssumptionKey::nonzero_key(ctx, b);
                                    let decision = crate::domain::can_cancel_factor_with_hint(
                                        domain_mode,
                                        proof,
                                        key,
                                        b,
                                        "Cancel Common Factors",
                                    );
                                    if !decision.allow {
                                        continue; // Skip in strict mode
                                    }
                                    den_factors.remove(j);
                                    found = true; // Remove num factor too
                                    changed = true;
                                    break;
                                }
                                let new_term = if new_exp.is_one() {
                                    b
                                } else {
                                    let exp_node = ctx.add(Expr::Number(new_exp));
                                    ctx.add(Expr::Pow(b, exp_node))
                                };
                                den_factors[j] = new_term;
                                found = true; // Remove num factor
                                changed = true;
                                break;
                            }
                        }
                    }
                }

                // Case 3: nf = base^n, df = base^m (integer exponents only)
                // Fractional exponents are handled atomically by QuotientOfPowersRule
                if let Some((b_n, e_n)) = nf_pow {
                    if let Some((b_d, e_d)) = df_pow {
                        if crate::ordering::compare_expr(ctx, b_n, b_d) == std::cmp::Ordering::Equal
                        {
                            if let (Expr::Number(n), Expr::Number(m)) = (ctx.get(e_n), ctx.get(e_d))
                            {
                                // Skip fractional exponents - QuotientOfPowersRule handles them
                                if !n.is_integer() || !m.is_integer() {
                                    // Continue to next factor, don't process this pair
                                } else if n > m {
                                    let new_exp = n - m;
                                    let new_term = if new_exp.is_one() {
                                        b_n
                                    } else {
                                        let exp_node = ctx.add(Expr::Number(new_exp));
                                        ctx.add(Expr::Pow(b_n, exp_node))
                                    };
                                    num_factors[i] = new_term;
                                    den_factors.remove(j);
                                    found = false;
                                    changed = true;
                                    break;
                                } else if m > n {
                                    let new_exp = m - n;
                                    let new_term = if new_exp.is_one() {
                                        b_d
                                    } else {
                                        let exp_node = ctx.add(Expr::Number(new_exp));
                                        ctx.add(Expr::Pow(b_d, exp_node))
                                    };
                                    den_factors[j] = new_term;
                                    found = true;
                                    changed = true;
                                    break;
                                } else {
                                    // x^n / x^n (n == m), remove both factors
                                    // DOMAIN GATE: check base is provably non-zero
                                    let proof = prove_nonzero(ctx, b_n);
                                    let key =
                                        crate::assumptions::AssumptionKey::nonzero_key(ctx, b_n);
                                    let decision = crate::domain::can_cancel_factor_with_hint(
                                        domain_mode,
                                        proof,
                                        key,
                                        b_n,
                                        "Cancel Common Factors",
                                    );
                                    if !decision.allow {
                                        continue; // Skip in strict mode
                                    }
                                    // Record assumption if made
                                    if decision.assumption.is_some() {
                                        assumption_events.push(
                                            crate::assumptions::AssumptionEvent::nonzero(ctx, b_n)
                                        );
                                    }
                                    den_factors.remove(j);
                                    found = true;
                                    changed = true;
                                    break;
                                } // end else for integer exponents
                            }
                        }
                    }
                }
            }
            if found {
                num_factors.remove(i);
            } else {
                i += 1;
            }
        }

        if changed {
            // Reconstruct
            let new_num = if num_factors.is_empty() {
                ctx.num(1)
            } else {
                let mut n = num_factors[0];
                for &f in num_factors.iter().skip(1) {
                    n = mul2_raw(ctx, n, f);
                }
                n
            };
            let new_den = if den_factors.is_empty() {
                ctx.num(1)
            } else {
                let mut d = den_factors[0];
                for &f in den_factors.iter().skip(1) {
                    d = mul2_raw(ctx, d, f);
                }
                d
            };

            let new_expr = ctx.add(Expr::Div(new_num, new_den));
            return Some(Rewrite::new(new_expr)
                .desc("Cancel common factors")
                .local(expr, new_expr)
                .assume_all(assumption_events));
        }

        None
    }
);

// Atomized rule for quotient of powers: a^n / a^m = a^(n-m)
// This is separated from CancelCommonFactorsRule for pedagogical clarity
define_rule!(
    QuotientOfPowersRule,
    "Quotient of Powers",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::helpers::prove_nonzero;
        use cas_ast::views::FractionParts;

        // Capture domain mode for cancellation decisions
        let domain_mode = parent_ctx.domain_mode();

        let fp = FractionParts::from(&*ctx, expr);
        if !fp.is_fraction() {
            return None;
        }

        let (num, den, _) = fp.to_num_den(ctx);
        // CLONE_OK: Multi-branch match on Add/Mul/Pow patterns
        let num_data = ctx.get(num).clone();
        // CLONE_OK: Denominator inspection for sign normalization
        let den_data = ctx.get(den).clone();

        // Case 1: a^n / a^m where both are Pow
        if let (Expr::Pow(b_n, e_n), Expr::Pow(b_d, e_d)) = (&num_data, &den_data) {
            // Check same base
            if crate::ordering::compare_expr(ctx, *b_n, *b_d) == std::cmp::Ordering::Equal {
                // Check if exponents are numeric (so we can subtract)
                if let (Expr::Number(n), Expr::Number(m)) = (ctx.get(*e_n), ctx.get(*e_d)) {
                    // Only handle fractional exponents here - integer case is in CancelCommonFactors
                    if n.is_integer() && m.is_integer() {
                        return None;
                    }

                    let diff = n - m;
                    if diff.is_zero() {
                        // a^n / a^n = 1
                        // DOMAIN GATE: check if base is provably non-zero
                        let proof = prove_nonzero(ctx, *b_n);
                        let key = crate::assumptions::AssumptionKey::nonzero_key(ctx, *b_n);
                        let decision = crate::domain::can_cancel_factor_with_hint(
                            domain_mode,
                            proof,
                            key,
                            *b_n,
                            "Quotient of Powers",
                        );
                        if !decision.allow {
                            return None; // In Strict mode, don't cancel unknown factors
                        }
                        return Some(Rewrite::new(ctx.num(1)).desc("a^n / a^n = 1"));
                    } else if diff.is_one() {
                        // Result is just the base
                        return Some(Rewrite::new(*b_n).desc("a^n / a^m = a^(n-m)"));
                    } else {
                        // Guard: Don't produce negative fractional exponents (anti-pattern for rationalization)
                        // E.g., sqrt(x)/x should NOT become x^(-1/2) as it undoes rationalization
                        if diff < num_rational::BigRational::zero() && !diff.is_integer() {
                            return None;
                        }
                        let new_exp = ctx.add(Expr::Number(diff));
                        let new_expr = ctx.add(Expr::Pow(*b_n, new_exp));
                        return Some(Rewrite::new(new_expr).desc("a^n / a^m = a^(n-m)"));
                    }
                }
            }
        }

        // Case 2: a^n / a (denominator has implicit exponent 1)
        if let Expr::Pow(b_n, e_n) = &num_data {
            if crate::ordering::compare_expr(ctx, *b_n, den) == std::cmp::Ordering::Equal {
                if let Expr::Number(n) = ctx.get(*e_n) {
                    if !n.is_integer() {
                        let new_exp_val = n - num_rational::BigRational::one();
                        // Guard: Don't produce negative fractional exponents
                        if new_exp_val < num_rational::BigRational::zero() {
                            return None;
                        }
                        if new_exp_val.is_one() {
                            return Some(Rewrite::new(*b_n).desc("a^n / a = a^(n-1)"));
                        } else {
                            let new_exp = ctx.add(Expr::Number(new_exp_val));
                            let new_expr = ctx.add(Expr::Pow(*b_n, new_exp));
                            return Some(Rewrite::new(new_expr).desc("a^n / a = a^(n-1)"));
                        }
                    }
                }
            }
        }

        // Case 3: a / a^m (numerator has implicit exponent 1)
        if let Expr::Pow(b_d, e_d) = &den_data {
            if crate::ordering::compare_expr(ctx, num, *b_d) == std::cmp::Ordering::Equal {
                if let Expr::Number(m) = ctx.get(*e_d) {
                    if !m.is_integer() {
                        let new_exp_val = num_rational::BigRational::one() - m;
                        // Guard: Don't produce negative fractional exponents
                        // This would undo rationalization: sqrt(x)/x should stay as-is, NOT become x^(-1/2)
                        if new_exp_val < num_rational::BigRational::zero() {
                            return None;
                        }
                        let new_exp = ctx.add(Expr::Number(new_exp_val));
                        let new_expr = ctx.add(Expr::Pow(num, new_exp));
                        return Some(Rewrite::new(new_expr).desc("a / a^m = a^(1-m)"));
                    }
                }
            }
        }

        None
    }
);

define_rule!(
    PullConstantFromFractionRule,
    "Pull Constant From Fraction",
    |ctx, expr| {
        // NOTE: Keep simple Div detection to avoid infinite loop with Combine Like Terms
        // when detecting Neg(Div(...)) as a fraction
        let (n, d) = if let Expr::Div(n, d) = ctx.get(expr) {
            (*n, *d)
        } else {
            return None;
        };

        // CLONE_OK: Multi-branch match on Add/Mul/Number patterns
        let num_data = ctx.get(n).clone();
        if let Expr::Mul(l, r) = num_data {
            // Check if l or r is a number/constant
            let l_is_const = matches!(ctx.get(l), Expr::Number(_) | Expr::Constant(_));
            let r_is_const = matches!(ctx.get(r), Expr::Number(_) | Expr::Constant(_));

            if l_is_const {
                // (c * x) / y -> c * (x / y)
                let div = ctx.add(Expr::Div(r, d));
                let new_expr = mul2_raw(ctx, l, div);
                return Some(Rewrite::new(new_expr).desc("Pull constant from numerator"));
            } else if r_is_const {
                // (x * c) / y -> c * (x / y)
                let div = ctx.add(Expr::Div(l, d));
                let new_expr = mul2_raw(ctx, r, div);
                return Some(Rewrite::new(new_expr).desc("Pull constant from numerator"));
            }
        }
        // Also handle Neg: (-x) / y -> -1 * (x / y)
        if let Expr::Neg(inner) = num_data {
            let minus_one = ctx.num(-1);
            let div = ctx.add(Expr::Div(inner, d));
            let new_expr = mul2_raw(ctx, minus_one, div);
            return Some(Rewrite::new(new_expr).desc("Pull negation from numerator"));
        }
        None
    }
);

define_rule!(
    FactorBasedLCDRule,
    "Factor-Based LCD",
    Some(vec!["Add"]),
    |ctx, expr| {
        use crate::ordering::compare_expr;
        use std::cmp::Ordering;

        // Normalize a binomial to canonical form: (a-b) where a < b alphabetically
        // Returns (canonical_expr, sign_flip) where sign_flip is true if we negated
        let normalize_binomial = |ctx: &mut Context, e: ExprId| -> (ExprId, bool) {
            match ctx.get(e).clone() {
                Expr::Add(l, r) => {
                    if let Expr::Neg(inner) = ctx.get(r).clone() {
                        // Form: l + (-inner) = l - inner
                        if compare_expr(ctx, l, inner) == Ordering::Less {
                            (e, false) // Already canonical
                        } else {
                            // Create: -(inner - l) = (l - inner) negated
                            let neg_l = ctx.add(Expr::Neg(l));
                            let canonical = ctx.add(Expr::Add(inner, neg_l));
                            (canonical, true)
                        }
                    } else {
                        (e, false) // Not a subtraction pattern
                    }
                }
                Expr::Sub(l, r) => {
                    if compare_expr(ctx, l, r) == Ordering::Less {
                        (e, false)
                    } else {
                        let canonical = ctx.add(Expr::Sub(r, l));
                        (canonical, true)
                    }
                }
                _ => (e, false),
            }
        };

        // Extract factors from a product expression
        let get_factors = |ctx: &Context, e: ExprId| -> Vec<ExprId> {
            let mut factors = Vec::new();
            let mut stack = vec![e];
            while let Some(curr) = stack.pop() {
                match ctx.get(curr) {
                    Expr::Mul(l, r) => {
                        stack.push(*l);
                        stack.push(*r);
                    }
                    _ => factors.push(curr),
                }
            }
            factors
        };

        // Check if expression is a binomial (Add with Neg or Sub)
        let is_binomial = |ctx: &Context, e: ExprId| -> bool {
            match ctx.get(e) {
                Expr::Add(_, r) => matches!(ctx.get(*r), Expr::Neg(_)),
                Expr::Sub(_, _) => true,
                _ => false,
            }
        };

        // Check if two expressions are equal (by compare_expr)
        let expr_eq = |ctx: &Context, a: ExprId, b: ExprId| -> bool {
            compare_expr(ctx, a, b) == Ordering::Equal
        };

        // ===== Main Logic =====

        // Collect all terms from the Add tree
        let mut terms = Vec::new();
        let mut stack = vec![expr];
        while let Some(curr) = stack.pop() {
            match ctx.get(curr) {
                Expr::Add(l, r) => {
                    stack.push(*l);
                    stack.push(*r);
                }
                _ => terms.push(curr),
            }
        }

        // Need at least 3 fractions - AddFractionsRule handles 2-fraction cases
        if terms.len() < 3 {
            return None;
        }

        // Extract (numerator, denominator) from each fraction
        let mut fractions: Vec<(ExprId, ExprId)> = Vec::new();
        for term in &terms {
            match ctx.get(*term) {
                Expr::Div(num, den) => fractions.push((*num, *den)),
                _ => return None, // Not all terms are fractions
            }
        }

        // For each denominator, extract and normalize binomial factors
        // Store: Vec<(canonical_factor, sign_flip)> for each fraction
        let mut all_factor_sets: Vec<Vec<(ExprId, bool)>> = Vec::new();

        for (_, den) in &fractions {
            let raw_factors = get_factors(ctx, *den);

            // All factors must be binomials for this rule to apply
            let mut normalized = Vec::new();
            for f in raw_factors {
                if !is_binomial(ctx, f) {
                    return None;
                }
                let (canonical, flipped) = normalize_binomial(ctx, f);
                normalized.push((canonical, flipped));
            }

            if normalized.is_empty() {
                return None;
            }

            all_factor_sets.push(normalized);
        }

        // Collect all unique canonical factors (the LCD factors)
        let mut unique_factors: Vec<ExprId> = Vec::new();
        for factor_set in &all_factor_sets {
            for (canonical, _) in factor_set {
                let exists = unique_factors.iter().any(|u| expr_eq(ctx, *u, *canonical));
                if !exists {
                    unique_factors.push(*canonical);
                }
            }
        }

        // Skip if all fractions already have the same denominator
        let all_same = all_factor_sets.iter().all(|fs| {
            fs.len() == unique_factors.len()
                && unique_factors
                    .iter()
                    .all(|uf| fs.iter().any(|(cf, _)| expr_eq(ctx, *cf, *uf)))
        });
        if all_same && fractions.len() == terms.len() {
            // If fractions share all factors (same LCD), AddFractionsRule handles it
            return None;
        }

        // Build LCD as product of all unique factors
        let lcd = if unique_factors.len() == 1 {
            unique_factors[0]
        } else {
            let (&first, rest) = unique_factors.split_first().unwrap();
            rest.iter()
                .copied()
                .fold(first, |acc, f| mul2_raw(ctx, acc, f))
        };

        // For each fraction, compute numerator contribution
        let mut numerator_terms: Vec<ExprId> = Vec::new();

        for (i, (num, _den)) in fractions.iter().enumerate() {
            let factor_set = &all_factor_sets[i];

            // Compute overall sign from normalization
            let sign_flips: usize = factor_set.iter().filter(|(_, f)| *f).count();
            let is_negative = sign_flips % 2 == 1;

            // Find missing factors (in unique but not in this denominator)
            let mut missing: Vec<ExprId> = Vec::new();
            for uf in &unique_factors {
                let present = factor_set.iter().any(|(cf, _)| expr_eq(ctx, *cf, *uf));
                if !present {
                    missing.push(*uf);
                }
            }

            // Multiply numerator by all missing factors
            let mut contribution = *num;
            for mf in missing {
                contribution = mul2_raw(ctx, contribution, mf);
            }

            // Apply sign
            if is_negative {
                contribution = ctx.add(Expr::Neg(contribution));
            }

            numerator_terms.push(contribution);
        }

        // Sum all numerator contributions
        let total_num = if numerator_terms.len() == 1 {
            numerator_terms[0]
        } else {
            let (&first, rest) = numerator_terms.split_first().unwrap();
            rest.iter()
                .copied()
                .fold(first, |acc, term| ctx.add(Expr::Add(acc, term)))
        };

        // Create the combined fraction
        let new_expr = ctx.add(Expr::Div(total_num, lcd));

        Some(Rewrite::new(new_expr).desc("Combine fractions with factor-based LCD"))
    }
);

// ========== Light Rationalization for Single Numeric Surd Denominators ==========
// Transforms: num / (k * √n) → (num * √n) / (k * n)
// Only applies when:
// - denominator contains exactly one numeric square root
// - base of the root is a positive integer
// - no variables inside the radical

define_rule!(
    RationalizeSingleSurdRule,
    "Rationalize Single Surd",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        use cas_ast::views::as_rational_const;
        use num_rational::BigRational;
        use num_traits::ToPrimitive;

        // Only match Div expressions - use zero-clone helper
        let (num, den) = crate::helpers::as_div(ctx, expr)?;

        // Check denominator for Pow(Number(n), 1/2) patterns
        // We need to find exactly one surd in the denominator factors

        // Helper to check if an expression is a numeric square root
        fn is_numeric_sqrt(ctx: &Context, id: ExprId) -> Option<i64> {
            if let Expr::Pow(base, exp) = ctx.get(id) {
                // Check exponent is 1/2 (using robust detection)
                let exp_val = as_rational_const(ctx, *exp, 8)?;
                let half = BigRational::new(1.into(), 2.into());
                if exp_val != half {
                    return None;
                }
                // Check base is a positive integer
                if let Expr::Number(n) = ctx.get(*base) {
                    if n.is_integer() {
                        return n.numer().to_i64().filter(|&x| x > 0);
                    }
                }
            }
            None
        }

        // Try different denominator patterns
        // CLONE_OK: Multi-branch match extracting sqrt value from Mul product
        let (sqrt_n_value, other_den_factors): (i64, Vec<ExprId>) = match ctx.get(den).clone() {
            // Case 1: Denominator is just √n
            Expr::Pow(_, _) => {
                if let Some(n) = is_numeric_sqrt(ctx, den) {
                    (n, vec![])
                } else {
                    return None;
                }
            }

            // Case 2: Denominator is k * √n or √n * k (one level of Mul)
            Expr::Mul(l, r) => {
                if let Some(n) = is_numeric_sqrt(ctx, l) {
                    // √n * k form
                    (n, vec![r])
                } else if let Some(n) = is_numeric_sqrt(ctx, r) {
                    // k * √n form
                    (n, vec![l])
                } else {
                    // Check if either side is a Mul containing √n (two levels)
                    // For simplicity, we only handle shallow cases
                    return None;
                }
            }

            // Case 3: Function("sqrt", [n])
            Expr::Function(name, ref args) if name == "sqrt" && args.len() == 1 => {
                if let Expr::Number(n) = ctx.get(args[0]) {
                    if n.is_integer() {
                        if let Some(n_int) = n.numer().to_i64().filter(|&x| x > 0) {
                            (n_int, vec![])
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                } else {
                    return None; // Variable inside sqrt
                }
            }

            _ => return None,
        };

        // Build the rationalized form: (num * √n) / (other_den * n)
        let n_expr = ctx.num(sqrt_n_value);
        let half = ctx.rational(1, 2);
        let sqrt_n = ctx.add(Expr::Pow(n_expr, half));

        // New numerator: num * √n
        let new_num = mul2_raw(ctx, num, sqrt_n);

        // New denominator: other_den_factors * n
        let n_in_den = ctx.num(sqrt_n_value);
        let new_den = if other_den_factors.is_empty() {
            n_in_den
        } else {
            let mut den_product = other_den_factors[0];
            for &f in &other_den_factors[1..] {
                den_product = mul2_raw(ctx, den_product, f);
            }
            mul2_raw(ctx, den_product, n_in_den)
        };

        let new_expr = ctx.add(Expr::Div(new_num, new_den));

        // Optional: Check node count didn't explode (shouldn't for this simple transform)
        if count_nodes(ctx, new_expr) > count_nodes(ctx, expr) + 10 {
            return None;
        }

        Some(Rewrite::new(new_expr).desc(format!(
            "{} / {} -> {} / {}",
            DisplayExpr {
                context: ctx,
                id: num
            },
            DisplayExpr {
                context: ctx,
                id: den
            },
            DisplayExpr {
                context: ctx,
                id: new_num
            },
            DisplayExpr {
                context: ctx,
                id: new_den
            }
        )))
    }
);

// ========== Binomial Conjugate Rationalization (Level 1) ==========
// Transforms: num / (A + B√n) → num * (A - B√n) / (A² - B²·n)
// Only applies when:
// - denominator is a binomial with exactly one numeric surd term
// - A, B are rational, n is a positive integer
// - uses closed-form arithmetic (no calls to general simplifier)

define_rule!(
    RationalizeBinomialSurdRule,
    "Rationalize Binomial Denominator",
    None,
    PhaseMask::RATIONALIZE,
    |ctx, expr| {
        use crate::rationalize_policy::RationalizeReason;
        use cas_ast::views::{as_rational_const, count_distinct_numeric_surds, is_surd_free};
        use num_rational::BigRational;
        use num_traits::ToPrimitive;

        // Only match Div expressions - use zero-clone helper
        let (num, den) = match crate::helpers::as_div(ctx, expr) {
            Some((n, d)) => (n, d),
            None => {
                tracing::trace!(target: "rationalize", "skipped: not a division");
                return None;
            }
        };

        // Budget guard: denominator shouldn't be too complex
        let den_nodes = count_nodes(ctx, den);
        if den_nodes > 30 {
            tracing::debug!(target: "rationalize", reason = ?RationalizeReason::BudgetExceeded, 
                            nodes = den_nodes, max = 30, "auto rationalize rejected");
            return None;
        }

        // Multi-surd guard: only rationalize if denominator has exactly 1 distinct surd
        // Level 1.5 blocks multi-surd expressions (reserved for `rationalize` command)
        let distinct_surds = count_distinct_numeric_surds(ctx, den, 50);
        if distinct_surds == 0 {
            tracing::trace!(target: "rationalize", "skipped: no surds found");
            return None;
        }
        if distinct_surds > 1 {
            tracing::debug!(target: "rationalize", reason = ?RationalizeReason::MultiSurdBlocked,
                            surds = distinct_surds, "auto rationalize rejected");
            return None;
        }

        // Try to parse denominator as A ± B√n (binomial surd)
        // Patterns: Add(A, Mul(B, √n)), Add(A, √n), Sub(A, Mul(B, √n)), etc.

        struct BinomialSurd {
            a: BigRational, // Rational constant term
            b: BigRational, // Coefficient of surd
            n: i64,         // Radicand (square-free positive integer)
            is_sub: bool,   // true if A - B√n, false if A + B√n
        }

        fn parse_binomial_surd(ctx: &Context, den: ExprId) -> Option<BinomialSurd> {
            // Helper to check if expression is a numeric √n
            fn is_numeric_sqrt(ctx: &Context, id: ExprId) -> Option<i64> {
                match ctx.get(id) {
                    Expr::Pow(base, exp) => {
                        let exp_val = as_rational_const(ctx, *exp, 8)?;
                        let half = BigRational::new(1.into(), 2.into());
                        if exp_val != half {
                            return None;
                        }
                        if let Expr::Number(n) = ctx.get(*base) {
                            if n.is_integer() {
                                return n.numer().to_i64().filter(|&x| x > 0);
                            }
                        }
                        None
                    }
                    Expr::Function(name, args) if name == "sqrt" && args.len() == 1 => {
                        if let Expr::Number(n) = ctx.get(args[0]) {
                            if n.is_integer() {
                                return n.numer().to_i64().filter(|&x| x > 0);
                            }
                        }
                        None
                    }
                    _ => None,
                }
            }

            // Helper to parse B*√n or √n (B=1), handling negation
            // Returns (signed_coefficient, radicand)
            fn parse_surd_term(ctx: &Context, id: ExprId) -> Option<(BigRational, i64)> {
                // Handle Neg(surd) → -(surd) with negated coefficient
                if let Expr::Neg(inner) = ctx.get(id) {
                    let (b, n) = parse_surd_term(ctx, *inner)?;
                    return Some((-b, n));
                }

                // Try √n directly (B=1)
                if let Some(n) = is_numeric_sqrt(ctx, id) {
                    return Some((BigRational::from_integer(1.into()), n));
                }

                // Try B * √n (including negative B)
                if let Expr::Mul(l, r) = ctx.get(id) {
                    if let Some(n) = is_numeric_sqrt(ctx, *r) {
                        if let Some(b) = as_rational_const(ctx, *l, 8) {
                            return Some((b, n)); // b is already signed
                        }
                    }
                    if let Some(n) = is_numeric_sqrt(ctx, *l) {
                        if let Some(b) = as_rational_const(ctx, *r, 8) {
                            return Some((b, n)); // b is already signed
                        }
                    }
                }
                None
            }

            match ctx.get(den) {
                // A + surd_term
                Expr::Add(l, r) => {
                    // Try l=A (rational), r=B√n
                    if let Some(a) = as_rational_const(ctx, *l, 8) {
                        if let Some((b, n)) = parse_surd_term(ctx, *r) {
                            return Some(BinomialSurd {
                                a,
                                b,
                                n,
                                is_sub: false,
                            });
                        }
                    }
                    // Try l=B√n, r=A
                    if let Some(a) = as_rational_const(ctx, *r, 8) {
                        if let Some((b, n)) = parse_surd_term(ctx, *l) {
                            return Some(BinomialSurd {
                                a,
                                b,
                                n,
                                is_sub: false,
                            });
                        }
                    }
                    None
                }
                // A - surd_term (or surd_term - A which is -(A - surd_term) with negated a)
                Expr::Sub(l, r) => {
                    // Try l=A (rational), r=B√n
                    if let Some(a) = as_rational_const(ctx, *l, 8) {
                        if let Some((b, n)) = parse_surd_term(ctx, *r) {
                            return Some(BinomialSurd {
                                a,
                                b,
                                n,
                                is_sub: true,
                            });
                        }
                    }
                    // Try l=B√n, r=A (symmetric case: surd - rational)
                    // This represents -A + B√n, so we negate a and use is_sub: false
                    if let Some(a) = as_rational_const(ctx, *r, 8) {
                        if let Some((b, n)) = parse_surd_term(ctx, *l) {
                            return Some(BinomialSurd {
                                a: -a, // negate since it's B√n - A = -A + B√n
                                b,
                                n,
                                is_sub: false, // After negation, this is -A + B√n
                            });
                        }
                    }
                    None
                }
                _ => None,
            }
        }

        // Helper to extract binomial factor from a Mul chain (Level 1.5)
        // Returns (k_factors, binomial) where k_factors are surd-free factors
        fn extract_binomial_from_product(
            ctx: &Context,
            den: ExprId,
        ) -> Option<(Vec<ExprId>, BinomialSurd)> {
            // First try direct binomial (Level 1)
            if let Some(surd) = parse_binomial_surd(ctx, den) {
                return Some((vec![], surd));
            }

            // Try Mul chain (Level 1.5)
            match ctx.get(den) {
                Expr::Mul(_, _) => {
                    // Flatten the Mul chain preserving order
                    fn collect_factors(ctx: &Context, id: ExprId, factors: &mut Vec<ExprId>) {
                        match ctx.get(id) {
                            Expr::Mul(l, r) => {
                                collect_factors(ctx, *l, factors);
                                collect_factors(ctx, *r, factors);
                            }
                            _ => factors.push(id),
                        }
                    }

                    let mut factors = Vec::new();
                    collect_factors(ctx, den, &mut factors);

                    // Find exactly one binomial factor; others must be surd-free
                    let mut binomial_idx = None;
                    for (i, &factor) in factors.iter().enumerate() {
                        if parse_binomial_surd(ctx, factor).is_some() {
                            if binomial_idx.is_some() {
                                // Multiple binomials → not Level 1.5
                                return None;
                            }
                            binomial_idx = Some(i);
                        } else if !is_surd_free(ctx, factor, 20) {
                            // Factor is neither binomial nor surd-free → skip
                            return None;
                        }
                    }

                    let binomial_idx = binomial_idx?;
                    let binomial = parse_binomial_surd(ctx, factors[binomial_idx])?;

                    // Collect K factors (those not the binomial)
                    let k_factors: Vec<_> = factors
                        .into_iter()
                        .enumerate()
                        .filter(|(i, _)| *i != binomial_idx)
                        .map(|(_, f)| f)
                        .collect();

                    Some((k_factors, binomial))
                }
                _ => None,
            }
        }

        let (k_factors, surd) = extract_binomial_from_product(ctx, den)?;

        // Build k_factor product (outside the helper to avoid borrow issues)
        let k_factor: Option<ExprId> = if k_factors.is_empty() {
            None
        } else if k_factors.len() == 1 {
            Some(k_factors[0])
        } else {
            let mut k = k_factors[0];
            for &f in &k_factors[1..] {
                k = ctx.add(Expr::Mul(k, f));
            }
            Some(k)
        };

        // Compute conjugate: if A + B√n, conjugate is A - B√n (and vice versa)
        // New denominator = A² - B²·n (always the same)
        let a_sq = &surd.a * &surd.a;
        let b_sq = &surd.b * &surd.b;
        let b_sq_n = &b_sq * BigRational::from_integer(surd.n.into());
        let new_den_val = &a_sq - &b_sq_n;

        // Check denominator is non-zero
        if new_den_val == BigRational::from_integer(0.into()) {
            return None;
        }

        // Build conjugate expression: A ∓ B√n
        let a_expr = ctx.add(Expr::Number(surd.a.clone()));
        let n_expr = ctx.num(surd.n);
        let half = ctx.rational(1, 2);
        let sqrt_n = ctx.add(Expr::Pow(n_expr, half));

        let b_sqrt_n = if surd.b == BigRational::from_integer(1.into()) {
            sqrt_n
        } else if surd.b == BigRational::from_integer((-1).into()) {
            ctx.add(Expr::Neg(sqrt_n))
        } else {
            let b_expr = ctx.add(Expr::Number(surd.b.clone()));
            mul2_raw(ctx, b_expr, sqrt_n)
        };

        // conjugate = A - B√n if original was A + B√n (is_sub=false)
        // conjugate = A + B√n if original was A - B√n (is_sub=true)
        let conjugate = if surd.is_sub {
            ctx.add(Expr::Add(a_expr, b_sqrt_n))
        } else {
            ctx.add(Expr::Sub(a_expr, b_sqrt_n))
        };

        // Build new numerator: num * conjugate
        // But first, handle negative denominator by absorbing sign into conjugate
        let (final_conjugate, final_den_val) = if new_den_val < BigRational::from_integer(0.into())
        {
            // Negative denominator: negate the entire conjugate
            // This produces -(A + B√n) instead of -A - B√n, which is cleaner for display
            let negated_conjugate = ctx.add(Expr::Neg(conjugate));
            (negated_conjugate, -new_den_val.clone())
        } else {
            (conjugate, new_den_val.clone())
        };

        let new_num = mul2_raw(ctx, num, final_conjugate);

        // Build new denominator as Number (now always positive or handled)
        let new_den = ctx.add(Expr::Number(final_den_val.clone()));

        // If denominator is 1, just return numerator (possibly divided by K)
        let new_expr = if final_den_val == BigRational::from_integer(1.into()) {
            // new_den = K (if present) or 1
            match k_factor {
                Some(k) => ctx.add(Expr::Div(new_num, k)),
                None => new_num,
            }
        } else {
            // new_den = K * (A² - B²n) or just (A² - B²n)
            let rationalized_den = ctx.add(Expr::Number(final_den_val.clone()));
            let full_den = match k_factor {
                Some(k) => mul2_raw(ctx, k, rationalized_den),
                None => rationalized_den,
            };
            ctx.add(Expr::Div(new_num, full_den))
        };

        // Verify we actually made progress (denominator is now rational)
        if count_nodes(ctx, new_expr) > count_nodes(ctx, expr) + 20 {
            return None;
        }

        Some(Rewrite::new(new_expr).desc(format!(
            "{} / {} -> {} / {}",
            DisplayExpr {
                context: ctx,
                id: num
            },
            DisplayExpr {
                context: ctx,
                id: den
            },
            DisplayExpr {
                context: ctx,
                id: new_num
            },
            DisplayExpr {
                context: ctx,
                id: new_den
            }
        )))
    }
);

// ============================================================================
// R1: Absorb Negation Into Difference Factor
// ============================================================================
// -1/((x-y)*...) → 1/((y-x)*...)
// Absorbs the negative sign by flipping one difference in the denominator.
// Differences can be Sub(x,y) or Add(x, Neg(y)) or Add(x, Mul(-1,y)).

/// Check if expression is a difference (x - y) in any canonical form
/// Returns Some((x, y)) if it's a difference
fn extract_difference(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr) {
        Expr::Sub(l, r) => Some((*l, *r)),
        Expr::Add(l, r) => {
            // Check if right is Neg(y)
            if let Expr::Neg(inner) = ctx.get(*r) {
                return Some((*l, *inner));
            }
            // Check if right is Mul(-1, y) or Mul(y, -1) with negative number
            if let Expr::Mul(a, b) = ctx.get(*r) {
                if let Expr::Number(n) = ctx.get(*a) {
                    if n.is_negative() && *n == num_rational::BigRational::from_integer((-1).into())
                    {
                        return Some((*l, *b));
                    }
                }
                if let Expr::Number(n) = ctx.get(*b) {
                    if n.is_negative() && *n == num_rational::BigRational::from_integer((-1).into())
                    {
                        return Some((*l, *a));
                    }
                }
            }
            // Check if left is Neg(x)
            if let Expr::Neg(inner) = ctx.get(*l) {
                return Some((*r, *inner));
            }
            None
        }
        _ => None,
    }
}

/// Build a difference expression: always use Sub form now that canonicalization
/// works properly with our fixes
fn build_difference(ctx: &mut Context, x: ExprId, y: ExprId) -> ExprId {
    ctx.add(Expr::Sub(x, y))
}

define_rule!(
    AbsorbNegationIntoDifferenceRule,
    "Absorb Negation Into Difference",
    |ctx, expr| {
        // Check for Neg(Div(...)) or Div with negative numerator
        let (is_neg_wrapped, div_num, div_den) = match ctx.get(expr) {
            Expr::Neg(inner) => {
                if let Expr::Div(n, d) = ctx.get(*inner) {
                    (true, *n, *d)
                } else {
                    return None;
                }
            }
            Expr::Div(n, d) => {
                if let Expr::Number(num_val) = ctx.get(*n) {
                    if num_val.is_negative() {
                        (false, *n, *d)
                    } else {
                        return None;
                    }
                } else if let Expr::Neg(_) = ctx.get(*n) {
                    (false, *n, *d)
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        // Collect all factors from denominator
        let mut factors: Vec<ExprId> = collect_mul_factors(ctx, div_den);

        // Find a difference factor to flip
        let mut flip_index: Option<usize> = None;
        let mut diff_pair: Option<(ExprId, ExprId)> = None;
        for (i, &f) in factors.iter().enumerate() {
            if let Some((x, y)) = extract_difference(ctx, f) {
                flip_index = Some(i);
                diff_pair = Some((x, y));
                break;
            }
        }

        let idx = flip_index?;
        let (x, y) = diff_pair?;

        // Flip the difference: (x - y) → (y - x)
        let flipped = build_difference(ctx, y, x);
        factors[idx] = flipped;

        // Rebuild denominator
        let new_den = factors.iter().copied().fold(None, |acc, f| {
            Some(match acc {
                Some(a) => mul2_raw(ctx, a, f),
                None => f,
            })
        })?;

        // Handle numerator: remove the negation
        let new_num = if is_neg_wrapped {
            div_num
        } else if let Expr::Number(n) = ctx.get(div_num) {
            ctx.add(Expr::Number(-n.clone()))
        } else if let Expr::Neg(inner) = ctx.get(div_num) {
            *inner
        } else {
            return None;
        };

        let new_expr = ctx.add(Expr::Div(new_num, new_den));

        Some(Rewrite::new(new_expr).desc("Absorb negation into difference factor"))
    }
);

// ============================================================================
// R2: Canonicalize Products of Same-Tail Differences
// ============================================================================
// 1/((p-t)*(q-t)) → 1/((t-p)*(t-q))
// When two difference factors share the same "tail" (right operand),
// flip both to have that common element first.
// Double-flip preserves the overall sign.

define_rule!(
    CanonicalDifferenceProductRule,
    "Canonicalize Difference Product",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let (num, den) = if let Expr::Div(n, d) = ctx.get(expr) {
            (*n, *d)
        } else {
            return None;
        };

        // Check if denominator is Mul of exactly two Sub expressions
        let (factor1, factor2) = if let Expr::Mul(l, r) = ctx.get(den) {
            (*l, *r)
        } else {
            return None;
        };

        // Both factors must be Sub
        let (p, t1) = if let Expr::Sub(a, b) = ctx.get(factor1) {
            (*a, *b)
        } else {
            return None;
        };
        let (q, t2) = if let Expr::Sub(a, b) = ctx.get(factor2) {
            (*a, *b)
        } else {
            return None;
        };

        // Check if they share the same tail
        if crate::ordering::compare_expr(ctx, t1, t2) != Ordering::Equal {
            return None;
        }

        let t = t1;

        // Only flip if the current form is NOT canonical
        // Canonical: (t - p) * (t - q) where t comes first in both
        // Current is (p - t) * (q - t) - needs flipping
        // Guard: if t already comes first in both, don't flip (avoid loops)
        let t_already_first_1 = if let Expr::Sub(a, _) = ctx.get(factor1) {
            crate::ordering::compare_expr(ctx, *a, t) == Ordering::Equal
        } else {
            false
        };
        let t_already_first_2 = if let Expr::Sub(a, _) = ctx.get(factor2) {
            crate::ordering::compare_expr(ctx, *a, t) == Ordering::Equal
        } else {
            false
        };

        if t_already_first_1 && t_already_first_2 {
            return None; // Already canonical
        }

        // Flip both: (p-t) → (t-p), (q-t) → (t-q)
        let new_factor1 = ctx.add(Expr::Sub(t, p));
        let new_factor2 = ctx.add(Expr::Sub(t, q));
        let new_den = mul2_raw(ctx, new_factor1, new_factor2);

        let new_expr = ctx.add(Expr::Div(num, new_den));

        Some(Rewrite::new(new_expr).desc("Canonicalize same-tail difference product"))
    }
);

// =============================================================================
// Combine same-denominator fractions in n-ary Add/Sub chains
// =============================================================================
// For expressions like: 1 + (a)/(d) - (b)/(d) + x
// This rule groups fractions with the same denominator and combines them:
// → 1 + (a - b)/(d) + x
//
// Domain Mode Policy:
// - Strict: only combine if prove_nonzero(d) == Proven
// - Assume: combine with domain_assumption warning "Assuming d ≠ 0"
// - Generic: combine unconditionally (educational mode)

define_rule!(
    CombineSameDenominatorFractionsRule,
    "Combine Same Denominator Fractions",
    |ctx, expr, parent_ctx| {
        use crate::domain::Proof;
        use crate::helpers::{flatten_add_sub_chain, prove_nonzero};
        use std::collections::HashMap;

        // Only handle Add or Sub at root
        if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
            return None;
        }

        // Flatten the Add/Sub chain into individual terms
        let terms = flatten_add_sub_chain(ctx, expr);
        if terms.len() < 2 {
            return None;
        }

        // Helper to get (num, den, is_negated) from a Div term - does NOT modify ctx
        let get_fraction = |ctx: &Context, term: ExprId| -> Option<(ExprId, ExprId, bool)> {
            match ctx.get(term) {
                Expr::Div(num, den) => Some((*num, *den, false)),
                Expr::Neg(inner) => {
                    if let Expr::Div(num, den) = ctx.get(*inner) {
                        // Neg(Div(n, d)) → mark as negated
                        Some((*num, *den, true))
                    } else {
                        None
                    }
                }
                _ => None,
            }
        };

        // Group terms by denominator
        // Key: ExprId of denominator, Value: list of (term_index, numerator, is_negated)
        let mut denom_groups: HashMap<ExprId, Vec<(usize, ExprId, bool)>> = HashMap::new();
        let mut non_fraction_indices: Vec<usize> = Vec::new();

        for (idx, &term) in terms.iter().enumerate() {
            if let Some((num, den, is_neg)) = get_fraction(ctx, term) {
                // Check if we already have this denominator (by structural equality)
                let mut found_key = None;
                for existing_den in denom_groups.keys() {
                    if crate::ordering::compare_expr(ctx, *existing_den, den) == Ordering::Equal {
                        found_key = Some(*existing_den);
                        break;
                    }
                }

                if let Some(key) = found_key {
                    denom_groups.get_mut(&key).unwrap().push((idx, num, is_neg));
                } else {
                    denom_groups.insert(den, vec![(idx, num, is_neg)]);
                }
            } else {
                non_fraction_indices.push(idx);
            }
        }

        // Find groups with more than one fraction (those can be combined)
        #[allow(clippy::type_complexity)]
        let mut combinable_groups: Vec<(ExprId, Vec<(usize, ExprId, bool)>)> = Vec::new();
        for (den, group) in denom_groups.iter() {
            if group.len() >= 2 {
                combinable_groups.push((*den, group.clone()));
            }
        }

        // If no groups can be combined, nothing to do
        if combinable_groups.is_empty() {
            return None;
        }

        // Take the first combinable group
        let (common_den, group) = &combinable_groups[0];

        // DOMAIN MODE GATE: Check if denominator is provably non-zero
        let den_nonzero = prove_nonzero(ctx, *common_den);
        let domain_mode = parent_ctx.domain_mode();

        // Determine if we should proceed and what warning to emit
        // Note: assumption_events not yet emitted for this rule
        let _domain_assumption: Option<&str> = match domain_mode {
            crate::DomainMode::Strict => {
                // Only combine if denominator is provably non-zero
                if den_nonzero != Proof::Proven {
                    return None;
                }
                None
            }
            crate::DomainMode::Assume => {
                // Combine with warning if not proven
                if den_nonzero != Proof::Proven {
                    Some("Assuming denominator ≠ 0")
                } else {
                    None
                }
            }
            crate::DomainMode::Generic => {
                // Educational mode: combine unconditionally
                None
            }
        };

        // Combine numerators: n1 + n2 + ... (handle negation)
        let combined_num_terms: Vec<ExprId> = group
            .iter()
            .map(|(_, num, is_neg)| {
                if *is_neg {
                    ctx.add(Expr::Neg(*num))
                } else {
                    *num
                }
            })
            .collect();

        let combined_num = if combined_num_terms.len() == 1 {
            combined_num_terms[0]
        } else {
            let mut acc = combined_num_terms[0];
            for term in &combined_num_terms[1..] {
                acc = ctx.add(Expr::Add(acc, *term));
            }
            acc
        };

        // Create combined fraction
        let combined_fraction = ctx.add(Expr::Div(combined_num, *common_den));

        // Build new expression: non-combined terms + combined fraction
        let _combined_indices: Vec<usize> = group.iter().map(|(idx, _, _)| *idx).collect();

        let mut new_terms: Vec<ExprId> = Vec::new();

        // Add non-fraction terms
        for &idx in &non_fraction_indices {
            new_terms.push(terms[idx]);
        }

        // Add uncombined fractions (those not in the combined group)
        for (den, single_group) in denom_groups.iter() {
            if single_group.len() == 1
                && crate::ordering::compare_expr(ctx, *den, *common_den) != Ordering::Equal
            {
                let (idx, _, _) = single_group[0];
                new_terms.push(terms[idx]);
            }
        }

        // Add the combined fraction
        new_terms.push(combined_fraction);

        // Build result expression
        if new_terms.is_empty() {
            return None;
        }

        let result = if new_terms.len() == 1 {
            new_terms[0]
        } else {
            let mut acc = new_terms[0];
            for term in &new_terms[1..] {
                acc = ctx.add(Expr::Add(acc, *term));
            }
            acc
        };

        // Avoid no-op
        if count_nodes(ctx, result) >= count_nodes(ctx, expr) {
            // Only proceed if we actually reduced something or combined fractions
            // Check that we did combine (reduced number of Div nodes)
            let old_divs = count_nodes_of_type(ctx, expr, "Div");
            let new_divs = count_nodes_of_type(ctx, result, "Div");
            if new_divs >= old_divs {
                return None;
            }
        }

        // ===== FOCUS CONSTRUCTION =====
        // Capture original fraction terms exactly as they appear (preserving signs)
        // This enables didactic display showing only the combined fractions
        let original_fractions: Vec<ExprId> = group
            .iter()
            .map(|&(idx, _, _)| terms[idx]) // term already has its sign
            .collect();

        // Build focus_before from original fraction terms
        let focus_before = if original_fractions.len() == 1 {
            original_fractions[0]
        } else {
            let mut acc = original_fractions[0];
            for &term in &original_fractions[1..] {
                acc = ctx.add(Expr::Add(acc, term));
            }
            acc
        };

        // focus_after is the combined fraction
        let focus_after = combined_fraction;

        Some(
            Rewrite::new(result)
                .desc("Combine fractions with same denominator")
                .local(focus_before, focus_after),
        )
    }
);
