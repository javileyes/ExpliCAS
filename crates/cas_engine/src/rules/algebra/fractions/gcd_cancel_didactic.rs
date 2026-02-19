//! Didactic factorization helpers for GCD-based cancellation.
//!
//! These helper functions detect common algebraic patterns (difference of squares,
//! perfect square trinomials, sum/difference of cubes, etc.) and produce didactic
//! rewrites that show the factorization step before cancellation.
//!
//! Used by `SimplifyFractionRule` in `gcd_cancel.rs` as early-return paths.

use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};

use super::core_rules::poly_eq;

// =============================================================================
// EARLY RETURN: Didactic expansion of perfect-square denominators for cancellation
// =============================================================================

/// Try to detect and cancel `(a^2 + 2ab + b^2) / (a+b)^2 → 1` with visible expansion step.
///
/// This avoids the "magic" GCD path by showing the user that (a+b)^2 = a^2 + 2ab + b^2.
/// Returns Some(Rewrite) if the pattern matches, None otherwise to fall through to GCD.
pub(super) fn try_expand_binomial_square_in_den_for_cancel(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    _domain_mode: crate::domain::DomainMode,
    _parent_ctx: &crate::parent_context::ParentContext,
) -> Option<Rewrite> {
    use crate::implicit_domain::ImplicitCondition;
    use cas_math::expr_rewrite::smart_mul;

    // STEP 1: Check if den = Pow(base, 2) where base = Add(a, b)
    let (base, exp) = crate::helpers::as_pow(ctx, den)?;

    // Exponent must be exactly 2 (integer)
    let exp_val = cas_math::numeric::as_i64(ctx, exp)?;
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
pub(super) fn try_difference_of_squares_in_num(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    _domain_mode: crate::domain::DomainMode,
    _parent_ctx: &crate::parent_context::ParentContext,
) -> Option<Rewrite> {
    use crate::implicit_domain::ImplicitCondition;
    use cas_math::expr_rewrite::smart_mul;

    // STEP 1: Check if num is a² - b² form
    // Try Sub(Pow(a,2), Pow(b,2)) first, also handle Pow(a,2) - Number(k²)
    let (a, b) = if let Some((left, right)) = crate::helpers::as_sub(ctx, num) {
        // left - right, check if left is a square
        let (a, exp_a) = crate::helpers::as_pow(ctx, left)?;
        let exp_a_val = cas_math::numeric::as_i64(ctx, exp_a)?;
        if exp_a_val != 2 {
            return None;
        }

        // right can be Pow(b, 2) or Number(k²)
        if let Some((b, exp_b)) = crate::helpers::as_pow(ctx, right) {
            let exp_b_val = cas_math::numeric::as_i64(ctx, exp_b)?;
            if exp_b_val != 2 {
                return None;
            }
            (a, b)
        } else if let Expr::Number(n) = ctx.get(right) {
            // Check if n is a perfect square
            if let Some(sqrt_n) = try_integer_sqrt(n) {
                let b = ctx.add(Expr::Number(sqrt_n));
                (a, b)
            } else {
                return None;
            }
        } else {
            return None;
        }
    } else if let Some((left, right)) = crate::helpers::as_add(ctx, num) {
        // Try Add(Pow(a,2), Neg(Pow(b,2))) which is how parser represents a² - b²
        let (a, exp_a) = crate::helpers::as_pow(ctx, left)?;
        let exp_a_val = cas_math::numeric::as_i64(ctx, exp_a)?;
        if exp_a_val != 2 {
            return None;
        }

        // right must be Neg(Pow(b,2)) or Neg(Number(k²))
        let neg_inner = crate::helpers::as_neg(ctx, right)?;
        if let Some((b, exp_b)) = crate::helpers::as_pow(ctx, neg_inner) {
            let exp_b_val = cas_math::numeric::as_i64(ctx, exp_b)?;
            if exp_b_val != 2 {
                return None;
            }
            (a, b)
        } else if let Expr::Number(n) = ctx.get(neg_inner) {
            if let Some(sqrt_n) = try_integer_sqrt(n) {
                let b = ctx.add(Expr::Number(sqrt_n));
                (a, b)
            } else {
                return None;
            }
        } else {
            return None;
        }
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
pub(super) fn try_perfect_square_minus_in_num(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    _domain_mode: crate::domain::DomainMode,
    _parent_ctx: &crate::parent_context::ParentContext,
) -> Option<Rewrite> {
    use crate::implicit_domain::ImplicitCondition;
    use cas_math::expr_rewrite::smart_mul;

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

    // Return the INTERMEDIATE form: (a-b)² / (a-b)
    // This allows CancelCommonFactorRule to fire as Step 2
    let after = ctx.add(Expr::Div(factored_num, den));

    let rewrite = Rewrite::new(after)
        .desc("Recognize: a² - 2ab + b² = (a-b)²")
        .local(num, factored_num)
        .requires(ImplicitCondition::NonZero(den));

    Some(rewrite)
}

// =============================================================================
// EARLY RETURN: Sum/Difference of cubes factorization
// =============================================================================

// Detects (a³ - b³) / (a-b) and recognizes it as (a-b)(a²+ab+b²)/(a-b) = a²+ab+b²
// Also handles (a³ + b³) / (a+b) = (a+b)(a²-ab+b²)/(a+b) = a²-ab+b²
pub(super) fn try_sum_diff_of_cubes_in_num(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    _domain_mode: crate::domain::DomainMode,
    _parent_ctx: &crate::parent_context::ParentContext,
) -> Option<Rewrite> {
    use crate::implicit_domain::ImplicitCondition;
    use cas_math::expr_rewrite::smart_mul;

    // Check if num is a³ - b³ (as Sub or Add with Neg)
    let (a, b, is_difference) = if let Some((left, right)) = crate::helpers::as_sub(ctx, num) {
        // a³ - b³
        let (a, exp_a) = crate::helpers::as_pow(ctx, left)?;
        if cas_math::numeric::as_i64(ctx, exp_a)? != 3 {
            return None;
        }

        let (b, exp_b) = crate::helpers::as_pow(ctx, right)?;
        if cas_math::numeric::as_i64(ctx, exp_b)? != 3 {
            return None;
        }

        (a, b, true)
    } else if let Some((left, right)) = crate::helpers::as_add(ctx, num) {
        // Try a³ + (-b³) for difference, or a³ + b³ for sum
        let (a, exp_a) = crate::helpers::as_pow(ctx, left)?;
        if cas_math::numeric::as_i64(ctx, exp_a)? != 3 {
            return None;
        }

        if let Some(neg_inner) = crate::helpers::as_neg(ctx, right) {
            // a³ + (-b³) = a³ - b³
            let (b, exp_b) = crate::helpers::as_pow(ctx, neg_inner)?;
            if cas_math::numeric::as_i64(ctx, exp_b)? != 3 {
                return None;
            }
            (a, b, true)
        } else {
            // a³ + b³
            let (b, exp_b) = crate::helpers::as_pow(ctx, right)?;
            if cas_math::numeric::as_i64(ctx, exp_b)? != 3 {
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

    // Return INTERMEDIATE form: (a-b)(a²+ab+b²) / (a-b) or (a+b)(...) / (a+b)
    // This allows CancelCommonFactorRule to fire as Step 2
    let after = ctx.add(Expr::Div(factored_num, den));

    let desc = if is_difference {
        "Factor: a³ - b³ = (a-b)(a² + ab + b²)"
    } else {
        "Factor: a³ + b³ = (a+b)(a² - ab + b²)"
    };

    let rewrite = Rewrite::new(after)
        .desc(desc)
        .local(num, factored_num)
        .requires(ImplicitCondition::NonZero(den));

    Some(rewrite)
}

// =============================================================================
// EARLY RETURN: Power quotient preserving factored form P^m / P^n → P^(m-n)
// =============================================================================
//
// V2.14.45: When both numerator and denominator are powers of the same base,
// simplify by subtracting exponents WITHOUT expanding to polynomial form.
// This preserves (x-1)⁴/(x-1)² → (x-1)² instead of x² - 2x + 1.
//
// Also handles: P^m / P → P^(m-1) and P / P^n → P^(1-n)
// =============================================================================
pub(super) fn try_power_quotient_preserve_form(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    _domain_mode: crate::domain::DomainMode,
    _parent_ctx: &crate::parent_context::ParentContext,
) -> Option<Rewrite> {
    use crate::implicit_domain::ImplicitCondition;

    // Extract base and exponent from numerator
    // Handle both Pow(base, exp) and just base (implicitly exp=1)
    let (num_base, num_exp) = if let Some((base, exp)) = crate::helpers::as_pow(ctx, num) {
        (base, Some(exp))
    } else {
        (num, None) // Treat as P^1
    };

    // Extract base and exponent from denominator
    let (den_base, den_exp) = if let Some((base, exp)) = crate::helpers::as_pow(ctx, den) {
        (base, Some(exp))
    } else {
        (den, None) // Treat as P^1
    };

    // Check if bases are structurally equal using polynomial comparison
    // This handles reordering like (x-1) vs (1-x) if signs are handled
    if !poly_eq(ctx, num_base, den_base) {
        return None;
    }

    // Both exponents must be numeric integers for safe simplification
    let num_exp_val: i64 = match num_exp {
        Some(e) => cas_math::numeric::as_i64(ctx, e)?,
        None => 1,
    };
    let den_exp_val: i64 = match den_exp {
        Some(e) => cas_math::numeric::as_i64(ctx, e)?,
        None => 1,
    };

    // Must have m > n for this to be a simplification (result is positive exponent)
    // Otherwise let other rules handle it
    if num_exp_val <= den_exp_val {
        return None;
    }

    let result_exp = num_exp_val - den_exp_val;

    // Build result: base^(m-n)
    let result = if result_exp == 1 {
        num_base // P^1 = P
    } else {
        let exp_expr = ctx.num(result_exp);
        ctx.add(Expr::Pow(num_base, exp_expr))
    };

    // Build local display: show P^m / P^n → P^(m-n)
    let desc = format!(
        "Cancel: P^{} / P^{} → P^{}",
        num_exp_val, den_exp_val, result_exp
    );

    Some(
        Rewrite::new(result)
            .desc(desc)
            .local(num, result)
            .requires(ImplicitCondition::NonZero(den)),
    )
}

// =============================================================================
// Helper: check if a rational number is a perfect square integer
// =============================================================================

/// If `n` is a positive integer that is a perfect square (1, 4, 9, 16, ...),
/// return its integer square root as a BigRational.
fn try_integer_sqrt(n: &num_rational::BigRational) -> Option<num_rational::BigRational> {
    use num_bigint::BigInt;
    use num_traits::Zero;

    // Must be a positive integer
    if !n.is_integer() {
        return None;
    }
    let val = n.to_integer();
    if val <= BigInt::zero() {
        return None;
    }

    // Compute integer square root via Newton's method
    let root = val.sqrt();
    // Verify it's a perfect square
    if &root * &root == val {
        Some(num_rational::BigRational::from_integer(root))
    } else {
        None
    }
}
