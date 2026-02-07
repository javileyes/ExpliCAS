//! Core fraction rules and helpers.
//!
//! This module contains the main simplification and cancellation rules,
//! along with helper functions for polynomial comparison and factor collection.

use crate::build::mul2_raw;
use crate::define_rule;
use crate::multipoly::{
    gcd_multivar_layer2, gcd_multivar_layer25, multipoly_from_expr, multipoly_to_expr, GcdBudget,
    GcdLayer, Layer25Budget, MultiPoly, PolyBudget,
};
use crate::polynomial::Polynomial;
use crate::rule::{ChainedRewrite, Rewrite};
use crate::rules::algebra::helpers::{
    collect_denominators, collect_variables, count_nodes_of_type, distribute, gcd_rational,
};
use cas_ast::{Context, DisplayExpr, Expr, ExprId};
use num_traits::{One, Zero};

// =============================================================================
// Context-aware helpers for AddFractionsRule gating
// =============================================================================

/// Check if a function name is trigonometric (sin, cos, tan and inverses/hyperbolics)
pub fn is_trig_function_name(name: &str) -> bool {
    matches!(
        name,
        "sin"
            | "cos"
            | "tan"
            | "csc"
            | "sec"
            | "cot"
            | "asin"
            | "acos"
            | "atan"
            | "sinh"
            | "cosh"
            | "tanh"
            | "asinh"
            | "acosh"
            | "atanh"
    )
}

/// Check if a function (by SymbolId) is trigonometric
pub fn is_trig_function(ctx: &Context, fn_id: usize) -> bool {
    is_trig_function_name(ctx.sym_name(fn_id))
}

/// Check if expression is a constant involving π (e.g., pi, pi/9, 2*pi/3)
pub fn is_pi_constant(ctx: &Context, id: ExprId) -> bool {
    crate::helpers::extract_rational_pi_multiple(ctx, id).is_some()
}

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

/// Relation between two polynomial expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SignRelation {
    Same,    // a == b
    Negated, // a == -b (e.g., x-y vs y-x)
}

/// Compare two expressions to detect if they are equal or negated.
/// Returns Some(Same) if a == b, Some(Negated) if a == -b, None otherwise.
fn poly_relation(ctx: &Context, a: ExprId, b: ExprId) -> Option<SignRelation> {
    let budget = PolyBudget {
        max_terms: 100,
        max_total_degree: 10,
        max_pow_exp: 5,
    };

    let pa = multipoly_from_expr(ctx, a, &budget).ok()?;
    let pb = multipoly_from_expr(ctx, b, &budget).ok()?;

    if pa == pb {
        return Some(SignRelation::Same);
    }

    // Check if pa == -pb
    let neg_pb = pb.neg();
    if pa == neg_pb {
        return Some(SignRelation::Negated);
    }

    None
}

// =============================================================================
// A1: Structural Factor Cancellation (without polynomial expansion)
// =============================================================================

/// Collect multiplicative factors with integer exponents from an expression.
/// - Mul(...) is flattened
/// - Pow(base, k) with integer k becomes (base, k)
/// - Neg(x) is unwrapped and factors collected from x (for intersection purposes)
/// - Everything else becomes (expr, 1)
pub fn collect_mul_factors_int_pow(ctx: &Context, expr: ExprId) -> Vec<(ExprId, i64)> {
    let mut factors = Vec::new();
    // Unwrap top-level Neg for factor collection (enables intersection with positive terms)
    let actual_expr = match ctx.get(expr) {
        Expr::Neg(inner) => *inner,
        _ => expr,
    };
    collect_mul_factors_recursive(ctx, actual_expr, 1, &mut factors);
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

/// Build a product from factors with integer exponents.
///
/// Uses canonical `MulBuilder` (right-fold with exponents).
/// (See ARCHITECTURE.md "Canonical Utilities Registry")
pub fn build_mul_from_factors_a1(ctx: &mut Context, factors: &[(ExprId, i64)]) -> ExprId {
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
pub fn extract_as_fraction(ctx: &mut Context, expr: ExprId) -> (ExprId, ExprId, bool) {
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
// STEP 1.5: Cancel same-base power fractions P^m/P^n → P^(m-n) (shallow, PRE-ORDER)
// =============================================================================

// V2.14.35: Ultra-light rule for Pow(base,m)/Pow(base,n) → base^(m-n)
// Uses shallow ExprId comparison to avoid recursion/stack depth issues.
// This handles cases like ((x+y)^10)/((x+y)^9) that would otherwise overflow stack.
define_rule!(
    CancelPowersDivisionRule,
    "Cancel Same-Base Powers",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::helpers::prove_nonzero;
        use crate::implicit_domain::ImplicitCondition;

        // Match Div(Pow(base_num, exp_num), Pow(base_den, exp_den))
        let (num, den) = crate::helpers::as_div(ctx, expr)?;
        let (base_num, exp_num) = crate::helpers::as_pow(ctx, num)?;
        let (base_den, exp_den) = crate::helpers::as_pow(ctx, den)?;

        // STRUCTURAL COMPARISON: Use compare_expr to check if bases are structurally equal
        // This handles cases where (x+y) is parsed separately in num and den
        if crate::ordering::compare_expr(ctx, base_num, base_den) != std::cmp::Ordering::Equal {
            return None;
        }

        // Get exponents as integers
        let m = crate::helpers::as_i64(ctx, exp_num)?;
        let n = crate::helpers::as_i64(ctx, exp_den)?;

        // Skip if both are zero (undefined)
        if m == 0 && n == 0 {
            return None;
        }

        // DOMAIN GATE: need base ≠ 0 (derived from original denominator P^n ≠ 0)
        let domain_mode = parent_ctx.domain_mode();
        let proof = prove_nonzero(ctx, base_num);
        let key = crate::assumptions::AssumptionKey::nonzero_key(ctx, base_num);
        let decision = crate::domain::can_cancel_factor_with_hint(
            domain_mode,
            proof,
            key,
            base_num,
            "Cancel Same-Base Powers",
        );

        if !decision.allow {
            return None;
        }

        // Compute effective exponent difference: P^m / P^n = P^(m-n)
        let diff = m - n;

        // Build result based on diff
        let (result, desc) = if diff == 0 {
            // P^n / P^n → 1
            (ctx.num(1), format!("Cancel: P^{}/P^{} → 1", m, n))
        } else if diff == 1 {
            // P^(n+1) / P^n → P
            (base_num, format!("Cancel: P^{}/P^{} → P", m, n))
        } else if diff == -1 {
            // P^n / P^(n+1) → 1/P
            let one = ctx.num(1);
            let result = ctx.add(Expr::Div(one, base_num));
            (result, format!("Cancel: P^{}/P^{} → 1/P", m, n))
        } else if diff > 0 {
            // P^m / P^n → P^(m-n) where m > n
            let new_exp = ctx.num(diff);
            let result = ctx.add(Expr::Pow(base_num, new_exp));
            (result, format!("Cancel: P^{}/P^{} → P^{}", m, n, diff))
        } else {
            // P^m / P^n → 1/P^(n-m) where m < n
            let pos_diff = -diff;
            let new_exp = ctx.num(pos_diff);
            let pow_result = ctx.add(Expr::Pow(base_num, new_exp));
            let one = ctx.num(1);
            let result = ctx.add(Expr::Div(one, pow_result));
            (result, format!("Cancel: P^{}/P^{} → 1/P^{}", m, n, pos_diff))
        };

        Some(Rewrite::new(result)
            .desc(&desc)
            .local(expr, result)
            .requires(ImplicitCondition::NonZero(base_num))
            .assume_all(decision.assumption_events(ctx, base_num)))
    }
);

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

// Rule to cancel P^n / P → P^(n-1) (didactic step 2 for perfect squares and similar)
// Handles patterns like (x-y)²/(x-y) → x-y
define_rule!(
    CancelPowerFractionRule,
    "Cancel Power Fraction",
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Definability
    ),
    |ctx, expr, parent_ctx| {
        use crate::helpers::prove_nonzero;
        use crate::implicit_domain::ImplicitCondition;

        // Match Div(Pow(base, exp), den)
        let (num, den) = crate::helpers::as_div(ctx, expr)?;
        let (base, exp) = crate::helpers::as_pow(ctx, num)?;

        // Check if base == den OR base == -den using poly_relation
        let relation = poly_relation(ctx, base, den)?;

        // Get exponent as integer
        let exp_val = crate::helpers::as_i64(ctx, exp)?;
        if exp_val < 1 {
            return None; // Only handle exp >= 1
        }

        // DOMAIN GATE
        let domain_mode = parent_ctx.domain_mode();
        let proof = prove_nonzero(ctx, den);
        let key = crate::assumptions::AssumptionKey::nonzero_key(ctx, den);
        let decision = crate::domain::can_cancel_factor_with_hint(
            domain_mode,
            proof,
            key,
            den,
            "Cancel Power Fraction",
        );

        if !decision.allow {
            return None;
        }

        // Build base result: P^(n-1) or 1 if n=1
        let base_result = if exp_val == 1 {
            ctx.num(1)
        } else if exp_val == 2 {
            base
        } else {
            let new_exp = ctx.num(exp_val - 1);
            ctx.add(Expr::Pow(base, new_exp))
        };

        // Apply sign based on relation
        let (result, desc) = match relation {
            SignRelation::Same => (
                base_result,
                "Cancel: P^n/P → P^(n-1)"
            ),
            SignRelation::Negated => {
                // P^n / (-P) = -P^(n-1)
                let negated = ctx.add(Expr::Neg(base_result));
                (negated, "Cancel: P^n/(-P) → -P^(n-1)")
            }
        };

        Some(Rewrite::new(result)
            .desc(desc)
            .local(expr, result)
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
fn try_power_quotient_preserve_form(
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
        Some(e) => crate::helpers::as_i64(ctx, e)?,
        None => 1,
    };
    let den_exp_val: i64 = match den_exp {
        Some(e) => crate::helpers::as_i64(ctx, e)?,
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
            .desc(&desc)
            .local(num, result)
            .requires(ImplicitCondition::NonZero(den)),
    )
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

        // EARLY RETURN: Power quotient preserving factored form
        // V2.14.45: (x-1)⁴/(x-1)² → (x-1)² instead of x² - 2x + 1
        if let Some(rewrite) = try_power_quotient_preserve_form(ctx, num, den, domain_mode, parent_ctx) {
            return Some(rewrite);
        }

        // NOTE: PR-2 shallow GCD integration deferred.
        // The gcd_shallow_for_fraction function exists in poly_gcd.rs but calling it
        // here adds stack depth that causes overflow on complex expressions.
        // Future work: investigate stack-safe approach for power cancellation.

        // 0. Try multivariate GCD (Layer 1: monomial + content)
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

                // Short-circuit: if new_num is 0, result is just 0 - avoid confusing 0*GCD/GCD pattern
                // Let DivZeroRule or subsequent simplification handle it cleanly
                let num_is_zero = matches!(ctx.get(new_num), Expr::Number(n) if n.is_zero());
                if num_is_zero {
                    // Just return 0 directly with a cleaner description
                    let zero = ctx.num(0);
                    use crate::implicit_domain::ImplicitCondition;
                    return Some(
                        Rewrite::new(zero)
                            .desc("Numerator simplifies to 0")
                            .local(num, zero)
                            .requires(ImplicitCondition::NonZero(den))
                    );
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

                // Layer tag for verbose description (omit for clean didactic display)
                let _layer_tag = match layer {
                    GcdLayer::Layer1MonomialContent => "Layer 1: monomial+content",
                    GcdLayer::Layer2HeuristicSeeds => "Layer 2: heuristic seeds",
                    GcdLayer::Layer25TensorGrid => "Layer 2.5: tensor grid",
                };

                // Compute final result
                let result = if let Expr::Number(n) = ctx.get(new_den) {
                    if n.is_one() {
                        new_num // Denominator simplified to 1
                    } else {
                        ctx.add(Expr::Div(new_num, new_den))
                    }
                } else {
                    ctx.add(Expr::Div(new_num, new_den))
                };

                // Normalize expressions to ensure Rule: and After: display consistently
                use crate::canonical_forms::normalize_core;
                let factored_form_norm = normalize_core(ctx, factored_form);
                let result_norm = normalize_core(ctx, result);

                // === ChainedRewrite Pattern: Factor -> Cancel ===
                // Step 1 (main): Factor - show the factored form
                // Use requires (not assume) to avoid duplicate Requires/Assumed display
                use crate::implicit_domain::ImplicitCondition;
                let gcd_display = format!("{}", DisplayExpr { context: ctx, id: gcd_expr });
                let factor_rw = Rewrite::new(factored_form_norm)
                    .desc(format!("Factor by GCD: {}", gcd_display))
                    .local(expr, factored_form_norm)
                    .requires(ImplicitCondition::NonZero(den));

                // Step 2 (chained): Cancel - reduce to final result
                let cancel = ChainedRewrite::new(result_norm)
                    .desc("Cancel common factor")
                    .local(factored_form_norm, result_norm);

                return Some(factor_rw.chain(cancel));
            }
        }


        // 1. Univariate path: require single variable
        if vars.len() != 1 {
            return None;
        }
        let var = vars.iter().next()?;

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
        let (new_num_poly, rem_num) = match p_num.div_rem(&full_gcd) {
            Ok(result) => result,
            Err(_) => return None,
        };
        let (new_den_poly, rem_den) = match p_den.div_rem(&full_gcd) {
            Ok(result) => result,
            Err(_) => return None,
        };

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

        // Short-circuit: if new_num is 0, result is just 0 - avoid confusing 0*GCD/GCD pattern
        let num_is_zero = matches!(ctx.get(new_num), Expr::Number(n) if n.is_zero());
        if num_is_zero {
            let zero = ctx.num(0);
            use crate::implicit_domain::ImplicitCondition;
            return Some(
                Rewrite::new(zero)
                    .desc("Numerator simplifies to 0")
                    .local(num, zero)
                    .requires(ImplicitCondition::NonZero(den))
            );
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

        // Compute final result
        let result = if let Expr::Number(n) = ctx.get(new_den) {
            if n.is_one() {
                new_num // Denominator simplified to 1
            } else {
                ctx.add(Expr::Div(new_num, new_den))
            }
        } else {
            ctx.add(Expr::Div(new_num, new_den))
        };

        // Normalize expressions to ensure Rule: and After: display consistently
        use crate::canonical_forms::normalize_core;
        let factored_form_norm = normalize_core(ctx, factored_form);
        let result_norm = normalize_core(ctx, result);

        // === ChainedRewrite Pattern: Factor -> Cancel ===
        // Step 1 (main): Factor - show the factored form
        // Use requires (not assume) to avoid duplicate Requires/Assumed display
        use crate::implicit_domain::ImplicitCondition;
        let gcd_display = format!("{}", DisplayExpr { context: ctx, id: gcd_expr });
        let factor_rw = Rewrite::new(factored_form_norm)
            .desc(format!("Factor by GCD: {}", gcd_display))
            .local(expr, factored_form_norm)
            .requires(ImplicitCondition::NonZero(den));

        // Step 2 (chained): Cancel - reduce to final result
        let cancel = ChainedRewrite::new(result_norm)
            .desc("Cancel common factor")
            .local(factored_form_norm, result_norm);

        return Some(factor_rw.chain(cancel));
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

        let (&first, rest) = unique_denoms.split_first()?;
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
pub fn check_divisible_denominators(
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
