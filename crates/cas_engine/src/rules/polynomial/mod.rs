//! Polynomial manipulation rules: distribution, annihilation, combining like terms,
//! expansion, and factoring.
//!
//! This module is split into submodules:
//! - `expansion`: Binomial/multinomial expansion, auto-expand, polynomial identity detection
//! - `factoring`: Heuristic common factor extraction

mod expansion;
mod expansion_normalize;
mod factoring;
pub(crate) mod polynomial_helpers;

pub use expansion::{
    AutoExpandPowSumRule, AutoExpandSubCancelRule, BinomialExpansionRule,
    SmallMultinomialExpansionRule,
};
pub use expansion_normalize::{
    ExpandSmallBinomialPowRule, HeuristicPolyNormalizeAddRule, PolynomialIdentityZeroRule,
};
pub use factoring::{ExtractCommonMulFactorRule, HeuristicExtractCommonFactorAddRule};

// Re-export helpers used within this module
use polynomial_helpers::{
    count_additive_terms, flatten_additive_terms, is_conjugate, poly_equal, select_best_focus,
    unwrap_hold,
};

use crate::define_rule;
use crate::nary::{build_balanced_add, AddView, Sign};
use crate::ordering::compare_expr;
use crate::phase::PhaseMask;
use crate::polynomial::Polynomial;
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{Context, Expr, ExprId};
use num_integer::Integer;
use num_traits::{One, Zero};
use std::cmp::Ordering;

/// Check if an expression is a binomial (sum or difference of exactly 2 terms)
/// Examples: (a + b), (a - b), (x + (-y))
fn is_binomial(ctx: &Context, e: ExprId) -> bool {
    matches!(ctx.get(e), Expr::Add(_, _) | Expr::Sub(_, _))
}

/// Check if a product of two expressions forms a sum/difference of cubes identity:
/// - `(X + c) * (X² - c·X + c²) = X³ + c³`
/// - `(X - c) * (X² + c·X + c²) = X³ - c³`
///   where `c` is a constant (Number) and X is any expression.
///
/// This is used as an exception in the binomial×binomial guard to allow
/// distribution of cube identity products, enabling the engine to simplify
/// expressions like `sin(u)³ + 1 - (sin(u)+1)·(sin(u)²-sin(u)+1)` → 0.
fn is_cube_identity_product(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    try_match_cube_identity(ctx, a, b) || try_match_cube_identity(ctx, b, a)
}

/// Try to match (binomial, trinomial) as a cube identity.
/// Returns true if `binomial = X ± c` and `trinomial = X² ∓ c·X + c²`.
fn try_match_cube_identity(ctx: &Context, binomial: ExprId, trinomial: ExprId) -> bool {
    let Some((x, c_val, is_sum)) = (match ctx.get(binomial) {
        Expr::Add(l, r) => {
            // Try X + c (c on right)
            if let Expr::Number(n) = ctx.get(*r) {
                Some((*l, n.clone(), true))
            }
            // Try c + X (c on left)
            else if let Expr::Number(n) = ctx.get(*l) {
                Some((*r, n.clone(), true))
            } else {
                None
            }
        }
        Expr::Sub(l, r) => {
            // X - c
            if let Expr::Number(n) = ctx.get(*r) {
                Some((*l, n.clone(), false))
            } else {
                None
            }
        }
        _ => None,
    }) else {
        return false;
    };

    // Step 2: Flatten the trinomial into additive terms
    let mut terms = Vec::new();
    flatten_additive_terms(ctx, trinomial, false, &mut terms);
    if terms.len() != 3 {
        return false;
    }

    // Step 3: Verify the 3 terms match X², ±c·X, c²
    let c_squared = &c_val * &c_val;
    let mut found_x_sq = false;
    let mut found_cx = false;
    let mut found_c_sq = false;

    for (term, is_neg) in &terms {
        // Check for X² (must be positive)
        if !found_x_sq && !is_neg {
            if let Expr::Pow(base, exp) = ctx.get(*term) {
                if compare_expr(ctx, *base, x) == Ordering::Equal {
                    if let Expr::Number(n) = ctx.get(*exp) {
                        if *n == num_rational::BigRational::from_integer(2.into()) {
                            found_x_sq = true;
                            continue;
                        }
                    }
                }
            }
        }

        // Check for c·X or X (when |c|=1)
        // The cube identity has middle coefficient = -c:
        //   (X+c): middle = -c·X → if c>0, negated; if c<0, positive
        //   (X-c): middle = +c·X → if c>0, positive; if c<0, negated
        // Account for sign of c: the "visible" sign is (-c for sum, +c for diff)
        // combined with the sign already captured by flatten's is_neg.
        if !found_cx {
            use num_traits::Signed;
            let c_is_neg = c_val.is_negative();
            // expect_neg: the middle term's sign in the canonical identity
            // (X+c)(X²-cX+c²): middle = -c·X → neg when c>0, pos when c<0
            // (X-c)(X²+cX+c²): middle = +c·X → pos when c>0, neg when c<0
            let expect_neg = is_sum ^ c_is_neg; // XOR

            if *is_neg == expect_neg {
                let c_abs = if c_is_neg {
                    -c_val.clone()
                } else {
                    c_val.clone()
                };
                // Check if |c|=1: middle term is just X
                if c_abs.is_one() && compare_expr(ctx, *term, x) == Ordering::Equal {
                    found_cx = true;
                    continue;
                }
                // General case: check Mul(|c|, X) or Mul(X, |c|)
                if let Expr::Mul(ml, mr) = ctx.get(*term) {
                    if let Expr::Number(n) = ctx.get(*ml) {
                        if *n == c_abs && compare_expr(ctx, *mr, x) == Ordering::Equal {
                            found_cx = true;
                            continue;
                        }
                    }
                    if let Expr::Number(n) = ctx.get(*mr) {
                        if *n == c_abs && compare_expr(ctx, *ml, x) == Ordering::Equal {
                            found_cx = true;
                            continue;
                        }
                    }
                }
            }
        }

        // Check for c² (must be positive)
        if !found_c_sq && !is_neg {
            if let Expr::Number(n) = ctx.get(*term) {
                if *n == c_squared {
                    found_c_sq = true;
                    continue;
                }
            }
        }
    }

    found_x_sq && found_cx && found_c_sq
}

/// Extract cube identity components from a product.
/// Returns `(x, c_cubed, is_sum)` where:
/// - `x`: the base expression
/// - `c_cubed`: c³ as a BigRational
/// - `is_sum`: true for X³+c³, false for X³-c³
fn extract_cube_identity(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
) -> Option<(ExprId, num_rational::BigRational, bool)> {
    extract_cube_identity_ordered(ctx, a, b).or_else(|| extract_cube_identity_ordered(ctx, b, a))
}

fn extract_cube_identity_ordered(
    ctx: &Context,
    binomial: ExprId,
    trinomial: ExprId,
) -> Option<(ExprId, num_rational::BigRational, bool)> {
    // Extract X and c from the binomial
    let (x, c_val, is_sum) = match ctx.get(binomial) {
        Expr::Add(l, r) => {
            if let Expr::Number(n) = ctx.get(*r) {
                Some((*l, n.clone(), true))
            } else if let Expr::Number(n) = ctx.get(*l) {
                Some((*r, n.clone(), true))
            } else {
                None
            }
        }
        Expr::Sub(l, r) => {
            if let Expr::Number(n) = ctx.get(*r) {
                Some((*l, n.clone(), false))
            } else {
                None
            }
        }
        _ => None,
    }?;

    // Verify trinomial matches the cube identity
    if !try_match_cube_identity(ctx, binomial, trinomial) {
        return None;
    }

    // Compute c³
    let c_cubed = &c_val * &c_val * &c_val;

    // Determine sign: (X+c)(X²-cX+c²) = X³+c³
    // For Add(x, c): X³ + c³ (sum if c > 0, diff if c < 0 since c³ < 0)
    // For Sub(x, c): X³ - c³ (diff)
    Some((x, c_cubed, is_sum))
}

// ── Sum/Difference of Cubes Contraction Rule ────────────────────────────
//
// Pre-order rule: (X + c)·(X² - c·X + c²) → X³ + c³
//                 (X - c)·(X² + c·X + c²) → X³ - c³
//
// This fires BEFORE DistributeRule to prevent suboptimal splitting of the
// trinomial factor. Works for any base X (polynomial, transcendental, etc.)
define_rule!(
    SumDiffCubesContractionRule,
    "Sum/Difference of Cubes Contraction",
    None,
    PhaseMask::CORE,
    |ctx, expr| {
        let (l, r) = crate::helpers::as_mul(ctx, expr)?;

        let (x, c_cubed, _is_sum) = extract_cube_identity(ctx, l, r)?;

        // Build X³
        let three = ctx.add(Expr::Number(num_rational::BigRational::from_integer(
            3.into(),
        )));
        let x_cubed = ctx.add(Expr::Pow(x, three));

        // Build c³ node
        let c_cubed_node = ctx.add(Expr::Number(c_cubed.clone()));

        // Build X³ ± c³
        let result = if c_cubed >= num_rational::BigRational::from_integer(0.into()) {
            ctx.add(Expr::Add(x_cubed, c_cubed_node))
        } else {
            // c³ < 0: X³ + c³ where c³ is negative → use Add with negative number
            ctx.add(Expr::Add(x_cubed, c_cubed_node))
        };

        Some(
            Rewrite::new(result)
                .desc("Sum/Difference of cubes")
                .local(expr, result),
        )
    }
);

// ── Sqrt Perfect-Square Trinomial Rule ───────────────────────────────────
//
// sqrt(A² + 2·A·B + B²) → |A + B|
//
// Detects perfect-square trinomials inside sqrt and simplifies directly.
// Works for any sub-expressions A, B (polynomial, transcendental, etc.)
//
// Example: sqrt(sin²(u) + 2·sin(u) + 1) → |sin(u) + 1|
//
// We support two forms:
//   (a) A² + 2·A·c + c²  where c is a Number (most common from CSV)
//   (b) Fully symbolic: both A² and B² are Pow(_, 2) nodes

/// Extract the square root of a term if it is a perfect square.
///
/// Recognizes:
/// - `Pow(base, 2k)` → `base^k` (even power)
/// - `Mul(n, Pow(base, 2k))` where `n` is a perfect square integer → `√n · base^k`
/// - `Number(n)` where `n` is a perfect square integer → `√n`
///
/// Returns `Some(root)` where `term = root²`, or `None`.
pub(crate) fn extract_square_root_of_term(ctx: &mut Context, term: ExprId) -> Option<ExprId> {
    // Case 1: Pow(base, 2k) → base^k
    if let Expr::Pow(base, exp) = ctx.get(term) {
        let (base, exp) = (*base, *exp);
        if let Expr::Number(n) = ctx.get(exp) {
            let n = n.clone();
            if n.is_integer() {
                let int_val = n.to_integer();
                let two: num_bigint::BigInt = 2.into();
                if &int_val % &two == 0.into() && int_val > 0.into() {
                    let half_exp = &int_val / &two;
                    let half_exp_rat = num_rational::BigRational::from_integer(half_exp);
                    if half_exp_rat == num_rational::BigRational::from_integer(1.into()) {
                        return Some(base);
                    } else {
                        let half_exp_id = ctx.add(Expr::Number(half_exp_rat));
                        return Some(ctx.add(Expr::Pow(base, half_exp_id)));
                    }
                }
            }
        }
        return None;
    }

    // Case 2: Mul(coeff, Pow(base, 2k)) where coeff is a perfect square integer
    if let Expr::Mul(l, r) = ctx.get(term) {
        let (l, r) = (*l, *r);
        // Try both orderings: Mul(coeff, pow) and Mul(pow, coeff)
        for (maybe_coeff, maybe_pow) in [(l, r), (r, l)] {
            if let Expr::Number(coeff) = ctx.get(maybe_coeff) {
                if coeff.is_integer() && *coeff > num_rational::BigRational::from_integer(0.into())
                {
                    let coeff_int = coeff.to_integer();
                    let coeff_root = coeff_int.sqrt();
                    if &coeff_root * &coeff_root == coeff_int {
                        // coeff is a perfect square, now check if maybe_pow is Pow(base, 2k)
                        if let Some(pow_root) = extract_square_root_of_term(ctx, maybe_pow) {
                            // A = √coeff · pow_root
                            let root_num = ctx.add(Expr::Number(
                                num_rational::BigRational::from_integer(coeff_root),
                            ));
                            return Some(ctx.add(Expr::Mul(root_num, pow_root)));
                        }
                    }
                }
            }
        }
    }

    // Case 3: Number(n) where n is a perfect square integer
    if let Expr::Number(n) = ctx.get(term) {
        use num_traits::Zero;
        if n.is_integer() && *n > num_rational::BigRational::zero() {
            let int_val = n.to_integer();
            let root = int_val.sqrt();
            if &root * &root == int_val {
                return Some(ctx.add(Expr::Number(num_rational::BigRational::from_integer(root))));
            }
        }
    }

    None
}

/// Try to match a 3-term additive expression as a perfect-square trinomial.
/// Returns `Some((A, B, is_sub))` such that the expression equals `(A ± B)²`.
///
/// Handles even-power squares: `Pow(base, 2k)` is recognized as `(base^k)²`,
/// e.g. `u⁴ + 2u² + 1` matches as `(u² + 1)²`.
///
/// Also handles coefficient-bearing squares: `Mul(n, Pow(base, 2k))` where `n` is
/// a perfect square integer, e.g. `4u² + 12u + 9` matches as `(2u + 3)²`.
pub(crate) fn try_match_perfect_square_trinomial(
    ctx: &mut Context,
    arg: ExprId,
) -> Option<(ExprId, ExprId, bool)> {
    let mut terms = Vec::new();
    flatten_additive_terms(ctx, arg, false, &mut terms);
    if terms.len() != 3 {
        return None;
    }

    // Identify which terms are "squared" — either Pow(x, 2) or Number(n) where n = k²
    // We try all permutations of assigning A², 2AB, B² to the 3 terms.

    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                continue;
            }
            let k = 3 - i - j; // the remaining index

            let (term_a_sq, neg_a_sq) = &terms[i];
            let (term_mid, neg_mid) = &terms[k];
            let (term_b_sq, neg_b_sq) = &terms[j];

            // A² must be positive
            if *neg_a_sq {
                continue;
            }

            // B² must be positive
            if *neg_b_sq {
                continue;
            }

            // Extract A from A² (Pow(base, 2k) → A = base^k)
            // Also handles Mul(n, Pow(base, 2k)) where n is a perfect square → A = √n · base^k
            let a_expr = extract_square_root_of_term(ctx, *term_a_sq);
            let Some(a) = a_expr else { continue };

            // Extract B from B² using the same helper
            let b: ExprId;
            let b_val: Option<num_rational::BigRational>;

            if let Some(b_root) = extract_square_root_of_term(ctx, *term_b_sq) {
                b = b_root;
                b_val = if let Expr::Number(bn) = ctx.get(b) {
                    Some(bn.clone())
                } else {
                    None
                };
            } else {
                continue;
            }

            // Check middle term = ±2·A·B
            // neg_mid tells us if the term was subtracted via additive sign.
            // However, after canonicalization, -2·x may be stored as Mul(-2, x)
            // with sign=Pos, so we must also check for a negative leading coefficient.
            use num_traits::Signed;
            let mut effective_neg_mid = *neg_mid;
            let effective_mid = *term_mid;

            // Normalize: if middle term has a negative leading coefficient,
            // absorb that into effective_neg_mid.
            // This handles the case where canonicalization turns -(2·x) into (-2)·x.
            let effective_mid = {
                let mut mid = effective_mid;
                // Check for Mul(Number(neg), _) or Mul(_, Number(neg))
                if let Expr::Mul(l, r) = ctx.get(mid) {
                    let (l, r) = (*l, *r);
                    if let Expr::Number(n) = ctx.get(l) {
                        if n.is_negative() {
                            // Negate the coefficient and flip the sign
                            let abs_n = ctx.add(Expr::Number(-n.clone()));
                            mid = ctx.add(Expr::Mul(abs_n, r));
                            effective_neg_mid = !effective_neg_mid;
                        }
                    } else if let Expr::Number(n) = ctx.get(r) {
                        if n.is_negative() {
                            let abs_n = ctx.add(Expr::Number(-n.clone()));
                            mid = ctx.add(Expr::Mul(l, abs_n));
                            effective_neg_mid = !effective_neg_mid;
                        }
                    }
                } else if let Expr::Number(n) = ctx.get(mid) {
                    if n.is_negative() {
                        mid = ctx.add(Expr::Number(-n.clone()));
                        effective_neg_mid = !effective_neg_mid;
                    }
                } else if let Expr::Neg(inner) = ctx.get(mid) {
                    mid = *inner;
                    effective_neg_mid = !effective_neg_mid;
                }
                mid
            };

            let mid_matches = check_middle_term_2ab(ctx, effective_mid, a, b, &b_val);
            if !mid_matches {
                continue;
            }

            // Determine if it's (A+B)² or (A-B)²
            // For (A+B)²: middle term is +2AB (effective_neg_mid = false)
            // For (A-B)²: middle term is -2AB (effective_neg_mid = true)
            let is_sub = effective_neg_mid;

            return Some((a, b, is_sub));
        }
    }
    None
}

/// Check if `term` equals `2·A·B` (ignoring sign, which is handled by the caller).
fn check_middle_term_2ab(
    ctx: &mut Context,
    term: ExprId,
    a: ExprId,
    b: ExprId,
    b_val: &Option<num_rational::BigRational>,
) -> bool {
    use std::cmp::Ordering;
    let two = num_rational::BigRational::from_integer(2.into());

    // The middle term should be 2·A·B in some Mul arrangement.
    // Possible shapes:
    //   Mul(Number(2), Mul(A, B))
    //   Mul(Mul(Number(2), A), B)
    //   Mul(A, Mul(Number(2), B))
    //   Mul(Number(2·b_val), A)  when B is a number
    //   etc.

    // Strategy: flatten the Mul chain and check we have exactly {2, A, B}
    // or if B is numeric, {2·B_val, A}
    let mut factors = Vec::new();
    flatten_mul_factors(ctx, term, &mut factors);

    // Case 1: B is a Number(k). Middle should be 2k·A or A·2k
    if let Some(bv) = b_val {
        let expected_coeff = &two * bv;
        // Look for (2k) * A or A * (2k)
        if factors.len() == 2 {
            for perm in [(0, 1), (1, 0)] {
                if let Expr::Number(n) = ctx.get(factors[perm.0]) {
                    if *n == expected_coeff
                        && compare_expr(ctx, factors[perm.1], a) == Ordering::Equal
                    {
                        return true;
                    }
                }
            }
        }
        // Also check 3-factor: {2, k, A}
        if factors.len() == 3 {
            let mut found_two = false;
            let mut found_bv = false;
            let mut found_a = false;
            for &f in &factors {
                if !found_two {
                    if let Expr::Number(n) = ctx.get(f) {
                        if *n == two {
                            found_two = true;
                            continue;
                        }
                    }
                }
                if !found_bv {
                    if let Expr::Number(n) = ctx.get(f) {
                        if n == bv {
                            found_bv = true;
                            continue;
                        }
                    }
                }
                if !found_a && compare_expr(ctx, f, a) == Ordering::Equal {
                    found_a = true;
                    continue;
                }
            }
            if found_two && found_bv && found_a {
                return true;
            }
        }
    }

    // Case 2: General. Factors should be {2, A, B}
    if factors.len() == 2 {
        // Could be Mul(2, Mul(A,B)) already flattened to [2, A, B] in 3-factor case
        // Or Mul(A, B) where one of them absorbed the 2
        // Check if one factor is Number(2) * A and other is B, etc.
        // This is complex — try the 3-factor check only
    }

    if factors.len() == 3 {
        let mut found_two = false;
        let mut found_a = false;
        let mut found_b = false;
        for &f in &factors {
            if !found_two {
                if let Expr::Number(n) = ctx.get(f) {
                    if *n == two {
                        found_two = true;
                        continue;
                    }
                }
            }
            if !found_a && compare_expr(ctx, f, a) == Ordering::Equal {
                found_a = true;
                continue;
            }
            if !found_b && compare_expr(ctx, f, b) == Ordering::Equal {
                found_b = true;
                continue;
            }
        }
        if found_two && found_a && found_b {
            return true;
        }
    }

    // Case 3: Semantic fallback — build 2·A·B and compare structurally.
    // This handles compound A/B (e.g. A = Mul(2, u)) that factor-flattening misses.
    // Try multiple orderings since canonicalization may arrange Mul differently.
    {
        let two_id = ctx.add(Expr::Number(two));
        // Form 1: 2 * (A * B)
        let ab = ctx.add(Expr::Mul(a, b));
        let expected_1 = ctx.add(Expr::Mul(two_id, ab));
        if compare_expr(ctx, term, expected_1) == Ordering::Equal {
            return true;
        }
        // Form 2: (2 * A) * B
        let two_a = ctx.add(Expr::Mul(two_id, a));
        let expected_2 = ctx.add(Expr::Mul(two_a, b));
        if compare_expr(ctx, term, expected_2) == Ordering::Equal {
            return true;
        }
        // Form 3: A * (2 * B)
        let two_b = ctx.add(Expr::Mul(two_id, b));
        let expected_3 = ctx.add(Expr::Mul(a, two_b));
        if compare_expr(ctx, term, expected_3) == Ordering::Equal {
            return true;
        }

        // Form 4: (2*b_val) * A when B is numeric (handles coefficient absorption)
        if let Some(bv) = b_val {
            let expected_coeff_val = num_rational::BigRational::from_integer(2.into()) * bv;
            let coeff_id = ctx.add(Expr::Number(expected_coeff_val));
            let expected_4 = ctx.add(Expr::Mul(coeff_id, a));
            if compare_expr(ctx, term, expected_4) == Ordering::Equal {
                return true;
            }
            // Also try A * (2*b_val)
            let expected_5 = ctx.add(Expr::Mul(a, coeff_id));
            if compare_expr(ctx, term, expected_5) == Ordering::Equal {
                return true;
            }
        }
    }

    false
}

/// Flatten a multiplication chain into leaf factors.
fn flatten_mul_factors(ctx: &Context, expr: ExprId, factors: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            flatten_mul_factors(ctx, *l, factors);
            flatten_mul_factors(ctx, *r, factors);
        }
        _ => factors.push(expr),
    }
}

define_rule!(
    SqrtPerfectSquareRule,
    "Sqrt Perfect Square",
    None,
    PhaseMask::CORE,
    |ctx, expr| {
        // Match Pow(arg, 1/2) — sqrt is canonicalized to x^(1/2) early
        let (arg, exp) = match ctx.get(expr) {
            Expr::Pow(base, exp) => (*base, *exp),
            _ => return None,
        };
        // Check exponent is exactly 1/2
        let half = num_rational::BigRational::new(1.into(), 2.into());
        match ctx.get(exp) {
            Expr::Number(n) if *n == half => {}
            _ => return None,
        }

        // Try to match arg as a perfect-square trinomial
        let (a, b, is_sub) = try_match_perfect_square_trinomial(ctx, arg)?;

        // Build |A ± B|
        let inner = if is_sub {
            ctx.add(Expr::Sub(a, b))
        } else {
            ctx.add(Expr::Add(a, b))
        };
        let result = ctx.call_builtin(cas_ast::BuiltinFn::Abs, vec![inner]);

        Some(
            Rewrite::new(result)
                .desc("√(A² ± 2AB + B²) = |A ± B|")
                .local(expr, result),
        )
    }
);

/// computationally expensive and should be skipped.
///
/// Returns true (skip distribution) when ALL of:
///   - The additive side contains variables (pure-constant sums always OK)
///   - The factor matches one of these expensive patterns:
///
/// | Case | Pattern | Example | Why expensive |
/// |------|---------|---------|---------------|
/// | 1 | Var-free complex constant | `(√6+√2)/4` (≥5 nodes) | Nested radical × polynomial |
/// | 2 | Fractional exponents | `(1-x^(1/3)+x^(2/3))/(1+x)` | Cube-root rationalization residual |
/// | 3 | Multi-variable fraction | `(-b+√(b²-4ac))/(2a)` | Quadratic formula × polynomial |
/// | 4 | Non-Number × ≥4 terms | `√2 * (x⁴+4x³+6x²+4x+1)` | Distribute↔Factor oscillation |
///
/// Harmless factors are always allowed through:
///   - Simple numbers: `3`, `-1/2` (always distribute, even across many terms)
///   - Simple surds vs short sums: `√2 * (a+b)` (< 4 terms OK)
///   - Single variables: `x` (already blocked by should_distribute)
fn is_expensive_factor(ctx: &Context, factor: ExprId, additive: ExprId) -> bool {
    // Pure-constant additive sums always distribute (e.g. x*(√3-2) → √3·x - 2·x)
    let additive_vars = cas_ast::collect_variables(ctx, additive);
    if additive_vars.is_empty() {
        return false;
    }

    let factor_nodes = cas_ast::count_nodes(ctx, factor);
    let factor_vars = cas_ast::collect_variables(ctx, factor);

    // Case 1: Variable-free complex constant (≥5 nodes)
    // e.g. (√6+√2)/4, √(10+2√5)/4
    if factor_vars.is_empty() && factor_nodes >= 5 {
        return true;
    }

    // Case 2: Expression with fractional exponents (≥5 nodes)
    // e.g. (1-x^(1/3)+x^(2/3))/(1+x) from cube-root rationalization
    if factor_nodes >= 5 && has_fractional_exponents(ctx, factor) {
        return true;
    }

    // Case 3: Multi-variable fraction (≥3 vars, ≥10 nodes)
    // e.g. (-b+√(b²-4ac))/(2a) — distributing creates 5+ copies of this monster
    if factor_vars.len() >= 3 && factor_nodes >= 10 {
        return true;
    }

    // Case 4: Non-Number factor × many-term polynomial (≥4 terms)
    // Distributing surds, functions, or expressions across large polynomials
    // creates a Distribute↔Factor oscillation: the distributed terms get
    // immediately re-factored by ExtractCommonMultiplicativeFactorRule.
    // Cost grows superlinearly with term count (3 terms: <1s, 5 terms: >7s).
    // Numbers are exempt because scalar distribution (2*poly) creates useful
    // merged coefficients and doesn't oscillate.
    let additive_terms = count_additive_terms(ctx, additive);
    if additive_terms >= 4 && !matches!(ctx.get(factor), Expr::Number(_)) {
        return true;
    }

    false
}

/// Check if an expression tree contains any fractional exponents.
/// e.g. x^(1/3), x^(2/3), x^(1/2) — but NOT x^2 or x^(-1).
fn has_fractional_exponents(ctx: &Context, root: ExprId) -> bool {
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Pow(base, exp) => {
                // Check if exponent is a non-integer rational
                if let Expr::Number(n) = ctx.get(*exp) {
                    if !n.is_integer() {
                        return true;
                    }
                }
                // Also check if exponent is Div(a,b) form (e.g. 1/3 as AST)
                if matches!(ctx.get(*exp), Expr::Div(_, _)) {
                    return true;
                }
                stack.push(*base);
                stack.push(*exp);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(e) => stack.push(*e),
            Expr::Function(_, args) => {
                for &a in args {
                    stack.push(a);
                }
            }
            _ => {} // Leaves: Number, Variable, Constant
        }
    }
    false
}

// DistributeRule: Runs in CORE, TRANSFORM, RATIONALIZE but NOT in POST
// This prevents Factor↔Distribute infinite loops (FactorCommonIntegerFromAdd runs in POST)
define_rule!(
    DistributeRule,
    "Distributive Property",
    None,
    // NO POST: evita ciclo con FactorCommonIntegerFromAdd (ver test_factor_distribute_no_loop)
    PhaseMask::CORE | PhaseMask::TRANSFORM | PhaseMask::RATIONALIZE,
    |ctx, expr, parent_ctx| {
        use crate::semantics::NormalFormGoal;

        // GATE: Don't distribute when goal is Collected or Factored
        // This prevents undoing the effect of collect() or factor() commands
        match parent_ctx.goal() {
            NormalFormGoal::Collected | NormalFormGoal::Factored => return None,
            _ => {}
        }

        // Don't distribute if expression is in canonical form (e.g., inside abs() or sqrt())
        // This protects patterns like abs((x-2)(x+2)) from expanding
        if crate::canonical_forms::is_canonical_form(ctx, expr) {
            return None;
        }

        // GUARD: Block distribution when sin(4x) identity pattern is detected
        // This allows Sin4xIdentityZeroRule to see 4*sin(t)*cos(t)*(cos²-sin²) as a single product
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.has_sin4x_identity_pattern {
                return None;
            }
        }
        // Use zero-clone destructuring pattern
        let (l, r) = crate::helpers::as_mul(ctx, expr)?;

        // GUARD: Skip distribution when a factor is 1.
        // 1*(a+b) -> 1*a + 1*b is a visual no-op (MulOne is applied in rendering),
        // and produces confusing "Before/After identical" steps.
        if crate::helpers::is_one(ctx, l) || crate::helpers::is_one(ctx, r) {
            return None;
        }

        // a * (b + c) -> a*b + a*c
        if let Some((b, c)) = crate::helpers::as_add(ctx, r) {
            // PERFORMANCE: Don't distribute expensive factors (complex irrationals,
            // fractional exponents, multi-variable fractions) across polynomials.
            if is_expensive_factor(ctx, l, r) {
                return None;
            }

            // Distribute if 'l' is a Number, Function, Add/Sub, Pow, Mul, or Div.
            // We exclude Var to keep x(x+1) factored, but allow x^2(x+1) to expand.
            // Exception: always allow if the additive side is variable-free (pure constants/surds)
            // so that x*(√3-2) -> √3·x - 2·x for like-term collection.
            let l_expr = ctx.get(l);
            let additive_is_constant = cas_ast::collect_variables(ctx, r).is_empty();
            let should_distribute = additive_is_constant
                || matches!(l_expr, Expr::Number(_))
                || matches!(l_expr, Expr::Function(_, _))
                || matches!(l_expr, Expr::Add(_, _))
                || matches!(l_expr, Expr::Sub(_, _))
                || matches!(l_expr, Expr::Pow(_, _))
                || matches!(l_expr, Expr::Mul(_, _))
                || matches!(l_expr, Expr::Div(_, _))
                || (matches!(l_expr, Expr::Variable(_))
                    && cas_ast::collect_variables(ctx, expr).len() > 1);

            if !should_distribute {
                return None;
            }

            // CRITICAL: Avoid undoing FactorDifferenceSquaresRule
            // If we have (A+B)(A-B), do NOT distribute.
            if is_conjugate(ctx, l, r) {
                return None;
            }

            // N-ary conjugate protection (secondary defense).
            // Primary defense is the pre-order conjugate pair contraction in
            // transform_binary. This guards against cases where the parent
            // references are still in the same form.
            if let Some(parent_id) = parent_ctx.immediate_parent() {
                if let Expr::Mul(pl, pr) = ctx.get(parent_id) {
                    if is_conjugate(ctx, r, *pl) || is_conjugate(ctx, r, *pr) {
                        return None;
                    }
                }
            }

            // CRITICAL: Don't expand binomial*binomial products like (a-b)*(a-c)
            // This preserves factored form for opposite denominator detection
            // EXCEPTION: Allow sum/difference of cubes identity products
            if is_binomial(ctx, l) && is_binomial(ctx, r) && !is_cube_identity_product(ctx, l, r) {
                return None;
            }

            // EDUCATIONAL: Don't distribute fractional coefficient over binomial
            // Preserves clean form like 1/2*(√2-1) instead of √2/2 - 1/2
            if let Expr::Number(n) = ctx.get(l) {
                if !n.is_integer() && is_binomial(ctx, r) {
                    return None;
                }
            }

            let ab = smart_mul(ctx, l, b);
            let ac = smart_mul(ctx, l, c);
            let new_expr = ctx.add(Expr::Add(ab, ac));
            return Some(
                Rewrite::new(new_expr)
                    .desc("Distribute")
                    .local(expr, new_expr),
            );
        }

        // a * (b - c) -> a*b - a*c
        if let Some((b, c)) = crate::helpers::as_sub(ctx, r) {
            // PERFORMANCE: Same expensive-factor guard as Add branch
            if is_expensive_factor(ctx, l, r) {
                return None;
            }

            let l_expr = ctx.get(l);
            let additive_is_constant = cas_ast::collect_variables(ctx, r).is_empty();
            let should_distribute = additive_is_constant
                || matches!(l_expr, Expr::Number(_))
                || matches!(l_expr, Expr::Function(_, _))
                || matches!(l_expr, Expr::Add(_, _))
                || matches!(l_expr, Expr::Sub(_, _))
                || matches!(l_expr, Expr::Pow(_, _))
                || matches!(l_expr, Expr::Mul(_, _))
                || matches!(l_expr, Expr::Div(_, _))
                || (matches!(l_expr, Expr::Variable(_))
                    && cas_ast::collect_variables(ctx, expr).len() > 1);

            if !should_distribute {
                return None;
            }

            // CRITICAL: Avoid undoing FactorDifferenceSquaresRule
            if is_conjugate(ctx, l, r) {
                return None;
            }

            // N-ary conjugate protection
            if let Some(parent_id) = parent_ctx.immediate_parent() {
                if let Expr::Mul(pl, pr) = ctx.get(parent_id) {
                    if is_conjugate(ctx, r, *pl) || is_conjugate(ctx, r, *pr) {
                        return None;
                    }
                }
            }

            // Don't expand binomial*binomial products
            // EXCEPTION: Allow sum/difference of cubes identity products
            if is_binomial(ctx, l) && is_binomial(ctx, r) && !is_cube_identity_product(ctx, l, r) {
                return None;
            }

            // EDUCATIONAL: Don't distribute fractional coefficient over binomial
            if let Expr::Number(n) = ctx.get(l) {
                if !n.is_integer() && is_binomial(ctx, r) {
                    return None;
                }
            }

            let ab = smart_mul(ctx, l, b);
            let ac = smart_mul(ctx, l, c);
            let new_expr = ctx.add(Expr::Sub(ab, ac));
            return Some(
                Rewrite::new(new_expr)
                    .desc("Distribute")
                    .local(expr, new_expr),
            );
        }

        // (b + c) * a -> b*a + c*a
        if let Some((b, c)) = crate::helpers::as_add(ctx, l) {
            // PERFORMANCE: Same expensive-factor guard (mirror of a*(b+c))
            if is_expensive_factor(ctx, r, l) {
                return None;
            }

            // Same logic for 'r', with variable-free bypass for constant sums
            let r_expr = ctx.get(r);
            let additive_is_constant = cas_ast::collect_variables(ctx, l).is_empty();
            let should_distribute = additive_is_constant
                || matches!(r_expr, Expr::Number(_))
                || matches!(r_expr, Expr::Function(_, _))
                || matches!(r_expr, Expr::Add(_, _))
                || matches!(r_expr, Expr::Sub(_, _))
                || matches!(r_expr, Expr::Pow(_, _))
                || matches!(r_expr, Expr::Mul(_, _))
                || matches!(r_expr, Expr::Div(_, _))
                || (matches!(r_expr, Expr::Variable(_))
                    && cas_ast::collect_variables(ctx, expr).len() > 1);

            if !should_distribute {
                return None;
            }

            // CRITICAL: Avoid undoing FactorDifferenceSquaresRule
            if is_conjugate(ctx, l, r) {
                return None;
            }

            // N-ary conjugate protection (mirror of RHS case above)
            if let Some(parent_id) = parent_ctx.immediate_parent() {
                if let Expr::Mul(pl, pr) = ctx.get(parent_id) {
                    if is_conjugate(ctx, l, *pl) || is_conjugate(ctx, l, *pr) {
                        return None;
                    }
                }
            }

            // CRITICAL: Don't expand binomial*binomial products (Policy A+)
            // This preserves factored form like (a+b)*(c+d)
            // EXCEPTION: Allow sum/difference of cubes identity products
            if is_binomial(ctx, l) && is_binomial(ctx, r) && !is_cube_identity_product(ctx, l, r) {
                return None;
            }

            // EDUCATIONAL: Don't distribute fractional coefficient over binomial
            // Preserves clean form like (√2-1)/2 instead of √2/2 - 1/2
            if let Expr::Number(n) = ctx.get(r) {
                if !n.is_integer() && is_binomial(ctx, l) {
                    return None;
                }
            }

            let ba = smart_mul(ctx, b, r);
            let ca = smart_mul(ctx, c, r);
            let new_expr = ctx.add(Expr::Add(ba, ca));
            return Some(
                Rewrite::new(new_expr)
                    .desc("Distribute")
                    .local(expr, new_expr),
            );
        }

        // (b - c) * a -> b*a - c*a
        if let Some((b, c)) = crate::helpers::as_sub(ctx, l) {
            // PERFORMANCE: Same expensive-factor guard (mirror of a*(b-c))
            if is_expensive_factor(ctx, r, l) {
                return None;
            }

            let r_expr = ctx.get(r);
            let additive_is_constant = cas_ast::collect_variables(ctx, l).is_empty();
            let should_distribute = additive_is_constant
                || matches!(r_expr, Expr::Number(_))
                || matches!(r_expr, Expr::Function(_, _))
                || matches!(r_expr, Expr::Add(_, _))
                || matches!(r_expr, Expr::Sub(_, _))
                || matches!(r_expr, Expr::Pow(_, _))
                || matches!(r_expr, Expr::Mul(_, _))
                || matches!(r_expr, Expr::Div(_, _))
                || (matches!(r_expr, Expr::Variable(_))
                    && cas_ast::collect_variables(ctx, expr).len() > 1);

            if !should_distribute {
                return None;
            }

            // CRITICAL: Avoid undoing FactorDifferenceSquaresRule
            if is_conjugate(ctx, l, r) {
                return None;
            }

            // N-ary conjugate protection
            if let Some(parent_id) = parent_ctx.immediate_parent() {
                if let Expr::Mul(pl, pr) = ctx.get(parent_id) {
                    if is_conjugate(ctx, l, *pl) || is_conjugate(ctx, l, *pr) {
                        return None;
                    }
                }
            }

            // Don't expand binomial*binomial products
            // EXCEPTION: Allow sum/difference of cubes identity products
            if is_binomial(ctx, l) && is_binomial(ctx, r) && !is_cube_identity_product(ctx, l, r) {
                return None;
            }

            // EDUCATIONAL: Don't distribute fractional coefficient over binomial
            if let Expr::Number(n) = ctx.get(r) {
                if !n.is_integer() && is_binomial(ctx, l) {
                    return None;
                }
            }

            let ba = smart_mul(ctx, b, r);
            let ca = smart_mul(ctx, c, r);
            let new_expr = ctx.add(Expr::Sub(ba, ca));
            return Some(
                Rewrite::new(new_expr)
                    .desc("Distribute")
                    .local(expr, new_expr),
            );
        }

        // Handle Division Distribution: (a + b) / c -> a/c + b/c
        // Using AddView for shape-independent n-ary handling
        if let Some((numer, denom)) = crate::helpers::as_div(ctx, expr) {
            // Helper to check if division simplifies (shares factors) and return factor size
            let get_simplification_reduction = |ctx: &Context, num: ExprId, den: ExprId| -> usize {
                if num == den {
                    return cas_ast::count_nodes(ctx, num);
                }

                // Structural factor check
                let get_factors = |e: ExprId| -> Vec<ExprId> {
                    let mut factors = Vec::new();
                    let mut stack = vec![e];
                    while let Some(curr) = stack.pop() {
                        if let Expr::Mul(a, b) = ctx.get(curr) {
                            stack.push(*a);
                            stack.push(*b);
                        } else {
                            factors.push(curr);
                        }
                    }
                    factors
                };

                let num_factors = get_factors(num);
                let den_factors = get_factors(den);

                for df in den_factors {
                    // Check for structural equality using compare_expr
                    let found = num_factors
                        .iter()
                        .any(|nf| compare_expr(ctx, *nf, df) == Ordering::Equal);

                    if found {
                        let factor_size = cas_ast::count_nodes(ctx, df);
                        // Factor removed from num and den -> 2 * size
                        let mut reduction = factor_size * 2;
                        // If factor is entire denominator, Div is removed -> +1
                        if df == den {
                            reduction += 1;
                        }
                        return reduction;
                    }

                    // Check for numeric GCD
                    if let Expr::Number(n_den) = ctx.get(df) {
                        let found_numeric = num_factors.iter().any(|nf| {
                            if let Expr::Number(n_num) = ctx.get(*nf) {
                                if n_num.is_integer() && n_den.is_integer() {
                                    let num_int = n_num.to_integer();
                                    let den_int = n_den.to_integer();
                                    if !num_int.is_zero() && !den_int.is_zero() {
                                        let gcd = num_int.gcd(&den_int);
                                        return gcd > One::one();
                                    }
                                }
                            }
                            false
                        });
                        if found_numeric {
                            return 1; // Conservative estimate for number simplification
                        }
                    }
                }

                // Fallback to Polynomial GCD
                let vars = cas_ast::collect_variables(ctx, num);
                if vars.is_empty() {
                    return 0;
                }

                for var in vars {
                    if let (Ok(p_num), Ok(p_den)) = (
                        Polynomial::from_expr(ctx, num, &var),
                        Polynomial::from_expr(ctx, den, &var),
                    ) {
                        if p_den.is_zero() {
                            continue;
                        }
                        let gcd = p_num.gcd(&p_den);
                        // println!("DistributeRule Poly GCD check: num={:?} den={:?} var={} gcd={:?}", ctx.get(num), ctx.get(den), var, gcd);
                        if gcd.degree() > 0 || !gcd.leading_coeff().is_one() {
                            // Estimate complexity of GCD
                            // If GCD cancels denominator (degree match), reduction is high
                            if gcd.degree() == p_den.degree() {
                                // Assume denominator is removed (size(den) + 1)
                                return cas_ast::count_nodes(ctx, den) + 1;
                            }
                            // Otherwise, just return 1
                            return 1;
                        }
                    }
                }
                0
            };

            // N-ARY: Use AddView for shape-independent handling of sums
            // This correctly handles ((a+b)+c), (a+(b+c)), and balanced trees
            let num_view = AddView::from_expr(ctx, numer);

            // Check if it's actually a sum (more than 1 term)
            if num_view.terms.len() > 1 {
                // Calculate total reduction potential
                let mut total_reduction: usize = 0;
                let mut any_simplifies = false;

                for &(term, _sign) in &num_view.terms {
                    let red = get_simplification_reduction(ctx, term, denom);
                    if red > 0 {
                        any_simplifies = true;
                        total_reduction += red;
                    }
                }

                // Only distribute if at least one term simplifies
                if any_simplifies {
                    // Build new terms: each term divided by denominator
                    let new_terms: Vec<ExprId> = num_view
                        .terms
                        .iter()
                        .map(|&(term, sign)| {
                            let div_term = ctx.add(Expr::Div(term, denom));
                            match sign {
                                Sign::Pos => div_term,
                                Sign::Neg => ctx.add(Expr::Neg(div_term)),
                            }
                        })
                        .collect();

                    // Rebuild as balanced sum
                    let new_expr = build_balanced_add(ctx, &new_terms);

                    // Check complexity to prevent cycles with AddFractionsRule
                    let old_complexity = cas_ast::count_nodes(ctx, expr);
                    let new_complexity = cas_ast::count_nodes(ctx, new_expr);

                    // Allow if predicted complexity (after simplification) is not worse
                    if new_complexity <= old_complexity + total_reduction {
                        return Some(
                            Rewrite::new(new_expr)
                                .desc("Distribute division (simplifying)")
                                .local(expr, new_expr),
                        );
                    }
                }
            }
        }
        None
    }
);

// AnnihilationRule: Detects and cancels terms like x - x or __hold(sum) - sum
// Domain Mode Policy: Like AddInverseRule, we must respect domain_mode
// because if `x` can be undefined (e.g., a/(a-1) when a=1), then x - x
// is undefined, not 0.
// - Strict: only if no term contains potentially-undefined subexpressions
// - Assume: always apply (educational mode assumption: all expressions are defined)
// - Generic: same as Assume
define_rule!(AnnihilationRule, "Annihilation", |ctx, expr, parent_ctx| {
    // Helper: check if expression contains any Div with non-literal denominator
    // Delegates to canonical implementation that handles Hold/Matrix
    fn has_undefined_risk(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
        crate::collect::has_undefined_risk(ctx, expr)
    }

    // Only process Add/Sub expressions
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    // Flatten all terms
    let mut terms: Vec<(ExprId, bool)> = Vec::new();
    flatten_additive_terms(ctx, expr, false, &mut terms);

    if terms.len() < 2 {
        return None;
    }

    // CASE 1: Look for simple pairs that cancel (term and its negation)
    for i in 0..terms.len() {
        for j in (i + 1)..terms.len() {
            let (term_i, neg_i) = &terms[i];
            let (term_j, neg_j) = &terms[j];

            // Only if opposite signs
            if neg_i == neg_j {
                continue;
            }

            // Unwrap __hold for comparison
            let unwrapped_i = unwrap_hold(ctx, *term_i);
            let unwrapped_j = unwrap_hold(ctx, *term_j);

            // Check structural or polynomial equality
            if poly_equal(ctx, unwrapped_i, unwrapped_j) {
                // These terms cancel. If they're the only 2 terms, result is 0
                if terms.len() == 2 {
                    // DOMAIN MODE GATE: Check for undefined risk
                    let domain_mode = parent_ctx.domain_mode();
                    let either_has_risk =
                        has_undefined_risk(ctx, *term_i) || has_undefined_risk(ctx, *term_j);

                    if domain_mode == crate::DomainMode::Strict && either_has_risk {
                        return None;
                    }

                    // Note: domain assumption would be emitted here if Assume mode and either_has_risk
                    // but assumption_events are not emitted for this case yet

                    let zero = ctx.num(0);
                    return Some(Rewrite::new(zero).desc("x - x = 0"));
                }
            }
        }
    }

    // CASE 2: Handle __hold(A+B+...) with distributed -(A) -(B) -(...)
    // Find __hold terms and check if remaining negated terms sum to their content
    for (idx, (term, is_neg)) in terms.iter().enumerate() {
        if *is_neg {
            continue; // Only check positive __hold terms
        }

        // Check if this is a __hold using canonical helper
        if cas_ast::hold::is_hold(ctx, *term) {
            // Unwrap the held content
            let held_content = cas_ast::hold::unwrap_hold(ctx, *term);

            // Flatten the held content to get its terms
            let mut held_terms: Vec<(ExprId, bool)> = Vec::new();
            flatten_additive_terms(ctx, held_content, false, &mut held_terms);

            // Get all other terms (excluding this __hold)
            let other_terms: Vec<(ExprId, bool)> = terms
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, t)| *t)
                .collect();

            // Check if held_terms and other_terms cancel out
            // They cancel if for each held term there's an opposite signed other term
            if held_terms.len() == other_terms.len() {
                let mut all_cancel = true;
                let mut used = vec![false; other_terms.len()];

                for (held_term, held_neg) in &held_terms {
                    let mut found = false;

                    for (j, (other_term, other_neg)) in other_terms.iter().enumerate() {
                        if used[j] {
                            continue;
                        }

                        // Check if terms cancel (one positive, one negative equivalently)
                        // Case 1: Same term with opposite flags
                        if *other_neg != *held_neg {
                            // Use poly_equal for more robust comparison
                            // This handles cases where expressions are semantically equal
                            // but structurally different (e.g., Mul(15,x) vs Mul(x,15))
                            if poly_equal(ctx, *held_term, *other_term) {
                                used[j] = true;
                                found = true;
                                break;
                            }
                        }

                        // Case 2: Number with same flag but opposite value (e.g., 1 vs -1)
                        if *other_neg == *held_neg {
                            if let (Expr::Number(n1), Expr::Number(n2)) =
                                (ctx.get(*held_term), ctx.get(*other_term))
                            {
                                if n1 == &-n2.clone() {
                                    used[j] = true;
                                    found = true;
                                    break;
                                }
                            }
                        }
                    }

                    if !found {
                        all_cancel = false;
                        break;
                    }
                }

                if all_cancel && used.iter().all(|&u| u) {
                    let zero = ctx.num(0);
                    return Some(Rewrite::new(zero).desc("__hold(sum) - sum = 0"));
                }
            }
        }
    }

    None
});

// CombineLikeTermsRule: Collects like terms in Add/Mul expressions
// Now uses collect_with_semantics for domain_mode awareness:
// - Strict: refuses to cancel terms with undefined risk (e.g., x/(x+1) - x/(x+1))
// - Assume: cancels with domain_assumption warning
// - Generic: cancels unconditionally
define_rule!(
    CombineLikeTermsRule,
    "Combine Like Terms",
    |ctx, expr, parent_ctx| {
        // Only try to collect if it's an Add or Mul
        let is_add_or_mul = matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Mul(_, _));

        if is_add_or_mul {
            // CRITICAL: Do NOT apply to non-commutative expressions (e.g., matrices)
            if !ctx.is_mul_commutative(expr) {
                return None;
            }

            // Use semantics-aware collect that respects domain_mode
            let result = crate::collect::collect_with_semantics(ctx, expr, parent_ctx)?;

            // Check if structurally different to avoid infinite loops with ID regeneration
            if crate::ordering::compare_expr(ctx, result.new_expr, expr) == Ordering::Equal {
                return None;
            }

            // V2.14.26: Skip trivial changes that only normalize -1 coefficients
            // without actually combining or cancelling any terms.
            // This avoids noisy steps like "-1·x → -x" that don't add didactic value.
            if result.cancelled.is_empty() && result.combined.is_empty() {
                return None;
            }

            // V2.9.18: Restore granular focus using CollectResult's cancelled/combined groups
            // This provides specific focus like "5 - 5 → 0" for didactic clarity
            // Timeline highlighting uses step.path separately for broader context
            let (before_local, after_local, description) = select_best_focus(ctx, &result);

            let mut rewrite = Rewrite::new(result.new_expr).desc(description);
            if let (Some(before), Some(after)) = (before_local, after_local) {
                rewrite = rewrite.local(before, after);
            }
            return Some(rewrite);
        }
        None
    }
);

/// BinomialExpansionRule: (a + b)^n → expanded polynomial
/// ONLY expands true binomials (exactly 2 terms).
/// Multinomial expansion (>2 terms) is NOT done by default to avoid explosion.
/// Use explicit expand() mode for multinomial expansion.
/// Implements Rule directly to access ParentContext
pub fn register(simplifier: &mut crate::Simplifier) {
    // Register cube identity contraction BEFORE distribution to prevent suboptimal splits
    simplifier.add_rule(Box::new(SumDiffCubesContractionRule));
    // Sqrt perfect-square trinomial: sqrt(A²+2AB+B²) → |A+B|
    simplifier.add_rule(Box::new(SqrtPerfectSquareRule));
    simplifier.add_rule(Box::new(DistributeRule));
    simplifier.add_rule(Box::new(AnnihilationRule));
    simplifier.add_rule(Box::new(CombineLikeTermsRule));
    simplifier.add_rule(Box::new(BinomialExpansionRule));
    simplifier.add_rule(Box::new(SmallMultinomialExpansionRule));
    // V2.15.8: ExpandSmallBinomialPowRule - controlled by autoexpand_binomials flag
    // Enable via REPL: set autoexpand_binomials on
    simplifier.add_rule(Box::new(ExpandSmallBinomialPowRule));
    simplifier.add_rule(Box::new(AutoExpandPowSumRule));
    simplifier.add_rule(Box::new(AutoExpandSubCancelRule));
    simplifier.add_rule(Box::new(PolynomialIdentityZeroRule));
    // V2.15.8: HeuristicPolyNormalizeAddRule - poly-normalize Add/Sub in Heuristic mode
    // V2.15.9: HeuristicExtractCommonFactorAddRule - extract common factors first (priority 110)
    simplifier.add_rule(Box::new(HeuristicExtractCommonFactorAddRule));
    // V2.16: ExtractCommonMulFactorRule - extract common multiplicative factors from n-ary sums
    // Fixes cross-product NF divergence in metamorphic Mul tests (priority 108)
    simplifier.add_rule(Box::new(ExtractCommonMulFactorRule));
    simplifier.add_rule(Box::new(HeuristicPolyNormalizeAddRule));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;

    #[test]
    fn test_distribute() {
        let mut ctx = Context::new();
        let rule = DistributeRule;
        // x^2 * (x + 3) - use x^2 (not an integer) so guard doesn't block
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let x_sq = ctx.add(Expr::Pow(x, two));
        let add = ctx.add(Expr::Add(x, three));
        let expr = ctx.add(Expr::Mul(x_sq, add));

        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        // Should be (x^2 * x) + (x^2 * 3) before further simplification
        // Note: x^2*x -> x^3 happens in a later pass, not in DistributeRule
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "x^2 * x + x^2 * 3" // Canonical: polynomial order (x terms before constants)
        );
    }

    #[test]
    fn test_annihilation() {
        let mut ctx = Context::new();
        let rule = AnnihilationRule;
        let x = ctx.var("x");
        let expr = ctx.add(Expr::Sub(x, x));
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "0"
        );
    }

    #[test]
    fn test_combine_like_terms() {
        let mut ctx = Context::new();
        let rule = CombineLikeTermsRule;

        // 2x + 3x -> 5x
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let term1 = ctx.add(Expr::Mul(two, x));
        let term2 = ctx.add(Expr::Mul(three, x));
        let expr = ctx.add(Expr::Add(term1, term2));

        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "5 * x"
        );

        // x + 2x -> 3x
        let term1 = x;
        let term2 = ctx.add(Expr::Mul(two, x));
        let expr2 = ctx.add(Expr::Add(term1, term2));
        let rewrite2 = rule
            .apply(
                &mut ctx,
                expr2,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite2.new_expr
                }
            ),
            "3 * x"
        );

        // ln(x) + ln(x) -> 2 * ln(x)
        let ln_x = ctx.call_builtin(cas_ast::BuiltinFn::Ln, vec![x]);
        let expr3 = ctx.add(Expr::Add(ln_x, ln_x));
        let rewrite3 = rule
            .apply(
                &mut ctx,
                expr3,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        // ln(x) is log(e, x), prints as ln(x)
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite3.new_expr
                }
            ),
            "2 * ln(x)"
        );
    }

    #[test]
    fn test_polynomial_identity_zero_rule() {
        // Test: (a+b)^2 - (a^2 + 2ab + b^2) = 0
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");

        // (a+b)^2
        let a_plus_b = ctx.add(Expr::Add(a, b));
        let two = ctx.num(2);
        let a_plus_b_sq = ctx.add(Expr::Pow(a_plus_b, two));

        // a^2 + 2ab + b^2
        let a_sq = ctx.add(Expr::Pow(a, two));
        let b_sq = ctx.add(Expr::Pow(b, two));
        let ab = ctx.add(Expr::Mul(a, b));
        let two_ab = ctx.add(Expr::Mul(two, ab));
        let sum1 = ctx.add(Expr::Add(a_sq, two_ab));
        let rhs = ctx.add(Expr::Add(sum1, b_sq));

        // (a+b)^2 - (a^2 + 2ab + b^2)
        let expr = ctx.add(Expr::Sub(a_plus_b_sq, rhs));

        let rule = PolynomialIdentityZeroRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        // Should simplify to 0
        assert!(rewrite.is_some(), "Polynomial identity should be detected");
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.unwrap().new_expr
                }
            ),
            "0"
        );
    }

    #[test]
    fn test_polynomial_identity_zero_rule_non_identity() {
        // Test: (a+b)^2 - a^2 ≠ 0 (not an identity)
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");

        // (a+b)^2
        let a_plus_b = ctx.add(Expr::Add(a, b));
        let two = ctx.num(2);
        let a_plus_b_sq = ctx.add(Expr::Pow(a_plus_b, two));

        // a^2
        let a_sq = ctx.add(Expr::Pow(a, two));

        // (a+b)^2 - a^2
        let expr = ctx.add(Expr::Sub(a_plus_b_sq, a_sq));

        let rule = PolynomialIdentityZeroRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        // Should NOT return a rewrite (not an identity to 0)
        assert!(rewrite.is_none(), "Non-identity should not trigger rule");
    }
}
