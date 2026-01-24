//! # Pythagorean Identity Simplification
//!
//! This module provides the `TrigPythagoreanSimplifyRule` which applies the
//! Pythagorean identity to simplify expressions of the form:
//!
//! - `k - k*sin²(x) → k*cos²(x)`
//! - `k - k*cos²(x) → k*sin²(x)`
//!
//! This rule was extracted from `CancelCommonFactorsRule` for better
//! step-by-step transparency and pedagogical clarity.

use crate::define_rule;
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::{Context, Expr, ExprId};
use num_traits::Zero;

define_rule!(
    TrigPythagoreanSimplifyRule,
    "Pythagorean Factor Form",
    |ctx, expr| {
        // Only apply to Add expressions (Sub is represented as Add with Neg)

        // We need exactly 2 terms: constant and trig term
        let mut terms = Vec::new();
        crate::helpers::flatten_add(ctx, expr, &mut terms);
        if terms.len() != 2 {
            return None;
        }

        let t1 = terms[0];
        let t2 = terms[1];

        // Try both orderings: (c_term, trig_term) or (trig_term, c_term)
        if let Some((result, desc)) = check_pythagorean_pattern(ctx, t1, t2) {
            return Some(Rewrite::new(result).desc(desc));
        }
        if let Some((result, desc)) = check_pythagorean_pattern(ctx, t2, t1) {
            return Some(Rewrite::new(result).desc(desc));
        }

        None
    }
);

// =============================================================================
// TrigPythagoreanChainRule: sin²(t) + cos²(t) → 1 (n-ary)
// =============================================================================
// Searches for sin²(t) and cos²(t) pairs with matching arguments in an additive
// chain of ANY length and replaces the pair with 1.
// This enables simplifications like: cos²(x/2) + sin²(x/2) - 1 → 0

define_rule!(
    TrigPythagoreanChainRule,
    "Pythagorean Chain Identity",
    |ctx, expr| {
        // Flatten the additive chain
        let mut terms = Vec::new();
        crate::helpers::flatten_add(ctx, expr, &mut terms);

        if terms.len() < 2 {
            return None;
        }

        // Collect sin² and cos² terms with their (argument, index, coefficient)
        let mut sin2_terms: Vec<(ExprId, usize, num_rational::BigRational)> = Vec::new();
        let mut cos2_terms: Vec<(ExprId, usize, num_rational::BigRational)> = Vec::new();

        for (i, &term) in terms.iter().enumerate() {
            if let Some((func_name, arg, coef)) = extract_trig_squared(ctx, term) {
                if func_name == "sin" {
                    sin2_terms.push((arg, i, coef));
                } else if func_name == "cos" {
                    cos2_terms.push((arg, i, coef));
                }
            }
        }

        // Find matching pairs: same argument, same coefficient (both coefficient 1)
        for (sin_arg, sin_idx, sin_coef) in sin2_terms.iter() {
            for (cos_arg, cos_idx, cos_coef) in cos2_terms.iter() {
                // Arguments must match
                if crate::ordering::compare_expr(ctx, *sin_arg, *cos_arg)
                    != std::cmp::Ordering::Equal
                {
                    continue;
                }

                // Coefficients must both be 1 (for basic sin²+cos² = 1)
                let one = num_rational::BigRational::from_integer(1.into());
                if *sin_coef != one || *cos_coef != one {
                    continue;
                }

                // Found a match! Replace sin²(t) + cos²(t) with 1
                let replacement = ctx.num(1);

                // Build new expression with the pair removed and 1 added
                let mut new_terms: Vec<ExprId> = Vec::new();
                for (j, &t) in terms.iter().enumerate() {
                    if j != *sin_idx && j != *cos_idx {
                        new_terms.push(t);
                    }
                }
                new_terms.push(replacement);

                // Build result as sum
                let result = if new_terms.len() == 1 {
                    new_terms[0]
                } else {
                    let mut acc = new_terms[0];
                    for &t in new_terms.iter().skip(1) {
                        acc = ctx.add(Expr::Add(acc, t));
                    }
                    acc
                };

                return Some(Rewrite::new(result).desc("sin²(x) + cos²(x) = 1"));
            }
        }

        None
    }
);

// =============================================================================
// TrigPythagoreanGenericCoefficientRule: A*sin²(t) + A*cos²(t) → A
// =============================================================================
// Extends the Pythagorean identity to work when the coefficient is any expression,
// not just a numeric constant. This enables simplifications like:
//   cos(u)²*sin(x)² + cos(u)²*cos(x)² → cos(u)²
// Which is needed to prove equivalences in combined identities.
//
// Key insight: when a term like cos(u)²*sin(x)² contains multiple trig², we
// extract ALL possible candidates and match across terms.

define_rule!(
    TrigPythagoreanGenericCoefficientRule,
    "Pythagorean with Generic Coefficient",
    |ctx, expr| {
        // Flatten the additive chain
        let mut terms = Vec::new();
        crate::helpers::flatten_add(ctx, expr, &mut terms);

        if terms.len() < 2 {
            return None;
        }

        // Extract ALL candidates from each term
        // Candidate: (term_index, trig_kind, trig_arg, sorted_coef_factors)
        // trig_kind: true = sin, false = cos
        let mut all_sin_candidates: Vec<(usize, ExprId, Vec<ExprId>)> = Vec::new();
        let mut all_cos_candidates: Vec<(usize, ExprId, Vec<ExprId>)> = Vec::new();

        for (i, &term) in terms.iter().enumerate() {
            let candidates = extract_all_trig_squared_candidates(ctx, term);
            for (is_sin, arg, mut coef_factors) in candidates {
                // Sort factors for canonical comparison
                coef_factors.sort_by(|a, b| crate::ordering::compare_expr(ctx, *a, *b));
                if is_sin {
                    all_sin_candidates.push((i, arg, coef_factors));
                } else {
                    all_cos_candidates.push((i, arg, coef_factors));
                }
            }
        }

        // Find matching pairs: same argument AND same coefficient factors from DIFFERENT terms
        for (sin_idx, sin_arg, sin_coef) in all_sin_candidates.iter() {
            for (cos_idx, cos_arg, cos_coef) in all_cos_candidates.iter() {
                // Must be different terms
                if sin_idx == cos_idx {
                    continue;
                }

                // Arguments must match
                if crate::ordering::compare_expr(ctx, *sin_arg, *cos_arg)
                    != std::cmp::Ordering::Equal
                {
                    continue;
                }

                // Coefficient factors must match (already sorted)
                if sin_coef.len() != cos_coef.len() {
                    continue;
                }

                // Skip if no coefficient (just sin² + cos², handled by basic rule)
                if sin_coef.is_empty() {
                    continue;
                }

                let mut all_match = true;
                for (sf, cf) in sin_coef.iter().zip(cos_coef.iter()) {
                    if crate::ordering::compare_expr(ctx, *sf, *cf) != std::cmp::Ordering::Equal {
                        all_match = false;
                        break;
                    }
                }
                if !all_match {
                    continue;
                }

                // Found a match! A*sin²(t) + A*cos²(t) → A
                // Build the coefficient expression from sorted factors
                let replacement = if sin_coef.len() == 1 {
                    sin_coef[0]
                } else {
                    let mut coef = sin_coef[0];
                    for &f in sin_coef.iter().skip(1) {
                        coef = ctx.add(Expr::Mul(coef, f));
                    }
                    coef
                };

                // Build new expression with the pair removed and A added
                let mut new_terms: Vec<ExprId> = Vec::new();
                for (j, &t) in terms.iter().enumerate() {
                    if j != *sin_idx && j != *cos_idx {
                        new_terms.push(t);
                    }
                }
                new_terms.push(replacement);

                // Build result as sum
                let result = if new_terms.len() == 1 {
                    new_terms[0]
                } else {
                    let mut acc = new_terms[0];
                    for &t in new_terms.iter().skip(1) {
                        acc = ctx.add(Expr::Add(acc, t));
                    }
                    acc
                };

                return Some(Rewrite::new(result).desc("A·sin²(x) + A·cos²(x) = A"));
            }
        }

        None
    }
);

/// Extract ALL possible (is_sin, argument, coefficient_factors) candidates from a term.
/// For a term like cos(u)²*sin(x)², returns TWO candidates:
///   1. (true, x, [cos(u)²])   -- interpreting sin(x)² as the trig, cos(u)² as coef
///   2. (false, u, [sin(x)²])  -- interpreting cos(u)² as the trig, sin(x)² as coef
fn extract_all_trig_squared_candidates(
    ctx: &Context,
    term: ExprId,
) -> Vec<(bool, ExprId, Vec<ExprId>)> {
    let mut results = Vec::new();

    // Case 1: Direct trig² (no coefficient)
    if let Expr::Pow(base, exp) = ctx.get(term) {
        if let Expr::Number(n) = ctx.get(*exp) {
            if *n == num_rational::BigRational::from_integer(2.into()) {
                if let Expr::Function(fn_id, args) = ctx.get(*base) { let name = ctx.sym_name(*fn_id);
                    if args.len() == 1 {
                        if name == "sin" {
                            results.push((true, args[0], vec![]));
                        } else if name == "cos" {
                            results.push((false, args[0], vec![]));
                        }
                    }
                }
            }
        }
        return results;
    }

    // Case 2: Mul - flatten and find ALL sin²/cos² factors
    if let Expr::Mul(_, _) = ctx.get(term) {
        let mut factors = Vec::new();
        let mut stack = vec![term];
        while let Some(curr) = stack.pop() {
            if let Expr::Mul(l, r) = ctx.get(curr) {
                stack.push(*r);
                stack.push(*l);
            } else {
                factors.push(curr);
            }
        }

        // Find ALL trig² factors
        for (i, &f) in factors.iter().enumerate() {
            if let Expr::Pow(base, exp) = ctx.get(f) {
                if let Expr::Number(n) = ctx.get(*exp) {
                    if *n == num_rational::BigRational::from_integer(2.into()) {
                        if let Expr::Function(fn_id, args) = ctx.get(*base) { let name = ctx.sym_name(*fn_id);
                            if args.len() == 1 && (name == "sin" || name == "cos") {
                                // Build coefficient from all OTHER factors
                                let coef_factors: Vec<ExprId> = factors
                                    .iter()
                                    .enumerate()
                                    .filter(|(j, _)| *j != i)
                                    .map(|(_, &g)| g)
                                    .collect();

                                let is_sin = name == "sin";
                                results.push((is_sin, args[0], coef_factors));
                            }
                        }
                    }
                }
            }
        }
    }

    results
}

// =============================================================================
// TrigPythagoreanLinearFoldRule: a·sin²(t) + b·cos²(t) + c → (a-b)·sin²(t) + (b+c)
// =============================================================================
// Uses the identity sin²(t) + cos²(t) = 1 to reduce linear combinations.
// Example: cos²(u) + 2·sin²(u) - 1 → sin²(u)
// This handles cases where we have both sin² and cos² of the same argument
// with numeric coefficients.

define_rule!(
    TrigPythagoreanLinearFoldRule,
    "Pythagorean Linear Fold",
    |ctx, expr| {
        use num_rational::BigRational;
        use num_traits::{One, Zero};

        // Flatten the additive chain
        let mut terms = Vec::new();
        crate::helpers::flatten_add(ctx, expr, &mut terms);

        if terms.len() < 2 {
            return None;
        }

        // Find sin²(t) and cos²(t) terms with NUMERIC coefficients and the same argument
        // We collect: (arg, is_sin, coef, term_index)
        let mut trig_sq_terms: Vec<(ExprId, bool, BigRational, usize)> = Vec::new();
        let mut numeric_constant: BigRational = BigRational::zero();
        let mut constant_indices: Vec<usize> = Vec::new();

        for (i, &term) in terms.iter().enumerate() {
            // Check for numeric constant
            if let Expr::Number(n) = ctx.get(term) {
                numeric_constant += n.clone();
                constant_indices.push(i);
                continue;
            }

            // Check for Neg(Number)
            if let Expr::Neg(inner) = ctx.get(term) {
                if let Expr::Number(n) = ctx.get(*inner) {
                    numeric_constant -= n.clone();
                    constant_indices.push(i);
                    continue;
                }
            }

            // Try to extract sin²(t) or cos²(t) with numeric coefficient
            if let Some((func_name, arg, coef)) = extract_trig_squared(ctx, term) {
                let is_sin = func_name == "sin";
                trig_sq_terms.push((arg, is_sin, coef, i));
            }
        }

        // Look for pairs: sin²(t) and cos²(t) with same argument
        for i in 0..trig_sq_terms.len() {
            for j in (i + 1)..trig_sq_terms.len() {
                let (arg_i, is_sin_i, coef_i, idx_i) = &trig_sq_terms[i];
                let (arg_j, is_sin_j, coef_j, idx_j) = &trig_sq_terms[j];

                // Must have same argument
                if crate::ordering::compare_expr(ctx, *arg_i, *arg_j) != std::cmp::Ordering::Equal {
                    continue;
                }

                // Must be different functions (one sin², one cos²)
                if is_sin_i == is_sin_j {
                    continue;
                }

                // Identify which is sin and which is cos
                let (sin_coef, cos_coef, sin_idx, cos_idx) = if *is_sin_i {
                    (coef_i.clone(), coef_j.clone(), *idx_i, *idx_j)
                } else {
                    (coef_j.clone(), coef_i.clone(), *idx_j, *idx_i)
                };

                let arg = *arg_i;

                // Apply the transformation: a·sin²(t) + b·cos²(t) + c → (a-b)·sin²(t) + (b+c)
                // where a = sin_coef, b = cos_coef, c = numeric_constant
                let a_minus_b = &sin_coef - &cos_coef;
                let b_plus_c = &cos_coef + &numeric_constant;

                // Only apply if this reduces complexity:
                // - Original: sin² term + cos² term + constant (if present)
                // - New: (a-b)·sin² + (b+c)
                // Reduction: we eliminate one trig² term
                //
                // Cases where we should apply:
                // - a == b: sin² term disappears, result is just (b+c)
                // - b + c == 0: constant term disappears, result is (a-b)·sin²
                //
                // Cases where we should NOT apply:
                // - a != b AND b+c != 0: just rearranging, not reducing (e.g., sin²-cos² → 2sin²-1)
                if !a_minus_b.is_zero() && !b_plus_c.is_zero() {
                    continue;
                }

                // Build the new expression
                let mut new_terms: Vec<ExprId> = Vec::new();

                // Add terms that are NOT sin²(arg), cos²(arg), or numeric constants
                for (k, &t) in terms.iter().enumerate() {
                    if k == sin_idx || k == cos_idx || constant_indices.contains(&k) {
                        continue;
                    }
                    new_terms.push(t);
                }

                // Add (a-b)·sin²(t) if non-zero
                if !a_minus_b.is_zero() {
                    let sin_t = ctx.call("sin", vec![arg]);
                    let two = ctx.num(2);
                    let sin_sq = ctx.add(Expr::Pow(sin_t, two));

                    let result_term = if a_minus_b.is_one() {
                        sin_sq
                    } else if a_minus_b == -BigRational::one() {
                        ctx.add(Expr::Neg(sin_sq))
                    } else {
                        let coef_expr = ctx.add(Expr::Number(a_minus_b.clone()));
                        ctx.add(Expr::Mul(coef_expr, sin_sq))
                    };
                    new_terms.push(result_term);
                }

                // Add (b+c) constant if non-zero
                if !b_plus_c.is_zero() {
                    let const_expr = ctx.add(Expr::Number(b_plus_c.clone()));
                    new_terms.push(const_expr);
                }

                // Build result
                let result = if new_terms.is_empty() {
                    ctx.num(0)
                } else if new_terms.len() == 1 {
                    new_terms[0]
                } else {
                    let mut acc = new_terms[0];
                    for &t in new_terms.iter().skip(1) {
                        acc = ctx.add(Expr::Add(acc, t));
                    }
                    acc
                };

                return Some(Rewrite::new(result).desc("a·sin²+b·cos²+c = (a-b)·sin²+(b+c)"));
            }
        }

        None
    }
);

// =============================================================================
// TrigPythagoreanLocalCollectFoldRule: k·R·sin²(t) + R·cos²(t) - R → (k-1)·R·sin²(t)
// =============================================================================
// Finds triplets in Add n-ary where:
// - Two terms share a common residual factor R multiplied by sin²(t) and cos²(t)
// - A third term is -R (or c·R where c allows folding)
// Example: 2·cos(x)²·sin(u)² + cos(u)²·cos(x)² - cos(x)² → cos(x)²·sin(u)²

define_rule!(
    TrigPythagoreanLocalCollectFoldRule,
    "Pythagorean Local Collect Fold",
    |ctx, expr| {
        use num_rational::BigRational;
        use num_traits::{One, Zero};

        // Flatten the additive chain
        let mut terms = Vec::new();
        crate::helpers::flatten_add(ctx, expr, &mut terms);

        if terms.len() < 3 {
            return None;
        }

        // Extract decompositions: (term_index, is_sin, arg, numeric_coef, residual_factors)
        // Use multi-candidate to handle terms with multiple trig² factors
        let mut decompositions: Vec<(usize, bool, ExprId, BigRational, Vec<ExprId>)> = Vec::new();
        // Pure residuals for -R matching: (index, factors, coef)
        let mut pure_residuals: Vec<(usize, Vec<ExprId>, BigRational)> = Vec::new();

        for (i, &term) in terms.iter().enumerate() {
            // Get ALL possible decompositions from this term
            for decomp in decompose_term_with_residual_multi(ctx, term) {
                decompositions.push((i, decomp.0, decomp.1, decomp.2, decomp.3));
            }
            if let Some((factors, coef)) = extract_as_product(ctx, term) {
                pure_residuals.push((i, factors, coef));
            }
        }

        // Find triplets: sin² term + cos² term + residual term
        for (sin_idx, is_sin, sin_arg, sin_coef, sin_residual) in decompositions.iter() {
            if !is_sin {
                continue;
            } // Must be sin² term

            for (cos_idx, is_cos_sin, cos_arg, cos_coef, cos_residual) in decompositions.iter() {
                if sin_idx == cos_idx {
                    continue;
                }
                if *is_cos_sin {
                    continue;
                } // Must be cos² term

                // Same argument
                if crate::ordering::compare_expr(ctx, *sin_arg, *cos_arg)
                    != std::cmp::Ordering::Equal
                {
                    continue;
                }

                // Same residual (sorted)
                let mut sin_res_sorted = sin_residual.clone();
                let mut cos_res_sorted = cos_residual.clone();
                sin_res_sorted.sort_by(|a, b| crate::ordering::compare_expr(ctx, *a, *b));
                cos_res_sorted.sort_by(|a, b| crate::ordering::compare_expr(ctx, *a, *b));

                if sin_res_sorted.len() != cos_res_sorted.len() {
                    continue;
                }
                if !sin_res_sorted
                    .iter()
                    .zip(cos_res_sorted.iter())
                    .all(|(a, b)| {
                        crate::ordering::compare_expr(ctx, *a, *b) == std::cmp::Ordering::Equal
                    })
                {
                    continue;
                }

                // Look for matching -R term
                for (res_idx, res_factors, res_coef) in pure_residuals.iter() {
                    if res_idx == sin_idx || res_idx == cos_idx {
                        continue;
                    }

                    let mut res_sorted = res_factors.clone();
                    res_sorted.sort_by(|a, b| crate::ordering::compare_expr(ctx, *a, *b));

                    if res_sorted.len() != sin_res_sorted.len() {
                        continue;
                    }
                    if !res_sorted.iter().zip(sin_res_sorted.iter()).all(|(a, b)| {
                        crate::ordering::compare_expr(ctx, *a, *b) == std::cmp::Ordering::Equal
                    }) {
                        continue;
                    }

                    // Apply: a·sin²+b·cos²+c = (a-b)·sin²+(b+c)
                    let a_minus_b = sin_coef - cos_coef;
                    let b_plus_c = cos_coef + res_coef;

                    // Only proceed if b+c = 0 (reduces to single term) or a-b = 0 (reduces to constant)
                    if !b_plus_c.is_zero() && !a_minus_b.is_zero() {
                        continue;
                    }

                    // Build result
                    let mut new_terms: Vec<ExprId> = Vec::new();
                    for (k, &t) in terms.iter().enumerate() {
                        if k != *sin_idx && k != *cos_idx && k != *res_idx {
                            new_terms.push(t);
                        }
                    }

                    if !a_minus_b.is_zero() {
                        let sin_t = ctx.call("sin", vec![*sin_arg]);
                        let two = ctx.num(2);
                        let sin_sq = ctx.add(Expr::Pow(sin_t, two));

                        let residual = if sin_residual.is_empty() {
                            sin_sq
                        } else {
                            let mut r = sin_residual[0];
                            for &f in sin_residual.iter().skip(1) {
                                r = ctx.add(Expr::Mul(r, f));
                            }
                            ctx.add(Expr::Mul(r, sin_sq))
                        };

                        let result_term = if a_minus_b.is_one() {
                            residual
                        } else if a_minus_b == -BigRational::one() {
                            ctx.add(Expr::Neg(residual))
                        } else {
                            let coef_expr = ctx.add(Expr::Number(a_minus_b.clone()));
                            ctx.add(Expr::Mul(coef_expr, residual))
                        };
                        new_terms.push(result_term);
                    }

                    let result = if new_terms.is_empty() {
                        ctx.num(0)
                    } else if new_terms.len() == 1 {
                        new_terms[0]
                    } else {
                        let mut acc = new_terms[0];
                        for &t in new_terms.iter().skip(1) {
                            acc = ctx.add(Expr::Add(acc, t));
                        }
                        acc
                    };

                    return Some(Rewrite::new(result).desc("k·R·sin²+R·cos²-R = (k-1)·R·sin²"));
                }
            }
        }

        None
    }
);

/// Multi-candidate version: returns ALL possible (is_sin, arg, numeric_coef, residual_factors)
/// decompositions from a term with potentially multiple trig² factors.
/// For example, `2·cos(x)²·sin(u)²` generates:
///   - (false, x, 2, [sin(u)²]) - treating cos(x)² as the main trig
///   - (true, u, 2, [cos(x)²]) - treating sin(u)² as the main trig
fn decompose_term_with_residual_multi(
    ctx: &Context,
    term: ExprId,
) -> Vec<(bool, ExprId, num_rational::BigRational, Vec<ExprId>)> {
    use num_rational::BigRational;
    use num_traits::One;

    let mut factors = Vec::new();
    let mut stack = vec![term];
    let mut is_negated = false;

    if let Expr::Neg(inner) = ctx.get(term) {
        is_negated = true;
        stack = vec![*inner];
    }

    while let Some(curr) = stack.pop() {
        match ctx.get(curr) {
            Expr::Mul(l, r) => {
                stack.push(*r);
                stack.push(*l);
            }
            Expr::Neg(inner) => {
                is_negated = !is_negated;
                stack.push(*inner);
            }
            _ => factors.push(curr),
        }
    }

    // Find ALL trig² factors (sin²/cos² with any argument)
    let mut trig_indices: Vec<(usize, bool, ExprId)> = Vec::new(); // (index, is_sin, arg)

    for (i, &f) in factors.iter().enumerate() {
        if let Expr::Pow(base, exp) = ctx.get(f) {
            if let Expr::Number(n) = ctx.get(*exp) {
                if *n == BigRational::from_integer(2.into()) {
                    if let Expr::Function(fn_id, args) = ctx.get(*base) { let name = ctx.sym_name(*fn_id);
                        if args.len() == 1 && (name == "sin" || name == "cos") {
                            trig_indices.push((i, name == "sin", args[0]));
                        }
                    }
                }
            }
        }
    }

    // Generate one candidate for each trig² factor
    let mut results = Vec::new();
    let base_numeric_coef: BigRational = if is_negated {
        -BigRational::one()
    } else {
        BigRational::one()
    };

    for (trig_idx, is_sin, arg) in trig_indices {
        let mut numeric_coef = base_numeric_coef.clone();
        let mut residual: Vec<ExprId> = Vec::new();

        for (i, &f) in factors.iter().enumerate() {
            if i == trig_idx {
                continue;
            }
            if let Expr::Number(n) = ctx.get(f) {
                numeric_coef *= n.clone();
            } else {
                residual.push(f);
            }
        }

        results.push((is_sin, arg, numeric_coef, residual));
    }

    // Cap at 6 candidates to avoid blow-up
    results.truncate(6);
    results
}

/// Extract term as product of factors with numeric coefficient
fn extract_as_product(
    ctx: &Context,
    term: ExprId,
) -> Option<(Vec<ExprId>, num_rational::BigRational)> {
    use num_rational::BigRational;
    use num_traits::One;

    let mut factors = Vec::new();
    let mut stack = vec![term];
    let mut is_negated = false;

    if let Expr::Neg(inner) = ctx.get(term) {
        is_negated = true;
        stack = vec![*inner];
    }

    while let Some(curr) = stack.pop() {
        match ctx.get(curr) {
            Expr::Mul(l, r) => {
                stack.push(*r);
                stack.push(*l);
            }
            Expr::Neg(inner) => {
                is_negated = !is_negated;
                stack.push(*inner);
            }
            _ => factors.push(curr),
        }
    }

    let mut numeric_coef = BigRational::one();
    let mut non_numeric: Vec<ExprId> = Vec::new();

    for &f in &factors {
        if let Expr::Number(n) = ctx.get(f) {
            numeric_coef *= n.clone();
        } else {
            non_numeric.push(f);
        }
    }

    if is_negated {
        numeric_coef = -numeric_coef;
    }

    Some((non_numeric, numeric_coef))
}

///
/// Handles: sin(t)^2, cos(t)^2, k*sin(t)^2, k*cos(t)^2
fn extract_trig_squared(
    ctx: &Context,
    term: ExprId,
) -> Option<(String, ExprId, num_rational::BigRational)> {
    use num_rational::BigRational;
    use num_traits::One;

    let mut coef = BigRational::one();
    let mut working = term;

    // Handle Neg wrapper: -sin²(t) has coefficient -1
    if let Expr::Neg(inner) = ctx.get(term) {
        coef = -coef;
        working = *inner;
    }

    // Check direct Pow(trig, 2)
    if let Expr::Pow(base, exp) = ctx.get(working) {
        if let Expr::Number(n) = ctx.get(*exp) {
            if *n == BigRational::from_integer(2.into()) {
                if let Expr::Function(fn_id, args) = ctx.get(*base) { let name = ctx.sym_name(*fn_id);
                    if (name == "sin" || name == "cos") && args.len() == 1 {
                        return Some((name.clone(), args[0], coef));
                    }
                }
            }
        }
    }

    // Check Mul(k, Pow(trig, 2)) or Mul(Pow(trig, 2), k)
    if let Expr::Mul(l, r) = ctx.get(working) {
        // Try both orderings
        for (maybe_coef, maybe_pow) in [(*l, *r), (*r, *l)] {
            if let Expr::Number(n) = ctx.get(maybe_coef) {
                if let Expr::Pow(base, exp) = ctx.get(maybe_pow) {
                    if let Expr::Number(e) = ctx.get(*exp) {
                        if *e == BigRational::from_integer(2.into()) {
                            if let Expr::Function(fn_id, args) = ctx.get(*base) { let name = ctx.sym_name(*fn_id);
                                if (name == "sin" || name == "cos") && args.len() == 1 {
                                    return Some((name.clone(), args[0], coef * n.clone()));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

// =============================================================================
// RecognizeSecSquaredRule: 1 + tan²(x) → sec²(x) (contraction to canonical form)
// =============================================================================
// This is the CANONICAL direction - contracting to sec² is "simpler" (fewer nodes).
// The reverse (expansion) should NOT be done in generic mode to avoid worsen.

define_rule!(
    RecognizeSecSquaredRule,
    "Recognize Secant Squared",
    |ctx, expr| {
        // Flatten the additive chain
        let mut terms = Vec::new();
        crate::helpers::flatten_add(ctx, expr, &mut terms);

        if terms.len() < 2 {
            return None;
        }

        // Look for pattern: 1 + tan²(x)
        let mut one_idx: Option<usize> = None;
        let mut tan2_idx: Option<usize> = None;
        let mut tan_arg: Option<ExprId> = None;

        for (i, &term) in terms.iter().enumerate() {
            // Check for literal 1
            if let Expr::Number(n) = ctx.get(term) {
                if *n == num_rational::BigRational::from_integer(1.into()) {
                    one_idx = Some(i);
                    continue;
                }
            }
            // Check for tan²(x)
            if let Some((func_name, arg, coef)) = extract_tan_or_cot_squared(ctx, term) {
                let one = num_rational::BigRational::from_integer(1.into());
                if func_name == "tan" && coef == one {
                    tan2_idx = Some(i);
                    tan_arg = Some(arg);
                }
            }
        }

        // If we found both 1 and tan²(x), replace with sec²(x)
        if let (Some(one_i), Some(tan_i), Some(arg)) = (one_idx, tan2_idx, tan_arg) {
            let sec_func = ctx.call("sec", vec![arg]);
            let two = ctx.num(2);
            let sec_squared = ctx.add(Expr::Pow(sec_func, two));

            // Build new expression with the pair removed and sec²(x) added
            let mut new_terms: Vec<ExprId> = Vec::new();
            for (j, &t) in terms.iter().enumerate() {
                if j != one_i && j != tan_i {
                    new_terms.push(t);
                }
            }
            new_terms.push(sec_squared);

            // Build result as sum
            let result = if new_terms.len() == 1 {
                new_terms[0]
            } else {
                let mut acc = new_terms[0];
                for &t in new_terms.iter().skip(1) {
                    acc = ctx.add(Expr::Add(acc, t));
                }
                acc
            };

            return Some(Rewrite::new(result).desc("1 + tan²(x) = sec²(x)"));
        }

        None
    }
);

// =============================================================================
// RecognizeCscSquaredRule: 1 + cot²(x) → csc²(x) (contraction to canonical form)
// =============================================================================

define_rule!(
    RecognizeCscSquaredRule,
    "Recognize Cosecant Squared",
    |ctx, expr| {
        // Flatten the additive chain
        let mut terms = Vec::new();
        crate::helpers::flatten_add(ctx, expr, &mut terms);

        if terms.len() < 2 {
            return None;
        }

        // Look for pattern: 1 + cot²(x)
        let mut one_idx: Option<usize> = None;
        let mut cot2_idx: Option<usize> = None;
        let mut cot_arg: Option<ExprId> = None;

        for (i, &term) in terms.iter().enumerate() {
            // Check for literal 1
            if let Expr::Number(n) = ctx.get(term) {
                if *n == num_rational::BigRational::from_integer(1.into()) {
                    one_idx = Some(i);
                    continue;
                }
            }
            // Check for cot²(x)
            if let Some((func_name, arg, coef)) = extract_tan_or_cot_squared(ctx, term) {
                let one = num_rational::BigRational::from_integer(1.into());
                if func_name == "cot" && coef == one {
                    cot2_idx = Some(i);
                    cot_arg = Some(arg);
                }
            }
        }

        // If we found both 1 and cot²(x), replace with csc²(x)
        if let (Some(one_i), Some(cot_i), Some(arg)) = (one_idx, cot2_idx, cot_arg) {
            let csc_func = ctx.call("csc", vec![arg]);
            let two = ctx.num(2);
            let csc_squared = ctx.add(Expr::Pow(csc_func, two));

            // Build new expression with the pair removed and csc²(x) added
            let mut new_terms: Vec<ExprId> = Vec::new();
            for (j, &t) in terms.iter().enumerate() {
                if j != one_i && j != cot_i {
                    new_terms.push(t);
                }
            }
            new_terms.push(csc_squared);

            // Build result as sum
            let result = if new_terms.len() == 1 {
                new_terms[0]
            } else {
                let mut acc = new_terms[0];
                for &t in new_terms.iter().skip(1) {
                    acc = ctx.add(Expr::Add(acc, t));
                }
                acc
            };

            return Some(Rewrite::new(result).desc("1 + cot²(x) = csc²(x)"));
        }

        None
    }
);

/// Extract (function_name, argument, coefficient) from tan²(t) or cot²(t) terms.
fn extract_tan_or_cot_squared(
    ctx: &Context,
    term: ExprId,
) -> Option<(String, ExprId, num_rational::BigRational)> {
    use num_rational::BigRational;
    use num_traits::One;

    let mut coef = BigRational::one();
    let mut working = term;

    // Handle Neg wrapper
    if let Expr::Neg(inner) = ctx.get(term) {
        coef = -coef;
        working = *inner;
    }

    // Check direct Pow(trig, 2)
    if let Expr::Pow(base, exp) = ctx.get(working) {
        if let Expr::Number(n) = ctx.get(*exp) {
            if *n == BigRational::from_integer(2.into()) {
                if let Expr::Function(fn_id, args) = ctx.get(*base) { let name = ctx.sym_name(*fn_id);
                    if (name == "tan" || name == "cot") && args.len() == 1 {
                        return Some((name.clone(), args[0], coef));
                    }
                }
            }
        }
    }

    // Check Mul(k, Pow(trig, 2)) or Mul(Pow(trig, 2), k)
    if let Expr::Mul(l, r) = ctx.get(working) {
        for (maybe_coef, maybe_pow) in [(*l, *r), (*r, *l)] {
            if let Expr::Number(n) = ctx.get(maybe_coef) {
                if let Expr::Pow(base, exp) = ctx.get(maybe_pow) {
                    if let Expr::Number(e) = ctx.get(*exp) {
                        if *e == BigRational::from_integer(2.into()) {
                            if let Expr::Function(fn_id, args) = ctx.get(*base) { let name = ctx.sym_name(*fn_id);
                                if (name == "tan" || name == "cot") && args.len() == 1 {
                                    return Some((name.clone(), args[0], coef * n.clone()));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

// =============================================================================
// SecToRecipCosRule: sec(x) → 1/cos(x) (canonical expansion)
// =============================================================================
// This ensures sec unifies with 1/cos forms from tan²+1 simplification.

define_rule!(
    SecToRecipCosRule,
    "Secant to Reciprocal Cosine",
    |ctx, expr| {
        if let Expr::Function(fn_id, args) = ctx.get(expr) { let name = ctx.sym_name(*fn_id);
            if name == "sec" && args.len() == 1 {
                let arg = args[0];
                let cos_func = ctx.call("cos", vec![arg]);
                let one = ctx.num(1);
                let result = ctx.add(Expr::Div(one, cos_func));
                return Some(Rewrite::new(result).desc("sec(x) = 1/cos(x)"));
            }
        }
        None
    }
);

// =============================================================================
// CscToRecipSinRule: csc(x) → 1/sin(x) (canonical expansion)
// =============================================================================

define_rule!(
    CscToRecipSinRule,
    "Cosecant to Reciprocal Sine",
    |ctx, expr| {
        if let Expr::Function(fn_id, args) = ctx.get(expr) { let name = ctx.sym_name(*fn_id);
            if name == "csc" && args.len() == 1 {
                let arg = args[0];
                let sin_func = ctx.call("sin", vec![arg]);
                let one = ctx.num(1);
                let result = ctx.add(Expr::Div(one, sin_func));
                return Some(Rewrite::new(result).desc("csc(x) = 1/sin(x)"));
            }
        }
        None
    }
);

// =============================================================================
// CotToCosSinRule: cot(x) → cos(x)/sin(x) (canonical expansion)
// =============================================================================
// This ensures cot unifies with cos/sin forms for csc²-cot²=1 simplification.

define_rule!(
    CotToCosSinRule,
    "Cotangent to Cosine over Sine",
    |ctx, expr| {
        if let Expr::Function(fn_id, args) = ctx.get(expr) { let name = ctx.sym_name(*fn_id);
            if name == "cot" && args.len() == 1 {
                let arg = args[0];
                let cos_func = ctx.call("cos", vec![arg]);
                let sin_func = ctx.call("sin", vec![arg]);
                let result = ctx.add(Expr::Div(cos_func, sin_func));
                return Some(Rewrite::new(result).desc("cot(x) = cos(x)/sin(x)"));
            }
        }
        None
    }
);

/// Check if (c_term, trig_term) matches the pattern k - k*trig²(x) = k*other²(x)
/// where trig is sin or cos, and other is the complementary function.
fn check_pythagorean_pattern(
    ctx: &mut Context,
    c_term: ExprId,
    t_term: ExprId,
) -> Option<(ExprId, String)> {
    // Parse t_term for k * trig^2 (possibly negated)
    let (base_term, is_neg) = if let Expr::Neg(inner) = ctx.get(t_term) {
        (*inner, true)
    } else {
        (t_term, false)
    };

    // Flatten multiplication factors
    let mut factors = Vec::new();
    let mut stack = vec![base_term];
    while let Some(curr) = stack.pop() {
        if let Expr::Mul(l, r) = ctx.get(curr) {
            stack.push(*r);
            stack.push(*l);
        } else {
            factors.push(curr);
        }
    }

    // Find trig²(x) factor
    let mut trig_idx = None;
    let mut func_name = String::new();
    let mut arg = None;

    for (i, &f) in factors.iter().enumerate() {
        if let Expr::Pow(b, exp) = ctx.get(f) {
            if let Expr::Number(n) = ctx.get(*exp) {
                if *n == num_rational::BigRational::from_integer(2.into()) {
                    if let Expr::Function(fn_id, args) = ctx.get(*b) { let name = ctx.sym_name(*fn_id);
                        if (name == "sin" || name == "cos") && args.len() == 1 {
                            trig_idx = Some(i);
                            func_name = name.clone();
                            arg = Some(args[0]);
                            break;
                        }
                    }
                }
            }
        }
    }

    let idx = trig_idx?;
    let arg = arg?;

    // Collect coefficient factors (excluding trig²)
    let mut coeff_factors = Vec::new();
    if is_neg {
        coeff_factors.push(ctx.num(-1));
    }
    for (i, &f) in factors.iter().enumerate() {
        if i != idx {
            coeff_factors.push(f);
        }
    }

    // Build coefficient
    let coeff = if coeff_factors.is_empty() {
        ctx.num(1)
    } else {
        let mut c = coeff_factors[0];
        for &f in coeff_factors.iter().skip(1) {
            c = smart_mul(ctx, c, f);
        }
        c
    };

    // The pattern is: c_term + (-coeff * trig²) where c_term == -(-coeff) = coeff
    // So we need c_term == -coeff (negative of the coefficient)
    let neg_coeff = if let Expr::Number(n) = ctx.get(coeff) {
        ctx.add(Expr::Number(-n.clone()))
    } else if let Expr::Neg(inner) = ctx.get(coeff) {
        *inner
    } else {
        ctx.add(Expr::Neg(coeff))
    };

    if crate::ordering::compare_expr(ctx, c_term, neg_coeff) == std::cmp::Ordering::Equal {
        // Pattern matches! Apply identity: k - k*sin² = k*cos², k - k*cos² = k*sin²
        let other_name = if func_name == "sin" { "cos" } else { "sin" };
        let func = ctx.add(Expr::Function(other_name.to_string(), vec![arg]));
        let two = ctx.num(2);
        let pow = ctx.add(Expr::Pow(func, two));
        let result = smart_mul(ctx, c_term, pow);

        let desc = format!("1 - {}²(x) = {}²(x)", func_name, other_name);

        return Some((result, desc));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::SimpleRule;

    #[test]
    fn test_one_minus_sin_squared() {
        let mut ctx = Context::new();
        // 1 - sin²(x) should become cos²(x)
        let x = ctx.var("x");
        let sin_x = ctx.call("sin", vec![x]);
        let two = ctx.num(2);
        let sin_sq = ctx.add(Expr::Pow(sin_x, two));
        let neg_sin_sq = ctx.add(Expr::Neg(sin_sq));
        let one = ctx.num(1);
        let expr = ctx.add(Expr::Add(one, neg_sin_sq)); // 1 + (-sin²(x))

        let rule = TrigPythagoreanSimplifyRule;
        let result = rule.apply_simple(&mut ctx, expr);

        assert!(result.is_some(), "Rule should apply to 1 - sin²(x)");
        let rewrite = result.unwrap();

        // Result should be cos²(x)
        if let Expr::Mul(_coeff, pow) = ctx.get(rewrite.new_expr) {
            if let Expr::Pow(base, _) = ctx.get(*pow) {
                if let Expr::Function(name, _) = ctx.get(*base) {
                    assert_eq!(name, "cos");
                }
            }
        }
    }
}

// ============================================================================
// TrigEvenPowerDifferenceRule
// ============================================================================
// Detects pairs like k*sin⁴(u) + (-k)*cos⁴(u) in Add expressions and reduces
// them using the factorization: sin⁴ - cos⁴ = (sin² - cos²)(sin² + cos²) = sin² - cos²
//
// This is a degree-reducing rule (4 → 2) so it cannot loop.

define_rule!(
    TrigEvenPowerDifferenceRule,
    "Trig Fourth Power Difference",
    |ctx, expr| {
        // Collect all additive terms
        let mut terms = Vec::new();
        crate::helpers::flatten_add(ctx, expr, &mut terms);

        if terms.len() < 2 {
            return None;
        }

        // For each term, try to extract (coefficient, trig_func, argument, power)
        // We're looking for sin⁴(u) or cos⁴(u) with coefficients
        let mut sin4_terms: Vec<(num_rational::BigRational, ExprId, usize)> = Vec::new(); // (coef, arg, term_index)
        let mut cos4_terms: Vec<(num_rational::BigRational, ExprId, usize)> = Vec::new();

        for (i, &term) in terms.iter().enumerate() {
            if let Some((coef, func_name, arg)) = extract_trig_fourth_power(ctx, term) {
                if func_name == "sin" {
                    sin4_terms.push((coef, arg, i));
                } else if func_name == "cos" {
                    cos4_terms.push((coef, arg, i));
                }
            }
        }

        // Find matching pairs: same argument, opposite coefficients
        for (sin_coef, sin_arg, sin_idx) in sin4_terms.iter() {
            for (cos_coef, cos_arg, cos_idx) in cos4_terms.iter() {
                // Same argument?
                if crate::ordering::compare_expr(ctx, *sin_arg, *cos_arg)
                    != std::cmp::Ordering::Equal
                {
                    continue;
                }

                // Opposite coefficients? sin_coef = -cos_coef
                let sum = sin_coef.clone() + cos_coef.clone();
                if !sum.is_zero() {
                    continue;
                }

                // Found a match! k*sin⁴(u) + (-k)*cos⁴(u) → k*(sin²(u) - cos²(u))
                // Build replacement: coef * (sin²(arg) - cos²(arg))
                let arg = *sin_arg;
                let coef = sin_coef.clone();

                let sin_func = ctx.call("sin", vec![arg]);
                let cos_func = ctx.call("cos", vec![arg]);
                let two = ctx.num(2);
                let sin_sq = ctx.add(Expr::Pow(sin_func, two));
                let cos_sq = ctx.add(Expr::Pow(cos_func, two));
                let diff = ctx.add(Expr::Sub(sin_sq, cos_sq));

                // Apply coefficient
                let replacement = if coef == num_rational::BigRational::from_integer(1.into()) {
                    diff
                } else if coef == num_rational::BigRational::from_integer((-1).into()) {
                    ctx.add(Expr::Neg(diff))
                } else {
                    let coef_expr = ctx.add(Expr::Number(coef));
                    ctx.add(Expr::Mul(coef_expr, diff))
                };

                // Build new expression: remove the two matched terms, add replacement
                let mut new_terms: Vec<ExprId> = Vec::new();
                for (j, &t) in terms.iter().enumerate() {
                    if j != *sin_idx && j != *cos_idx {
                        new_terms.push(t);
                    }
                }
                new_terms.push(replacement);

                // Build result as sum
                let result = if new_terms.len() == 1 {
                    new_terms[0]
                } else {
                    let mut acc = new_terms[0];
                    for &t in new_terms.iter().skip(1) {
                        acc = ctx.add(Expr::Add(acc, t));
                    }
                    acc
                };

                return Some(Rewrite::new(result).desc("sin⁴(x) - cos⁴(x) = sin²(x) - cos²(x)"));
            }
        }

        None
    }
);

/// Extract (coefficient, trig_function_name, argument) from a term like k * sin(u)^4
/// Returns None if the term doesn't match this pattern.
fn extract_trig_fourth_power(
    ctx: &Context,
    term: ExprId,
) -> Option<(num_rational::BigRational, String, ExprId)> {
    use num_rational::BigRational;
    use num_traits::One;

    let mut coef = BigRational::one();
    let mut working = term;

    // Handle Neg wrapper
    if let Expr::Neg(inner) = ctx.get(term) {
        coef = -coef;
        working = *inner;
    }

    // Flatten multiplication
    let mut factors = Vec::new();
    let mut stack = vec![working];
    while let Some(curr) = stack.pop() {
        match ctx.get(curr) {
            Expr::Mul(l, r) => {
                stack.push(*r);
                stack.push(*l);
            }
            _ => factors.push(curr),
        }
    }

    // Look for trig⁴(arg) and extract numeric coefficients
    let mut trig_func_name = None;
    let mut trig_arg = None;
    let mut trig_idx = None;

    for (i, &f) in factors.iter().enumerate() {
        // Check for Number (accumulate into coefficient)
        if let Expr::Number(n) = ctx.get(f) {
            coef *= n.clone();
            continue;
        }

        // Check for Pow(trig_func, 4)
        if let Expr::Pow(base, exp) = ctx.get(f) {
            if let Expr::Number(n) = ctx.get(*exp) {
                if *n == BigRational::from_integer(4.into()) {
                    if let Expr::Function(fn_id, args) = ctx.get(*base) { let name = ctx.sym_name(*fn_id);
                        if (name == "sin" || name == "cos") && args.len() == 1 {
                            trig_func_name = Some(name.clone());
                            trig_arg = Some(args[0]);
                            trig_idx = Some(i);
                        }
                    }
                }
            }
        }
    }

    // Verify we found exactly one trig⁴ and possibly some numeric factors
    let func_name = trig_func_name?;
    let arg = trig_arg?;
    let idx = trig_idx?;

    // Verify remaining factors are all numbers (already accumulated into coef)
    for (i, &f) in factors.iter().enumerate() {
        if i == idx {
            continue;
        }
        if !matches!(ctx.get(f), Expr::Number(_)) {
            // Non-numeric factor that isn't the trig⁴, pattern doesn't match cleanly
            return None;
        }
    }

    Some((coef, func_name, arg))
}
