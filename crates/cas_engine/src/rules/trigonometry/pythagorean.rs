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
use cas_ast::{Expr, ExprId};
use cas_math::trig_power_identity_support::{
    check_pythagorean_pattern, decompose_term_with_residual_multi,
    extract_all_trig_squared_candidates, extract_as_product, extract_coeff_tan_or_cot_pow2,
    extract_coeff_trig_pow2, try_high_power_pythagorean,
};

define_rule!(
    TrigPythagoreanSimplifyRule,
    "Pythagorean Factor Form",
    |ctx, expr| {
        // Only apply to Add expressions (Sub is represented as Add with Neg)

        // We need exactly 2 terms: constant and trig term
        let terms = crate::nary::add_leaves(ctx, expr);
        if terms.len() != 2 {
            return None;
        }

        let t1 = terms[0];
        let t2 = terms[1];

        // Try both orderings: (c_term, trig_term) or (trig_term, c_term)
        if let Some((result, trig, other)) = check_pythagorean_pattern(ctx, t1, t2) {
            let desc = format!("1 - {}²(x) = {}²(x)", trig.name(), other.name());
            return Some(Rewrite::new(result).desc(desc));
        }
        if let Some((result, trig, other)) = check_pythagorean_pattern(ctx, t2, t1) {
            let desc = format!("1 - {}²(x) = {}²(x)", trig.name(), other.name());
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
        let terms = crate::nary::add_leaves(ctx, expr);

        if terms.len() < 2 {
            return None;
        }

        // Collect sin² and cos² terms with their (argument, index, coefficient)
        let mut sin2_terms: Vec<(ExprId, usize, num_rational::BigRational)> = Vec::new();
        let mut cos2_terms: Vec<(ExprId, usize, num_rational::BigRational)> = Vec::new();

        for (i, &term) in terms.iter().enumerate() {
            if let Some((coef, func_name, arg)) = extract_coeff_trig_pow2(ctx, term) {
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
        let terms = crate::nary::add_leaves(ctx, expr);

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
        let terms = crate::nary::add_leaves(ctx, expr);

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
            if let Some((coef, func_name, arg)) = extract_coeff_trig_pow2(ctx, term) {
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
                    let sin_t = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![arg]);
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
        let terms = crate::nary::add_leaves(ctx, expr);

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
                        let sin_t = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![*sin_arg]);
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
        let terms = crate::nary::add_leaves(ctx, expr);

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
            if let Some((coef, func_name, arg)) = extract_coeff_tan_or_cot_pow2(ctx, term) {
                let one = num_rational::BigRational::from_integer(1.into());
                if func_name == "tan" && coef == one {
                    tan2_idx = Some(i);
                    tan_arg = Some(arg);
                }
            }
        }

        // If we found both 1 and tan²(x), replace with sec²(x)
        if let (Some(one_i), Some(tan_i), Some(arg)) = (one_idx, tan2_idx, tan_arg) {
            let sec_func = ctx.call_builtin(cas_ast::BuiltinFn::Sec, vec![arg]);
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
        let terms = crate::nary::add_leaves(ctx, expr);

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
            if let Some((coef, func_name, arg)) = extract_coeff_tan_or_cot_pow2(ctx, term) {
                let one = num_rational::BigRational::from_integer(1.into());
                if func_name == "cot" && coef == one {
                    cot2_idx = Some(i);
                    cot_arg = Some(arg);
                }
            }
        }

        // If we found both 1 and cot²(x), replace with csc²(x)
        if let (Some(one_i), Some(cot_i), Some(arg)) = (one_idx, cot2_idx, cot_arg) {
            let csc_func = ctx.call_builtin(cas_ast::BuiltinFn::Csc, vec![arg]);
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

// =============================================================================
// TrigPythagoreanHighPowerRule: R − R·trig²(x) → R·other²(x)
// =============================================================================
// Handles cases where trig² is embedded in a higher trig power, e.g.:
//   4·sin(x) − 4·sin³(x) → 4·cos²(x)·sin(x)
//   sin²(x) − sin²(x)·cos²(x) → sin⁴(x)
//
// Strategy: flatten both Add terms into multiplicative factor lists, decompose
// any trig^n (n≥2) into trig^(n-2)·trig², and check if the "bigger" term has
// exactly one extra trig² factor compared to the "smaller" one.

define_rule!(
    TrigPythagoreanHighPowerRule,
    "Pythagorean High-Power Factor",
    |ctx, expr| {
        if !matches!(ctx.get(expr), Expr::Add(_, _)) {
            return None;
        }
        let terms = crate::nary::add_leaves(ctx, expr);
        if terms.len() != 2 {
            return None;
        }

        // Try both orderings: (small, big) where big = small * trig²
        for (small_term, big_term) in [(terms[0], terms[1]), (terms[1], terms[0])] {
            if let Some((result, trig, other)) =
                try_high_power_pythagorean(ctx, small_term, big_term)
            {
                let desc = format!("R − R·{}²(x) = R·{}²(x)", trig.name(), other.name());
                return Some(Rewrite::new(result).desc(desc));
            }
        }

        None
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::SimpleRule;
    use cas_ast::Context;

    #[test]
    fn test_one_minus_sin_squared() {
        let mut ctx = Context::new();
        // 1 - sin²(x) should become cos²(x)
        let x = ctx.var("x");
        let sin_x = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![x]);
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
                if let Expr::Function(name_id, _) = ctx.get(*base) {
                    assert_eq!(ctx.sym_name(*name_id), "cos");
                }
            }
        }
    }
}
