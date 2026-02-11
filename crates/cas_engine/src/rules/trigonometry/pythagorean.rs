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
        let terms = crate::nary::add_leaves(ctx, expr);

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

/// Extract ALL possible (is_sin, argument, coefficient_factors) candidates from a term.
/// For a term like cos(u)²*sin(x)², returns TWO candidates:
///   1. (true, x, [cos(u)²])   -- interpreting sin(x)² as the trig, cos(u)² as coef
///   2. (false, u, [sin(x)²])  -- interpreting cos(u)² as the trig, sin(x)² as coef
///
/// Also decomposes trig^n (n ≥ 3) into trig^(n-2) · trig², so cos³(u) emits
/// candidate (false, u, [cos(u)]) — enabling Pythagorean matches on higher powers.
fn extract_all_trig_squared_candidates(
    ctx: &mut Context,
    term: ExprId,
) -> Vec<(bool, ExprId, Vec<ExprId>)> {
    let mut results = Vec::new();

    // Case 1: Direct trig^n (no multiplication coefficient)
    // Extract all needed info from ctx.get() first, then call ctx.add() separately
    {
        let mut case1_info = None; // (is_sin, arg, base_id, remainder_exp_opt)
        if let Expr::Pow(base, exp) = ctx.get(term) {
            let base = *base;
            let exp = *exp;
            if let Expr::Number(n) = ctx.get(exp) {
                if let Expr::Function(fn_id, args) = ctx.get(base) {
                    let builtin = ctx.builtin_of(*fn_id);
                    if args.len() == 1
                        && matches!(
                            builtin,
                            Some(cas_ast::BuiltinFn::Sin | cas_ast::BuiltinFn::Cos)
                        )
                    {
                        let is_sin = matches!(builtin, Some(cas_ast::BuiltinFn::Sin));
                        let arg = args[0];
                        let two = num_rational::BigRational::from_integer(2.into());
                        if *n == two {
                            case1_info = Some((is_sin, arg, base, None));
                        } else if *n > two && n.is_integer() {
                            let remainder = n - &two;
                            case1_info = Some((is_sin, arg, base, Some(remainder)));
                        }
                    }
                }
            }
        }

        if let Some((is_sin, arg, base, remainder_opt)) = case1_info {
            match remainder_opt {
                None => {
                    results.push((is_sin, arg, vec![]));
                }
                Some(remainder) => {
                    let one = num_rational::BigRational::from_integer(1.into());
                    let leftover = if remainder == one {
                        base // trig^3 → trig · trig², leftover is just trig
                    } else {
                        let remainder_id = ctx.add(Expr::Number(remainder));
                        ctx.add(Expr::Pow(base, remainder_id))
                    };
                    results.push((is_sin, arg, vec![leftover]));
                }
            }
            return results;
        }
        // If we matched Pow but no trig was found, still return (no candidates from non-trig Pow)
        if matches!(ctx.get(term), Expr::Pow(_, _)) {
            return results;
        }
    }

    // Case 2: Mul - flatten and find ALL sin²/cos² factors (including higher powers)
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

        // First pass: collect all candidate info without mutating ctx
        struct CandidateInfo {
            factor_idx: usize,
            is_sin: bool,
            arg: ExprId,
            base: ExprId,
            remainder: Option<num_rational::BigRational>, // None = exact trig²
        }
        let mut candidates_info: Vec<CandidateInfo> = Vec::new();

        for (i, &f) in factors.iter().enumerate() {
            if let Expr::Pow(base, exp) = ctx.get(f) {
                let base = *base;
                let exp = *exp;
                if let Expr::Number(n) = ctx.get(exp) {
                    if let Expr::Function(fn_id, args) = ctx.get(base) {
                        let builtin = ctx.builtin_of(*fn_id);
                        if args.len() == 1
                            && matches!(
                                builtin,
                                Some(cas_ast::BuiltinFn::Sin | cas_ast::BuiltinFn::Cos)
                            )
                        {
                            let is_sin = matches!(builtin, Some(cas_ast::BuiltinFn::Sin));
                            let arg = args[0];
                            let two = num_rational::BigRational::from_integer(2.into());
                            if *n == two {
                                candidates_info.push(CandidateInfo {
                                    factor_idx: i,
                                    is_sin,
                                    arg,
                                    base,
                                    remainder: None,
                                });
                            } else if *n > two && n.is_integer() {
                                let rem = n - &two;
                                candidates_info.push(CandidateInfo {
                                    factor_idx: i,
                                    is_sin,
                                    arg,
                                    base,
                                    remainder: Some(rem),
                                });
                            }
                        }
                    }
                }
            }
        }

        // Second pass: build results, calling ctx.add() only for higher-power decompositions
        for info in candidates_info {
            let mut coef_factors: Vec<ExprId> = factors
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != info.factor_idx)
                .map(|(_, &g)| g)
                .collect();

            if let Some(remainder) = info.remainder {
                let one = num_rational::BigRational::from_integer(1.into());
                let leftover = if remainder == one {
                    info.base // trig^3 → trig · trig², leftover is just trig
                } else {
                    let remainder_id = ctx.add(Expr::Number(remainder));
                    ctx.add(Expr::Pow(info.base, remainder_id))
                };
                coef_factors.push(leftover);
            }

            results.push((info.is_sin, info.arg, coef_factors));
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
                    if let Expr::Function(fn_id, args) = ctx.get(*base) {
                        let builtin = ctx.builtin_of(*fn_id);
                        if args.len() == 1
                            && matches!(
                                builtin,
                                Some(cas_ast::BuiltinFn::Sin | cas_ast::BuiltinFn::Cos)
                            )
                        {
                            trig_indices.push((
                                i,
                                matches!(builtin, Some(cas_ast::BuiltinFn::Sin)),
                                args[0],
                            ));
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
                if let Expr::Function(fn_id, args) = ctx.get(*base) {
                    let builtin = ctx.builtin_of(*fn_id);
                    if matches!(
                        builtin,
                        Some(cas_ast::BuiltinFn::Sin | cas_ast::BuiltinFn::Cos)
                    ) && args.len() == 1
                    {
                        return Some((builtin.unwrap().name().to_string(), args[0], coef));
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
                            if let Expr::Function(fn_id, args) = ctx.get(*base) {
                                let builtin = ctx.builtin_of(*fn_id);
                                if matches!(
                                    builtin,
                                    Some(cas_ast::BuiltinFn::Sin | cas_ast::BuiltinFn::Cos)
                                ) && args.len() == 1
                                {
                                    return Some((
                                        builtin.unwrap().name().to_string(),
                                        args[0],
                                        coef * n.clone(),
                                    ));
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
                if let Expr::Function(fn_id, args) = ctx.get(*base) {
                    let builtin = ctx.builtin_of(*fn_id);
                    if matches!(
                        builtin,
                        Some(cas_ast::BuiltinFn::Tan | cas_ast::BuiltinFn::Cot)
                    ) && args.len() == 1
                    {
                        return Some((builtin.unwrap().name().to_string(), args[0], coef));
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
                            if let Expr::Function(fn_id, args) = ctx.get(*base) {
                                let builtin = ctx.builtin_of(*fn_id);
                                if matches!(
                                    builtin,
                                    Some(cas_ast::BuiltinFn::Tan | cas_ast::BuiltinFn::Cot)
                                ) && args.len() == 1
                                {
                                    return Some((
                                        builtin.unwrap().name().to_string(),
                                        args[0],
                                        coef * n.clone(),
                                    ));
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
            if let Some(result) = try_high_power_pythagorean(ctx, small_term, big_term) {
                return Some(result);
            }
        }

        None
    }
);

/// Try to match: small_term + big_term where big_term = −small_term · trig²(x)
/// If matched, rewrite as small_term · other²(x)
fn try_high_power_pythagorean(
    ctx: &mut Context,
    small_term: ExprId,
    big_term: ExprId,
) -> Option<Rewrite> {
    use num_rational::BigRational;

    // Flatten both terms into (is_negated, sorted_factors) where factors are atomic
    // For trig^n, decompose into trig^(n-2) and trig² separately
    let (small_neg, small_factors) = flatten_with_trig_decomp(ctx, small_term);
    let (big_neg, big_factors) = flatten_with_trig_decomp(ctx, big_term);

    // We need opposite signs: small + big = R − R·trig² → R·other²
    if small_neg == big_neg {
        return None; // Same sign, can't cancel
    }

    // big_factors should be a superset of small_factors + one extra trig² factor
    if big_factors.len() != small_factors.len() + 1 {
        return None;
    }

    // Sort both factor lists for comparison
    let mut small_sorted: Vec<ExprId> = small_factors.clone();
    let mut big_sorted: Vec<ExprId> = big_factors.clone();
    small_sorted.sort_by(|a, b| crate::ordering::compare_expr(ctx, *a, *b));
    big_sorted.sort_by(|a, b| crate::ordering::compare_expr(ctx, *a, *b));

    // Find the one extra factor in big that's not in small
    let mut extra_factor = None;
    let mut si = 0;
    let mut bi = 0;
    let mut mismatches = 0;

    while bi < big_sorted.len() {
        if si < small_sorted.len()
            && crate::ordering::compare_expr(ctx, small_sorted[si], big_sorted[bi])
                == std::cmp::Ordering::Equal
        {
            si += 1;
            bi += 1;
        } else {
            // big has an extra factor
            mismatches += 1;
            if mismatches > 1 {
                return None;
            }
            extra_factor = Some(big_sorted[bi]);
            bi += 1;
        }
    }
    // Any remaining in small means they didn't match
    if si != small_sorted.len() {
        return None;
    }

    let extra = extra_factor?;

    // Check if excess factor is trig²(x)
    if let Expr::Pow(base, exp) = ctx.get(extra) {
        let base = *base;
        let exp = *exp;
        if let Expr::Number(n) = ctx.get(exp) {
            if *n != BigRational::from_integer(2.into()) {
                return None;
            }
            if let Expr::Function(fn_id, args) = ctx.get(base) {
                let builtin = ctx.builtin_of(*fn_id);
                if !matches!(
                    builtin,
                    Some(cas_ast::BuiltinFn::Sin | cas_ast::BuiltinFn::Cos)
                ) || args.len() != 1
                {
                    return None;
                }

                let trig_arg = args[0];
                let other_builtin = if matches!(builtin, Some(cas_ast::BuiltinFn::Sin)) {
                    cas_ast::BuiltinFn::Cos
                } else {
                    cas_ast::BuiltinFn::Sin
                };

                // Pattern confirmed! small_term + big_term = R + (−R·trig²(x))
                // Since we checked big has the extra trig² and signs differ:
                //   If small = +R, big = −R·trig² → sum = R(1−trig²) = +R·other²
                //   If small = −R, big = +R·trig² → sum = −R + R·trig² = R(trig²−1) = −R·other²
                // In both cases the result sign matches small_term's sign.
                // Use the original small_term as the R multiplier.

                let other_fn = ctx.call_builtin(other_builtin, vec![trig_arg]);
                let two = ctx.num(2);
                let other_sq = ctx.add(Expr::Pow(other_fn, two));

                let result = crate::rules::algebra::helpers::smart_mul(ctx, small_term, other_sq);

                let func_name = builtin.unwrap().name();
                let other_name = other_builtin.name();
                let desc = format!("R − R·{}²(x) = R·{}²(x)", func_name, other_name);
                return Some(Rewrite::new(result).desc(desc));
            }
        }
    }

    None
}

/// Flatten a term into (is_negated, factor_list).
/// For trig^n (n≥2), decompose into [trig^(n-2), trig²] to expose the trig² for matching.
fn flatten_with_trig_decomp(ctx: &mut Context, term: ExprId) -> (bool, Vec<ExprId>) {
    use num_rational::BigRational;
    use num_traits::{One, Signed};

    let mut is_neg = false;
    let mut factors = Vec::new();
    let mut stack = vec![term];

    // Handle outer Neg
    if let Expr::Neg(inner) = ctx.get(term) {
        is_neg = true;
        stack = vec![*inner];
    }

    // Flatten Mul tree
    while let Some(curr) = stack.pop() {
        match ctx.get(curr) {
            Expr::Mul(l, r) => {
                stack.push(*r);
                stack.push(*l);
            }
            Expr::Neg(inner) => {
                is_neg = !is_neg;
                stack.push(*inner);
            }
            _ => factors.push(curr),
        }
    }

    // Extract sign from negative numeric coefficients
    // e.g., Mul(-4, sin³(x)) should be is_neg=true with factor 4
    let mut final_factors = Vec::with_capacity(factors.len());
    for f in factors {
        if let Expr::Number(n) = ctx.get(f) {
            if n.is_negative() {
                is_neg = !is_neg;
                let abs_val = -n.clone();
                if abs_val == BigRational::one() {
                    // Factor of -1 → just flip sign, don't add factor 1
                    continue;
                }
                let abs_id = ctx.add(Expr::Number(abs_val));
                final_factors.push(abs_id);
                continue;
            }
        }
        final_factors.push(f);
    }
    let factors = final_factors;

    // Decompose any trig^n (n≥3) into trig^(n-2) + trig²
    let mut decomposed = Vec::new();
    for &f in &factors {
        if let Expr::Pow(base, exp) = ctx.get(f) {
            let base = *base;
            let exp = *exp;
            if let Expr::Number(n) = ctx.get(exp) {
                let two = BigRational::from_integer(2.into());
                if *n > two && n.is_integer() {
                    if let Expr::Function(fn_id, args) = ctx.get(base) {
                        let builtin = ctx.builtin_of(*fn_id);
                        if matches!(
                            builtin,
                            Some(cas_ast::BuiltinFn::Sin | cas_ast::BuiltinFn::Cos)
                        ) && args.len() == 1
                        {
                            // Decompose: trig^n → trig^(n-2) · trig²
                            let remainder = n - &two;
                            let leftover = if remainder == BigRational::one() {
                                base // trig^3 → trig · trig²
                            } else {
                                let rem_id = ctx.add(Expr::Number(remainder));
                                ctx.add(Expr::Pow(base, rem_id))
                            };
                            let two_id = ctx.num(2);
                            let trig_sq = ctx.add(Expr::Pow(base, two_id));
                            decomposed.push(leftover);
                            decomposed.push(trig_sq);
                            continue;
                        }
                    }
                }
            }
        }
        decomposed.push(f);
    }

    (is_neg, decomposed)
}

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
                    if let Expr::Function(fn_id, args) = ctx.get(*b) {
                        let builtin = ctx.builtin_of(*fn_id);
                        if matches!(
                            builtin,
                            Some(cas_ast::BuiltinFn::Sin | cas_ast::BuiltinFn::Cos)
                        ) && args.len() == 1
                        {
                            trig_idx = Some(i);
                            func_name = builtin.unwrap().name().to_string();
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
        let func = ctx.call(other_name, vec![arg]);
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
