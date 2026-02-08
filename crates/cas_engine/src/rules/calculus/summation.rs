//! Finite summation and product rules.
//!
//! Contains `SumRule`, `ProductRule`, and their helper functions:
//! - Telescoping sum/product detection
//! - Factorizable product detection
//! - Variable substitution

use crate::build::mul2_raw;
use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};

// =============================================================================
// SUM RULE: Evaluate finite summations
// =============================================================================
// Syntax: sum(expr, var, start, end)
// Example: sum(k, k, 1, 10) → 55
// Example: sum(k^2, k, 1, 5) → 1 + 4 + 9 + 16 + 25 = 55

define_rule!(SumRule, "Finite Summation", |ctx, expr| {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        let name = ctx.sym_name(*fn_id);
        if name == "sum" && args.len() == 4 {
            let summand = args[0];
            let var_expr = args[1];
            let start_expr = args[2];
            let end_expr = args[3];

            // Extract variable name
            let var_name = if let Expr::Variable(sym_id) = ctx.get(var_expr) {
                ctx.sym_name(*sym_id).to_string()
            } else {
                return None;
            };

            // =====================================================================
            // TELESCOPING DETECTION: Check for rational sums that telescope
            // =====================================================================
            // Pattern: 1/(k*(k+a)) = (1/a) * (1/k - 1/(k+a))
            // Telescoping sum: sum from m to n = (1/a) * (1/m - 1/(n+a))

            if let Some(result) =
                try_telescoping_rational_sum(ctx, summand, &var_name, start_expr, end_expr)
            {
                return Some(Rewrite::new(result).desc_lazy(|| {
                    format!(
                        "Telescoping sum: Σ({}, {}) from {} to {}",
                        cas_ast::DisplayExpr {
                            context: ctx,
                            id: summand
                        },
                        var_name,
                        cas_ast::DisplayExpr {
                            context: ctx,
                            id: start_expr
                        },
                        cas_ast::DisplayExpr {
                            context: ctx,
                            id: end_expr
                        }
                    )
                }));
            }

            // Try to evaluate start and end as integers for numeric evaluation
            let start = get_integer(ctx, start_expr);
            let end = get_integer(ctx, end_expr);

            if let (Some(start), Some(end)) = (start, end) {
                // Safety limit for direct evaluation
                if end - start > 1000 {
                    return None; // Too many terms
                }

                // Direct numeric evaluation
                if start <= end {
                    let mut result = ctx.num(0);
                    for k in start..=end {
                        let k_expr = ctx.num(k);
                        let term = substitute_var(ctx, summand, &var_name, k_expr);
                        result = ctx.add(Expr::Add(result, term));
                    }

                    // Simplify the result
                    let mut simplifier = crate::Simplifier::with_default_rules();
                    simplifier.context = ctx.clone();
                    let (simplified, _) = simplifier.simplify(result);
                    *ctx = simplifier.context;

                    return Some(Rewrite::new(simplified).desc_lazy(|| {
                        format!(
                            "sum({}, {}, {}, {})",
                            cas_ast::DisplayExpr {
                                context: ctx,
                                id: summand
                            },
                            var_name,
                            start,
                            end
                        )
                    }));
                }
            }
        }
    }
    None
});

/// Try to detect and evaluate telescoping rational sums
/// Pattern: 1/(k*(k+a)) where a is an integer constant
/// Result: (1/a) * (1/start - 1/(end+a))
fn try_telescoping_rational_sum(
    ctx: &mut Context,
    summand: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    // Check if summand is 1/(k*(k+a)) or 1/((k+b)*(k+c))
    let (num, den) = if let Expr::Div(num, den) = ctx.get(summand) {
        (*num, *den)
    } else {
        return None;
    };
    {
        // Numerator should be 1
        if let Expr::Number(n) = ctx.get(num) {
            if !n.is_one() {
                return None;
            }
        } else {
            return None;
        }
    }

    // Extract denominator structure: (k+b)*(k+c)
    let (factor1, factor2) = if let Expr::Mul(l, r) = ctx.get(den) {
        (*l, *r)
    } else {
        return None;
    };

    // Get linear offsets from each factor
    let offset1 = extract_linear_offset(ctx, factor1, var)?;
    let offset2 = extract_linear_offset(ctx, factor2, var)?;

    // The difference a = offset2 - offset1 is the telescoping shift
    let a = offset2 - offset1;
    if a == 0 {
        return None;
    }

    // Verify pattern is valid: we need 1/((k+offset1)*(k+offset2))
    // which equals (1/a) * (1/(k+offset1) - 1/(k+offset2)) for positive a

    {
        // Build result: (1/a) * (1/(start+offset1) - 1/(end+offset2))
        let a_expr = ctx.num(a.abs());

        // Build start + offset1
        let start_shifted = if offset1 == 0 {
            start
        } else {
            let offset = ctx.num(offset1);
            ctx.add(Expr::Add(start, offset))
        };

        // Build end + offset2
        let end_shifted = if offset2 == 0 {
            end
        } else {
            let offset = ctx.num(offset2);
            ctx.add(Expr::Add(end, offset))
        };

        // Build 1/(start+offset1) - 1/(end+offset2)
        let one1 = ctx.num(1);
        let one2 = ctx.num(1);
        let first_term = ctx.add(Expr::Div(one1, start_shifted));
        let second_term = ctx.add(Expr::Div(one2, end_shifted));

        // Result = (1/a) * (first - second)
        let diff = ctx.add(Expr::Sub(first_term, second_term));

        let result = if a.abs() == 1 {
            if a > 0 {
                diff
            } else {
                ctx.add(Expr::Neg(diff))
            }
        } else {
            let unsigned_result = ctx.add(Expr::Div(diff, a_expr));
            if a > 0 {
                unsigned_result
            } else {
                ctx.add(Expr::Neg(unsigned_result))
            }
        };

        // Simplify the result
        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx.clone();
        let (simplified, _) = simplifier.simplify(result);
        *ctx = simplifier.context;

        Some(simplified)
    }
}

/// Extract the constant offset from a linear expression: k+offset or k
/// Returns Some(offset) if expr = var + offset, None otherwise
fn extract_linear_offset(ctx: &Context, expr: ExprId, var: &str) -> Option<i64> {
    match ctx.get(expr) {
        // Just the variable: k+0
        Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var => Some(0),

        // k + c
        Expr::Add(l, r) => {
            if let Expr::Variable(sym_id) = ctx.get(*l) {
                if ctx.sym_name(*sym_id) == var {
                    return get_integer(ctx, *r);
                }
            }
            if let Expr::Variable(sym_id) = ctx.get(*r) {
                if ctx.sym_name(*sym_id) == var {
                    return get_integer(ctx, *l);
                }
            }
            None
        }

        // k - c = k + (-c)
        Expr::Sub(l, r) => {
            if let Expr::Variable(sym_id) = ctx.get(*l) {
                if ctx.sym_name(*sym_id) == var {
                    return get_integer(ctx, *r).map(|c| -c);
                }
            }
            None
        }

        _ => None,
    }
}

/// Get integer value from expression.
///
/// Uses canonical implementation from helpers.rs.
/// (See ARCHITECTURE.md "Canonical Utilities Registry")
fn get_integer(ctx: &Context, expr: ExprId) -> Option<i64> {
    crate::helpers::get_integer(ctx, expr)
}

/// Substitute variable with value in expression
fn substitute_var(ctx: &mut Context, expr: ExprId, var: &str, value: ExprId) -> ExprId {
    match ctx.get(expr) {
        Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var => value,
        Expr::Variable(_) | Expr::Number(_) | Expr::Constant(_) => expr,
        Expr::Add(l, r) => {
            let (l, r) = (*l, *r);
            let new_l = substitute_var(ctx, l, var, value);
            let new_r = substitute_var(ctx, r, var, value);
            ctx.add(Expr::Add(new_l, new_r))
        }
        Expr::Sub(l, r) => {
            let (l, r) = (*l, *r);
            let new_l = substitute_var(ctx, l, var, value);
            let new_r = substitute_var(ctx, r, var, value);
            ctx.add(Expr::Sub(new_l, new_r))
        }
        Expr::Mul(l, r) => {
            let (l, r) = (*l, *r);
            let new_l = substitute_var(ctx, l, var, value);
            let new_r = substitute_var(ctx, r, var, value);
            mul2_raw(ctx, new_l, new_r)
        }
        Expr::Div(l, r) => {
            let (l, r) = (*l, *r);
            let new_l = substitute_var(ctx, l, var, value);
            let new_r = substitute_var(ctx, r, var, value);
            ctx.add(Expr::Div(new_l, new_r))
        }
        Expr::Pow(l, r) => {
            let (l, r) = (*l, *r);
            let new_l = substitute_var(ctx, l, var, value);
            let new_r = substitute_var(ctx, r, var, value);
            ctx.add(Expr::Pow(new_l, new_r))
        }
        Expr::Neg(e) => {
            let e = *e;
            let new_e = substitute_var(ctx, e, var, value);
            ctx.add(Expr::Neg(new_e))
        }
        Expr::Function(name, ref args) => {
            let (name, args) = (*name, args.clone());
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|a| substitute_var(ctx, *a, var, value))
                .collect();
            ctx.add(Expr::Function(name, new_args))
        }
        Expr::Matrix {
            rows,
            cols,
            ref data,
        } => {
            let (rows, cols, data) = (*rows, *cols, data.clone());
            let new_data: Vec<ExprId> = data
                .iter()
                .map(|a| substitute_var(ctx, *a, var, value))
                .collect();
            ctx.add(Expr::Matrix {
                rows,
                cols,
                data: new_data,
            })
        }
        // SessionRef is a leaf - no substitution needed
        Expr::SessionRef(_) => expr,
        // Hold: substitute inside and rewrap
        Expr::Hold(inner) => {
            let inner = *inner;
            let new_inner = substitute_var(ctx, inner, var, value);
            ctx.add(Expr::Hold(new_inner))
        }
    }
}

// =============================================================================
// PRODUCT RULE: Evaluate finite products (productorio)
// =============================================================================
// Syntax: product(expr, var, start, end)
// Example: product(k, k, 1, 5) → 120  (5!)
// Example: product((k+1)/k, k, 1, n) → n+1  (telescoping)

define_rule!(ProductRule, "Finite Product", |ctx, expr| {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        let name = ctx.sym_name(*fn_id);
        if name == "product" && args.len() == 4 {
            let factor = args[0];
            let var_expr = args[1];
            let start_expr = args[2];
            let end_expr = args[3];

            // Extract variable name
            let var_name = if let Expr::Variable(sym_id) = ctx.get(var_expr) {
                ctx.sym_name(*sym_id).to_string()
            } else {
                return None;
            };

            // =====================================================================
            // TELESCOPING DETECTION: Check for rational products that telescope
            // =====================================================================
            // Pattern: (k+a)/k → product = (end+a)! / (start-1+a)! * (start-1)! / end!
            // Simple case: (k+1)/k → product = (n+1)/1 = n+1

            if let Some(result) =
                try_telescoping_product(ctx, factor, &var_name, start_expr, end_expr)
            {
                return Some(Rewrite::new(result).desc_lazy(|| {
                    format!(
                        "Telescoping product: Π({}, {}) from {} to {}",
                        cas_ast::DisplayExpr {
                            context: ctx,
                            id: factor
                        },
                        var_name,
                        cas_ast::DisplayExpr {
                            context: ctx,
                            id: start_expr
                        },
                        cas_ast::DisplayExpr {
                            context: ctx,
                            id: end_expr
                        }
                    )
                }));
            }

            // =====================================================================
            // FACTORIZABLE PRODUCT: Handle patterns like 1 - 1/k²
            // =====================================================================
            // Pattern: 1 - 1/k² = (k²-1)/k² = (k-1)(k+1)/k² = [(k-1)/k]·[(k+1)/k]
            // Each factor telescopes separately, then combine results

            if let Some(result) =
                try_factorizable_product(ctx, factor, &var_name, start_expr, end_expr)
            {
                return Some(Rewrite::new(result).desc_lazy(|| {
                    format!(
                        "Factorized telescoping product: Π({}, {}) from {} to {}",
                        cas_ast::DisplayExpr {
                            context: ctx,
                            id: factor
                        },
                        var_name,
                        cas_ast::DisplayExpr {
                            context: ctx,
                            id: start_expr
                        },
                        cas_ast::DisplayExpr {
                            context: ctx,
                            id: end_expr
                        }
                    )
                }));
            }

            // Try to evaluate start and end as integers for numeric evaluation
            let start = get_integer(ctx, start_expr);
            let end = get_integer(ctx, end_expr);

            if let (Some(start), Some(end)) = (start, end) {
                // Safety limit for direct evaluation
                if end - start > 1000 {
                    return None; // Too many terms
                }

                // Direct numeric evaluation
                if start <= end {
                    let mut result = ctx.num(1);
                    for k in start..=end {
                        let k_expr = ctx.num(k);
                        let term = substitute_var(ctx, factor, &var_name, k_expr);
                        result = mul2_raw(ctx, result, term);
                    }

                    // Simplify the result
                    let mut simplifier = crate::Simplifier::with_default_rules();
                    simplifier.context = ctx.clone();
                    let (simplified, _) = simplifier.simplify(result);
                    *ctx = simplifier.context;

                    return Some(Rewrite::new(simplified).desc_lazy(|| {
                        format!(
                            "product({}, {}, {}, {})",
                            cas_ast::DisplayExpr {
                                context: ctx,
                                id: factor
                            },
                            var_name,
                            start,
                            end
                        )
                    }));
                }
            }
        }
    }
    None
});

/// Try to detect and evaluate telescoping products
/// Pattern: (k+a)/(k+b) where a > b → product = (end+a)!/(start-1+a)! * (start-1+b)!/(end+b)!
/// Simple case: (k+1)/k → (n+1)/start
fn try_telescoping_product(
    ctx: &mut Context,
    factor: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    // Check if factor is (k+a)/(k+b) pattern
    if let Expr::Div(num, den) = ctx.get(factor) {
        let (num, den) = (*num, *den);
        // Extract offsets from numerator and denominator
        let num_offset = extract_linear_offset(ctx, num, var)?;
        let den_offset = extract_linear_offset(ctx, den, var)?;

        // For telescoping, we need num_offset > den_offset
        // (k+1)/k means num_offset=1, den_offset=0
        let shift = num_offset - den_offset;

        if shift <= 0 {
            return None; // Not a telescoping pattern
        }

        // For (k+a)/(k+b) with shift = a-b:
        // Product telescopes to: (end+a) * (end+a-1) * ... * (end+b+1) / (start+b-1) * ... / (start+a-1)
        //
        // Simple case shift=1: (k+1)/k from 1 to n
        // = (2/1) * (3/2) * ... * ((n+1)/n) = (n+1)/1 = n+1
        //
        // In general for shift=1:
        // Result = (end + num_offset) / (start + den_offset - 1 + 1) = (end + num_offset) / start_shifted

        if shift == 1 {
            // Simple telescoping: result = (end + num_offset) / (start + den_offset)
            // For (k+1)/k: result = (n+1) / 1 = n+1

            let end_plus_offset = if num_offset == 0 {
                end
            } else {
                let offset = ctx.num(num_offset);
                ctx.add(Expr::Add(end, offset))
            };

            let start_plus_offset = if den_offset == 0 {
                start
            } else {
                let offset = ctx.num(den_offset);
                ctx.add(Expr::Add(start, offset))
            };

            let result = ctx.add(Expr::Div(end_plus_offset, start_plus_offset));

            // Simplify the result
            let mut simplifier = crate::Simplifier::with_default_rules();
            simplifier.context = ctx.clone();
            let (simplified, _) = simplifier.simplify(result);
            *ctx = simplifier.context;

            return Some(simplified);
        }

        // For shift > 1, the pattern is more complex
        // We can still handle it but leave for future enhancement
    }

    None
}

/// Try to factor and evaluate products of factorizable expressions
/// Pattern: 1 - 1/k² = (k²-1)/k² = (k-1)(k+1)/k² = [(k-1)/k]·[(k+1)/k]
///
/// This handles:
/// - 1 - 1/k² (difference with reciprocal square)
/// - (k² - 1)/k² (already as fraction with factorable numerator)
///
/// Result for product from 2 to n:
/// - ∏(k-1)/k = 1/n
/// - ∏(k+1)/k = (n+1)/2
/// - Total: (n+1)/(2n)
fn try_factorizable_product(
    ctx: &mut Context,
    factor: ExprId,
    var: &str,
    start: ExprId,
    end: ExprId,
) -> Option<ExprId> {
    // Pattern 1: 1 - 1/k² or 1 - k^(-2)
    // This is the most common form
    if let Some((base_var, power)) = detect_one_minus_reciprocal_power(ctx, factor, var) {
        if power == 2 && base_var == var {
            // This is 1 - 1/k²
            // Factor as (k-1)(k+1)/k² = [(k-1)/k]·[(k+1)/k]

            // Evaluate ∏(k-1)/k from start to end
            // = (start-1)/start · start/(start+1) · ... · (end-1)/end
            // = (start-1) / end (telescopes to first numerator / last denominator)
            let start_minus_1 = if let Some(n) = get_integer(ctx, start) {
                ctx.num(n - 1)
            } else {
                let one = ctx.num(1);
                ctx.add(Expr::Sub(start, one))
            };

            // Evaluate ∏(k+1)/k from start to end
            // = (start+1)/start · (start+2)/(start+1) · ... · (end+1)/end
            // = (end+1) / start (telescopes to last numerator / first denominator)
            let end_plus_1 = if let Some(n) = get_integer(ctx, end) {
                ctx.num(n + 1)
            } else {
                let one = ctx.num(1);
                ctx.add(Expr::Add(end, one))
            };

            // Combine: (start-1)/end * (end+1)/start = (start-1)*(end+1) / (start*end)
            // Build as a single fraction for better simplification
            let combined_num = mul2_raw(ctx, start_minus_1, end_plus_1);
            let combined_den = mul2_raw(ctx, start, end);
            let result = ctx.add(Expr::Div(combined_num, combined_den));

            return Some(result);
        }
    }

    None
}

/// Detect pattern: 1 - 1/var^power or 1 - var^(-power)
/// Returns (variable name, power) if matched
fn detect_one_minus_reciprocal_power(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> Option<(String, i64)> {
    // Pattern: 1 - 1/k² or k^(-2) - 1 (canonicalized)
    // Also handles: 1 - k^(-2)

    if let Expr::Sub(left, right) = ctx.get(expr) {
        // Check if left is 1
        if let Expr::Number(n) = ctx.get(*left) {
            if n.is_one() {
                // Right should be 1/k² or k^(-2)
                return detect_reciprocal_power(ctx, *right, var);
            }
        }
    }

    // Also check Add with negative: 1 + (-1/k²) (canonical form: -1/k² + 1)
    if let Expr::Add(left, right) = ctx.get(expr) {
        // Check for -1/k² + 1 pattern
        if let Expr::Number(n) = ctx.get(*right) {
            if n.is_one() {
                if let Expr::Neg(inner) = ctx.get(*left) {
                    return detect_reciprocal_power(ctx, *inner, var);
                }
            }
        }
        // Check for 1 + (-1/k²) pattern
        if let Expr::Number(n) = ctx.get(*left) {
            if n.is_one() {
                if let Expr::Neg(inner) = ctx.get(*right) {
                    return detect_reciprocal_power(ctx, *inner, var);
                }
            }
        }
    }

    None
}

/// Detect 1/var^power or var^(-power)
fn detect_reciprocal_power(ctx: &Context, expr: ExprId, var: &str) -> Option<(String, i64)> {
    // Pattern 1: 1/k^n
    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*num) {
            if n.is_one() {
                // den should be k^power
                if let Expr::Pow(base, exp) = ctx.get(*den) {
                    if let Expr::Variable(sym_id) = ctx.get(*base) {
                        if ctx.sym_name(*sym_id) == var {
                            if let Some(power) = get_integer(ctx, *exp) {
                                return Some((ctx.sym_name(*sym_id).to_string(), power));
                            }
                        }
                    }
                }
                // Also check if den is just k (power = 1)
                if let Expr::Variable(sym_id) = ctx.get(*den) {
                    if ctx.sym_name(*sym_id) == var {
                        return Some((ctx.sym_name(*sym_id).to_string(), 1));
                    }
                }
            }
        }
    }

    // Pattern 2: k^(-n)
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if let Expr::Variable(sym_id) = ctx.get(*base) {
            if ctx.sym_name(*sym_id) == var {
                if let Expr::Neg(inner_exp) = ctx.get(*exp) {
                    if let Some(power) = get_integer(ctx, *inner_exp) {
                        return Some((ctx.sym_name(*sym_id).to_string(), power));
                    }
                }
                // Check for negative number exponent
                if let Expr::Number(n) = ctx.get(*exp) {
                    if *n < num_rational::BigRational::from_integer(0.into()) {
                        if let Some(power) = get_integer(ctx, *exp) {
                            return Some((ctx.sym_name(*sym_id).to_string(), -power));
                        }
                    }
                }
            }
        }
    }

    None
}

use num_traits::One;
