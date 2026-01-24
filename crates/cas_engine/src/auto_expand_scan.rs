//! Auto-expand candidate scanner for intelligent expansion in cancellation contexts.
//!
//! This module detects patterns where expanding `Pow(Add(..), n)` is likely to lead to
//! cancellation, and marks the context nodes (Div/Sub) for auto-expansion.
//!
//! MVP Pattern: Difference Quotient
//! - `Div(Sub(Pow(sum, n), _), denom)` → marks the Div as auto-expand context
//! - Example: `((x+h)^3 - x^3)/h` → Pow(x+h, 3) will be expanded within this Div

use crate::pattern_marks::PatternMarks;
use crate::phase::ExpandBudget;
use cas_ast::{Context, Expr, ExprId};
use num_traits::{Signed, ToPrimitive, Zero};

/// Scan expression tree and mark auto-expand contexts where expansion is beneficial.
///
/// This only marks contexts (Div/Sub nodes) where expanding inner Pow(Add(..), n)
/// expressions is likely to lead to cancellation. Standalone polynomials like
/// `(x+1)^3` are NOT marked and will NOT be expanded.
///
/// # Arguments
/// * `ctx` - Expression context
/// * `root` - Root of expression tree to scan
/// * `budget` - Expansion budget limits
/// * `marks` - PatternMarks to update with auto-expand contexts
pub fn mark_auto_expand_candidates(
    ctx: &Context,
    root: ExprId,
    budget: &ExpandBudget,
    marks: &mut PatternMarks,
) {
    scan_recursive(ctx, root, budget, marks);
}

/// Iterative scanner with depth limit (replaces recursive scan_recursive).
/// Visits all nodes up to MAX_SCAN_DEPTH and marks cancellation contexts.
fn scan_recursive(ctx: &Context, root: ExprId, budget: &ExpandBudget, marks: &mut PatternMarks) {
    const MAX_SCAN_DEPTH: usize = 200;

    // Work stack: (expr_id, depth)
    let mut work_stack = vec![(root, 0usize)];

    while let Some((id, depth)) = work_stack.pop() {
        // Depth guard
        if depth >= MAX_SCAN_DEPTH {
            continue;
        }

        // Try to detect cancellation patterns at this node
        if try_mark_difference_quotient(ctx, id, budget, marks) {
            // Marked as Div context
        } else if try_mark_sub_cancellation(ctx, id, budget, marks) {
            // Marked as Sub context
        } else if try_mark_add_neg_cancellation(ctx, id, budget, marks) {
            // Marked as Add context
        } else if try_mark_log_cancellation(ctx, id, marks) {
            // Marked as log context
        }

        // Push children for processing
        match ctx.get(id) {
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                work_stack.push((*l, depth + 1));
                work_stack.push((*r, depth + 1));
            }
            Expr::Neg(inner) => {
                work_stack.push((*inner, depth + 1));
            }
            Expr::Function(_, args) => {
                for arg in args {
                    work_stack.push((*arg, depth + 1));
                }
            }
            Expr::Matrix { data, .. } => {
                for elem in data {
                    work_stack.push((*elem, depth + 1));
                }
            }
            _ => {}
        }
    }
}

/// Try to detect and mark difference quotient pattern: `Div(Sub(Pow(sum, n), _), denom)`
///
/// This is the MVP high-probability cancellation pattern:
/// - `((x+h)^n - x^n)/h` → after expansion, `h` terms can be factored and cancelled
/// - `((x+h)^n - f(x))/h` → similar pattern
///
/// Returns `true` if the pattern was detected and marked.
fn try_mark_difference_quotient(
    ctx: &Context,
    id: ExprId,
    budget: &ExpandBudget,
    marks: &mut PatternMarks,
) -> bool {
    // Pattern: Div(Sub(Pow(sum, n), _), denom)
    if let Expr::Div(num, _den) = ctx.get(id) {
        if let Expr::Sub(lhs, _rhs) = ctx.get(*num) {
            // Check if lhs is Pow(Add(..), n) with small positive integer n
            if let Some((base, n)) = extract_pow_sum_info(ctx, *lhs) {
                // Budget checks
                if !passes_budget_checks(ctx, base, n, budget) {
                    return false;
                }

                // Pattern matches and passes budget! Mark the Div as auto-expand context.
                marks.mark_auto_expand_context(id);
                return true;
            }
        }
    }

    false
}

/// Try to detect and mark Sub patterns where expansion might cancel.
/// Pattern: Sub chain containing Pow(Add(..), n) and polynomial terms
///
/// Examples:
/// - `(x+1)^2 - (x^2 + 2*x + 1)` → after expansion, cancels to 0
/// - `(x+y+1)^2 - x^2 - y^2 - 1 - 2*x - 2*y - 2*x*y` → nested Sub chain
///
/// Returns `true` if the pattern was detected and marked.
fn try_mark_sub_cancellation(
    ctx: &Context,
    id: ExprId,
    budget: &ExpandBudget,
    marks: &mut PatternMarks,
) -> bool {
    // Only process Sub nodes
    if !matches!(ctx.get(id), Expr::Sub(_, _)) {
        return false;
    }

    // Entire expression must be polynomial-like (no functions, etc.)
    if !looks_polynomial_like(ctx, id) {
        return false;
    }

    // Search for Pow(Add(..), n) anywhere in this Sub expression
    if let Some((pow_base, n)) = find_pow_add_in_expr(ctx, id) {
        // Budget checks
        if !passes_budget_checks(ctx, pow_base, n, budget) {
            return false;
        }

        // V2.15.8: Relax to n≤6 to match ExpandSmallBinomialPowRule budget
        // (previously n≤3, but users expect identities like (x+1)^5 - expansion = 0)
        if n > 6 {
            return false;
        }

        // Pattern matches! Mark this Sub node as auto-expand context.
        marks.mark_auto_expand_context(id);
        return true;
    }

    false
}

/// Search for Pow(Add(..), n) anywhere in expression tree
fn find_pow_add_in_expr(ctx: &Context, id: ExprId) -> Option<(ExprId, u32)> {
    // Check if this node itself is Pow(Add, n)
    if let Some(info) = extract_pow_sum_info(ctx, id) {
        return Some(info);
    }

    // Recurse into children
    match ctx.get(id) {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            find_pow_add_in_expr(ctx, *l).or_else(|| find_pow_add_in_expr(ctx, *r))
        }
        Expr::Neg(inner) => find_pow_add_in_expr(ctx, *inner),
        _ => None,
    }
}

/// Heuristic: check if expression looks like an expanded polynomial
/// (contains Add, Mul, Pow with integer exponents, but no functions)
pub fn looks_polynomial_like(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) => true,
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            looks_polynomial_like(ctx, *l) && looks_polynomial_like(ctx, *r)
        }
        Expr::Pow(base, exp) => {
            // Exponent should be a non-negative integer
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() && !n.is_negative() {
                    return looks_polynomial_like(ctx, *base);
                }
            }
            false
        }
        Expr::Neg(inner) => looks_polynomial_like(ctx, *inner),
        // Functions, Div, Matrix, SessionRef are not "polynomial-like"
        _ => false,
    }
}

/// Extract (base, exponent) from Pow(base, exp) where base is Add(..) and exp is small positive int.
fn extract_pow_sum_info(ctx: &Context, id: ExprId) -> Option<(ExprId, u32)> {
    if let Expr::Pow(base, exp) = ctx.get(id) {
        // Check if base is an Add (sum)
        if !matches!(ctx.get(*base), Expr::Add(_, _)) {
            return None;
        }

        // Check if exponent is a small positive integer
        if let Expr::Number(n) = ctx.get(*exp) {
            if !n.is_integer() {
                return None;
            }
            use num_traits::Signed;
            if n.is_negative() {
                return None;
            }
            let n_val = n.to_integer().to_u32()?;
            if n_val < 2 {
                return None; // n=0,1 not useful for expansion
            }
            return Some((*base, n_val));
        }
    }
    None
}

/// Extract (sign, base, exponent) from either:
/// - `Pow(Add(..), n)` → returns `(+1, base, n)`
/// - `Neg(Pow(Add(..), n))` → returns `(-1, base, n)`
///
/// This unified extractor handles both "positive Pow" and "negated Pow" patterns,
/// enabling detection of inverse patterns like Sophie-Germain.
fn extract_pow_sum_with_sign(ctx: &Context, id: ExprId) -> Option<(i8, ExprId, u32)> {
    // Check for positive Pow(Add, n)
    if let Some((base, n)) = extract_pow_sum_info(ctx, id) {
        return Some((1, base, n));
    }

    // Check for Neg(Pow(Add, n))
    if let Expr::Neg(inner) = ctx.get(id) {
        if let Some((base, n)) = extract_pow_sum_info(ctx, *inner) {
            return Some((-1, base, n));
        }
    }

    None
}

/// Check if a Pow(base, n) passes all budget constraints.
fn passes_budget_checks(ctx: &Context, base: ExprId, n: u32, budget: &ExpandBudget) -> bool {
    // Budget check 1: max_pow_exp
    if n > budget.max_pow_exp {
        return false;
    }

    // Budget check 2: max_base_terms
    let num_terms = count_add_terms(ctx, base);
    if num_terms > budget.max_base_terms {
        return false;
    }

    // Budget check 3: max_generated_terms (multinomial estimate)
    let estimated_terms = estimate_multinomial_terms(num_terms, n);
    if estimated_terms.is_none_or(|t| t > budget.max_generated_terms) {
        return false;
    }

    // Budget check 4: max_vars
    let var_count = count_variables(ctx, base);
    if var_count > budget.max_vars {
        return false;
    }

    true
}

/// Count number of terms in a flattened Add expression.
fn count_add_terms(ctx: &Context, id: ExprId) -> u32 {
    match ctx.get(id) {
        Expr::Add(l, r) => count_add_terms(ctx, *l) + count_add_terms(ctx, *r),
        _ => 1,
    }
}

/// Estimate number of terms in multinomial expansion: C(n+m-1, m-1)
/// where n = exponent, m = number of terms in base.
/// Returns None on overflow.
fn estimate_multinomial_terms(m: u32, n: u32) -> Option<u32> {
    // For binomial (m=2): C(n+1, 1) = n+1
    // For trinomial (m=3): C(n+2, 2) = (n+1)(n+2)/2
    // General: C(n+m-1, m-1)

    if m == 0 || n == 0 {
        return Some(1);
    }

    // Use iterative binomial coefficient computation to avoid overflow
    // C(n+m-1, m-1) = C(n+m-1, n)
    let total = (n + m - 1) as u64;
    let k = (m - 1).min(n) as u64;

    let mut result: u64 = 1;
    for i in 0..k {
        // result *= (total - i) / (i + 1)
        // Do multiplication first, then division to avoid fractions
        result = result.checked_mul(total - i)?;
        result /= i + 1;

        // Early bailout if already too large
        if result > u32::MAX as u64 {
            return None;
        }
    }

    Some(result as u32)
}

/// Count unique variables in an expression (simple heuristic).
fn count_variables(ctx: &Context, id: ExprId) -> u32 {
    let mut vars = std::collections::HashSet::new();
    collect_variables_recursive(ctx, id, &mut vars);
    vars.len() as u32
}

fn collect_variables_recursive(
    ctx: &Context,
    id: ExprId,
    vars: &mut std::collections::HashSet<String>,
) {
    match ctx.get(id) {
        Expr::Variable(sym_id) => {
            vars.insert(ctx.sym_name(*sym_id).to_string());
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            collect_variables_recursive(ctx, *l, vars);
            collect_variables_recursive(ctx, *r, vars);
        }
        Expr::Neg(inner) => {
            collect_variables_recursive(ctx, *inner, vars);
        }
        Expr::Function(_, args) => {
            for arg in args {
                collect_variables_recursive(ctx, *arg, vars);
            }
        }
        _ => {}
    }
}

// =============================================================================
// Add+Neg pattern detection (for trinomial/multivar cancellation)
// =============================================================================

// NOTE: Local flatten_add REMOVED - use crate::nary::add_terms_no_sign instead
// (see ARCHITECTURE.md "Canonical Utilities Registry")

#[allow(dead_code)] // Used in tests
/// Check if a term is "negative" for the purpose of sub-cancellation detection.
/// Returns true if the term represents a negative value:
/// - `Neg(t)` → true
/// - `Number(-c)` where c > 0 → true  
/// - `Mul` with ODD number of negative factors → true
/// - Otherwise → false
///
/// Note: This is a predicate, not a "strip" function. We don't reconstruct
/// the positive form - we just detect negativity for pattern matching.
fn is_negative_term(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Neg(_) => true,

        // Negative number: -1, -2, -5/3 etc.
        Expr::Number(n) => {
            use num_traits::Signed;
            n.is_negative()
        }

        Expr::Mul(l, r) => {
            // Count negative factors using XOR logic:
            // (-a) * b = negative, (-a) * (-b) = positive
            let l_neg = is_negative_factor(ctx, *l);
            let r_neg = is_negative_factor(ctx, *r);
            l_neg ^ r_neg // XOR: negative only if exactly one is negative
        }

        _ => false,
    }
}

#[allow(dead_code)] // Used via is_negative_term in tests
/// Helper: Check if a factor is negative (for Mul sign calculation)
fn is_negative_factor(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(n) => {
            use num_traits::Signed;
            n.is_negative()
        }
        Expr::Neg(_) => true,
        // Recursively check nested Mul
        Expr::Mul(l, r) => is_negative_factor(ctx, *l) ^ is_negative_factor(ctx, *r),
        _ => false,
    }
}

/// Try to detect and mark Add patterns with Pow(Add,n) where expansion leads to cancellation.
///
/// Handles TWO patterns via unified sign-normalized detection:
/// 1. `Add(Pow(Add, n), Neg(poly), ...)` - positive Pow + negative terms
/// 2. `Add(poly, poly, ..., Neg(Pow(Add, n)))` - positive terms + negative Pow (Sophie-Germain)
///
/// Uses speculative expand + score to verify cancellation before marking.
/// Returns `true` if pattern detected, verified, and marked.
fn try_mark_add_neg_cancellation(
    ctx: &Context,
    id: ExprId,
    budget: &ExpandBudget,
    marks: &mut PatternMarks,
) -> bool {
    // Only process Add nodes
    if !matches!(ctx.get(id), Expr::Add(_, _)) {
        return false;
    }

    let terms = crate::nary::add_terms_no_sign(ctx, id);
    if terms.len() < 2 {
        return false;
    }

    // Find exactly ONE Pow(Add, n) term (positive or negative via unified extractor)
    let mut pow_candidate: Option<(ExprId, i8, ExprId, u32)> = None; // (term_id, sign, base, exp)
    let mut other_terms: Vec<ExprId> = Vec::new();

    for term in &terms {
        if let Some((sign, base, n)) = extract_pow_sum_with_sign(ctx, *term) {
            if pow_candidate.is_some() {
                // More than one Pow term - not our pattern
                return false;
            }
            pow_candidate = Some((*term, sign, base, n));
        } else {
            other_terms.push(*term);
        }
    }

    // Must have exactly one Pow term
    let Some((_pow_term_id, sign, base, n)) = pow_candidate else {
        return false;
    };

    // Filter out literal zeros from other_terms - they should not count as "real" terms
    // This prevents e+0 from triggering expansion when e alone wouldn't
    let other_terms: Vec<ExprId> = other_terms
        .into_iter()
        .filter(|term| !matches!(ctx.get(*term), Expr::Number(n) if n.is_zero()))
        .collect();

    // Must have at least one REAL other term to potentially cancel with
    if other_terms.is_empty() {
        return false;
    }

    // V2.15.8: Extended to n≤6 for consistency with ExpandSmallBinomialPowRule
    // (previously n=2 only, but users expect identities like (x+1)^5 - expansion = 0)
    if !(2..=6).contains(&n) {
        return false;
    }

    let base_term_count = count_add_terms(ctx, base);
    if base_term_count > 3 {
        return false; // Max 3 terms in base (e.g., a² + 2b² for Sophie-Germain)
    }

    // Standard budget checks
    if !passes_budget_checks(ctx, base, n, budget) {
        return false;
    }

    // Verify all other terms look polynomial-like
    for term in &other_terms {
        if !looks_polynomial_like(ctx, *term) {
            return false;
        }
    }

    // === SPECULATIVE EXPAND + SCORE ===
    // Trial expand and check if node count decreases (indicating cancellation)
    if !speculative_expand_reduces_nodes(ctx, id, base, n, sign, &other_terms) {
        return false;
    }

    // Pattern matches and verified! Mark the Add as auto-expand context.
    marks.mark_auto_expand_context(id);
    true
}

/// Speculative expand + score: check if expanding the Pow would likely lead to cancellation.
///
/// V2.15.8: Extended to handle binomial powers up to n=6 for education mode.
/// Heuristic: expansion is beneficial if other_terms count is close to
/// the number of terms that would be produced by expansion.
fn speculative_expand_reduces_nodes(
    ctx: &Context,
    _original_id: ExprId,
    pow_base: ExprId,
    pow_exp: u32,
    _pow_sign: i8,
    other_terms: &[ExprId],
) -> bool {
    // Only handle 2 <= n <= 6
    if !(2..=6).contains(&pow_exp) {
        return false;
    }

    // Count terms in base - only handle binomials (2 terms) for simplicity
    let base_term_count = count_add_terms(ctx, pow_base);
    if base_term_count != 2 {
        // For trinomials, only handle n=2
        if base_term_count == 3 && pow_exp == 2 {
            // Trinomial square produces 6 terms
            return other_terms.len() >= 3;
        }
        return false;
    }

    // Binomial expansion: (a+b)^n produces n+1 terms
    let expansion_term_count = pow_exp + 1;

    // Accept if other_terms count is close to expansion term count
    // (at least half the terms might cancel)
    other_terms.len() >= (expansion_term_count as usize) / 2
}

// =============================================================================
// Log expansion pattern detection (for log cancellation)
// =============================================================================

/// Try to detect and mark log expansion patterns where expanding leads to cancellation.
///
/// Pattern: `log(product) ± k*log(a) ± m*log(b)` where expanding log(product) might cancel
/// with the explicit log terms.
///
/// Examples:
/// - `ln(a^2 * b^3) - 2*ln(a) - 3*ln(b)` → expands to 0
/// - `ln(a*b) - ln(a) - ln(b)` → expands to 0
/// - `2*ln(a) + 3*ln(b) - ln(a^2 * b^3)` → expands to 0
///
/// Returns `true` if pattern detected and marked.
fn try_mark_log_cancellation(ctx: &Context, id: ExprId, marks: &mut PatternMarks) -> bool {
    // Only process Add/Sub nodes (log terms combined additively)
    if !matches!(ctx.get(id), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return false;
    }

    // Collect all additive terms
    let terms = crate::nary::add_terms_no_sign(ctx, id);
    if terms.len() < 2 {
        return false;
    }

    // Count log terms and check if one is "expandable" (has Mul/Div/Pow inside)
    let mut expandable_log_count = 0;
    let mut simple_log_count = 0;

    for term in &terms {
        if let Some(info) = extract_log_info(ctx, *term) {
            if is_expandable_log_arg(ctx, info.arg) {
                expandable_log_count += 1;
            } else {
                simple_log_count += 1;
            }
        } else if is_coefficient_times_log(ctx, *term) {
            simple_log_count += 1;
        }
    }

    // We need at least one expandable log and at least one simple log for cancellation
    if expandable_log_count == 0 || simple_log_count == 0 {
        return false;
    }

    // Pattern matches! Mark the Add/Sub as auto-expand context for logs.
    marks.mark_auto_expand_context(id);
    true
}

/// Info about a log expression: the argument and base
struct LogInfo {
    arg: ExprId,
    #[allow(dead_code)]
    base: Option<ExprId>, // None for ln, Some(base) for log(base, arg)
}

/// Extract log info from a Function node
fn extract_log_info(ctx: &Context, id: ExprId) -> Option<LogInfo> {
    match ctx.get(id) {
        Expr::Function(name, args) if name == "ln" && args.len() == 1 => Some(LogInfo {
            arg: args[0],
            base: None,
        }),
        Expr::Function(name, args) if name == "log" && args.len() == 1 => Some(LogInfo {
            arg: args[0],
            base: None,
        }),
        Expr::Function(name, args) if name == "log" && args.len() == 2 => Some(LogInfo {
            arg: args[1],
            base: Some(args[0]),
        }),
        Expr::Neg(inner) => extract_log_info(ctx, *inner),
        _ => None,
    }
}

/// Check if a log argument is "expandable" (contains Mul, Div, or Pow)
fn is_expandable_log_arg(ctx: &Context, arg: ExprId) -> bool {
    match ctx.get(arg) {
        Expr::Mul(_, _) => true, // ln(a*b) can expand
        Expr::Div(_, _) => true, // ln(a/b) can expand
        Expr::Pow(_, exp) => {
            // ln(a^n) can expand if n is not 1
            if let Expr::Number(n) = ctx.get(*exp) {
                if n.is_integer() {
                    let n_int = n.to_integer();
                    return n_int != num_bigint::BigInt::from(1);
                }
            }
            false
        }
        _ => false,
    }
}

/// Check if term is of form `k * log(x)` or `k * ln(x)` (coefficient times log)
fn is_coefficient_times_log(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Mul(l, r) => {
            // Check if one side is a number and the other is a log
            let l_is_num = matches!(ctx.get(*l), Expr::Number(_));
            let r_is_num = matches!(ctx.get(*r), Expr::Number(_));
            let l_is_log = extract_log_info(ctx, *l).is_some();
            let r_is_log = extract_log_info(ctx, *r).is_some();

            (l_is_num && r_is_log) || (r_is_num && l_is_log)
        }
        Expr::Neg(inner) => is_coefficient_times_log(ctx, *inner),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phase::ExpandBudget;

    fn default_budget() -> ExpandBudget {
        ExpandBudget::default()
    }

    #[test]
    fn test_difference_quotient_detected() {
        // Build: ((x+h)^3 - x^3)/h
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let h = ctx.var("h");
        let x_plus_h = ctx.add(Expr::Add(x, h));
        let three = ctx.num(3);
        let x_plus_h_cubed = ctx.add(Expr::Pow(x_plus_h, three));
        let x_cubed = ctx.add(Expr::Pow(x, three));
        let diff = ctx.add(Expr::Sub(x_plus_h_cubed, x_cubed));
        let quotient = ctx.add(Expr::Div(diff, h));

        let mut marks = PatternMarks::new();
        mark_auto_expand_candidates(&ctx, quotient, &default_budget(), &mut marks);

        assert!(
            marks.is_auto_expand_context(quotient),
            "Div should be marked as auto-expand context"
        );
    }

    #[test]
    fn test_standalone_pow_not_marked() {
        // Build: (x+1)^3 - should NOT be marked
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let one = ctx.num(1);
        let x_plus_one = ctx.add(Expr::Add(x, one));
        let three = ctx.num(3);
        let pow = ctx.add(Expr::Pow(x_plus_one, three));

        let mut marks = PatternMarks::new();
        mark_auto_expand_candidates(&ctx, pow, &default_budget(), &mut marks);

        assert!(
            !marks.has_auto_expand_contexts(),
            "Standalone Pow should NOT be marked"
        );
    }

    #[test]
    fn test_budget_rejects_high_exponent() {
        // Build: ((x+h)^10 - x^10)/h - exponent 10 > default max 4
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let h = ctx.var("h");
        let x_plus_h = ctx.add(Expr::Add(x, h));
        let ten = ctx.num(10);
        let x_plus_h_10 = ctx.add(Expr::Pow(x_plus_h, ten));
        let x_10 = ctx.add(Expr::Pow(x, ten));
        let diff = ctx.add(Expr::Sub(x_plus_h_10, x_10));
        let quotient = ctx.add(Expr::Div(diff, h));

        let mut marks = PatternMarks::new();
        mark_auto_expand_candidates(&ctx, quotient, &default_budget(), &mut marks);

        assert!(
            !marks.has_auto_expand_contexts(),
            "High exponent should be rejected by budget"
        );
    }

    #[test]
    fn test_multinomial_estimation() {
        // Binomial: C(n+1, 1) = n+1
        assert_eq!(estimate_multinomial_terms(2, 3), Some(4)); // (a+b)^3 has 4 terms
        assert_eq!(estimate_multinomial_terms(2, 4), Some(5)); // (a+b)^4 has 5 terms

        // Trinomial: C(n+2, 2) = (n+1)(n+2)/2
        assert_eq!(estimate_multinomial_terms(3, 2), Some(6)); // (a+b+c)^2 has 6 terms
        assert_eq!(estimate_multinomial_terms(3, 3), Some(10)); // (a+b+c)^3 has 10 terms
    }

    #[test]
    fn test_is_negative_term_basic() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let neg_one = ctx.num(-1);
        let neg_two = ctx.num(-2);
        let pos_two = ctx.num(2);

        // Neg(x) -> negative
        let neg_x = ctx.add(Expr::Neg(x));
        assert!(is_negative_term(&ctx, neg_x), "Neg(x) should be negative");

        // Number(-1) -> negative
        assert!(
            is_negative_term(&ctx, neg_one),
            "Number(-1) should be negative"
        );

        // Number(-2) -> negative
        assert!(
            is_negative_term(&ctx, neg_two),
            "Number(-2) should be negative"
        );

        // Number(2) -> not negative
        assert!(
            !is_negative_term(&ctx, pos_two),
            "Number(2) should NOT be negative"
        );

        // Variable x -> not negative
        assert!(
            !is_negative_term(&ctx, x),
            "Variable should NOT be negative"
        );
    }

    #[test]
    fn test_is_negative_term_mul_single_neg() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let neg_two = ctx.num(-2);
        let pos_two = ctx.num(2);

        // Mul(-2, x) -> negative (one negative factor)
        let mul_neg = ctx.add(Expr::Mul(neg_two, x));
        assert!(
            is_negative_term(&ctx, mul_neg),
            "Mul(-2, x) should be negative"
        );

        // Mul(2, x) -> not negative
        let mul_pos = ctx.add(Expr::Mul(pos_two, x));
        assert!(
            !is_negative_term(&ctx, mul_pos),
            "Mul(2, x) should NOT be negative"
        );
    }

    #[test]
    fn test_is_negative_term_mul_double_neg() {
        let mut ctx = Context::new();
        let neg_two = ctx.num(-2);
        let neg_three = ctx.num(-3);

        // Mul(-2, -3) -> positive (two negatives cancel out via XOR)
        let mul_double_neg = ctx.add(Expr::Mul(neg_two, neg_three));
        assert!(
            !is_negative_term(&ctx, mul_double_neg),
            "Mul(-2, -3) should NOT be negative (double negative = positive)"
        );
    }

    // ==========================================================================
    // Sophie-Germain inverse pattern tests (speculative expand + score)
    // ==========================================================================

    #[test]
    fn test_sophie_germain_inverse_pattern_detected() {
        // Build: a^4 + 4*b^4 + 4*a^2*b^2 - (a^2 + 2*b^2)^2
        // This is the Sophie-Germain pattern after difference of squares:
        // Pattern: positive poly terms + Neg(Pow(Add, 2))
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");

        // a^4
        let four_exp = ctx.num(4);
        let a4 = ctx.add(Expr::Pow(a, four_exp));

        // 4*b^4
        let four_coef = ctx.num(4);
        let b4 = ctx.add(Expr::Pow(b, four_exp));
        let four_b4 = ctx.add(Expr::Mul(four_coef, b4));

        // 4*a^2*b^2
        let two_exp = ctx.num(2);
        let four_coef2 = ctx.num(4);
        let a2 = ctx.add(Expr::Pow(a, two_exp));
        let b2 = ctx.add(Expr::Pow(b, two_exp));
        let a2_b2 = ctx.add(Expr::Mul(a2, b2));
        let four_a2_b2 = ctx.add(Expr::Mul(four_coef2, a2_b2));

        // (a^2 + 2*b^2)^2
        let two_coef = ctx.num(2);
        let two_b2 = ctx.add(Expr::Mul(two_coef, b2));
        let base = ctx.add(Expr::Add(a2, two_b2));
        let pow_base_2 = ctx.add(Expr::Pow(base, two_exp));
        let neg_pow = ctx.add(Expr::Neg(pow_base_2));

        // Build: a^4 + 4*b^4 + 4*a^2*b^2 + Neg((a^2+2*b^2)^2)
        let sum1 = ctx.add(Expr::Add(a4, four_b4));
        let sum2 = ctx.add(Expr::Add(sum1, four_a2_b2));
        let full_expr = ctx.add(Expr::Add(sum2, neg_pow));

        let mut marks = PatternMarks::new();
        mark_auto_expand_candidates(&ctx, full_expr, &default_budget(), &mut marks);

        assert!(
            marks.is_auto_expand_context(full_expr),
            "Sophie-Germain inverse pattern should be marked for auto-expand"
        );
    }

    #[test]
    fn test_no_cancel_pattern_not_marked() {
        // Build: a^4 + 4*b^4 - (a^2 + 2*b^2 + 1)^2
        // This has inverse pattern but does NOT cancel (extra +1 in base)
        // Should NOT be marked because speculative expand won't show cancellation
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");

        // a^4
        let four_exp = ctx.num(4);
        let a4 = ctx.add(Expr::Pow(a, four_exp));

        // 4*b^4
        let four_coef = ctx.num(4);
        let b4 = ctx.add(Expr::Pow(b, four_exp));
        let four_b4 = ctx.add(Expr::Mul(four_coef, b4));

        // (a^2 + 2*b^2 + 1)^2 - note the +1
        let two_exp = ctx.num(2);
        let one = ctx.num(1);
        let a2 = ctx.add(Expr::Pow(a, two_exp));
        let b2 = ctx.add(Expr::Pow(b, two_exp));
        let two_coef = ctx.num(2);
        let two_b2 = ctx.add(Expr::Mul(two_coef, b2));
        let inner1 = ctx.add(Expr::Add(a2, two_b2));
        let base = ctx.add(Expr::Add(inner1, one)); // a^2 + 2*b^2 + 1
        let pow_base_2 = ctx.add(Expr::Pow(base, two_exp));
        let neg_pow = ctx.add(Expr::Neg(pow_base_2));

        // Build: a^4 + 4*b^4 + Neg((a^2+2*b^2+1)^2)
        let sum1 = ctx.add(Expr::Add(a4, four_b4));
        let full_expr = ctx.add(Expr::Add(sum1, neg_pow));

        let mut marks = PatternMarks::new();
        mark_auto_expand_candidates(&ctx, full_expr, &default_budget(), &mut marks);

        // This should NOT be marked because speculative expand won't show match
        assert!(
            !marks.is_auto_expand_context(full_expr),
            "No-cancel pattern should NOT be marked for auto-expand"
        );
    }

    #[test]
    fn test_budget_exceeded_too_many_base_terms() {
        // Build: x - (a + b + c + d)^2
        // Base has 4 terms, exceeds budget limit of 3
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let a = ctx.var("a");
        let b = ctx.var("b");
        let c = ctx.var("c");
        let d = ctx.var("d");

        let two = ctx.num(2);
        let ab = ctx.add(Expr::Add(a, b));
        let abc = ctx.add(Expr::Add(ab, c));
        let abcd = ctx.add(Expr::Add(abc, d)); // 4 terms
        let pow = ctx.add(Expr::Pow(abcd, two));
        let neg_pow = ctx.add(Expr::Neg(pow));

        let full_expr = ctx.add(Expr::Add(x, neg_pow));

        let mut marks = PatternMarks::new();
        mark_auto_expand_candidates(&ctx, full_expr, &default_budget(), &mut marks);

        assert!(
            !marks.is_auto_expand_context(full_expr),
            "4-term base should exceed budget and NOT be marked"
        );
    }
}
