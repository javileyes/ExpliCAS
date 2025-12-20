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
use num_traits::{Signed, ToPrimitive};

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

/// Recursive scanner that visits all nodes and marks cancellation contexts.
fn scan_recursive(ctx: &Context, id: ExprId, budget: &ExpandBudget, marks: &mut PatternMarks) {
    // Try to detect cancellation patterns at this node
    // First try Div(Sub(Pow(..), _), _) pattern (difference quotient)
    if try_mark_difference_quotient(ctx, id, budget, marks) {
        // Marked as Div context, continue to recurse
    }
    // Then try Sub(Pow(..), _) pattern (direct subtraction with potential cancellation)
    else if try_mark_sub_cancellation(ctx, id, budget, marks) {
        // Marked as Sub context, continue to recurse
    }
    // Then try Add(Pow(..), Neg(..), ...) pattern (Sub canonicalized to Add+Neg)
    else if try_mark_add_neg_cancellation(ctx, id, budget, marks) {
        // Marked as Add context (trinomial/multivar case), continue to recurse
    }

    // Recurse into children
    match ctx.get(id) {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            scan_recursive(ctx, *l, budget, marks);
            scan_recursive(ctx, *r, budget, marks);
        }
        Expr::Neg(inner) => {
            scan_recursive(ctx, *inner, budget, marks);
        }
        Expr::Function(_, args) => {
            for arg in args {
                scan_recursive(ctx, *arg, budget, marks);
            }
        }
        Expr::Matrix { data, .. } => {
            for elem in data {
                scan_recursive(ctx, *elem, budget, marks);
            }
        }
        // Atoms: Number, Variable, Constant, SessionRef - no children
        _ => {}
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

        // For Sub, use stricter budget (max exp 3)
        if n > 3 {
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
fn looks_polynomial_like(ctx: &Context, id: ExprId) -> bool {
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
    if estimated_terms.map_or(true, |t| t > budget.max_generated_terms) {
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
        Expr::Variable(name) => {
            vars.insert(name.clone());
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

/// Flatten an Add expression into a list of terms.
/// E.g., Add(Add(a, b), c) → [a, b, c]
fn flatten_add(ctx: &Context, id: ExprId) -> Vec<ExprId> {
    let mut terms = Vec::new();
    flatten_add_recursive(ctx, id, &mut terms);
    terms
}

fn flatten_add_recursive(ctx: &Context, id: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(id) {
        Expr::Add(l, r) => {
            flatten_add_recursive(ctx, *l, terms);
            flatten_add_recursive(ctx, *r, terms);
        }
        _ => {
            terms.push(id);
        }
    }
}

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

/// Try to detect and mark Add patterns with Pow and negated polynomial terms.
/// Pattern: `Add(Pow(Add(..), n), Neg(poly), Neg(poly), ...)`
///
/// This handles the case where `Sub(a, b)` has been canonicalized to `Add(a, Neg(b))`.
///
/// Returns `true` if pattern detected and marked.
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

    let terms = flatten_add(ctx, id);
    if terms.len() < 2 {
        return false;
    }

    // Find exactly ONE Pow(Add(..), n) term (the "positive" polynomial power)
    let mut pow_term: Option<(ExprId, ExprId, u32)> = None; // (pow_id, base, exp)
    let mut neg_terms: Vec<ExprId> = Vec::new();
    let mut other_positive_terms = 0;

    for term in &terms {
        if let Some((base, n)) = extract_pow_sum_info(ctx, *term) {
            if pow_term.is_some() {
                // More than one Pow term - not our pattern
                return false;
            }
            pow_term = Some((*term, base, n));
        } else if is_negative_term(ctx, *term) {
            // This is a negated term
            neg_terms.push(*term);
        } else {
            // Positive term that's not a Pow(Add, n)
            other_positive_terms += 1;
        }
    }

    // Must have exactly one Pow term and at least one negated term
    let Some((pow_id, base, n)) = pow_term else {
        return false;
    };
    if neg_terms.is_empty() {
        return false;
    }

    // Allow some positive constant terms (like the constant in the expansion)
    // but don't allow too many random positive terms
    if other_positive_terms > 3 {
        return false;
    }

    // Budget checks for the Pow
    if !passes_budget_checks(ctx, base, n, budget) {
        return false;
    }

    // Stricter exponent limit for Add+Neg patterns (same as Sub)
    if n > 3 {
        return false;
    }

    // Verify negated terms look polynomial-like
    for neg_term in &neg_terms {
        // For negative terms, check they're still polynomial-like
        if !looks_polynomial_like(ctx, *neg_term) {
            return false;
        }
    }

    // Pattern matches! Mark the Add as auto-expand context.
    marks.mark_auto_expand_context(id);

    // Also mark the Pow term's parent Add for good measure
    let _ = pow_id; // Silence unused warning

    true
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
}
