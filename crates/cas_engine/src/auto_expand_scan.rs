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
use num_traits::ToPrimitive;

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
    // Try to detect cancellation pattern at this node
    if try_mark_difference_quotient(ctx, id, budget, marks) {
        // If we marked this node, don't need to recurse deeper for pattern detection
        // (but still recurse for children that might have their own patterns)
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
}
