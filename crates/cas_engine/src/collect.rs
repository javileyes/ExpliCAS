use crate::build::mul2_raw;
use crate::helpers::flatten_add_sub_chain;
use crate::parent_context::ParentContext;
use crate::DomainMode;
use cas_ast::{Context, Expr, ExprId};

use num_rational::BigRational;
use num_traits::{One, Zero};

/// A group of terms that cancelled to zero (e.g., 5 + (-5) → 0)
#[derive(Debug, Clone)]
pub struct CancelledGroup {
    /// The canonical key (term part) that identified this group
    pub key: ExprId,
    /// Original terms from the input expression that cancelled
    pub original_terms: smallvec::SmallVec<[ExprId; 4]>,
    /// Whether this is a pure constant cancellation (prioritized for focus)
    pub is_constant: bool,
}

/// A group of terms that were combined (e.g., x + x → 2x)
#[derive(Debug, Clone)]
pub struct CombinedGroup {
    /// The canonical key (term part) that identifies this group
    pub key: ExprId,
    /// Original terms from the input expression
    pub original_terms: smallvec::SmallVec<[ExprId; 4]>,
    /// The combined result term
    pub combined_term: ExprId,
}

/// Result of a semantics-aware collection operation.
/// Contains the new expression and tracking of what was cancelled/combined.
#[derive(Debug, Clone)]
pub struct CollectResult {
    pub new_expr: ExprId,
    pub assumption: Option<String>,
    /// Groups of terms that cancelled to zero
    pub cancelled: Vec<CancelledGroup>,
    /// Groups of terms that were combined
    pub combined: Vec<CombinedGroup>,
}

/// Check if an expression contains any Div with a denominator that is not proven non-zero.
/// This indicates "undefined risk" - the expression could be undefined at some points.
pub fn has_undefined_risk(ctx: &Context, expr: ExprId) -> bool {
    use crate::domain::Proof;
    use crate::helpers::prove_nonzero;

    let mut stack = vec![expr];
    while let Some(e) = stack.pop() {
        match ctx.get(e) {
            Expr::Div(num, den) => {
                if prove_nonzero(ctx, *den) != Proof::Proven {
                    return true;
                }
                // Still need to check num for nested issues
                stack.push(*num);
                stack.push(*den);
            }
            Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) => {
                stack.push(*inner);
            }
            Expr::Function(_, args) => {
                for arg in args {
                    stack.push(*arg);
                }
            }
            // Leaf nodes: nothing to push
            _ => {}
        }
    }
    false
}

/// Collects like terms with domain_mode awareness.
/// In Strict mode, refuses to cancel terms that might be undefined.
/// In Assume mode, cancels with a warning.
/// In Generic mode, cancels unconditionally.
///
/// Returns None if the result would be identical to input or if blocked by Strict mode.
pub fn collect_with_semantics(
    ctx: &mut Context,
    expr: ExprId,
    parent_ctx: &ParentContext,
) -> Option<CollectResult> {
    // CRITICAL: Do NOT collect non-commutative expressions (e.g., matrices)
    if !ctx.is_mul_commutative(expr) {
        return None;
    }

    // Check for undefined risk in the entire expression
    let risk = has_undefined_risk(ctx, expr);
    let domain_mode = parent_ctx.domain_mode();

    // Determine if we should proceed based on domain_mode
    let (allowed, assumption) = match domain_mode {
        DomainMode::Strict => {
            if risk {
                // In Strict mode, don't cancel terms with undefined risk
                (false, None)
            } else {
                (true, None)
            }
        }
        DomainMode::Assume => {
            // In Assume mode, allow with warning if there's risk
            let assumption = if risk {
                Some("Assuming expression is defined (denominators ≠ 0)".to_string())
            } else {
                None
            };
            (true, assumption)
        }
        DomainMode::Generic => {
            // In Generic mode, always allow without warning
            (true, None)
        }
    };

    if !allowed {
        return None;
    }

    // Run the actual collection logic
    let impl_result = collect_impl(ctx, expr);

    // Only return if something changed
    if impl_result.new_expr == expr {
        return None;
    }

    Some(CollectResult {
        new_expr: impl_result.new_expr,
        assumption,
        cancelled: impl_result.cancelled,
        combined: impl_result.combined,
    })
}

/// Collects like terms in an expression (legacy wrapper, uses Generic mode).
/// e.g. 2*x + 3*x -> 5*x
///      x + x -> 2*x
///      x^2 + 2*x^2 -> 3*x^2
pub fn collect(ctx: &mut Context, expr: ExprId) -> ExprId {
    // Use Generic mode for backward compatibility (no blocking, no warnings)
    let fake_parent = ParentContext::root();
    match collect_with_semantics(ctx, expr, &fake_parent) {
        Some(result) => result.new_expr,
        None => expr, // No change or blocked
    }
}

/// Internal result from collect_impl with tracking info
struct CollectImplResult {
    new_expr: ExprId,
    cancelled: Vec<CancelledGroup>,
    combined: Vec<CombinedGroup>,
}

/// Internal implementation of collect logic (no semantics checking)
/// Now tracks original terms per group for didactic focus display
fn collect_impl(ctx: &mut Context, expr: ExprId) -> CollectImplResult {
    // CRITICAL: Do NOT collect non-commutative expressions (e.g., matrices)
    if !ctx.is_mul_commutative(expr) {
        return CollectImplResult {
            new_expr: expr,
            cancelled: vec![],
            combined: vec![],
        };
    }

    // Flatten terms (using shared helper from crate::helpers)
    let terms = flatten_add_sub_chain(ctx, expr);

    // Group terms by their "non-coefficient" part, tracking original terms
    // Each group: (accumulated_coeff, key_term_part, original_terms)
    let mut groups: Vec<(BigRational, ExprId, smallvec::SmallVec<[ExprId; 4]>)> = Vec::new();

    for term in terms {
        let (coeff, term_part) = extract_numerical_coeff(ctx, term);

        // Find if we already have this term_part in groups
        let mut found = false;
        for (g_coeff, g_term, g_originals) in groups.iter_mut() {
            if are_structurally_equal(ctx, *g_term, term_part) {
                *g_coeff += coeff.clone();
                g_originals.push(term); // Track original term
                found = true;
                break;
            }
        }

        if !found {
            let mut originals = smallvec::SmallVec::new();
            originals.push(term);
            groups.push((coeff, term_part, originals));
        }
    }

    // Sort groups to ensure canonical order
    groups.sort_by(|a, b| crate::ordering::compare_expr(ctx, a.1, b.1));

    // Track cancelled and combined groups
    let mut cancelled = Vec::new();
    let mut combined = Vec::new();

    // Reconstruct expression
    let mut new_terms = Vec::new();
    for (coeff, term_part, originals) in groups {
        if coeff.is_zero() {
            // This group cancelled to zero
            let is_constant = is_one_term(ctx, term_part);
            cancelled.push(CancelledGroup {
                key: term_part,
                original_terms: originals,
                is_constant,
            });
            continue;
        }

        let term = if is_one_term(ctx, term_part) {
            // Just the coefficient (constant term)
            ctx.add(Expr::Number(coeff.clone()))
        } else if coeff.is_one() {
            term_part
        } else if coeff == BigRational::from_integer((-1).into()) {
            // Use Neg(x) instead of Mul(-1, x) for conciseness
            ctx.add(Expr::Neg(term_part))
        } else {
            let coeff_expr = ctx.add(Expr::Number(coeff.clone()));
            mul2_raw(ctx, coeff_expr, term_part)
        };

        // Track combined groups (more than 1 original term merged into 1)
        if originals.len() > 1 {
            combined.push(CombinedGroup {
                key: term_part,
                original_terms: originals,
                combined_term: term,
            });
        }

        new_terms.push(term);
    }

    // Sort terms by global canonical order to match CanonicalizeAddRule
    new_terms.sort_by(|a, b| crate::ordering::compare_expr(ctx, *a, *b));

    let new_expr = if new_terms.is_empty() {
        ctx.num(0)
    } else {
        // Construct Add chain (Right-Associative to match CanonicalizeAddRule)
        let mut result = *new_terms.last().unwrap();
        for t in new_terms.iter().rev().skip(1) {
            result = ctx.add(Expr::Add(*t, result));
        }
        result
    };

    CollectImplResult {
        new_expr,
        cancelled,
        combined,
    }
}

// --- Helpers ---

// flatten_add_chain is now provided by crate::helpers::flatten_add_sub_chain

fn extract_numerical_coeff(ctx: &mut Context, expr: ExprId) -> (BigRational, ExprId) {
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Number(n) => (n, ctx.num(1)), // 5 -> 5 * 1
        Expr::Neg(e) => {
            let (c, t) = extract_numerical_coeff(ctx, e);
            (-c, t)
        }
        Expr::Mul(l, r) => {
            // Check if l is number
            if let Expr::Number(n) = ctx.get(l) {
                (n.clone(), r)
            } else {
                (BigRational::one(), expr)
            }
        }
        _ => (BigRational::one(), expr),
    }
}

fn are_structurally_equal(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    crate::ordering::compare_expr(ctx, a, b) == std::cmp::Ordering::Equal
}

fn is_one_term(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        n.is_one()
    } else {
        false
    }
}

/// Simplify numeric sums in exponents throughout an expression tree.
/// e.g., x^(1/2 + 1/3) → x^(5/6)
/// This is applied during the collect phase for early, visible simplification.
pub fn simplify_numeric_exponents(ctx: &mut Context, expr: ExprId) -> ExprId {
    let expr_data = ctx.get(expr).clone();

    match expr_data {
        Expr::Pow(base, exp) => {
            // Recursively simplify base first
            let new_base = simplify_numeric_exponents(ctx, base);

            // Try to sum numeric fractions in exponent
            if let Some(sum) = try_sum_numeric_fractions(ctx, exp) {
                let new_exp = ctx.add(Expr::Number(sum));
                ctx.add(Expr::Pow(new_base, new_exp))
            } else {
                // Recursively simplify exponent if not purely numeric
                let new_exp = simplify_numeric_exponents(ctx, exp);
                if new_base != base || new_exp != exp {
                    ctx.add(Expr::Pow(new_base, new_exp))
                } else {
                    expr
                }
            }
        }
        Expr::Add(l, r) => {
            let nl = simplify_numeric_exponents(ctx, l);
            let nr = simplify_numeric_exponents(ctx, r);
            if nl != l || nr != r {
                ctx.add(Expr::Add(nl, nr))
            } else {
                expr
            }
        }
        Expr::Sub(l, r) => {
            let nl = simplify_numeric_exponents(ctx, l);
            let nr = simplify_numeric_exponents(ctx, r);
            if nl != l || nr != r {
                ctx.add(Expr::Sub(nl, nr))
            } else {
                expr
            }
        }
        Expr::Mul(l, r) => {
            let nl = simplify_numeric_exponents(ctx, l);
            let nr = simplify_numeric_exponents(ctx, r);
            if nl != l || nr != r {
                mul2_raw(ctx, nl, nr)
            } else {
                expr
            }
        }
        Expr::Div(l, r) => {
            let nl = simplify_numeric_exponents(ctx, l);
            let nr = simplify_numeric_exponents(ctx, r);
            if nl != l || nr != r {
                ctx.add(Expr::Div(nl, nr))
            } else {
                expr
            }
        }
        Expr::Neg(e) => {
            let ne = simplify_numeric_exponents(ctx, e);
            if ne != e {
                ctx.add(Expr::Neg(ne))
            } else {
                expr
            }
        }
        Expr::Function(name, args) => {
            let mut changed = false;
            let new_args: Vec<ExprId> = args
                .iter()
                .map(|a| {
                    let na = simplify_numeric_exponents(ctx, *a);
                    if na != *a {
                        changed = true;
                    }
                    na
                })
                .collect();
            if changed {
                ctx.add(Expr::Function(name, new_args))
            } else {
                expr
            }
        }
        _ => expr, // Number, Variable, Constant, Matrix - no processing needed
    }
}

/// Try to extract and sum all numeric fractions from an Add chain.
/// Returns Some(sum) if the entire chain is numeric fractions, None otherwise.
fn try_sum_numeric_fractions(ctx: &Context, exp: ExprId) -> Option<BigRational> {
    let mut addends: Vec<BigRational> = Vec::new();
    let mut stack = vec![exp];

    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Add(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Number(n) => {
                addends.push(n.clone());
            }
            Expr::Div(num, den) => {
                if let (Expr::Number(n), Expr::Number(d)) = (ctx.get(*num), ctx.get(*den)) {
                    if !d.is_zero() {
                        addends.push(n / d);
                    } else {
                        return None; // Division by zero
                    }
                } else {
                    return None; // Non-numeric fraction
                }
            }
            _ => return None, // Non-numeric term in Add chain
        }
    }

    // Only return sum if there are at least 2 addends (actual simplification)
    if addends.len() >= 2 {
        Some(addends.iter().sum())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::DisplayExpr;
    use cas_parser::parse;

    fn s(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn test_collect_integers() {
        let mut ctx = Context::new();
        let expr = parse("1 + 2", &mut ctx).unwrap();
        let res = collect(&mut ctx, expr);
        assert_eq!(s(&ctx, res), "3");
    }

    #[test]
    fn test_collect_variables() {
        let mut ctx = Context::new();
        let expr = parse("x + x", &mut ctx).unwrap();
        let res = collect(&mut ctx, expr);
        assert_eq!(s(&ctx, res), "2 * x");
    }

    #[test]
    fn test_collect_mixed() {
        let mut ctx = Context::new();
        let expr = parse("2*x + 3*y + 4*x", &mut ctx).unwrap();
        let res = collect(&mut ctx, expr);
        // Order depends on implementation, but should have 6*x and 3*y
        let res_str = s(&ctx, res);
        assert!(res_str.contains("6 * x"));
        assert!(res_str.contains("3 * y"));
    }

    #[test]
    fn test_collect_cancel() {
        let mut ctx = Context::new();
        let expr = parse("x - x", &mut ctx).unwrap();
        let res = collect(&mut ctx, expr);
        assert_eq!(s(&ctx, res), "0");
    }

    #[test]
    fn test_collect_powers() {
        let mut ctx = Context::new();
        let expr = parse("x^2 + 2*x^2", &mut ctx).unwrap();
        let res = collect(&mut ctx, expr);
        assert_eq!(s(&ctx, res), "3 * x^2");
    }

    #[test]
    fn test_simplify_numeric_exponents() {
        let mut ctx = Context::new();
        // x^(1/2 + 1/3) should become x^(5/6)
        let expr = parse("x^(1/2 + 1/3)", &mut ctx).unwrap();
        let res = simplify_numeric_exponents(&mut ctx, expr);
        // The result should be different (simplified)
        assert_ne!(res, expr, "Expression should be simplified");
        assert_eq!(s(&ctx, res), "x^(5/6)");
    }
}
