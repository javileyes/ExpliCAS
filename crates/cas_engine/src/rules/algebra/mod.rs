use crate::define_rule;
use crate::polynomial::Polynomial;
use crate::rule::Rewrite;
use cas_ast::expression::count_nodes;
use cas_ast::{Context, Expr, ExprId};

use num_traits::{One, Zero};

#[cfg(test)]
mod tests;

pub mod helpers;
pub use helpers::*;

pub mod fractions;
pub use fractions::*;

// get_variant_name and is_one are now imported from crate::helpers

define_rule!(ExpandRule, "Expand Polynomial", |ctx, expr| {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if name == "expand" && args.len() == 1 {
            let arg = args[0];
            let new_expr = crate::expand::expand(ctx, arg);
            if new_expr != expr {
                return Some(Rewrite {
                    new_expr,
                    description: "expand()".to_string(),
                    before_local: None,
                    after_local: None,
                });
            } else {
                // If expand didn't change anything, maybe we should just unwrap?
                // "expand(x)" -> "x"
                return Some(Rewrite {
                    new_expr: arg,
                    description: "expand(atom)".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }
        }
    }
    None
});

define_rule!(
    ConservativeExpandRule,
    "Conservative Expand",
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            // If explicit expand() call, always expand
            if name == "expand" && args.len() == 1 {
                let arg = args[0];
                let new_expr = crate::expand::expand(ctx, arg);
                if new_expr != expr {
                    return Some(Rewrite {
                        new_expr,
                        description: "expand()".to_string(),
                        before_local: None,
                        after_local: None,
                    });
                } else {
                    return Some(Rewrite {
                        new_expr: arg,
                        description: "expand(atom)".to_string(),
                        before_local: None,
                        after_local: None,
                    });
                }
            }
        }

        // Implicit expansion (e.g. (x+1)^2)
        // Only expand if complexity does not increase
        let new_expr = crate::expand::expand(ctx, expr);
        if new_expr != expr {
            let old_count = count_nodes(ctx, expr);
            let new_count = count_nodes(ctx, new_expr);

            if new_count <= old_count {
                // Check for structural equality to avoid loops with ID regeneration
                if crate::ordering::compare_expr(ctx, new_expr, expr) == std::cmp::Ordering::Equal {
                    return None;
                }
                return Some(Rewrite {
                    new_expr,
                    description: "Conservative Expansion".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }
        }
        None
    }
);

define_rule!(DistributeRule, "Distributive Property", |ctx, expr| {
    // Skip canonical (elegant) forms - even in aggressive mode
    // e.g., (x+y)*(x-y) should stay factored, not be distributed
    if crate::canonical_forms::is_canonical_form(ctx, expr) {
        return None;
    }
    if let Expr::Mul(l, r) = ctx.get(expr) {
        let l_id = *l;
        let r_id = *r;

        // Try to distribute l into r if r is an Add/Sub
        if matches!(ctx.get(r_id), Expr::Add(_, _) | Expr::Sub(_, _)) {
            let new_expr = distribute(ctx, r_id, l_id);
            if new_expr != expr {
                return Some(Rewrite {
                    new_expr,
                    description: "Distribute (RHS)".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }
        }
        // Try to distribute r into l if l is an Add/Sub
        if matches!(ctx.get(l_id), Expr::Add(_, _) | Expr::Sub(_, _)) {
            let new_expr = distribute(ctx, l_id, r_id);
            if new_expr != expr {
                return Some(Rewrite {
                    new_expr,
                    description: "Distribute (LHS)".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }
        }
    }
    None
});

define_rule!(FactorRule, "Factor Polynomial", |ctx, expr| {
    if let Expr::Function(name, args) = ctx.get(expr) {
        if name == "factor" && args.len() == 1 {
            let arg = args[0];
            // Use the general factor entry point which tries polynomial then diff squares
            let new_expr = crate::factor::factor(ctx, arg);
            if new_expr != arg {
                return Some(Rewrite {
                    new_expr,
                    description: "Factorization".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }
        }
    }
    None
});

define_rule!(
    FactorDifferenceSquaresRule,
    "Factor Difference of Squares",
    |ctx, expr| {
        // 1. Check if it's an Add/Sub chain
        match ctx.get(expr) {
            Expr::Add(_, _) | Expr::Sub(_, _) => {}
            _ => return None,
        }

        // 2. Flatten the chain
        let mut terms = Vec::new();
        crate::helpers::flatten_add(ctx, expr, &mut terms);

        // 3. Separate into potential squares and negative squares
        // We look for pairs (A, B) where A is a square and B is a negative square (B = -C^2)
        // i.e. A - C^2.

        // Optimization: We only need to find ONE pair that works.
        // We can iterate O(N^2) or sort? O(N^2) is fine for small N.

        for i in 0..terms.len() {
            for j in 0..terms.len() {
                if i == j {
                    continue;
                }

                let t1 = terms[i];
                let t2 = terms[j];

                // Check if t1 + t2 forms a difference of squares
                // We construct a temporary Add(t1, t2) and call factor_difference_squares
                // This reuses the existing logic (including get_square_root and is_pythagorean)
                let pair = ctx.add(Expr::Add(t1, t2));

                if let Some(factored) = crate::factor::factor_difference_squares(ctx, pair) {
                    // Complexity check
                    let old_count = count_nodes(ctx, pair);
                    let new_count = count_nodes(ctx, factored);

                    // Smart Check:
                    // If the result is a Mul, it means we factored into (A-B)(A+B).
                    // This usually increases complexity and blocks cancellation (due to DistributeRule guard).
                    // So we require STRICT reduction (<).
                    //
                    // If the result is NOT a Mul, it means we used a Pythagorean identity (sin^2+cos^2=1).
                    // The result is just (A-B). This is a simplification we want, even if size is same.
                    // So we allow SAME size (<=).

                    let is_mul = matches!(ctx.get(factored), Expr::Mul(_, _));
                    let allowed = if is_mul {
                        new_count < old_count
                    } else {
                        new_count <= old_count
                    };

                    if allowed {
                        // Found a pair!
                        // Construct the new expression: Factored + (Terms - {t1, t2})
                        let mut new_terms = Vec::new();
                        new_terms.push(factored);
                        for k in 0..terms.len() {
                            if k != i && k != j {
                                new_terms.push(terms[k]);
                            }
                        }

                        // Rebuild Add chain
                        if new_terms.is_empty() {
                            return Some(Rewrite {
                                new_expr: ctx.num(0),
                                description: "Factor difference of squares (Empty)".to_string(),
                                before_local: None,
                                after_local: None,
                            });
                        }

                        let mut new_expr = new_terms[0];
                        for t in new_terms.iter().skip(1) {
                            new_expr = ctx.add(Expr::Add(new_expr, *t));
                        }

                        return Some(Rewrite {
                            new_expr,
                            description: "Factor difference of squares (N-ary)".to_string(),
                            before_local: None,
                            after_local: None,
                        });
                    }
                }
            }
        }
        None
    }
);

define_rule!(
    AutomaticFactorRule,
    "Automatic Factorization",
    |ctx, expr| {
        // Only try to factor if it's an Add or Sub (polynomial-like)
        match ctx.get(expr) {
            Expr::Add(_, _) | Expr::Sub(_, _) => {}
            _ => return None,
        }

        // Try factor_polynomial first
        if let Some(new_expr) = crate::factor::factor_polynomial(ctx, expr) {
            if new_expr != expr {
                // Complexity check: Only accept if it strictly reduces size
                // This prevents loops with ExpandRule which usually increases size (or keeps it same)
                let old_count = count_nodes(ctx, expr);
                let new_count = count_nodes(ctx, new_expr);

                if new_count < old_count {
                    // Check for structural equality (though unlikely if count reduced)
                    if crate::ordering::compare_expr(ctx, new_expr, expr)
                        == std::cmp::Ordering::Equal
                    {
                        return None;
                    }
                    return Some(Rewrite {
                        new_expr,
                        description: "Automatic Factorization (Reduced Size)".to_string(),
                        before_local: None,
                        after_local: None,
                    });
                }
            }
        }

        // Try difference of squares
        // Note: Diff squares usually increases size: a^2 - b^2 (5) -> (a-b)(a+b) (7)
        // So this will rarely trigger with strict size check unless terms simplify further.
        // e.g. x^4 - 1 -> (x^2-1)(x^2+1) -> (x-1)(x+1)(x^2+1).
        // x^4 - 1 (5 nodes). (x-1)(x+1)(x^2+1) (many nodes).
        // So auto-factoring diff squares is risky for loops.
        // We'll skip it for now in AutomaticFactorRule, or only if it reduces size.
        if let Some(new_expr) = crate::factor::factor_difference_squares(ctx, expr) {
            if new_expr != expr {
                let old_count = count_nodes(ctx, expr);
                let new_count = count_nodes(ctx, new_expr);
                if new_count < old_count {
                    return Some(Rewrite {
                        new_expr,
                        description: "Automatic Factorization (Diff Squares)".to_string(),
                        before_local: None,
                        after_local: None,
                    });
                }
            }
        }

        None
    }
);

// Removed local is_sin_cos_pair, is_negative_term, negate_term as they are now in factor module (or internal to it)
// If other rules need them, I should expose them from factor module or helpers.
// Checking file... is_negative_term was used by FactorDifferenceSquaresRule only.
// negate_term was used by FactorDifferenceSquaresRule only.
// is_sin_cos_pair was used by FactorDifferenceSquaresRule only.
// So safe to remove.

// Tests moved to tests.rs

// Helper function: Check if two expressions are structurally opposite (e.g., a-b vs b-a)

define_rule!(AddFractionsRule, "Add Fractions", |ctx, expr| {
    let expr_data = ctx.get(expr).clone();
    if let Expr::Add(l, r) = expr_data {
        let one = ctx.num(1);

        // Helper to extract num/den
        let mut get_num_den = |e: ExprId| -> (ExprId, ExprId, bool) {
            let expr_data = ctx.get(e).clone();
            match expr_data {
                Expr::Div(n, d) => (n, d, true),
                Expr::Neg(inner) => {
                    match ctx.get(inner).clone() {
                        Expr::Div(n, d) => (ctx.add(Expr::Neg(n)), d, true),
                        Expr::Pow(b, e_inner) => {
                            if let Expr::Number(n_inner) = ctx.get(e_inner) {
                                if n_inner.is_integer()
                                    && *n_inner
                                        == num_rational::BigRational::from_integer((-1).into())
                                {
                                    (ctx.add(Expr::Neg(one)), b, true)
                                } else {
                                    (e, one, false)
                                }
                            } else {
                                (e, one, false)
                            }
                        }
                        Expr::Mul(ml, mr) => {
                            // Check ml * mr^-1
                            let mr_data = ctx.get(mr).clone();
                            if let Expr::Pow(b, e_inner) = mr_data {
                                if let Expr::Number(n_inner) = ctx.get(e_inner) {
                                    if n_inner.is_integer()
                                        && *n_inner
                                            == num_rational::BigRational::from_integer((-1).into())
                                    {
                                        return (ctx.add(Expr::Neg(ml)), b, true);
                                    }
                                }
                            }
                            // Check mr * ml^-1
                            let ml_data = ctx.get(ml).clone();
                            if let Expr::Pow(b, e_inner) = ml_data {
                                if let Expr::Number(n_inner) = ctx.get(e_inner) {
                                    if n_inner.is_integer()
                                        && *n_inner
                                            == num_rational::BigRational::from_integer((-1).into())
                                    {
                                        return (ctx.add(Expr::Neg(mr)), b, true);
                                    }
                                }
                            }
                            (e, one, false)
                        }
                        _ => (e, one, false),
                    }
                }
                Expr::Pow(b, exp) => {
                    if let Expr::Number(n) = ctx.get(exp) {
                        if n.is_integer()
                            && *n == num_rational::BigRational::from_integer((-1).into())
                        {
                            (one, b, true)
                        } else {
                            (e, one, false)
                        }
                    } else {
                        (e, one, false)
                    }
                }
                Expr::Mul(ml, mr) => {
                    // Check ml * mr^-1
                    let mr_data = ctx.get(mr).clone();
                    if let Expr::Pow(b, exp) = mr_data {
                        if let Expr::Number(n) = ctx.get(exp) {
                            if n.is_integer()
                                && *n == num_rational::BigRational::from_integer((-1).into())
                            {
                                return (ml, b, true);
                            }
                        }
                    }
                    // Check mr * ml^-1
                    let ml_data = ctx.get(ml).clone();
                    if let Expr::Pow(b, exp) = ml_data {
                        if let Expr::Number(n) = ctx.get(exp) {
                            if n.is_integer()
                                && *n == num_rational::BigRational::from_integer((-1).into())
                            {
                                return (mr, b, true);
                            }
                        }
                    }
                    (e, one, false)
                }
                _ => (e, one, false),
            }
        };

        let (n1, d1, is_frac1) = get_num_den(l);
        let (n2, d2, is_frac2) = get_num_den(r);

        if !is_frac1 && !is_frac2 {
            return None;
        }
        // eprintln!("AddFractionsRule visiting: {:?}", ctx.get(expr));

        // Check if d2 = -d1 (e.g. x-1 and 1-x) OR d2 == d1
        let (n2, d2, opposite_denom, same_denom) = {
            let mut found_neg = false;
            let mut found_same = false;

            // First check if denominators are exactly equal
            if crate::ordering::compare_expr(ctx, d1, d2) == std::cmp::Ordering::Equal {
                found_same = true;
            }

            // If not same, check if opposite
            if !found_same {
                // First try structural comparison (works for sqrt and other non-polynomials)
                if are_denominators_opposite(ctx, d1, d2) {
                    found_neg = true;
                }

                // Fallback to polynomial check if structural didn't match
                if !found_neg {
                    let vars = collect_variables(ctx, d1);
                    if !vars.is_empty() {
                        for var in vars {
                            if let (Ok(p1), Ok(p2)) = (
                                Polynomial::from_expr(ctx, d1, &var),
                                Polynomial::from_expr(ctx, d2, &var),
                            ) {
                                if p1.add(&p2).is_zero() {
                                    found_neg = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            if found_neg {
                (ctx.add(Expr::Neg(n2)), d1, true, false) // Opposite: negate n2, use d1
            } else if found_same {
                (n2, d1, false, true) // Same: keep n2, use d1
            } else {
                (n2, d2, false, false) // Different: keep both
            }
        };

        // a/b + c/d = (ad + bc) / bd
        // If denominators are same, (a+c)/b

        // Try to compute LCM of denominators to keep expression size small
        let (common_den, mult1, mult2) = {
            let mut result = None;
            let vars = collect_variables(ctx, d1);

            if !vars.is_empty() {
                for var in vars {
                    if let (Ok(p1), Ok(p2)) = (
                        Polynomial::from_expr(ctx, d1, &var),
                        Polynomial::from_expr(ctx, d2, &var),
                    ) {
                        if p1.is_zero() || p2.is_zero() {
                            continue;
                        }
                        let gcd = p1.gcd(&p2);
                        // println!("AddFractions LCM: d1={:?} d2={:?} gcd={:?}", ctx.get(d1), ctx.get(d2), gcd);
                        if gcd.degree() > 0 || !gcd.leading_coeff().is_one() {
                            // Found non-trivial GCD.
                            // LCM = (p1 * p2) / gcd
                            // mult1 = p2 / gcd
                            // mult2 = p1 / gcd

                            let (m1_poly, rem1) = p2.div_rem(&gcd);
                            let (m2_poly, rem2) = p1.div_rem(&gcd);

                            if rem1.is_zero() && rem2.is_zero() {
                                let m1 = m1_poly.to_expr(ctx);
                                let m2 = m2_poly.to_expr(ctx);

                                // common_den = d1 * m1
                                let cd = ctx.add(Expr::Mul(d1, m1));
                                result = Some((cd, m1, m2));
                                break;
                            }
                        }
                    }
                }
            }

            if let Some(res) = result {
                res
            } else {
                // Fallback to naive product
                if crate::ordering::compare_expr(ctx, d1, d2) == std::cmp::Ordering::Equal {
                    (d1, ctx.num(1), ctx.num(1))
                } else if matches!(ctx.get(d1), Expr::Number(n) if n.is_one()) {
                    (d2, d2, ctx.num(1))
                } else if matches!(ctx.get(d2), Expr::Number(n) if n.is_one()) {
                    (d1, ctx.num(1), d1)
                } else {
                    (ctx.add(Expr::Mul(d1, d2)), d2, d1)
                }
            }
        };

        let term1 = ctx.add(Expr::Mul(n1, mult1));
        let term2 = ctx.add(Expr::Mul(n2, mult2));

        let new_num = ctx.add(Expr::Add(term1, term2));
        let new_expr = ctx.add(Expr::Div(new_num, common_den));

        // Complexity check to prevent cycles with DistributeRule and ensure simplification
        let old_complexity = cas_ast::expression::count_nodes(ctx, expr);
        let new_complexity = cas_ast::expression::count_nodes(ctx, new_expr);

        // Check if the new fraction simplifies (shares factors)
        // This allows temporary complexity increase if it leads to simplification
        // Check if the new fraction simplifies (shares factors)
        // This allows temporary complexity increase if it leads to simplification
        let simplifies = |ctx: &mut Context, num: ExprId, den: ExprId| -> bool {
            // If numerator is zero, it simplifies to 0
            if let Expr::Number(n) = ctx.get(num) {
                if n.is_zero() {
                    return true;
                }
            }

            // Check for structural cancellation: A + (-A)
            let add_parts = if let Expr::Add(a, b) = ctx.get(num) {
                Some((*a, *b))
            } else {
                None
            };

            if let Some((a, b)) = add_parts {
                // Helper to check if e1 == -e2
                fn is_negation(ctx: &Context, e1: ExprId, e2: ExprId) -> bool {
                    // Direct negation check
                    if let Expr::Neg(inner) = ctx.get(e1) {
                        if is_sem_equal(ctx, *inner, e2) {
                            return true;
                        }
                    }
                    if let Expr::Neg(inner) = ctx.get(e2) {
                        if is_sem_equal(ctx, *inner, e1) {
                            return true;
                        }
                    }

                    // Helper to expand Neg(Add(a, b)) -> Add(-a, -b) virtual view
                    let get_add_parts_expanded =
                        |e: ExprId| -> Option<(ExprId, ExprId, bool, bool)> {
                            // Returns (l, r, neg_l, neg_r)
                            match ctx.get(e) {
                                Expr::Add(l, r) => Some((*l, *r, false, false)),
                                Expr::Sub(l, r) => Some((*l, *r, false, true)),
                                Expr::Neg(inner) => {
                                    match ctx.get(*inner) {
                                        Expr::Add(l, r) => Some((*l, *r, true, true)),
                                        Expr::Sub(l, r) => Some((*l, *r, true, false)), // -(l - r) = -l + r
                                        _ => None,
                                    }
                                }
                                _ => None,
                            }
                        };

                    if let (Some((l1, r1, nl1, nr1)), Some((l2, r2, nl2, nr2))) =
                        (get_add_parts_expanded(e1), get_add_parts_expanded(e2))
                    {
                        // Check e1 == -e2
                        // e1 = s1*l1 + s2*r1. e2 = s3*l2 + s4*r2.
                        // Need e1 + e2 == 0.
                        // (s1*l1 + s3*l2 == 0 && s2*r1 + s4*r2 == 0) ...

                        let check_term_neg = |t1: ExprId, n1: bool, t2: ExprId, n2: bool| -> bool {
                            // t1 (sign n1) vs t2 (sign n2).
                            // If n1 == n2, need is_negation(t1, t2).
                            // If n1 != n2, need is_equal(t1, t2).
                            if n1 == n2 {
                                is_negation(ctx, t1, t2)
                            } else {
                                crate::ordering::compare_expr(ctx, t1, t2)
                                    == std::cmp::Ordering::Equal
                            }
                        };

                        return (check_term_neg(l1, nl1, l2, nl2)
                            && check_term_neg(r1, nr1, r2, nr2))
                            || (check_term_neg(l1, nl1, r2, nr2)
                                && check_term_neg(r1, nr1, l2, nl2));
                    }

                    // Check Mul(-1, x) vs y
                    let is_mul_neg_of = |e: ExprId, target: ExprId| -> bool {
                        if let Expr::Mul(l, r) = ctx.get(e) {
                            if let Expr::Number(n) = ctx.get(*l) {
                                if n.is_integer()
                                    && *n == num_rational::BigRational::from_integer((-1).into())
                                {
                                    return is_negation(ctx, *r, target);
                                }
                            }
                            if let Expr::Number(n) = ctx.get(*r) {
                                if n.is_integer()
                                    && *n == num_rational::BigRational::from_integer((-1).into())
                                {
                                    return is_negation(ctx, *l, target);
                                }
                            }
                        }
                        false
                    };

                    if is_mul_neg_of(e1, e2) || is_mul_neg_of(e2, e1) {
                        return true;
                    }

                    // Helper to check semantic equality (handling Neg distribution)
                    fn is_sem_equal(ctx: &Context, e1: ExprId, e2: ExprId) -> bool {
                        if crate::ordering::compare_expr(ctx, e1, e2) == std::cmp::Ordering::Equal {
                            return true;
                        }

                        // Check Neg(e1) == Neg(e2) -> is_negation(e1, Neg(e2))? No.
                        // Check e1 == e2.
                        // If e1 = Neg(x). e2 = y. Check x == -y -> is_negation(x, y).
                        if let Expr::Neg(inner) = ctx.get(e1) {
                            return is_negation(ctx, *inner, e2);
                        }
                        if let Expr::Neg(inner) = ctx.get(e2) {
                            return is_negation(ctx, e1, *inner);
                        }

                        // Check Mul(-1, x) == y -> x == -y -> is_negation(x, y)
                        if let Expr::Mul(l, r) = ctx.get(e1) {
                            if let Expr::Number(n) = ctx.get(*l) {
                                if n.is_integer()
                                    && *n == num_rational::BigRational::from_integer((-1).into())
                                {
                                    return is_negation(ctx, *r, e2);
                                }
                            }
                            if let Expr::Number(n) = ctx.get(*r) {
                                if n.is_integer()
                                    && *n == num_rational::BigRational::from_integer((-1).into())
                                {
                                    return is_negation(ctx, *l, e2);
                                }
                            }
                        }
                        if let Expr::Mul(l, r) = ctx.get(e2) {
                            if let Expr::Number(n) = ctx.get(*l) {
                                if n.is_integer()
                                    && *n == num_rational::BigRational::from_integer((-1).into())
                                {
                                    return is_negation(ctx, e1, *r);
                                }
                            }
                            if let Expr::Number(n) = ctx.get(*r) {
                                if n.is_integer()
                                    && *n == num_rational::BigRational::from_integer((-1).into())
                                {
                                    return is_negation(ctx, e1, *l);
                                }
                            }
                        }

                        false
                    }

                    // Check Mul
                    if let (Expr::Mul(l1, r1), Expr::Mul(l2, r2)) = (ctx.get(e1), ctx.get(e2)) {
                        // e1 = l1*r1. e2 = l2*r2.
                        // e1 == -e2.
                        // (l1 == -l2 && r1 == r2) || (l1 == l2 && r1 == -r2) ...

                        // Helper to check equality
                        let eq = |a, b| is_sem_equal(ctx, a, b);
                        // Helper to check negation
                        let neg = |a, b| is_negation(ctx, a, b);

                        // Case 1: l1 matches l2 (equal or neg)
                        if eq(*l1, *l2) {
                            return neg(*r1, *r2);
                        }
                        if neg(*l1, *l2) {
                            return eq(*r1, *r2);
                        }

                        // Case 2: l1 matches r2
                        if eq(*l1, *r2) {
                            return neg(*r1, *l2);
                        }
                        if neg(*l1, *r2) {
                            return eq(*r1, *l2);
                        }
                    }

                    // Check Number negation
                    if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(e1), ctx.get(e2)) {
                        return n1 == &-n2;
                    }

                    // Check Mul(-1, x)
                    let is_mul_neg = |e: ExprId, target: ExprId| -> bool {
                        if let Expr::Mul(l, r) = ctx.get(e) {
                            if let Expr::Number(n) = ctx.get(*l) {
                                if n.is_integer()
                                    && *n == num_rational::BigRational::from_integer((-1).into())
                                {
                                    return *r == target;
                                }
                            }
                            if let Expr::Number(n) = ctx.get(*r) {
                                if n.is_integer()
                                    && *n == num_rational::BigRational::from_integer((-1).into())
                                {
                                    return *l == target;
                                }
                            }
                        }
                        false
                    };

                    if is_mul_neg(e1, e2) || is_mul_neg(e2, e1) {
                        return true;
                    }

                    false
                }

                if is_negation(ctx, a, b) {
                    return true;
                }
            }

            let get_factors = |e: ExprId| -> Vec<ExprId> {
                let mut factors = Vec::new();
                let mut stack = vec![e];
                while let Some(curr) = stack.pop() {
                    match ctx.get(curr) {
                        Expr::Mul(a, b) => {
                            stack.push(*a);
                            stack.push(*b);
                        }
                        Expr::Pow(b, e) => {
                            if let Expr::Number(n) = ctx.get(*e) {
                                if n.is_integer() && *n > num_rational::BigRational::zero() {
                                    stack.push(*b);
                                } else {
                                    factors.push(curr);
                                }
                            } else {
                                factors.push(curr);
                            }
                        }
                        _ => factors.push(curr),
                    }
                }
                factors
            };

            let num_factors = get_factors(num);
            let den_factors = get_factors(den);

            for df in den_factors {
                let found = num_factors.iter().any(|nf| {
                    crate::ordering::compare_expr(ctx, *nf, df) == std::cmp::Ordering::Equal
                });
                if found {
                    return true;
                }

                // Check for numeric GCD
                if let Expr::Number(n_den) = ctx.get(df) {
                    let found_numeric = num_factors.iter().any(|nf| {
                        if let Expr::Number(n_num) = ctx.get(*nf) {
                            if n_num.is_integer() && n_den.is_integer() {
                                let num_int = n_num.to_integer();
                                let den_int = n_den.to_integer();
                                if !num_int.is_zero() && !den_int.is_zero() {
                                    use num_integer::Integer;
                                    let gcd = num_int.gcd(&den_int);
                                    return gcd > One::one();
                                }
                            }
                        }
                        false
                    });
                    if found_numeric {
                        return true;
                    }
                }
            }

            // Polynomial GCD check
            let vars = collect_variables(ctx, num);
            // println!("simplifies: num={:?} vars={:?}", ctx.get(num), vars);
            if !vars.is_empty() {
                for var in vars {
                    if let (Ok(p_num), Ok(p_den)) = (
                        Polynomial::from_expr(ctx, num, &var),
                        Polynomial::from_expr(ctx, den, &var),
                    ) {
                        if p_den.is_zero() {
                            continue;
                        }
                        let gcd = p_num.gcd(&p_den);
                        if gcd.degree() > 0 || !gcd.leading_coeff().is_one() {
                            // Debug: Poly GCD found
                            return true;
                        }
                    }
                }
            }
            false
        };

        let does_simplify = simplifies(ctx, new_num, common_den);
        // Allow simplification if complexity doesn't increase too much (1.5x)
        // OR if we detected opposite/same denominators (always beneficial to combine)
        if opposite_denom
            || same_denom
            || new_complexity <= old_complexity
            || (does_simplify && new_complexity < (old_complexity * 3) / 2)
        {
            return Some(Rewrite {
                new_expr,
                description: "Add fractions: a/b + c/d -> (ad+bc)/bd".to_string(),
                before_local: None,
                after_local: None,
            });
        }
    }
    None
});

define_rule!(
    RationalizeDenominatorRule,
    "Rationalize Denominator",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();

        // Helper to extract num/den from Div, Pow(x, -1), or Mul(x, Pow(y, -1))
        let (num, den) = match expr_data {
            Expr::Div(n, d) => (n, d),
            Expr::Pow(b, e) => {
                if let Expr::Number(n) = ctx.get(e) {
                    if n.is_integer() && *n == num_rational::BigRational::from_integer((-1).into())
                    {
                        (ctx.num(1), b)
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            Expr::Mul(l, r) => {
                // Check l * r^-1
                if let Expr::Pow(b, e) = ctx.get(r) {
                    if let Expr::Number(n) = ctx.get(*e) {
                        if n.is_integer()
                            && *n == num_rational::BigRational::from_integer((-1).into())
                        {
                            (l, *b)
                        } else {
                            // Check r * l^-1
                            if let Expr::Pow(b_l, e_l) = ctx.get(l) {
                                if let Expr::Number(n_l) = ctx.get(*e_l) {
                                    if n_l.is_integer()
                                        && *n_l
                                            == num_rational::BigRational::from_integer((-1).into())
                                    {
                                        (r, *b_l)
                                    } else {
                                        return None;
                                    }
                                } else {
                                    return None;
                                }
                            } else {
                                return None;
                            }
                        }
                    } else {
                        // Check r * l^-1
                        if let Expr::Pow(b_l, e_l) = ctx.get(l) {
                            if let Expr::Number(n_l) = ctx.get(*e_l) {
                                if n_l.is_integer()
                                    && *n_l == num_rational::BigRational::from_integer((-1).into())
                                {
                                    (r, *b_l)
                                } else {
                                    return None;
                                }
                            } else {
                                return None;
                            }
                        } else {
                            return None;
                        }
                    }
                } else {
                    // Check r * l^-1
                    if let Expr::Pow(b_l, e_l) = ctx.get(l) {
                        if let Expr::Number(n_l) = ctx.get(*e_l) {
                            if n_l.is_integer()
                                && *n_l == num_rational::BigRational::from_integer((-1).into())
                            {
                                (r, *b_l)
                            } else {
                                return None;
                            }
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                }
            }
            _ => return None,
        };

        // Check if denominator has roots
        // We look for Add(a, b) or Sub(a, b) where one or both involve roots.
        // Simple case: sqrt(x) + 1, sqrt(x) - 1, sqrt(x) + sqrt(y)

        let den_data = ctx.get(den).clone();
        let (l, r, is_add) = match den_data {
            Expr::Add(l, r) => (l, r, true),
            Expr::Sub(l, r) => (l, r, false),
            _ => return None,
        };

        // Check for roots
        let has_root = |e: ExprId| -> bool {
            match ctx.get(e) {
                Expr::Pow(_, exp) => {
                    if let Expr::Number(n) = ctx.get(*exp) {
                        !n.is_integer()
                    } else {
                        false
                    }
                }
                Expr::Function(name, _) => name == "sqrt",
                Expr::Mul(_, _) => false, // Simplified
                _ => false,
            }
        };

        let l_root = has_root(l);
        let r_root = has_root(r);

        // eprintln!("Rationalize Check: {:?} has_root(l)={} has_root(r)={}", den_data, l_root, r_root);
        // eprintln!("Rationalize Check: {:?} has_root(l)={} has_root(r)={} l={:?} r={:?}", den_data, l_root, r_root, ctx.get(l), ctx.get(r));

        if !l_root && !r_root {
            return None;
        }

        // Construct conjugate
        let conjugate = if is_add {
            ctx.add(Expr::Sub(l, r))
        } else {
            ctx.add(Expr::Add(l, r))
        };

        // Multiply num by conjugate
        let new_num = ctx.add(Expr::Mul(num, conjugate));

        // Compute new den = l^2 - r^2
        // (l+r)(l-r) = l^2 - r^2
        // (l-r)(l+r) = l^2 - r^2
        let two = ctx.num(2);
        let l_sq = ctx.add(Expr::Pow(l, two));
        let r_sq = ctx.add(Expr::Pow(r, two));
        let new_den = ctx.add(Expr::Sub(l_sq, r_sq));

        let new_expr = ctx.add(Expr::Div(new_num, new_den));
        return Some(Rewrite {
            new_expr,
            description: "Rationalize denominator (diff squares)".to_string(),
            before_local: None,
            after_local: None,
        });
    }
);

define_rule!(
    CancelCommonFactorsRule,
    "Cancel Common Factors",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();

        // Helper to collect factors
        fn collect_factors(ctx: &Context, e: ExprId) -> Vec<ExprId> {
            let mut factors = Vec::new();
            let mut stack = vec![e];
            while let Some(curr) = stack.pop() {
                if let Expr::Mul(l, r) = ctx.get(curr) {
                    stack.push(*r);
                    stack.push(*l);
                } else {
                    factors.push(curr);
                }
            }
            factors
        }

        let mut num_factors = Vec::new();
        let mut den_factors = Vec::new();

        match expr_data {
            Expr::Div(n, d) => {
                num_factors = collect_factors(ctx, n);
                den_factors = collect_factors(ctx, d);
            }
            Expr::Pow(b, e) => {
                if let Expr::Number(n) = ctx.get(e) {
                    if n.is_integer() && *n == num_rational::BigRational::from_integer((-1).into())
                    {
                        den_factors = collect_factors(ctx, b);
                        num_factors.push(ctx.num(1));
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            Expr::Mul(_, _) => {
                let factors = collect_factors(ctx, expr);
                for f in factors {
                    if let Expr::Pow(b, e) = ctx.get(f) {
                        if let Expr::Number(n) = ctx.get(*e) {
                            if n.is_integer()
                                && *n == num_rational::BigRational::from_integer((-1).into())
                            {
                                den_factors.extend(collect_factors(ctx, *b));
                                continue;
                            }
                        }
                    }
                    num_factors.push(f);
                }
                if den_factors.is_empty() {
                    return None;
                }
            }
            _ => return None,
        }

        // Helper to simplify trig binomials: 1 - sin^2 -> cos^2, 2 - 2sin^2 -> 2cos^2
        let simplify_trig_binomial = |ctx: &mut Context, e: ExprId| -> Option<ExprId> {
            let mut terms = Vec::new();
            crate::helpers::flatten_add(ctx, e, &mut terms);
            if terms.len() != 2 {
                return None;
            }

            let t1 = terms[0];
            let t2 = terms[1];

            let check_term =
                |ctx: &mut Context, c_term: ExprId, t_term: ExprId| -> Option<ExprId> {
                    // Parse t_term for k * trig^2
                    let (base_term, is_neg) = if let Expr::Neg(inner) = ctx.get(t_term) {
                        (*inner, true)
                    } else {
                        (t_term, false)
                    };

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

                    let mut trig_idx = None;
                    let mut func_name = String::new();
                    let mut arg = None;

                    for (i, &f) in factors.iter().enumerate() {
                        if let Expr::Pow(b, exp) = ctx.get(f) {
                            if let Expr::Number(n) = ctx.get(*exp) {
                                if *n == num_rational::BigRational::from_integer(2.into()) {
                                    if let Expr::Function(name, args) = ctx.get(*b) {
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

                    if let Some(idx) = trig_idx {
                        let arg = arg.unwrap();
                        let mut coeff_factors = Vec::new();
                        if is_neg {
                            coeff_factors.push(ctx.num(-1));
                        }
                        for (i, &f) in factors.iter().enumerate() {
                            if i != idx {
                                coeff_factors.push(f);
                            }
                        }

                        let coeff = if coeff_factors.is_empty() {
                            ctx.num(1)
                        } else {
                            let mut c = coeff_factors[0];
                            for &f in coeff_factors.iter().skip(1) {
                                c = ctx.add(Expr::Mul(c, f));
                            }
                            c
                        };

                        // Check if c_term == -coeff
                        let neg_coeff = if let Expr::Number(n) = ctx.get(coeff) {
                            ctx.add(Expr::Number(-n.clone()))
                        } else if let Expr::Neg(inner) = ctx.get(coeff) {
                            *inner
                        } else {
                            ctx.add(Expr::Neg(coeff))
                        };

                        if crate::ordering::compare_expr(ctx, c_term, neg_coeff)
                            == std::cmp::Ordering::Equal
                        {
                            let other_name = if func_name == "sin" { "cos" } else { "sin" };
                            let func = ctx.add(Expr::Function(other_name.to_string(), vec![arg]));
                            let two = ctx.num(2);
                            let pow = ctx.add(Expr::Pow(func, two));
                            return Some(ctx.add(Expr::Mul(c_term, pow)));
                        }
                    }
                    None
                };

            if let Some(res) = check_term(ctx, t1, t2) {
                return Some(res);
            }
            if let Some(res) = check_term(ctx, t2, t1) {
                return Some(res);
            }
            None
        };

        let mut changed = false;

        // Simplify factors
        let mut new_num_factors = Vec::new();
        for &f in &num_factors {
            if let Some(new_expr) = simplify_trig_binomial(ctx, f) {
                if let Expr::Mul(_, _) = ctx.get(new_expr) {
                    crate::helpers::flatten_mul(ctx, new_expr, &mut new_num_factors);
                } else {
                    new_num_factors.push(new_expr);
                }
                changed = true;
            } else {
                new_num_factors.push(f);
            }
        }
        num_factors = new_num_factors;

        let mut new_den_factors = Vec::new();
        for &f in &den_factors {
            if let Some(new_expr) = simplify_trig_binomial(ctx, f) {
                if let Expr::Mul(_, _) = ctx.get(new_expr) {
                    crate::helpers::flatten_mul(ctx, new_expr, &mut new_den_factors);
                } else {
                    new_den_factors.push(new_expr);
                }
                changed = true;
            } else {
                new_den_factors.push(f);
            }
        }
        den_factors = new_den_factors;

        let mut i = 0;
        while i < num_factors.len() {
            let nf = num_factors[i];
            // println!("Processing num factor: {:?}", ctx.get(nf));
            let mut found = false;
            for j in 0..den_factors.len() {
                let df = den_factors[j];

                // Check exact match
                if crate::ordering::compare_expr(ctx, nf, df) == std::cmp::Ordering::Equal {
                    den_factors.remove(j);
                    found = true;
                    changed = true;
                    break;
                }

                // Check power cancellation: nf = x^n, df = x^m
                // Case 1: nf = base^n, df = base. (n > 1)
                let nf_pow = if let Expr::Pow(b, e) = ctx.get(nf) {
                    Some((*b, *e))
                } else {
                    None
                };
                if let Some((b, e)) = nf_pow {
                    // println!("Checking power cancellation: nf={:?} df={:?}", ctx.get(nf), ctx.get(df));
                    if crate::ordering::compare_expr(ctx, b, df) == std::cmp::Ordering::Equal {
                        // println!("  Base matches!");
                        if let Expr::Number(n) = ctx.get(e) {
                            let new_exp = n - num_rational::BigRational::one();
                            let new_term = if new_exp.is_one() {
                                b
                            } else {
                                let exp_node = ctx.add(Expr::Number(new_exp));
                                ctx.add(Expr::Pow(b, exp_node))
                            };
                            num_factors[i] = new_term;
                            den_factors.remove(j);
                            found = false; // Modified num factor
                            changed = true;
                            break;
                        }
                    }
                }

                // Case 2: nf = base, df = base^m. (m > 1)
                let df_pow = if let Expr::Pow(b, e) = ctx.get(df) {
                    Some((*b, *e))
                } else {
                    None
                };
                if let Some((b, e)) = df_pow {
                    if crate::ordering::compare_expr(ctx, nf, b) == std::cmp::Ordering::Equal {
                        if let Expr::Number(n) = ctx.get(e) {
                            let new_exp = n - num_rational::BigRational::one();
                            let new_term = if new_exp.is_one() {
                                b
                            } else {
                                let exp_node = ctx.add(Expr::Number(new_exp));
                                ctx.add(Expr::Pow(b, exp_node))
                            };
                            den_factors[j] = new_term;
                            found = true; // Remove num factor
                            changed = true;
                            break;
                        }
                    }
                }

                // Case 3: nf = base^n, df = base^m.
                if let Some((b_n, e_n)) = nf_pow {
                    if let Some((b_d, e_d)) = df_pow {
                        if crate::ordering::compare_expr(ctx, b_n, b_d) == std::cmp::Ordering::Equal
                        {
                            if let (Expr::Number(n), Expr::Number(m)) = (ctx.get(e_n), ctx.get(e_d))
                            {
                                if n > m {
                                    let new_exp = n - m;
                                    let new_term = if new_exp.is_one() {
                                        b_n
                                    } else {
                                        let exp_node = ctx.add(Expr::Number(new_exp));
                                        ctx.add(Expr::Pow(b_n, exp_node))
                                    };
                                    num_factors[i] = new_term;
                                    den_factors.remove(j);
                                    found = false;
                                    changed = true;
                                    break;
                                } else if m > n {
                                    let new_exp = m - n;
                                    let new_term = if new_exp.is_one() {
                                        b_d
                                    } else {
                                        let exp_node = ctx.add(Expr::Number(new_exp));
                                        ctx.add(Expr::Pow(b_d, exp_node))
                                    };
                                    den_factors[j] = new_term;
                                    found = true;
                                    changed = true;
                                    break;
                                } else {
                                    den_factors.remove(j);
                                    found = true;
                                    changed = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            if found {
                num_factors.remove(i);
            } else {
                i += 1;
            }
        }

        if changed {
            let new_num = if num_factors.is_empty() {
                ctx.num(1)
            } else {
                let mut n = num_factors[0];
                for &f in num_factors.iter().skip(1) {
                    n = ctx.add(Expr::Mul(n, f));
                }
                n
            };

            let new_den = if den_factors.is_empty() {
                ctx.num(1)
            } else {
                let mut res = den_factors[0];
                for f in den_factors.iter().skip(1) {
                    res = ctx.add(Expr::Mul(res, *f));
                }
                res
            };

            // If denominator is 1, return numerator
            if let Expr::Number(n) = ctx.get(new_den) {
                if n.is_one() {
                    return Some(Rewrite {
                        new_expr: new_num,
                        description: "Cancel common factors (to scalar)".to_string(),
                        before_local: None,
                        after_local: None,
                    });
                }
            }

            let new_expr = ctx.add(Expr::Div(new_num, new_den));
            return Some(Rewrite {
                new_expr,
                description: "Cancel common factors".to_string(),
                before_local: None,
                after_local: None,
            });
        }
        None
    }
);

define_rule!(RootDenestingRule, "Root Denesting", |ctx, expr| {
    let expr_data = ctx.get(expr).clone();

    // We look for sqrt(A + B) or sqrt(A - B)
    // Also handle Pow(inner, 1/2)
    let inner = if let Expr::Function(name, args) = &expr_data {
        if name == "sqrt" && args.len() == 1 {
            Some(args[0])
        } else {
            None
        }
    } else if let Expr::Pow(b, e) = &expr_data {
        if let Expr::Number(n) = ctx.get(*e) {
            if *n.numer() == 1.into() && *n.denom() == 2.into() {
                Some(*b)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    if inner.is_none() {
        return None;
    }
    let inner = inner.unwrap();
    let inner_data = ctx.get(inner).clone();
    //println!("RootDenesting checking inner: {:?}", inner_data);

    let (a, b, is_add) = match inner_data {
        Expr::Add(l, r) => (l, r, true),
        Expr::Sub(l, r) => (l, r, false),
        _ => return None,
    };

    // Helper to identify if a term is C*sqrt(D) or sqrt(D)
    // Returns (Option<C>, D). If C is None, it means 1.
    fn analyze_sqrt_term(
        ctx: &Context,
        e: cas_ast::ExprId,
    ) -> Option<(Option<cas_ast::ExprId>, cas_ast::ExprId)> {
        match ctx.get(e) {
            Expr::Function(fname, fargs) if fname == "sqrt" && fargs.len() == 1 => {
                Some((None, fargs[0]))
            }
            Expr::Pow(b, e) => {
                // Check for b^(3/2) -> b * sqrt(b)
                if let Expr::Number(n) = ctx.get(*e) {
                    // Debug: Checking Pow for root denesting
                    if *n.numer() == 3.into() && *n.denom() == 2.into() {
                        return Some((Some(*b), *b));
                    }
                }
                None
            }
            Expr::Mul(l, r) => {
                // Helper to check for sqrt/pow(1/2)
                let is_sqrt = |e: cas_ast::ExprId| -> Option<cas_ast::ExprId> {
                    match ctx.get(e) {
                        Expr::Function(fname, fargs) if fname == "sqrt" && fargs.len() == 1 => {
                            Some(fargs[0])
                        }
                        Expr::Pow(b, e) => {
                            if let Expr::Number(n) = ctx.get(*e) {
                                if *n.numer() == 1.into() && *n.denom() == 2.into() {
                                    return Some(*b);
                                }
                            }
                            None
                        }
                        _ => None,
                    }
                };

                // Check l * sqrt(r)
                if let Some(inner) = is_sqrt(*r) {
                    return Some((Some(*l), inner));
                }
                // Check sqrt(l) * r
                if let Some(inner) = is_sqrt(*l) {
                    return Some((Some(*r), inner));
                }
                None
            }
            _ => None,
        }
    }

    // We need to determine which is the "rational" part A and which is the "surd" part sqrt(B).
    // Try both permutations.

    // We can't use a closure that captures ctx mutably and calls methods on it easily.
    // So we inline the logic or use a macro/helper that takes ctx.

    let check_permutation = |ctx: &mut Context,
                             term_a: cas_ast::ExprId,
                             term_b: cas_ast::ExprId|
     -> Option<crate::rule::Rewrite> {
        // Assume term_a is A, term_b is C*sqrt(D)
        // We need to call analyze_sqrt_term which takes &Context.
        // But we need to mutate ctx later.
        // So we analyze first.

        let sqrt_parts = analyze_sqrt_term(ctx, term_b);

        if let Some((c_opt, d)) = sqrt_parts {
            // We have sqrt(A +/- C*sqrt(D))
            // Effective B_eff = C^2 * D
            let c = c_opt.unwrap_or_else(|| ctx.num(1));

            // We need numerical values to check the condition
            if let (Expr::Number(val_a), Expr::Number(val_c), Expr::Number(val_d)) =
                (ctx.get(term_a), ctx.get(c), ctx.get(d))
            {
                let val_c2 = val_c * val_c;
                let val_beff = val_c2 * val_d;
                let val_a2 = val_a * val_a;
                let val_delta = val_a2 - val_beff.clone();

                if val_delta >= num_rational::BigRational::zero() {
                    if val_delta.is_integer() {
                        let int_delta = val_delta.to_integer();
                        let sqrt_delta = int_delta.sqrt();

                        if sqrt_delta.clone() * sqrt_delta.clone() == int_delta {
                            // Perfect square!
                            let z_val = ctx.add(Expr::Number(
                                num_rational::BigRational::from_integer(sqrt_delta),
                            ));

                            // Found Z!
                            // Result = sqrt((A+Z)/2) +/- sqrt((A-Z)/2)
                            let two = ctx.num(2);

                            let term1_num = ctx.add(Expr::Add(term_a, z_val));
                            let term1_frac = ctx.add(Expr::Div(term1_num, two));
                            let term1 =
                                ctx.add(Expr::Function("sqrt".to_string(), vec![term1_frac]));

                            let term2_num = ctx.add(Expr::Sub(term_a, z_val));
                            let term2_frac = ctx.add(Expr::Div(term2_num, two));
                            let term2 =
                                ctx.add(Expr::Function("sqrt".to_string(), vec![term2_frac]));

                            // Check sign of C
                            let c_is_negative = if let Expr::Number(n) = ctx.get(c) {
                                n < &num_rational::BigRational::zero()
                            } else {
                                false
                            };

                            // If is_add is true, we have A + C*sqrt(D).
                            // If C is negative, effective operation is subtraction.
                            // If is_add is false, we have A - C*sqrt(D).
                            // If C is negative, effective operation is addition.

                            let effective_sub = if is_add {
                                c_is_negative
                            } else {
                                !c_is_negative
                            };

                            let new_expr = if effective_sub {
                                ctx.add(Expr::Sub(term1, term2))
                            } else {
                                ctx.add(Expr::Add(term1, term2))
                            };

                            return Some(crate::rule::Rewrite {
                                new_expr,
                                description: "Denest square root".to_string(),
                                before_local: None,
                                after_local: None,
                            });
                        }
                    }
                }
            }
        }
        None
    };

    if let Some(rw) = check_permutation(ctx, a, b) {
        return Some(rw);
    }
    if let Some(rw) = check_permutation(ctx, b, a) {
        return Some(rw);
    }
    None
});

// Rule to combine N fractions with binomial product denominators.
// Generalizes the common LCD approach for fractions like:
// - 1/(a-b) + 1/(b-a) → 0
// - 1/((a-b)(a-c)) + 1/((b-c)(b-a)) + 1/((c-a)(c-b)) → 0
// Works with any number of fractions and any number of binomial factors per denominator.
define_rule!(
    FactorBasedLCDRule,
    "Factor-Based LCD",
    Some(vec!["Add"]),
    |ctx, expr| {
        use crate::ordering::compare_expr;
        use std::cmp::Ordering;

        // ===== Helper Functions =====

        // Normalize a binomial to canonical form: (a-b) where a < b alphabetically
        // Returns (canonical_expr, sign_flip) where sign_flip is true if we negated
        let normalize_binomial = |ctx: &mut Context, e: ExprId| -> (ExprId, bool) {
            match ctx.get(e).clone() {
                Expr::Add(l, r) => {
                    if let Expr::Neg(inner) = ctx.get(r).clone() {
                        // Form: l + (-inner) = l - inner
                        if compare_expr(ctx, l, inner) == Ordering::Less {
                            (e, false) // Already canonical
                        } else {
                            // Create: -(inner - l) = (l - inner) negated
                            let neg_l = ctx.add(Expr::Neg(l));
                            let canonical = ctx.add(Expr::Add(inner, neg_l));
                            (canonical, true)
                        }
                    } else {
                        (e, false) // Not a subtraction pattern
                    }
                }
                Expr::Sub(l, r) => {
                    if compare_expr(ctx, l, r) == Ordering::Less {
                        (e, false)
                    } else {
                        let canonical = ctx.add(Expr::Sub(r, l));
                        (canonical, true)
                    }
                }
                _ => (e, false),
            }
        };

        // Extract factors from a product expression
        let get_factors = |ctx: &Context, e: ExprId| -> Vec<ExprId> {
            let mut factors = Vec::new();
            let mut stack = vec![e];
            while let Some(curr) = stack.pop() {
                match ctx.get(curr) {
                    Expr::Mul(l, r) => {
                        stack.push(*l);
                        stack.push(*r);
                    }
                    _ => factors.push(curr),
                }
            }
            factors
        };

        // Check if expression is a binomial (Add with Neg or Sub)
        let is_binomial = |ctx: &Context, e: ExprId| -> bool {
            match ctx.get(e) {
                Expr::Add(_, r) => matches!(ctx.get(*r), Expr::Neg(_)),
                Expr::Sub(_, _) => true,
                _ => false,
            }
        };

        // Check if two expressions are equal (by compare_expr)
        let expr_eq = |ctx: &Context, a: ExprId, b: ExprId| -> bool {
            compare_expr(ctx, a, b) == Ordering::Equal
        };

        // ===== Main Logic =====

        // Collect all terms from the Add tree
        let mut terms = Vec::new();
        let mut stack = vec![expr];
        while let Some(curr) = stack.pop() {
            match ctx.get(curr) {
                Expr::Add(l, r) => {
                    stack.push(*l);
                    stack.push(*r);
                }
                _ => terms.push(curr),
            }
        }

        // Need at least 3 fractions - AddFractionsRule handles 2-fraction cases
        if terms.len() < 3 {
            return None;
        }

        // Extract (numerator, denominator) from each fraction
        let mut fractions: Vec<(ExprId, ExprId)> = Vec::new();
        for term in &terms {
            match ctx.get(*term) {
                Expr::Div(num, den) => fractions.push((*num, *den)),
                _ => return None, // Not all terms are fractions
            }
        }

        // For each denominator, extract and normalize binomial factors
        // Store: Vec<(canonical_factor, sign_flip)> for each fraction
        let mut all_factor_sets: Vec<Vec<(ExprId, bool)>> = Vec::new();

        for (_, den) in &fractions {
            let raw_factors = get_factors(ctx, *den);

            // All factors must be binomials for this rule to apply
            let mut normalized = Vec::new();
            for f in raw_factors {
                if !is_binomial(ctx, f) {
                    return None;
                }
                let (canonical, flipped) = normalize_binomial(ctx, f);
                normalized.push((canonical, flipped));
            }

            if normalized.is_empty() {
                return None;
            }

            all_factor_sets.push(normalized);
        }

        // Collect all unique canonical factors (the LCD factors)
        let mut unique_factors: Vec<ExprId> = Vec::new();
        for factor_set in &all_factor_sets {
            for (canonical, _) in factor_set {
                let exists = unique_factors.iter().any(|u| expr_eq(ctx, *u, *canonical));
                if !exists {
                    unique_factors.push(*canonical);
                }
            }
        }

        // Skip if all fractions already have the same denominator
        let all_same = all_factor_sets.iter().all(|fs| {
            fs.len() == unique_factors.len()
                && unique_factors
                    .iter()
                    .all(|uf| fs.iter().any(|(cf, _)| expr_eq(ctx, *cf, *uf)))
        });
        if all_same && fractions.len() == terms.len() {
            // If fractions share all factors (same LCD), AddFractionsRule handles it
            return None;
        }

        // Build LCD as product of all unique factors
        let lcd = if unique_factors.len() == 1 {
            unique_factors[0]
        } else {
            let mut product = unique_factors[0];
            for i in 1..unique_factors.len() {
                product = ctx.add(Expr::Mul(product, unique_factors[i]));
            }
            product
        };

        // For each fraction, compute numerator contribution
        let mut numerator_terms: Vec<ExprId> = Vec::new();

        for (i, (num, _den)) in fractions.iter().enumerate() {
            let factor_set = &all_factor_sets[i];

            // Compute overall sign from normalization
            let sign_flips: usize = factor_set.iter().filter(|(_, f)| *f).count();
            let is_negative = sign_flips % 2 == 1;

            // Find missing factors (in unique but not in this denominator)
            let mut missing: Vec<ExprId> = Vec::new();
            for uf in &unique_factors {
                let present = factor_set.iter().any(|(cf, _)| expr_eq(ctx, *cf, *uf));
                if !present {
                    missing.push(*uf);
                }
            }

            // Multiply numerator by all missing factors
            let mut contribution = *num;
            for mf in missing {
                contribution = ctx.add(Expr::Mul(contribution, mf));
            }

            // Apply sign
            if is_negative {
                contribution = ctx.add(Expr::Neg(contribution));
            }

            numerator_terms.push(contribution);
        }

        // Sum all numerator contributions
        let total_num = if numerator_terms.len() == 1 {
            numerator_terms[0]
        } else {
            let mut sum = numerator_terms[0];
            for i in 1..numerator_terms.len() {
                sum = ctx.add(Expr::Add(sum, numerator_terms[i]));
            }
            sum
        };

        // Create the combined fraction
        let new_expr = ctx.add(Expr::Div(total_num, lcd));

        Some(Rewrite {
            new_expr,
            description: "Combine fractions with factor-based LCD".to_string(),
            before_local: None,
            after_local: None,
        })
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(SimplifyFractionRule));
    simplifier.add_rule(Box::new(NestedFractionRule));
    simplifier.add_rule(Box::new(SimplifyMulDivRule));
    simplifier.add_rule(Box::new(AddFractionsRule));
    simplifier.add_rule(Box::new(RationalizeDenominatorRule));
    simplifier.add_rule(Box::new(FactorRule));
    simplifier.add_rule(Box::new(CancelCommonFactorsRule));
    simplifier.add_rule(Box::new(RootDenestingRule));
    simplifier.add_rule(Box::new(SimplifySquareRootRule));
    simplifier.add_rule(Box::new(PullConstantFromFractionRule));
    simplifier.add_rule(Box::new(ExpandRule));
    simplifier.add_rule(Box::new(FactorBasedLCDRule));
    // simplifier.add_rule(Box::new(FactorDifferenceSquaresRule)); // Too aggressive for default, causes loops with DistributeRule
}

define_rule!(
    SimplifySquareRootRule,
    "Simplify Square Root",
    |ctx, expr| {
        let arg = if let Expr::Function(name, args) = ctx.get(expr) {
            if name == "sqrt" && args.len() == 1 {
                Some(args[0])
            } else {
                None
            }
        } else if let Expr::Pow(b, e) = ctx.get(expr) {
            if let Expr::Number(n) = ctx.get(*e) {
                if *n.numer() == 1.into() && *n.denom() == 2.into() {
                    Some(*b)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        if let Some(arg) = arg {
            // Only try to factor if argument is Add/Sub (polynomial)
            match ctx.get(arg) {
                Expr::Add(_, _) | Expr::Sub(_, _) => {}
                _ => return None,
            }

            use crate::polynomial::Polynomial;
            use crate::rules::algebra::collect_variables;

            let vars = collect_variables(ctx, arg);
            if vars.len() == 1 {
                let var = vars.iter().next().unwrap();
                if let Ok(poly) = Polynomial::from_expr(ctx, arg, var) {
                    let factors = poly.factor_rational_roots();
                    if !factors.is_empty() {
                        let first = &factors[0];
                        if factors.iter().all(|f| f == first) {
                            let count = factors.len() as u32;
                            if count >= 2 {
                                let base = first.to_expr(ctx);
                                let k = count / 2;
                                let rem = count % 2;

                                let abs_base =
                                    ctx.add(Expr::Function("abs".to_string(), vec![base]));

                                let term1 = if k == 1 {
                                    abs_base
                                } else {
                                    let k_expr = ctx.num(k as i64);
                                    ctx.add(Expr::Pow(abs_base, k_expr))
                                };

                                if rem == 0 {
                                    return Some(Rewrite {
                                        new_expr: term1,
                                        description: "Simplify perfect square root".to_string(),
                                        before_local: None,
                                        after_local: None,
                                    });
                                } else {
                                    let sqrt_base =
                                        ctx.add(Expr::Function("sqrt".to_string(), vec![base]));
                                    let new_expr = ctx.add(Expr::Mul(term1, sqrt_base));
                                    return Some(Rewrite {
                                        new_expr,
                                        description: "Simplify square root factors".to_string(),
                                        before_local: None,
                                        after_local: None,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }
);
