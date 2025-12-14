use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::count_nodes;
use cas_ast::Expr;
use std::cmp::Ordering;

/// Check if two expressions form a conjugate pair: (A+B) and (A-B) or vice versa
/// Returns Some((a, b)) if they are conjugates, None otherwise
fn is_conjugate_pair(
    ctx: &cas_ast::Context,
    l: cas_ast::ExprId,
    r: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    use crate::ordering::compare_expr;

    let l_expr = ctx.get(l);
    let r_expr = ctx.get(r);

    match (l_expr, r_expr) {
        (Expr::Add(a1, a2), Expr::Sub(b1, b2)) | (Expr::Sub(b1, b2), Expr::Add(a1, a2)) => {
            // Copy the ExprIds (they're Copy types from pattern match)
            let (a1, a2, b1, b2) = (*a1, *a2, *b1, *b2);

            // Direct match: (A+B) vs (A-B)
            if compare_expr(ctx, a1, b1) == Ordering::Equal
                && compare_expr(ctx, a2, b2) == Ordering::Equal
            {
                return Some((a1, a2));
            }
            // Commutative: (B+A) vs (A-B) → A=b1, B=a1
            if compare_expr(ctx, a2, b1) == Ordering::Equal
                && compare_expr(ctx, a1, b2) == Ordering::Equal
            {
                return Some((b1, b2));
            }
            None
        }
        // Handle canonicalized form: Sub(a, b) becomes Add(-b, a) or Add(a, -b)
        (Expr::Add(a1, a2), Expr::Add(b1, b2)) => {
            let (a1, a2, b1, b2) = (*a1, *a2, *b1, *b2);

            // Case 1: (A+B) vs (A+(-B)) where b2 = -a2
            if is_negation(ctx, a2, b2) && compare_expr(ctx, a1, b1) == Ordering::Equal {
                return Some((a1, a2));
            }
            // Case 2: (A+B) vs ((-B)+A) where b1 = -a2
            if is_negation(ctx, a2, b1) && compare_expr(ctx, a1, b2) == Ordering::Equal {
                return Some((a1, a2));
            }
            // Case 3: (A+B) vs (B+(-A)) where b2 = -a1
            if is_negation(ctx, a1, b2) && compare_expr(ctx, a2, b1) == Ordering::Equal {
                return Some((a2, a1)); // b^2 - a^2
            }
            // Case 4: (A+B) vs ((-A)+B) where b1 = -a1
            if is_negation(ctx, a1, b1) && compare_expr(ctx, a2, b2) == Ordering::Equal {
                return Some((a2, a1));
            }
            None
        }
        _ => None,
    }
}

/// Check if `b` is the negation of `a` (Neg(a) or Mul(-1, a) or Number(-n) vs Number(n))
fn is_negation(ctx: &cas_ast::Context, a: cas_ast::ExprId, b: cas_ast::ExprId) -> bool {
    use num_bigint::BigInt;
    use num_rational::BigRational;

    // Check numeric negation: Number(n) vs Number(-n)
    if let (Expr::Number(n_a), Expr::Number(n_b)) = (ctx.get(a), ctx.get(b)) {
        if n_a == &(-n_b.clone()) {
            return true;
        }
    }

    match ctx.get(b) {
        Expr::Neg(inner) if *inner == a => true,
        Expr::Mul(l, r) => {
            // Check for -1 * a or a * -1
            let l_id = *l;
            let r_id = *r;
            (is_minus_one(ctx, l_id) && r_id == a) || (is_minus_one(ctx, r_id) && l_id == a)
        }
        _ => false,
    }
}

/// Check if expression is -1
fn is_minus_one(ctx: &cas_ast::Context, e: cas_ast::ExprId) -> bool {
    use num_bigint::BigInt;
    use num_rational::BigRational;
    if let Expr::Number(n) = ctx.get(e) {
        n == &BigRational::from_integer(BigInt::from(-1))
    } else if let Expr::Neg(inner) = ctx.get(e) {
        if let Expr::Number(n) = ctx.get(*inner) {
            n == &BigRational::from_integer(BigInt::from(1))
        } else {
            false
        }
    } else {
        false
    }
}

// DifferenceOfSquaresRule: Expands conjugate products
// (a - b) * (a + b) → a² - b²
// Phase: CORE | POST (structural simplification, not expansion)
define_rule!(
    DifferenceOfSquaresRule,
    "Difference of Squares (Product to Difference)",
    None,
    PhaseMask::CORE | PhaseMask::POST,
    |ctx, expr| {
        // Match Mul(l, r) where l and r are conjugate binomials
        if let Expr::Mul(l, r) = ctx.get(expr) {
            let l = *l;
            let r = *r;

            if let Some((a, b)) = is_conjugate_pair(ctx, l, r) {
                // Create a² - b²
                let two = ctx.num(2);
                let a_squared = ctx.add(Expr::Pow(a, two));
                let b_squared = ctx.add(Expr::Pow(b, two));
                let new_expr = ctx.add(Expr::Sub(a_squared, b_squared));

                return Some(Rewrite {
                    new_expr,
                    description: "(a-b)(a+b) = a² - b²".to_string(),
                    before_local: None,
                    after_local: None, domain_assumption: None,
                });
            }
        }
        None
    }
);

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
                    after_local: None, domain_assumption: None,
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
        // match Expr::Add(l, r)
        let expr_data = ctx.get(expr).clone();
        println!("FactorDifferenceSquaresRule visiting: {:?}", ctx.get(expr));
        if let Expr::Add(_, _) | Expr::Sub(_, _) = expr_data {
            // Check
        } else {
            return None;
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
                                after_local: None, domain_assumption: None,
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
                            after_local: None, domain_assumption: None,
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
                        after_local: None, domain_assumption: None,
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
                        after_local: None, domain_assumption: None,
                    });
                }
            }
        }

        None
    }
);
