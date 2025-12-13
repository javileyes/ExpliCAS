use crate::define_rule;
use crate::ordering::compare_expr;
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::Expr;
use num_integer::Integer;
use num_traits::Zero;
use std::cmp::Ordering;

/// Helper: Build a 2-factor product (no normalization, right-assoc, safe for canonicalization).
#[inline]
fn mul2_raw(ctx: &mut cas_ast::Context, a: cas_ast::ExprId, b: cas_ast::ExprId) -> cas_ast::ExprId {
    ctx.add(Expr::Mul(a, b))
}

define_rule!(
    CanonicalizeNegationRule,
    "Canonicalize Negation",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();

        // 1. Subtraction: a - b -> a + (-b)
        if let Expr::Sub(lhs, rhs) = expr_data {
            let neg_rhs = ctx.add(Expr::Neg(rhs));
            let new_expr = ctx.add(Expr::Add(lhs, neg_rhs));
            return Some(Rewrite {
                new_expr,
                description: "Convert Subtraction to Addition (a - b -> a + (-b))".to_string(),
                before_local: None,
                after_local: None,
            });
        }

        // 2. Negation: -x -> -1 * x
        if let Expr::Neg(inner) = expr_data {
            // eprintln!("CanonicalizeNegationRule checking: {:?}", expr);
            let inner_data = ctx.get(inner).clone();
            if let Expr::Number(n) = inner_data {
                // -(-5) -> 5 (Handled by parser usually, but good to have)
                // Actually parser produces Neg(Number(5)).
                // If we have Neg(Number(5)), we want Number(-5).
                let neg_n = -n.clone();

                // CRITICAL: Normalize -0 to 0
                let normalized_n = if neg_n.is_zero() {
                    num_rational::BigRational::from_integer(0.into())
                } else {
                    neg_n
                };

                let new_expr = ctx.add(Expr::Number(normalized_n.clone()));
                return Some(Rewrite {
                    new_expr,
                    description: format!("-({}) = {}", n, normalized_n),
                    before_local: None,
                    after_local: None,
                });
            }

            // -(-x) -> x
            if let Expr::Neg(double_inner) = inner_data {
                return Some(Rewrite {
                    new_expr: double_inner,
                    description: "-(-x) = x".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }

            // -(a + b) -> -a + -b
            if let Expr::Add(lhs, rhs) = inner_data {
                let neg_lhs = if let Expr::Number(n) = ctx.get(lhs) {
                    ctx.add(Expr::Number(-n.clone()))
                } else {
                    ctx.add(Expr::Neg(lhs))
                };

                let neg_rhs = if let Expr::Number(n) = ctx.get(rhs) {
                    ctx.add(Expr::Number(-n.clone()))
                } else {
                    ctx.add(Expr::Neg(rhs))
                };

                let new_expr = ctx.add(Expr::Add(neg_lhs, neg_rhs));
                return Some(Rewrite {
                    new_expr,
                    description: "-(a + b) = -a - b".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }

            // -(c * x) -> (-c) * x
            if let Expr::Mul(lhs, rhs) = inner_data {
                let n_opt = if let Expr::Number(n) = ctx.get(lhs) {
                    Some(n.clone())
                } else {
                    None
                };

                if let Some(n) = n_opt {
                    let neg_n = -n.clone();
                    let neg_n_expr = ctx.add(Expr::Number(neg_n.clone()));
                    let new_expr = mul2_raw(ctx, neg_n_expr, rhs);
                    return Some(Rewrite {
                        new_expr,
                        description: format!("-({} * x) = {} * x", n, neg_n),
                        before_local: None,
                        after_local: None,
                    });
                }
            }

            if false {
                // Dummy block to handle the 'else' structure from previous code if needed, but here we just fall through

                // -x -> -x (Keep as Neg)
                // We do NOT want to convert to -1 * x because it's verbose.
                return None;
            }
        }

        // 3. Multiplication: a * (-b) -> -(a * b)
        if let Expr::Mul(lhs, rhs) = expr_data {
            // Check for (-a) * b
            let lhs_is_neg = if let Expr::Neg(inner) = ctx.get(lhs) {
                Some(*inner)
            } else {
                None
            };
            if let Some(inner_l) = lhs_is_neg {
                let new_mul = mul2_raw(ctx, inner_l, rhs);
                let new_expr = ctx.add(Expr::Neg(new_mul));
                return Some(Rewrite {
                    new_expr,
                    description: "(-a) * b = -(a * b)".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }

            // Check for a * (-b)
            let rhs_is_neg = if let Expr::Neg(inner) = ctx.get(rhs) {
                Some(*inner)
            } else {
                None
            };

            if let Some(inner_r) = rhs_is_neg {
                // Special case: if a is a Number, we prefer (-a) * b
                let n_opt = if let Expr::Number(n) = ctx.get(lhs) {
                    Some(n.clone())
                } else {
                    None
                };

                if let Some(n) = n_opt {
                    let neg_n = -n.clone();
                    let neg_n_expr = ctx.add(Expr::Number(neg_n.clone()));
                    let new_expr = mul2_raw(ctx, neg_n_expr, inner_r);
                    return Some(Rewrite {
                        new_expr,
                        description: format!("{} * (-x) = {} * x", n, neg_n),
                        before_local: None,
                        after_local: None,
                    });
                }

                let new_mul = mul2_raw(ctx, lhs, inner_r);
                let new_expr = ctx.add(Expr::Neg(new_mul));
                return Some(Rewrite {
                    new_expr,
                    description: "a * (-b) = -(a * b)".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }
        }

        // 4. Division: (-a) / b -> -(a / b), a / (-b) -> -(a / b)
        if let Expr::Div(lhs, rhs) = expr_data {
            let lhs_data = ctx.get(lhs);
            let rhs_data = ctx.get(rhs);

            if let Expr::Neg(inner_l) = lhs_data {
                let new_div = ctx.add(Expr::Div(*inner_l, rhs));
                let new_expr = ctx.add(Expr::Neg(new_div));
                return Some(Rewrite {
                    new_expr,
                    description: "(-a) / b = -(a / b)".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }

            if let Expr::Neg(inner_r) = rhs_data {
                let new_div = ctx.add(Expr::Div(lhs, *inner_r));
                let new_expr = ctx.add(Expr::Neg(new_div));
                return Some(Rewrite {
                    new_expr,
                    description: "a / (-b) = -(a / b)".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }
        }

        None
    }
);

define_rule!(CanonicalizeAddRule, "Canonicalize Addition", |ctx, expr| {
    if let Expr::Add(_, _) = ctx.get(expr) {
        // 1. Flatten
        let mut terms = Vec::new();
        let mut stack = vec![expr];
        while let Some(id) = stack.pop() {
            if let Expr::Add(lhs, rhs) = ctx.get(id) {
                stack.push(*rhs);
                stack.push(*lhs);
            } else {
                terms.push(id);
            }
        }

        // 2. Check if already sorted
        let mut is_sorted = true;
        for i in 0..terms.len() - 1 {
            if compare_expr(ctx, terms[i], terms[i + 1]) == Ordering::Greater {
                is_sorted = false;
                break;
            }
        }

        // 3. Check if right-associative (if sorted)
        // If sorted, we only need to rewrite if the structure is NOT right-associative.
        // Right-associative means: t0 + (t1 + (t2 + ...))
        // The flattened traversal above (push rhs, push lhs) produces [t0, t1, t2...] for a right-associative tree.
        // It ALSO produces [t0, t1, t2...] for a left-associative tree ((t0+t1)+t2).
        // So flattening loses structure information.
        // We need to check the structure of `expr` directly?
        // Or just rebuild and compare?
        // Since Context doesn't dedupe, we can't compare IDs.
        // But we can check if we *need* to do anything.

        // If it is NOT sorted, we MUST sort and rebuild.
        if !is_sorted {
            terms.sort_by(|a, b| compare_expr(ctx, *a, *b));
            // Rebuild right-associative
            let mut new_expr = *terms.last().unwrap();
            for term in terms.iter().rev().skip(1) {
                new_expr = ctx.add(Expr::Add(*term, new_expr));
            }
            return Some(Rewrite {
                new_expr,
                description: "Sort addition terms".to_string(),
                before_local: None,
                after_local: None,
            });
        }

        // If it IS sorted, we might still need to fix associativity.
        // e.g. (a+b)+c -> a+(b+c).
        // We can check if the root has an Add as LHS.
        // If LHS is Add, it's left-associative (at the top).
        // We want right-associative, so LHS should NOT be Add (unless it's a parenthesized group, but here we flattened it).
        // Wait, if we flatten, we treat nested Adds as part of the same sum.
        // So if LHS is an Add, it means we have (a+b)+... which we want to convert to a+(b+...).
        if let Expr::Add(lhs, _) = ctx.get(expr) {
            if let Expr::Add(_, _) = ctx.get(*lhs) {
                // Left-associative at root. Rewrite.
                let mut new_expr = *terms.last().unwrap();
                for term in terms.iter().rev().skip(1) {
                    new_expr = ctx.add(Expr::Add(*term, new_expr));
                }
                return Some(Rewrite {
                    new_expr,
                    description: "Fix associativity (a+b)+c -> a+(b+c)".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }
        }

        // Also check if any RHS is NOT an Add (except the last term).
        // In a+(b+(c+d)), RHS of first Add is Add. RHS of second is Add. RHS of third is d (not Add).
        // If we have a+(b+c), terms are [a,b,c].
        // Root: Add(a, X). X should be Add(b, c).
        // If X is NOT Add, but we have > 2 terms, then structure is wrong?
        // No, if X is not Add, it means we only have 2 terms.
        // If terms.len() > 2, then RHS MUST be Add.
        if terms.len() > 2 {
            if let Expr::Add(_, rhs) = ctx.get(expr) {
                if !matches!(ctx.get(*rhs), Expr::Add(_, _)) {
                    // This case is weird if LHS is not Add.
                    // e.g. a + b. terms=[a,b]. len=2.
                    // e.g. a + (b+c). terms=[a,b,c]. len=3. RHS is Add(b,c). OK.
                    // e.g. (a+b) + c. terms=[a,b,c]. LHS is Add. Caught above.
                    // Is there a case where LHS is not Add, but structure is wrong?
                    // Maybe mixed? a + ((b+c) + d)?
                    // Flatten: [a, b, c, d].
                    // Root: Add(a, Y). Y = Add(Add(b,c), d).
                    // Y's LHS is Add.
                    // So recursively, Y would be fixed by this rule when visiting Y.
                    // But we are at Root.
                    // If we only fix Root, Y will be fixed later/before?
                    // Bottom-up simplification means children are simplified first.
                    // So Y is already canonicalized to b+(c+d).
                    // So Root is a + (b+(c+d)).
                    // This is correct.
                    // So checking LHS is Add is sufficient?
                    // Yes, if children are already canonical.
                }
            }
        }
    }
    None
});

define_rule!(
    CanonicalizeMulRule,
    "Canonicalize Multiplication",
    |ctx, expr| {
        use crate::ordering::compare_expr;
        use std::cmp::Ordering;

        if let Expr::Mul(_, _) = ctx.get(expr) {
            // 1. Flatten the chain into factors
            let mut factors = Vec::new();
            let mut stack = vec![expr];
            while let Some(id) = stack.pop() {
                if let Expr::Mul(lhs, rhs) = ctx.get(id) {
                    stack.push(*rhs);
                    stack.push(*lhs);
                } else {
                    factors.push(id);
                }
            }

            // 2. Check if already sorted
            let mut is_sorted = true;
            for i in 0..factors.len().saturating_sub(1) {
                if compare_expr(ctx, factors[i], factors[i + 1]) == Ordering::Greater {
                    is_sorted = false;
                    break;
                }
            }

            if !is_sorted {
                // Sort factors canonically
                factors.sort_by(|a, b| compare_expr(ctx, *a, *b));

                // Rebuild right-associative: a*(b*(c*d))
                let mut new_expr = *factors.last().unwrap();
                for factor in factors.iter().rev().skip(1) {
                    new_expr = mul2_raw(ctx, *factor, new_expr);
                }

                return Some(Rewrite {
                    new_expr,
                    description: "Sort multiplication factors".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }

            // 3. Check associativity: (a*b)*c -> a*(b*c)
            if let Expr::Mul(lhs, _) = ctx.get(expr) {
                if let Expr::Mul(_, _) = ctx.get(*lhs) {
                    let mut new_expr = *factors.last().unwrap();
                    for factor in factors.iter().rev().skip(1) {
                        new_expr = mul2_raw(ctx, *factor, new_expr);
                    }
                    return Some(Rewrite {
                        new_expr,
                        description: "Fix associativity (a*b)*c -> a*(b*c)".to_string(),
                        before_local: None,
                        after_local: None,
                    });
                }
            }
        }
        None
    }
);

define_rule!(CanonicalizeDivRule, "Canonicalize Division", |ctx, expr| {
    let expr_data = ctx.get(expr).clone();
    if let Expr::Div(lhs, rhs) = expr_data {
        // x / c -> (1/c) * x
        let n_opt = if let Expr::Number(n) = ctx.get(rhs) {
            Some(n.clone())
        } else {
            None
        };

        if let Some(n) = n_opt {
            if !n.is_zero() {
                // n is Ratio<BigInt>.
                // We want 1/n.
                // Ratio::recip() exists.
                let inv = n.recip();
                let inv_expr = ctx.add(Expr::Number(inv));
                let new_expr = smart_mul(ctx, inv_expr, lhs);
                return Some(Rewrite {
                    new_expr,
                    description: format!("x / {} = (1/{}) * x", n, n),
                    before_local: None,
                    after_local: None,
                });
            }
        }
    }
    None
});

define_rule!(CanonicalizeRootRule, "Canonicalize Roots", |ctx, expr| {
    let expr_data = ctx.get(expr).clone();
    if let Expr::Function(name, args) = expr_data {
        if name == "sqrt" {
            if args.len() == 1 {
                let arg = args[0];

                // Simplified: Just convert to power.
                // Complex simplification (like sqrt(x^2+2x+1)) belongs in a separate simplification rule.

                // Check for simple sqrt(x^2) -> |x|
                if let Expr::Pow(b, e) = ctx.get(arg).clone() {
                    if let Expr::Number(n) = ctx.get(e) {
                        if n.is_integer() && n.to_integer().is_even() {
                            // sqrt(x^(2k)) -> |x|^k
                            let two = ctx.num(2);
                            let k = ctx.add(Expr::Div(e, two)); // This will simplify to integer
                            let abs_base = ctx.add(Expr::Function("abs".to_string(), vec![b]));
                            let new_expr = ctx.add(Expr::Pow(abs_base, k));
                            return Some(Rewrite {
                                new_expr,
                                description: "sqrt(x^2k) -> |x|^k".to_string(),
                                before_local: None,
                                after_local: None,
                            });
                        }
                    }
                }

                // sqrt(x) -> x^(1/2)
                let half = ctx.rational(1, 2);
                let new_expr = ctx.add(Expr::Pow(args[0], half));
                return Some(Rewrite {
                    new_expr,
                    description: "sqrt(x) = x^(1/2)".to_string(),
                    before_local: None,
                    after_local: None,
                });
            } else if args.len() == 2 {
                // sqrt(x, n) -> x^(1/n)
                let one = ctx.num(1);
                let exp = ctx.add(Expr::Div(one, args[1]));
                let new_expr = ctx.add(Expr::Pow(args[0], exp));
                return Some(Rewrite {
                    new_expr,
                    description: "sqrt(x, n) = x^(1/n)".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }
        } else if name == "root" && args.len() == 2 {
            // root(x, n) -> x^(1/n)
            let one = ctx.num(1);
            let exp = ctx.add(Expr::Div(one, args[1]));
            let new_expr = ctx.add(Expr::Pow(args[0], exp));
            return Some(Rewrite {
                new_expr,
                description: "root(x, n) = x^(1/n)".to_string(),
                before_local: None,
                after_local: None,
            });
        }
    }
    None
});

define_rule!(NormalizeSignsRule, "Normalize Signs", |ctx, expr| {
    // Pattern 1: -c + x -> x - c (if c is positive number)
    if let Expr::Add(l, r) = ctx.get(expr) {
        if let Expr::Neg(inner_neg) = ctx.get(*l) {
            if let Expr::Number(n) = ctx.get(*inner_neg) {
                if *n > num_rational::BigRational::zero() {
                    let n_clone = n.clone();
                    let new_expr = ctx.add(Expr::Sub(*r, *inner_neg));
                    return Some(Rewrite {
                        new_expr,
                        description: format!("-{} + x -> x - {}", n_clone, n_clone),
                        before_local: None,
                        after_local: None,
                    });
                }
            }
        }
        // Also check the opposite order: x + (-c) -> x - c
        if let Expr::Neg(inner_neg) = ctx.get(*r) {
            if let Expr::Number(n) = ctx.get(*inner_neg) {
                if *n > num_rational::BigRational::zero() {
                    let n_clone = n.clone();
                    let new_expr = ctx.add(Expr::Sub(*l, *inner_neg));
                    return Some(Rewrite {
                        new_expr,
                        description: format!("x + (-{}) -> x - {}", n_clone, n_clone),
                        before_local: None,
                        after_local: None,
                    });
                }
            }
        }
    }

    None
});

// Normalize binomial order: (b-a) -> -(a-b) when a < b alphabetically
// This ensures consistent representation of binomials like (y-x) vs (x-y)
// so they can be recognized as opposites in fraction simplification.
define_rule!(
    NormalizeBinomialOrderRule,
    "Normalize Binomial Order",
    |ctx, expr| {
        use crate::ordering::compare_expr;
        use std::cmp::Ordering;

        // Pattern: Add(y, Neg(x)) where x < y -> Neg(Add(x, Neg(y)))
        // This converts (y - x) to -(x - y) when x should come first
        let expr_data = ctx.get(expr).clone();
        if let Expr::Add(l, r) = expr_data {
            let r_data = ctx.get(r).clone();
            // Check if r is Neg(x) - this is the pattern for (l - x)
            if let Expr::Neg(inner) = r_data {
                // We have: l + (-inner) which represents (l - inner)
                // If inner < l, we should reorder to -(inner - l) = -(inner + (-l))
                if compare_expr(ctx, inner, l) == Ordering::Less {
                    // Create: Neg(inner + Neg(l)) = Neg(inner - l) = -(inner - l)
                    let neg_l = ctx.add(Expr::Neg(l));
                    let inner_minus_l = ctx.add(Expr::Add(inner, neg_l));
                    let new_expr = ctx.add(Expr::Neg(inner_minus_l));
                    return Some(Rewrite {
                        new_expr,
                        description: "(y-x) -> -(x-y) for canonical order".to_string(),
                        before_local: None,
                        after_local: None,
                    });
                }
            }
        }
        None
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::{Context, DisplayExpr};
    use cas_parser::parse;

    #[test]
    fn test_canonicalize_negation() {
        let mut ctx = Context::new();
        let rule = CanonicalizeNegationRule;
        // -5 -> -5 (Number)
        let expr = parse("-5", &mut ctx).unwrap(); // Neg(Number(5))
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        // The display might look the same "-5", but the structure is different.
        // Let's check if it's a Number.
        if let Expr::Number(n) = ctx.get(rewrite.new_expr) {
            assert_eq!(format!("{}", n), "-5");
        } else {
            panic!("Expected Number, got {:?}", ctx.get(rewrite.new_expr));
        }
    }

    #[test]
    fn test_canonicalize_sqrt() {
        let mut ctx = Context::new();
        let rule = CanonicalizeRootRule;
        // sqrt(x)
        let x = ctx.var("x");
        let expr = ctx.add(Expr::Function("sqrt".to_string(), vec![x]));
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        // Should be x^(1/2)
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "x^(1/2)"
        );
    }

    #[test]
    fn test_canonicalize_nth_root() {
        let mut ctx = Context::new();
        let rule = CanonicalizeRootRule;

        // sqrt(x, 3) -> x^(1/3)
        let x = ctx.var("x");
        let three = ctx.num(3);
        let expr = ctx.add(Expr::Function("sqrt".to_string(), vec![x, three]));
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "x^(1 / 3)"
        );

        // root(x, 4) -> x^(1/4)
        let four = ctx.num(4);
        let expr2 = ctx.add(Expr::Function("root".to_string(), vec![x, four]));
        let rewrite2 = rule
            .apply(
                &mut ctx,
                expr2,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite2.new_expr
                }
            ),
            "x^(1 / 4)"
        );
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    // RE-ENABLED: Needed for -0 → 0 normalization
    // The non-determinism issue with Sub→Add(Neg) is now handled by canonical ordering
    simplifier.add_rule(Box::new(CanonicalizeNegationRule));

    simplifier.add_rule(Box::new(CanonicalizeAddRule));
    simplifier.add_rule(Box::new(CanonicalizeMulRule));
    simplifier.add_rule(Box::new(CanonicalizeDivRule));
    simplifier.add_rule(Box::new(CanonicalizeRootRule));
    simplifier.add_rule(Box::new(NormalizeSignsRule));
    // NormalizeBinomialOrderRule disabled - causes infinite loop with other rules
}
