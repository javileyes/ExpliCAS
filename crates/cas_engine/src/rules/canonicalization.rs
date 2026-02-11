use crate::build::mul2_raw;
use crate::define_rule;
use crate::helpers::{as_add, as_div, as_mul, as_neg, as_pow, as_sub};
use crate::ordering::compare_expr;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use crate::rules::algebra::helpers::smart_mul;
use cas_ast::Expr;
use num_integer::Integer;
use num_traits::Zero;
use std::cmp::Ordering;

define_rule!(
    CanonicalizeNegationRule,
    "Canonicalize Negation",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        // 1. Subtraction: a - b -> a + (-b)
        if let Some((lhs, rhs)) = as_sub(ctx, expr) {
            let neg_rhs = ctx.add(Expr::Neg(rhs));
            let new_expr = ctx.add(Expr::Add(lhs, neg_rhs));
            return Some(Rewrite::new(new_expr).desc("Convert Subtraction to Addition (a - b -> a + (-b))"));
        }

        // 2. Negation: -x -> -1 * x
        if let Some(inner) = as_neg(ctx, expr) {
            if let Expr::Number(n) = ctx.get(inner) {
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

                let n_display = n.clone();
                let new_expr = ctx.add(Expr::Number(normalized_n.clone()));
                return Some(Rewrite::new(new_expr).desc_lazy(|| format!("-({}) = {}", n_display, normalized_n)));
            }

            // -(-x) -> x
            if let Some(double_inner) = as_neg(ctx, inner) {
                return Some(Rewrite::new(double_inner).desc("-(-x) = x"));
            }

            // -(a + b) -> -a + -b
            if let Some((lhs, rhs)) = as_add(ctx, inner) {
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
                return Some(Rewrite::new(new_expr).desc("-(a + b) = -a - b"));
            }

            // -(c * x) -> (-c) * x
            if let Some((lhs, rhs)) = as_mul(ctx, inner) {
                if let Expr::Number(n) = ctx.get(lhs) {
                    let n = n.clone();
                    let neg_n = -n.clone();
                    let neg_n_expr = ctx.add(Expr::Number(neg_n.clone()));
                    let new_expr = mul2_raw(ctx, neg_n_expr, rhs);
                    return Some(Rewrite::new(new_expr).desc_lazy(|| format!("-({} * x) = {} * x", n, neg_n)));
                }
            }

            // NOTE: DISABLED to break canonicalization loop.
            // The rule -(a/b) → (-a)/b was creating loops with other rules that
            // pull negation out. The canonical form is now -(a/b), not (-a)/b.
            // if let Expr::Div(num, den) = inner_data {
            //     let neg_num = if let Expr::Number(n) = ctx.get(num) {
            //         ctx.add(Expr::Number(-n.clone()))
            //     } else {
            //         ctx.add(Expr::Neg(num))
            //     };
            //     let new_expr = ctx.add(Expr::Div(neg_num, den));
            //     return Some(Rewrite {
            //         new_expr,
            //         description: "-(a/b) = (-a)/b".to_string(),
            //         before_local: None,
            //         after_local: None,
            //            //     });
            // }

            if false {
                // Dummy block to handle the 'else' structure from previous code if needed, but here we just fall through

                // -x -> -x (Keep as Neg)
                // We do NOT want to convert to -1 * x because it's verbose.
                return None;
            }
        }

        // 3. Multiplication: a * (-b) -> -(a * b)
        if let Some((lhs, rhs)) = as_mul(ctx, expr) {
            // Check for (-a) * b
            if let Some(inner_l) = as_neg(ctx, lhs) {
                let new_mul = mul2_raw(ctx, inner_l, rhs);
                let new_expr = ctx.add(Expr::Neg(new_mul));
                return Some(Rewrite::new(new_expr).desc("(-a) * b = -(a * b)"));
            }

            // Check for a * (-b)
            if let Some(inner_r) = as_neg(ctx, rhs) {
                // Special case: if a is a Number, we prefer (-a) * b
                if let Expr::Number(n) = ctx.get(lhs) {
                    let n = n.clone();
                    let neg_n = -n.clone();
                    let neg_n_expr = ctx.add(Expr::Number(neg_n.clone()));
                    let new_expr = mul2_raw(ctx, neg_n_expr, inner_r);
                    return Some(Rewrite::new(new_expr).desc_lazy(|| format!("{} * (-x) = {} * x", n, neg_n)));
                }

                let new_mul = mul2_raw(ctx, lhs, inner_r);
                let new_expr = ctx.add(Expr::Neg(new_mul));
                return Some(Rewrite::new(new_expr).desc("a * (-b) = -(a * b)"));
            }
        }

        // 4. Division: a / (-b) -> -(a / b) [only for denominator, NOT numerator]
        // NOTE: We do NOT convert (-a)/b -> -(a/b) because that creates a loop with
        // the -(a/b) -> (-a)/b rule at line 114. The canonical form is (-a)/b.
        if let Some((lhs, rhs)) = as_div(ctx, expr) {
            if let Some(inner_r) = as_neg(ctx, rhs) {
                let new_div = ctx.add(Expr::Div(lhs, inner_r));
                let new_expr = ctx.add(Expr::Neg(new_div));
                return Some(Rewrite::new(new_expr).desc("a / (-b) = -(a / b)"));
            }
        }

        None
    }
);

define_rule!(CanonicalizeAddRule, "Canonicalize Addition", importance: crate::step::ImportanceLevel::Low, |ctx, expr| {
    if as_add(ctx, expr).is_some() {
        // 1. Flatten
        let mut terms = Vec::new();
        let mut stack = vec![expr];
        while let Some(id) = stack.pop() {
            if let Some((lhs, rhs)) = as_add(ctx, id) {
                stack.push(rhs);
                stack.push(lhs);
            } else {
                terms.push(id);
            }
        }

        // 2. Check if already sorted
        let is_sorted = !terms.windows(2).any(|w| compare_expr(ctx, w[0], w[1]) == Ordering::Greater);

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
            let mut new_expr = match terms.last() {
            Some(t) => *t,
            None => return None,
        };
            for term in terms.iter().rev().skip(1) {
                new_expr = ctx.add(Expr::Add(*term, new_expr));
            }
            return Some(Rewrite::new(new_expr).desc("Sort addition terms"));
        }

        // If it IS sorted, we might still need to fix associativity.
        // e.g. (a+b)+c -> a+(b+c).
        // We can check if the root has an Add as LHS.
        // If LHS is Add, it's left-associative (at the top).
        // We want right-associative, so LHS should NOT be Add (unless it's a parenthesized group, but here we flattened it).
        // Wait, if we flatten, we treat nested Adds as part of the same sum.
        // So if LHS is an Add, it means we have (a+b)+... which we want to convert to a+(b+...).
        if let Some((lhs, _)) = as_add(ctx, expr) {
            if as_add(ctx, lhs).is_some() {
                // Left-associative at root. Rewrite.
                let mut new_expr = match terms.last() {
            Some(t) => *t,
            None => return None,
        };
                for term in terms.iter().rev().skip(1) {
                    new_expr = ctx.add(Expr::Add(*term, new_expr));
                }
                return Some(Rewrite::new(new_expr).desc("Fix associativity (a+b)+c -> a+(b+c)"));
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
            if let Some((_, rhs)) = as_add(ctx, expr) {
                if as_add(ctx, rhs).is_none() {
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
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        use crate::ordering::compare_expr;
        use std::cmp::Ordering;

        if as_mul(ctx, expr).is_some() {
            // 1. Flatten the chain into factors
            let mut factors = Vec::new();
            let mut stack = vec![expr];
            while let Some(id) = stack.pop() {
                if let Some((lhs, rhs)) = as_mul(ctx, id) {
                    stack.push(rhs);
                    stack.push(lhs);
                } else {
                    factors.push(id);
                }
            }

            // CRITICAL: Skip canonicalization if any factor is non-commutative
            // Matrix multiplication is non-commutative: A*B ≠ B*A
            if factors.iter().any(|f| !ctx.is_mul_commutative(*f)) {
                return None;
            }

            // 2. Check if already sorted
            let is_sorted = !factors.windows(2).any(|w| compare_expr(ctx, w[0], w[1]) == Ordering::Greater);

            if !is_sorted {
                // Sort factors canonically
                factors.sort_by(|a, b| compare_expr(ctx, *a, *b));

                // Rebuild right-associative: a*(b*(c*d))
                let mut new_expr = match factors.last() {
            Some(f) => *f,
            None => return None,
        };
                for factor in factors.iter().rev().skip(1) {
                    new_expr = mul2_raw(ctx, *factor, new_expr);
                }

                return Some(Rewrite::new(new_expr).desc("Sort multiplication factors"));
            }

            // 3. Check associativity: (a*b)*c -> a*(b*c)
            if let Some((lhs, _)) = as_mul(ctx, expr) {
                if as_mul(ctx, lhs).is_some() {
                    let mut new_expr = match factors.last() {
            Some(f) => *f,
            None => return None,
        };
                    for factor in factors.iter().rev().skip(1) {
                        new_expr = mul2_raw(ctx, *factor, new_expr);
                    }
                    return Some(Rewrite::new(new_expr).desc("Fix associativity (a*b)*c -> a*(b*c)"));
                }
            }
        }
        None
    }
);

define_rule!(CanonicalizeDivRule, "Canonicalize Division", importance: crate::step::ImportanceLevel::Low, |ctx, expr| {
    if let Some((lhs, rhs)) = as_div(ctx, expr) {
        // x / c -> (1/c) * x
        if let Expr::Number(n) = ctx.get(rhs) {
            let n = n.clone();
            if !n.is_zero() {
                // SPECIAL CASE: (a/b) / c → a / (b·c)
                // Flatten nested division instead of creating (1/c)*(a/b),
                // which would diverge from the canonical flat form a/(b·c).
                // This ensures "(sqrt(5)/x²)/5" and "sqrt(5)/(5·x²)" converge.
                if let Some((inner_num, inner_den)) = as_div(ctx, lhs) {
                    let c_expr = ctx.add(Expr::Number(n.clone()));
                    let new_den = smart_mul(ctx, inner_den, c_expr);
                    let new_expr = ctx.add(Expr::Div(inner_num, new_den));
                    return Some(Rewrite::new(new_expr).desc_lazy(|| format!("(a/b) / {} = a / (b·{})", n, n)));
                }

                let inv = n.recip();
                let inv_expr = ctx.add(Expr::Number(inv));
                let new_expr = smart_mul(ctx, inv_expr, lhs);
                return Some(Rewrite::new(new_expr).desc_lazy(|| format!("x / {} = (1/{}) * x", n, n)));
            }
        }
    }
    None
});

define_rule!(CanonicalizeRootRule, "Canonicalize Roots", importance: crate::step::ImportanceLevel::Low, |ctx, expr| {
    if let Some((fn_id, args)) = crate::helpers::as_function(ctx, expr) {
        if ctx.builtin_of(fn_id) == Some(cas_ast::BuiltinFn::Sqrt) {
            if args.len() == 1 {
                let arg = args[0];

                // Check for simple sqrt(x^2) -> |x|
                if let Some((b, e)) = as_pow(ctx, arg) {
                    if let Expr::Number(n) = ctx.get(e) {
                        if n.is_integer() && n.to_integer().is_even() {
                            let two = ctx.num(2);
                            let k = ctx.add(Expr::Div(e, two));
                            let abs_base = ctx.call_builtin(cas_ast::BuiltinFn::Abs, vec![b]);
                            let new_expr = ctx.add(Expr::Pow(abs_base, k));
                            return Some(Rewrite::new(new_expr).desc("sqrt(x^2k) -> |x|^k"));
                        }
                    }
                }

                // sqrt(x) -> x^(1/2)
                let half = ctx.rational(1, 2);
                let new_expr = ctx.add(Expr::Pow(arg, half));
                return Some(Rewrite::new(new_expr).desc("sqrt(x) = x^(1/2)"));
            } else if args.len() == 2 {
                // sqrt(x, n) -> x^(1/n)
                let (base_arg, index) = (args[0], args[1]);
                if !matches!(ctx.get(index), Expr::Number(_)) {
                    return None;
                }
                let one = ctx.num(1);
                let exp = ctx.add(Expr::Div(one, index));
                let new_expr = ctx.add(Expr::Pow(base_arg, exp));
                return Some(Rewrite::new(new_expr).desc("sqrt(x, n) = x^(1/n)"));
            }
        } else if ctx.builtin_of(fn_id) == Some(cas_ast::BuiltinFn::Root) && args.len() == 2 {
            // root(x, n) -> x^(1/n)
            let (base_arg, index) = (args[0], args[1]);
            if !matches!(ctx.get(index), Expr::Number(_)) {
                return None;
            }
            let one = ctx.num(1);
            let exp = ctx.add(Expr::Div(one, index));
            let new_expr = ctx.add(Expr::Pow(base_arg, exp));
            return Some(Rewrite::new(new_expr).desc("root(x, n) = x^(1/n)"));
        }
    }
    None
});

define_rule!(NormalizeSignsRule, "Normalize Signs", |ctx, expr| {
    // Pattern 1: -c + x -> x - c (if c is positive number)
    if let Some((l, r)) = as_add(ctx, expr) {
        if let Some(inner_neg) = as_neg(ctx, l) {
            if let Expr::Number(n) = ctx.get(inner_neg) {
                if *n > num_rational::BigRational::zero() {
                    let n_clone = n.clone();
                    let new_expr = ctx.add(Expr::Sub(r, inner_neg));
                    return Some(
                        Rewrite::new(new_expr)
                            .desc_lazy(|| format!("-{} + x -> x - {}", n_clone, n_clone)),
                    );
                }
            }
        }
        // Also check the opposite order: x + (-c) -> x - c
        if let Some(inner_neg) = as_neg(ctx, r) {
            if let Expr::Number(n) = ctx.get(inner_neg) {
                if *n > num_rational::BigRational::zero() {
                    let n_clone = n.clone();
                    let new_expr = ctx.add(Expr::Sub(l, inner_neg));
                    return Some(
                        Rewrite::new(new_expr)
                            .desc_lazy(|| format!("x + (-{}) -> x - {}", n_clone, n_clone)),
                    );
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
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        use crate::ordering::compare_expr;
        use std::cmp::Ordering;

        // Pattern: Add(y, Neg(x)) where x < y -> Neg(Add(x, Neg(y)))
        // This converts (y - x) to -(x - y) when x should come first
        if let Some((l, r)) = as_add(ctx, expr) {
            // Check if r is Neg(x) - this is the pattern for (l - x)
            if let Some(inner) = as_neg(ctx, r) {
                // We have: l + (-inner) which represents (l - inner)
                // If inner < l, we should reorder to -(inner - l) = -(inner + (-l))
                if compare_expr(ctx, inner, l) == Ordering::Less {
                    // Create: Neg(inner + Neg(l)) = Neg(inner - l) = -(inner - l)
                    let neg_l = ctx.add(Expr::Neg(l));
                    let inner_minus_l = ctx.add(Expr::Add(inner, neg_l));
                    let new_expr = ctx.add(Expr::Neg(inner_minus_l));
                    return Some(Rewrite::new(new_expr).desc("(y-x) -> -(x-y) for canonical order"));
                }
            }
        }
        None
    }
);

// Rule: -(a - b) → (b - a) ONLY when inner is non-canonical (a > b)
// This prevents the 2-cycle with NormalizeBinomialOrderRule:
// - Normalize: (a-b) → -(b-a) when a > b (produces canonical inner with b < a)
// - Flip: -(a-b) → (b-a) ONLY when a > b (inner is non-canonical)
// Since a > b and b < a are mutually exclusive, no cycle can occur.
define_rule!(
    NegSubFlipRule,
    "Flip Negative Subtraction",
    None,
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        use std::cmp::Ordering;

        let Expr::Neg(inner) = ctx.get(expr) else { return None; };
        let inner_id = *inner;

        // Use as_sub_like to handle both Sub(a,b) and Add(a, Neg(b))
        // Note: can't use `?` operator inside define_rule! closure
        #[allow(clippy::question_mark)]
        let (a, b) = match as_sub_like(ctx, inner_id) {
            Some(pair) => pair,
            None => return None,
        };

        // Guard: only flip if the inner subtraction is NON-canonical (a > b)
        // If a <= b, the inner is already canonical, so don't flip
        if crate::ordering::compare_expr(ctx, a, b) != Ordering::Greater {
            return None;
        }

        // -(a-b) where a > b => (b-a) which is canonical form
        let new_expr = build_sub_like(ctx, b, a);

        Some(Rewrite::new(new_expr).desc("-(a - b) → (b - a) (canonical orientation)").local(inner_id, new_expr))
    }
);

/// Detect (a - b) represented as Sub(a,b) or Add(a, Neg(b)) or Add(Neg(b), a)
/// Returns Some((a, b)) where the expression represents a - b
fn as_sub_like(
    ctx: &cas_ast::Context,
    id: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    match ctx.get(id) {
        Expr::Sub(a, b) => Some((*a, *b)),
        Expr::Add(l, r) => {
            // Check for Add(a, Neg(b)) = a - b
            if let Expr::Neg(x) = ctx.get(*r) {
                return Some((*l, *x));
            }
            // Check for Add(Neg(b), a) = a - b (after canonicalization)
            if let Expr::Neg(x) = ctx.get(*l) {
                return Some((*r, *x));
            }
            None
        }
        _ => None,
    }
}

/// Build (a - b) in the canonical sub-like form: Add(a, Neg(b))
/// This ensures all rules construct subtraction consistently
fn build_sub_like(
    ctx: &mut cas_ast::Context,
    a: cas_ast::ExprId,
    b: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let neg_b = ctx.add(Expr::Neg(b));
    ctx.add(Expr::Add(a, neg_b))
}

/// Peel negation from an expression, returning (core, was_negation).
/// Recognizes:
/// - Neg(t) → (t, true)
/// - Mul(-1, t) or Mul(t, -1) → (t, true)
/// - Sub(a, b) where a > b in canonical order → (Sub(b, a), true)
///   This treats (1 - x) as the negation of (x - 1) when x < 1 canonically
/// - Otherwise → (original, false)
fn peel_negation(ctx: &cas_ast::Context, id: cas_ast::ExprId) -> (cas_ast::ExprId, bool) {
    match ctx.get(id) {
        // Case 1: Explicit Neg(t)
        Expr::Neg(inner) => (*inner, true),

        // Case 2: Mul(-1, t) or Mul(t, -1)
        Expr::Mul(l, r) => {
            let minus_one = num_rational::BigRational::from_integer((-1).into());
            if let Expr::Number(n) = ctx.get(*l) {
                if *n == minus_one {
                    return (*r, true);
                }
            }
            if let Expr::Number(n) = ctx.get(*r) {
                if *n == minus_one {
                    return (*l, true);
                }
            }
            (id, false)
        }

        // Case 3: Sub(a, b) where a > b canonically
        // This means (a - b) = -(b - a), so the "core" is (b - a)
        _ => {
            if let Some((a, b)) = as_sub_like(ctx, id) {
                // Check if a < b in canonical order
                // If so, (a - b) is the "negative form" of (b - a)
                // e.g., (1 - x) = -(x - 1) when 1 < x canonically
                if compare_expr(ctx, a, b) == Ordering::Less {
                    // Return (b - a) as the core, but we need to build it
                    // We can't build here (ctx is immutable), so we return a flag
                    // that the caller can use to build the flipped version
                    return (id, true); // Signal that this is a "negated" sub
                }
            }
            (id, false)
        }
    }
}

/// Build the "un-negated" version of a sub-like expression.
/// For Sub(a, b) where a < b, returns Sub(b, a).
fn build_unnegated_sub(ctx: &mut cas_ast::Context, id: cas_ast::ExprId) -> cas_ast::ExprId {
    if let Some((a, b)) = as_sub_like(ctx, id) {
        // Build (b - a) in canonical form: Add(b, Neg(a))
        build_sub_like(ctx, b, a)
    } else {
        // For Neg(t) or Mul(-1, t), just return the inner
        match ctx.get(id) {
            Expr::Neg(inner) => *inner,
            Expr::Mul(l, r) => {
                let minus_one = num_rational::BigRational::from_integer((-1).into());
                if let Expr::Number(n) = ctx.get(*l) {
                    if *n == minus_one {
                        return *r;
                    }
                }
                if let Expr::Number(n) = ctx.get(*r) {
                    if *n == minus_one {
                        return *l;
                    }
                }
                id
            }
            _ => id,
        }
    }
}

// Rule: (-A)/(-B) → A/B - Cancel double negation in fractions
// This handles cases like (1-√x)/(1-x) → (√x-1)/(x-1)
// by recognizing that (1-√x) is the negation of (√x-1), etc.
//
// No loop risk: produces canonical order which won't match again.
define_rule!(
    CancelFractionSignsRule,
    "Cancel Fraction Signs",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        // Match Div(num, den)
        let Expr::Div(num, den) = ctx.get(expr) else { return None; };
        let num_id = *num;
        let den_id = *den;

        // Check if both num and den are "negations"
        let (_, num_is_neg) = peel_negation(ctx, num_id);
        let (_, den_is_neg) = peel_negation(ctx, den_id);

        if !(num_is_neg && den_is_neg) {
            return None;
        }

        // Both are negations - cancel the double sign
        let new_num = build_unnegated_sub(ctx, num_id);
        let new_den = build_unnegated_sub(ctx, den_id);

        let new_expr = ctx.add(Expr::Div(new_num, new_den));

        Some(Rewrite::new(new_expr).desc("(-A)/(-B) = A/B (cancel double sign)"))
    }
);

// Rule: (-k) * (...) * (a - b) → k * (...) * (b - a) when k > 0
// This produces cleaner output like "1/2 * x * (√2 - 1)" instead of "-1/2 * x * (1 - √2)"
// No loop risk: produces positive coefficient which won't match again
define_rule!(
    NegCoeffFlipBinomialRule,
    "Flip binomial under negative coefficient",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        use num_traits::Signed;

        // Match Mul(l, r) - we work with binary Mul, not flat vec
        if let Expr::Mul(l, r) = ctx.get(expr) {
            let l_id = *l;
            let r_id = *r;

            // Case 1: Left is negative number, right is binomial
            if let Expr::Number(n) = ctx.get(l_id) {
                if n.is_negative() {
                    if let Some((a, b)) = as_sub_like(ctx, r_id) {
                        // (-k) * (a - b) → k * (b - a)
                        let pos_n = ctx.add(Expr::Number(-n.clone()));
                        let neg_a = ctx.add(Expr::Neg(a));
                        let b_minus_a = ctx.add(Expr::Add(b, neg_a));
                        let new_expr = ctx.add(Expr::Mul(pos_n, b_minus_a));
                        return Some(Rewrite::new(new_expr).desc("(-k) * (a-b) → k * (b-a)"));
                    }
                }
            }

            // Case 2: Left is negative number, right is Mul containing binomial
            if let Expr::Number(n) = ctx.get(l_id) {
                if n.is_negative() {
                    let n_clone = n.clone();
                    if let Expr::Mul(ml, mr) = ctx.get(r_id) {
                        let ml_id = *ml;
                        let mr_id = *mr;
                        // (-k) * (x * (a-b)) → k * (x * (b-a))
                        if let Some((a, b)) = as_sub_like(ctx, mr_id) {
                            let pos_n = ctx.add(Expr::Number(-n_clone.clone()));
                            let neg_a = ctx.add(Expr::Neg(a));
                            let b_minus_a = ctx.add(Expr::Add(b, neg_a));
                            let new_inner = ctx.add(Expr::Mul(ml_id, b_minus_a));
                            let new_expr = ctx.add(Expr::Mul(pos_n, new_inner));
                            return Some(Rewrite::new(new_expr).desc("(-k) * (x * (a-b)) → k * (x * (b-a))"));
                        }
                        if let Some((a, b)) = as_sub_like(ctx, ml_id) {
                            let pos_n = ctx.add(Expr::Number(-n_clone));
                            let neg_a = ctx.add(Expr::Neg(a));
                            let b_minus_a = ctx.add(Expr::Add(b, neg_a));
                            let new_inner = ctx.add(Expr::Mul(b_minus_a, mr_id));
                            let new_expr = ctx.add(Expr::Mul(pos_n, new_inner));
                            return Some(Rewrite::new(new_expr).desc("(-k) * ((a-b) * x) → k * ((b-a) * x)"));
                        }
                    }
                }
            }
        }
        None
    }
);
/// ExpToEPowRule: Convert exp(x) → e^x
///
/// GATE: Only applies in RealOnly mode.
/// In ComplexEnabled, exp(z) is univalued while e^z (via pow) could imply
/// multivalued logarithm semantics. Keeping exp() as a function preserves
/// the intended univalued semantics in complex domain.
///
/// This allows ExponentialLogRule to match e^(ln(x)) → x patterns.
pub struct ExpToEPowRule;

impl crate::rule::Rule for ExpToEPowRule {
    fn name(&self) -> &str {
        "Convert exp to Power"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::semantics::ValueDomain;

        // GATE: Only in RealOnly (exp is univalued; ComplexEnabled needs special handling)
        if parent_ctx.value_domain() != ValueDomain::RealOnly {
            return None;
        }

        if let Some((fn_id, args)) = crate::helpers::as_function(ctx, expr) {
            if ctx.builtin_of(fn_id) == Some(cas_ast::BuiltinFn::Exp) && args.len() == 1 {
                let arg = args[0];
                let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                let new_expr = ctx.add(Expr::Pow(e, arg));
                return Some(Rewrite::new(new_expr).desc("exp(x) = e^x"));
            }
        }
        None
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::FUNCTION)
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    // RE-ENABLED: Needed for -0 → 0 normalization
    // The non-determinism issue with Sub→Add(Neg) is now handled by canonical ordering
    simplifier.add_rule(Box::new(CanonicalizeNegationRule));

    simplifier.add_rule(Box::new(CanonicalizeAddRule));
    simplifier.add_rule(Box::new(CanonicalizeMulRule));
    simplifier.add_rule(Box::new(CanonicalizeDivRule));
    simplifier.add_rule(Box::new(CancelFractionSignsRule)); // (-A)/(-B) → A/B
    simplifier.add_rule(Box::new(CanonicalizeRootRule));
    simplifier.add_rule(Box::new(NormalizeSignsRule));
    // NormalizeBinomialOrderRule DISABLED - causes stack overflow in asin_acos tests
    // even with guarded NegSubFlipRule. The cycle likely involves other rules.
    // EvenPowSubSwapRule handles the specific (x-y)^2 - (y-x)^2 = 0 case safely.
    // simplifier.add_rule(Box::new(NormalizeBinomialOrderRule));
    simplifier.add_rule(Box::new(NegSubFlipRule)); // -(a-b) → (b-a) only when a > b
    simplifier.add_rule(Box::new(NegCoeffFlipBinomialRule)); // (-k)*(a-b) → k*(b-a)

    // exp(x) → e^x (RealOnly only - preserves complex semantics)
    simplifier.add_rule(Box::new(ExpToEPowRule));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::{Context, DisplayExpr};

    #[test]
    fn test_canonicalize_negation() {
        let mut ctx = Context::new();
        let rule = CanonicalizeNegationRule;
        // -5 -> -5 (Number)
        // Use add_raw to bypass Context::add's canonicalization which already converts Neg(Number(n)) -> Number(-n)
        let five = ctx.num(5);
        let expr = ctx.add_raw(Expr::Neg(five));
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
        let expr = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x]);
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
        let expr = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x, three]);
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
        let expr2 = ctx.call_builtin(cas_ast::BuiltinFn::Root, vec![x, four]);
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

    #[test]
    fn test_cancel_fraction_signs_explicit_neg() {
        // (-a)/(-b) -> a/b
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let neg_a = ctx.add(Expr::Neg(a));
        let neg_b = ctx.add(Expr::Neg(b));
        let expr = ctx.add(Expr::Div(neg_a, neg_b));

        let rule = CancelFractionSignsRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(rewrite.is_some(), "Should apply to (-a)/(-b)");
        let result = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.unwrap().new_expr
            }
        );
        assert_eq!(result, "a / b");
    }

    #[test]
    fn test_cancel_fraction_signs_sub_implicit() {
        // (1-x)/(1-y) -> (x-1)/(y-1) because 1 < x and 1 < y canonically
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let x = ctx.var("x");
        let y = ctx.var("y");
        // Build Sub(1, x) and Sub(1, y)
        let num = ctx.add(Expr::Sub(one, x));
        let den = ctx.add(Expr::Sub(one, y));
        let expr = ctx.add(Expr::Div(num, den));

        let rule = CancelFractionSignsRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(rewrite.is_some(), "Should apply to (1-x)/(1-y)");
    }

    #[test]
    fn test_cancel_fraction_signs_single_neg_unchanged() {
        // (-a)/b should NOT be changed by this rule (only one is negative)
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let neg_a = ctx.add(Expr::Neg(a));
        let expr = ctx.add(Expr::Div(neg_a, b));

        let rule = CancelFractionSignsRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(rewrite.is_none(), "Should NOT apply to (-a)/b");
    }

    #[test]
    fn test_cancel_fraction_signs_single_neg_den_unchanged() {
        // a/(-b) should NOT be changed by this rule (only one is negative)
        let mut ctx = Context::new();
        let a = ctx.var("a");
        let b = ctx.var("b");
        let neg_b = ctx.add(Expr::Neg(b));
        let expr = ctx.add(Expr::Div(a, neg_b));

        let rule = CancelFractionSignsRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(rewrite.is_none(), "Should NOT apply to a/(-b)");
    }
}
