use crate::helpers::{get_variant_name, is_one};
use crate::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed};
use std::cmp::Ordering;

pub(crate) fn gcd_rational(a: BigRational, b: BigRational) -> BigRational {
    if a.is_integer() && b.is_integer() {
        use num_integer::Integer;
        let num_a = a.to_integer();
        let num_b = b.to_integer();
        let g = num_a.gcd(&num_b);
        return BigRational::from_integer(g);
    }
    BigRational::one()
}

/// Count nodes of a specific variant type.
///
/// Uses canonical `cas_ast::traversal::count_nodes_matching` with variant name predicate.
/// (See POLICY.md "Traversal Contract")
pub(crate) fn count_nodes_of_type(ctx: &Context, expr: ExprId, variant: &str) -> usize {
    cas_ast::traversal::count_nodes_matching(ctx, expr, |node| get_variant_name(node) == variant)
}

/// Create a Mul but avoid trivial 1*x or x*1.
///
/// This is the "simplifying" builder - use for rule outputs where 1*x → x is desired.
/// Uses `add_raw` internally to preserve operand order after simplification.
///
/// Alias: `mul2_simpl` (same behavior, clearer intent)
pub(crate) fn smart_mul(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    if is_one(ctx, a) {
        return b;
    }
    if is_one(ctx, b) {
        return a;
    }
    ctx.add_raw(Expr::Mul(a, b))
}

pub(crate) fn distribute(ctx: &mut Context, target: ExprId, multiplier: ExprId) -> ExprId {
    enum Shape {
        Add(ExprId, ExprId),
        Sub(ExprId, ExprId),
        Mul(ExprId, ExprId),
        Div(ExprId, ExprId),
        Other,
    }
    let shape = match ctx.get(target) {
        Expr::Add(l, r) => Shape::Add(*l, *r),
        Expr::Sub(l, r) => Shape::Sub(*l, *r),
        Expr::Mul(l, r) => Shape::Mul(*l, *r),
        Expr::Div(l, r) => Shape::Div(*l, *r),
        _ => Shape::Other,
    };
    match shape {
        Shape::Add(l, r) => {
            let dl = distribute(ctx, l, multiplier);
            let dr = distribute(ctx, r, multiplier);
            ctx.add(Expr::Add(dl, dr))
        }
        Shape::Sub(l, r) => {
            let dl = distribute(ctx, l, multiplier);
            let dr = distribute(ctx, r, multiplier);
            ctx.add(Expr::Sub(dl, dr))
        }
        Shape::Mul(l, r) => {
            // Try to distribute into the side that has denominators
            let l_denoms = collect_denominators(ctx, l);
            if !l_denoms.is_empty() {
                let dl = distribute(ctx, l, multiplier);
                // Chain distribution: result is dl * r = (l*m) * r.
                // We want to distribute dl into r to clear r's denominators if any.
                return distribute(ctx, r, dl);
            }
            let r_denoms = collect_denominators(ctx, r);
            if !r_denoms.is_empty() {
                let dr = distribute(ctx, r, multiplier);
                // Chain distribution: result is l * dr = l * (r*m).
                // Distribute dr into l.
                return distribute(ctx, l, dr);
            }
            // If neither has explicit denominators, just multiply
            smart_mul(ctx, target, multiplier)
        }
        Shape::Div(l, r) => {
            // (l / r) * m.
            // Check if m is a multiple of r.
            if let Some(quotient) = get_quotient(ctx, multiplier, r) {
                // m = q * r.
                // (l / r) * (q * r) = l * q
                return smart_mul(ctx, l, quotient);
            }
            // If not, we are stuck with (l/r)*m.
            let div_expr = ctx.add(Expr::Div(l, r));
            smart_mul(ctx, div_expr, multiplier)
        }
        Shape::Other => smart_mul(ctx, target, multiplier),
    }
}

pub(crate) fn get_quotient(ctx: &mut Context, dividend: ExprId, divisor: ExprId) -> Option<ExprId> {
    if dividend == divisor {
        return Some(ctx.num(1));
    }

    let mul_parts = match ctx.get(dividend) {
        Expr::Mul(l, r) => Some((*l, *r)),
        _ => None,
    };

    if let Some((l, r)) = mul_parts {
        if let Some(q) = get_quotient(ctx, l, divisor) {
            return Some(smart_mul(ctx, q, r));
        }
        if let Some(q) = get_quotient(ctx, r, divisor) {
            return Some(smart_mul(ctx, l, q));
        }
    }
    None
}

pub(crate) fn collect_denominators(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut denoms = Vec::new();
    match ctx.get(expr) {
        Expr::Div(_, den) => {
            denoms.push(*den);
            // Recurse? Maybe not needed for simple cases.
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) => {
            denoms.extend(collect_denominators(ctx, *l));
            denoms.extend(collect_denominators(ctx, *r));
        }
        Expr::Pow(b, e) => {
            // Check for negative exponent?
            if let Expr::Number(n) = ctx.get(*e) {
                if n.is_negative() {
                    // b^-k = 1/b^k. Denominator is b^k (or b if k=-1)
                    // For simplicity, let's just handle 1/x style Divs first.
                }
            }
            denoms.extend(collect_denominators(ctx, *b));
        }
        _ => {}
    }
    denoms
}

// Helper function: Check if two expressions are structurally opposite (e.g., a-b vs b-a)
pub(crate) fn are_denominators_opposite(ctx: &Context, e1: ExprId, e2: ExprId) -> bool {
    match (ctx.get(e1), ctx.get(e2)) {
        // Case 1: (a - b) vs (b - a)
        (Expr::Sub(l1, r1), Expr::Sub(l2, r2)) => {
            compare_expr(ctx, *l1, *r2) == Ordering::Equal
                && compare_expr(ctx, *r1, *l2) == Ordering::Equal
        }
        // Case 2: (-a + b) vs (a - b) where a and b are ANY expressions
        // e.g., -1 + x vs 1 - x OR -x^(1/2) + c vs x^(1/2) - c
        (Expr::Add(l1, r1), Expr::Sub(l2, r2)) => {
            // Pattern: Add(Neg(a), b) vs Sub(a, b)
            // This matches: -a + b vs a - b, which are opposites
            if let Expr::Neg(neg_inner) = ctx.get(*l1) {
                if compare_expr(ctx, *neg_inner, *l2) == Ordering::Equal
                    && compare_expr(ctx, *r1, *r2) == Ordering::Equal
                {
                    // eprintln!("  -> MATCH case 2a: Add(Neg(a), b) vs Sub(a, b)");
                    return true;
                }
            }
            // Also check if l1 is Number(-n) and l2 is Number(n)
            if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(*l1), ctx.get(*l2)) {
                let neg_n2 = -n2.clone();

                if n1 == &neg_n2 && compare_expr(ctx, *r1, *r2) == Ordering::Equal {
                    // eprintln!("  -> MATCH case 2b: Add(Number(-n), b) vs Sub(Number(n), b)");
                    return true;
                }
            }
            false
        }
        // Case 3: Reverse of case 2: Sub(a, b) vs Add(Neg(a), b) or Add(Number(-n), b)
        (Expr::Sub(_, _), Expr::Add(_, _)) => are_denominators_opposite(ctx, e2, e1),
        // Case 4: Both Add - various patterns
        (Expr::Add(l1, r1), Expr::Add(l2, r2)) => {
            // Pattern 4a: Add(Neg(a), b) vs Add(Neg(b), a) -- both terms negated, swapped
            if let (Expr::Neg(neg_l1), Expr::Neg(neg_l2)) = (ctx.get(*l1), ctx.get(*l2)) {
                if compare_expr(ctx, *neg_l1, *r2) == Ordering::Equal
                    && compare_expr(ctx, *r1, *neg_l2) == Ordering::Equal
                {
                    return true;
                }
            }

            // Pattern 4b: Add(Number(-n), x) vs Add(Number(m), Neg(x))
            // e.g., -1 + x vs 1 + (-x) which should be detected as opposite
            if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(*l1), ctx.get(*l2)) {
                // Check if r2 = -r1 and n1 = -n2
                if let Expr::Neg(neg_r2) = ctx.get(*r2) {
                    let neg_n2 = -n2.clone();
                    if n1 == &neg_n2 && compare_expr(ctx, *r1, *neg_r2) == Ordering::Equal {
                        return true;
                    }
                }
                // Also check reverse: r1 = -r2
                if let Expr::Neg(neg_r1) = ctx.get(*r1) {
                    let neg_n2 = -n2.clone();

                    if n1 == &neg_n2 && compare_expr(ctx, *neg_r1, *r2) == Ordering::Equal {
                        return true;
                    }
                }
            }

            // Pattern 4c: Add(a, Neg(b)) vs Add(b, Neg(a))
            // This is the canonical form of (a - b) vs (b - a)
            // e.g., a + (-b) vs b + (-a)
            if let (Expr::Neg(neg_r1), Expr::Neg(neg_r2)) = (ctx.get(*r1), ctx.get(*r2)) {
                // l1 == neg_r2 (a == a) and l2 == neg_r1 (b == b)
                if compare_expr(ctx, *l1, *neg_r2) == Ordering::Equal
                    && compare_expr(ctx, *l2, *neg_r1) == Ordering::Equal
                {
                    return true;
                }
            }
            false
        }
        // Case 5: Mul(a, b) vs Mul(c, d) where one factor matches and one is opposite
        // e.g., (a-b)*(a-c) vs (a-c)*(b-a) → one shared factor (a-c), one opposite (a-b) vs (b-a)
        (Expr::Mul(m1_l, m1_r), Expr::Mul(m2_l, m2_r)) => {
            // Helper to check if two factors are opposite (recursive call)
            let factors_opposite =
                |f1: ExprId, f2: ExprId| -> bool { are_denominators_opposite(ctx, f1, f2) };

            // Check all 4 combinations:
            // 1. m1_l == m2_l and m1_r opposite to m2_r
            if compare_expr(ctx, *m1_l, *m2_l) == Ordering::Equal && factors_opposite(*m1_r, *m2_r)
            {
                return true;
            }
            // 2. m1_l == m2_r and m1_r opposite to m2_l
            if compare_expr(ctx, *m1_l, *m2_r) == Ordering::Equal && factors_opposite(*m1_r, *m2_l)
            {
                return true;
            }
            // 3. m1_r == m2_l and m1_l opposite to m2_r
            if compare_expr(ctx, *m1_r, *m2_l) == Ordering::Equal && factors_opposite(*m1_l, *m2_r)
            {
                return true;
            }
            // 4. m1_r == m2_r and m1_l opposite to m2_l
            if compare_expr(ctx, *m1_r, *m2_r) == Ordering::Equal && factors_opposite(*m1_l, *m2_l)
            {
                return true;
            }
            false
        }
        _ => false,
    }
}
