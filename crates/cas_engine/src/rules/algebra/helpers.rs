use crate::helpers::is_one;
use crate::target_kind::TargetKind;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed};

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
/// Uses canonical `cas_ast::traversal::count_nodes_matching` with `TargetKind` predicate.
/// (See POLICY.md "Traversal Contract")
pub(crate) fn count_nodes_of_type(ctx: &Context, expr: ExprId, kind: TargetKind) -> usize {
    cas_ast::traversal::count_nodes_matching(ctx, expr, |node| TargetKind::from_expr(node) == kind)
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
