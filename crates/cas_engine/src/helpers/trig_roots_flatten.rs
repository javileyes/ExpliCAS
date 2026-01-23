// ========== Trig, Roots, and Flatten Helpers ==========

use cas_ast::{Context, Expr, ExprId};
use num_traits::{One, Signed, ToPrimitive};

/// Check if expression is a trigonometric function raised to a specific power.
///
/// # Example
/// ```ignore
/// // Matches sin(x)^2
/// is_trig_pow(ctx, expr, "sin", 2)
/// ```
pub fn is_trig_pow(context: &Context, expr: ExprId, name: &str, power: i64) -> bool {
    if let Expr::Pow(base, exp) = context.get(expr) {
        if let Expr::Number(n) = context.get(*exp) {
            if n.is_integer() && n.to_integer() == power.into() {
                if let Expr::Function(func_name, args) = context.get(*base) {
                    return func_name == name && args.len() == 1;
                }
            }
        }
    }
    false
}

/// Extract the argument from a trigonometric power expression.
///
/// For an expression like `sin(x)^2`, returns `Some(x)`.
pub fn get_trig_arg(context: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Pow(base, _) = context.get(expr) {
        if let Expr::Function(_, args) = context.get(*base) {
            if args.len() == 1 {
                return Some(args[0]);
            }
        }
    }
    None
}

pub fn get_square_root(context: &mut Context, expr: ExprId) -> Option<ExprId> {
    // We need to clone the expression to avoid borrowing issues if we need to inspect it deeply
    // But context.get returns reference.
    // We can't hold reference to context while mutating it.
    // So we should extract necessary data first.

    let expr_data = context.get(expr).clone();

    match expr_data {
        Expr::Pow(b, e) => {
            if let Expr::Number(n) = context.get(e) {
                if n.is_integer() {
                    let val = n.to_integer();
                    if &val % 2 == 0.into() {
                        let two = num_bigint::BigInt::from(2);
                        let new_exp_val = (val / two).to_i64()?;
                        let new_exp = context.num(new_exp_val);

                        // If new_exp is 1, simplify to b
                        if let Expr::Number(ne) = context.get(new_exp) {
                            if ne.is_one() {
                                return Some(b);
                            }
                        }
                        return Some(context.add(Expr::Pow(b, new_exp)));
                    }
                }
            }
            None
        }
        Expr::Number(n) => {
            if n.is_integer() && !n.is_negative() {
                let val = n.to_integer();
                let sqrt = val.sqrt();
                if &sqrt * &sqrt == val {
                    return Some(context.num(sqrt.to_i64()?));
                }
            }
            None
        }
        _ => None,
    }
}

pub fn extract_double_angle_arg(context: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Mul(lhs, rhs) = context.get(expr) {
        if let Expr::Number(n) = context.get(*lhs) {
            if n.is_integer() && n.to_integer() == 2.into() {
                return Some(*rhs);
            }
        }
        if let Expr::Number(n) = context.get(*rhs) {
            if n.is_integer() && n.to_integer() == 2.into() {
                return Some(*lhs);
            }
        }
    }
    None
}

/// Extract inner variable from 3*x pattern (for triple angle identities).
/// Matches both Mul(3, x) and Mul(x, 3).
pub fn extract_triple_angle_arg(context: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Mul(lhs, rhs) = context.get(expr) {
        if let Expr::Number(n) = context.get(*lhs) {
            if n.is_integer() && n.to_integer() == 3.into() {
                return Some(*rhs);
            }
        }
        if let Expr::Number(n) = context.get(*rhs) {
            if n.is_integer() && n.to_integer() == 3.into() {
                return Some(*lhs);
            }
        }
    }
    None
}

/// Extract inner variable from 5*x pattern (for quintuple angle identities).
/// Matches both Mul(5, x) and Mul(x, 5).
pub fn extract_quintuple_angle_arg(context: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Mul(lhs, rhs) = context.get(expr) {
        if let Expr::Number(n) = context.get(*lhs) {
            if n.is_integer() && n.to_integer() == 5.into() {
                return Some(*rhs);
            }
        }
        if let Expr::Number(n) = context.get(*rhs) {
            if n.is_integer() && n.to_integer() == 5.into() {
                return Some(*lhs);
            }
        }
    }
    None
}

/// Flatten an Add chain into a list of terms (simple version).
///
/// This only handles `Add` nodes. For handling `Sub` and `Neg`, use
/// `flatten_add_sub_chain` instead.
/// Uses iterative traversal to prevent stack overflow on deep expressions.
pub fn flatten_add(ctx: &Context, root: ExprId, terms: &mut Vec<ExprId>) {
    let mut stack = vec![root];

    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Add(l, r) => {
                // Push right first so left is processed first
                stack.push(*r);
                stack.push(*l);
            }
            _ => terms.push(expr),
        }
    }
}

/// Flatten an Add/Sub chain into a list of terms, converting subtractions to Neg.
/// This is used by collect and grouping modules for like-term collection.
///
/// Unlike `flatten_add`, this handles:
/// - `Add(a, b)` → [a, b]
/// - `Sub(a, b)` → [a, Neg(b)]
/// - `Neg(Neg(x))` → [x]
pub fn flatten_add_sub_chain(ctx: &mut Context, expr: ExprId) -> Vec<ExprId> {
    let mut terms = Vec::new();
    flatten_add_sub_recursive(ctx, expr, &mut terms, false);
    terms
}

fn flatten_add_sub_recursive(
    ctx: &mut Context,
    expr: ExprId,
    terms: &mut Vec<ExprId>,
    negate: bool,
) {
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Add(l, r) => {
            flatten_add_sub_recursive(ctx, l, terms, negate);
            flatten_add_sub_recursive(ctx, r, terms, negate);
        }
        Expr::Sub(l, r) => {
            flatten_add_sub_recursive(ctx, l, terms, negate);
            flatten_add_sub_recursive(ctx, r, terms, !negate);
        }
        Expr::Neg(inner) => {
            // Handle double negation: Neg(Neg(x)) -> x
            flatten_add_sub_recursive(ctx, inner, terms, !negate);
        }
        _ => {
            if negate {
                terms.push(ctx.add(Expr::Neg(expr)));
            } else {
                terms.push(expr);
            }
        }
    }
}

/// Flatten a Mul chain into a list of factors (simple version).
///
/// This only handles `Mul` nodes. For handling `Neg` as `-1 * expr`,
/// use `flatten_mul_chain` instead.
/// Uses iterative traversal to prevent stack overflow on deep expressions.
pub fn flatten_mul(ctx: &Context, root: ExprId, factors: &mut Vec<ExprId>) {
    let mut stack = vec![root];

    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Mul(l, r) => {
                // Push right first so left is processed first
                stack.push(*r);
                stack.push(*l);
            }
            _ => factors.push(expr),
        }
    }
}

/// Flatten a Mul chain into a list of factors, handling Neg as -1 multiplication.
/// Returns Vec<ExprId> where Neg(e) is converted to [num(-1), ...factors of e...]
pub fn flatten_mul_chain(ctx: &mut Context, expr: ExprId) -> Vec<ExprId> {
    let mut factors = Vec::new();
    flatten_mul_recursive(ctx, expr, &mut factors);
    factors
}

fn flatten_mul_recursive(ctx: &mut Context, expr: ExprId, factors: &mut Vec<ExprId>) {
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Mul(l, r) => {
            flatten_mul_recursive(ctx, l, factors);
            flatten_mul_recursive(ctx, r, factors);
        }
        Expr::Neg(e) => {
            // Treat Neg(e) as -1 * e
            let neg_one = ctx.num(-1);
            factors.push(neg_one);
            flatten_mul_recursive(ctx, e, factors);
        }
        _ => {
            factors.push(expr);
        }
    }
}

pub fn get_parts(context: &mut Context, e: ExprId) -> (num_rational::BigRational, ExprId) {
    match context.get(e) {
        Expr::Mul(a, b) => {
            if let Expr::Number(n) = context.get(*a) {
                (n.clone(), *b)
            } else if let Expr::Number(n) = context.get(*b) {
                (n.clone(), *a)
            } else {
                (num_rational::BigRational::one(), e)
            }
        }
        Expr::Number(n) => (n.clone(), context.num(1)), // Treat constant as c * 1
        _ => (num_rational::BigRational::one(), e),
    }
}
