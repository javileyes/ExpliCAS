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
pub(crate) fn is_trig_pow(context: &Context, expr: ExprId, name: &str, power: i64) -> bool {
    if let Expr::Pow(base, exp) = context.get(expr) {
        if let Expr::Number(n) = context.get(*exp) {
            if n.is_integer() && n.to_integer() == power.into() {
                if let Expr::Function(func_name, args) = context.get(*base) {
                    return context
                        .builtin_of(*func_name)
                        .is_some_and(|b| b.name() == name)
                        && args.len() == 1;
                }
            }
        }
    }
    false
}

/// Extract the argument from a trigonometric power expression.
///
/// For an expression like `sin(x)^2`, returns `Some(x)`.
pub(crate) fn get_trig_arg(context: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Pow(base, _) = context.get(expr) {
        if let Expr::Function(_, args) = context.get(*base) {
            if args.len() == 1 {
                return Some(args[0]);
            }
        }
    }
    None
}

pub(crate) fn get_square_root(context: &mut Context, expr: ExprId) -> Option<ExprId> {
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

/// Extract an *exact* integer factor `±target` from a single term.
///
/// Returns `Some((is_positive, inner))` if the term is:
/// - `Mul(Number(±target), inner)` or `Mul(inner, Number(±target))`
/// - `Number(±target)`  (inner = sentinel, caller replaces with 1)
/// - `Neg(x)` → inverts sign and retries on `x`
///
/// This is a building-block for `extract_int_multiple_additive`.
fn extract_exact_int_factor(
    context: &Context,
    term: ExprId,
    target_big: &num_bigint::BigInt,
    neg_target_big: &num_bigint::BigInt,
) -> Option<(bool, ExprId)> {
    // Neg(x) → invert sign and retry
    if let Expr::Neg(inner) = context.get(term) {
        return extract_exact_int_factor(context, *inner, target_big, neg_target_big)
            .map(|(sign, id)| (!sign, id));
    }

    // Number(±target) → (sign, sentinel)
    if let Expr::Number(n) = context.get(term) {
        if n.is_integer() {
            let val = n.to_integer();
            if val == *target_big {
                return Some((true, term)); // sentinel — caller replaces with 1
            }
            if val == *neg_target_big {
                return Some((false, term)); // sentinel — caller replaces with 1
            }
        }
    }

    // Mul(Number(±target), inner) or Mul(inner, Number(±target))
    if let Expr::Mul(lhs, rhs) = context.get(term) {
        if let Expr::Number(n) = context.get(*lhs) {
            if n.is_integer() {
                let val = n.to_integer();
                if val == *target_big {
                    return Some((true, *rhs));
                }
                if val == *neg_target_big {
                    return Some((false, *rhs));
                }
            }
        }
        if let Expr::Number(n) = context.get(*rhs) {
            if n.is_integer() {
                let val = n.to_integer();
                if val == *target_big {
                    return Some((true, *lhs));
                }
                if val == *neg_target_big {
                    return Some((false, *lhs));
                }
            }
        }
    }
    None
}

/// Extract inner variable from `±target * x` pattern for any small positive integer `target`.
///
/// Returns `Some((is_positive, inner))` if the expression matches:
/// - `Mul(Number(k), inner)` or `Mul(inner, Number(k))` where `|k| == target`
/// - `Neg(Mul(Number(k), inner))` where `k == target`
///
/// `is_positive` is `true` when the sign is `+target`, `false` when `-target`.
///
/// **Note**: This function does NOT handle additive forms like `Add(Mul(2,a), 2)`.
/// Use [`extract_int_multiple_additive`] for that (requires `&mut Context`).
pub(crate) fn extract_int_multiple(
    context: &Context,
    expr: ExprId,
    target: i64,
) -> Option<(bool, ExprId)> {
    debug_assert!(target > 0, "target must be a positive integer");
    let target_big: num_bigint::BigInt = target.into();
    let neg_target_big: num_bigint::BigInt = (-target).into();

    // Check Neg(Mul(target, inner))
    if let Expr::Neg(inner_neg) = context.get(expr) {
        if let Expr::Mul(lhs, rhs) = context.get(*inner_neg) {
            if let Expr::Number(n) = context.get(*lhs) {
                if n.is_integer() && n.to_integer() == target_big {
                    return Some((false, *rhs));
                }
            }
            if let Expr::Number(n) = context.get(*rhs) {
                if n.is_integer() && n.to_integer() == target_big {
                    return Some((false, *lhs));
                }
            }
        }
        return None;
    }

    // Check Mul(±target, inner) or Mul(inner, ±target)
    if let Expr::Mul(lhs, rhs) = context.get(expr) {
        if let Expr::Number(n) = context.get(*lhs) {
            if n.is_integer() {
                let val = n.to_integer();
                if val == target_big {
                    return Some((true, *rhs));
                }
                if val == neg_target_big {
                    return Some((false, *rhs));
                }
            }
        }
        if let Expr::Number(n) = context.get(*rhs) {
            if n.is_integer() {
                let val = n.to_integer();
                if val == target_big {
                    return Some((true, *lhs));
                }
                if val == neg_target_big {
                    return Some((false, *lhs));
                }
            }
        }
    }
    None
}

/// Like [`extract_int_multiple`] but also handles **additive** forms where
/// all terms share a common integer factor `target`.
///
/// Recognized extra patterns (2-term Add/Sub only):
/// - `Add(Mul(k, a), Mul(k, b))` → `(true, Add(a, b))`
/// - `Add(Mul(k, a), k)` → `(true, Add(a, 1))`
/// - `Sub(Mul(k, a), k)` → `(true, Sub(a, 1))`
/// - `Add(x, Neg(y))` is treated as `Sub(x, y)`
///
/// Requires `&mut Context` because it may build new AST nodes for the inner sum.
pub(crate) fn extract_int_multiple_additive(
    context: &mut Context,
    expr: ExprId,
    target: i64,
) -> Option<(bool, ExprId)> {
    // First try the basic Mul/Neg patterns (no allocation needed)
    if let Some(result) = extract_int_multiple(context, expr, target) {
        return Some(result);
    }

    // Additive fallback: Add/Sub of two terms sharing factor `target`
    let target_big: num_bigint::BigInt = target.into();
    let neg_target_big: num_bigint::BigInt = (-target).into();

    // Normalize Add(x, Neg(y)) → treat as Sub(x, y)
    let (lhs, rhs, is_sub) = match context.get(expr) {
        Expr::Add(l, r) => {
            if let Expr::Neg(neg_inner) = context.get(*r) {
                (*l, *neg_inner, true)
            } else {
                (*l, *r, false)
            }
        }
        Expr::Sub(l, r) => (*l, *r, true),
        _ => return None,
    };

    // Extract factor from each term
    let (l_sign, l_inner) = extract_exact_int_factor(context, lhs, &target_big, &neg_target_big)?;
    let (r_sign, r_inner) = extract_exact_int_factor(context, rhs, &target_big, &neg_target_big)?;

    // Replace sentinel Number(±target) with 1 for bare-constant terms
    let one = context.num(1);
    let l_inner = if matches!(context.get(l_inner), Expr::Number(n) if {
        let v = n.to_integer();
        v == target_big || v == neg_target_big
    }) {
        one
    } else {
        l_inner
    };
    let r_inner = if matches!(context.get(r_inner), Expr::Number(n) if {
        let v = n.to_integer();
        v == target_big || v == neg_target_big
    }) {
        one
    } else {
        r_inner
    };

    // Compute effective signs:
    //   outer_sign = l_sign (the global ±)
    //   The rhs enters with r_sign, possibly negated by is_sub
    let r_effective_positive = if is_sub { !r_sign } else { r_sign };

    if l_sign == r_effective_positive {
        // Same sign → Add
        let inner_sum = context.add(Expr::Add(l_inner, r_inner));
        Some((l_sign, inner_sum))
    } else {
        // Different signs → Sub
        let inner_diff = context.add(Expr::Sub(l_inner, r_inner));
        Some((l_sign, inner_diff))
    }
}

/// Extract inner variable from `2*x` pattern (for double angle identities).
/// Backward-compatible wrapper around [`extract_int_multiple`].
#[inline]
pub(crate) fn extract_double_angle_arg(context: &Context, expr: ExprId) -> Option<ExprId> {
    extract_int_multiple(context, expr, 2).and_then(|(positive, inner)| positive.then_some(inner))
}

/// Extract inner variable from `3*x` pattern (for triple angle identities).
/// Backward-compatible wrapper around [`extract_int_multiple`].
#[inline]
pub(crate) fn extract_triple_angle_arg(context: &Context, expr: ExprId) -> Option<ExprId> {
    extract_int_multiple(context, expr, 3).and_then(|(positive, inner)| positive.then_some(inner))
}

/// Extract inner variable from `5*x` pattern (for quintuple angle identities).
/// Backward-compatible wrapper around [`extract_int_multiple`].
#[inline]
pub(crate) fn extract_quintuple_angle_arg(context: &Context, expr: ExprId) -> Option<ExprId> {
    extract_int_multiple(context, expr, 5).and_then(|(positive, inner)| positive.then_some(inner))
}

/// Flatten an Add/Sub chain into a list of terms, converting subtractions to Neg.
/// This is used by collect and grouping modules for like-term collection.
///
/// Unlike `nary::add_leaves`, this handles:
/// - `Add(a, b)` → [a, b]
/// - `Sub(a, b)` → [a, Neg(b)]
/// - `Neg(Neg(x))` → [x]
pub(crate) fn flatten_add_sub_chain(ctx: &mut Context, expr: ExprId) -> Vec<ExprId> {
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

/// Flatten a Mul chain into a list of factors, handling Neg as -1 multiplication.
/// Returns Vec<ExprId> where Neg(e) is converted to [num(-1), ...factors of e...]
pub(crate) fn flatten_mul_chain(ctx: &mut Context, expr: ExprId) -> Vec<ExprId> {
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
