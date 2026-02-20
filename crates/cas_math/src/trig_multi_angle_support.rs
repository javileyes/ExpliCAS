use crate::trig_sum_product_support::extract_trig_arg;
use cas_ast::{Context, Expr, ExprId};

/// Check if a trig argument is "trivial": variable, constant, number,
/// or a simple numeric multiple of one of those.
pub fn is_trivial_angle(ctx: &Context, arg: ExprId) -> bool {
    match ctx.get(arg) {
        Expr::Variable(_) | Expr::Constant(_) | Expr::Number(_) => true,
        Expr::Mul(l, r) => {
            let l_simple = matches!(
                ctx.get(*l),
                Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_)
            );
            let r_simple = matches!(
                ctx.get(*r),
                Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_)
            );
            l_simple && r_simple
        }
        Expr::Neg(inner) => is_trivial_angle(ctx, *inner),
        _ => false,
    }
}

/// Check whether `arg` is a multiple-angle form `n*x` (or `x*n`)
/// with integer `|n| > 1`.
pub fn is_multiple_angle(ctx: &Context, arg: ExprId) -> bool {
    let Expr::Mul(l, r) = ctx.get(arg) else {
        return false;
    };

    let is_large_integer_factor = |id: ExprId| -> bool {
        if let Expr::Number(n) = ctx.get(id) {
            if n.is_integer() {
                let val = n.numer().clone();
                return val > 1.into() || val < (-1).into();
            }
        }
        false
    };

    is_large_integer_factor(*l) || is_large_integer_factor(*r)
}

/// Check whether `arg` has a large trig coefficient.
///
/// Preserves engine behavior:
/// - `n*x` with integer `|n| > 2` is considered large.
/// - For `a+b` / `a-b`, if either side is a multiple-angle (`|n| > 1`),
///   it is considered large.
pub fn has_large_coefficient(ctx: &Context, arg: ExprId) -> bool {
    if let Expr::Mul(l, r) = ctx.get(arg) {
        let is_very_large_integer_factor = |id: ExprId| -> bool {
            if let Expr::Number(n) = ctx.get(id) {
                if n.is_integer() {
                    let val = n.numer().clone();
                    return val > 2.into() || val < (-2).into();
                }
            }
            false
        };

        if is_very_large_integer_factor(*l) || is_very_large_integer_factor(*r) {
            return true;
        }
    }

    if let Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) = ctx.get(arg) {
        return is_multiple_angle(ctx, *lhs) || is_multiple_angle(ctx, *rhs);
    }

    false
}

/// Check if `expr` is a binary trig add/sub operation:
/// - `Add(trig(A), trig(B))`
/// - `Sub(trig(A), trig(B))`
/// - `Add(trig(A), Neg(trig(B)))`
pub fn is_binary_trig_op(ctx: &Context, expr: ExprId, fn_name: &str) -> bool {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            if extract_trig_arg(ctx, *l, fn_name).is_some()
                && extract_trig_arg(ctx, *r, fn_name).is_some()
            {
                return true;
            }
            if let Expr::Neg(inner) = ctx.get(*r) {
                if extract_trig_arg(ctx, *l, fn_name).is_some()
                    && extract_trig_arg(ctx, *inner, fn_name).is_some()
                {
                    return true;
                }
            }
            if let Expr::Neg(inner) = ctx.get(*l) {
                if extract_trig_arg(ctx, *r, fn_name).is_some()
                    && extract_trig_arg(ctx, *inner, fn_name).is_some()
                {
                    return true;
                }
            }
            false
        }
        Expr::Sub(l, r) => {
            extract_trig_arg(ctx, *l, fn_name).is_some()
                && extract_trig_arg(ctx, *r, fn_name).is_some()
        }
        _ => false,
    }
}

/// Check if `expr` is `Add(trig(A), trig(B))`.
pub fn is_trig_sum(ctx: &Context, expr: ExprId, fn_name: &str) -> bool {
    if let Expr::Add(l, r) = ctx.get(expr) {
        return extract_trig_arg(ctx, *l, fn_name).is_some()
            && extract_trig_arg(ctx, *r, fn_name).is_some();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn trivial_angle_detection_handles_basic_shapes() {
        let mut ctx = Context::new();
        let var = parse("x", &mut ctx).expect("x");
        let mul = parse("2*x", &mut ctx).expect("2*x");
        let neg = parse("-pi", &mut ctx).expect("-pi");
        let complex = parse("x+y", &mut ctx).expect("x+y");

        assert!(is_trivial_angle(&ctx, var));
        assert!(is_trivial_angle(&ctx, mul));
        assert!(is_trivial_angle(&ctx, neg));
        assert!(!is_trivial_angle(&ctx, complex));
    }

    #[test]
    fn binary_trig_op_matches_add_sub_and_add_neg_forms() {
        let mut ctx = Context::new();
        let add = parse("sin(a)+sin(b)", &mut ctx).expect("add");
        let sub = parse("sin(a)-sin(b)", &mut ctx).expect("sub");
        let add_neg = parse("sin(a)+(-sin(b))", &mut ctx).expect("add_neg");
        let mixed = parse("sin(a)+cos(b)", &mut ctx).expect("mixed");

        assert!(is_binary_trig_op(&ctx, add, "sin"));
        assert!(is_binary_trig_op(&ctx, sub, "sin"));
        assert!(is_binary_trig_op(&ctx, add_neg, "sin"));
        assert!(!is_binary_trig_op(&ctx, mixed, "sin"));
    }

    #[test]
    fn trig_sum_matches_only_add_of_same_trig_family() {
        let mut ctx = Context::new();
        let sum = parse("cos(a)+cos(b)", &mut ctx).expect("sum");
        let diff = parse("cos(a)-cos(b)", &mut ctx).expect("diff");

        assert!(is_trig_sum(&ctx, sum, "cos"));
        assert!(!is_trig_sum(&ctx, diff, "cos"));
    }

    #[test]
    fn multiple_angle_detection_matches_integer_multiplier_policy() {
        let mut ctx = Context::new();
        let two_x = parse("2*x", &mut ctx).expect("2*x");
        let neg_three_x = parse("-3*x", &mut ctx).expect("-3*x");
        let half_x = parse("x/2", &mut ctx).expect("x/2");
        let one_x = parse("1*x", &mut ctx).expect("1*x");

        assert!(is_multiple_angle(&ctx, two_x));
        assert!(is_multiple_angle(&ctx, neg_three_x));
        assert!(!is_multiple_angle(&ctx, half_x));
        assert!(!is_multiple_angle(&ctx, one_x));
    }

    #[test]
    fn large_coefficient_detection_matches_existing_engine_behavior() {
        let mut ctx = Context::new();
        let three_x = parse("3*x", &mut ctx).expect("3*x");
        let two_x = parse("2*x", &mut ctx).expect("2*x");
        let sum_with_multiple = parse("x + 2*y", &mut ctx).expect("x+2*y");
        let simple_sum = parse("x + y", &mut ctx).expect("x+y");

        assert!(has_large_coefficient(&ctx, three_x));
        assert!(!has_large_coefficient(&ctx, two_x));
        assert!(has_large_coefficient(&ctx, sum_with_multiple));
        assert!(!has_large_coefficient(&ctx, simple_sum));
    }
}
