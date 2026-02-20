use crate::expr_rewrite::smart_mul;
use crate::pi_helpers::extract_rational_pi_multiple;
use crate::trig_sum_product_support::extract_trig_arg;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use std::cmp::Ordering;

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

/// Check whether `large` and `small` satisfy a double-angle relation.
///
/// Recognized shapes:
/// - `large = 2 * small`
/// - `small = large / 2`
/// - `small = (1/2) * large`
pub fn is_double_angle_relation(ctx: &Context, large: ExprId, small: ExprId) -> bool {
    // Case 1: large = 2 * small
    if let Expr::Mul(l, r) = ctx.get(large) {
        if let Expr::Number(n) = ctx.get(*l) {
            if n == &num_rational::BigRational::from_integer(2.into())
                && compare_expr(ctx, *r, small) == Ordering::Equal
            {
                return true;
            }
        }
        if let Expr::Number(n) = ctx.get(*r) {
            if n == &num_rational::BigRational::from_integer(2.into())
                && compare_expr(ctx, *l, small) == Ordering::Equal
            {
                return true;
            }
        }
    }

    // Case 2: small = large / 2
    if let Expr::Div(n, d) = ctx.get(small) {
        if let Expr::Number(val) = ctx.get(*d) {
            if val == &num_rational::BigRational::from_integer(2.into())
                && compare_expr(ctx, *n, large) == Ordering::Equal
            {
                return true;
            }
        }
    }

    // Case 3: small = (1/2) * large
    if let Expr::Mul(l, r) = ctx.get(small) {
        if let Expr::Number(n) = ctx.get(*l) {
            if n == &num_rational::BigRational::new(1.into(), 2.into())
                && compare_expr(ctx, *r, large) == Ordering::Equal
            {
                return true;
            }
        }
        if let Expr::Number(n) = ctx.get(*r) {
            if n == &num_rational::BigRational::new(1.into(), 2.into())
                && compare_expr(ctx, *l, large) == Ordering::Equal
            {
                return true;
            }
        }
    }

    false
}

/// Verify that trig args form a dyadic sequence: `theta, 2*theta, 4*theta, ...`.
///
/// This matcher is robust to normalization by comparing extracted rational
/// coefficients in `k*pi` space rather than raw AST shapes.
pub fn verify_dyadic_pi_sequence(ctx: &Context, theta: ExprId, trig_args: &[ExprId]) -> bool {
    let n = trig_args.len() as u32;
    if n == 0 {
        return false;
    }

    let base_coeff = match extract_rational_pi_multiple(ctx, theta) {
        Some(k) => k,
        None => return false,
    };

    let mut coeffs: Vec<BigRational> = Vec::with_capacity(n as usize);
    for &arg in trig_args {
        match extract_rational_pi_multiple(ctx, arg) {
            Some(k) => coeffs.push(k),
            None => return false,
        }
    }

    let mut expected: Vec<BigRational> = Vec::with_capacity(n as usize);
    for k in 0..n {
        let multiplier = BigRational::from_integer((1u64 << k).into());
        expected.push(&base_coeff * &multiplier);
    }

    let mut used = vec![false; expected.len()];
    for coeff in &coeffs {
        let mut found = false;
        for (i, exp) in expected.iter().enumerate() {
            if !used[i] && coeff == exp {
                used[i] = true;
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }

    used.iter().all(|&u| u)
}

/// Collect all arguments from unary `sin`, `cos` and `tan` calls in an expression tree.
pub fn collect_trig_args_recursive(ctx: &Context, expr: ExprId, args: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Function(fn_id, fargs) => {
            if matches!(
                ctx.builtin_of(*fn_id),
                Some(BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan)
            ) && fargs.len() == 1
            {
                args.push(fargs[0]);
            }
            for arg in fargs {
                collect_trig_args_recursive(ctx, *arg, args);
            }
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            collect_trig_args_recursive(ctx, *l, args);
            collect_trig_args_recursive(ctx, *r, args);
        }
        Expr::Neg(e) => collect_trig_args_recursive(ctx, *e, args),
        _ => {}
    }
}

/// Expand `sin/cos/tan(large_angle)` nodes to half-angle forms using `small_angle`.
pub fn expand_trig_angle(
    ctx: &mut Context,
    expr: ExprId,
    large_angle: ExprId,
    small_angle: ExprId,
) -> ExprId {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if args.len() == 1 && compare_expr(ctx, args[0], large_angle) == Ordering::Equal {
            let fn_id = *fn_id;
            match ctx.builtin_of(fn_id) {
                Some(BuiltinFn::Sin) => {
                    let two = ctx.num(2);
                    let sin_half = ctx.call_builtin(BuiltinFn::Sin, vec![small_angle]);
                    let cos_half = ctx.call_builtin(BuiltinFn::Cos, vec![small_angle]);
                    let term = smart_mul(ctx, sin_half, cos_half);
                    return smart_mul(ctx, two, term);
                }
                Some(BuiltinFn::Cos) => {
                    let two = ctx.num(2);
                    let one = ctx.num(1);
                    let cos_half = ctx.call_builtin(BuiltinFn::Cos, vec![small_angle]);
                    let cos_sq = ctx.add(Expr::Pow(cos_half, two));
                    let term = smart_mul(ctx, two, cos_sq);
                    return ctx.add(Expr::Sub(term, one));
                }
                Some(BuiltinFn::Tan) => {
                    let two = ctx.num(2);
                    let one = ctx.num(1);
                    let tan_half = ctx.call_builtin(BuiltinFn::Tan, vec![small_angle]);
                    let num = smart_mul(ctx, two, tan_half);
                    let tan_sq = ctx.add(Expr::Pow(tan_half, two));
                    let den = ctx.add(Expr::Sub(one, tan_sq));
                    return ctx.add(Expr::Div(num, den));
                }
                _ => {}
            }
        }
    }

    enum Shape {
        Add(ExprId, ExprId),
        Sub(ExprId, ExprId),
        Mul(ExprId, ExprId),
        Div(ExprId, ExprId),
        Pow(ExprId, ExprId),
        Neg(ExprId),
        Func(usize, Vec<ExprId>),
        Other,
    }

    let shape = match ctx.get(expr) {
        Expr::Add(l, r) => Shape::Add(*l, *r),
        Expr::Sub(l, r) => Shape::Sub(*l, *r),
        Expr::Mul(l, r) => Shape::Mul(*l, *r),
        Expr::Div(l, r) => Shape::Div(*l, *r),
        Expr::Pow(b, e) => Shape::Pow(*b, *e),
        Expr::Neg(e) => Shape::Neg(*e),
        Expr::Function(fn_id, args) => Shape::Func(*fn_id, args.clone()),
        _ => Shape::Other,
    };

    match shape {
        Shape::Add(l, r) => {
            let nl = expand_trig_angle(ctx, l, large_angle, small_angle);
            let nr = expand_trig_angle(ctx, r, large_angle, small_angle);
            if nl != l || nr != r {
                ctx.add(Expr::Add(nl, nr))
            } else {
                expr
            }
        }
        Shape::Sub(l, r) => {
            let nl = expand_trig_angle(ctx, l, large_angle, small_angle);
            let nr = expand_trig_angle(ctx, r, large_angle, small_angle);
            if nl != l || nr != r {
                ctx.add(Expr::Sub(nl, nr))
            } else {
                expr
            }
        }
        Shape::Mul(l, r) => {
            let nl = expand_trig_angle(ctx, l, large_angle, small_angle);
            let nr = expand_trig_angle(ctx, r, large_angle, small_angle);
            if nl != l || nr != r {
                smart_mul(ctx, nl, nr)
            } else {
                expr
            }
        }
        Shape::Div(l, r) => {
            let nl = expand_trig_angle(ctx, l, large_angle, small_angle);
            let nr = expand_trig_angle(ctx, r, large_angle, small_angle);
            if nl != l || nr != r {
                ctx.add(Expr::Div(nl, nr))
            } else {
                expr
            }
        }
        Shape::Pow(b, e) => {
            let nb = expand_trig_angle(ctx, b, large_angle, small_angle);
            let ne = expand_trig_angle(ctx, e, large_angle, small_angle);
            if nb != b || ne != e {
                ctx.add(Expr::Pow(nb, ne))
            } else {
                expr
            }
        }
        Shape::Neg(e) => {
            let ne = expand_trig_angle(ctx, e, large_angle, small_angle);
            if ne != e {
                ctx.add(Expr::Neg(ne))
            } else {
                expr
            }
        }
        Shape::Func(fn_id, args) => {
            let mut new_args = Vec::with_capacity(args.len());
            let mut changed = false;
            for arg in args {
                let na = expand_trig_angle(ctx, arg, large_angle, small_angle);
                if na != arg {
                    changed = true;
                }
                new_args.push(na);
            }
            if changed {
                ctx.add(Expr::Function(fn_id, new_args))
            } else {
                expr
            }
        }
        Shape::Other => expr,
    }
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

    #[test]
    fn double_angle_relation_matches_all_supported_forms() {
        let mut ctx = Context::new();
        let two_x = parse("2*x", &mut ctx).expect("2*x");
        let x = parse("x", &mut ctx).expect("x");
        let x_over_2 = parse("x/2", &mut ctx).expect("x/2");
        let three_x = parse("3*x", &mut ctx).expect("3*x");

        assert!(is_double_angle_relation(&ctx, two_x, x));
        assert!(is_double_angle_relation(&ctx, x, x_over_2));
        assert!(!is_double_angle_relation(&ctx, three_x, x));
    }

    #[test]
    fn dyadic_pi_sequence_detection_matches_expected_patterns() {
        let mut ctx = Context::new();
        let theta = parse("pi/9", &mut ctx).expect("theta");
        let args = vec![
            parse("2*pi/9", &mut ctx).expect("2pi/9"),
            parse("4*pi/9", &mut ctx).expect("4pi/9"),
            parse("pi/9", &mut ctx).expect("pi/9"),
        ];
        let wrong = vec![
            parse("pi/9", &mut ctx).expect("pi/9"),
            parse("3*pi/9", &mut ctx).expect("3pi/9"),
        ];

        assert!(verify_dyadic_pi_sequence(&ctx, theta, &args));
        assert!(!verify_dyadic_pi_sequence(&ctx, theta, &wrong));
    }

    #[test]
    fn collect_trig_args_recursive_finds_nested_unary_trig_arguments() {
        let mut ctx = Context::new();
        let expr = parse("sin(2*x) + cos(x+y) + tan(z) + ln(sin(w))", &mut ctx).expect("expr");
        let mut args = Vec::new();
        collect_trig_args_recursive(&ctx, expr, &mut args);

        let mut expected = [
            parse("2*x", &mut ctx).expect("2*x"),
            parse("x+y", &mut ctx).expect("x+y"),
            parse("z", &mut ctx).expect("z"),
            parse("w", &mut ctx).expect("w"),
        ];

        args.sort_by(|a, b| compare_expr(&ctx, *a, *b));
        expected.sort_by(|a, b| compare_expr(&ctx, *a, *b));

        assert_eq!(args.len(), expected.len());
        for (a, b) in args.iter().zip(expected.iter()) {
            assert_eq!(compare_expr(&ctx, *a, *b), Ordering::Equal);
        }
    }

    #[test]
    fn expand_trig_angle_rewrites_large_angle_trig_nodes_recursively() {
        let mut ctx = Context::new();
        let large = parse("2*x", &mut ctx).expect("large");
        let small = parse("x", &mut ctx).expect("small");

        let sin_expr = parse("sin(2*x)", &mut ctx).expect("sin_expr");
        let sin_got = expand_trig_angle(&mut ctx, sin_expr, large, small);
        let two = ctx.num(2);
        let sin_half = ctx.call_builtin(BuiltinFn::Sin, vec![small]);
        let cos_half = ctx.call_builtin(BuiltinFn::Cos, vec![small]);
        let sin_term = smart_mul(&mut ctx, sin_half, cos_half);
        let sin_expected = smart_mul(&mut ctx, two, sin_term);
        assert_eq!(compare_expr(&ctx, sin_got, sin_expected), Ordering::Equal);

        let cos_expr = parse("cos(2*x)", &mut ctx).expect("cos_expr");
        let cos_got = expand_trig_angle(&mut ctx, cos_expr, large, small);
        let one = ctx.num(1);
        let cos_half = ctx.call_builtin(BuiltinFn::Cos, vec![small]);
        let cos_sq = ctx.add(Expr::Pow(cos_half, two));
        let cos_term = smart_mul(&mut ctx, two, cos_sq);
        let cos_expected = ctx.add(Expr::Sub(cos_term, one));
        assert_eq!(compare_expr(&ctx, cos_got, cos_expected), Ordering::Equal);

        let tan_expr = parse("tan(2*x)", &mut ctx).expect("tan_expr");
        let tan_got = expand_trig_angle(&mut ctx, tan_expr, large, small);
        let tan_half = ctx.call_builtin(BuiltinFn::Tan, vec![small]);
        let tan_num = smart_mul(&mut ctx, two, tan_half);
        let tan_sq = ctx.add(Expr::Pow(tan_half, two));
        let tan_den = ctx.add(Expr::Sub(one, tan_sq));
        let tan_expected = ctx.add(Expr::Div(tan_num, tan_den));
        assert_eq!(compare_expr(&ctx, tan_got, tan_expected), Ordering::Equal);
    }
}
