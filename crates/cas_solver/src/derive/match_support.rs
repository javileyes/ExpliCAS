use std::cmp::Ordering;

use cas_ast::{BuiltinFn, Expr, ExprId};
use cas_math::expr_nary::AddView;
use cas_math::expr_rewrite::smart_mul;
use num_traits::{One, Signed};

pub(crate) fn presentational_target_match(
    ctx: &mut cas_ast::Context,
    actual: ExprId,
    target: ExprId,
) -> bool {
    if actual == target {
        return true;
    }

    if trig_additive_multiplicative_shape_mismatch(ctx, actual, target) {
        return false;
    }

    if normalized_structural_match(ctx, actual, target) {
        return true;
    }

    if projected_root_power_match(ctx, actual, target) {
        return true;
    }

    if commutative_add_multiset_match(ctx, actual, target) {
        return true;
    }

    if commutative_mul_multiset_match(ctx, actual, target) {
        return true;
    }

    if signed_global_match(ctx, actual, target) {
        return true;
    }

    false
}

fn trig_additive_multiplicative_shape_mismatch(
    ctx: &cas_ast::Context,
    actual: ExprId,
    target: ExprId,
) -> bool {
    let actual_is_additive = matches!(ctx.get(actual), Expr::Add(_, _) | Expr::Sub(_, _));
    let target_is_additive = matches!(ctx.get(target), Expr::Add(_, _) | Expr::Sub(_, _));
    let actual_is_multiplicative = matches!(ctx.get(actual), Expr::Mul(_, _) | Expr::Div(_, _));
    let target_is_multiplicative = matches!(ctx.get(target), Expr::Mul(_, _) | Expr::Div(_, _));

    ((actual_is_additive && target_is_multiplicative)
        || (actual_is_multiplicative && target_is_additive))
        && (contains_circular_trig_fn(ctx, actual) || contains_circular_trig_fn(ctx, target))
}

fn contains_circular_trig_fn(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Function(fn_id, args) => {
                if matches!(
                    ctx.builtin_of(*fn_id),
                    Some(
                        BuiltinFn::Sin
                            | BuiltinFn::Cos
                            | BuiltinFn::Tan
                            | BuiltinFn::Sec
                            | BuiltinFn::Csc
                            | BuiltinFn::Cot
                    )
                ) {
                    return true;
                }
                stack.extend(args.iter().copied());
            }
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

pub(crate) fn strong_target_match(
    ctx: &mut cas_ast::Context,
    actual: ExprId,
    target: ExprId,
) -> bool {
    if presentational_target_match(ctx, actual, target) {
        return true;
    }

    let checker = cas_math::semantic_equality::SemanticEqualityChecker::new(ctx);
    if checker.are_equal_for_cycle_check(actual, target) {
        return true;
    }

    let projected_actual = project_root_like_calls_to_powers(ctx, actual);
    let projected_target = project_root_like_calls_to_powers(ctx, target);
    cas_math::semantic_equality::SemanticEqualityChecker::new(ctx)
        .are_equal_for_cycle_check(projected_actual, projected_target)
}

fn normalized_structural_match(ctx: &mut cas_ast::Context, actual: ExprId, target: ExprId) -> bool {
    let normalized_actual = cas_math::canonical_forms::normalize_core(ctx, actual);
    let normalized_target = cas_math::canonical_forms::normalize_core(ctx, target);
    cas_ast::ordering::compare_expr(ctx, normalized_actual, normalized_target) == Ordering::Equal
}

fn projected_root_power_match(ctx: &mut cas_ast::Context, actual: ExprId, target: ExprId) -> bool {
    let projected_actual = project_root_like_calls_to_powers(ctx, actual);
    let projected_target = project_root_like_calls_to_powers(ctx, target);

    let normalized_actual = cas_math::canonical_forms::normalize_core(ctx, projected_actual);
    let normalized_target = cas_math::canonical_forms::normalize_core(ctx, projected_target);
    cas_ast::ordering::compare_expr(ctx, normalized_actual, normalized_target) == Ordering::Equal
}

fn commutative_mul_multiset_match(
    ctx: &mut cas_ast::Context,
    actual: ExprId,
    target: ExprId,
) -> bool {
    if !ctx.is_mul_commutative(actual) || !ctx.is_mul_commutative(target) {
        return false;
    }

    let actual_factors = cas_math::trig_roots_flatten::flatten_mul_chain(ctx, actual);
    let target_factors = cas_math::trig_roots_flatten::flatten_mul_chain(ctx, target);
    if actual_factors.len() != target_factors.len() {
        return false;
    }

    let mut used = vec![false; target_factors.len()];
    for actual_factor in actual_factors {
        let mut matched = false;
        for (index, target_factor) in target_factors.iter().enumerate() {
            if used[index] {
                continue;
            }
            if factors_match(ctx, actual_factor, *target_factor) {
                used[index] = true;
                matched = true;
                break;
            }
        }
        if !matched {
            return false;
        }
    }

    true
}

fn commutative_add_multiset_match(
    ctx: &mut cas_ast::Context,
    actual: ExprId,
    target: ExprId,
) -> bool {
    let actual_terms = AddView::from_expr(ctx, actual).terms;
    let target_terms = AddView::from_expr(ctx, target).terms;
    if actual_terms.len() != target_terms.len() {
        return false;
    }
    if actual_terms.len() <= 1 {
        return false;
    }

    let mut used = vec![false; target_terms.len()];
    for (actual_term, actual_sign) in actual_terms {
        let mut matched = false;
        for (index, (target_term, target_sign)) in target_terms.iter().enumerate() {
            if used[index] || actual_sign != *target_sign {
                continue;
            }
            if factors_match(ctx, actual_term, *target_term) {
                used[index] = true;
                matched = true;
                break;
            }
        }
        if !matched {
            return false;
        }
    }

    true
}

fn signed_global_match(ctx: &mut cas_ast::Context, actual: ExprId, target: ExprId) -> bool {
    let (actual_negative, actual_core) = extract_global_sign(ctx, actual);
    let (target_negative, target_core) = extract_global_sign(ctx, target);
    actual_negative == target_negative
        && (normalized_structural_match(ctx, actual_core, target_core)
            || projected_root_power_match(ctx, actual_core, target_core)
            || commutative_mul_multiset_match(ctx, actual_core, target_core))
}

fn factors_match(ctx: &mut cas_ast::Context, actual: ExprId, target: ExprId) -> bool {
    if actual == target {
        return true;
    }

    if matches!(ctx.get(actual), Expr::Mul(_, _))
        && matches!(ctx.get(target), Expr::Mul(_, _))
        && commutative_mul_multiset_match(ctx, actual, target)
    {
        return true;
    }

    if normalized_structural_match(ctx, actual, target) {
        return true;
    }

    if projected_root_power_match(ctx, actual, target) {
        return true;
    }

    let checker = cas_math::semantic_equality::SemanticEqualityChecker::new(ctx);
    if checker.are_equal_for_cycle_check(actual, target) {
        return true;
    }

    let projected_actual = project_root_like_calls_to_powers(ctx, actual);
    let projected_target = project_root_like_calls_to_powers(ctx, target);
    cas_math::semantic_equality::SemanticEqualityChecker::new(ctx)
        .are_equal_for_cycle_check(projected_actual, projected_target)
}

fn extract_global_sign(ctx: &mut cas_ast::Context, expr: ExprId) -> (bool, ExprId) {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let (negative, core) = extract_global_sign(ctx, inner);
            (!negative, core)
        }
        Expr::Number(value) if value.is_negative() => (true, ctx.add(Expr::Number(-value))),
        _ if ctx.is_mul_commutative(expr) => {
            let mut negative = false;
            let mut positive_factors = Vec::new();
            for factor in cas_math::trig_roots_flatten::flatten_mul_chain(ctx, expr) {
                match ctx.get(factor).clone() {
                    Expr::Neg(inner) => {
                        negative = !negative;
                        positive_factors.push(inner);
                    }
                    Expr::Number(value) if value.is_negative() => {
                        negative = !negative;
                        let positive = ctx.add(Expr::Number(-value));
                        if !matches!(ctx.get(positive), Expr::Number(n) if n.is_one()) {
                            positive_factors.push(positive);
                        }
                    }
                    _ => positive_factors.push(factor),
                }
            }
            (negative, rebuild_mul_from_factors(ctx, &positive_factors))
        }
        _ => (false, expr),
    }
}

fn rebuild_mul_from_factors(ctx: &mut cas_ast::Context, factors: &[ExprId]) -> ExprId {
    let mut iter = factors.iter().copied();
    let Some(first) = iter.next() else {
        return ctx.num(1);
    };
    iter.fold(first, |acc, factor| smart_mul(ctx, acc, factor))
}

fn project_root_like_calls_to_powers(ctx: &mut cas_ast::Context, expr: ExprId) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => expr,
        Expr::Add(left, right) => rebuild_binary(ctx, expr, Expr::Add, left, right),
        Expr::Sub(left, right) => rebuild_binary(ctx, expr, Expr::Sub, left, right),
        Expr::Mul(left, right) => rebuild_binary(ctx, expr, Expr::Mul, left, right),
        Expr::Div(left, right) => rebuild_binary(ctx, expr, Expr::Div, left, right),
        Expr::Pow(base, exp) => rebuild_binary(ctx, expr, Expr::Pow, base, exp),
        Expr::Neg(inner) => {
            let rewritten = project_root_like_calls_to_powers(ctx, inner);
            if rewritten == inner {
                expr
            } else {
                ctx.add(Expr::Neg(rewritten))
            }
        }
        Expr::Function(name, args) => {
            let rewritten_args = args
                .iter()
                .map(|arg| project_root_like_calls_to_powers(ctx, *arg))
                .collect::<Vec<_>>();

            if ctx.is_builtin(name, BuiltinFn::Exp) && rewritten_args.len() == 1 {
                let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
                return ctx.add(Expr::Pow(e, rewritten_args[0]));
            }

            if (ctx.is_builtin(name, BuiltinFn::Sqrt) || ctx.is_builtin(name, BuiltinFn::Root))
                && rewritten_args.len() == 1
            {
                let half = ctx.rational(1, 2);
                return ctx.add(Expr::Pow(rewritten_args[0], half));
            }

            if (ctx.is_builtin(name, BuiltinFn::Sqrt) || ctx.is_builtin(name, BuiltinFn::Root))
                && rewritten_args.len() == 2
            {
                let exponent = build_root_exponent(ctx, rewritten_args[1]);
                return ctx.add(Expr::Pow(rewritten_args[0], exponent));
            }

            if rewritten_args == args {
                expr
            } else {
                ctx.add(Expr::Function(name, rewritten_args))
            }
        }
        Expr::Matrix { rows, cols, data } => {
            let rewritten_data = data
                .iter()
                .map(|item| project_root_like_calls_to_powers(ctx, *item))
                .collect::<Vec<_>>();
            if rewritten_data == data {
                expr
            } else {
                ctx.add(Expr::Matrix {
                    rows,
                    cols,
                    data: rewritten_data,
                })
            }
        }
        Expr::Hold(inner) => {
            let rewritten = project_root_like_calls_to_powers(ctx, inner);
            if rewritten == inner {
                expr
            } else {
                ctx.add(Expr::Hold(rewritten))
            }
        }
    }
}

fn rebuild_binary<F>(
    ctx: &mut cas_ast::Context,
    original: ExprId,
    make_expr: F,
    left: ExprId,
    right: ExprId,
) -> ExprId
where
    F: FnOnce(ExprId, ExprId) -> Expr,
{
    let rewritten_left = project_root_like_calls_to_powers(ctx, left);
    let rewritten_right = project_root_like_calls_to_powers(ctx, right);
    if rewritten_left == left && rewritten_right == right {
        original
    } else {
        ctx.add(make_expr(rewritten_left, rewritten_right))
    }
}

fn build_root_exponent(ctx: &mut cas_ast::Context, index: ExprId) -> ExprId {
    match ctx.get(index) {
        Expr::Number(value) if value.is_integer() && value.numer() != &0.into() => {
            if let Some(index_value) = num_traits::ToPrimitive::to_i64(value.numer()) {
                ctx.rational(1, index_value)
            } else {
                let one = ctx.num(1);
                ctx.add(Expr::Div(one, index))
            }
        }
        _ => {
            let one = ctx.num(1);
            ctx.add(Expr::Div(one, index))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::strong_target_match;

    #[test]
    fn matches_sqrt_target_against_fractional_power() {
        let mut ctx = cas_ast::Context::new();
        let actual = cas_parser::parse("x^(1/2)", &mut ctx).expect("actual");
        let target = cas_parser::parse("sqrt(x)", &mut ctx).expect("target");
        assert!(strong_target_match(&mut ctx, actual, target));
        assert!(strong_target_match(&mut ctx, target, actual));
    }

    #[test]
    fn matches_rationalized_target_with_sqrt_notation() {
        let mut ctx = cas_ast::Context::new();
        let actual = cas_parser::parse("(x^(1/2) + 1)/(x - 1)", &mut ctx).expect("actual");
        let target = cas_parser::parse("(sqrt(x) + 1)/(x - 1)", &mut ctx).expect("target");
        assert!(strong_target_match(&mut ctx, actual, target));
    }

    #[test]
    fn matches_additive_targets_with_reordered_terms() {
        let mut ctx = cas_ast::Context::new();
        let actual = cas_parser::parse("a^2 + b^2 + 2*a*b", &mut ctx).expect("actual");
        let target = cas_parser::parse("a^2 + 2*a*b + b^2", &mut ctx).expect("target");
        assert!(strong_target_match(&mut ctx, actual, target));
        assert!(strong_target_match(&mut ctx, target, actual));
    }

    #[test]
    fn matches_additive_targets_with_reordered_negative_terms() {
        let mut ctx = cas_ast::Context::new();
        let actual = cas_parser::parse("a - b - c", &mut ctx).expect("actual");
        let target = cas_parser::parse("a - c - b", &mut ctx).expect("target");
        assert!(strong_target_match(&mut ctx, actual, target));
        assert!(strong_target_match(&mut ctx, target, actual));
    }

    #[test]
    fn matches_global_negation_against_negative_coefficient_product() {
        let mut ctx = cas_ast::Context::new();
        let actual = cas_parser::parse("-(2*sin(2*x)*sin(3*x))", &mut ctx).expect("actual");
        let target = cas_parser::parse("-2*sin(3*x)*sin(2*x)", &mut ctx).expect("target");
        assert!(strong_target_match(&mut ctx, actual, target));
        assert!(strong_target_match(&mut ctx, target, actual));
    }

    #[test]
    fn matches_exp_call_targets_against_power_notation() {
        let mut ctx = cas_ast::Context::new();
        let actual = cas_parser::parse("exp(x)-exp(-x)", &mut ctx).expect("actual");
        let target = cas_parser::parse("e^x-e^(-x)", &mut ctx).expect("target");
        assert!(strong_target_match(&mut ctx, actual, target));
        assert!(strong_target_match(&mut ctx, target, actual));
    }

    #[test]
    fn matches_hyperbolic_additive_products_with_reordered_factors() {
        let mut ctx = cas_ast::Context::new();
        let actual =
            cas_parser::parse("sinh(x)*cosh(y) + sinh(y)*cosh(x)", &mut ctx).expect("actual");
        let target =
            cas_parser::parse("sinh(x)*cosh(y) + cosh(x)*sinh(y)", &mut ctx).expect("target");
        assert!(strong_target_match(&mut ctx, actual, target));
        assert!(strong_target_match(&mut ctx, target, actual));
    }

    #[test]
    fn matches_hyperbolic_additive_terms_with_reordered_internal_product_factors() {
        let mut ctx = cas_ast::Context::new();
        let actual = cas_parser::parse("4*cosh(x)^2*sinh(x)-2*sinh(x)", &mut ctx).expect("actual");
        let target = cas_parser::parse("4*sinh(x)*cosh(x)^2-2*sinh(x)", &mut ctx).expect("target");
        assert!(strong_target_match(&mut ctx, actual, target));
        assert!(strong_target_match(&mut ctx, target, actual));
    }
}
