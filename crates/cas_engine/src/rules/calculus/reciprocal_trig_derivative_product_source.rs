use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::Zero;

pub(crate) struct ReciprocalTrigDerivativeProductPair {
    pub(crate) first_index: usize,
    pub(crate) second_index: usize,
    pub(crate) arg: ExprId,
    pub(crate) denominator: BuiltinFn,
}

pub(crate) fn constant_scaled_reciprocal_trig_derivative_product_source(
    ctx: &Context,
    target: ExprId,
    var_name: &str,
) -> bool {
    let factors = cas_math::expr_nary::mul_leaves(ctx, target);
    let Some(pair) = reciprocal_trig_derivative_product_pair(ctx, &factors) else {
        return false;
    };

    for (index, factor) in factors.iter().copied().enumerate() {
        if index == pair.first_index || index == pair.second_index {
            continue;
        }
        if cas_math::expr_predicates::contains_named_var(ctx, factor, var_name) {
            return false;
        }
    }

    cas_math::expr_predicates::contains_named_var(ctx, pair.arg, var_name)
        && expr_is_affine_in_named_var(ctx, pair.arg, var_name)
}

pub(crate) fn reciprocal_trig_derivative_product_pair(
    ctx: &Context,
    factors: &[ExprId],
) -> Option<ReciprocalTrigDerivativeProductPair> {
    let mut pair = None;
    for (index, factor) in factors.iter().copied().enumerate() {
        if pair.is_none() {
            pair = reciprocal_trig_derivative_product_factor_pair(ctx, factors, index, factor);
        }
    }
    pair
}

pub(crate) fn reciprocal_trig_derivative_integrand_quotient(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let quotient = reciprocal_trig_derivative_integrand_quotient(ctx, inner)?;
        return Some(ctx.add(Expr::Neg(quotient)));
    }

    reciprocal_trig_derivative_integrand_quotient_for_builtin(ctx, expr, BuiltinFn::Sec).or_else(
        || reciprocal_trig_derivative_integrand_quotient_for_builtin(ctx, expr, BuiltinFn::Csc),
    )
}

fn reciprocal_trig_derivative_integrand_quotient_for_builtin(
    ctx: &mut Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<ExprId> {
    let (ratio_builtin, denominator_builtin) = match builtin {
        BuiltinFn::Sec => (BuiltinFn::Tan, BuiltinFn::Cos),
        BuiltinFn::Csc => (BuiltinFn::Cot, BuiltinFn::Sin),
        _ => return None,
    };

    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    let mut reciprocal_index = None;
    let mut ratio_index = None;
    let mut matched_arg = None;

    for (idx, factor) in factors.iter().enumerate() {
        let Some(arg) = unary_builtin_arg(ctx, *factor, builtin) else {
            continue;
        };
        if reciprocal_index.is_some() {
            return None;
        }
        reciprocal_index = Some(idx);
        matched_arg = Some(arg);
    }

    let arg = matched_arg?;
    for (idx, factor) in factors.iter().enumerate() {
        let Some(ratio_arg) = unary_builtin_arg(ctx, *factor, ratio_builtin) else {
            continue;
        };
        if !cas_math::expr_domain::exprs_equivalent(ctx, ratio_arg, arg) {
            continue;
        }
        if ratio_index.is_some() {
            return None;
        }
        ratio_index = Some(idx);
    }

    let reciprocal_index = reciprocal_index?;
    ratio_index?;

    let numerator_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != reciprocal_index).then_some(*factor))
        .collect();
    let numerator = cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors);
    let denominator = ctx.call_builtin(denominator_builtin, vec![arg]);
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(crate) fn verified_reciprocal_trig_derivative_product_source_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId)> {
    let mut factors = Vec::new();
    collect_reciprocal_trig_product_factors(ctx, target, &mut factors);

    let ReciprocalTrigDerivativeProductPair {
        first_index,
        second_index,
        arg,
        denominator,
    } = reciprocal_trig_derivative_product_pair(ctx, &factors)?;

    if !remaining_factors_are_constant_scaled_arg_derivative(
        ctx,
        &factors,
        first_index,
        second_index,
        arg,
        var_name,
    ) {
        return None;
    }

    let required_nonzero = ctx.call_builtin(denominator, vec![arg]);
    Some((target, required_nonzero))
}

pub(crate) fn verified_affine_reciprocal_trig_derivative_product_source_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, ExprId)> {
    if !constant_scaled_reciprocal_trig_derivative_product_source(ctx, target, var_name) {
        return None;
    }

    let normalized = reciprocal_trig_derivative_integrand_quotient(ctx, target)?;
    cas_math::symbolic_integration_support::integrate_symbolic_expr(ctx, normalized, var_name)?;

    let mut factors = Vec::new();
    collect_reciprocal_trig_product_factors(ctx, target, &mut factors);
    let ReciprocalTrigDerivativeProductPair {
        arg, denominator, ..
    } = reciprocal_trig_derivative_product_pair(ctx, &factors)?;
    let required_nonzero = ctx.call_builtin(denominator, vec![arg]);
    Some((target, required_nonzero))
}

fn reciprocal_trig_derivative_product_factor_pair(
    ctx: &Context,
    factors: &[ExprId],
    first_index: usize,
    first_factor: ExprId,
) -> Option<ReciprocalTrigDerivativeProductPair> {
    for (second_index, second_factor) in factors.iter().copied().enumerate().skip(first_index + 1) {
        if let Some(arg) = same_arg_unary_pair(
            ctx,
            first_factor,
            BuiltinFn::Csc,
            second_factor,
            BuiltinFn::Cot,
        )
        .or_else(|| {
            same_arg_unary_pair(
                ctx,
                second_factor,
                BuiltinFn::Csc,
                first_factor,
                BuiltinFn::Cot,
            )
        }) {
            return Some(ReciprocalTrigDerivativeProductPair {
                first_index,
                second_index,
                arg,
                denominator: BuiltinFn::Sin,
            });
        }

        if let Some(arg) = same_arg_unary_pair(
            ctx,
            first_factor,
            BuiltinFn::Sec,
            second_factor,
            BuiltinFn::Tan,
        )
        .or_else(|| {
            same_arg_unary_pair(
                ctx,
                second_factor,
                BuiltinFn::Sec,
                first_factor,
                BuiltinFn::Tan,
            )
        }) {
            return Some(ReciprocalTrigDerivativeProductPair {
                first_index,
                second_index,
                arg,
                denominator: BuiltinFn::Cos,
            });
        }
    }

    None
}

fn same_arg_unary_pair(
    ctx: &Context,
    left: ExprId,
    left_builtin: BuiltinFn,
    right: ExprId,
    right_builtin: BuiltinFn,
) -> Option<ExprId> {
    let left_arg = signed_unary_builtin_arg(ctx, left, left_builtin)?;
    let right_arg = signed_unary_builtin_arg(ctx, right, right_builtin)?;
    cas_math::expr_witness::exprs_equal(ctx, left_arg, right_arg).then_some(left_arg)
}

fn unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() == 1 && ctx.is_builtin(*fn_id, builtin) {
        Some(args[0])
    } else {
        None
    }
}

fn signed_unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    let expr = match ctx.get(expr) {
        Expr::Neg(inner) => *inner,
        _ => expr,
    };
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() == 1 && ctx.is_builtin(*fn_id, builtin) {
        Some(args[0])
    } else {
        None
    }
}

fn collect_reciprocal_trig_product_factors(
    ctx: &mut Context,
    expr: ExprId,
    factors: &mut Vec<ExprId>,
) {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Mul(left, right) => {
            collect_reciprocal_trig_product_factors(ctx, left, factors);
            collect_reciprocal_trig_product_factors(ctx, right, factors);
        }
        _ => factors.push(expr),
    }
}

fn remaining_factors_are_constant_scaled_arg_derivative(
    ctx: &mut Context,
    factors: &[ExprId],
    first_index: usize,
    second_index: usize,
    arg: ExprId,
    var_name: &str,
) -> bool {
    let remaining_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(index, factor)| {
            (index != first_index && index != second_index).then_some(*factor)
        })
        .collect();
    if remaining_factors.is_empty() {
        return false;
    }

    let remaining = cas_math::expr_nary::build_balanced_mul(ctx, &remaining_factors);
    if let Some(arg_derivative) =
        cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(ctx, arg, var_name)
    {
        if remaining_factors_contain_explicit_derivative_and_constant_scale(
            ctx,
            &remaining_factors,
            arg_derivative,
            var_name,
        ) {
            return true;
        }
        if expr_matches_signed_derivative(ctx, remaining, arg_derivative)
            || exprs_equivalent_up_to_sign(ctx, remaining, arg_derivative)
            || remaining_is_constant_scaled_derivative(ctx, remaining, arg_derivative, var_name)
        {
            return true;
        }
    }

    let Some(polynomial_arg_derivative) =
        polynomial_derivative_of_dependent_arg(ctx, arg, var_name)
    else {
        return false;
    };
    remaining_is_constant_scaled_derivative(ctx, remaining, polynomial_arg_derivative, var_name)
}

fn remaining_factors_contain_explicit_derivative_and_constant_scale(
    ctx: &mut Context,
    remaining_factors: &[ExprId],
    derivative: ExprId,
    var_name: &str,
) -> bool {
    let mut saw_derivative_factor = false;
    for factor in remaining_factors.iter().copied() {
        if !saw_derivative_factor
            && (expr_matches_signed_derivative(ctx, factor, derivative)
                || exprs_equivalent_up_to_sign(ctx, factor, derivative))
        {
            saw_derivative_factor = true;
            continue;
        }
        if cas_math::expr_predicates::contains_named_var(ctx, factor, var_name) {
            return false;
        }
    }
    saw_derivative_factor
}

fn exprs_equivalent_up_to_sign(ctx: &mut Context, expr: ExprId, derivative: ExprId) -> bool {
    if cas_math::expr_domain::exprs_equivalent(ctx, expr, derivative) {
        return true;
    }
    let neg_expr = ctx.add(Expr::Neg(expr));
    cas_math::expr_domain::exprs_equivalent(ctx, neg_expr, derivative)
}

fn expr_matches_signed_derivative(ctx: &mut Context, expr: ExprId, derivative: ExprId) -> bool {
    if cas_math::expr_witness::exprs_equal(ctx, expr, derivative) {
        return true;
    }
    let neg_factor = ctx.add(Expr::Neg(expr));
    cas_math::expr_witness::exprs_equal(ctx, neg_factor, derivative)
}

fn remaining_is_constant_scaled_derivative(
    ctx: &mut Context,
    remaining: ExprId,
    derivative: ExprId,
    var_name: &str,
) -> bool {
    let Ok(remaining_poly) = Polynomial::from_expr(ctx, remaining, var_name) else {
        return false;
    };
    let Ok(derivative_poly) = Polynomial::from_expr(ctx, derivative, var_name) else {
        return false;
    };

    constant_polynomial_ratio(&remaining_poly, &derivative_poly)
        .is_some_and(|scale| !scale.is_zero())
}

fn polynomial_derivative_of_dependent_arg(
    ctx: &mut Context,
    arg: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let dependent = additive_var_dependent_part(ctx, arg, var_name)?;
    let poly = Polynomial::from_expr(ctx, dependent, var_name).ok()?;
    Some(poly.derivative().to_expr(ctx))
}

fn additive_var_dependent_part(ctx: &mut Context, expr: ExprId, var_name: &str) -> Option<ExprId> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => match (
            additive_var_dependent_part(ctx, left, var_name),
            additive_var_dependent_part(ctx, right, var_name),
        ) {
            (Some(left), Some(right)) => Some(ctx.add(Expr::Add(left, right))),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            (None, None) => None,
        },
        Expr::Sub(left, right) => match (
            additive_var_dependent_part(ctx, left, var_name),
            additive_var_dependent_part(ctx, right, var_name),
        ) {
            (Some(left), Some(right)) => Some(ctx.add(Expr::Sub(left, right))),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(ctx.add(Expr::Neg(right))),
            (None, None) => None,
        },
        Expr::Neg(inner) => {
            let inner = additive_var_dependent_part(ctx, inner, var_name)?;
            Some(ctx.add(Expr::Neg(inner)))
        }
        Expr::Hold(inner) => additive_var_dependent_part(ctx, inner, var_name),
        _ if cas_math::expr_predicates::contains_named_var(ctx, expr, var_name) => Some(expr),
        _ => None,
    }
}

fn expr_is_affine_in_named_var(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    if !cas_math::expr_predicates::contains_named_var(ctx, expr, var_name) {
        return true;
    }

    match ctx.get(expr) {
        Expr::Variable(sym_id) => ctx.sym_name(*sym_id) == var_name,
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            expr_is_affine_in_named_var(ctx, *left, var_name)
                && expr_is_affine_in_named_var(ctx, *right, var_name)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => expr_is_affine_in_named_var(ctx, *inner, var_name),
        Expr::Mul(left, right) => {
            let left_has_var = cas_math::expr_predicates::contains_named_var(ctx, *left, var_name);
            let right_has_var =
                cas_math::expr_predicates::contains_named_var(ctx, *right, var_name);
            match (left_has_var, right_has_var) {
                (false, false) => true,
                (false, true) => expr_is_affine_in_named_var(ctx, *right, var_name),
                (true, false) => expr_is_affine_in_named_var(ctx, *left, var_name),
                (true, true) => false,
            }
        }
        Expr::Div(num, den) => {
            !cas_math::expr_predicates::contains_named_var(ctx, *den, var_name)
                && expr_is_affine_in_named_var(ctx, *num, var_name)
        }
        _ => false,
    }
}

fn constant_polynomial_ratio(
    numerator: &Polynomial,
    denominator: &Polynomial,
) -> Option<BigRational> {
    if denominator.is_zero() {
        return None;
    }

    let pivot = denominator
        .coeffs
        .iter()
        .position(|coeff| !coeff.is_zero())?;
    let numerator_pivot = numerator
        .coeffs
        .get(pivot)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let scale = numerator_pivot / denominator.coeffs[pivot].clone();
    let len = numerator.coeffs.len().max(denominator.coeffs.len());

    for idx in 0..len {
        let left = numerator
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        let right = denominator
            .coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(BigRational::zero)
            * scale.clone();
        if left != right {
            return None;
        }
    }

    Some(scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;

    fn parse_in(ctx: &mut Context, raw: &str) -> ExprId {
        cas_parser::parse(raw, ctx).unwrap()
    }

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn recognizes_constant_scaled_affine_reciprocal_trig_derivative_products() {
        let mut ctx = Context::new();

        for raw in [
            "sec(x)*tan(x)",
            "k*sec(x)*tan(x)",
            "k*a*sec(a*x+b)*tan(a*x+b)",
            "-k*a*sec(b-a*x)*tan(b-a*x)",
            "k*a*csc(a*x+b)*cot(a*x+b)",
            "-k*a*csc(b-a*x)*cot(b-a*x)",
        ] {
            let target = parse_in(&mut ctx, raw);
            assert!(
                constant_scaled_reciprocal_trig_derivative_product_source(&ctx, target, "x"),
                "{raw}"
            );
        }
    }

    #[test]
    fn rejects_non_affine_reciprocal_trig_products_without_derivative_factor() {
        let mut ctx = Context::new();

        for raw in [
            "sec(x^2)*tan(x^2)",
            "k*sec(x^2+b)*tan(x^2+b)",
            "csc(x^2)*cot(x^2)",
            "k*csc(x^2+b)*cot(x^2+b)",
            "sec(sqrt(x))*tan(sqrt(x))",
        ] {
            let target = parse_in(&mut ctx, raw);
            assert!(
                !constant_scaled_reciprocal_trig_derivative_product_source(&ctx, target, "x"),
                "{raw}"
            );
        }
    }

    #[test]
    fn quotient_normalization_feeds_symbolic_integration_for_affine_products() {
        let mut ctx = Context::new();

        for (raw, expected_primitive) in [
            ("sec(2*x+1)*tan(2*x+1)", "sec(2 * x + 1) / 2"),
            ("csc(2*x+1)*cot(2*x+1)", "-csc(2 * x + 1) / 2"),
            ("sec(1-2*x)*tan(1-2*x)", "-sec(1 - 2 * x) / 2"),
            ("csc(1-2*x)*cot(1-2*x)", "csc(1 - 2 * x) / 2"),
        ] {
            let target = parse_in(&mut ctx, raw);
            let normalized = reciprocal_trig_derivative_integrand_quotient(&mut ctx, target)
                .unwrap_or_else(|| panic!("expected quotient normalization for {raw}"));
            let primitive = cas_math::symbolic_integration_support::integrate_symbolic_expr(
                &mut ctx, normalized, "x",
            )
            .unwrap_or_else(|| panic!("expected normalized integrand to integrate for {raw}"));

            assert_eq!(rendered(&ctx, primitive), expected_primitive, "{raw}");
        }
    }

    #[test]
    fn verified_source_requires_explicit_derivative_factor_and_reports_pole() {
        let mut ctx = Context::new();

        for (raw, expected_required) in [
            ("a*sec(a*x+b)*tan(a*x+b)", "cos(a * x + b)"),
            ("-a*csc(a*x+b)*cot(a*x+b)", "sin(a * x + b)"),
            ("2*x*sec(x^2+b)*tan(x^2+b)", "cos(x^2 + b)"),
            ("2*x*csc(x^2+b)*cot(x^2+b)", "sin(x^2 + b)"),
        ] {
            let target = parse_in(&mut ctx, raw);
            let (_source, required_nonzero) =
                verified_reciprocal_trig_derivative_product_source_with_domain(
                    &mut ctx, target, "x",
                )
                .unwrap_or_else(|| panic!("expected verified source for {raw}"));

            assert_eq!(rendered(&ctx, required_nonzero), expected_required, "{raw}");
        }

        for raw in [
            "sec(x^2+b)*tan(x^2+b)",
            "k*sec(x^2+b)*tan(x^2+b)",
            "cosh(x^2+b)/sinh(x^2+b)^2",
        ] {
            let target = parse_in(&mut ctx, raw);
            assert!(
                verified_reciprocal_trig_derivative_product_source_with_domain(
                    &mut ctx, target, "x",
                )
                .is_none(),
                "{raw}"
            );
        }
    }

    #[test]
    fn verified_affine_source_normalizes_and_reports_pole() {
        let mut ctx = Context::new();

        for (raw, expected_required) in [
            ("sec(2*x+1)*tan(2*x+1)", "cos(2 * x + 1)"),
            ("csc(2*x+1)*cot(2*x+1)", "sin(2 * x + 1)"),
            ("sec(1-2*x)*tan(1-2*x)", "cos(1 - 2 * x)"),
            ("csc(1-2*x)*cot(1-2*x)", "sin(1 - 2 * x)"),
        ] {
            let target = parse_in(&mut ctx, raw);
            let (_source, required_nonzero) =
                verified_affine_reciprocal_trig_derivative_product_source_with_domain(
                    &mut ctx, target, "x",
                )
                .unwrap_or_else(|| panic!("expected verified affine source for {raw}"));

            assert_eq!(rendered(&ctx, required_nonzero), expected_required, "{raw}");
        }

        for raw in [
            "sec(x^2)*tan(x^2)",
            "k*sec(x^2+b)*tan(x^2+b)",
            "csc(x^2)*cot(x^2)",
        ] {
            let target = parse_in(&mut ctx, raw);
            assert!(
                verified_affine_reciprocal_trig_derivative_product_source_with_domain(
                    &mut ctx, target, "x",
                )
                .is_none(),
                "{raw}"
            );
        }
    }
}
