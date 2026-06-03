//! Presentation for `ln(sqrt(f) + sqrt(g))` routes with equal derivatives.

use super::diff_rule_support::finalize_diff_rewrite_with_conditions;
use super::polynomial_support::{
    polynomial_is_strictly_positive_everywhere, polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    scale_expr_for_calculus_presentation,
};
use crate::rule::Rewrite;
use crate::symbolic_calculus_call_support::NamedVarCall;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::One;

pub(super) fn ln_sum_of_equal_derivative_roots_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (compact, _) = ln_sum_of_equal_derivative_roots_derivative_presentation_with_domain(
        ctx, target, var_name,
    )?;
    Some(compact)
}

pub(super) fn ln_sum_of_equal_derivative_roots_derivative_rewrite(
    ctx: &mut Context,
    call: &NamedVarCall,
    target: ExprId,
) -> Option<Rewrite> {
    let result =
        ln_sum_of_equal_derivative_roots_derivative_presentation(ctx, target, &call.var_name)?;
    Some(finalize_diff_rewrite_with_conditions(
        ctx,
        call,
        target,
        result,
        Vec::new(),
    ))
}

pub(crate) fn ln_sum_of_equal_derivative_roots_derivative_presentation_with_domain(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (fn_id, args) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let terms = cas_math::expr_nary::add_terms_signed(ctx, args[0]);
    if terms.len() != 2 {
        return None;
    }

    let mut radicands = Vec::with_capacity(2);
    for (term, sign) in terms {
        if sign == cas_math::expr_nary::Sign::Neg {
            return None;
        }
        radicands.push(extract_square_root_base(ctx, term)?);
    }

    let left_poly = polynomial_radicand_for_calculus_presentation(ctx, radicands[0], var_name)?;
    let right_poly = polynomial_radicand_for_calculus_presentation(ctx, radicands[1], var_name)?;
    let left_positive_everywhere = polynomial_is_strictly_positive_everywhere(&left_poly);
    let right_positive_everywhere = polynomial_is_strictly_positive_everywhere(&right_poly);
    let affine_pair = left_poly.degree() <= 1 && right_poly.degree() <= 1;
    let strictly_positive_quadratic_pair = left_poly.degree() <= 2
        && right_poly.degree() <= 2
        && left_positive_everywhere
        && right_positive_everywhere;
    if !affine_pair && !strictly_positive_quadratic_pair {
        return None;
    }
    let derivative_poly = left_poly.derivative();
    if derivative_poly != right_poly.derivative() {
        return None;
    }
    if derivative_poly.is_zero() {
        let required_conditions =
            positive_radicand_conditions_for_equal_derivative_roots_presentation(
                radicands[0],
                left_positive_everywhere,
                radicands[1],
                right_positive_everywhere,
            );
        return Some((ctx.num(0), required_conditions));
    }

    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let left_sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicands[0]]);
    let right_sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicands[1]]);
    let left_sqrt = cas_ast::hold::wrap_hold(ctx, left_sqrt);
    let right_sqrt = cas_ast::hold::wrap_hold(ctx, right_sqrt);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[left_sqrt, right_sqrt]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    let compact = ctx.add(Expr::Div(numerator, denominator));
    let required_conditions = positive_radicand_conditions_for_equal_derivative_roots_presentation(
        radicands[0],
        left_positive_everywhere,
        radicands[1],
        right_positive_everywhere,
    );
    Some((cas_ast::hold::wrap_hold(ctx, compact), required_conditions))
}

fn positive_radicand_conditions_for_equal_derivative_roots_presentation(
    left_radicand: ExprId,
    left_positive_everywhere: bool,
    right_radicand: ExprId,
    right_positive_everywhere: bool,
) -> Vec<crate::ImplicitCondition> {
    let mut conditions = Vec::with_capacity(2);
    if !left_positive_everywhere {
        conditions.push(crate::ImplicitCondition::Positive(left_radicand));
    }
    if !right_positive_everywhere {
        conditions.push(crate::ImplicitCondition::Positive(right_radicand));
    }
    conditions
}

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::{
        ln_sum_of_equal_derivative_roots_derivative_presentation,
        ln_sum_of_equal_derivative_roots_derivative_rewrite,
    };
    use crate::symbolic_calculus_call_support::NamedVarCall;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn ln_sum_of_equal_derivative_roots_presentation_accepts_scaled_affines() {
        let mut ctx = Context::new();
        let expr = parse("ln(sqrt(2*x+1)+sqrt(2*x+3))", &mut ctx).unwrap();
        let compact = ln_sum_of_equal_derivative_roots_derivative_presentation(&mut ctx, expr, "x")
            .unwrap_or_else(|| {
                panic!("scaled affine equal-derivative root sum should be recognized")
            });

        assert_eq!(
            rendered(&ctx, compact),
            "1 / (sqrt(2 * x + 1) * sqrt(2 * x + 3))"
        );
    }

    #[test]
    fn ln_sum_of_equal_derivative_roots_rewrite_preserves_result() {
        let mut ctx = Context::new();
        let target = parse("ln(sqrt(2*x+1)+sqrt(2*x+3))", &mut ctx).unwrap();
        let call = NamedVarCall {
            target,
            var_name: "x".to_string(),
        };
        let rewrite =
            ln_sum_of_equal_derivative_roots_derivative_rewrite(&mut ctx, &call, target).unwrap();

        assert_eq!(
            rendered(&ctx, rewrite.new_expr),
            "1 / (sqrt(2 * x + 1) * sqrt(2 * x + 3))"
        );
    }
}
