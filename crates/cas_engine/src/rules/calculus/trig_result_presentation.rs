use super::presentation_utils::unwrap_internal_hold_for_calculus;
use super::scalar_presentation::scale_expr_for_calculus_presentation;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};

fn collect_scaled_trig_square_terms_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
    scale: BigRational,
    linear_coeff: &mut BigRational,
    sin_terms: &mut Vec<(ExprId, BigRational)>,
) -> Option<()> {
    if scale.is_zero() {
        return Some(());
    }

    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Number(value) if value.is_zero() => Some(()),
        Expr::Add(left, right) => {
            collect_scaled_trig_square_terms_for_integration_presentation(
                ctx,
                left,
                var_name,
                scale.clone(),
                linear_coeff,
                sin_terms,
            )?;
            collect_scaled_trig_square_terms_for_integration_presentation(
                ctx,
                right,
                var_name,
                scale,
                linear_coeff,
                sin_terms,
            )
        }
        Expr::Sub(left, right) => {
            collect_scaled_trig_square_terms_for_integration_presentation(
                ctx,
                left,
                var_name,
                scale.clone(),
                linear_coeff,
                sin_terms,
            )?;
            collect_scaled_trig_square_terms_for_integration_presentation(
                ctx,
                right,
                var_name,
                -scale,
                linear_coeff,
                sin_terms,
            )
        }
        Expr::Neg(inner) => collect_scaled_trig_square_terms_for_integration_presentation(
            ctx,
            inner,
            var_name,
            -scale,
            linear_coeff,
            sin_terms,
        ),
        Expr::Div(num, den) => {
            let den_scale = cas_ast::views::as_rational_const(ctx, den, 8)?;
            if den_scale.is_zero() {
                return None;
            }
            collect_scaled_trig_square_terms_for_integration_presentation(
                ctx,
                num,
                var_name,
                scale / den_scale,
                linear_coeff,
                sin_terms,
            )
        }
        Expr::Mul(_, _) => {
            let mut term_scale = scale;
            let mut non_numeric = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    term_scale *= value;
                    continue;
                }
                if non_numeric.replace(factor).is_some() {
                    return None;
                }
            }
            collect_scaled_trig_square_terms_for_integration_presentation(
                ctx,
                non_numeric?,
                var_name,
                term_scale,
                linear_coeff,
                sin_terms,
            )
        }
        Expr::Variable(sym_id) if ctx.sym_name(sym_id) == var_name => {
            *linear_coeff += scale;
            Some(())
        }
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(fn_id) == Some(BuiltinFn::Sin) =>
        {
            sin_terms.push((args[0], scale));
            Some(())
        }
        _ => None,
    }
}

pub(super) fn compact_trig_square_reduction_primitive_for_integration_presentation(
    ctx: &mut Context,
    result: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let mut linear_coeff = BigRational::zero();
    let mut sin_terms = Vec::new();
    collect_scaled_trig_square_terms_for_integration_presentation(
        ctx,
        result,
        var_name,
        BigRational::one(),
        &mut linear_coeff,
        &mut sin_terms,
    )?;

    if linear_coeff != BigRational::new(1.into(), 2.into()) {
        return None;
    }

    let (sin_arg, sin_coeff) = sin_terms
        .iter()
        .find_map(|(arg, coeff)| (!coeff.is_zero()).then_some((*arg, coeff.clone())))?;
    if sin_terms.iter().any(|(arg, coeff)| {
        !coeff.is_zero() && compare_expr(ctx, *arg, sin_arg) != std::cmp::Ordering::Equal
    }) {
        return None;
    }

    let var = ctx.var(var_name);
    let linear = scale_expr_for_calculus_presentation(ctx, linear_coeff, var);
    let sin = ctx.call_builtin(BuiltinFn::Sin, vec![sin_arg]);
    let trig = scale_expr_for_calculus_presentation(ctx, sin_coeff, sin);
    let compact = ctx.add(Expr::Add(linear, trig));

    (compact != result).then_some(compact)
}

#[cfg(test)]
mod tests {
    use super::compact_trig_square_reduction_primitive_for_integration_presentation;
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn compact_trig_square_reduction_primitive_expands_coefficients() {
        let cases = [
            ("1/4*(2*x - sin(2*x))", "1/2 * x - 1/4 * sin(2 * x)"),
            ("1/4*(sin(2*x) + 2*x)", "1/4 * sin(2 * x) + 1/2 * x"),
            ("1/8*(4*x - sin(4*x+2))", "1/2 * x - 1/8 * sin(4 * x + 2)"),
            ("1/8*(sin(4*x+2) + 4*x)", "1/8 * sin(4 * x + 2) + 1/2 * x"),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).unwrap();
            let compact = compact_trig_square_reduction_primitive_for_integration_presentation(
                &mut ctx, expr, "x",
            )
            .unwrap();

            assert_eq!(rendered(&ctx, compact), expected, "input: {input}");
        }
    }
}
