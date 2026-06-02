use super::presentation_utils::unwrap_internal_hold_for_calculus;
use super::scalar_presentation::scale_expr_for_calculus_presentation;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

fn trig_power_term_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId, u32)> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    if let Expr::Function(fn_id, args) = ctx.get(expr).clone() {
        if args.len() != 1 {
            return None;
        }
        return match ctx.builtin_of(fn_id) {
            Some(BuiltinFn::Sin | BuiltinFn::Cos) => Some((ctx.builtin_of(fn_id)?, args[0], 1)),
            _ => None,
        };
    }

    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    let power = cas_ast::views::as_rational_const(ctx, exp, 4)?;
    if power.denom() != &1.into() {
        return None;
    }
    let power = power.numer().to_u32()?;
    if !matches!(power, 3 | 5 | 7) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(base).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Sin | BuiltinFn::Cos) => Some((ctx.builtin_of(fn_id)?, args[0], power)),
        _ => None,
    }
}

fn collect_scaled_trig_power_terms_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    scale: BigRational,
    terms: &mut Vec<(BuiltinFn, ExprId, u32, BigRational)>,
) -> Option<()> {
    if scale.is_zero() {
        return Some(());
    }

    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Number(value) if value.is_zero() => Some(()),
        Expr::Add(left, right) => {
            collect_scaled_trig_power_terms_for_integration_presentation(
                ctx,
                left,
                scale.clone(),
                terms,
            )?;
            collect_scaled_trig_power_terms_for_integration_presentation(ctx, right, scale, terms)
        }
        Expr::Sub(left, right) => {
            collect_scaled_trig_power_terms_for_integration_presentation(
                ctx,
                left,
                scale.clone(),
                terms,
            )?;
            collect_scaled_trig_power_terms_for_integration_presentation(ctx, right, -scale, terms)
        }
        Expr::Neg(inner) => {
            collect_scaled_trig_power_terms_for_integration_presentation(ctx, inner, -scale, terms)
        }
        Expr::Div(num, den) => {
            let den_scale = cas_ast::views::as_rational_const(ctx, den, 8)?;
            if den_scale.is_zero() {
                return None;
            }
            collect_scaled_trig_power_terms_for_integration_presentation(
                ctx,
                num,
                scale / den_scale,
                terms,
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
            collect_scaled_trig_power_terms_for_integration_presentation(
                ctx,
                non_numeric?,
                term_scale,
                terms,
            )
        }
        _ => {
            let (builtin, arg, power) = trig_power_term_for_integration_presentation(ctx, expr)?;
            terms.push((builtin, arg, power, scale));
            Some(())
        }
    }
}

fn trig_power_presentation_coeff(
    terms: &[(BuiltinFn, ExprId, u32, BigRational)],
    power: u32,
) -> BigRational {
    terms
        .iter()
        .filter_map(|(_, _, term_power, coeff)| (*term_power == power).then_some(coeff.clone()))
        .fold(BigRational::zero(), |acc, coeff| acc + coeff)
}

pub(super) fn compact_trig_odd_power_reduction_primitive_for_integration_presentation(
    ctx: &mut Context,
    result: ExprId,
) -> Option<ExprId> {
    let mut terms = Vec::new();
    collect_scaled_trig_power_terms_for_integration_presentation(
        ctx,
        result,
        BigRational::one(),
        &mut terms,
    )?;
    if terms.len() < 2 {
        return None;
    }

    let (builtin, arg) = terms
        .iter()
        .find_map(|(builtin, arg, _, coeff)| (!coeff.is_zero()).then_some((*builtin, *arg)))?;
    if terms.iter().any(|(term_builtin, term_arg, power, coeff)| {
        !coeff.is_zero()
            && (*term_builtin != builtin
                || compare_expr(ctx, *term_arg, arg) != std::cmp::Ordering::Equal
                || !matches!(*power, 1 | 3 | 5))
    }) {
        return None;
    }

    let linear_coeff = trig_power_presentation_coeff(&terms, 1);
    if linear_coeff.is_zero() {
        return None;
    }

    let scale = match builtin {
        BuiltinFn::Cos if linear_coeff.is_negative() => -linear_coeff.clone(),
        BuiltinFn::Sin if linear_coeff.is_positive() => linear_coeff.clone(),
        _ => return None,
    };
    if scale.is_zero() {
        return None;
    }

    let cubic_coeff = trig_power_presentation_coeff(&terms, 3);
    if cubic_coeff.is_zero() {
        return None;
    }
    let fifth_coeff = trig_power_presentation_coeff(&terms, 5);
    let has_fifth = !fifth_coeff.is_zero();

    let one_third = BigRational::new(1.into(), 3.into());
    let two_thirds = BigRational::new(2.into(), 3.into());
    let one_fifth = BigRational::new(1.into(), 5.into());
    let expected_cubic = if has_fifth {
        match builtin {
            BuiltinFn::Cos => two_thirds.clone() * scale.clone(),
            BuiltinFn::Sin => -two_thirds.clone() * scale.clone(),
            _ => return None,
        }
    } else {
        match builtin {
            BuiltinFn::Cos => one_third.clone() * scale.clone(),
            BuiltinFn::Sin => -one_third.clone() * scale.clone(),
            _ => return None,
        }
    };
    let expected_fifth = if has_fifth {
        match builtin {
            BuiltinFn::Cos => -one_fifth.clone() * scale.clone(),
            BuiltinFn::Sin => one_fifth.clone() * scale.clone(),
            _ => return None,
        }
    } else {
        BigRational::zero()
    };

    if cubic_coeff != expected_cubic || fifth_coeff != expected_fifth {
        return None;
    }

    let three = ctx.num(3);
    let five = ctx.num(5);
    let base = ctx.call_builtin(builtin, vec![arg]);
    let base_cubed = ctx.add(Expr::Pow(base, three));
    let base_fifth = ctx.add(Expr::Pow(base, five));

    let linear = scale_expr_for_calculus_presentation(ctx, linear_coeff, base);
    let cubic = scale_expr_for_calculus_presentation(ctx, expected_cubic, base_cubed);

    let compact = if has_fifth {
        let fifth = scale_expr_for_calculus_presentation(ctx, expected_fifth, base_fifth);
        match builtin {
            BuiltinFn::Cos => {
                let first_two = ctx.add(Expr::Add(cubic, linear));
                ctx.add(Expr::Add(first_two, fifth))
            }
            BuiltinFn::Sin => {
                let first_two = ctx.add(Expr::Add(linear, fifth));
                ctx.add(Expr::Add(first_two, cubic))
            }
            _ => return None,
        }
    } else {
        match builtin {
            BuiltinFn::Cos => ctx.add(Expr::Add(cubic, linear)),
            BuiltinFn::Sin => ctx.add(Expr::Add(linear, cubic)),
            _ => return None,
        }
    };

    (compact != result).then_some(compact)
}

#[cfg(test)]
mod tests {
    use super::compact_trig_odd_power_reduction_primitive_for_integration_presentation;
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn compact_trig_odd_power_reduction_primitive_expands_coefficients() {
        let cases = [
            ("1/3*(cos(x)^3 - 3*cos(x))", "1/3 * cos(x)^3 - cos(x)"),
            ("1/3*(3*sin(x) - sin(x)^3)", "sin(x) - 1/3 * sin(x)^3"),
            (
                "1/6*(cos(2*x+1)^3 - 3*cos(2*x+1))",
                "1/6 * cos(2 * x + 1)^3 - 1/2 * cos(2 * x + 1)",
            ),
            (
                "1/5*(10/3*cos(x)^3 - cos(x)^5 - 5*cos(x))",
                "2/3 * cos(x)^3 - cos(x) - 1/5 * cos(x)^5",
            ),
            (
                "1/5*(sin(x)^5 + 5*sin(x) - 10/3*sin(x)^3)",
                "sin(x) + 1/5 * sin(x)^5 - 2/3 * sin(x)^3",
            ),
            (
                "1/10*(10/3*cos(2*x+1)^3 - cos(2*x+1)^5 - 5*cos(2*x+1))",
                "1/3 * cos(2 * x + 1)^3 - 1/2 * cos(2 * x + 1) - 1/10 * cos(2 * x + 1)^5",
            ),
        ];

        for (input, expected) in cases {
            let mut ctx = Context::new();
            let expr = parse(input, &mut ctx).unwrap();
            let compact = compact_trig_odd_power_reduction_primitive_for_integration_presentation(
                &mut ctx, expr,
            )
            .unwrap();

            assert_eq!(rendered(&ctx, compact), expected, "input: {input}");
        }
    }
}
