use super::arctan_companion_result_presentation::{
    compact_arctan_presentation_other_terms,
    ln_polynomial_coefficient_degree_for_calculus_presentation,
};
use super::presentation_utils::{unwrap_internal_hold_for_calculus, variable_named};
use super::scalar_presentation::fold_numeric_mul_constants_for_hold;
use cas_ast::ordering::compare_expr;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;

fn arctan_arg_matches_for_calculus_presentation(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
    var_name: &str,
) -> bool {
    if compare_expr(ctx, left, right) == std::cmp::Ordering::Equal {
        return true;
    }

    let Ok(left_poly) = Polynomial::from_expr(ctx, left, var_name) else {
        return false;
    };
    let Ok(right_poly) = Polynomial::from_expr(ctx, right, var_name) else {
        return false;
    };
    left_poly == right_poly
}

fn extract_arctan_term_for_calculus_presentation(
    ctx: &mut Context,
    term: ExprId,
) -> Option<(ExprId, ExprId)> {
    let term = unwrap_internal_hold_for_calculus(ctx, term);
    if let Expr::Neg(inner) = ctx.get(term).clone() {
        let (arg, coeff) = extract_arctan_term_for_calculus_presentation(ctx, inner)?;
        let coeff = ctx.add(Expr::Neg(coeff));
        return Some((arg, coeff));
    }
    if let Expr::Div(num, den) = ctx.get(term).clone() {
        let (arg, coeff) = extract_arctan_term_for_calculus_presentation(ctx, num)?;
        let coeff = ctx.add(Expr::Div(coeff, den));
        return Some((arg, coeff));
    }

    let factors = cas_math::expr_nary::MulView::from_expr(ctx, term).factors;
    let mut arctan_arg = None;
    let mut coefficient_factors = Vec::new();

    for factor in factors {
        let factor = unwrap_internal_hold_for_calculus(ctx, factor);
        match ctx.get(factor).clone() {
            Expr::Function(fn_id, args)
                if args.len() == 1
                    && matches!(
                        ctx.builtin_of(fn_id),
                        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
                    ) =>
            {
                if arctan_arg.is_some() {
                    return None;
                }
                arctan_arg = Some(args[0]);
            }
            Expr::Div(num, den) => {
                let Expr::Function(fn_id, args) = ctx.get(num).clone() else {
                    coefficient_factors.push(factor);
                    continue;
                };
                if args.len() == 1
                    && matches!(
                        ctx.builtin_of(fn_id),
                        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
                    )
                {
                    if arctan_arg.is_some() {
                        return None;
                    }
                    arctan_arg = Some(args[0]);
                    let one = ctx.num(1);
                    coefficient_factors.push(ctx.add(Expr::Div(one, den)));
                } else {
                    coefficient_factors.push(factor);
                }
            }
            Expr::Neg(inner) => {
                let inner = unwrap_internal_hold_for_calculus(ctx, inner);
                let Expr::Function(fn_id, args) = ctx.get(inner).clone() else {
                    coefficient_factors.push(factor);
                    continue;
                };
                if args.len() == 1
                    && matches!(
                        ctx.builtin_of(fn_id),
                        Some(BuiltinFn::Arctan | BuiltinFn::Atan)
                    )
                {
                    if arctan_arg.is_some() {
                        return None;
                    }
                    arctan_arg = Some(args[0]);
                    let minus_one = ctx.num(-1);
                    coefficient_factors.push(minus_one);
                } else {
                    coefficient_factors.push(factor);
                }
            }
            _ => coefficient_factors.push(factor),
        }
    }

    let arg = arctan_arg?;
    let coeff = if coefficient_factors.is_empty() {
        ctx.num(1)
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &coefficient_factors)
    };
    Some((arg, coeff))
}

fn negate_term_for_calculus_presentation(ctx: &mut Context, term: ExprId) -> ExprId {
    let term = unwrap_internal_hold_for_calculus(ctx, term);
    if let Expr::Number(value) = ctx.get(term).clone() {
        return ctx.add(Expr::Number(-value));
    }
    if let Expr::Div(num, den) = ctx.get(term).clone() {
        let num = negate_term_for_calculus_presentation(ctx, num);
        return ctx.add(Expr::Div(num, den));
    }

    let factors = cas_math::expr_nary::MulView::from_expr(ctx, term).factors;
    if factors.len() > 1 {
        let mut replaced = false;
        let mut negated_factors = Vec::with_capacity(factors.len());
        for factor in factors {
            if !replaced {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    negated_factors.push(ctx.add(Expr::Number(-value)));
                    replaced = true;
                    continue;
                }
            }
            negated_factors.push(factor);
        }
        if replaced {
            return cas_math::expr_nary::build_balanced_mul(ctx, &negated_factors);
        }
    }

    ctx.add(Expr::Neg(term))
}

fn contains_nontrivial_arctan_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let mut stack = vec![cas_ast::hold::unwrap_internal_hold(ctx, expr)];
    while let Some(current) = stack.pop() {
        match ctx.get(cas_ast::hold::unwrap_internal_hold(ctx, current)) {
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(fn_id, args) => {
                if ctx.builtin_of(*fn_id) == Some(BuiltinFn::Arctan)
                    && args.len() == 1
                    && !variable_named(ctx, args[0], var_name)
                {
                    return true;
                }
                stack.extend(args.iter().copied());
            }
            _ => {}
        }
    }
    false
}

pub(super) fn flatten_subtracting_additive_group_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    let Expr::Sub(left, right) = ctx.get(expr).clone() else {
        return None;
    };
    let right = unwrap_internal_hold_for_calculus(ctx, right);
    if !matches!(ctx.get(right), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }
    if !contains_nontrivial_arctan_for_calculus_presentation(ctx, right, var_name) {
        return None;
    }
    if ln_polynomial_coefficient_degree_for_calculus_presentation(ctx, left, var_name)? > 5 {
        return None;
    }

    let mut additive_terms = Vec::new();
    collect_additive_terms_for_arctan_calculus_presentation(
        ctx,
        left,
        cas_math::expr_nary::Sign::Pos,
        &mut additive_terms,
    );
    collect_additive_terms_for_arctan_calculus_presentation(
        ctx,
        right,
        cas_math::expr_nary::Sign::Neg,
        &mut additive_terms,
    );
    if additive_terms.len() < 3 {
        return None;
    }

    let terms = additive_terms
        .into_iter()
        .map(|(term, sign)| {
            let signed = match sign {
                cas_math::expr_nary::Sign::Pos => term,
                cas_math::expr_nary::Sign::Neg => negate_term_for_calculus_presentation(ctx, term),
            };
            fold_numeric_mul_constants_for_hold(ctx, signed)
        })
        .collect::<Vec<_>>();

    Some(cas_math::expr_nary::build_balanced_add(ctx, &terms))
}

pub(super) fn compact_arctan_additive_terms_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);

    match ctx.get(expr).clone() {
        Expr::Mul(left, right) => {
            if cas_ast::views::as_rational_const(ctx, left, 8).is_some() {
                let compact =
                    compact_arctan_additive_terms_for_calculus_presentation(ctx, right, var_name)?;
                let compact = cas_ast::hold::wrap_hold(ctx, compact);
                return Some(ctx.add(Expr::Mul(left, compact)));
            }
            if cas_ast::views::as_rational_const(ctx, right, 8).is_some() {
                let compact =
                    compact_arctan_additive_terms_for_calculus_presentation(ctx, left, var_name)?;
                let compact = cas_ast::hold::wrap_hold(ctx, compact);
                return Some(ctx.add(Expr::Mul(compact, right)));
            }
        }
        Expr::Div(num, den) if cas_ast::views::as_rational_const(ctx, den, 8).is_some() => {
            let compact =
                compact_arctan_additive_terms_for_calculus_presentation(ctx, num, var_name)?;
            let compact = cas_ast::hold::wrap_hold(ctx, compact);
            return Some(ctx.add(Expr::Div(compact, den)));
        }
        _ => {}
    }

    let mut additive_terms = Vec::new();
    collect_additive_terms_for_arctan_calculus_presentation(
        ctx,
        expr,
        cas_math::expr_nary::Sign::Pos,
        &mut additive_terms,
    );
    if additive_terms.len() < 2 {
        return None;
    }

    let mut arctan_arg = None;
    let mut arctan_coefficients = Vec::new();
    let mut other_terms = Vec::new();
    let mut arctan_term_count = 0usize;

    for (term, sign) in additive_terms {
        if let Some((arg, coeff)) = extract_arctan_term_for_calculus_presentation(ctx, term) {
            if let Some(existing_arg) = arctan_arg {
                if !arctan_arg_matches_for_calculus_presentation(ctx, existing_arg, arg, var_name) {
                    return None;
                }
            } else {
                arctan_arg = Some(arg);
            }
            arctan_term_count += 1;
            let coeff = match sign {
                cas_math::expr_nary::Sign::Pos => coeff,
                cas_math::expr_nary::Sign::Neg => negate_term_for_calculus_presentation(ctx, coeff),
            };
            let coeff = fold_numeric_mul_constants_for_hold(ctx, coeff);
            arctan_coefficients.push(coeff);
        } else {
            let signed_term = match sign {
                cas_math::expr_nary::Sign::Pos => term,
                cas_math::expr_nary::Sign::Neg => negate_term_for_calculus_presentation(ctx, term),
            };
            let signed_term = fold_numeric_mul_constants_for_hold(ctx, signed_term);
            other_terms.push(signed_term);
        }
    }

    if arctan_term_count < 2 {
        return None;
    }

    let arg = arctan_arg?;
    if Polynomial::from_expr(ctx, arg, var_name).is_err() {
        return None;
    };
    let coeff = cas_math::expr_nary::build_balanced_add(ctx, &arctan_coefficients);
    let coeff = cas_ast::hold::wrap_hold(ctx, coeff);
    let arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![arg]);
    let arctan_term = ctx.add(Expr::Mul(coeff, arctan));

    let mut terms = vec![arctan_term];
    terms.extend(compact_arctan_presentation_other_terms(
        ctx,
        other_terms,
        var_name,
    ));
    Some(cas_math::expr_nary::build_balanced_add(ctx, &terms))
}

fn collect_additive_terms_for_arctan_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    sign: cas_math::expr_nary::Sign,
    out: &mut Vec<(ExprId, cas_math::expr_nary::Sign)>,
) {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            collect_additive_terms_for_arctan_calculus_presentation(ctx, left, sign, out);
            collect_additive_terms_for_arctan_calculus_presentation(ctx, right, sign, out);
        }
        Expr::Sub(left, right) => {
            collect_additive_terms_for_arctan_calculus_presentation(ctx, left, sign, out);
            collect_additive_terms_for_arctan_calculus_presentation(ctx, right, sign.negate(), out);
        }
        Expr::Neg(inner) => {
            collect_additive_terms_for_arctan_calculus_presentation(ctx, inner, sign.negate(), out);
        }
        _ => out.push((expr, sign)),
    }
}

#[cfg(test)]
mod tests {
    use super::super::arctan_polynomial_integrand_presentation::polynomial_times_arctan_affine_integrand_for_diff_shortcut;
    use super::super::integration::integrate;
    use super::super::scalar_presentation::fold_numeric_mul_constants_for_hold;
    use super::compact_arctan_additive_terms_for_calculus_presentation;
    use cas_ast::{Context, Expr, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn compact_arctan_additive_terms_accepts_negative_affine_argument() {
        let mut ctx = Context::new();
        let expr = parse(
            "1/3*x^3*arctan(1-x) + 1/3*ln(x^2+2-2*x) + 2/3*arctan(1-x) + 1/6*x^2 + 2/3*x",
            &mut ctx,
        )
        .unwrap();
        let compact =
            compact_arctan_additive_terms_for_calculus_presentation(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact).matches("arctan(1 - x)").count(), 1);

        let raw_by_parts = parse(
            "1/3*x^3*arctan(1-x) - (-1/3*ln(x^2+2-2*x) - 2/3*arctan(1-x) - 1/6*x^2 - 2/3*x)",
            &mut ctx,
        )
        .unwrap();
        let compact =
            compact_arctan_additive_terms_for_calculus_presentation(&mut ctx, raw_by_parts, "x")
                .unwrap();
        assert_eq!(rendered(&ctx, compact).matches("arctan(1 - x)").count(), 1);

        let duplicate_companions = parse(
            "1/3*ln(x^2+2-2*x) + 1/2*ln(x^2+2-2*x) + 1/3*x^3*arctan(1-x) + 1/2*x^2*arctan(1-x) + 2/3*arctan(1-x) + 1/6*x^2 + 1/2*x + 2/3*x",
            &mut ctx,
        )
        .unwrap();
        let compact = compact_arctan_additive_terms_for_calculus_presentation(
            &mut ctx,
            duplicate_companions,
            "x",
        )
        .unwrap();
        let rendered = rendered(&ctx, compact);
        assert_eq!(rendered.matches("ln(x^2 + 2 - 2 * x)").count(), 1);
        assert!(rendered.contains("5/6 * ln(x^2 + 2 - 2 * x)"));
        assert!(rendered.contains("7/6 * x"));
    }

    #[test]
    fn integrate_pipeline_compacts_negative_affine_arctan_by_parts_result() {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse("integrate(x^2*arctan(1-x), x)", &mut simplifier.context).unwrap();
        let target = match simplifier.context.get(expr) {
            Expr::Function(_, args) => args[0],
            _ => expr,
        };
        assert!(
            cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_arctan_affine_target(
                &mut simplifier.context,
                target,
                "x",
            )
        );
        assert!(polynomial_times_arctan_affine_integrand_for_diff_shortcut(
            &simplifier.context,
            target,
            "x"
        ));
        let raw = integrate(&mut simplifier.context, target, "x").unwrap();
        let raw = fold_numeric_mul_constants_for_hold(&mut simplifier.context, raw);
        let compact = compact_arctan_additive_terms_for_calculus_presentation(
            &mut simplifier.context,
            raw,
            "x",
        )
        .unwrap();
        assert_eq!(
            rendered(&simplifier.context, compact)
                .matches("arctan(1 - x)")
                .count(),
            1
        );
        let (result, _steps) = simplifier.simplify(expr);
        let rendered = rendered(&simplifier.context, result);

        assert_eq!(rendered.matches("arctan(1 - x)").count(), 1);
    }
}
