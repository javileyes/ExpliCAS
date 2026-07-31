//! `focused_rule_substeps`: familia `radicals`.
//!
//! Ver la cabecera de `focused_rule_substeps.rs` para el contexto.

use super::*;

pub(super) fn generate_canonicalize_roots_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    if before == after {
        return Vec::new();
    }

    vec![concrete_expr_substep(
        ctx,
        "Reescribir la raíz como potencia con exponente 1/2",
        before,
        after,
    )]
}

pub(super) fn generate_hidden_radical_extraction_before_like_terms_substeps(
    ctx: &Context,
    step: &Step,
    literal_factors: &[ExprId],
) -> Vec<SubStep> {
    let [literal] = literal_factors else {
        return Vec::new();
    };
    let Some(target_radicand) = numeric_square_root_radicand(ctx, *literal) else {
        return Vec::new();
    };
    let Some(global_before) = step.global_before else {
        return Vec::new();
    };
    let Some((raw_sqrt, outer_factor, reduced_radicand)) =
        find_extractable_numeric_sqrt_for_radicand(ctx, global_before, target_radicand)
    else {
        return Vec::new();
    };

    let mut work = ctx.clone();
    let reduced = build_integer_sqrt_factor_expr(&mut work, outer_factor, reduced_radicand);
    let (before_plain, before_latex) = render_temp_expr(&work, raw_sqrt);
    let (after_plain, after_latex) = render_temp_expr(&work, reduced);

    vec![formula_substep(
        "Extraer el cuadrado perfecto dentro de la raíz",
        &before_plain,
        &after_plain,
        &before_latex,
        &after_latex,
    )]
}

fn find_extractable_numeric_sqrt_for_radicand(
    ctx: &Context,
    expr: ExprId,
    target_radicand: i64,
) -> Option<(ExprId, i64, i64)> {
    if let Some(raw_radicand) = numeric_square_root_radicand(ctx, expr) {
        let (outer_factor, reduced_radicand) = square_free_decompose(raw_radicand);
        if outer_factor > 1 && reduced_radicand == target_radicand {
            return Some((expr, outer_factor, reduced_radicand));
        }
    }

    match ctx.get(expr) {
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            find_extractable_numeric_sqrt_for_radicand(ctx, *left, target_radicand).or_else(|| {
                find_extractable_numeric_sqrt_for_radicand(ctx, *right, target_radicand)
            })
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            find_extractable_numeric_sqrt_for_radicand(ctx, *inner, target_radicand)
        }
        Expr::Function(_, args) => args
            .iter()
            .find_map(|arg| find_extractable_numeric_sqrt_for_radicand(ctx, *arg, target_radicand)),
        Expr::Matrix { data, .. } => data.iter().find_map(|item| {
            find_extractable_numeric_sqrt_for_radicand(ctx, *item, target_radicand)
        }),
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => None,
    }
}

fn numeric_square_root_radicand(ctx: &Context, expr: ExprId) -> Option<i64> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            integer_value(ctx, args[0])
        }
        Expr::Pow(base, exp)
            if as_rational_const(ctx, *exp, 8)? == BigRational::new(1.into(), 2.into()) =>
        {
            integer_value(ctx, *base)
        }
        _ => None,
    }
}

fn build_integer_sqrt_factor_expr(
    ctx: &mut Context,
    outer_factor: i64,
    reduced_radicand: i64,
) -> ExprId {
    let radicand = ctx.num(reduced_radicand);
    let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    if outer_factor == 1 {
        sqrt
    } else {
        let coefficient = ctx.num(outer_factor);
        ctx.add(Expr::Mul(coefficient, sqrt))
    }
}

pub(super) fn generate_sqrt_perfect_square_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let direct_substeps = generate_sqrt_perfect_square_core_substeps(ctx, before, after);
    if !direct_substeps.is_empty() {
        return direct_substeps;
    }

    let mut work = ctx.clone();
    let Some(plan) = try_cancel_common_additive_terms_expr(&mut work, before, after) else {
        return Vec::new();
    };
    if plan.new_lhs == before && plan.new_rhs == after {
        return Vec::new();
    }

    generate_sqrt_perfect_square_core_substeps(&work, plan.new_lhs, plan.new_rhs)
}

fn generate_sqrt_perfect_square_core_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Vec<SubStep> {
    let Some(radicand) = sqrt_radicand(ctx, before) else {
        return Vec::new();
    };
    let Some(abs_arg) = abs_argument(ctx, after) else {
        return Vec::new();
    };

    let square_display = squared_display(ctx, abs_arg);
    let square_latex = squared_latex(ctx, abs_arg);

    if is_direct_square_of(ctx, radicand, abs_arg) {
        let base_display = display_expr(ctx, abs_arg);
        let base_latex = latex_expr(ctx, abs_arg);
        return vec![
            SubStep::new(
                "Identificar la base del cuadrado",
                display_expr(ctx, radicand),
                format!("u = {base_display}"),
            )
            .with_before_latex(latex_expr(ctx, radicand))
            .with_after_latex(format!("u = {base_latex}")),
            SubStep::new(
                "La raíz de un cuadrado da un valor absoluto",
                "sqrt(u^2)",
                "|u|",
            )
            .with_before_latex("\\sqrt{{u}^{2}}")
            .with_after_latex("|u|"),
        ];
    }

    vec![
        SubStep::new(
            "Reescribir el radicando como un cuadrado perfecto",
            display_expr(ctx, radicand),
            square_display.clone(),
        )
        .with_before_latex(latex_expr(ctx, radicand))
        .with_after_latex(square_latex.clone()),
        SubStep::new(
            "La raíz de un cuadrado da un valor absoluto",
            format!("sqrt({})", square_display),
            display_expr(ctx, after),
        )
        .with_before_latex(format!("\\sqrt{{{}}}", square_latex))
        .with_after_latex(latex_expr(ctx, after)),
    ]
}

pub(super) fn generate_square_of_square_root_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(radicand) = square_of_square_root_radicand(ctx, before) else {
        return Vec::new();
    };

    vec![
        SubStep::new(
            "Identificar el radicando de la raíz principal",
            display_expr(ctx, before),
            format!("u = {}", display_expr(ctx, radicand)),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(format!("u = {}", latex_expr(ctx, radicand))),
        SubStep::new(
            "El cuadrado deshace la raíz bajo la condición u ≥ 0",
            "sqrt(u)^2",
            display_expr(ctx, after),
        )
        .with_before_latex("{\\sqrt{u}}^{2}")
        .with_after_latex(latex_expr(ctx, after)),
    ]
}

fn square_of_square_root_radicand(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if !is_integer_literal(ctx, *exponent, 2) {
        return None;
    }

    match ctx.get(*base) {
        Expr::Function(fn_id, args)
            if *fn_id == ctx.builtin_id(BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            Some(args[0])
        }
        Expr::Pow(radicand, inner_exponent) if is_one_half(ctx, *inner_exponent) => Some(*radicand),
        _ => None,
    }
}

pub(super) fn sqrt_empty_positive_domain_diff_substep(
    ctx: &Context,
    target: ExprId,
    after: ExprId,
    var_name: &str,
) -> Option<SubStep> {
    let Expr::Constant(Constant::Undefined) = ctx.get(after) else {
        return None;
    };
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if ctx.builtin_of(*fn_id) != Some(BuiltinFn::Sqrt) || args.len() != 1 {
        return None;
    }

    let radicand = args[0];
    let empty_positive_domain = if contains_named_var(ctx, radicand, var_name) {
        let mut scratch = ctx.clone();
        cas_math::calculus_domain_support::positive_condition_is_impossible_over_reals(
            &mut scratch,
            radicand,
            8,
        )
    } else {
        as_rational_const(ctx, radicand, 8).is_some_and(|value| value.is_negative())
    };
    if !empty_positive_domain {
        return None;
    }

    Some(
        SubStep::new(
            "Detectar dominio real vacío de la raíz",
            display_expr(ctx, target),
            display_expr(ctx, after),
        )
        .with_before_latex(latex_expr(ctx, target))
        .with_after_latex(latex_expr(ctx, after)),
    )
}

/// A vector `integrate`/`diff` narrated component by component.
///
/// The engine already works component-wise and SAYS so in the rule description
/// (`integrate_rule.rs:73`, "Integrar cada componente del vector"), but the
/// didactic chain did not recognise the `Expr::Matrix` shape and returned empty,
/// so `integrate([cos(x), e^x], x)` and `diff([x^2, sin(x)], x)` were mute while
/// their scalar halves narrate fine.
///
/// Unlike the additive narrator, pairing IS positional here and that is sound:
/// component `i` of the answer is by definition the image of component `i` of
/// the input — a matrix has coordinates, a sum does not.
/// The RootSum frontier narrated from the RESULT itself.
///
/// `integrate(1/(x^5-x-1), x)` answers with a correct
/// `root_sum(R(t), t, t·ln(x − w(t)))` and published ZERO substeps: not even the
/// name of the method, let alone why a closed form in radicals does not exist.
/// These are precisely the rows the corpus advertises as the differentiator
/// against sympy.
///
/// No engine signature is touched: the backend sums the RootSum node with the
/// elementary part, so the resolvent `R` and the witness `w` travel INSIDE the
/// answer and can be read back out of it.
pub(super) fn generate_root_sum_integration_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if step.rule_name != "Symbolic Integration" {
        return Vec::new();
    }
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return Vec::new();
    }

    let mut result = after;
    loop {
        let unwrapped = cas_ast::hold::unwrap_internal_hold(ctx, result);
        if unwrapped == result {
            break;
        }
        result = unwrapped;
    }

    // The RootSum may sit alone or added to an elementary part.
    let (root_sum, elementary) = match ctx.get(result) {
        Expr::Add(lhs, rhs) => {
            if is_root_sum_call(ctx, *lhs) {
                (*lhs, Some(*rhs))
            } else if is_root_sum_call(ctx, *rhs) {
                (*rhs, Some(*lhs))
            } else {
                return Vec::new();
            }
        }
        _ if is_root_sum_call(ctx, result) => (result, None),
        _ => return Vec::new(),
    };
    let Expr::Function(_, root_sum_args) = ctx.get(root_sum) else {
        return Vec::new();
    };
    let (resolvent, summand) = (root_sum_args[0], root_sum_args[2]);

    let mut substeps = Vec::new();
    if let Some(elementary) = elementary {
        substeps.push(
            SubStep::keyed(
                "rootsum.split_rational_part",
                vec![],
                display_expr(ctx, args[0]),
                format!(
                    "{} + {}",
                    display_expr(ctx, elementary),
                    display_expr(ctx, root_sum)
                ),
            )
            .with_before_latex(latex_expr(ctx, args[0]))
            .with_after_latex(format!(
                "{} + {}",
                latex_expr(ctx, elementary),
                latex_expr(ctx, root_sum)
            )),
        );
    }
    substeps.push(
        SubStep::keyed(
            "rootsum.no_radicals",
            vec![],
            display_expr(ctx, args[0]),
            format!("R(t) = {}", display_expr(ctx, resolvent)),
        )
        .with_before_latex(latex_expr(ctx, args[0]))
        .with_after_latex(format!("R(t) = {}", latex_expr(ctx, resolvent))),
    );
    substeps.push(
        SubStep::keyed(
            "rootsum.read_the_sum",
            vec![],
            format!("R(t) = {}", display_expr(ctx, resolvent)),
            display_expr(ctx, root_sum),
        )
        .with_before_latex(format!("R(t) = {}", latex_expr(ctx, resolvent)))
        .with_after_latex(latex_expr(ctx, root_sum)),
    );
    let _ = summand;
    substeps
}

fn is_root_sum_call(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Function(fn_id, args)
        if ctx.sym_name(*fn_id) == "root_sum" && args.len() == 3)
}

pub(super) fn polynomial_base_sqrt_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let (builtin, arg) = unary_builtin_arg(ctx, expr)?;
    (builtin == BuiltinFn::Sqrt).then_some(arg)
}

pub(super) fn generate_arctan_sqrt_reciprocal_table_integration_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    if step.rule_name != "Symbolic Integration" {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return Vec::new();
    }
    if let Expr::Function(after_fn_id, _) = ctx.get(after) {
        if ctx.sym_name(*after_fn_id) == "integrate" {
            return Vec::new();
        }
    }

    let Expr::Variable(var_sym) = ctx.get(args[1]) else {
        return Vec::new();
    };
    let var_name = ctx.sym_name(*var_sym);
    let is_arctan_sqrt_target =
        cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_sqrt_var_reciprocal_target(
            ctx,
            args[0],
            var_name,
        )
        || cas_math::symbolic_integration_support::integrate_symbolic_is_arctan_sqrt_affine_derivative_target(
            ctx,
            args[0],
            var_name,
        );
    if !is_arctan_sqrt_target {
        return Vec::new();
    }
    let Some(table_match) = arctan_sqrt_var_result_match(ctx, after, var_name) else {
        return Vec::new();
    };

    vec![
        SubStep::new(
            "Usar la regla de u'/(1+u^2) -> arctan(u)",
            display_expr(ctx, args[0]),
            display_expr(ctx, after),
        )
        .with_before_latex(latex_expr(ctx, args[0]))
        .with_after_latex(latex_expr(ctx, after)),
        SubStep::keyed(
            "usub.identify_u_du",
            vec![],
            format!("u = {}", display_expr(ctx, table_match.arg)),
            format!("du = {} dx", table_match.derivative_display),
        )
        .with_before_latex(format!("u = {}", latex_expr(ctx, table_match.arg)))
        .with_after_latex(format!("du = {}\\,dx", table_match.derivative_latex)),
    ]
}

fn arctan_sqrt_var_result_match(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ArctanSqrtVarTableMatch> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Function(_, _) => arctan_sqrt_var_function_match(ctx, expr, var_name),
        Expr::Neg(inner) | Expr::Hold(inner) => arctan_sqrt_var_result_match(ctx, *inner, var_name),
        Expr::Mul(left, right) => {
            if as_rational_const(ctx, *left, 8).is_some() {
                arctan_sqrt_var_result_match(ctx, *right, var_name)
            } else if as_rational_const(ctx, *right, 8).is_some() {
                arctan_sqrt_var_result_match(ctx, *left, var_name)
            } else if !contains_named_var(ctx, *left, var_name) {
                arctan_sqrt_var_result_match(ctx, *right, var_name)
            } else if !contains_named_var(ctx, *right, var_name) {
                arctan_sqrt_var_result_match(ctx, *left, var_name)
            } else {
                None
            }
        }
        Expr::Div(num, den) => {
            if !contains_named_var(ctx, *den, var_name) {
                arctan_sqrt_var_result_match(ctx, *num, var_name)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn arctan_sqrt_var_function_match(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ArctanSqrtVarTableMatch> {
    let (builtin, arg) = unary_builtin_arg(ctx, expr)?;
    if !matches!(builtin, BuiltinFn::Arctan | BuiltinFn::Atan) {
        return None;
    }
    let arg_match = arctan_sqrt_var_arg_match(ctx, arg, var_name)?;
    if !arg_match.scale.is_positive() {
        return None;
    }
    let (derivative_display, derivative_latex) =
        if let Some(parameter) = arg_match.symbolic_denominator {
            symbolic_denominator_sqrt_affine_derivative_display_and_latex(
                ctx,
                &arg_match.scale,
                arg_match.radicand,
                parameter,
                var_name,
            )?
        } else if let Some(parameter) = arg_match.symbolic_multiplier {
            symbolic_multiplier_sqrt_affine_derivative_display_and_latex(
                ctx,
                &arg_match.scale,
                arg_match.radicand,
                parameter,
                var_name,
            )?
        } else {
            scaled_sqrt_affine_derivative_display_and_latex(
                ctx,
                &arg_match.scale,
                arg_match.radicand,
                var_name,
            )?
        };
    Some(ArctanSqrtVarTableMatch {
        arg,
        derivative_display,
        derivative_latex,
    })
}

fn arctan_sqrt_var_arg_match(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> Option<ArctanSqrtArgMatch> {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    if let Some(radicand) = sqrt_affine_radicand(ctx, expr, var_name) {
        return Some(ArctanSqrtArgMatch {
            scale: BigRational::one(),
            radicand,
            symbolic_denominator: None,
            symbolic_multiplier: None,
        });
    }
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            if let Some(coeff) = as_rational_const(ctx, *left, 8) {
                return arctan_sqrt_var_arg_match(ctx, *right, var_name).map(|arg_match| {
                    ArctanSqrtArgMatch {
                        scale: coeff * arg_match.scale,
                        radicand: arg_match.radicand,
                        symbolic_denominator: arg_match.symbolic_denominator,
                        symbolic_multiplier: arg_match.symbolic_multiplier,
                    }
                });
            }
            if let Some(coeff) = as_rational_const(ctx, *right, 8) {
                return arctan_sqrt_var_arg_match(ctx, *left, var_name).map(|arg_match| {
                    ArctanSqrtArgMatch {
                        scale: coeff * arg_match.scale,
                        radicand: arg_match.radicand,
                        symbolic_denominator: arg_match.symbolic_denominator,
                        symbolic_multiplier: arg_match.symbolic_multiplier,
                    }
                });
            }
            if !contains_named_var(ctx, *left, var_name) {
                return arctan_sqrt_var_arg_match(ctx, *right, var_name).and_then(|arg_match| {
                    if arg_match.symbolic_denominator.is_some()
                        || arg_match.symbolic_multiplier.is_some()
                    {
                        return None;
                    }
                    Some(ArctanSqrtArgMatch {
                        scale: arg_match.scale,
                        radicand: arg_match.radicand,
                        symbolic_denominator: None,
                        symbolic_multiplier: Some(*left),
                    })
                });
            }
            if !contains_named_var(ctx, *right, var_name) {
                return arctan_sqrt_var_arg_match(ctx, *left, var_name).and_then(|arg_match| {
                    if arg_match.symbolic_denominator.is_some()
                        || arg_match.symbolic_multiplier.is_some()
                    {
                        return None;
                    }
                    Some(ArctanSqrtArgMatch {
                        scale: arg_match.scale,
                        radicand: arg_match.radicand,
                        symbolic_denominator: None,
                        symbolic_multiplier: Some(*right),
                    })
                });
            }
            None
        }
        Expr::Div(num, den) => {
            if let Some(coeff) = as_rational_const(ctx, *den, 8) {
                if coeff.is_zero() {
                    return None;
                }
                return arctan_sqrt_var_arg_match(ctx, *num, var_name).map(|arg_match| {
                    ArctanSqrtArgMatch {
                        scale: arg_match.scale / coeff,
                        radicand: arg_match.radicand,
                        symbolic_denominator: arg_match.symbolic_denominator,
                        symbolic_multiplier: arg_match.symbolic_multiplier,
                    }
                });
            }
            if contains_named_var(ctx, *den, var_name) {
                return None;
            }
            arctan_sqrt_var_arg_match(ctx, *num, var_name).and_then(|arg_match| {
                if arg_match.symbolic_denominator.is_some() {
                    return None;
                }
                Some(ArctanSqrtArgMatch {
                    scale: arg_match.scale,
                    radicand: arg_match.radicand,
                    symbolic_denominator: Some(*den),
                    symbolic_multiplier: None,
                })
            })
        }
        Expr::Neg(inner) | Expr::Hold(inner) => arctan_sqrt_var_arg_match(ctx, *inner, var_name)
            .map(|arg_match| ArctanSqrtArgMatch {
                scale: -arg_match.scale,
                radicand: arg_match.radicand,
                symbolic_denominator: arg_match.symbolic_denominator,
                symbolic_multiplier: arg_match.symbolic_multiplier,
            }),
        _ => None,
    }
}

fn sqrt_affine_radicand(ctx: &Context, expr: ExprId, var_name: &str) -> Option<ExprId> {
    let (sqrt_builtin, radicand) = unary_builtin_arg(ctx, expr)?;
    if sqrt_builtin != BuiltinFn::Sqrt {
        return None;
    }
    let poly = Polynomial::from_expr(ctx, radicand, var_name).ok()?;
    (poly.degree() == 1).then_some(radicand)
}

pub(super) fn sqrt_chain_symbolic_scale_trace(
    scratch: &mut Context,
    input: SqrtChainSymbolicScaleTraceInput<'_>,
) -> Option<(String, String)> {
    let SqrtChainSymbolicScaleTraceInput {
        numerator_negative,
        numerator_factors,
        denominator_factors,
        radicand,
        radicand_derivative,
        derivative_simplified,
        var_name,
    } = input;

    let mut symbolic_scale_factors = Vec::new();
    let mut derivative_numerator_factors = Vec::new();
    for factor in numerator_factors {
        if !contains_named_var(scratch, *factor, var_name)
            && as_rational_const(scratch, *factor, 8).is_none()
        {
            symbolic_scale_factors.push(*factor);
        } else {
            derivative_numerator_factors.push(*factor);
        }
    }
    if symbolic_scale_factors.is_empty() {
        return None;
    }

    let mut signed_derivative_numerator_factors = Vec::new();
    if numerator_negative {
        signed_derivative_numerator_factors
            .push(scratch.add(Expr::Number(BigRational::from_integer((-1).into()))));
    }
    signed_derivative_numerator_factors.extend(derivative_numerator_factors);
    let rational_scale = sqrt_chain_rational_cofactor_scale(
        scratch,
        &signed_derivative_numerator_factors,
        denominator_factors,
        radicand,
        radicand_derivative,
        var_name,
    )
    .or_else(|| {
        let derivative_cofactor_expr = build_quotient_from_factors(
            scratch,
            &signed_derivative_numerator_factors,
            denominator_factors,
        );
        let derivative_cofactor_simplified =
            simplify_expr_in_context(scratch, derivative_cofactor_expr);
        let ratio = scratch.add(Expr::Div(
            derivative_cofactor_simplified,
            derivative_simplified,
        ));
        let ratio_simplified = simplify_expr_in_context(scratch, ratio);
        as_rational_const(scratch, ratio_simplified, 8)
    })?;
    if rational_scale.is_zero() {
        return None;
    }

    let symbolic_scale = build_mul_expr_from_factors(scratch, &symbolic_scale_factors);
    let scaled_symbolic_scale = if rational_scale.is_one() {
        symbolic_scale
    } else {
        let rational_expr = scratch.add(Expr::Number(rational_scale));
        scratch.add(Expr::Mul(rational_expr, symbolic_scale))
    };
    let scaled_symbolic_scale = simplify_expr_in_context(scratch, scaled_symbolic_scale);
    if contains_named_var(scratch, scaled_symbolic_scale, var_name) {
        return None;
    }

    Some((
        display_expr(scratch, scaled_symbolic_scale),
        latex_expr(scratch, scaled_symbolic_scale),
    ))
}

fn sqrt_chain_rational_cofactor_scale(
    scratch: &mut Context,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
    radicand: ExprId,
    radicand_derivative: &Polynomial,
    var_name: &str,
) -> Option<BigRational> {
    let two = BigRational::from_integer(2.into());
    let half_derivative = radicand_derivative.div_scalar(&two);

    for (idx, factor) in denominator_factors.iter().enumerate() {
        let Some(factor_radicand) = sqrt_chain_arg_radicand(scratch, *factor) else {
            continue;
        };
        if compare_expr(scratch, factor_radicand, radicand) != Ordering::Equal {
            continue;
        }

        let remaining_denominator = denominator_factors
            .iter()
            .enumerate()
            .filter_map(|(factor_idx, factor)| (factor_idx != idx).then_some(*factor))
            .collect::<Vec<_>>();
        return quotient_scale_against_polynomial_trace(
            scratch,
            numerator_factors,
            &remaining_denominator,
            &half_derivative,
            var_name,
        );
    }

    None
}

pub(super) fn sqrt_chain_arg_radicand(ctx: &Context, arg: ExprId) -> Option<ExprId> {
    if let Some((BuiltinFn::Sqrt, radicand)) = unary_builtin_arg(ctx, arg) {
        return Some(radicand);
    }

    let (base, exponent) = as_pow(ctx, arg)?;
    let exponent = as_rational_const(ctx, exponent, 8)?;
    (exponent == BigRational::new(1.into(), 2.into())).then_some(base)
}

fn signed_sqrt_chain_arg_radicand(ctx: &Context, arg: ExprId) -> Option<(ExprId, BigRational)> {
    match ctx.get(arg) {
        Expr::Neg(inner) => {
            let radicand = sqrt_chain_arg_radicand(ctx, *inner)?;
            Some((radicand, -BigRational::one()))
        }
        _ => sqrt_chain_arg_radicand(ctx, arg).map(|radicand| (radicand, BigRational::one())),
    }
}

pub(super) fn sqrt_chain_arg_radicand_and_sign(
    ctx: &Context,
    arg: ExprId,
    var_name: &str,
) -> Option<(ExprId, BigRational)> {
    if let Some(radicand) = sqrt_chain_arg_radicand(ctx, arg) {
        return Some((radicand, BigRational::one()));
    }

    match ctx.get(arg) {
        Expr::Add(left, right) => {
            if !contains_named_var(ctx, *left, var_name) {
                return signed_sqrt_chain_arg_radicand(ctx, *right);
            }
            if !contains_named_var(ctx, *right, var_name) {
                return signed_sqrt_chain_arg_radicand(ctx, *left);
            }
            None
        }
        Expr::Sub(left, right) => {
            if !contains_named_var(ctx, *left, var_name) {
                let (radicand, sign) = signed_sqrt_chain_arg_radicand(ctx, *right)?;
                return Some((radicand, -sign));
            }
            if !contains_named_var(ctx, *right, var_name) {
                return signed_sqrt_chain_arg_radicand(ctx, *left);
            }
            None
        }
        _ => None,
    }
}

pub(super) fn same_sqrt_chain_arg(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    if compare_expr(ctx, left, right) == Ordering::Equal {
        return true;
    }
    let Some(left_radicand) = sqrt_chain_arg_radicand(ctx, left) else {
        return false;
    };
    let Some(right_radicand) = sqrt_chain_arg_radicand(ctx, right) else {
        return false;
    };
    compare_expr(ctx, left_radicand, right_radicand) == Ordering::Equal
}

/// Extract the radicand `P` of a square root written either as the `sqrt(P)`
/// builtin or as `P^(1/2)`. Returns `None` for any other shape.
pub(super) fn as_sqrt_radicand(ctx: &Context, e: ExprId) -> Option<ExprId> {
    if let Expr::Function(fn_id, args) = ctx.get(e) {
        if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) {
            return Some(args[0]);
        }
    }
    let (base, exp) = as_pow(ctx, e)?;
    match ctx.get(exp) {
        Expr::Number(n) if *n.numer() == 1.into() && *n.denom() == 2.into() => Some(base),
        _ => None,
    }
}

pub(super) fn cube_root_power_term(ctx: &Context, expr: ExprId) -> Option<(ExprId, i64)> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    let exponent = positive_integer_literal_value(ctx, *exponent)?.to_i64()?;
    if exponent <= 3 || exponent % 3 != 0 {
        return None;
    }
    let cube_root_exponent = exponent / 3;
    if cube_root_exponent <= 2 {
        return None;
    }
    Some((*base, cube_root_exponent))
}

pub(super) fn sqrt_radicand(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if *fn_id == ctx.builtin_id(BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            Some(args[0])
        }
        Expr::Pow(base, exponent) if is_one_half(ctx, *exponent) => Some(*base),
        _ => None,
    }
}
