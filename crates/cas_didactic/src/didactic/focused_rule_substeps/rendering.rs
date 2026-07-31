//! `focused_rule_substeps`: familia `rendering`.
//!
//! Ver la cabecera de `focused_rule_substeps.rs` para el contexto.

use super::*;

pub(super) fn grouped_substitution_display(ctx: &Context, expr: ExprId) -> String {
    let display = human_expr(ctx, expr);
    if needs_grouped_substitution_expr(ctx.get(expr)) {
        format!("({display})")
    } else {
        display
    }
}

pub(super) fn grouped_substitution_latex(ctx: &Context, expr: ExprId) -> String {
    let latex = latex_expr(ctx, expr);
    if needs_grouped_substitution_expr(ctx.get(expr)) {
        format!("\\left({latex}\\right)")
    } else {
        latex
    }
}

pub(super) fn finite_sum_closed_form_title(description: &str) -> Option<&'static str> {
    if description.starts_with("Sum of first integers:") {
        Some("Usar la fórmula cerrada para la suma de enteros")
    } else if description.starts_with("Sum of squares:") {
        Some("Usar la fórmula cerrada para la suma de cuadrados")
    } else if description.starts_with("Sum of cubes:") {
        Some("Usar la fórmula cerrada para la suma de cubos")
    } else if description.starts_with("Sum of constant term:") {
        Some("Contar términos iguales en la suma")
    } else if description.starts_with("Geometric sum:") {
        Some("Usar la fórmula cerrada para la suma geométrica")
    } else {
        None
    }
}

pub(super) fn finite_product_closed_form_title(description: &str) -> Option<&'static str> {
    if description.starts_with("Product of first integers:") {
        Some("Usar factorial para el producto de enteros consecutivos")
    } else if description.starts_with("Product of powers:") {
        Some("Convertir el producto de potencias en potencia de factoriales")
    } else if description.starts_with("Product of constant factor:") {
        Some("Contar factores iguales en el producto")
    } else {
        None
    }
}

pub(super) fn render_finite_aggregate_endpoint_series(
    ctx: &Context,
    call: &FiniteAggregateCall,
    separator_plain: &str,
    separator_latex: &str,
) -> (String, String) {
    let mut temp_ctx = ctx.clone();
    let first = substitute_expr_by_id(&mut temp_ctx, call.term, call.var_expr, call.start_expr);
    let second_index = finite_aggregate_successor_index(ctx, &mut temp_ctx, call.start_expr);
    let second = substitute_expr_by_id(&mut temp_ctx, call.term, call.var_expr, second_index);
    let last = substitute_expr_by_id(&mut temp_ctx, call.term, call.var_expr, call.end_expr);

    let (first_plain, first_latex) = render_temp_expr(&temp_ctx, first);
    let (second_plain, second_latex) = render_temp_expr(&temp_ctx, second);
    let (last_plain, last_latex) = render_temp_expr(&temp_ctx, last);

    (
        format!(
            "{first_plain}{separator_plain}{second_plain}{separator_plain}…{separator_plain}{last_plain}"
        ),
        format!(
            "{first_latex}{separator_latex}{second_latex}{separator_latex}\\cdots{separator_latex}{last_latex}"
        ),
    )
}

pub(super) fn binom_latex(n: i64, k: i64) -> String {
    format!("\\binom{{{n}}}{{{k}}}")
}

pub(super) fn identity_title_with_optional_u(base_title: &str, u_plain: &str) -> String {
    if u_plain == "u" {
        base_title.to_string()
    } else {
        format!("{base_title} con u = {u_plain}")
    }
}

pub(super) fn dirichlet_kernel_identity_title(base_title: &str, n: usize, u_plain: &str) -> String {
    if u_plain == "u" {
        format!("{base_title} con n = {n}")
    } else {
        format!("{base_title} con n = {n} y u = {u_plain}")
    }
}

pub(super) fn infer_sum_to_product_kind_from_display(ctx: &Context, expr: ExprId) -> Option<&str> {
    let before = cas_formatter::clean_display_string(&format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: expr
        }
    ));
    let sin_count = before.matches("sin(").count();
    let cos_count = before.matches("cos(").count();
    let is_difference = before.contains(" - ");

    match (sin_count >= 2, cos_count >= 2, is_difference) {
        (true, false, false) => Some("sine sum"),
        (true, false, true) => Some("sine difference"),
        (false, true, false) => Some("cosine sum"),
        (false, true, true) => Some("cosine difference"),
        _ => None,
    }
}

pub(super) fn render_numeric_sum(coeffs: &[BigRational]) -> (String, String) {
    let mut temp_ctx = Context::new();
    let expr = build_numeric_sum_expr(&mut temp_ctx, coeffs);
    render_temp_expr(&temp_ctx, expr)
}

pub(super) fn render_numeric_value(value: &BigRational) -> (String, String) {
    let mut temp_ctx = Context::new();
    let expr = temp_ctx.add(Expr::Number(value.clone()));
    render_temp_expr(&temp_ctx, expr)
}

pub(super) fn render_square_difference_plain(base: &str) -> String {
    format!("({base} - 1) · ({base} + 1)")
}

pub(super) fn render_square_difference_latex(base: &str) -> String {
    format!("\\left({base} - 1\\right)\\cdot \\left({base} + 1\\right)")
}

pub(super) fn render_contribution_sum(var: &str, terms: &[PolyContribution]) -> (String, String) {
    let mut temp_ctx = Context::new();
    let expr = build_sum_expr_from_contributions(&mut temp_ctx, var, terms);
    (
        cas_formatter::clean_display_string(&format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &temp_ctx,
                id: expr,
            }
        )),
        cas_formatter::LaTeXExpr {
            context: &temp_ctx,
            id: expr,
        }
        .to_latex(),
    )
}

pub(super) fn render_grouped_contributions(
    var: &str,
    grouped: &BTreeMap<usize, Vec<PolyContribution>>,
) -> (String, String) {
    #[derive(Debug)]
    struct GroupRender {
        display: String,
        latex: String,
        negative: bool,
    }

    let mut rendered = Vec::new();

    for contributions in grouped.values().rev() {
        if contributions.is_empty() {
            continue;
        }

        if contributions.len() == 1 {
            let term = &contributions[0];
            let (display, latex) = render_contribution_sum(
                var,
                &[PolyContribution {
                    coeff: term.coeff.abs(),
                    degree: term.degree,
                }],
            );
            rendered.push(GroupRender {
                display,
                latex,
                negative: term.coeff.is_negative(),
            });
            continue;
        }

        let mut positives = Vec::new();
        let mut negatives = Vec::new();
        for term in contributions {
            if term.coeff.is_negative() {
                negatives.push(term.clone());
            } else {
                positives.push(term.clone());
            }
        }

        let (display, latex, negative) = if positives.is_empty() {
            let abs_terms = negatives
                .into_iter()
                .map(|term| PolyContribution {
                    coeff: term.coeff.abs(),
                    degree: term.degree,
                })
                .collect::<Vec<_>>();
            let (display, latex) = render_contribution_sum(var, &abs_terms);
            (display, latex, true)
        } else {
            let mut ordered = positives;
            ordered.extend(negatives);
            let (display, latex) = render_contribution_sum(var, &ordered);
            (display, latex, false)
        };

        rendered.push(GroupRender {
            display: format!("({display})"),
            latex: format!("\\left({latex}\\right)"),
            negative,
        });
    }

    if rendered.is_empty() {
        return ("0".to_string(), "0".to_string());
    }

    let mut display = String::new();
    let mut latex = String::new();

    for (index, group) in rendered.into_iter().enumerate() {
        if index == 0 {
            if group.negative {
                display.push('-');
                latex.push('-');
            }
            display.push_str(&group.display);
            latex.push_str(&group.latex);
            continue;
        }

        if group.negative {
            display.push_str(" - ");
            latex.push_str(" - ");
        } else {
            display.push_str(" + ");
            latex.push_str(" + ");
        }
        display.push_str(&group.display);
        latex.push_str(&group.latex);
    }

    (display, latex)
}

pub(super) fn inverse_function_empty_domain_diff_substep_title(
    ctx: &Context,
    expr: ExprId,
) -> Option<&'static str> {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Function(fn_id, args) => {
                let builtin = ctx.builtin_of(*fn_id);
                if matches!(
                    builtin,
                    Some(
                        BuiltinFn::Asin
                            | BuiltinFn::Acos
                            | BuiltinFn::Asec
                            | BuiltinFn::Acsc
                            | BuiltinFn::Arcsin
                            | BuiltinFn::Arccos
                            | BuiltinFn::Arcsec
                            | BuiltinFn::Arccsc
                            | BuiltinFn::Acosh
                            | BuiltinFn::Atanh
                    )
                ) {
                    let mut scratch = ctx.clone();
                    return match cas_math::calculus_domain_support::bounded_inverse_real_domain_rejection_over_reals(
                        &mut scratch,
                        builtin,
                        args,
                        8,
                    ) {
                        Some(
                            cas_math::calculus_domain_support::BoundedInverseRealDomainRejection::SourceDomainEmpty,
                        ) => Some("Detectar dominio real vacío de la función inversa"),
                        Some(
                            cas_math::calculus_domain_support::BoundedInverseRealDomainRejection::DerivativeDomainEmpty,
                        ) => Some(
                            "Detectar dominio real vacío de la derivada de la función inversa",
                        ),
                        None => Some("Detectar dominio real vacío de la función inversa"),
                    };
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
            Expr::Number(_)
            | Expr::Constant(_)
            | Expr::Variable(_)
            | Expr::SessionRef(_)
            | Expr::Matrix { .. } => {}
        }
    }
    None
}

pub(super) fn differentiation_rule_title(
    ctx: &Context,
    target: ExprId,
    var_name: &str,
) -> Option<&'static str> {
    match ctx.get(target) {
        Expr::Add(_, _) | Expr::Sub(_, _) => Some("derivative.linearity"),
        Expr::Neg(inner) if contains_named_var(ctx, *inner, var_name) => {
            Some("derivative.linearity")
        }
        Expr::Mul(_, _)
            if differentiation_constant_multiple_inner(ctx, target, var_name).is_some() =>
        {
            Some("derivative.constant_multiple")
        }
        Expr::Mul(_, _) => Some("derivative.product_rule"),
        Expr::Div(_, _) => Some("derivative.quotient_rule"),
        Expr::Pow(base, exponent) => {
            let base_depends = contains_named_var(ctx, *base, var_name);
            let exponent_depends = contains_named_var(ctx, *exponent, var_name);
            match (base_depends, exponent_depends) {
                (true, true) => Some("derivative.use_logarithmic_diff"),
                (true, false) if !is_named_var(ctx, *base, var_name) => {
                    Some("derivative.power_rule_with_chain")
                }
                (true, false) => Some("derivative.power_rule"),
                (false, true) => Some("derivative.exponential_rule"),
                _ => None,
            }
        }
        Expr::Function(fn_id, args)
            if args.len() == 1 && contains_named_var(ctx, args[0], var_name) =>
        {
            match ctx.builtin_of(*fn_id)? {
                BuiltinFn::Sin => Some("derivative.rule_sin_u"),
                BuiltinFn::Cos => Some("derivative.rule_cos_u"),
                BuiltinFn::Tan => Some("derivative.rule_tan_u"),
                BuiltinFn::Ln => Some("derivative.rule_ln_u"),
                BuiltinFn::Exp => Some("derivative.rule_exp_u"),
                BuiltinFn::Sqrt => Some("derivative.rule_sqrt_u"),
                BuiltinFn::Arctan | BuiltinFn::Atan => Some("derivative.rule_arctan_u"),
                BuiltinFn::Arcsin | BuiltinFn::Asin => Some("derivative.rule_arcsin_u"),
                BuiltinFn::Arccos | BuiltinFn::Acos => Some("derivative.rule_arccos_u"),
                BuiltinFn::Sec => Some("derivative.rule_sec_u"),
                BuiltinFn::Csc => Some("derivative.rule_csc_u"),
                BuiltinFn::Cot => Some("derivative.rule_cot_u"),
                BuiltinFn::Sinh => Some("derivative.rule_sinh_u"),
                BuiltinFn::Cosh => Some("derivative.rule_cosh_u"),
                BuiltinFn::Tanh => Some("derivative.rule_tanh_u"),
                BuiltinFn::Sign => Some("derivative.rule_sign_u_away_from_zero"),
                _ => Some("derivative.chain_rule"),
            }
        }
        _ => None,
    }
}

pub(super) fn group_display_for_product(value: &str) -> String {
    if value.contains(" + ") || value.contains(" - ") {
        format!("({value})")
    } else {
        value.to_string()
    }
}

pub(super) fn group_latex_for_product(value: &str) -> String {
    if value.contains(" + ") || value.contains(" - ") {
        format!("\\left({value}\\right)")
    } else {
        value.to_string()
    }
}

pub(super) fn inverse_table_function_display(
    ctx: &Context,
    builtin: BuiltinFn,
    arg: ExprId,
) -> String {
    format!(
        "{}({})",
        inverse_table_function_name(builtin),
        display_expr(ctx, arg)
    )
}

pub(super) fn inverse_table_function_latex(
    ctx: &Context,
    builtin: BuiltinFn,
    arg: ExprId,
) -> String {
    format!(
        "\\{}\\left({}\\right)",
        inverse_table_function_name(builtin),
        latex_expr(ctx, arg)
    )
}

pub(super) fn same_math_render(left: &str, right: &str) -> bool {
    let mut left = left.to_string();
    let mut right = right.to_string();
    left.retain(|ch| !ch.is_whitespace());
    right.retain(|ch| !ch.is_whitespace());
    left == right
}

pub(super) fn perfect_square_form_latex(
    ctx: &Context,
    numerator: ExprId,
    denominator: ExprId,
) -> Option<String> {
    let mut temp_ctx = ctx.clone();
    let (left, right, is_sub) =
        cas_math::perfect_square_support::try_match_perfect_square_trinomial(
            &mut temp_ctx,
            numerator,
        )?;
    let base = if is_sub {
        temp_ctx.add_raw(Expr::Sub(left, right))
    } else {
        temp_ctx.add_raw(Expr::Add(left, right))
    };
    if !same_presentational_expr(ctx, denominator, &temp_ctx, base) {
        return None;
    }

    let exponent = temp_ctx.num(2);
    let squared = temp_ctx.add_raw(Expr::Pow(base, exponent));
    Some(latex_expr(&temp_ctx, squared))
}

pub(super) fn cube_identity_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
) -> String {
    let op = match kind {
        CubeIdentityKind::Sum => " + ",
        CubeIdentityKind::Difference => " - ",
    };
    format!(
        "{}{}{}",
        cubed_display(ctx, left_base),
        op,
        cubed_display(ctx, right_base)
    )
}

pub(super) fn cube_identity_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
) -> String {
    let op = match kind {
        CubeIdentityKind::Sum => " + ",
        CubeIdentityKind::Difference => " - ",
    };
    format!(
        "{}{}{}",
        cubed_latex(ctx, left_base),
        op,
        cubed_latex(ctx, right_base)
    )
}

pub(super) fn sophie_germain_identity_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
) -> String {
    format!(
        "{} + 4 · {}",
        fourth_power_display(ctx, left_base),
        fourth_power_display(ctx, right_base)
    )
}

pub(super) fn sophie_germain_identity_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
) -> String {
    format!(
        "{} + 4\\cdot {}",
        fourth_power_latex(ctx, left_base),
        fourth_power_latex(ctx, right_base)
    )
}

fn cubed_display(ctx: &Context, expr: ExprId) -> String {
    let display = display_expr(ctx, expr);
    if is_simple_power_base(ctx, expr) {
        format!("{display}^3")
    } else {
        format!("({display})^3")
    }
}

fn cubed_latex(ctx: &Context, expr: ExprId) -> String {
    let latex = latex_expr(ctx, expr);
    format!("{{{latex}}}^{{3}}")
}

pub(super) fn squared_display(ctx: &Context, expr: ExprId) -> String {
    let display = display_expr(ctx, expr);
    if is_simple_power_base(ctx, expr) {
        format!("{display}^2")
    } else {
        format!("({display})^2")
    }
}

pub(super) fn squared_latex(ctx: &Context, expr: ExprId) -> String {
    let latex = latex_expr(ctx, expr);
    format!("{{{latex}}}^{{2}}")
}
