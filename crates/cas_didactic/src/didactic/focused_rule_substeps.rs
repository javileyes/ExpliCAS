use super::nested_fraction_analysis::NestedFractionPattern;
use super::SubStep;
use crate::didactic::try_as_fraction;
use crate::runtime::Step;
use cas_ast::ordering::compare_expr;
use cas_ast::views::as_rational_const;
use cas_ast::{substitute_expr_by_id, BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_destructure::as_div;
use cas_math::expr_extract::extract_i64_multiplier_and_base_factors;
use cas_math::expr_nary::build_balanced_mul;
use cas_math::expr_nary::{self, AddView, Sign};
use cas_math::poly_compare::poly_eq;
use cas_math::polynomial::Polynomial;
use cas_math::summation_support::{
    detect_factorized_telescoping_square_base, extract_linear_offset, extract_unit_shifted_base,
    try_extract_finite_aggregate_call,
};
use cas_solver_core::quadratic_coeffs::{
    extract_quadratic_coefficients, extract_simplified_nonzero_quadratic_coefficients_with_state,
};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::cmp::Ordering;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinomialSquareKind {
    Sum,
    Difference,
}

type SignedTerms = Vec<(ExprId, Sign)>;
type ConcreteLogExpansion = (ExprId, ExprId, SignedTerms);

pub(crate) fn generate_focused_rule_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let sixth_power_substeps = generate_sum_difference_sixth_powers_substeps(ctx, step);
    if !sixth_power_substeps.is_empty() {
        return sixth_power_substeps;
    }

    let cube_expansion_substeps = generate_sum_difference_cubes_expansion_substeps(ctx, step);
    if !cube_expansion_substeps.is_empty() {
        return cube_expansion_substeps;
    }

    let sixth_power_expansion_substeps =
        generate_sum_difference_sixth_powers_expansion_substeps(ctx, step);
    if !sixth_power_expansion_substeps.is_empty() {
        return sixth_power_expansion_substeps;
    }

    let phase_shift_substeps = generate_phase_shift_identity_substeps(ctx, step);
    if !phase_shift_substeps.is_empty() {
        return phase_shift_substeps;
    }

    let log_cancellation_substeps = generate_log_cancellation_substeps(ctx, step);
    if !log_cancellation_substeps.is_empty() {
        return log_cancellation_substeps;
    }

    match step.rule_name.as_str() {
        "Combine Like Terms" => generate_combine_like_terms_substeps(ctx, step),
        "Distribute Division" => generate_fraction_expansion_substeps(ctx, step),
        "Add Fractions" => generate_add_subtract_fractions_substeps(ctx, step),
        "Subtract Fractions" => generate_add_subtract_fractions_substeps(ctx, step),
        "Mixed Fraction Split" => generate_mixed_fraction_split_substeps(ctx, step),
        "Mixed Fraction Combine" => generate_mixed_fraction_combine_substeps(ctx, step),
        "Telescoping Fraction Combine" => generate_telescoping_fraction_combine_substeps(ctx, step),
        "Telescoping Fraction Split" => generate_telescoping_fraction_split_substeps(ctx, step),
        "Canonicalize Roots" => generate_canonicalize_roots_substeps(ctx, step),
        "Combine powers with same base (n-ary)" => {
            generate_same_base_power_merge_substeps(ctx, step)
        }
        "Expand Odd Half Power" => generate_odd_half_power_substeps(ctx, step),
        "Expand" => generate_expand_substeps(ctx, step),
        "Collect Terms" => generate_collect_terms_substeps(ctx, step),
        "Factor Out With Division" => generate_factor_out_with_division_substeps(ctx, step),
        "Factorization" => generate_factorization_substeps(ctx, step),
        "Binomial Expansion" | "Auto Expand Power Sum" => {
            generate_binomial_expansion_substeps(ctx, step)
        }
        "expand_log" => generate_expand_log_substeps(ctx, step),
        "Simplify" | "Canonicalize" => generate_simplify_substeps(ctx, step),
        "Evaluate Logarithms" => generate_evaluate_logarithms_substeps(ctx, step),
        "Factor Perfect Square in Logarithm" => {
            generate_factor_perfect_square_log_substeps(ctx, step)
        }
        "Log Contraction" => generate_log_contraction_substeps(ctx, step),
        "Finite Product" => generate_finite_product_substeps(ctx, step),
        "Finite Summation" => generate_finite_summation_substeps(ctx, step),
        "Cos Product Telescoping" => generate_cos_product_telescoping_substeps(ctx, step),
        "Dirichlet Kernel Identity" => generate_dirichlet_kernel_substeps(ctx, step),
        "Complete the Square" => generate_complete_square_substeps(ctx, step),
        "Product-to-Sum Identity" => generate_product_to_sum_substeps(step),
        "Sum-to-Product Identity" | "Sum-to-Product Identity Cancellation Bridge" => {
            generate_sum_to_product_substeps(ctx, step)
        }
        "Hyperbolic Angle Sum/Difference Identity" => {
            generate_hyperbolic_angle_sum_diff_substeps(ctx, step)
        }
        "Double Angle Expansion" => generate_double_angle_expansion_substeps(ctx, step),
        "Double Angle Contraction" => generate_double_angle_contraction_substeps(ctx, step),
        "Half-Angle Square Identity" | "Angle Consistency (Half-Angle)" => {
            generate_half_angle_square_identity_substeps(ctx, step)
        }
        "Expand Secant Squared" | "Expand Cosecant Squared" => {
            generate_sec_csc_squared_expansion_substeps(ctx, step)
        }
        "Recognize Secant Squared" | "Recognize Cosecant Squared" => {
            generate_sec_csc_squared_contraction_substeps(ctx, step)
        }
        "Reciprocal Product Identity" => generate_reciprocal_product_identity_substeps(ctx, step),
        "Reciprocal Pythagorean Identity" => generate_reciprocal_pythagorean_substeps(ctx, step),
        "Cos 2x Additive Contraction" => generate_cos_2x_additive_contraction_substeps(ctx, step),
        "Triple Angle Identity" => generate_triple_angle_identity_substeps(ctx, step),
        "Half-Angle Tangent Identity" => generate_half_angle_tangent_substeps(ctx, step),
        "Reciprocal Trig Identity" => generate_reciprocal_trig_identity_substeps(ctx, step),
        "Trig Expansion" => generate_trig_expansion_substeps(ctx, step),
        "Trig Quotient" => generate_trig_quotient_substeps(ctx, step),
        "Cos-Diff / Sin-Diff Quotient" => generate_cos_diff_sin_diff_quotient_substeps(ctx, step),
        "Distributive Property"
        | "Distributive Property (Simple)"
        | "Pull Constant From Fraction" => {
            generate_reverse_nested_fraction_rule_substeps(ctx, step)
        }
        "Pythagorean Factor Form" => generate_pythagorean_factor_form_substeps(ctx, step),
        "Pythagorean High-Power Factor" => {
            generate_pythagorean_high_power_factor_substeps(ctx, step)
        }
        "Pythagorean Chain Identity" => generate_pythagorean_chain_identity_substeps(ctx, step),
        "Consecutive Factorial Ratio" => generate_consecutive_factorial_ratio_substeps(ctx, step),
        "Simplify Nested Fraction" => generate_simplify_nested_fraction_substeps(ctx, step),
        "Pre-order Perfect Square Minus Cancel" => {
            generate_perfect_square_fraction_cancel_substeps(ctx, step)
        }
        "Pre-order Common Factor Cancel" => generate_common_factor_cancel_substeps(ctx, step),
        "Pre-order Difference of Squares Cancel" => {
            generate_difference_of_squares_cancel_substeps(ctx, step)
        }
        "Canonicalize Nested Power" => generate_canonicalize_nested_power_substeps(ctx, step),
        "Identity Property of Addition" => generate_identity_addition_substeps(ctx, step),
        "Identity Property of Multiplication" => {
            generate_identity_multiplication_substeps(ctx, step)
        }
        "Evaluate Numeric Power" => generate_evaluate_numeric_power_substeps(ctx, step),
        "Pre-order Sum/Difference of Cubes" => generate_sum_difference_cubes_substeps(ctx, step),
        "Pre-order Sum/Difference of Cubes Cancel" => {
            generate_sum_difference_cubes_cancel_substeps(ctx, step)
        }
        "Cancel Sum/Difference of Cubes Fraction" => {
            generate_sum_difference_cubes_cancel_substeps(ctx, step)
        }
        "Inverse Tan Relations" => generate_inverse_tan_relation_substeps(ctx, step),
        "Subtraction Self-Cancel" => generate_subtraction_self_cancel_substeps(ctx, step),
        "Cancel Reciprocal Exponents" => generate_cancel_reciprocal_exponents_substeps(ctx, step),
        "Polynomial Identity" => generate_polynomial_identity_exact_cancel_substeps(ctx, step),
        "Subtract Expanded Sum/Difference of Cubes Quotient" => {
            generate_subtract_expanded_cubes_quotient_substeps(ctx, step)
        }
        "Polynomial Product Normalize" => generate_polynomial_product_normalize_substeps(ctx, step),
        "Sqrt Perfect Square" | "Simplify Square Root" => {
            generate_sqrt_perfect_square_substeps(ctx, step)
        }
        _ => Vec::new(),
    }
}

fn generate_phase_shift_identity_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let is_phase_shift = super::visible_rule_names::visible_rule_name_for_step(
        step.rule_name.as_str(),
        step.description.as_str(),
    )
    .as_ref()
        == "Aplicar identidad de desfase";
    if !is_phase_shift {
        return Vec::new();
    }

    let local_before = step.before_local().unwrap_or(step.before);
    let local_after = step.after_local().unwrap_or(step.after);
    let first_title = if matches!(ctx.get(local_before), Expr::Add(_, _) | Expr::Sub(_, _))
        && !matches!(ctx.get(local_after), Expr::Add(_, _) | Expr::Sub(_, _))
    {
        "Usar a·sin(u) + b·cos(u) = R·sin(u + φ)"
    } else if !matches!(ctx.get(local_before), Expr::Add(_, _) | Expr::Sub(_, _))
        && matches!(ctx.get(local_after), Expr::Add(_, _) | Expr::Sub(_, _))
    {
        "Expandir R·sin(u + φ)"
    } else {
        "Usar una identidad de desfase"
    };
    let mut substeps = if local_before != local_after {
        vec![concrete_expr_substep(
            ctx,
            first_title,
            local_before,
            local_after,
        )]
    } else {
        Vec::new()
    };

    let Some(global_before) = step.global_before else {
        return substeps;
    };
    let Some(global_after) = step.global_after else {
        return substeps;
    };

    let mut work = ctx.clone();
    let intermediate = substitute_expr_by_id(&mut work, global_before, local_before, local_after);
    if intermediate == global_after {
        return substeps;
    }

    let (intermediate_plain, intermediate_latex) = render_temp_expr(&work, intermediate);
    let (global_after_plain, global_after_latex) = render_temp_expr(&work, global_after);
    substeps.push(formula_substep(
        "Cancelar términos iguales",
        &intermediate_plain,
        &global_after_plain,
        &intermediate_latex,
        &global_after_latex,
    ));

    substeps
}

fn generate_combine_like_terms_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if step.description.contains("Cancel opposite terms") {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let Some((coeffs, literal_display)) = combine_like_terms_coeff_sum_plan(ctx, before) else {
        return Vec::new();
    };

    let (before_display, before_latex) = render_numeric_sum(&coeffs);
    let total = coeffs
        .iter()
        .fold(BigRational::from_integer(0.into()), |acc, coeff| {
            acc + coeff.clone()
        });
    let (after_display, after_latex) = render_numeric_value(&total);

    vec![SubStep::new(
        format!("Sumar los coeficientes que acompañan a {literal_display}"),
        before_display,
        after_display,
    )
    .with_before_latex(before_latex)
    .with_after_latex(after_latex)]
}

fn generate_collect_terms_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let Some(focus) = step.description.strip_prefix("Collect terms by ") else {
        return Vec::new();
    };
    let display_focus = human_collect_focus(focus);
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    if before == after {
        return Vec::new();
    }

    if is_simple_collect_focus(focus) {
        return vec![concrete_expr_substep(
            ctx,
            format!("Agrupar los términos que llevan la misma potencia de {display_focus}"),
            before,
            after,
        )];
    }

    vec![concrete_expr_substep(
        ctx,
        format!("Agrupar los términos que llevan el mismo factor {display_focus}"),
        before,
        after,
    )]
}

fn is_simple_collect_focus(focus: &str) -> bool {
    !focus.is_empty()
        && focus
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn human_collect_focus(focus: &str) -> String {
    focus.replace(" * ", "·").replace('*', "·")
}

fn generate_factor_out_with_division_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    if before == after {
        return Vec::new();
    }

    let factor_expr = detect_factor_out_with_division_substep_factor(
        ctx,
        step.after_local().unwrap_or(step.after),
    );
    let Some(factor_display) = factor_expr.map(|expr| human_expr(ctx, expr)).or_else(|| {
        step.description
            .strip_prefix("Factor out ")
            .and_then(|tail| tail.strip_suffix(" from the whole expression"))
            .map(str::to_string)
    }) else {
        return Vec::new();
    };
    vec![concrete_expr_substep(
        ctx,
        format!("Reescribir los términos que no llevan {factor_display} usando el factor común"),
        before,
        after,
    )]
}

fn detect_factor_out_with_division_substep_factor(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if !ctx.is_mul_commutative(expr) {
        return None;
    }

    let factors = collect_mul_chain_factors_readonly(ctx, expr);
    for (idx, factor) in factors.iter().copied().enumerate() {
        let mut remaining = factors.clone();
        remaining.remove(idx);
        if remaining
            .into_iter()
            .any(|inner_expr| contains_division_by_exact_factor(ctx, inner_expr, factor))
        {
            return Some(factor);
        }
    }

    None
}

fn collect_mul_chain_factors_readonly(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut out = Vec::new();
    collect_mul_chain_factors_readonly_into(ctx, expr, &mut out);
    out
}

fn collect_mul_chain_factors_readonly_into(ctx: &Context, expr: ExprId, out: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            collect_mul_chain_factors_readonly_into(ctx, *left, out);
            collect_mul_chain_factors_readonly_into(ctx, *right, out);
        }
        _ => out.push(expr),
    }
}

fn contains_division_by_exact_factor(ctx: &Context, expr: ExprId, factor: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Div(_, den) => compare_expr(ctx, *den, factor) == std::cmp::Ordering::Equal,
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            contains_division_by_exact_factor(ctx, *left, factor)
                || contains_division_by_exact_factor(ctx, *right, factor)
        }
        Expr::Pow(base, exp) => {
            contains_division_by_exact_factor(ctx, *base, factor)
                || contains_division_by_exact_factor(ctx, *exp, factor)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            contains_division_by_exact_factor(ctx, *inner, factor)
        }
        Expr::Function(_, args) => args
            .iter()
            .any(|arg| contains_division_by_exact_factor(ctx, *arg, factor)),
        Expr::Matrix { data, .. } => data
            .iter()
            .any(|item| contains_division_by_exact_factor(ctx, *item, factor)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}

fn generate_complete_square_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some((var_name, leading_coeff, linear_coeff, constant_term)) =
        complete_square_substep_plan(ctx, before)
    else {
        return Vec::new();
    };

    let _ = (&var_name, leading_coeff, linear_coeff, constant_term);
    vec![formula_substep(
        "Usar la fórmula de completar el cuadrado",
        &human_expr(ctx, before),
        &human_expr(ctx, after),
        &latex_expr(ctx, before),
        &latex_expr(ctx, after),
    )]
}

fn complete_square_substep_plan(
    ctx: &Context,
    expr: ExprId,
) -> Option<(String, ExprId, ExprId, ExprId)> {
    let mut work = ctx.clone();
    let mut vars: Vec<_> = cas_ast::collect_variables(&work, expr)
        .into_iter()
        .collect();
    vars.sort();

    for var_name in vars {
        let Some((leading_coeff, linear_coeff, constant_term)) =
            extract_simplified_nonzero_quadratic_coefficients_with_state(
                &mut work,
                expr,
                &var_name,
                extract_quadratic_coefficients,
                simplify_expr_in_context,
                expr_is_zero_in_context,
            )
        else {
            continue;
        };

        if expr_is_zero_in_context(&mut work, linear_coeff) {
            continue;
        }

        return Some((var_name, leading_coeff, linear_coeff, constant_term));
    }

    None
}

fn simplify_expr_in_context(ctx: &mut Context, expr: ExprId) -> ExprId {
    let mut simplifier = cas_solver::runtime::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (rewritten, _steps, _stats) =
        simplifier.simplify_with_stats(expr, cas_solver::runtime::SimplifyOptions::default());
    std::mem::swap(&mut simplifier.context, ctx);
    rewritten
}

fn expr_is_zero_in_context(ctx: &mut Context, expr: ExprId) -> bool {
    let simplified = simplify_expr_in_context(ctx, expr);
    matches!(ctx.get(simplified), Expr::Number(n) if n.is_zero())
}

fn generate_fraction_expansion_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let local_after = step.after_local().unwrap_or(step.after);
    let mut out = Vec::new();
    if before != local_after {
        let first_title = match ctx.get(before) {
            Expr::Div(numerator, _denominator) => {
                let numerator_terms = AddView::from_expr(ctx, *numerator);
                if numerator_terms.terms.len() >= 3 {
                    "Repartir el mismo denominador sobre cada término del numerador"
                } else {
                    "Repartir el denominador entre los términos del numerador"
                }
            }
            _ => "Repartir el denominador entre los términos del numerador",
        };
        out.push(concrete_expr_substep(ctx, first_title, before, local_after));
    }

    if step.before_local().is_none() {
        if let Some(intermediate) = step.after_local() {
            if intermediate != step.after {
                out.push(
                    SubStep::new(
                        fraction_expansion_cleanup_title(ctx, intermediate),
                        human_expr(ctx, intermediate),
                        human_expr(ctx, step.after),
                    )
                    .with_before_latex(latex_expr(ctx, intermediate))
                    .with_after_latex(latex_expr(ctx, step.after)),
                );
            }
        }
    }

    out
}

fn generate_add_subtract_fractions_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let unit_numerators = both_fraction_numerators_are_one(ctx, before);

    let mut work = ctx.clone();
    let Some(intermediate) = build_two_fraction_common_denominator_intermediate(&mut work, before)
    else {
        return Vec::new();
    };

    let intermediate_display = display_expr(&work, intermediate);
    let after_display = display_expr(ctx, after);
    let intermediate_latex = latex_expr(&work, intermediate);
    let after_latex = latex_expr(ctx, after);

    let mut out = vec![SubStep::new(
        "Llevar a denominador común",
        display_expr(ctx, before),
        intermediate_display.clone(),
    )
    .with_before_latex(latex_expr(ctx, before))
    .with_after_latex(intermediate_latex.clone())];

    if unit_numerators
        && (intermediate_display != after_display || intermediate_latex != after_latex)
    {
        out.push(
            SubStep::new(
                "Simplificar el numerador y el denominador",
                intermediate_display,
                after_display,
            )
            .with_before_latex(intermediate_latex)
            .with_after_latex(after_latex),
        );
    }

    out
}

fn both_fraction_numerators_are_one(ctx: &Context, expr: ExprId) -> bool {
    let Some((left, right, _is_subtraction)) = extract_fraction_add_sub_operands(ctx, expr) else {
        return false;
    };

    let Some((left_num, _)) = as_div(ctx, left) else {
        return false;
    };
    let Some((right_num, _)) = as_div(ctx, right) else {
        return false;
    };

    is_one(ctx, left_num) && is_one(ctx, right_num)
}

fn build_two_fraction_common_denominator_intermediate(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let (left, right, is_subtraction) = extract_fraction_add_sub_operands(ctx, expr)?;

    let (left_num, left_den) = as_div(ctx, left)?;
    let (right_num, right_den) = as_div(ctx, right)?;

    let common_den = ctx.add(Expr::Mul(left_den, right_den));
    let lifted_left = ctx.add(Expr::Mul(left_num, right_den));
    let lifted_right = ctx.add(Expr::Mul(right_num, left_den));
    let numerator = if is_subtraction {
        ctx.add(Expr::Sub(lifted_left, lifted_right))
    } else {
        ctx.add(Expr::Add(lifted_left, lifted_right))
    };

    Some(ctx.add(Expr::Div(numerator, common_den)))
}

fn extract_fraction_add_sub_operands(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, bool)> {
    match ctx.get(expr) {
        Expr::Add(left, right) => match ctx.get(*right) {
            Expr::Neg(inner) => Some((*left, *inner, true)),
            _ => Some((*left, *right, false)),
        },
        Expr::Sub(left, right) => Some((*left, *right, true)),
        _ => None,
    }
}

fn fraction_expansion_cleanup_title(ctx: &Context, intermediate: ExprId) -> String {
    match count_fraction_terms_with_common_factor(ctx, intermediate) {
        0 => "Simplificar las fracciones resultantes".to_string(),
        1 => "Cancelar los factores comunes en la fracción que queda".to_string(),
        _ => "Cancelar los factores comunes en las fracciones resultantes".to_string(),
    }
}

fn count_fraction_terms_with_common_factor(ctx: &Context, expr: ExprId) -> usize {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() <= 1 {
        return usize::from(fraction_term_has_common_factor(ctx, expr));
    }

    terms
        .into_iter()
        .filter(|(term, _sign)| fraction_term_has_common_factor(ctx, *term))
        .count()
}

fn fraction_term_has_common_factor(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Div(numerator, denominator) => {
            first_common_factor(ctx, *numerator, *denominator).is_some()
        }
        _ => false,
    }
}

fn generate_mixed_fraction_split_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    if before == after {
        return Vec::new();
    }

    vec![concrete_expr_substep(
        ctx,
        "Separar la parte entera y la fracción restante",
        before,
        after,
    )]
}

fn generate_mixed_fraction_combine_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    if before == after {
        return Vec::new();
    }

    vec![concrete_expr_substep(
        ctx,
        "Poner la parte entera sobre el mismo denominador y combinar",
        before,
        after,
    )]
}

fn generate_canonicalize_roots_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
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

fn generate_same_base_power_merge_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    if before == after {
        return Vec::new();
    }

    vec![concrete_expr_substep(
        ctx,
        "Sumar los exponentes de la misma base",
        before,
        after,
    )]
}

fn generate_odd_half_power_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if let Some(substeps) = generate_odd_half_power_simplify_substeps(ctx, step) {
        return substeps;
    }

    vec![
        formula_substep(
            "Separar la mitad entera de la mitad radical",
            "u^((2k+1)/2)",
            "|u|^k · sqrt(u)",
            "u^{\\frac{2k+1}{2}}",
            "|u|^k\\cdot \\sqrt{u}",
        ),
        formula_substep(
            "Usar que queda una raíz cuadrada del mismo factor",
            "u^(k + 1/2)",
            "|u|^k · sqrt(u)",
            "u^{k + \\frac{1}{2}}",
            "|u|^k\\cdot \\sqrt{u}",
        ),
    ]
}

fn generate_expand_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    if polynomial_product_didactic_plan(ctx, before).is_some() {
        return generate_polynomial_product_normalize_substeps(ctx, step);
    }

    let perfect_square_cancel = generate_perfect_square_fraction_cancel_substeps(ctx, step);
    if !perfect_square_cancel.is_empty() {
        return perfect_square_cancel;
    }

    let difference_of_squares_cancel = generate_difference_of_squares_cancel_substeps(ctx, step);
    if !difference_of_squares_cancel.is_empty() {
        return difference_of_squares_cancel;
    }

    let sum_difference_cubes_cancel = generate_sum_difference_cubes_cancel_substeps(ctx, step);
    if !sum_difference_cubes_cancel.is_empty() {
        return sum_difference_cubes_cancel;
    }

    if let Some((factor, kind)) = common_factor_factorization_plan(ctx, after, before) {
        let factor_display = human_expr(ctx, factor);
        let factor_latex = latex_expr(ctx, factor);
        let _ = (factor_display, factor_latex, kind);
        return vec![formula_substep(
            "Usar la distributiva",
            &human_expr(ctx, before),
            &human_expr(ctx, after),
            &latex_expr(ctx, before),
            &latex_expr(ctx, after),
        )];
    }

    if let Some((a, b)) = sophie_germain_expansion_plan(ctx, before, after) {
        let a_display = human_expr(ctx, a);
        let b_display = human_expr(ctx, b);

        let _ = (a_display, b_display);
        return vec![formula_substep(
            "Usar (a^2 - 2ab + 2b^2) · (a^2 + 2ab + 2b^2) = a^4 + 4b^4",
            &human_expr(ctx, before),
            &human_expr(ctx, after),
            &latex_expr(ctx, before),
            &latex_expr(ctx, after),
        )];
    }

    Vec::new()
}

fn generate_factorization_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some(substeps) = generate_consecutive_telescoping_fraction_substeps(ctx, before, after) {
        return substeps;
    }

    if let Some((base, power)) = geometric_difference_factor_plan(ctx, before, after) {
        let base_display = human_expr(ctx, base);
        let base_latex = latex_expr(ctx, base);

        let _ = (base_display, base_latex, power);
        return vec![formula_substep(
            "Usar a^n - 1 = (a - 1) · (a^(n-1) + a^(n-2) + ... + a + 1)",
            &human_expr(ctx, before),
            &human_expr(ctx, after),
            &latex_expr(ctx, before),
            &latex_expr(ctx, after),
        )];
    }

    if let Some((factor, kind)) = common_factor_factorization_plan(ctx, before, after) {
        let factor_display = human_expr(ctx, factor);
        let _ = (factor_display, kind);
        return vec![formula_substep(
            "Usar el factor común",
            &human_expr(ctx, before),
            &human_expr(ctx, after),
            &latex_expr(ctx, before),
            &latex_expr(ctx, after),
        )];
    }

    if let Some((left, right)) = difference_of_squares_bases(ctx, before) {
        let left_display = human_expr(ctx, left);
        let right_display = human_expr(ctx, right);
        let left_latex = latex_expr(ctx, left);
        let right_latex = latex_expr(ctx, right);

        let _ = (left_display, right_display, left_latex, right_latex);
        return vec![formula_substep(
            "Usar a^2 - b^2 = (a - b) · (a + b)",
            &human_expr(ctx, before),
            &human_expr(ctx, after),
            &latex_expr(ctx, before),
            &latex_expr(ctx, after),
        )];
    }

    if cube_identity_plan(ctx, before, after).is_some() {
        return generate_sum_difference_cubes_substeps(ctx, step);
    }

    if let Some((left, right, kind, power)) = binomial_power_terms(ctx, after) {
        if power == 3 {
            let (left, right) = prefer_non_constant_term_first(ctx, left, right);
            let left_display = human_expr(ctx, left);
            let right_display = human_expr(ctx, right);
            let left_latex = latex_expr(ctx, left);
            let right_latex = latex_expr(ctx, right);
            let (title, before_display, before_latex) = match kind {
                BinomialSquareKind::Sum => (
                    "Usar a^3 + 3a^2b + 3ab^2 + b^3 = (a + b)^3",
                    format!(
                        "{left_display}^3 + 3 · {left_display}^2 · {right_display} + 3 · {left_display} · {right_display}^2 + {right_display}^3"
                    ),
                    format!(
                        "{left_latex}^3 + 3\\cdot {left_latex}^2\\cdot {right_latex} + 3\\cdot {left_latex}\\cdot {right_latex}^2 + {right_latex}^3"
                    ),
                ),
                BinomialSquareKind::Difference => (
                    "Usar a^3 - 3a^2b + 3ab^2 - b^3 = (a - b)^3",
                    format!(
                        "{left_display}^3 - 3 · {left_display}^2 · {right_display} + 3 · {left_display} · {right_display}^2 - {right_display}^3"
                    ),
                    format!(
                        "{left_latex}^3 - 3\\cdot {left_latex}^2\\cdot {right_latex} + 3\\cdot {left_latex}\\cdot {right_latex}^2 - {right_latex}^3"
                    ),
                ),
            };

            let _ = (left_display, right_display, left_latex, right_latex);
            return vec![formula_substep(
                title,
                &before_display,
                &human_expr(ctx, after),
                &before_latex,
                &latex_expr(ctx, after),
            )];
        }
    }

    if let Some((left, right, kind)) = binomial_square_terms(ctx, after) {
        let (left, right) = prefer_non_constant_term_first(ctx, left, right);
        let left_display = human_expr(ctx, left);
        let right_display = human_expr(ctx, right);
        let left_latex = latex_expr(ctx, left);
        let right_latex = latex_expr(ctx, right);
        let (title, before_display, before_latex) = match kind {
            BinomialSquareKind::Sum => (
                "Usar a^2 + 2ab + b^2 = (a + b)^2",
                format!(
                    "{left_display}^2 + 2 · {left_display} · {right_display} + {right_display}^2"
                ),
                format!(
                    "{left_latex}^2 + 2\\cdot {left_latex}\\cdot {right_latex} + {right_latex}^2"
                ),
            ),
            BinomialSquareKind::Difference => (
                "Usar a^2 - 2ab + b^2 = (a - b)^2",
                format!(
                    "{left_display}^2 - 2 · {left_display} · {right_display} + {right_display}^2"
                ),
                format!(
                    "{left_latex}^2 - 2\\cdot {left_latex}\\cdot {right_latex} + {right_latex}^2"
                ),
            ),
        };

        let _ = (left_display, right_display, left_latex, right_latex);
        return vec![formula_substep(
            title,
            &before_display,
            &human_expr(ctx, after),
            &before_latex,
            &latex_expr(ctx, after),
        )];
    }

    if let Some((a, b)) = sophie_germain_terms(ctx, before) {
        let a_display = human_expr(ctx, a);
        let b_display = human_expr(ctx, b);
        let a_latex = latex_expr(ctx, a);
        let b_latex = latex_expr(ctx, b);

        let _ = (a_display, b_display, a_latex, b_latex);
        return vec![formula_substep(
            "Usar a^4 + 4b^4 = (a^2 - 2ab + 2b^2) · (a^2 + 2ab + 2b^2)",
            &human_expr(ctx, before),
            &human_expr(ctx, after),
            &latex_expr(ctx, before),
            &latex_expr(ctx, after),
        )];
    }

    if let Some((a, b, c)) = alternating_cubic_vandermonde_plan(ctx, before, after) {
        let partial_display = format!("({a} - {b}) · ({a} - {c}) · ({b} - {c}) · L({a},{b},{c})");
        let partial_latex =
            format!("\\left({a} - {b}\\right)\\left({a} - {c}\\right)\\left({b} - {c}\\right)L({a},{b},{c})");

        return vec![
            formula_substep(
                "Si dos variables coinciden, la expresión vale 0",
                "F(a,b,c)",
                "(a - b) · (a - c) · (b - c) · L(a,b,c)",
                "F(a,b,c)",
                "\\left(a - b\\right)\\left(a - c\\right)\\left(b - c\\right)L(a,b,c)",
            ),
            formula_substep(
                "El factor restante es lineal y simétrico",
                &partial_display,
                &human_expr(ctx, after),
                &partial_latex,
                &latex_expr(ctx, after),
            ),
        ];
    }

    Vec::new()
}

fn alternating_cubic_vandermonde_plan(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<(String, String, String)> {
    let mut vars: Vec<_> = cas_ast::collect_variables(ctx, before)
        .into_iter()
        .collect();
    if vars.len() != 3 {
        return None;
    }
    vars.sort();

    if !matches_alternating_cubic_vandermonde_before(ctx, before, &vars) {
        return None;
    }
    if !matches_alternating_cubic_vandermonde_after(ctx, after, &vars) {
        return None;
    }

    Some((vars[0].clone(), vars[1].clone(), vars[2].clone()))
}

fn matches_alternating_cubic_vandermonde_before(
    ctx: &Context,
    before: ExprId,
    vars: &[String],
) -> bool {
    let terms = AddView::from_expr(ctx, before).terms;
    if terms.len() != 3 {
        return false;
    }

    let expected = [
        (&vars[0], &vars[1], &vars[2]),
        (&vars[1], &vars[2], &vars[0]),
        (&vars[2], &vars[0], &vars[1]),
    ];

    expected.iter().all(|(main, left, right)| {
        terms.iter().any(|(term, sign)| {
            *sign == Sign::Pos
                && matches_pow_three_times_difference(
                    ctx,
                    *term,
                    main.as_str(),
                    left.as_str(),
                    right.as_str(),
                )
        })
    })
}

fn matches_alternating_cubic_vandermonde_after(
    ctx: &Context,
    after: ExprId,
    vars: &[String],
) -> bool {
    let factors = expr_nary::mul_leaves(ctx, after);
    if factors.len() != 4 {
        return false;
    }

    let required_differences = [
        (&vars[0], &vars[1]),
        (&vars[0], &vars[2]),
        (&vars[1], &vars[2]),
    ];

    let has_all_differences = required_differences.iter().all(|(left, right)| {
        factors
            .iter()
            .any(|factor| matches_linear_difference(ctx, *factor, left.as_str(), right.as_str()))
    });

    has_all_differences
        && factors
            .iter()
            .any(|factor| matches_three_variable_sum(ctx, *factor, vars))
}

fn matches_pow_three_times_difference(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
    left_name: &str,
    right_name: &str,
) -> bool {
    let factors = expr_nary::mul_leaves(ctx, expr);
    if factors.len() != 2 {
        return false;
    }

    factors
        .iter()
        .any(|factor| matches_var_pow(ctx, *factor, var_name, 3))
        && factors
            .iter()
            .any(|factor| matches_linear_difference(ctx, *factor, left_name, right_name))
}

fn matches_var_pow(ctx: &Context, expr: ExprId, var_name: &str, exponent: i64) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            matches_var_name(ctx, *base, var_name)
                && matches!(
                    ctx.get(*exp),
                    Expr::Number(n) if n.is_integer() && *n.numer() == exponent.into()
                )
        }
        _ => false,
    }
}

fn matches_linear_difference(
    ctx: &Context,
    expr: ExprId,
    left_name: &str,
    right_name: &str,
) -> bool {
    match ctx.get(expr) {
        Expr::Sub(left, right) => {
            matches_var_name(ctx, *left, left_name) && matches_var_name(ctx, *right, right_name)
        }
        _ => false,
    }
}

fn matches_three_variable_sum(ctx: &Context, expr: ExprId, vars: &[String]) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 3 {
        return false;
    }

    let mut actual = Vec::with_capacity(3);
    for (term, sign) in terms {
        if sign != Sign::Pos {
            return false;
        }
        let Expr::Variable(sym_id) = ctx.get(term) else {
            return false;
        };
        actual.push(ctx.sym_name(*sym_id).to_string());
    }
    actual.sort();

    actual == vars
}

fn matches_var_name(ctx: &Context, expr: ExprId, expected: &str) -> bool {
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == expected)
}

fn needs_grouped_substitution_expr(expr: &Expr) -> bool {
    !matches!(
        expr,
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::Function(_, _)
    )
}

fn grouped_substitution_display(ctx: &Context, expr: ExprId) -> String {
    let display = human_expr(ctx, expr);
    if needs_grouped_substitution_expr(ctx.get(expr)) {
        format!("({display})")
    } else {
        display
    }
}

fn grouped_substitution_latex(ctx: &Context, expr: ExprId) -> String {
    let latex = latex_expr(ctx, expr);
    if needs_grouped_substitution_expr(ctx.get(expr)) {
        format!("\\left({latex}\\right)")
    } else {
        latex
    }
}

fn generate_binomial_expansion_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some((left, right, kind, power)) = binomial_power_terms(ctx, before) else {
        return Vec::new();
    };

    let _ = (left, right);
    let identity_title = match (kind, power) {
        (BinomialSquareKind::Sum, 2) => "Usar (a + b)^2 = a^2 + 2ab + b^2",
        (BinomialSquareKind::Difference, 2) => "Usar (a - b)^2 = a^2 - 2ab + b^2",
        (BinomialSquareKind::Sum, 3) => "Usar (a + b)^3 = a^3 + 3a^2b + 3ab^2 + b^3",
        (BinomialSquareKind::Difference, 3) => "Usar (a - b)^3 = a^3 - 3a^2b + 3ab^2 - b^3",
        _ => return Vec::new(),
    };

    vec![concrete_expr_substep(ctx, identity_title, before, after)]
}

fn generate_expand_log_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let (title, before_display, after_display, before_latex, after_latex) =
        if let Some(snippet) = log_formula_snippet(ctx, before, true) {
            snippet
        } else {
            (
                "Usar que el logaritmo de un producto se separa en una suma".to_string(),
                "ln(u · v)".to_string(),
                "ln(u) + ln(v)".to_string(),
                "\\ln(u\\cdot v)".to_string(),
                "\\ln(u) + \\ln(v)".to_string(),
            )
        };

    if before != after {
        return vec![concrete_expr_substep(ctx, title, before, after)];
    }

    vec![formula_substep(
        title,
        &before_display,
        &after_display,
        &before_latex,
        &after_latex,
    )]
}

fn generate_log_cancellation_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let visible_rule = super::visible_rule_names::visible_rule_name_for_step(
        step.rule_name.as_str(),
        step.description.as_str(),
    );
    if visible_rule.as_ref() != "Expandir logaritmos y cancelar términos iguales" {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    if !is_zero(ctx, after) {
        return Vec::new();
    }

    let mut work = ctx.clone();
    let Some((expansion_focus_before, expansion_focus_after, expanded_expr)) =
        build_log_cancellation_expansion_plan(&mut work, before)
    else {
        return Vec::new();
    };

    let mut substeps = vec![SubStep::new(
        "Expandir el logaritmo del producto o del cociente",
        human_expr(&work, expansion_focus_before),
        human_expr(&work, expansion_focus_after),
    )
    .with_before_latex(latex_expr(&work, expansion_focus_before))
    .with_after_latex(latex_expr(&work, expansion_focus_after))];

    let cancel_before = if let Some((extract_before, extract_after, extracted_expr)) =
        build_log_cancellation_exponent_plan(&mut work, expanded_expr)
    {
        substeps.push(
            SubStep::new(
                "Sacar exponentes fuera del logaritmo cuando sea necesario",
                human_expr(&work, extract_before),
                human_expr(&work, extract_after),
            )
            .with_before_latex(latex_expr(&work, extract_before))
            .with_after_latex(latex_expr(&work, extract_after)),
        );
        extracted_expr
    } else {
        expanded_expr
    };

    substeps.push(
        SubStep::new(
            "Cancelar términos iguales",
            human_expr(&work, cancel_before),
            human_expr(&work, after),
        )
        .with_before_latex(latex_expr(&work, cancel_before))
        .with_after_latex(latex_expr(&work, after)),
    );
    substeps
}

fn build_log_cancellation_expansion_plan(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    let mut rebuilt_terms = Vec::new();
    let mut focus_before = None;
    let mut focus_after = None;
    let terms = AddView::from_expr(ctx, expr).terms.to_vec();

    for (term, sign) in terms {
        if focus_before.is_none() {
            if let Some((term_before, term_after, expanded_terms)) =
                expand_log_term_into_signed_terms(ctx, term)
            {
                focus_before = Some(term_before);
                focus_after = Some(term_after);
                for (expanded_term, inner_sign) in expanded_terms {
                    rebuilt_terms.push((
                        expanded_term,
                        if sign == Sign::Pos {
                            inner_sign
                        } else {
                            inner_sign.negate()
                        },
                    ));
                }
                continue;
            }
        }

        rebuilt_terms.push((term, sign));
    }

    Some((
        focus_before?,
        focus_after?,
        build_add_from_signed_terms(ctx, &rebuilt_terms),
    ))
}

fn build_log_cancellation_exponent_plan(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    let mut focus_before_terms = Vec::new();
    let mut focus_after_terms = Vec::new();
    let mut rebuilt_terms = Vec::new();
    let terms = AddView::from_expr(ctx, expr).terms.to_vec();

    for (term, sign) in terms {
        if let Some(rewritten) = rewrite_log_power_term_concretely(ctx, term) {
            focus_before_terms.push((term, sign));
            focus_after_terms.push((rewritten, sign));
            rebuilt_terms.push((rewritten, sign));
        } else {
            rebuilt_terms.push((term, sign));
        }
    }

    if focus_before_terms.is_empty() {
        return None;
    }

    Some((
        build_add_from_signed_terms(ctx, &focus_before_terms),
        build_add_from_signed_terms(ctx, &focus_after_terms),
        build_add_from_signed_terms(ctx, &rebuilt_terms),
    ))
}

fn expand_log_term_into_signed_terms(
    ctx: &mut Context,
    term: ExprId,
) -> Option<ConcreteLogExpansion> {
    let (coeff, log_expr) = scaled_log_term(ctx, term).unwrap_or_else(|| (1.into(), term));
    let family = extract_log_didactic_family(ctx, log_expr)?;
    let arg = match ctx.get(log_expr) {
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Ln) && args.len() == 1 => {
            args[0]
        }
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Log) => {
            match args.as_slice() {
                [arg] => *arg,
                [_base, arg] => *arg,
                _ => return None,
            }
        }
        _ => return None,
    };
    let (inner_arg, wrap_abs) = abs_argument(ctx, arg)
        .map(|inner| (inner, true))
        .unwrap_or((arg, false));

    let signed_args = match ctx.get(inner_arg) {
        Expr::Mul(_, _) => expr_nary::mul_factors(ctx, inner_arg)
            .into_iter()
            .map(|factor| (factor, Sign::Pos))
            .collect::<Vec<_>>(),
        Expr::Div(numerator, denominator) => {
            let mut out = expr_nary::mul_factors(ctx, *numerator)
                .into_iter()
                .map(|factor| (factor, Sign::Pos))
                .collect::<Vec<_>>();
            out.extend(
                expr_nary::mul_factors(ctx, *denominator)
                    .into_iter()
                    .map(|factor| (factor, Sign::Neg)),
            );
            out
        }
        _ => return None,
    };

    let expanded_terms = signed_args
        .into_iter()
        .map(|(factor, sign)| {
            let term_arg = if wrap_abs {
                ctx.add(Expr::Function(ctx.builtin_id(BuiltinFn::Abs), vec![factor]))
            } else {
                factor
            };
            let log_term = build_log_call_for_family(ctx, family, term_arg);
            (scale_expr_by_positive_bigint(ctx, &coeff, log_term), sign)
        })
        .collect::<Vec<_>>();
    let expanded_expr = build_add_from_signed_terms(ctx, &expanded_terms);
    Some((term, expanded_expr, expanded_terms))
}

fn rewrite_log_power_term_concretely(ctx: &mut Context, term: ExprId) -> Option<ExprId> {
    let (coeff, log_expr) = scaled_log_term(ctx, term).unwrap_or_else(|| (1.into(), term));
    let (family, base, exponent) = log_power_extraction_family(ctx, log_expr)?;
    let power = positive_integer_literal_value(ctx, exponent)?;
    let target_log = if matches!(family, LogDidacticFamily::Ln) && (&power % 2) == 0.into() {
        let abs_base = ctx.add(Expr::Function(ctx.builtin_id(BuiltinFn::Abs), vec![base]));
        build_log_call_for_family(ctx, family, abs_base)
    } else {
        build_log_call_for_family(ctx, family, base)
    };

    Some(scale_expr_by_positive_bigint(
        ctx,
        &(coeff * power),
        target_log,
    ))
}

fn build_log_call_for_family(ctx: &mut Context, family: LogDidacticFamily, arg: ExprId) -> ExprId {
    match family {
        LogDidacticFamily::Ln => ctx.add(Expr::Function(ctx.builtin_id(BuiltinFn::Ln), vec![arg])),
        LogDidacticFamily::Log10 => {
            ctx.add(Expr::Function(ctx.builtin_id(BuiltinFn::Log), vec![arg]))
        }
        LogDidacticFamily::LogBase(base) => ctx.add(Expr::Function(
            ctx.builtin_id(BuiltinFn::Log),
            vec![base, arg],
        )),
    }
}

fn scale_expr_by_positive_bigint(
    ctx: &mut Context,
    coeff: &num_bigint::BigInt,
    expr: ExprId,
) -> ExprId {
    if coeff == &1.into() {
        expr
    } else {
        let coeff_expr = ctx.add(Expr::Number(BigRational::from_integer(coeff.clone())));
        ctx.add(Expr::Mul(coeff_expr, expr))
    }
}

fn build_add_from_signed_terms(ctx: &mut Context, terms: &[(ExprId, Sign)]) -> ExprId {
    let Some((first_term, first_sign)) = terms.first().copied() else {
        return ctx.num(0);
    };

    let mut acc = if first_sign == Sign::Pos {
        first_term
    } else {
        ctx.add(Expr::Neg(first_term))
    };

    for (term, sign) in terms.iter().copied().skip(1) {
        acc = if sign == Sign::Pos {
            ctx.add(Expr::Add(acc, term))
        } else {
            ctx.add(Expr::Sub(acc, term))
        };
    }

    acc
}

fn log_power_extraction_family(
    ctx: &Context,
    expr: ExprId,
) -> Option<(LogDidacticFamily, ExprId, ExprId)> {
    let (family, arg) = match ctx.get(expr) {
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Ln) && args.len() == 1 => {
            (LogDidacticFamily::Ln, args[0])
        }
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Log) => {
            match args.as_slice() {
                [arg] => (LogDidacticFamily::Log10, *arg),
                [base, arg] => (LogDidacticFamily::LogBase(*base), *arg),
                _ => return None,
            }
        }
        _ => return None,
    };

    let Expr::Pow(base, exponent) = ctx.get(arg) else {
        return None;
    };
    let Expr::Number(value) = ctx.get(*exponent) else {
        return None;
    };
    if !value.is_integer() || value <= &BigRational::zero() {
        return None;
    }

    Some((family, *base, *exponent))
}

fn generate_simplify_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some(substeps) = generate_odd_half_power_simplify_substeps(ctx, step) {
        return substeps;
    }

    if let Some(substeps) = generate_reverse_nested_fraction_substeps(ctx, before, after) {
        return substeps;
    }

    if let Some(substeps) = generate_log_change_of_base_chain_substeps(ctx, before, after) {
        return substeps;
    }

    if let Some(substeps) = generate_consecutive_telescoping_fraction_substeps(ctx, before, after) {
        return substeps;
    }

    generate_log_power_contraction_substep(ctx, before, after)
        .into_iter()
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct OddHalfPowerSimplifyPlan {
    base: ExprId,
    outside_power: i64,
}

fn generate_odd_half_power_simplify_substeps(ctx: &Context, step: &Step) -> Option<Vec<SubStep>> {
    let local_before = step.before_local().unwrap_or(step.before);
    let local_after = step.after_local().unwrap_or(step.after);
    if let Some(plan) = odd_half_power_simplify_plan(ctx, local_before, local_after) {
        return Some(build_odd_half_power_simplify_substeps(
            ctx,
            local_before,
            local_after,
            plan,
            odd_half_power_replacement_pair(step, local_before, local_after),
        ));
    }

    let (focus_before, focus_after, plan) =
        find_additive_odd_half_power_simplify_focus(ctx, step.before, step.after)?;
    Some(build_odd_half_power_simplify_substeps(
        ctx,
        focus_before,
        focus_after,
        plan,
        Some((step.before, step.after)),
    ))
}

fn build_odd_half_power_simplify_substeps(
    ctx: &Context,
    focus_before: ExprId,
    focus_after: ExprId,
    plan: OddHalfPowerSimplifyPlan,
    replacement_pair: Option<(ExprId, ExprId)>,
) -> Vec<SubStep> {
    let radicand = sqrt_radicand(ctx, focus_before).expect("odd-half-power focus should be a root");
    let (outside_power_display, _outside_power_latex) =
        power_display_and_latex(ctx, plan.base, plan.outside_power);
    let (even_power_display, even_power_latex) =
        power_display_and_latex(ctx, plan.base, 2 * plan.outside_power);
    let base_grouped_display = grouped_substitution_display(ctx, plan.base);
    let base_grouped_latex = grouped_substitution_latex(ctx, plan.base);
    let factorized_radicand_display = format!("{even_power_display} · {base_grouped_display}");
    let factorized_radicand_latex = format!("{even_power_latex}\\cdot {base_grouped_latex}");
    let factorized_root_display = format!("sqrt({factorized_radicand_display})");
    let factorized_root_latex = format!("\\sqrt{{{factorized_radicand_latex}}}");

    let mut out = vec![
        SubStep::new(
            "Separar el radicando en una potencia par y un factor",
            human_expr(ctx, radicand),
            factorized_radicand_display.clone(),
        )
        .with_before_latex(latex_expr(ctx, radicand))
        .with_after_latex(factorized_radicand_latex.clone()),
        SubStep::new(
            format!(
                "Como {} ≥ 0, sacar {} fuera de la raíz",
                human_expr(ctx, plan.base),
                outside_power_display
            ),
            factorized_root_display,
            human_expr(ctx, focus_after),
        )
        .with_before_latex(factorized_root_latex)
        .with_after_latex(latex_expr(ctx, focus_after)),
    ];

    if let Some((replacement_before, replacement_after)) = replacement_pair {
        out.push(
            SubStep::new(
                "Reemplazar ese bloque en la expresión",
                human_expr(ctx, replacement_before),
                human_expr(ctx, replacement_after),
            )
            .with_before_latex(latex_expr(ctx, replacement_before))
            .with_after_latex(latex_expr(ctx, replacement_after)),
        );
    }

    out
}

fn odd_half_power_replacement_pair(
    step: &Step,
    local_before: ExprId,
    local_after: ExprId,
) -> Option<(ExprId, ExprId)> {
    if let (Some(global_before), Some(global_after)) = (step.global_before, step.global_after) {
        if global_before != local_before || global_after != local_after {
            return Some((global_before, global_after));
        }
    }

    ((step.before != local_before) || (step.after != local_after))
        .then_some((step.before, step.after))
}

fn odd_half_power_simplify_plan(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<OddHalfPowerSimplifyPlan> {
    let radicand = sqrt_radicand(ctx, before)?;
    let Expr::Pow(base, exponent) = ctx.get(radicand) else {
        return None;
    };
    let numerator = small_positive_integer_value(ctx, *exponent)?;
    if numerator < 3 || numerator % 2 == 0 {
        return None;
    }

    let outside_power = (numerator - 1) / 2;
    matches_odd_half_power_simplified_after(ctx, after, *base, outside_power).then_some(
        OddHalfPowerSimplifyPlan {
            base: *base,
            outside_power,
        },
    )
}

fn matches_odd_half_power_simplified_after(
    ctx: &Context,
    expr: ExprId,
    base: ExprId,
    outside_power: i64,
) -> bool {
    let factors = expr_nary::mul_leaves(ctx, expr);
    if factors.len() != 2 {
        return false;
    }

    let mut saw_sqrt = false;
    let mut saw_outer = false;
    for factor in factors {
        if !saw_sqrt
            && sqrt_radicand(ctx, factor)
                .is_some_and(|radicand| compare_expr(ctx, radicand, base) == Ordering::Equal)
        {
            saw_sqrt = true;
            continue;
        }

        if !saw_outer && matches_odd_half_power_outer_factor(ctx, factor, base, outside_power) {
            saw_outer = true;
            continue;
        }

        return false;
    }

    saw_sqrt && saw_outer
}

fn matches_odd_half_power_outer_factor(
    ctx: &Context,
    factor: ExprId,
    base: ExprId,
    outside_power: i64,
) -> bool {
    if outside_power == 1 && compare_expr(ctx, factor, base) == Ordering::Equal {
        return true;
    }

    if outside_power == 1
        && abs_argument(ctx, factor)
            .is_some_and(|inner| compare_expr(ctx, inner, base) == Ordering::Equal)
    {
        return true;
    }

    match ctx.get(factor) {
        Expr::Pow(pow_base, exponent)
            if small_positive_integer_value(ctx, *exponent) == Some(outside_power) =>
        {
            compare_expr(ctx, *pow_base, base) == Ordering::Equal
                || abs_argument(ctx, *pow_base)
                    .is_some_and(|inner| compare_expr(ctx, inner, base) == Ordering::Equal)
        }
        _ => false,
    }
}

fn power_display_and_latex(ctx: &Context, base: ExprId, exponent: i64) -> (String, String) {
    if exponent == 1 {
        return (human_expr(ctx, base), latex_expr(ctx, base));
    }

    let mut temp_ctx = ctx.clone();
    let exponent_expr = temp_ctx.num(exponent);
    let power_expr = temp_ctx.add_raw(Expr::Pow(base, exponent_expr));
    (
        human_expr(&temp_ctx, power_expr),
        latex_expr(&temp_ctx, power_expr),
    )
}

fn find_additive_odd_half_power_simplify_focus(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<(ExprId, ExprId, OddHalfPowerSimplifyPlan)> {
    let before_terms = expr_nary::add_terms_signed(ctx, before);
    let after_terms = expr_nary::add_terms_signed(ctx, after);
    if before_terms.len() < 2 || after_terms.len() != before_terms.len() {
        return None;
    }

    for (before_index, (before_focus, _before_sign)) in before_terms.iter().copied().enumerate() {
        for (after_index, (after_focus, _after_sign)) in after_terms.iter().copied().enumerate() {
            let Some(plan) = odd_half_power_simplify_plan(ctx, before_focus, after_focus) else {
                continue;
            };

            let before_passthrough =
                collect_signed_passthrough_terms_excluding_index(&before_terms, before_index);
            let after_passthrough =
                collect_signed_passthrough_terms_excluding_index(&after_terms, after_index);
            if signed_additive_term_multiset_matches(ctx, &before_passthrough, &after_passthrough) {
                return Some((before_focus, after_focus, plan));
            }
        }
    }

    None
}

fn collect_signed_passthrough_terms_excluding_index(
    terms: &[(ExprId, Sign)],
    excluded_index: usize,
) -> Vec<(ExprId, Sign)> {
    terms
        .iter()
        .enumerate()
        .filter_map(|(index, term)| (index != excluded_index).then_some(*term))
        .collect()
}

fn signed_additive_term_multiset_matches(
    ctx: &Context,
    lhs_terms: &[(ExprId, Sign)],
    rhs_terms: &[(ExprId, Sign)],
) -> bool {
    if lhs_terms.len() != rhs_terms.len() {
        return false;
    }

    let mut lhs = lhs_terms.to_vec();
    let mut rhs = rhs_terms.to_vec();
    lhs.sort_by(|(left_expr, left_sign), (right_expr, right_sign)| {
        compare_expr(ctx, *left_expr, *right_expr)
            .then_with(|| sign_sort_key(*left_sign).cmp(&sign_sort_key(*right_sign)))
    });
    rhs.sort_by(|(left_expr, left_sign), (right_expr, right_sign)| {
        compare_expr(ctx, *left_expr, *right_expr)
            .then_with(|| sign_sort_key(*left_sign).cmp(&sign_sort_key(*right_sign)))
    });

    lhs == rhs
}

fn sign_sort_key(sign: Sign) -> u8 {
    match sign {
        Sign::Pos => 0,
        Sign::Neg => 1,
    }
}

fn generate_reverse_nested_fraction_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<Vec<SubStep>> {
    let pattern = reverse_nested_fraction_pattern(ctx, after)?;

    match pattern {
        NestedFractionPattern::OneOverSumWithFraction
        | NestedFractionPattern::FractionOverSumWithFraction => {
            let Expr::Div(_, before_den) = ctx.get(before) else {
                return None;
            };
            let Expr::Div(_, after_den) = ctx.get(after) else {
                return None;
            };
            let (_, common_den) = split_add_with_single_fraction(ctx, *after_den)?;
            let common_den_display = human_expr(ctx, common_den);
            let common_den_grouped_display = grouped_substitution_display(ctx, common_den);
            let common_den_grouped_latex = grouped_substitution_latex(ctx, common_den);
            let after_den_display = human_expr(ctx, *after_den);
            let after_den_latex = latex_expr(ctx, *after_den);

            return Some(vec![SubStep::new(
                format!("Reescribir el denominador sacando factor común {common_den_display}"),
                human_expr(ctx, *before_den),
                format!("{common_den_grouped_display} · ({after_den_display})"),
            )
            .with_before_latex(latex_expr(ctx, *before_den))
            .with_after_latex(format!(
                "{common_den_grouped_latex}\\cdot \\left({after_den_latex}\\right)"
            ))]);
        }
        NestedFractionPattern::SumWithFractionOverScalar => {
            let Expr::Div(before_num, _) = ctx.get(before) else {
                return None;
            };
            let Expr::Div(after_num, _) = ctx.get(after) else {
                return None;
            };
            let (_, common_den) = split_add_with_single_fraction(ctx, *after_num)?;
            let common_den_display = human_expr(ctx, common_den);
            let common_den_grouped_display = grouped_substitution_display(ctx, common_den);
            let common_den_grouped_latex = grouped_substitution_latex(ctx, common_den);
            let after_num_display = human_expr(ctx, *after_num);
            let after_num_latex = latex_expr(ctx, *after_num);

            return Some(vec![SubStep::new(
                format!("Reescribir el numerador sacando factor común {common_den_display}"),
                human_expr(ctx, *before_num),
                format!("{common_den_grouped_display} · ({after_num_display})"),
            )
            .with_before_latex(latex_expr(ctx, *before_num))
            .with_after_latex(format!(
                "{common_den_grouped_latex}\\cdot \\left({after_num_latex}\\right)"
            ))]);
        }
        NestedFractionPattern::OneOverSumWithUnitFraction | NestedFractionPattern::General => {}
    }

    None
}

fn generate_reverse_nested_fraction_rule_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.global_before.unwrap_or(step.before);
    let after = step.global_after.unwrap_or(step.after);

    generate_reverse_nested_fraction_substeps(ctx, before, after)
        .or_else(|| {
            let before = step.before_local().unwrap_or(step.before);
            let after = step.after_local().unwrap_or(step.after);
            generate_reverse_nested_fraction_substeps(ctx, before, after)
        })
        .unwrap_or_default()
}

fn reverse_nested_fraction_pattern(ctx: &Context, after: ExprId) -> Option<NestedFractionPattern> {
    let pattern = super::nested_fraction_analysis::classify_nested_fraction(ctx, after)?;
    match pattern {
        NestedFractionPattern::OneOverSumWithFraction
        | NestedFractionPattern::FractionOverSumWithFraction
        | NestedFractionPattern::SumWithFractionOverScalar => {}
        _ => return None,
    }

    let Expr::Div(num, den) = ctx.get(after) else {
        return None;
    };

    match pattern {
        NestedFractionPattern::OneOverSumWithFraction
        | NestedFractionPattern::FractionOverSumWithFraction => {
            let denominator = *den;
            let _ = split_add_with_single_fraction(ctx, denominator)?;
        }
        NestedFractionPattern::SumWithFractionOverScalar => {
            let numerator = *num;
            let _ = split_add_with_single_fraction(ctx, numerator)?;
        }
        NestedFractionPattern::OneOverSumWithUnitFraction | NestedFractionPattern::General => {
            return None
        }
    }

    Some(pattern)
}

fn split_add_with_single_fraction(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };

    match (ctx.get(*left), ctx.get(*right)) {
        (Expr::Div(_, left_den), _) if !matches!(ctx.get(*right), Expr::Div(_, _)) => {
            Some((*left, *left_den))
        }
        (_, Expr::Div(_, right_den)) if !matches!(ctx.get(*left), Expr::Div(_, _)) => {
            Some((*right, *right_den))
        }
        _ => None,
    }
}

fn generate_telescoping_fraction_combine_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some((u, gap_display, gap_is_one)) = telescoping_fraction_base_and_gap(ctx, after, before)
    else {
        return Vec::new();
    };
    let _ = u;

    if gap_is_one {
        return vec![formula_substep(
            "Usar 1 / u - 1 / (u + 1) = 1 / (u · (u + 1))",
            &human_expr(ctx, before),
            &human_expr(ctx, after),
            &latex_expr(ctx, before),
            &latex_expr(ctx, after),
        )];
    }

    let _ = gap_display;
    vec![formula_substep(
        "Usar 1 / k · (1 / u - 1 / (u + k)) = 1 / (u · (u + k))",
        &human_expr(ctx, before),
        &human_expr(ctx, after),
        &latex_expr(ctx, before),
        &latex_expr(ctx, after),
    )]
}

fn generate_telescoping_fraction_split_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    generate_consecutive_telescoping_fraction_substeps(ctx, before, after).unwrap_or_default()
}

fn generate_consecutive_telescoping_fraction_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<Vec<SubStep>> {
    let (u, gap_display, gap_is_one) = telescoping_fraction_base_and_gap(ctx, before, after)?;
    let _ = u;

    if gap_is_one {
        return Some(vec![formula_substep(
            "Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)",
            &human_expr(ctx, before),
            &human_expr(ctx, after),
            &latex_expr(ctx, before),
            &latex_expr(ctx, after),
        )]);
    }

    let _ = gap_display;
    Some(vec![formula_substep(
        "Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))",
        &human_expr(ctx, before),
        &human_expr(ctx, after),
        &latex_expr(ctx, before),
        &latex_expr(ctx, after),
    )])
}

fn generate_finite_product_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if step
        .description
        .starts_with("Factorized telescoping product:")
    {
        return generate_factorized_finite_product_substeps(ctx, step);
    }

    if !step.description.starts_with("Telescoping product:") {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(call) = try_extract_finite_aggregate_call(ctx, before, "product") else {
        return Vec::new();
    };
    let Expr::Div(num, den) = ctx.get(call.term) else {
        return Vec::new();
    };
    let numeric_offsets = if let (Some(num_offset), Some(den_offset)) = (
        extract_linear_offset(ctx, *num, &call.var_name),
        extract_linear_offset(ctx, *den, &call.var_name),
    ) {
        if num_offset - den_offset != 1 {
            return Vec::new();
        }
        Some((num_offset, den_offset))
    } else {
        None
    };

    let affine_symbolic_pattern = if numeric_offsets.is_none() {
        detect_affine_consecutive_telescoping_sum_pattern(ctx, *den, *num, &call.var_name)
    } else {
        None
    };

    let (
        first_num_plain,
        first_num_latex,
        first_den_plain,
        first_den_latex,
        second_num_plain,
        second_num_latex,
        second_den_plain,
        second_den_latex,
        last_num_plain,
        last_num_latex,
        last_den_plain,
        last_den_latex,
    ) = if let Some((base, next_base, _gap)) = affine_symbolic_pattern {
        let mut temp_ctx = ctx.clone();
        let one = temp_ctx.num(1);
        let start_next_index = temp_ctx.add(Expr::Add(call.start_expr, one));
        let start_base = substitute_expr_by_id(&mut temp_ctx, base, call.var_expr, call.start_expr);
        let start_next_base =
            substitute_expr_by_id(&mut temp_ctx, next_base, call.var_expr, call.start_expr);
        let second_base =
            substitute_expr_by_id(&mut temp_ctx, base, call.var_expr, start_next_index);
        let second_next_base =
            substitute_expr_by_id(&mut temp_ctx, next_base, call.var_expr, start_next_index);
        let end_base = substitute_expr_by_id(&mut temp_ctx, base, call.var_expr, call.end_expr);
        let end_next_base =
            substitute_expr_by_id(&mut temp_ctx, next_base, call.var_expr, call.end_expr);
        let (first_den_plain, first_den_latex) = render_temp_expr(&temp_ctx, start_base);
        let (first_num_plain, first_num_latex) = render_temp_expr(&temp_ctx, start_next_base);
        let (second_den_plain, second_den_latex) = render_temp_expr(&temp_ctx, second_base);
        let (second_num_plain, second_num_latex) = render_temp_expr(&temp_ctx, second_next_base);
        let (last_den_plain, last_den_latex) = render_temp_expr(&temp_ctx, end_base);
        let (last_num_plain, last_num_latex) = render_temp_expr(&temp_ctx, end_next_base);
        (
            first_num_plain,
            first_num_latex,
            first_den_plain,
            first_den_latex,
            second_num_plain,
            second_num_latex,
            second_den_plain,
            second_den_latex,
            last_num_plain,
            last_num_latex,
            last_den_plain,
            last_den_latex,
        )
    } else if numeric_offsets.is_none() {
        let Some(base) = extract_unit_shifted_base(ctx, *den, &call.var_name) else {
            return Vec::new();
        };
        let mut temp_ctx = ctx.clone();
        let expected_num = shifted_expr(&mut temp_ctx, base, 1);
        if compare_expr(&temp_ctx, *num, expected_num) != std::cmp::Ordering::Equal {
            return Vec::new();
        }

        let mut temp_ctx = ctx.clone();
        let start_base = substitute_expr_by_id(&mut temp_ctx, base, call.var_expr, call.start_expr);
        let end_base = substitute_expr_by_id(&mut temp_ctx, base, call.var_expr, call.end_expr);
        let (first_den_plain, first_den_latex) = render_temp_expr(&temp_ctx, start_base);
        let (first_num_plain, first_num_latex) = shifted_expr_strings(&temp_ctx, start_base, 1);
        let (second_den_plain, second_den_latex) = shifted_expr_strings(&temp_ctx, start_base, 1);
        let (second_num_plain, second_num_latex) = shifted_expr_strings(&temp_ctx, start_base, 2);
        let (last_den_plain, last_den_latex) = render_temp_expr(&temp_ctx, end_base);
        let (last_num_plain, last_num_latex) = shifted_expr_strings(&temp_ctx, end_base, 1);
        (
            first_num_plain,
            first_num_latex,
            first_den_plain,
            first_den_latex,
            second_num_plain,
            second_num_latex,
            second_den_plain,
            second_den_latex,
            last_num_plain,
            last_num_latex,
            last_den_plain,
            last_den_latex,
        )
    } else if let Some((num_offset, den_offset)) = numeric_offsets {
        let (first_num_plain, first_num_latex) =
            shifted_expr_strings(ctx, call.start_expr, num_offset);
        let (first_den_plain, first_den_latex) =
            shifted_expr_strings(ctx, call.start_expr, den_offset);
        let (second_num_plain, second_num_latex) =
            shifted_expr_strings(ctx, call.start_expr, num_offset + 1);
        let (second_den_plain, second_den_latex) =
            shifted_expr_strings(ctx, call.start_expr, den_offset + 1);
        let (last_num_plain, last_num_latex) = shifted_expr_strings(ctx, call.end_expr, num_offset);
        let (last_den_plain, last_den_latex) = shifted_expr_strings(ctx, call.end_expr, den_offset);
        (
            first_num_plain,
            first_num_latex,
            first_den_plain,
            first_den_latex,
            second_num_plain,
            second_num_latex,
            second_den_plain,
            second_den_latex,
            last_num_plain,
            last_num_latex,
            last_den_plain,
            last_den_latex,
        )
    } else {
        return Vec::new();
    };

    let expansion_plain = format!(
        "{} · {} · … · {}",
        render_fraction_plain(&first_num_plain, &first_den_plain),
        render_fraction_plain(&second_num_plain, &second_den_plain),
        render_fraction_plain(&last_num_plain, &last_den_plain),
    );
    let expansion_latex = format!(
        "{}\\cdot {}\\cdot \\cdots \\cdot {}",
        render_fraction_latex(&first_num_latex, &first_den_latex),
        render_fraction_latex(&second_num_latex, &second_den_latex),
        render_fraction_latex(&last_num_latex, &last_den_latex),
    );
    let endpoint_plain = render_fraction_plain(&last_num_plain, &first_den_plain);
    let endpoint_latex = render_fraction_latex(&last_num_latex, &first_den_latex);
    let after_plain = human_expr(ctx, after);
    let after_latex = latex_expr(ctx, after);

    let mut out = vec![
        formula_substep(
            "Escribir los primeros y últimos factores del producto",
            &human_expr(ctx, before),
            &expansion_plain,
            &latex_expr(ctx, before),
            &expansion_latex,
        ),
        formula_substep(
            "Los factores intermedios se cancelan por parejas",
            &expansion_plain,
            &endpoint_plain,
            &expansion_latex,
            &endpoint_latex,
        ),
    ];

    if !same_math_render(&endpoint_latex, &after_latex) {
        out.push(formula_substep(
            "Solo quedan el último numerador y el primer denominador",
            &endpoint_plain,
            &after_plain,
            &endpoint_latex,
            &after_latex,
        ));
    }

    out
}

fn generate_factorized_finite_product_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(call) = try_extract_finite_aggregate_call(ctx, before, "product") else {
        return Vec::new();
    };
    let Some(u_expr) = detect_factorized_telescoping_square_base(ctx, call.term, &call.var_name)
    else {
        return Vec::new();
    };

    let mut temp_ctx = ctx.clone();
    let start_base = substitute_expr_by_id(&mut temp_ctx, u_expr, call.var_expr, call.start_expr);
    let end_base = substitute_expr_by_id(&mut temp_ctx, u_expr, call.var_expr, call.end_expr);
    let (first_plain, first_latex) = render_temp_expr(&temp_ctx, start_base);
    let (second_plain, second_latex) = shifted_expr_strings(&temp_ctx, start_base, 1);
    let (last_plain, last_latex) = render_temp_expr(&temp_ctx, end_base);
    let (last_plus_one_plain, last_plus_one_latex) = shifted_expr_strings(&temp_ctx, end_base, 1);
    let (first_minus_one_plain, first_minus_one_latex) =
        shifted_expr_strings(&temp_ctx, start_base, -1);

    let factorized_series_plain = format!(
        "{} · {} · … · {}",
        render_fraction_plain(
            &render_square_difference_plain(&first_plain),
            &render_power2_plain(&first_plain),
        ),
        render_fraction_plain(
            &render_square_difference_plain(&second_plain),
            &render_power2_plain(&second_plain),
        ),
        render_fraction_plain(
            &render_square_difference_plain(&last_plain),
            &render_power2_plain(&last_plain),
        ),
    );
    let factorized_series_latex = format!(
        "{}\\cdot {}\\cdot \\cdots \\cdot {}",
        render_fraction_latex(
            &render_square_difference_latex(&first_latex),
            &render_power2_latex(&first_latex),
        ),
        render_fraction_latex(
            &render_square_difference_latex(&second_latex),
            &render_power2_latex(&second_latex),
        ),
        render_fraction_latex(
            &render_square_difference_latex(&last_latex),
            &render_power2_latex(&last_latex),
        ),
    );
    let telescoped_plain = render_fraction_plain(
        &format!("{first_minus_one_plain} · {last_plus_one_plain}"),
        &format!("{first_plain} · {last_plain}"),
    );
    let telescoped_latex = render_fraction_latex(
        &format!("{first_minus_one_latex}\\cdot {last_plus_one_latex}"),
        &format!("{first_latex}\\cdot {last_latex}"),
    );
    let after_plain = human_expr(ctx, after);
    let after_latex = latex_expr(ctx, after);

    let mut out = vec![
        formula_substep(
            "Usar (u^2 - 1) / u^2 = ((u - 1) · (u + 1)) / u^2",
            &human_expr(ctx, before),
            &factorized_series_plain,
            &factorized_series_latex,
            &latex_expr(ctx, before),
        ),
        formula_substep(
            "Los factores (u + 1) y (u - 1) se cancelan telescópicamente",
            &factorized_series_plain,
            &telescoped_plain,
            &factorized_series_latex,
            &telescoped_latex,
        ),
    ];

    if !same_math_render(&telescoped_latex, &after_latex) {
        out.push(formula_substep(
            "Solo quedan el primer factor u - 1 y el último factor u + 1",
            &telescoped_plain,
            &after_plain,
            &telescoped_latex,
            &after_latex,
        ));
    }
    out
}

fn generate_finite_summation_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if !step.description.starts_with("Telescoping sum:") {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(call) = try_extract_finite_aggregate_call(ctx, before, "sum") else {
        return Vec::new();
    };
    let Expr::Div(num, den) = ctx.get(call.term) else {
        return Vec::new();
    };
    let Expr::Number(n) = ctx.get(*num) else {
        return Vec::new();
    };
    if !n.is_one() {
        return Vec::new();
    }
    let Expr::Mul(factor1, factor2) = ctx.get(*den) else {
        return Vec::new();
    };

    if let (Some(offset1), Some(offset2)) = (
        extract_linear_offset(ctx, *factor1, &call.var_name),
        extract_linear_offset(ctx, *factor2, &call.var_name),
    ) {
        if (offset1 - offset2).abs() == 1 {
            let low_offset = offset1.min(offset2);
            let high_offset = offset1.max(offset2);
            let (term1_plain, term1_latex) = shifted_expr_strings(ctx, call.var_expr, low_offset);
            let (term2_plain, term2_latex) = shifted_expr_strings(ctx, call.var_expr, high_offset);
            let (first_plain, first_latex) = shifted_expr_strings(ctx, call.start_expr, low_offset);
            let (second_plain, second_latex) =
                shifted_expr_strings(ctx, call.start_expr, high_offset);
            let (third_plain, third_latex) =
                shifted_expr_strings(ctx, call.start_expr, high_offset + 1);
            let (penultimate_plain, penultimate_latex) =
                shifted_expr_strings(ctx, call.end_expr, low_offset);
            let (last_plain, last_latex) = shifted_expr_strings(ctx, call.end_expr, high_offset);

            let decomposed_plain = format!(
                "{} - {}",
                render_unit_fraction_plain(&term1_plain),
                render_unit_fraction_plain(&term2_plain),
            );
            let decomposed_latex = format!(
                "{} - {}",
                render_unit_fraction_latex(&term1_latex),
                render_unit_fraction_latex(&term2_latex),
            );
            let telescoping_series_plain = format!(
                "{} - {} + {} - {} + … + {} - {}",
                render_unit_fraction_plain(&first_plain),
                render_unit_fraction_plain(&second_plain),
                render_unit_fraction_plain(&second_plain),
                render_unit_fraction_plain(&third_plain),
                render_unit_fraction_plain(&penultimate_plain),
                render_unit_fraction_plain(&last_plain),
            );
            let telescoping_series_latex = format!(
                "{} - {} + {} - {} + \\cdots + {} - {}",
                render_unit_fraction_latex(&first_latex),
                render_unit_fraction_latex(&second_latex),
                render_unit_fraction_latex(&second_latex),
                render_unit_fraction_latex(&third_latex),
                render_unit_fraction_latex(&penultimate_latex),
                render_unit_fraction_latex(&last_latex),
            );

            return vec![
                formula_substep(
                    "Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)",
                    &human_expr(ctx, call.term),
                    &decomposed_plain,
                    &latex_expr(ctx, call.term),
                    &decomposed_latex,
                ),
                formula_substep(
                    "La suma telescópica cancela los términos intermedios",
                    &telescoping_series_plain,
                    &human_expr(ctx, after),
                    &telescoping_series_latex,
                    &latex_expr(ctx, after),
                ),
            ];
        }
    }

    if let Some((base, next_base, gap)) =
        detect_affine_consecutive_telescoping_sum_pattern(ctx, *factor1, *factor2, &call.var_name)
    {
        let mut temp_ctx = ctx.clone();
        let one = temp_ctx.num(1);
        let start_next_index = temp_ctx.add(Expr::Add(call.start_expr, one));
        let start_base = substitute_expr_by_id(&mut temp_ctx, base, call.var_expr, call.start_expr);
        let start_next_base =
            substitute_expr_by_id(&mut temp_ctx, next_base, call.var_expr, call.start_expr);
        let second_next_base =
            substitute_expr_by_id(&mut temp_ctx, next_base, call.var_expr, start_next_index);
        let end_base = substitute_expr_by_id(&mut temp_ctx, base, call.var_expr, call.end_expr);
        let end_next_base =
            substitute_expr_by_id(&mut temp_ctx, next_base, call.var_expr, call.end_expr);

        let (u_plain, _) = render_temp_expr(ctx, base);
        let (gap_plain, gap_latex) = render_temp_expr(ctx, gap);
        let (first_plain, first_latex) = render_temp_expr(&temp_ctx, start_base);
        let (second_plain, second_latex) = render_temp_expr(&temp_ctx, start_next_base);
        let (third_plain, third_latex) = render_temp_expr(&temp_ctx, second_next_base);
        let (penultimate_plain, penultimate_latex) = render_temp_expr(&temp_ctx, end_base);
        let (last_plain, last_latex) = render_temp_expr(&temp_ctx, end_next_base);

        let decomposed_plain = format!(
            "{} · ({} - {})",
            render_unit_fraction_plain(&gap_plain),
            render_unit_fraction_plain(&u_plain),
            render_unit_fraction_plain(&human_expr(ctx, next_base)),
        );
        let decomposed_latex = format!(
            "{}\\cdot \\left({} - {}\\right)",
            render_unit_fraction_latex(&gap_latex),
            render_unit_fraction_latex(&latex_expr(ctx, base)),
            render_unit_fraction_latex(&latex_expr(ctx, next_base)),
        );
        let telescoping_series_plain = format!(
            "{} · ({} - {}) + {} · ({} - {}) + … + {} · ({} - {})",
            render_unit_fraction_plain(&gap_plain),
            render_unit_fraction_plain(&first_plain),
            render_unit_fraction_plain(&second_plain),
            render_unit_fraction_plain(&gap_plain),
            render_unit_fraction_plain(&second_plain),
            render_unit_fraction_plain(&third_plain),
            render_unit_fraction_plain(&gap_plain),
            render_unit_fraction_plain(&penultimate_plain),
            render_unit_fraction_plain(&last_plain),
        );
        let telescoping_series_latex = format!(
            "{}\\cdot \\left({} - {}\\right) + {}\\cdot \\left({} - {}\\right) + \\cdots + {}\\cdot \\left({} - {}\\right)",
            render_unit_fraction_latex(&gap_latex),
            render_unit_fraction_latex(&first_latex),
            render_unit_fraction_latex(&second_latex),
            render_unit_fraction_latex(&gap_latex),
            render_unit_fraction_latex(&second_latex),
            render_unit_fraction_latex(&third_latex),
            render_unit_fraction_latex(&gap_latex),
            render_unit_fraction_latex(&penultimate_latex),
            render_unit_fraction_latex(&last_latex),
        );

        return vec![
            formula_substep(
                "Usar 1 / (u · (u + g)) = 1 / g · (1 / u - 1 / (u + g))",
                &human_expr(ctx, call.term),
                &decomposed_plain,
                &latex_expr(ctx, call.term),
                &decomposed_latex,
            ),
            formula_substep(
                "La suma telescópica cancela los términos intermedios",
                &telescoping_series_plain,
                &human_expr(ctx, after),
                &telescoping_series_latex,
                &latex_expr(ctx, after),
            ),
        ];
    }

    let Some(base) =
        detect_consecutive_telescoping_sum_base(ctx, *factor1, *factor2, &call.var_name)
    else {
        return Vec::new();
    };

    let mut temp_ctx = ctx.clone();
    let start_base = substitute_expr_by_id(&mut temp_ctx, base, call.var_expr, call.start_expr);
    let end_base = substitute_expr_by_id(&mut temp_ctx, base, call.var_expr, call.end_expr);

    let (term1_plain, term1_latex) = render_temp_expr(&temp_ctx, start_base);
    let (term2_plain, term2_latex) = shifted_expr_strings(&temp_ctx, start_base, 1);
    let (first_plain, first_latex) = render_temp_expr(&temp_ctx, start_base);
    let (second_plain, second_latex) = shifted_expr_strings(&temp_ctx, start_base, 1);
    let (third_plain, third_latex) = shifted_expr_strings(&temp_ctx, start_base, 2);
    let (penultimate_plain, penultimate_latex) = render_temp_expr(&temp_ctx, end_base);
    let (last_plain, last_latex) = shifted_expr_strings(&temp_ctx, end_base, 1);

    let decomposed_plain = format!(
        "{} - {}",
        render_unit_fraction_plain(&term1_plain),
        render_unit_fraction_plain(&term2_plain),
    );
    let decomposed_latex = format!(
        "{} - {}",
        render_unit_fraction_latex(&term1_latex),
        render_unit_fraction_latex(&term2_latex),
    );
    let telescoping_series_plain = format!(
        "{} - {} + {} - {} + … + {} - {}",
        render_unit_fraction_plain(&first_plain),
        render_unit_fraction_plain(&second_plain),
        render_unit_fraction_plain(&second_plain),
        render_unit_fraction_plain(&third_plain),
        render_unit_fraction_plain(&penultimate_plain),
        render_unit_fraction_plain(&last_plain),
    );
    let telescoping_series_latex = format!(
        "{} - {} + {} - {} + \\cdots + {} - {}",
        render_unit_fraction_latex(&first_latex),
        render_unit_fraction_latex(&second_latex),
        render_unit_fraction_latex(&second_latex),
        render_unit_fraction_latex(&third_latex),
        render_unit_fraction_latex(&penultimate_latex),
        render_unit_fraction_latex(&last_latex),
    );

    vec![
        formula_substep(
            "Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)",
            &human_expr(ctx, call.term),
            &decomposed_plain,
            &decomposed_latex,
            &latex_expr(ctx, call.term),
        ),
        formula_substep(
            "La suma telescópica cancela los términos intermedios",
            &telescoping_series_plain,
            &human_expr(ctx, after),
            &telescoping_series_latex,
            &latex_expr(ctx, after),
        ),
    ]
}

fn detect_consecutive_telescoping_sum_base(
    ctx: &Context,
    factor1: ExprId,
    factor2: ExprId,
    var: &str,
) -> Option<ExprId> {
    for (base_candidate, other_factor) in [(factor1, factor2), (factor2, factor1)] {
        let Some(base) = extract_unit_shifted_base(ctx, base_candidate, var) else {
            continue;
        };
        let mut temp_ctx = ctx.clone();
        let shifted = shifted_expr(&mut temp_ctx, base, 1);
        if compare_expr(&temp_ctx, other_factor, shifted) == std::cmp::Ordering::Equal {
            return Some(base);
        }
    }
    None
}

fn detect_affine_consecutive_telescoping_sum_pattern(
    ctx: &Context,
    factor1: ExprId,
    factor2: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId, ExprId)> {
    for (base_candidate, other_factor) in [(factor1, factor2), (factor2, factor1)] {
        let coeff = extract_non_unit_affine_var_coeff(ctx, base_candidate, var)?;
        if additive_gap_relation_holds(ctx, base_candidate, coeff, other_factor) {
            return Some((base_candidate, other_factor, coeff));
        }
    }
    None
}

fn extract_non_unit_affine_var_coeff(ctx: &Context, expr: ExprId, var: &str) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            if !contains_named_var(ctx, *left, var) {
                return extract_non_unit_affine_var_coeff(ctx, *right, var);
            }
            if !contains_named_var(ctx, *right, var) {
                return extract_non_unit_affine_var_coeff(ctx, *left, var);
            }
            None
        }
        Expr::Sub(left, right) => {
            if contains_named_var(ctx, *right, var) {
                return None;
            }
            extract_non_unit_affine_var_coeff(ctx, *left, var)
        }
        _ => extract_non_unit_affine_linear_coeff(ctx, expr, var),
    }
}

fn extract_non_unit_affine_linear_coeff(ctx: &Context, expr: ExprId, var: &str) -> Option<ExprId> {
    if is_named_var(ctx, expr, var) {
        return None;
    }

    let factors = expr_nary::mul_leaves(ctx, expr);
    let mut saw_var = false;
    let mut coeff_factors = Vec::new();

    for factor in factors {
        if is_named_var(ctx, factor, var) {
            if saw_var {
                return None;
            }
            saw_var = true;
        } else if contains_named_var(ctx, factor, var) {
            return None;
        } else {
            coeff_factors.push(factor);
        }
    }

    if !saw_var {
        return None;
    }

    match coeff_factors.as_slice() {
        [] => None,
        [single] => Some(*single),
        _ => None,
    }
}

fn contains_named_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match ctx.get(current) {
            Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var => return true,
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push(*left);
                stack.push(*right);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

fn is_named_var(ctx: &Context, expr: ExprId, var: &str) -> bool {
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var)
}

fn generate_cos_product_telescoping_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let (base_multiplier, base_factors, factor_count, expands_morrie) =
        if let Some((base_multiplier, base_factors, factor_count)) =
            cos_product_telescoping_base_and_len(ctx, before)
        {
            (base_multiplier, base_factors, factor_count, false)
        } else if let Some((base_multiplier, base_factors, factor_count)) =
            cos_product_telescoping_base_and_len(ctx, after)
        {
            (base_multiplier, base_factors, factor_count, true)
        } else {
            return Vec::new();
        };

    let power = 1i64 << factor_count;
    let product_plain = (0..factor_count)
        .map(|idx| {
            let coeff = 1i64 << idx;
            if coeff == 1 {
                "cos(u)".to_string()
            } else {
                format!("cos({coeff}u)")
            }
        })
        .collect::<Vec<_>>()
        .join(" · ");
    let quotient_plain = format!("sin({power}u) / ({power} · sin(u))");
    let product_latex = (0..factor_count)
        .map(|idx| {
            let coeff = 1i64 << idx;
            if coeff == 1 {
                "\\cos(u)".to_string()
            } else {
                format!("\\cos({coeff}u)")
            }
        })
        .collect::<Vec<_>>()
        .join("\\cdot ");
    let quotient_latex = format!("\\frac{{\\sin({power}u)}}{{{power}\\cdot \\sin(u)}}");
    let (base_u_plain, base_u_latex) = render_factor_basis(ctx, &base_factors);
    let (u_plain, u_latex) = if base_multiplier == 1 {
        (base_u_plain, base_u_latex)
    } else {
        (
            format!("{base_multiplier} · {base_u_plain}"),
            format!("{base_multiplier}\\cdot {base_u_latex}"),
        )
    };
    let tautological_u = u_plain == "u" && u_latex == "u";
    let title = if expands_morrie {
        ("Expandir la ley de Morrie",)
    } else {
        ("Usar el telescopado de cosenos",)
    }
    .0;

    let _ = (
        product_plain,
        quotient_plain,
        product_latex,
        quotient_latex,
        u_plain,
        u_latex,
        tautological_u,
    );
    vec![formula_substep(
        title,
        &human_expr(ctx, before),
        &human_expr(ctx, after),
        &latex_expr(ctx, before),
        &latex_expr(ctx, after),
    )]
}

fn generate_dirichlet_kernel_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let (n, base_multiplier, base_factors, expands_kernel) = if let Some((
        n,
        base_multiplier,
        base_factors,
    )) =
        dirichlet_kernel_base_and_n(ctx, before)
    {
        (n, base_multiplier, base_factors, false)
    } else if let Some((n, base_multiplier, base_factors)) = dirichlet_kernel_base_and_n(ctx, after)
    {
        (n, base_multiplier, base_factors, true)
    } else {
        return Vec::new();
    };

    let (base_u_plain, base_u_latex) = render_factor_basis(ctx, &base_factors);
    let (u_plain, u_latex) = if base_multiplier == 1 {
        (base_u_plain, base_u_latex)
    } else {
        (
            format!("{base_multiplier} · {base_u_plain}"),
            format!("{base_multiplier}\\cdot {base_u_latex}"),
        )
    };
    let tautological_u = u_plain == "u" && u_latex == "u";
    let n_plain = n.to_string();
    let title = if expands_kernel {
        ("Expandir el núcleo de Dirichlet",)
    } else {
        ("Usar el núcleo de Dirichlet",)
    }
    .0;

    let _ = (n_plain, u_plain, u_latex, tautological_u);
    vec![formula_substep(
        title,
        &human_expr(ctx, before),
        &human_expr(ctx, after),
        &latex_expr(ctx, before),
        &latex_expr(ctx, after),
    )]
}

fn dirichlet_kernel_base_and_n(
    ctx: &Context,
    before: ExprId,
) -> Option<(usize, usize, Vec<ExprId>)> {
    let view = AddView::from_expr(ctx, before);
    let mut has_one = false;
    let mut multiples = Vec::new();
    let mut base_factors: Option<Vec<ExprId>> = None;

    for &(term, sign) in &view.terms {
        if sign != Sign::Pos {
            return None;
        }

        if is_one(ctx, term) {
            has_one = true;
            continue;
        }

        let (multiple, candidate_base) = dirichlet_cosine_multiple(ctx, term)?;
        if let Some(existing_base) = &base_factors {
            if !same_factor_basis(ctx, existing_base, &candidate_base) {
                return None;
            }
        } else {
            base_factors = Some(candidate_base);
        }
        multiples.push(multiple);
    }

    if !has_one || multiples.is_empty() {
        return None;
    }

    multiples.sort_unstable();
    multiples.sort_unstable();
    let n = multiples.len();
    let base_multiplier = multiples.iter().copied().reduce(gcd_usize)?;
    if multiples
        .iter()
        .enumerate()
        .any(|(idx, multiple)| *multiple != (idx + 1) * base_multiplier)
    {
        return None;
    }

    Some((n, base_multiplier, base_factors?))
}

fn dirichlet_cosine_multiple(ctx: &Context, expr: ExprId) -> Option<(usize, Vec<ExprId>)> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };

    let cosine = if is_integer_literal(ctx, *left, 2) {
        *right
    } else if is_integer_literal(ctx, *right, 2) {
        *left
    } else {
        return None;
    };

    let Expr::Function(fn_id, args) = ctx.get(cosine) else {
        return None;
    };
    if ctx.sym_name(*fn_id) != "cos" || args.len() != 1 {
        return None;
    }

    let (multiple, base_u) = extract_i64_multiplier_and_base_factors(ctx, args[0]);
    if multiple <= 0 {
        return None;
    }
    Some((multiple as usize, base_u.into_vec()))
}

fn gcd_usize(a: usize, b: usize) -> usize {
    let mut a = a;
    let mut b = b;
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn telescoping_fraction_base_and_gap(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<(ExprId, String, bool)> {
    let (num, den) = as_div(ctx, before)?;
    if !is_one(ctx, num) {
        return None;
    }

    let (u, u_plus_gap, gap_expr) = extract_telescoping_fraction_split_pattern(ctx, after)?;
    if !matches_telescoping_fraction_denominator(ctx, den, u, u_plus_gap) {
        return None;
    }

    if let Some(gap_expr) = gap_expr {
        if !additive_gap_relation_holds(ctx, u, gap_expr, u_plus_gap) {
            return None;
        }
        Some((u, human_expr(ctx, gap_expr), false))
    } else {
        if !unit_gap_relation_holds(ctx, u, u_plus_gap) {
            return None;
        }
        Some((u, "1".to_string(), true))
    }
}

fn matches_telescoping_fraction_denominator(
    ctx: &Context,
    denominator: ExprId,
    u: ExprId,
    u_plus_gap: ExprId,
) -> bool {
    let factors = expr_nary::mul_leaves(ctx, denominator);
    if factors.len() == 2 {
        let same_order = cas_ast::ordering::compare_expr(ctx, factors[0], u)
            == std::cmp::Ordering::Equal
            && cas_ast::ordering::compare_expr(ctx, factors[1], u_plus_gap)
                == std::cmp::Ordering::Equal;
        let swapped_order = cas_ast::ordering::compare_expr(ctx, factors[1], u)
            == std::cmp::Ordering::Equal
            && cas_ast::ordering::compare_expr(ctx, factors[0], u_plus_gap)
                == std::cmp::Ordering::Equal;
        if same_order || swapped_order {
            return true;
        }
    }

    // Also accept denominators that are algebraically the same product even when
    // they are still expanded, like x^2 + 3x + 2 instead of (x + 1)(x + 2).
    let mut temp_ctx = ctx.clone();
    let expected_product = temp_ctx.add_raw(Expr::Mul(u, u_plus_gap));
    poly_eq(&temp_ctx, denominator, expected_product)
}

fn cos_product_telescoping_base_and_len(
    ctx: &Context,
    expr: ExprId,
) -> Option<(i64, Vec<ExprId>, usize)> {
    let factors = expr_nary::mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    let mut cos_info = Vec::new();
    for &factor in &factors {
        let Expr::Function(fn_id, args) = ctx.get(factor) else {
            return None;
        };
        if ctx.builtin_of(*fn_id) != Some(BuiltinFn::Cos) || args.len() != 1 {
            return None;
        }
        let (multiplier, base_u) = extract_i64_multiplier_and_base_factors(ctx, args[0]);
        cos_info.push((multiplier, base_u.into_vec()));
    }

    let base_u = cos_info.first()?.1.clone();
    let mut multipliers = Vec::with_capacity(cos_info.len());
    for (multiplier, u) in cos_info {
        if !same_factor_basis(ctx, &u, &base_u) {
            return None;
        }
        multipliers.push(multiplier);
    }

    multipliers.sort_unstable();
    let base_multiplier = *multipliers.first()?;
    if base_multiplier <= 0 {
        return None;
    }

    for (idx, multiplier) in multipliers.iter().enumerate() {
        let expected = base_multiplier * (1i64 << idx);
        if *multiplier != expected {
            return None;
        }
    }

    Some((base_multiplier, base_u, multipliers.len()))
}

fn same_factor_basis(ctx: &Context, left: &[ExprId], right: &[ExprId]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(lhs, rhs)| compare_expr(ctx, *lhs, *rhs).is_eq())
}

fn render_factor_basis(ctx: &Context, factors: &[ExprId]) -> (String, String) {
    match factors {
        [] => {
            let mut temp_ctx = ctx.clone();
            let basis = build_balanced_mul(&mut temp_ctx, factors);
            render_temp_expr(&temp_ctx, basis)
        }
        [single] => render_temp_expr(ctx, *single),
        _ => {
            let plain = factors
                .iter()
                .map(|factor| render_factor_piece_plain(ctx, *factor))
                .collect::<Vec<_>>()
                .join(" · ");
            let latex = factors
                .iter()
                .map(|factor| render_factor_piece_latex(ctx, *factor))
                .collect::<Vec<_>>()
                .join("\\cdot ");
            (plain, latex)
        }
    }
}

fn render_factor_piece_plain(ctx: &Context, expr: ExprId) -> String {
    let (plain, _) = render_temp_expr(ctx, expr);
    match ctx.get(expr) {
        Expr::Add(_, _) | Expr::Sub(_, _) => format!("({plain})"),
        _ => plain,
    }
}

fn render_factor_piece_latex(ctx: &Context, expr: ExprId) -> String {
    let (_, latex) = render_temp_expr(ctx, expr);
    match ctx.get(expr) {
        Expr::Add(_, _) | Expr::Sub(_, _) => format!("\\left({latex}\\right)"),
        _ => latex,
    }
}

fn additive_signature(
    ctx: &Context,
    expr: ExprId,
) -> (Vec<(Vec<ExprId>, BigRational)>, BigRational) {
    let mut terms: Vec<(Vec<ExprId>, BigRational)> = Vec::new();
    let mut constant = BigRational::from_integer(0.into());

    for (term, sign) in AddView::from_expr(ctx, expr).terms {
        if let Some(value) = as_rational_const(ctx, term, 4) {
            match sign {
                Sign::Pos => constant += value,
                Sign::Neg => constant -= value,
            }
        } else {
            let (basis, coeff) = scaled_term_signature(ctx, term);
            let signed_coeff = match sign {
                Sign::Pos => coeff,
                Sign::Neg => -coeff,
            };
            if let Some((_, existing_coeff)) = terms
                .iter_mut()
                .find(|(existing_basis, _)| same_signature_basis(ctx, existing_basis, &basis))
            {
                *existing_coeff += signed_coeff;
            } else {
                terms.push((basis, signed_coeff));
            }
        }
    }

    terms.retain(|(_, coeff)| *coeff != BigRational::from_integer(0.into()));
    sort_signature_terms(ctx, &mut terms);

    (terms, constant)
}

fn scaled_term_signature(ctx: &Context, expr: ExprId) -> (Vec<ExprId>, BigRational) {
    let factors = expr_nary::mul_leaves(ctx, expr);
    let mut numeric_coeff = BigRational::from_integer(1.into());
    let mut basis = Vec::new();

    for factor in factors {
        if let Some(value) = as_rational_const(ctx, factor, 4) {
            numeric_coeff *= value;
        } else {
            basis.push(factor);
        }
    }

    basis.sort_by(|left, right| cas_ast::ordering::compare_expr(ctx, *left, *right));
    if basis.is_empty() {
        basis.push(expr);
    }
    (basis, numeric_coeff)
}

fn same_signature_basis(ctx: &Context, left: &[ExprId], right: &[ExprId]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(l, r)| cas_ast::ordering::compare_expr(ctx, *l, *r) == std::cmp::Ordering::Equal)
}

fn sort_signature_terms(ctx: &Context, terms: &mut [(Vec<ExprId>, BigRational)]) {
    terms.sort_by(|(left_basis, _), (right_basis, _)| {
        compare_signature_basis(ctx, left_basis, right_basis)
    });
}

fn compare_signature_basis(ctx: &Context, left: &[ExprId], right: &[ExprId]) -> std::cmp::Ordering {
    for (l, r) in left.iter().zip(right.iter()) {
        let ord = cas_ast::ordering::compare_expr(ctx, *l, *r);
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    left.len().cmp(&right.len())
}

fn extract_telescoping_fraction_split_pattern(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, Option<ExprId>)> {
    if let Some((numerator, denominator)) = as_div(ctx, expr) {
        if let Some((u, u_plus_gap)) = extract_telescoping_fraction_core(ctx, numerator) {
            return Some((u, u_plus_gap, Some(denominator)));
        }
    }

    let factors = expr_nary::mul_leaves(ctx, expr);
    for core_index in 0..factors.len() {
        let Some((u, u_plus_gap)) = extract_telescoping_fraction_core(ctx, factors[core_index])
        else {
            continue;
        };

        let residual = factors
            .iter()
            .enumerate()
            .filter_map(|(index, factor)| (index != core_index).then_some(*factor))
            .collect::<Vec<_>>();
        match residual.as_slice() {
            [] => return Some((u, u_plus_gap, None)),
            [single] => {
                if let Some(denominator) = extract_unit_reciprocal_denominator(ctx, *single) {
                    return Some((u, u_plus_gap, Some(denominator)));
                }
            }
            _ => {}
        }
    }

    None
}

fn extract_telescoping_fraction_core(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return None;
    }

    let mut saw_u = None;
    let mut saw_u_plus_gap = None;
    for (term, sign) in terms {
        match sign {
            Sign::Pos => saw_u = Some(extract_unit_fraction_denominator(ctx, term)?),
            Sign::Neg => saw_u_plus_gap = Some(extract_unit_fraction_denominator(ctx, term)?),
        }
    }

    Some((saw_u?, saw_u_plus_gap?))
}

fn extract_unit_fraction_denominator(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let (num, den) = as_div(ctx, expr)?;
    is_one(ctx, num).then_some(den)
}

fn extract_unit_reciprocal_denominator(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let (num, den) = as_div(ctx, expr)?;
    is_one(ctx, num).then_some(den)
}

fn additive_gap_relation_holds(ctx: &Context, base: ExprId, gap: ExprId, target: ExprId) -> bool {
    let (base_terms, base_constant) = additive_signature(ctx, base);
    let (gap_terms, gap_constant) = additive_signature(ctx, gap);
    let (target_terms, target_constant) = additive_signature(ctx, target);

    let mut combined_terms = base_terms;
    for (basis, coeff) in gap_terms {
        if let Some((_, existing_coeff)) = combined_terms
            .iter_mut()
            .find(|(existing_basis, _)| same_signature_basis(ctx, existing_basis, &basis))
        {
            *existing_coeff += coeff.clone();
        } else {
            combined_terms.push((basis, coeff));
        }
    }
    combined_terms.retain(|(_, coeff)| *coeff != BigRational::from_integer(0.into()));
    sort_signature_terms(ctx, &mut combined_terms);

    if combined_terms == target_terms
        && base_constant.clone() + gap_constant.clone() == target_constant
    {
        return true;
    }

    let mut temp_ctx = ctx.clone();
    let combined = temp_ctx.add_raw(Expr::Add(base, gap));
    poly_eq(&temp_ctx, combined, target)
}

fn unit_gap_relation_holds(ctx: &Context, base: ExprId, target: ExprId) -> bool {
    let (base_terms, base_constant) = additive_signature(ctx, base);
    let (target_terms, target_constant) = additive_signature(ctx, target);
    base_terms == target_terms
        && base_constant + BigRational::from_integer(1.into()) == target_constant
}

fn generate_log_power_contraction_substep(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<SubStep> {
    if matches_even_abs_ln_power_contraction(ctx, before, after) {
        return Some(formula_substep(
            "Usar n · ln(|u|) = ln(u^n) cuando n es par",
            "n · ln(|u|)",
            "ln(u^n)",
            "n\\cdot \\ln(|u|)",
            "\\ln(u^n)",
        ));
    }

    if matches_general_log_power_contraction(ctx, before, after) {
        return Some(formula_substep(
            "Usar n · log_b(u) = log_b(u^n)",
            "n · log_b(u)",
            "log_b(u^n)",
            "n\\cdot \\log_b(u)",
            "\\log_b(u^n)",
        ));
    }

    None
}

fn generate_log_change_of_base_chain_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<Vec<SubStep>> {
    if let Some(chain_len) = log_change_of_base_chain_contraction_len(ctx, before, after) {
        if chain_len == 2 {
            return Some(vec![formula_substep(
                "Usar log_b(a) · log_a(c) = log_b(c)",
                "log_b(a) · log_a(c)",
                "log_b(c)",
                "\\log_b(a)\\cdot \\log_a(c)",
                "\\log_b(c)",
            )]);
        }

        return Some(vec![formula_substep(
            "Encadenar los cambios de base intermedios",
            "log_{u0}(u1) · log_{u1}(u2) · ... · log_{u_{n-1}}(u_n)",
            "log_{u0}(u_n)",
            "\\log_{u_0}(u_1)\\cdot \\log_{u_1}(u_2)\\cdots \\log_{u_{n-1}}(u_n)",
            "\\log_{u_0}(u_n)",
        )]);
    }

    if let Some(chain_len) = log_change_of_base_chain_expansion_len(ctx, before, after) {
        if chain_len == 2 {
            return Some(vec![formula_substep(
                "Usar log_b(c) = log_a(c) · log_b(a)",
                "log_b(c)",
                "log_a(c) · log_b(a)",
                "\\log_b(c)",
                "\\log_a(c)\\cdot \\log_b(a)",
            )]);
        }

        return Some(vec![formula_substep(
            "Desplegar un logaritmo en una cadena de cambios de base",
            "log_{u0}(u_n)",
            "log_{u0}(u1) · log_{u1}(u2) · ... · log_{u_{n-1}}(u_n)",
            "\\log_{u_0}(u_n)",
            "\\log_{u_0}(u_1)\\cdot \\log_{u_1}(u_2)\\cdots \\log_{u_{n-1}}(u_n)",
        )]);
    }

    None
}

fn log_change_of_base_chain_contraction_len(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<usize> {
    let Some((Some(target_base), target_arg)) = general_log_base_and_arg(ctx, after) else {
        return None;
    };

    let factors = expr_nary::mul_leaves(ctx, before);
    log_change_of_base_chain_len(ctx, &factors, target_base, target_arg)
}

fn log_change_of_base_chain_expansion_len(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<usize> {
    let Some((Some(target_base), target_arg)) = general_log_base_and_arg(ctx, before) else {
        return None;
    };

    let factors = expr_nary::mul_leaves(ctx, after);
    log_change_of_base_chain_len(ctx, &factors, target_base, target_arg)
}

fn log_change_of_base_chain_len(
    ctx: &Context,
    factors: &[ExprId],
    target_base: ExprId,
    target_arg: ExprId,
) -> Option<usize> {
    if factors.len() < 2 {
        return None;
    }

    let chain_nodes: Option<Vec<(ExprId, ExprId)>> = factors
        .iter()
        .map(|factor| {
            let (Some(base), arg) = general_log_base_and_arg(ctx, *factor)? else {
                return None;
            };
            Some((base, arg))
        })
        .collect();
    let chain_nodes = chain_nodes?;

    for start in 0..chain_nodes.len() {
        if cas_ast::ordering::compare_expr(ctx, chain_nodes[start].0, target_base)
            != std::cmp::Ordering::Equal
        {
            continue;
        }

        let mut used = vec![false; chain_nodes.len()];
        used[start] = true;
        if log_change_of_base_chain_dfs(ctx, &chain_nodes, start, target_arg, 1, &mut used) {
            return Some(chain_nodes.len());
        }
    }

    None
}

fn log_change_of_base_chain_dfs(
    ctx: &Context,
    nodes: &[(ExprId, ExprId)],
    current: usize,
    target_arg: ExprId,
    depth: usize,
    used: &mut [bool],
) -> bool {
    if depth == nodes.len() {
        return cas_ast::ordering::compare_expr(ctx, nodes[current].1, target_arg)
            == std::cmp::Ordering::Equal;
    }

    let current_arg = nodes[current].1;
    for next in 0..nodes.len() {
        if used[next] {
            continue;
        }
        if cas_ast::ordering::compare_expr(ctx, current_arg, nodes[next].0)
            != std::cmp::Ordering::Equal
        {
            continue;
        }
        used[next] = true;
        if log_change_of_base_chain_dfs(ctx, nodes, next, target_arg, depth + 1, used) {
            return true;
        }
        used[next] = false;
    }

    false
}

fn matches_even_abs_ln_power_contraction(ctx: &Context, before: ExprId, after: ExprId) -> bool {
    let Some((coeff, log_expr)) = scaled_log_term(ctx, before) else {
        return false;
    };
    if coeff <= 0.into() || (&coeff % 2) != 0.into() {
        return false;
    }

    let Expr::Function(fn_id, args) = ctx.get(log_expr) else {
        return false;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Ln) || args.len() != 1 {
        return false;
    }
    let Some(inner) = abs_argument(ctx, args[0]) else {
        return false;
    };

    let Expr::Function(after_fn, after_args) = ctx.get(after) else {
        return false;
    };
    if !ctx.is_builtin(*after_fn, BuiltinFn::Ln) || after_args.len() != 1 {
        return false;
    }

    let Expr::Pow(after_base, after_exp) = ctx.get(after_args[0]) else {
        return false;
    };
    let Some(exponent) = positive_integer_literal_value(ctx, *after_exp) else {
        return false;
    };

    exponent == coeff
        && cas_ast::ordering::compare_expr(ctx, inner, *after_base) == std::cmp::Ordering::Equal
}

fn matches_general_log_power_contraction(ctx: &Context, before: ExprId, after: ExprId) -> bool {
    let Some((coeff, log_expr)) = scaled_log_term(ctx, before) else {
        return false;
    };
    if coeff <= 0.into() {
        return false;
    }

    let Some((before_base, before_arg)) = general_log_base_and_arg(ctx, log_expr) else {
        return false;
    };
    let Some((after_base, after_arg)) = general_log_base_and_arg(ctx, after) else {
        return false;
    };
    if before_base != after_base {
        return false;
    }

    let Expr::Pow(after_pow_base, after_exp) = ctx.get(after_arg) else {
        return false;
    };
    let Some(exponent) = positive_integer_literal_value(ctx, *after_exp) else {
        return false;
    };

    exponent == coeff
        && cas_ast::ordering::compare_expr(ctx, before_arg, *after_pow_base)
            == std::cmp::Ordering::Equal
}

fn generate_evaluate_logarithms_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    match step.description.as_str() {
        "log(b, x^y) = y * log(b, x)" => vec![concrete_expr_substep(
            ctx,
            "Sacar el exponente fuera del logaritmo",
            before,
            after,
        )],
        _ => Vec::new(),
    }
}

fn generate_factor_perfect_square_log_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    vec![concrete_expr_substep(
        ctx,
        "Sacar un exponente par fuera del logaritmo",
        step.before_local().unwrap_or(step.before),
        step.after_local().unwrap_or(step.after),
    )]
}

fn generate_log_contraction_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some(substeps) = generate_log_change_of_base_chain_substeps(ctx, before, after) {
        return substeps;
    }

    let Some((title, before_display, after_display, before_latex, after_latex)) =
        log_formula_snippet(ctx, before, false)
    else {
        return Vec::new();
    };

    if before != after {
        return vec![concrete_expr_substep(ctx, title, before, after)];
    }

    vec![formula_substep(
        title,
        &before_display,
        &after_display,
        &before_latex,
        &after_latex,
    )]
}

fn generate_double_angle_expansion_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before_expr = step.before_local().unwrap_or(step.before);
    let after_expr = step.after_local().unwrap_or(step.after);
    let before_display = human_expr(ctx, before_expr);
    let after_display = human_expr(ctx, after_expr);

    let title = if before_display.contains("1 - 2")
        && before_display.contains("sin(")
        && after_display.contains("cos(2")
    {
        "Reconocer el patrón 1 - 2 · sin(u)^2 = cos(2u)"
    } else if step.description.contains("1 - 2·sin")
        || step.description.contains("2·cos")
        || step.description.contains("sine")
        || step.description.contains("cosine")
    {
        "Usar la identidad de ángulo doble"
    } else {
        return Vec::new();
    };

    vec![concrete_expr_substep(ctx, title, before_expr, after_expr)]
}

fn generate_sum_to_product_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let local_before = step.before_local().unwrap_or(step.before);

    let inferred_kind = match ctx.get(local_before) {
        Expr::Add(left, right) => match (
            extract_trig_function_name(ctx, *left),
            extract_trig_function_name(ctx, *right),
        ) {
            (Some("sin"), Some("sin")) => Some("sine sum"),
            (Some("cos"), Some("cos")) => Some("cosine sum"),
            _ => None,
        },
        Expr::Sub(left, right) => match (
            extract_trig_function_name(ctx, *left),
            extract_trig_function_name(ctx, *right),
        ) {
            (Some("sin"), Some("sin")) => Some("sine difference"),
            (Some("cos"), Some("cos")) => Some("cosine difference"),
            _ => None,
        },
        _ => None,
    };

    let kind = inferred_kind
        .or_else(|| infer_sum_to_product_kind_from_display(ctx, local_before))
        .or_else(|| infer_sum_to_product_kind_from_description(step.description.as_str()));
    let title = match kind {
        Some("sine sum") => "Usar sin(A) + sin(B) = 2 · sin((A+B)/2) · cos((A-B)/2)",
        Some("sine difference") => "Usar sin(A) - sin(B) = 2 · cos((A+B)/2) · sin((A-B)/2)",
        Some("cosine sum") => "Usar cos(A) + cos(B) = 2 · cos((A+B)/2) · cos((A-B)/2)",
        Some("cosine difference") => "Usar cos(A) - cos(B) = -2 · sin((A+B)/2) · sin((A-B)/2)",
        _ => return Vec::new(),
    };

    vec![concrete_expr_substep(
        ctx,
        title,
        local_before,
        step.after_local().unwrap_or(step.after),
    )]
}

fn generate_hyperbolic_angle_sum_diff_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let local_before = step.before_local().unwrap_or(step.before);
    let Expr::Function(fn_id, args) = ctx.get(local_before) else {
        return Vec::new();
    };
    if args.len() != 1 {
        return Vec::new();
    }
    let (left, right, is_sum) = match ctx.get(args[0]) {
        Expr::Add(left, right) => (*left, *right, true),
        Expr::Sub(left, right) => (*left, *right, false),
        _ => return Vec::new(),
    };
    let _ = (left, right);

    let title = match (ctx.builtin_of(*fn_id), is_sum) {
        (Some(BuiltinFn::Sinh), true) => "Usar sinh(A+B) = sinh(A) · cosh(B) + cosh(A) · sinh(B)",
        (Some(BuiltinFn::Sinh), false) => "Usar sinh(A-B) = sinh(A) · cosh(B) - cosh(A) · sinh(B)",
        (Some(BuiltinFn::Cosh), true) => "Usar cosh(A+B) = cosh(A) · cosh(B) + sinh(A) · sinh(B)",
        (Some(BuiltinFn::Cosh), false) => "Usar cosh(A-B) = cosh(A) · cosh(B) - sinh(A) · sinh(B)",
        _ => return Vec::new(),
    };

    vec![concrete_expr_substep(
        ctx,
        title,
        local_before,
        step.after_local().unwrap_or(step.after),
    )]
}

fn extract_trig_function_name(ctx: &Context, expr: ExprId) -> Option<&str> {
    let Expr::Function(name, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    if ctx.is_builtin(*name, BuiltinFn::Sin) {
        Some("sin")
    } else if ctx.is_builtin(*name, BuiltinFn::Cos) {
        Some("cos")
    } else {
        None
    }
}

fn infer_sum_to_product_kind_from_display(ctx: &Context, expr: ExprId) -> Option<&str> {
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

fn infer_sum_to_product_kind_from_description(description: &str) -> Option<&'static str> {
    match description {
        "Expand sine sum to product" => Some("sine sum"),
        "Expand sine difference to product" => Some("sine difference"),
        "Expand cosine sum to product" => Some("cosine sum"),
        "Expand cosine difference to product" => Some("cosine difference"),
        _ => None,
    }
}

fn generate_product_to_sum_substeps(step: &Step) -> Vec<SubStep> {
    let _ = step;
    Vec::new()
}

fn generate_double_angle_contraction_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    if step.rule_name == "Double Angle Contraction"
        || step.description.contains("double-angle")
        || step.description.contains("Double Angle")
    {
        return vec![concrete_expr_substep(
            ctx,
            "Reconocer el patrón 2 · sin(u) · cos(u) = sin(2u)",
            before,
            after,
        )];
    }
    Vec::new()
}

fn generate_half_angle_square_identity_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if step.rule_name == "Angle Consistency (Half-Angle)"
        || step.description.contains("Half-Angle Expansion")
    {
        let after = human_expr(ctx, step.after_local().unwrap_or(step.after)).replace(' ', "");
        if after.contains("2·cos(") || after.contains("2*cos(") {
            return vec![concrete_expr_substep(
                ctx,
                "Usar cos(2u) = 2 · cos(u)^2 - 1",
                step.before_local().unwrap_or(step.before),
                step.after_local().unwrap_or(step.after),
            )];
        }
        if after.contains("1-2·sin(") || after.contains("1-2*sin(") {
            return vec![concrete_expr_substep(
                ctx,
                "Usar cos(2u) = 1 - 2 · sin(u)^2",
                step.before_local().unwrap_or(step.before),
                step.after_local().unwrap_or(step.after),
            )];
        }
    }

    if step.description.contains("Expand sin²") {
        return vec![concrete_expr_substep(
            ctx,
            "Usar sin²(u) = (1 - cos(2u)) / 2",
            step.before_local().unwrap_or(step.before),
            step.after_local().unwrap_or(step.after),
        )];
    }

    if step.description.contains("Expand cos²") {
        return vec![concrete_expr_substep(
            ctx,
            "Usar cos²(u) = (1 + cos(2u)) / 2",
            step.before_local().unwrap_or(step.before),
            step.after_local().unwrap_or(step.after),
        )];
    }

    if step.description.contains("Recognize (1 - cos(2u))/2") {
        return vec![concrete_expr_substep(
            ctx,
            "Usar (1 - cos(2u)) / 2 = sin²(u)",
            step.before_local().unwrap_or(step.before),
            step.after_local().unwrap_or(step.after),
        )];
    }

    if step.description.contains("Recognize (1 + cos(2u))/2") {
        return vec![concrete_expr_substep(
            ctx,
            "Usar (1 + cos(2u)) / 2 = cos²(u)",
            step.before_local().unwrap_or(step.before),
            step.after_local().unwrap_or(step.after),
        )];
    }

    Vec::new()
}

fn generate_triple_angle_identity_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let local_before = step.before_local().unwrap_or(step.before);
    let local_after = step.after_local().unwrap_or(step.after);
    let Expr::Function(fn_id, args) = ctx.get(local_before) else {
        return Vec::new();
    };
    if args.len() != 1 {
        return Vec::new();
    }

    if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
        return vec![concrete_expr_substep(
            ctx,
            "Usar sin(3u) = 3 · sin(u) - 4 · sin(u)^3",
            local_before,
            local_after,
        )];
    }

    if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
        return vec![concrete_expr_substep(
            ctx,
            "Usar cos(3u) = 4 · cos(u)^3 - 3 · cos(u)",
            local_before,
            local_after,
        )];
    }

    Vec::new()
}

fn generate_half_angle_tangent_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if step.description.contains("half-angle tangent")
        || step.rule_name == "Half-Angle Tangent Identity"
    {
        let before = step.before_local().unwrap_or(step.before);
        let after = step.after_local().unwrap_or(step.after);
        let variant = half_angle_tangent_variant(ctx, before)
            .or_else(|| half_angle_tangent_variant(ctx, after));
        return match variant {
            Some(HalfAngleTangentVariant::OneMinusCosOverSin) => vec![concrete_expr_substep(
                ctx,
                "Usar (1 - cos(2u)) / sin(2u) = tan(u)",
                before,
                after,
            )],
            Some(HalfAngleTangentVariant::SinOverOnePlusCos) => vec![concrete_expr_substep(
                ctx,
                "Usar sin(2u) / (1 + cos(2u)) = tan(u)",
                before,
                after,
            )],
            None => vec![concrete_expr_substep(
                ctx,
                "Usar tan(u) = (1 - cos(2u)) / sin(2u)",
                before,
                after,
            )],
        };
    }
    Vec::new()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HalfAngleTangentVariant {
    OneMinusCosOverSin,
    SinOverOnePlusCos,
}

fn half_angle_tangent_variant(ctx: &Context, expr: ExprId) -> Option<HalfAngleTangentVariant> {
    let (num, den) = as_div(ctx, expr)?;

    if is_one_minus_cos_double(ctx, num) && is_sin_double(ctx, den) {
        return Some(HalfAngleTangentVariant::OneMinusCosOverSin);
    }
    if is_sin_double(ctx, num) && is_one_plus_cos_double(ctx, den) {
        return Some(HalfAngleTangentVariant::SinOverOnePlusCos);
    }

    None
}

fn is_sin_double(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return false;
    };
    ctx.is_builtin(*fn_id, BuiltinFn::Sin)
        && args.len() == 1
        && is_double_angle(ctx, args[0]).is_some()
}

fn is_one_minus_cos_double(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Sub(lhs, rhs) = ctx.get(expr) else {
        return false;
    };
    is_one(ctx, *lhs) && is_cos_double(ctx, *rhs)
}

fn is_one_plus_cos_double(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Add(lhs, rhs) => {
            (is_one(ctx, *lhs) && is_cos_double(ctx, *rhs))
                || (is_one(ctx, *rhs) && is_cos_double(ctx, *lhs))
        }
        _ => false,
    }
}

fn is_cos_double(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return false;
    };
    ctx.is_builtin(*fn_id, BuiltinFn::Cos)
        && args.len() == 1
        && is_double_angle(ctx, args[0]).is_some()
}

fn is_double_angle(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };
    if is_two_half_angle(ctx, *left) {
        return Some(*right);
    }
    if is_two_half_angle(ctx, *right) {
        return Some(*left);
    }
    None
}

fn is_two_half_angle(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_integer() && n.to_integer() == 2.into())
}

fn generate_pythagorean_high_power_factor_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let local_before = step.before_local().unwrap_or(step.before);
    let before = human_expr(ctx, local_before);

    if before.contains("sin(") {
        if let Some(substeps) = build_pythagorean_high_power_sine_substeps(ctx, local_before) {
            return substeps;
        }
    }

    if before.starts_with("4 · cos(") && before.contains("^3") {
        if let Some(substeps) = build_pythagorean_high_power_cos_substeps(ctx, local_before, false)
        {
            return substeps;
        }
    }

    if before.contains("cos(") && before.contains("^3") {
        if let Some(substeps) = build_pythagorean_high_power_cos_substeps(ctx, local_before, true) {
            return substeps;
        }
    }

    Vec::new()
}

fn build_pythagorean_high_power_sine_substeps(
    ctx: &Context,
    local_before: ExprId,
) -> Option<Vec<SubStep>> {
    let mut work = ctx.clone();
    let arg = first_trig_argument_with_builtin(&work, local_before, BuiltinFn::Sin)?;
    let one = work.num(1);
    let two = work.num(2);
    let three = work.num(3);
    let four = work.num(4);
    let sin_u = work.call_builtin(BuiltinFn::Sin, vec![arg]);
    let cos_u = work.call_builtin(BuiltinFn::Cos, vec![arg]);
    let sin_sq = work.add(Expr::Pow(sin_u, two));
    let cos_sq = work.add(Expr::Pow(cos_u, two));
    let sin_cubed = work.add(Expr::Pow(sin_u, three));
    let four_sin = work.add(Expr::Mul(four, sin_u));
    let _four_sin_cubed = work.add(Expr::Mul(four, sin_cubed));
    let one_minus_sin_sq = work.add(Expr::Sub(one, sin_sq));
    let factorized = work.add(Expr::Mul(four_sin, one_minus_sin_sq));
    let four_sin_again = work.add(Expr::Mul(four, sin_u));
    let pythagorean = work.add(Expr::Mul(four_sin_again, cos_sq));
    let double_arg = work.add(Expr::Mul(two, arg));
    let sin_2u = work.call_builtin(BuiltinFn::Sin, vec![double_arg]);
    let two_sin_2u = work.add(Expr::Mul(two, sin_2u));
    let final_expr = work.add(Expr::Mul(two_sin_2u, cos_u));

    Some(vec![
        mixed_ctx_substep(
            "Sacar factor común 4 · sin(u)",
            ctx,
            local_before,
            &work,
            factorized,
        ),
        temp_ctx_substep(
            "Usar 1 - sin(u)^2 = cos(u)^2",
            &work,
            factorized,
            pythagorean,
        ),
        temp_ctx_substep(
            "Usar 2 · sin(u) · cos(u) = sin(2u)",
            &work,
            pythagorean,
            final_expr,
        ),
    ])
}

fn build_pythagorean_high_power_cos_substeps(
    ctx: &Context,
    local_before: ExprId,
    negated: bool,
) -> Option<Vec<SubStep>> {
    let mut work = ctx.clone();
    let arg = first_trig_argument_with_builtin(&work, local_before, BuiltinFn::Cos)?;
    let one = work.num(1);
    let two = work.num(2);
    let four = work.num(4);
    let neg_four = work.num(-4);
    let cos_u = work.call_builtin(BuiltinFn::Cos, vec![arg]);
    let sin_u = work.call_builtin(BuiltinFn::Sin, vec![arg]);
    let cos_sq = work.add(Expr::Pow(cos_u, two));
    let sin_sq = work.add(Expr::Pow(sin_u, two));
    let lead_coeff = if negated { neg_four } else { four };
    let lead_cos = work.add(Expr::Mul(lead_coeff, cos_u));
    let one_minus_cos_sq = work.add(Expr::Sub(one, cos_sq));
    let factorized = work.add(Expr::Mul(lead_cos, one_minus_cos_sq));
    let lead_cos_again = work.add(Expr::Mul(lead_coeff, cos_u));
    let pythagorean = work.add(Expr::Mul(lead_cos_again, sin_sq));
    let double_arg = work.add(Expr::Mul(two, arg));
    let sin_2u = work.call_builtin(BuiltinFn::Sin, vec![double_arg]);
    let final_coeff = if negated { work.num(-2) } else { two };
    let final_prefix = work.add(Expr::Mul(final_coeff, sin_2u));
    let final_expr = work.add(Expr::Mul(final_prefix, sin_u));

    Some(vec![
        mixed_ctx_substep(
            if negated {
                "Sacar factor común -4 · cos(u)"
            } else {
                "Sacar factor común 4 · cos(u)"
            },
            ctx,
            local_before,
            &work,
            factorized,
        ),
        temp_ctx_substep(
            "Usar 1 - cos(u)^2 = sin(u)^2",
            &work,
            factorized,
            pythagorean,
        ),
        temp_ctx_substep(
            "Usar 2 · sin(u) · cos(u) = sin(2u)",
            &work,
            pythagorean,
            final_expr,
        ),
    ])
}

fn first_trig_argument_with_builtin(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 && ctx.is_builtin(*fn_id, builtin) => {
            Some(args[0])
        }
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            first_trig_argument_with_builtin(ctx, *left, builtin)
                .or_else(|| first_trig_argument_with_builtin(ctx, *right, builtin))
        }
        Expr::Div(left, right) | Expr::Pow(left, right) => {
            first_trig_argument_with_builtin(ctx, *left, builtin)
                .or_else(|| first_trig_argument_with_builtin(ctx, *right, builtin))
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            first_trig_argument_with_builtin(ctx, *inner, builtin)
        }
        Expr::Function(_, args) => args
            .iter()
            .find_map(|arg| first_trig_argument_with_builtin(ctx, *arg, builtin)),
        Expr::Matrix { data, .. } => data
            .iter()
            .find_map(|item| first_trig_argument_with_builtin(ctx, *item, builtin)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => None,
    }
}

fn temp_ctx_substep(
    title: impl Into<String>,
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> SubStep {
    SubStep::new(title, human_expr(ctx, before), human_expr(ctx, after))
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(latex_expr(ctx, after))
}

fn mixed_ctx_substep(
    title: impl Into<String>,
    before_ctx: &Context,
    before: ExprId,
    after_ctx: &Context,
    after: ExprId,
) -> SubStep {
    SubStep::new(
        title,
        human_expr(before_ctx, before),
        human_expr(after_ctx, after),
    )
    .with_before_latex(latex_expr(before_ctx, before))
    .with_after_latex(latex_expr(after_ctx, after))
}

fn generate_cos_2x_additive_contraction_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let before_display = human_expr(ctx, before);

    if before_display.contains("sin(") {
        return vec![concrete_expr_substep(
            ctx,
            "Reconocer el patrón 1 - 2 · sin(u)^2 = cos(2u)",
            before,
            after,
        )];
    }

    if before_display.contains("cos(") {
        return vec![concrete_expr_substep(
            ctx,
            "Reconocer el patrón 2 · cos(u)^2 - 1 = cos(2u)",
            before,
            after,
        )];
    }

    Vec::new()
}

fn generate_sec_csc_squared_expansion_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    match step.description.as_str() {
        "Expand sec²(u) as 1 + tan(u)^2" => vec![concrete_expr_substep(
            ctx,
            "Usar sec²(u) = 1 + tan²(u)",
            before,
            after,
        )],
        "Expand csc²(u) as 1 + cot(u)^2" => vec![concrete_expr_substep(
            ctx,
            "Usar csc²(u) = 1 + cot²(u)",
            before,
            after,
        )],
        _ => Vec::new(),
    }
}

fn generate_sec_csc_squared_contraction_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    match step.description.as_str() {
        "Recognize 1 + tan²(u) as sec²(u)" => vec![concrete_expr_substep(
            ctx,
            "Usar 1 + tan²(u) = sec²(u)",
            before,
            after,
        )],
        "Recognize 1 + cot²(u) as csc²(u)" => vec![concrete_expr_substep(
            ctx,
            "Usar 1 + cot²(u) = csc²(u)",
            before,
            after,
        )],
        _ => Vec::new(),
    }
}

fn generate_pythagorean_factor_form_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    match step.description.as_str() {
        desc if desc.starts_with("1 - sin²") => vec![concrete_expr_substep(
            ctx,
            "Usar 1 - sin²(u) = cos²(u)",
            before,
            after,
        )],
        desc if desc.starts_with("1 - cos²") => vec![concrete_expr_substep(
            ctx,
            "Usar 1 - cos²(u) = sin²(u)",
            before,
            after,
        )],
        _ => Vec::new(),
    }
}

fn generate_pythagorean_chain_identity_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let after = step.after_local().unwrap_or(step.after);
    if !is_one(ctx, after) {
        return Vec::new();
    }

    let global_after = step.global_after.unwrap_or(step.after);
    let after_display = display_expr(ctx, global_after);
    if after_display.contains("sec(") && after_display.contains("csc(") {
        let mut work = ctx.clone();
        let mut out = Vec::new();
        for factor in collect_mul_chain_factors_readonly(&work, global_after) {
            let Some((title, reciprocal_before)) =
                reciprocal_rewrite_substep_for_factor(&mut work, factor)
            else {
                continue;
            };
            out.push(
                SubStep::new(
                    title,
                    display_expr(&work, reciprocal_before),
                    display_expr(&work, factor),
                )
                .with_before_latex(latex_expr(&work, reciprocal_before))
                .with_after_latex(latex_expr(&work, factor)),
            );
        }
        if !out.is_empty() {
            return out;
        }
    }

    Vec::new()
}

fn reciprocal_rewrite_substep_for_factor(
    ctx: &mut Context,
    factor: ExprId,
) -> Option<(String, ExprId)> {
    let (builtin, arg) = match ctx.get(factor) {
        Expr::Function(fn_id, args) if args.len() == 1 => (ctx.builtin_of(*fn_id), args[0]),
        _ => return None,
    };

    if matches!(builtin, Some(BuiltinFn::Sec)) {
        let cos_expr = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
        let one = ctx.num(1);
        let before = ctx.add(Expr::Div(one, cos_expr));
        return Some(("Usar 1 / cos(u) = sec(u)".to_string(), before));
    }

    if matches!(builtin, Some(BuiltinFn::Csc)) {
        let sin_expr = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
        let one = ctx.num(1);
        let before = ctx.add(Expr::Div(one, sin_expr));
        return Some(("Usar 1 / sin(u) = csc(u)".to_string(), before));
    }

    None
}

fn generate_trig_expansion_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    if step.description.contains("tangent to sine over cosine") {
        return vec![concrete_expr_substep(
            ctx,
            "Usar tan(u) = sin(u) / cos(u)",
            before,
            after,
        )];
    }
    Vec::new()
}

fn generate_reciprocal_trig_identity_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    match step.description.as_str() {
        "Expand sec(u) as 1 / cos(u)" => vec![concrete_expr_substep(
            ctx,
            "Usar sec(u) = 1 / cos(u)",
            before,
            after,
        )],
        "Expand csc(u) as 1 / sin(u)" => vec![concrete_expr_substep(
            ctx,
            "Usar csc(u) = 1 / sin(u)",
            before,
            after,
        )],
        "Expand cot(u) as cos(u) / sin(u)" => vec![concrete_expr_substep(
            ctx,
            "Usar cot(u) = cos(u) / sin(u)",
            before,
            after,
        )],
        "Recognize 1 / cos(u) as sec(u)" => vec![concrete_expr_substep(
            ctx,
            "Usar 1 / cos(u) = sec(u)",
            before,
            after,
        )],
        "Recognize 1 / sin(u) as csc(u)" => vec![concrete_expr_substep(
            ctx,
            "Usar 1 / sin(u) = csc(u)",
            before,
            after,
        )],
        "Recognize cos(u) / sin(u) as cot(u)" => vec![concrete_expr_substep(
            ctx,
            "Usar cos(u) / sin(u) = cot(u)",
            before,
            after,
        )],
        _ => Vec::new(),
    }
}

fn generate_reciprocal_product_identity_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    vec![concrete_expr_substep(
        ctx,
        "Usar tan(u) · cot(u) = 1",
        before,
        after,
    )]
}

fn generate_reciprocal_pythagorean_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    match step.description.as_str() {
        "Recognize sec²(u) - tan²(u) = 1" => vec![concrete_expr_substep(
            ctx,
            "Usar sec²(u) - tan²(u) = 1",
            before,
            after,
        )],
        "Recognize csc²(u) - cot²(u) = 1" => vec![concrete_expr_substep(
            ctx,
            "Usar csc²(u) - cot²(u) = 1",
            before,
            after,
        )],
        _ => Vec::new(),
    }
}

fn generate_trig_quotient_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    if step.description.contains("tan") || step.rule_name == "Trig Quotient" {
        return vec![concrete_expr_substep(
            ctx,
            "Reconocer el patrón sin(u) / cos(u) = tan(u)",
            before,
            after,
        )];
    }
    Vec::new()
}

fn generate_cos_diff_sin_diff_quotient_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let local_before = step.before_local().unwrap_or(step.before);
    let local_after = step.after_local().unwrap_or(step.after);

    if is_tan_call(ctx, local_after) || is_tan_call(ctx, step.after) {
        return vec![
            formula_substep(
                "Cancelar el factor común del numerador y del denominador",
                "(k · sin(u)) / (k · cos(u))",
                "sin(u) / cos(u)",
                "\\frac{k\\cdot \\sin(u)}{k\\cdot \\cos(u)}",
                "\\frac{\\sin(u)}{\\cos(u)}",
            ),
            formula_substep(
                "Reconocer el patrón sin(u) / cos(u) = tan(u)",
                "sin(u) / cos(u)",
                "tan(u)",
                "\\frac{\\sin(u)}{\\cos(u)}",
                "\\tan(u)",
            ),
        ];
    }

    let before_div = as_div(ctx, local_before).or_else(|| as_div(ctx, step.before));
    let after_div = as_div(ctx, local_after).or_else(|| as_div(ctx, step.after));
    let (Some((before_num, before_den)), Some((after_num, after_den))) = (before_div, after_div)
    else {
        return Vec::new();
    };

    if before_den == after_den && before_num != after_num {
        return vec![formula_substep(
            "Usar cos(A) - cos(B) = 2 · sin((A+B)/2) · sin((B-A)/2)",
            "cos(A) - cos(B)",
            "2 · sin((A+B)/2) · sin((B-A)/2)",
            "\\cos(A) - \\cos(B)",
            "2\\cdot \\sin\\!\\left(\\frac{A+B}{2}\\right)\\cdot \\sin\\!\\left(\\frac{B-A}{2}\\right)",
        )];
    }

    if before_num == after_num && before_den != after_den {
        return vec![formula_substep(
            "Usar sin(B) - sin(A) = 2 · cos((A+B)/2) · sin((B-A)/2)",
            "sin(B) - sin(A)",
            "2 · cos((A+B)/2) · sin((B-A)/2)",
            "\\sin(B) - \\sin(A)",
            "2\\cdot \\cos\\!\\left(\\frac{A+B}{2}\\right)\\cdot \\sin\\!\\left(\\frac{B-A}{2}\\right)",
        )];
    }

    Vec::new()
}

fn generate_consecutive_factorial_ratio_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some((expanded, gap)) = build_consecutive_factorial_ratio_expansion(ctx, before) else {
        return vec![
            formula_substep(
                "Escribir el factorial superior como el siguiente número por el factorial anterior",
                "(k + 1)! / k!",
                "((k + 1) · k!) / k!",
                "\\frac{(k+1)!}{k!}",
                "\\frac{(k+1)\\cdot k!}{k!}",
            ),
            formula_substep(
                "Cancelar el factorial común",
                "((k + 1) · k!) / k!",
                "k + 1",
                "\\frac{(k+1)\\cdot k!}{k!}",
                "k + 1",
            ),
        ];
    };

    let mut work = ctx.clone();
    let expanded_in_work = rebuild_consecutive_factorial_ratio_expansion(&mut work, before)
        .map(|(expr, _)| expr)
        .unwrap_or(expanded);
    let (before_display, before_latex) = render_temp_expr(&work, before);
    let (expanded_display, expanded_latex) = render_temp_expr(&work, expanded_in_work);
    let (after_display, after_latex) = render_temp_expr(&work, after);

    let first_title = if gap == 1 {
        "Escribir el factorial superior como el siguiente número por el factorial anterior"
            .to_string()
    } else {
        "Expandir el factorial superior hasta llegar al factorial inferior".to_string()
    };

    vec![
        SubStep::new(first_title, before_display, expanded_display)
            .with_before_latex(before_latex)
            .with_after_latex(expanded_latex),
        SubStep::new(
            "Cancelar el factorial común",
            human_expr(&work, expanded_in_work),
            after_display,
        )
        .with_before_latex(latex_expr(&work, expanded_in_work))
        .with_after_latex(after_latex),
    ]
}

fn extract_factorial_call_arg_local(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && matches!(ctx.sym_name(*fn_id), "fact" | "factorial") =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

fn small_integer(ctx: &Context, expr: ExprId) -> Option<i64> {
    match ctx.get(expr) {
        Expr::Number(n) if n.is_integer() => n.to_integer().try_into().ok(),
        Expr::Neg(inner) => small_integer(ctx, *inner).map(|value| -value),
        _ => None,
    }
}

fn extract_additive_base_and_offset_local(ctx: &Context, expr: ExprId) -> Option<(ExprId, i64)> {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            if let Some(offset) = small_integer(ctx, *left) {
                return Some((*right, offset));
            }
            if let Some(offset) = small_integer(ctx, *right) {
                return Some((*left, offset));
            }
            None
        }
        Expr::Sub(left, right) => small_integer(ctx, *right).map(|offset| (*left, -offset)),
        _ => Some((expr, 0)),
    }
}

fn rebuild_expr_with_offset_local(ctx: &mut Context, base: ExprId, offset: i64) -> ExprId {
    if offset == 0 {
        return base;
    }

    let amount = ctx.num(offset.checked_abs().expect("factorial offset fits in i64"));
    if offset > 0 {
        ctx.add(Expr::Add(base, amount))
    } else {
        ctx.add(Expr::Sub(base, amount))
    }
}

fn rebuild_consecutive_factorial_ratio_expansion(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, i64)> {
    let (num, den) = as_div(ctx, expr)?;
    let num_arg = extract_factorial_call_arg_local(ctx, num)?;
    let den_arg = extract_factorial_call_arg_local(ctx, den)?;
    let (num_base, num_offset) = extract_additive_base_and_offset_local(ctx, num_arg)?;
    let (den_base, den_offset) = extract_additive_base_and_offset_local(ctx, den_arg)?;
    if compare_expr(ctx, num_base, den_base) != Ordering::Equal {
        return None;
    }

    let gap = num_offset - den_offset;
    if gap <= 0 {
        return None;
    }

    let mut descending_factors = Vec::with_capacity(gap as usize);
    for shift in (1..=gap).rev() {
        let factor_offset = den_offset + shift;
        let factor = if factor_offset == num_offset {
            num_arg
        } else {
            rebuild_expr_with_offset_local(ctx, den_base, factor_offset)
        };
        descending_factors.push(factor);
    }

    let leading = build_balanced_mul(ctx, &descending_factors);
    let expanded_num = ctx.add(Expr::Mul(leading, den));
    Some((ctx.add(Expr::Div(expanded_num, den)), gap))
}

fn build_consecutive_factorial_ratio_expansion(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, i64)> {
    let mut work = ctx.clone();
    rebuild_consecutive_factorial_ratio_expansion(&mut work, expr)
}

fn combine_like_terms_coeff_sum_plan(
    ctx: &Context,
    before: ExprId,
) -> Option<(Vec<BigRational>, String)> {
    let terms = AddView::from_expr(ctx, before).terms;
    if terms.len() < 2 {
        return None;
    }

    let mut coeffs = Vec::with_capacity(terms.len());
    let mut literal_factors_key: Option<Vec<ExprId>> = None;

    for (term, sign) in terms {
        let (coeff, literal_factors) = extract_signed_coeff_and_literal(ctx, term, sign)?;
        if literal_factors.is_empty() {
            return None;
        }
        match &literal_factors_key {
            Some(existing) if *existing != literal_factors => return None,
            Some(_) => {}
            None => literal_factors_key = Some(literal_factors),
        }
        coeffs.push(coeff);
    }

    let literal_factors = literal_factors_key?;
    let literal_display = display_literal_factors(ctx, &literal_factors);
    Some((coeffs, literal_display))
}

fn extract_signed_coeff_and_literal(
    ctx: &Context,
    term: ExprId,
    sign: Sign,
) -> Option<(BigRational, Vec<ExprId>)> {
    let mut coeff = if sign == Sign::Neg {
        BigRational::from_integer((-1).into())
    } else {
        BigRational::from_integer(1.into())
    };
    let mut literal_factors = Vec::new();

    for factor in expr_nary::mul_leaves(ctx, term) {
        if let Some(numeric) = try_as_fraction(ctx, factor) {
            coeff *= numeric;
        } else {
            literal_factors.push(factor);
        }
    }

    Some((coeff, literal_factors))
}

fn display_literal_factors(ctx: &Context, literal_factors: &[ExprId]) -> String {
    let mut parts = literal_factors
        .iter()
        .map(|factor| display_expr(ctx, *factor))
        .collect::<Vec<_>>();
    if parts.len() == 1 {
        return parts.remove(0);
    }
    cas_formatter::clean_display_string(&parts.join(" · "))
}

fn render_numeric_sum(coeffs: &[BigRational]) -> (String, String) {
    let mut temp_ctx = Context::new();
    let expr = build_numeric_sum_expr(&mut temp_ctx, coeffs);
    render_temp_expr(&temp_ctx, expr)
}

fn render_numeric_value(value: &BigRational) -> (String, String) {
    let mut temp_ctx = Context::new();
    let expr = temp_ctx.add(Expr::Number(value.clone()));
    render_temp_expr(&temp_ctx, expr)
}

fn build_numeric_sum_expr(ctx: &mut Context, coeffs: &[BigRational]) -> ExprId {
    let mut iter = coeffs.iter();
    let first = iter.next().expect("nonempty coefficient sum");
    let mut acc = build_signed_number_expr(ctx, first);

    for coeff in iter {
        let rhs = ctx.add(Expr::Number(coeff.abs()));
        acc = if coeff.is_negative() {
            ctx.add(Expr::Sub(acc, rhs))
        } else {
            ctx.add(Expr::Add(acc, rhs))
        };
    }

    acc
}

fn build_signed_number_expr(ctx: &mut Context, coeff: &BigRational) -> ExprId {
    if coeff.is_negative() {
        let abs_expr = ctx.add(Expr::Number(coeff.abs()));
        ctx.add(Expr::Neg(abs_expr))
    } else {
        ctx.add(Expr::Number(coeff.clone()))
    }
}

fn render_temp_expr(ctx: &Context, expr: ExprId) -> (String, String) {
    (
        cas_formatter::clean_display_string(&format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: ctx,
                id: expr
            }
        )),
        cas_formatter::LaTeXExpr {
            context: ctx,
            id: expr,
        }
        .to_latex(),
    )
}

fn shifted_expr_strings(ctx: &Context, base: ExprId, offset: i64) -> (String, String) {
    if offset == 0 {
        return render_temp_expr(ctx, base);
    }

    let mut temp_ctx = ctx.clone();
    let shifted = shifted_expr(&mut temp_ctx, base, offset);
    render_temp_expr(&temp_ctx, shifted)
}

fn shifted_expr(ctx: &mut Context, base: ExprId, offset: i64) -> ExprId {
    if offset == 0 {
        return base;
    }

    let offset_expr = ctx.num(offset.abs());
    if offset > 0 {
        ctx.add(Expr::Add(base, offset_expr))
    } else {
        ctx.add(Expr::Sub(base, offset_expr))
    }
}

fn render_fraction_plain(numerator: &str, denominator: &str) -> String {
    format!("({numerator}) / ({denominator})")
}

fn render_fraction_latex(numerator: &str, denominator: &str) -> String {
    format!("\\frac{{{numerator}}}{{{denominator}}}")
}

fn render_power2_plain(base: &str) -> String {
    format!("{base}^2")
}

fn render_power2_latex(base: &str) -> String {
    format!("\\left({base}\\right)^{{2}}")
}

fn render_square_difference_plain(base: &str) -> String {
    format!("({base} - 1) · ({base} + 1)")
}

fn render_square_difference_latex(base: &str) -> String {
    format!("\\left({base} - 1\\right)\\cdot \\left({base} + 1\\right)")
}

fn render_unit_fraction_plain(denominator: &str) -> String {
    render_fraction_plain("1", denominator)
}

fn render_unit_fraction_latex(denominator: &str) -> String {
    render_fraction_latex("1", denominator)
}

fn formula_substep(
    description: impl Into<String>,
    before_expr: &str,
    after_expr: &str,
    before_latex: &str,
    after_latex: &str,
) -> SubStep {
    SubStep::new(description, before_expr, after_expr)
        .with_before_latex(before_latex)
        .with_after_latex(after_latex)
}

fn concrete_expr_substep(
    ctx: &Context,
    description: impl Into<String>,
    before: ExprId,
    after: ExprId,
) -> SubStep {
    SubStep::new(
        description,
        display_expr(ctx, before),
        display_expr(ctx, after),
    )
    .with_before_latex(latex_expr(ctx, before))
    .with_after_latex(latex_expr(ctx, after))
}

fn difference_of_squares_bases(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let Expr::Sub(left, right) = ctx.get(expr) else {
        return None;
    };
    let left_base = squared_base(ctx, *left)?;
    let right_base = squared_base(ctx, *right)?;
    Some((left_base, right_base))
}

fn squared_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) if is_small_positive_integer(ctx, *exp, 2) => Some(*base),
        Expr::Number(n) if n.is_integer() => {
            let int = n.to_integer();
            if !int.is_negative() {
                Some(expr)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn binomial_square_terms(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, BinomialSquareKind)> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if !is_small_positive_integer(ctx, *exp, 2) {
        return None;
    }
    match ctx.get(*base) {
        Expr::Add(left, right) => Some((*left, *right, BinomialSquareKind::Sum)),
        Expr::Sub(left, right) => Some((*left, *right, BinomialSquareKind::Difference)),
        _ => None,
    }
}

fn binomial_power_terms(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, BinomialSquareKind, i64)> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let power = if is_small_positive_integer(ctx, *exp, 2) {
        2
    } else if is_small_positive_integer(ctx, *exp, 3) {
        3
    } else {
        return None;
    };
    match ctx.get(*base) {
        Expr::Add(left, right) => Some((*left, *right, BinomialSquareKind::Sum, power)),
        Expr::Sub(left, right) => Some((*left, *right, BinomialSquareKind::Difference, power)),
        _ => None,
    }
}

fn geometric_difference_factor_plan(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<(ExprId, i64)> {
    let Expr::Sub(lhs, rhs) = ctx.get(before) else {
        return None;
    };
    if !is_small_positive_integer(ctx, *rhs, 1) {
        return None;
    }
    let Expr::Pow(base, exp) = ctx.get(*lhs) else {
        return None;
    };
    let power = small_positive_integer_value(ctx, *exp)?;
    if power < 2 {
        return None;
    }

    let Expr::Mul(left, right) = ctx.get(after) else {
        return None;
    };
    let series = if is_base_minus_one(ctx, *left, *base) {
        *right
    } else if is_base_minus_one(ctx, *right, *base) {
        *left
    } else {
        return None;
    };

    let terms = AddView::from_expr(ctx, series).terms;
    if terms.len() != power as usize {
        return None;
    }

    let mut seen = BTreeMap::new();
    for (term, sign) in terms {
        if sign != Sign::Pos {
            return None;
        }
        let exponent = geometric_series_term_exponent(ctx, *base, term)?;
        seen.insert(exponent, ());
    }

    if seen.len() != power as usize {
        return None;
    }
    if !(0..power).all(|exp| seen.contains_key(&exp)) {
        return None;
    }

    Some((*base, power))
}

fn sophie_germain_expansion_plan(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<(ExprId, ExprId)> {
    let (a, b) = sophie_germain_terms(ctx, after)?;
    let factors = expr_nary::mul_leaves(ctx, before);
    if factors.len() != 2 {
        return None;
    }

    let has_minus_factor = factors
        .iter()
        .any(|factor| matches_sophie_germain_quadratic(ctx, *factor, a, b, Sign::Neg));
    let has_plus_factor = factors
        .iter()
        .any(|factor| matches_sophie_germain_quadratic(ctx, *factor, a, b, Sign::Pos));

    (has_minus_factor && has_plus_factor).then_some((a, b))
}

fn is_base_minus_one(ctx: &Context, expr: ExprId, base: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Sub(lhs, rhs) if *lhs == base && is_small_positive_integer(ctx, *rhs, 1))
}

fn geometric_series_term_exponent(ctx: &Context, base: ExprId, term: ExprId) -> Option<i64> {
    if is_small_positive_integer(ctx, term, 1) {
        return Some(0);
    }
    if term == base {
        return Some(1);
    }
    match ctx.get(term) {
        Expr::Pow(pow_base, exponent) if *pow_base == base => {
            small_positive_integer_value(ctx, *exponent)
        }
        _ => None,
    }
}

fn sophie_germain_terms(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };

    try_match_sophie_germain(ctx, *left, *right)
        .or_else(|| try_match_sophie_germain(ctx, *right, *left))
}

fn try_match_sophie_germain(
    ctx: &Context,
    fourth_power_term: ExprId,
    four_times_fourth_power_term: ExprId,
) -> Option<(ExprId, ExprId)> {
    Some((
        fourth_power_base(ctx, fourth_power_term)?,
        four_times_fourth_power_base(ctx, four_times_fourth_power_term)?,
    ))
}

fn fourth_power_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) if is_small_positive_integer(ctx, *exp, 4) => Some(*base),
        _ => None,
    }
}

fn prefer_non_constant_term_first(ctx: &Context, left: ExprId, right: ExprId) -> (ExprId, ExprId) {
    if is_constant_like(ctx, left) && !is_constant_like(ctx, right) {
        (right, left)
    } else {
        (left, right)
    }
}

fn is_constant_like(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(_) | Expr::Constant(_))
}

fn four_times_fourth_power_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };

    if matches!(ctx.get(*left), Expr::Number(n) if n.is_integer() && n.to_integer() == 4.into()) {
        return fourth_power_base(ctx, *right);
    }
    if matches!(ctx.get(*right), Expr::Number(n) if n.is_integer() && n.to_integer() == 4.into()) {
        return fourth_power_base(ctx, *left);
    }

    None
}

fn matches_sophie_germain_quadratic(
    ctx: &Context,
    expr: ExprId,
    a: ExprId,
    b: ExprId,
    cross_sign: Sign,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 3 {
        return false;
    }

    let has_a_squared = terms
        .iter()
        .any(|(term, sign)| *sign == Sign::Pos && matches_scaled_square(ctx, *term, 1, a));
    let has_two_b_squared = terms
        .iter()
        .any(|(term, sign)| *sign == Sign::Pos && matches_scaled_square(ctx, *term, 2, b));
    let has_cross_term = terms
        .iter()
        .any(|(term, sign)| *sign == cross_sign && matches_scaled_product(ctx, *term, 2, a, b));

    has_a_squared && has_two_b_squared && has_cross_term
}

fn matches_scaled_square(ctx: &Context, expr: ExprId, coeff: i64, base: ExprId) -> bool {
    let factors = expr_nary::mul_leaves(ctx, expr);
    let expected_coeff = BigRational::from_integer(coeff.into());

    if coeff == 1 && factors.len() == 1 {
        return matches_square_of(ctx, factors[0], base);
    }

    if factors.len() != 2 {
        return false;
    }

    let mut saw_coeff = false;
    let mut saw_square = false;
    for factor in factors {
        match ctx.get(factor) {
            Expr::Number(n) if *n == expected_coeff => saw_coeff = true,
            _ if matches_square_of(ctx, factor, base) => saw_square = true,
            _ => return false,
        }
    }

    saw_coeff && saw_square
}

fn matches_square_of(ctx: &Context, expr: ExprId, base: ExprId) -> bool {
    if is_one(ctx, base) {
        return is_one(ctx, expr);
    }
    matches!(
        ctx.get(expr),
        Expr::Pow(pow_base, exp)
            if is_small_positive_integer(ctx, *exp, 2)
                && cas_ast::ordering::compare_expr(ctx, *pow_base, base)
                    == std::cmp::Ordering::Equal
    )
}

fn matches_fourth_power_of(ctx: &Context, expr: ExprId, base: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Pow(pow_base, exp)
            if is_small_positive_integer(ctx, *exp, 4)
                && cas_ast::ordering::compare_expr(ctx, *pow_base, base)
                    == std::cmp::Ordering::Equal
    )
}

fn matches_unscaled_product(ctx: &Context, expr: ExprId, left: ExprId, right: ExprId) -> bool {
    if is_one(ctx, left) && is_one(ctx, right) {
        return is_one(ctx, expr);
    }
    if is_one(ctx, left) {
        return cas_ast::ordering::compare_expr(ctx, expr, right) == std::cmp::Ordering::Equal;
    }
    if is_one(ctx, right) {
        return cas_ast::ordering::compare_expr(ctx, expr, left) == std::cmp::Ordering::Equal;
    }

    let factors = expr_nary::mul_leaves(ctx, expr);
    if factors.len() != 2 {
        return false;
    }

    let mut saw_left = false;
    let mut saw_right = false;
    for factor in factors {
        if !saw_left
            && cas_ast::ordering::compare_expr(ctx, factor, left) == std::cmp::Ordering::Equal
        {
            saw_left = true;
            continue;
        }
        if !saw_right
            && cas_ast::ordering::compare_expr(ctx, factor, right) == std::cmp::Ordering::Equal
        {
            saw_right = true;
            continue;
        }
        return false;
    }

    saw_left && saw_right
}

fn matches_product_of_squares(ctx: &Context, expr: ExprId, left: ExprId, right: ExprId) -> bool {
    let factors = expr_nary::mul_leaves(ctx, expr);
    if factors.len() != 2 {
        return false;
    }

    let mut saw_left = false;
    let mut saw_right = false;
    for factor in factors {
        if !saw_left && matches_square_of(ctx, factor, left) {
            saw_left = true;
            continue;
        }
        if !saw_right && matches_square_of(ctx, factor, right) {
            saw_right = true;
            continue;
        }
        return false;
    }

    saw_left && saw_right
}

fn matches_scaled_product(
    ctx: &Context,
    expr: ExprId,
    coeff: i64,
    left: ExprId,
    right: ExprId,
) -> bool {
    let factors = expr_nary::mul_leaves(ctx, expr);
    if factors.len() != 3 {
        return false;
    }

    let expected_coeff = BigRational::from_integer(coeff.into());
    let mut saw_coeff = false;
    let mut saw_left = false;
    let mut saw_right = false;

    for factor in factors {
        match ctx.get(factor) {
            Expr::Number(n) if *n == expected_coeff => saw_coeff = true,
            _ if cas_ast::ordering::compare_expr(ctx, factor, left)
                == std::cmp::Ordering::Equal =>
            {
                saw_left = true
            }
            _ if cas_ast::ordering::compare_expr(ctx, factor, right)
                == std::cmp::Ordering::Equal =>
            {
                saw_right = true
            }
            _ => return false,
        }
    }

    saw_coeff && saw_left && saw_right
}

fn log_formula_snippet(
    ctx: &Context,
    expr: ExprId,
    expand: bool,
) -> Option<(String, String, String, String, String)> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if ctx.is_builtin(*fn_id, BuiltinFn::Ln) && args.len() == 1 => {
            match ctx.get(args[0]) {
                Expr::Mul(_, _) => {
                    if expand {
                        Some((
                            "Usar que el logaritmo de un producto se separa en una suma"
                                .to_string(),
                            "ln(a · b)".to_string(),
                            "ln(a) + ln(b)".to_string(),
                            "\\ln(ab)".to_string(),
                            "\\ln(a) + \\ln(b)".to_string(),
                        ))
                    } else {
                        Some((
                            "Usar que una suma de logaritmos se puede reunir en un producto"
                                .to_string(),
                            "ln(a) + ln(b)".to_string(),
                            "ln(a · b)".to_string(),
                            "\\ln(a) + \\ln(b)".to_string(),
                            "\\ln(ab)".to_string(),
                        ))
                    }
                }
                Expr::Div(_, _) if expand => Some((
                    "Usar que el logaritmo de un cociente se separa en una resta".to_string(),
                    "ln(a / b)".to_string(),
                    "ln(a) - ln(b)".to_string(),
                    "\\ln\\left(\\frac{a}{b}\\right)".to_string(),
                    "\\ln(a) - \\ln(b)".to_string(),
                )),
                _ => None,
            }
        }
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Log) && args.len() == 2 && expand =>
        {
            let base_display = human_expr(ctx, args[0]);
            let base_latex = latex_expr(ctx, args[0]);
            match ctx.get(args[1]) {
                Expr::Mul(_, _) => Some((
                    "Usar que el logaritmo de un producto se separa en una suma".to_string(),
                    format!("log_{base_display}(a · b)"),
                    format!("log_{base_display}(a) + log_{base_display}(b)"),
                    format!("\\log_{{{base_latex}}}(ab)"),
                    format!("\\log_{{{base_latex}}}(a) + \\log_{{{base_latex}}}(b)"),
                )),
                Expr::Div(_, _) => Some((
                    "Usar que el logaritmo de un cociente se separa en una resta".to_string(),
                    format!("log_{base_display}(a / b)"),
                    format!("log_{base_display}(a) - log_{base_display}(b)"),
                    format!("\\log_{{{base_latex}}}\\left(\\frac{{a}}{{b}}\\right)"),
                    format!("\\log_{{{base_latex}}}(a) - \\log_{{{base_latex}}}(b)"),
                )),
                _ => None,
            }
        }
        Expr::Add(_, _) | Expr::Sub(_, _) if !expand => {
            if let Some(snippet) = scaled_log_formula_snippet(ctx, expr) {
                return Some(snippet);
            }

            let (left, right) = match ctx.get(expr) {
                Expr::Add(left, right) | Expr::Sub(left, right) => (*left, *right),
                _ => return None,
            };
            let (left_fn, right_fn) = (ctx.get(left), ctx.get(right));
            match (left_fn, right_fn) {
                (Expr::Function(left_id, left_args), Expr::Function(right_id, right_args))
                    if ctx.is_builtin(*left_id, BuiltinFn::Ln)
                        && ctx.is_builtin(*right_id, BuiltinFn::Ln)
                        && left_args.len() == 1
                        && right_args.len() == 1 =>
                {
                    Some(if matches!(ctx.get(expr), Expr::Add(_, _)) {
                        (
                            "Usar que una suma de logaritmos se puede reunir en un producto"
                                .to_string(),
                            "ln(a) + ln(b)".to_string(),
                            "ln(a · b)".to_string(),
                            "\\ln(a) + \\ln(b)".to_string(),
                            "\\ln(ab)".to_string(),
                        )
                    } else {
                        (
                            "Usar que una resta de logaritmos se puede reunir en un cociente"
                                .to_string(),
                            "ln(a) - ln(b)".to_string(),
                            "ln(a / b)".to_string(),
                            "\\ln(a) - \\ln(b)".to_string(),
                            "\\ln\\left(\\frac{a}{b}\\right)".to_string(),
                        )
                    })
                }
                (Expr::Function(left_id, left_args), Expr::Function(right_id, right_args))
                    if ctx.is_builtin(*left_id, BuiltinFn::Log)
                        && ctx.is_builtin(*right_id, BuiltinFn::Log)
                        && left_args.len() == 2
                        && right_args.len() == 2
                        && cas_ast::ordering::compare_expr(ctx, left_args[0], right_args[0])
                            == std::cmp::Ordering::Equal =>
                {
                    let base_display = human_expr(ctx, left_args[0]);
                    let base_latex = latex_expr(ctx, left_args[0]);
                    Some(if matches!(ctx.get(expr), Expr::Add(_, _)) {
                        (
                            "Usar que una suma de logaritmos se puede reunir en un producto"
                                .to_string(),
                            format!("log_{base_display}(a) + log_{base_display}(b)"),
                            format!("log_{base_display}(a · b)"),
                            format!("\\log_{{{base_latex}}}(a) + \\log_{{{base_latex}}}(b)"),
                            format!("\\log_{{{base_latex}}}(ab)"),
                        )
                    } else {
                        (
                            "Usar que una resta de logaritmos se puede reunir en un cociente"
                                .to_string(),
                            format!("log_{base_display}(a) - log_{base_display}(b)"),
                            format!("log_{base_display}(a / b)"),
                            format!("\\log_{{{base_latex}}}(a) - \\log_{{{base_latex}}}(b)"),
                            format!("\\log_{{{base_latex}}}\\left(\\frac{{a}}{{b}}\\right)"),
                        )
                    })
                }
                _ => None,
            }
        }
        _ => None,
    }
}

#[derive(Clone, Copy)]
enum LogDidacticFamily {
    Ln,
    Log10,
    LogBase(ExprId),
}

struct ScaledLogDidacticTerm {
    family: LogDidacticFamily,
    coeff: num_bigint::BigInt,
}

fn scaled_log_formula_snippet(
    ctx: &Context,
    expr: ExprId,
) -> Option<(String, String, String, String, String)> {
    let (left, right, is_sub) = match ctx.get(expr) {
        Expr::Add(left, right) => (*left, *right, false),
        Expr::Sub(left, right) => (*left, *right, true),
        _ => return None,
    };

    let left_term = extract_scaled_log_didactic_term(ctx, left)?;
    let right_term = extract_scaled_log_didactic_term(ctx, right)?;
    if left_term.coeff == 1.into() && right_term.coeff == 1.into() {
        return None;
    }
    if !same_log_didactic_family(ctx, left_term.family, right_term.family) {
        return None;
    }

    let (log_display, log_latex) = log_family_formula_name(ctx, left_term.family);
    if is_sub {
        Some((
            "Meter los coeficientes dentro de los logaritmos y reunir la resta en un cociente"
                .to_string(),
            format!(
                "{} · {log_display}(u) - {} · {log_display}(v)",
                left_term.coeff, right_term.coeff
            ),
            format!(
                "{log_display}(u^{} / v^{})",
                left_term.coeff, right_term.coeff
            ),
            format!(
                "{}\\cdot {log_latex}(u) - {}\\cdot {log_latex}(v)",
                left_term.coeff, right_term.coeff
            ),
            format!(
                "{log_latex}\\left(\\frac{{u^{}}}{{v^{}}}\\right)",
                left_term.coeff, right_term.coeff
            ),
        ))
    } else {
        Some((
            "Meter los coeficientes dentro de los logaritmos como exponentes".to_string(),
            format!(
                "{} · {log_display}(u) + {} · {log_display}(v)",
                left_term.coeff, right_term.coeff
            ),
            format!(
                "{log_display}(u^{} · v^{})",
                left_term.coeff, right_term.coeff
            ),
            format!(
                "{}\\cdot {log_latex}(u) + {}\\cdot {log_latex}(v)",
                left_term.coeff, right_term.coeff
            ),
            format!(
                "{log_latex}(u^{}\\cdot v^{})",
                left_term.coeff, right_term.coeff
            ),
        ))
    }
}

fn extract_scaled_log_didactic_term(ctx: &Context, expr: ExprId) -> Option<ScaledLogDidacticTerm> {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            if let Some(coeff) = positive_integer_literal_value(ctx, *left) {
                let family = extract_log_didactic_family(ctx, *right)?;
                return Some(ScaledLogDidacticTerm { family, coeff });
            }
            if let Some(coeff) = positive_integer_literal_value(ctx, *right) {
                let family = extract_log_didactic_family(ctx, *left)?;
                return Some(ScaledLogDidacticTerm { family, coeff });
            }
            None
        }
        _ => Some(ScaledLogDidacticTerm {
            family: extract_log_didactic_family(ctx, expr)?,
            coeff: 1.into(),
        }),
    }
}

fn extract_log_didactic_family(ctx: &Context, expr: ExprId) -> Option<LogDidacticFamily> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };

    if ctx.is_builtin(*fn_id, BuiltinFn::Ln) && args.len() == 1 {
        return Some(LogDidacticFamily::Ln);
    }
    if ctx.is_builtin(*fn_id, BuiltinFn::Log) {
        return match args.as_slice() {
            [_arg] => Some(LogDidacticFamily::Log10),
            [base, _arg] => Some(LogDidacticFamily::LogBase(*base)),
            _ => None,
        };
    }

    None
}

fn same_log_didactic_family(
    ctx: &Context,
    left: LogDidacticFamily,
    right: LogDidacticFamily,
) -> bool {
    match (left, right) {
        (LogDidacticFamily::Ln, LogDidacticFamily::Ln) => true,
        (LogDidacticFamily::Log10, LogDidacticFamily::Log10) => true,
        (LogDidacticFamily::LogBase(left), LogDidacticFamily::LogBase(right)) => {
            cas_ast::ordering::compare_expr(ctx, left, right) == std::cmp::Ordering::Equal
        }
        _ => false,
    }
}

fn log_family_formula_name(ctx: &Context, family: LogDidacticFamily) -> (String, String) {
    match family {
        LogDidacticFamily::Ln => ("ln".to_string(), "\\ln".to_string()),
        LogDidacticFamily::Log10 => ("log".to_string(), "\\log".to_string()),
        LogDidacticFamily::LogBase(base) => {
            let base_display = human_expr(ctx, base);
            let base_latex = latex_expr(ctx, base);
            (
                format!("log_{base_display}"),
                format!("\\log_{{{base_latex}}}"),
            )
        }
    }
}

fn is_small_positive_integer(ctx: &Context, expr: ExprId, value: i64) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_integer() && n.to_integer() == value.into())
}

fn small_positive_integer_value(ctx: &Context, expr: ExprId) -> Option<i64> {
    let Expr::Number(n) = ctx.get(expr) else {
        return None;
    };
    if !n.is_integer() || n <= &BigRational::zero() {
        return None;
    }
    n.to_integer().try_into().ok()
}

fn is_tan_call(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Function(fn_id, args) if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Tan))
}

const MAX_FULL_POLY_PRODUCT_SUBSTEP_TERMS: usize = 14;

#[derive(Debug, Clone)]
struct PolyContribution {
    coeff: BigRational,
    degree: usize,
}

#[derive(Debug, Clone)]
struct PolyProductDidacticPlan {
    expanded_display: String,
    expanded_latex: String,
    grouped_display: Option<String>,
    grouped_latex: Option<String>,
    expanded_terms: usize,
    repeated_degree_groups: usize,
    cancelled_degree_groups: usize,
}

fn generate_polynomial_product_normalize_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(plan) = polynomial_product_didactic_plan(ctx, before) else {
        return Vec::new();
    };

    let before_display = display_expr(ctx, before);
    let before_latex = latex_expr(ctx, before);
    let after_display = display_expr(ctx, after);
    let after_latex = latex_expr(ctx, after);

    if plan.expanded_terms > MAX_FULL_POLY_PRODUCT_SUBSTEP_TERMS {
        let summary = if plan.cancelled_degree_groups > 0 {
            "Multiplicar y reagrupar por grados para cancelar términos intermedios"
        } else {
            "Multiplicar y reagrupar por grados"
        };
        let grouped_display = plan
            .grouped_display
            .clone()
            .unwrap_or(plan.expanded_display.clone());
        let grouped_latex = plan
            .grouped_latex
            .clone()
            .unwrap_or(plan.expanded_latex.clone());
        return vec![SubStep::new(summary, grouped_display, after_display)
            .with_before_latex(grouped_latex)
            .with_after_latex(after_latex)];
    }

    let mut out = vec![SubStep::new(
        "Distribuir cada término del producto",
        before_display,
        plan.expanded_display.clone(),
    )
    .with_before_latex(before_latex)
    .with_after_latex(plan.expanded_latex.clone())];

    match (plan.grouped_display.clone(), plan.grouped_latex.clone()) {
        (Some(grouped_display), Some(grouped_latex))
            if grouped_display != plan.expanded_display =>
        {
            out.push(
                SubStep::new(
                    "Agrupar los términos del mismo grado",
                    plan.expanded_display.clone(),
                    grouped_display.clone(),
                )
                .with_before_latex(plan.expanded_latex)
                .with_after_latex(grouped_latex.clone()),
            );

            let finish_title = if plan.cancelled_degree_groups >= 2 {
                "Los términos intermedios se cancelan por parejas"
            } else if plan.cancelled_degree_groups == 1 {
                "Al combinar esos términos, se cancelan"
            } else if plan.repeated_degree_groups > 0 {
                "Sumar los términos del mismo grado"
            } else {
                "Escribir el resultado ya ordenado por grados"
            };

            out.push(
                SubStep::new(finish_title, grouped_display, after_display)
                    .with_before_latex(grouped_latex)
                    .with_after_latex(after_latex),
            );
        }
        _ => {
            let finish_title = if plan.repeated_degree_groups > 0 {
                "Sumar los términos del mismo grado"
            } else {
                "Escribir el resultado ya ordenado por grados"
            };
            out.push(
                SubStep::new(finish_title, plan.expanded_display, after_display)
                    .with_before_latex(plan.expanded_latex)
                    .with_after_latex(after_latex),
            );
        }
    }

    out
}

fn polynomial_product_didactic_plan(
    ctx: &Context,
    before: ExprId,
) -> Option<PolyProductDidacticPlan> {
    let Expr::Mul(_, _) = ctx.get(before) else {
        return None;
    };

    let vars = cas_ast::collect_variables(ctx, before);
    if vars.len() != 1 {
        return None;
    }
    let var = vars.iter().next()?.to_string();

    let factors = cas_math::expr_nary::mul_leaves(ctx, before);
    if factors.len() < 2 {
        return None;
    }

    let factor_terms = factors
        .iter()
        .map(|factor| factor_polynomial_terms(ctx, *factor, &var))
        .collect::<Option<Vec<_>>>()?;

    let expanded_terms = expand_polynomial_term_products(&factor_terms);
    if expanded_terms.len() < 2 {
        return None;
    }

    let (expanded_display, expanded_latex) = render_contribution_sum(&var, &expanded_terms);
    let grouped_by_degree = group_contributions_by_degree(&expanded_terms);
    let repeated_degree_groups = grouped_by_degree
        .values()
        .filter(|group| group.len() > 1)
        .count();
    let cancelled_degree_groups = grouped_by_degree
        .values()
        .filter(|group| group.len() > 1 && contribution_group_sum(group).is_zero())
        .count();

    let (grouped_display, grouped_latex) = if repeated_degree_groups > 0 {
        let (display, latex) = render_grouped_contributions(&var, &grouped_by_degree);
        (Some(display), Some(latex))
    } else {
        (None, None)
    };

    Some(PolyProductDidacticPlan {
        expanded_display,
        expanded_latex,
        grouped_display,
        grouped_latex,
        expanded_terms: expanded_terms.len(),
        repeated_degree_groups,
        cancelled_degree_groups,
    })
}

fn factor_polynomial_terms(
    ctx: &Context,
    factor: ExprId,
    var: &str,
) -> Option<Vec<PolyContribution>> {
    let poly = Polynomial::from_expr(ctx, factor, var).ok()?;
    let mut terms = Vec::new();
    for (degree, coeff) in poly.coeffs.iter().enumerate().rev() {
        if coeff.is_zero() {
            continue;
        }
        terms.push(PolyContribution {
            coeff: coeff.clone(),
            degree,
        });
    }
    Some(terms)
}

fn expand_polynomial_term_products(factors: &[Vec<PolyContribution>]) -> Vec<PolyContribution> {
    let mut acc = vec![PolyContribution {
        coeff: BigRational::from_integer(1.into()),
        degree: 0,
    }];

    for factor in factors {
        let mut next = Vec::new();
        for partial in &acc {
            for term in factor {
                next.push(PolyContribution {
                    coeff: partial.coeff.clone() * term.coeff.clone(),
                    degree: partial.degree + term.degree,
                });
            }
        }
        acc = next;
    }

    acc.retain(|term| !term.coeff.is_zero());
    acc
}

fn group_contributions_by_degree(
    contributions: &[PolyContribution],
) -> BTreeMap<usize, Vec<PolyContribution>> {
    let mut out: BTreeMap<usize, Vec<PolyContribution>> = BTreeMap::new();
    for contribution in contributions {
        out.entry(contribution.degree)
            .or_default()
            .push(contribution.clone());
    }
    out
}

fn contribution_group_sum(group: &[PolyContribution]) -> BigRational {
    group
        .iter()
        .fold(BigRational::from_integer(0.into()), |acc, term| {
            acc + term.coeff.clone()
        })
}

fn render_contribution_sum(var: &str, terms: &[PolyContribution]) -> (String, String) {
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

fn build_sum_expr_from_contributions(
    ctx: &mut Context,
    var: &str,
    contributions: &[PolyContribution],
) -> ExprId {
    if contributions.is_empty() {
        return ctx.num(0);
    }

    let mut iter = contributions.iter();
    let first = iter.next().expect("nonempty");
    let mut expr = build_signed_monomial_expr(ctx, var, first);

    for term in iter {
        let rhs = build_unsigned_monomial_expr(ctx, var, &term.coeff.abs(), term.degree);
        expr = if term.coeff.is_negative() {
            ctx.add_raw(Expr::Sub(expr, rhs))
        } else {
            ctx.add_raw(Expr::Add(expr, rhs))
        };
    }

    expr
}

fn build_signed_monomial_expr(ctx: &mut Context, var: &str, term: &PolyContribution) -> ExprId {
    let abs = term.coeff.abs();
    let unsigned = build_unsigned_monomial_expr(ctx, var, &abs, term.degree);
    if term.coeff.is_negative() {
        ctx.add_raw(Expr::Neg(unsigned))
    } else {
        unsigned
    }
}

fn build_unsigned_monomial_expr(
    ctx: &mut Context,
    var: &str,
    coeff: &BigRational,
    degree: usize,
) -> ExprId {
    if degree == 0 {
        return ctx.add(Expr::Number(coeff.clone()));
    }

    let var_expr = ctx.var(var);
    let power_expr = if degree == 1 {
        var_expr
    } else {
        let exp = ctx.num(degree as i64);
        ctx.add(Expr::Pow(var_expr, exp))
    };

    if coeff == &BigRational::from_integer(1.into()) {
        power_expr
    } else {
        let coeff_expr = ctx.add(Expr::Number(coeff.clone()));
        ctx.add(Expr::Mul(coeff_expr, power_expr))
    }
}

fn render_grouped_contributions(
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

fn generate_canonicalize_nested_power_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let mut before = step.before_local().unwrap_or(step.before);
    let mut after = step.after_local().unwrap_or(step.after);

    let uses_local_pow = matches!(ctx.get(before), Expr::Pow(base, _)
        if matches!(ctx.get(*base), Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Sqrt) && args.len() == 1));

    if !uses_local_pow {
        before = step.before;
        after = step.after;
    }

    if before == after {
        return Vec::new();
    }

    vec![SubStep::new(
        "Pasar la potencia al interior de la raíz",
        display_expr(ctx, before),
        display_expr(ctx, after),
    )
    .with_before_latex(latex_expr(ctx, before))
    .with_after_latex(latex_expr(ctx, after))]
}

fn generate_common_factor_cancel_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Div(numerator, denominator) = ctx.get(before) else {
        return Vec::new();
    };
    let Some(common_factor) = first_common_factor(ctx, *numerator, *denominator) else {
        return Vec::new();
    };

    let factor_display = display_expr(ctx, common_factor);
    let before_display = display_expr(ctx, before);
    let before_latex = latex_expr(ctx, before);
    let final_display = display_expr(ctx, after);
    let final_latex = latex_expr(ctx, after);

    let Some((intermediate_display, intermediate_latex)) =
        quotient_after_cancel_once(ctx, *numerator, *denominator, common_factor)
    else {
        return Vec::new();
    };

    if cas_formatter::clean_display_string(&intermediate_display)
        == cas_formatter::clean_display_string(&final_display)
        && intermediate_latex == final_latex
    {
        return Vec::new();
    }

    let mut out = vec![SubStep::new(
        format!("Cancelar el factor común {}", factor_display),
        before_display,
        intermediate_display.clone(),
    )
    .with_before_latex(before_latex)
    .with_after_latex(intermediate_latex.clone())];

    if cas_formatter::clean_display_string(&intermediate_display)
        != cas_formatter::clean_display_string(&final_display)
        || intermediate_latex != final_latex
    {
        let finish_title =
            next_common_factor_after_cancel(ctx, *numerator, *denominator, common_factor)
                .map(|next_factor| {
                    format!(
                        "Cancelar también el factor común {}",
                        display_expr(ctx, next_factor)
                    )
                })
                .unwrap_or_else(|| "Reducir la fracción que queda".to_string());
        out.push(
            SubStep::new(finish_title, intermediate_display, final_display)
                .with_before_latex(intermediate_latex)
                .with_after_latex(final_latex),
        );
    }

    out
}

fn generate_simplify_nested_fraction_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let repeated_factor = generate_perfect_square_fraction_cancel_substeps(ctx, step);
    if !repeated_factor.is_empty() {
        return repeated_factor;
    }

    let common_factor = generate_common_factor_cancel_substeps(ctx, step);
    if !common_factor.is_empty() {
        return common_factor;
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Div(_, _) = ctx.get(before) else {
        return Vec::new();
    };

    {
        let mut work = ctx.clone();
        if let Some((den_before, den_after, full_intermediate)) =
            build_one_over_fraction_plus_minus_one_intermediates(&mut work, before)
        {
            return vec![
                SubStep::new(
                    "Llevar a denominador común dentro del denominador",
                    display_expr(&work, den_before),
                    display_expr(&work, den_after),
                )
                .with_before_latex(latex_expr(&work, den_before))
                .with_after_latex(latex_expr(&work, den_after)),
                SubStep::new(
                    "Invertir la fracción del denominador",
                    display_expr(&work, full_intermediate),
                    display_expr(ctx, after),
                )
                .with_before_latex(latex_expr(&work, full_intermediate))
                .with_after_latex(latex_expr(ctx, after)),
            ];
        }
    }

    {
        let mut work = ctx.clone();
        if let Some(intermediate) =
            build_sum_difference_reciprocal_complex_fraction_intermediate(&mut work, before)
        {
            return vec![
                SubStep::new(
                    "Llevar el numerador y el denominador a común denominador",
                    display_expr(ctx, before),
                    display_expr(&work, intermediate),
                )
                .with_before_latex(latex_expr(ctx, before))
                .with_after_latex(latex_expr(&work, intermediate)),
                SubStep::new(
                    "Cancelar el denominador común de numerador y denominador",
                    display_expr(&work, intermediate),
                    display_expr(ctx, after),
                )
                .with_before_latex(latex_expr(&work, intermediate))
                .with_after_latex(latex_expr(ctx, after)),
            ];
        }
    }

    vec![SubStep::new(
        "Cancelar los factores comunes del numerador y del denominador",
        display_expr(ctx, before),
        display_expr(ctx, after),
    )
    .with_before_latex(latex_expr(ctx, before))
    .with_after_latex(latex_expr(ctx, after))]
}

fn unit_fraction_denominator(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Div(num, den) if is_one(ctx, *num) => Some(*den),
        _ => None,
    }
}

fn build_reciprocal_pair_with_common_denominator(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            let left_den = unit_fraction_denominator(ctx, *left)?;
            let right_den = unit_fraction_denominator(ctx, *right)?;
            let common_den = build_balanced_mul(ctx, &[left_den, right_den]);
            let numerator = ctx.add(Expr::Add(right_den, left_den));
            Some(ctx.add(Expr::Div(numerator, common_den)))
        }
        Expr::Sub(left, right) => {
            let left_den = unit_fraction_denominator(ctx, *left)?;
            let right_den = unit_fraction_denominator(ctx, *right)?;
            let common_den = build_balanced_mul(ctx, &[left_den, right_den]);
            let numerator = ctx.add(Expr::Sub(right_den, left_den));
            Some(ctx.add(Expr::Div(numerator, common_den)))
        }
        _ => None,
    }
}

fn build_sum_difference_reciprocal_complex_fraction_intermediate(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let (numerator, denominator) = as_div(ctx, expr)?;
    let numerator = build_reciprocal_pair_with_common_denominator(ctx, numerator)?;
    let denominator = build_reciprocal_pair_with_common_denominator(ctx, denominator)?;
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn build_one_over_fraction_plus_minus_one_intermediates(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    let (numerator, denominator) = as_div(ctx, expr)?;
    if !is_one(ctx, numerator) {
        return None;
    }

    let (_fraction_num, fraction_den, den_after) = match ctx.get(denominator).clone() {
        Expr::Add(left, right) => {
            if is_one(ctx, left) {
                let (frac_num, frac_den) = as_div(ctx, right)?;
                let combined_num = ctx.add(Expr::Add(frac_den, frac_num));
                let den_after = ctx.add(Expr::Div(combined_num, frac_den));
                (frac_num, frac_den, den_after)
            } else if is_one(ctx, right) {
                let (frac_num, frac_den) = as_div(ctx, left)?;
                let combined_num = ctx.add(Expr::Add(frac_num, frac_den));
                let den_after = ctx.add(Expr::Div(combined_num, frac_den));
                (frac_num, frac_den, den_after)
            } else {
                return None;
            }
        }
        Expr::Sub(left, right) => {
            if is_one(ctx, left) {
                let (frac_num, frac_den) = as_div(ctx, right)?;
                let combined_num = ctx.add(Expr::Sub(frac_den, frac_num));
                let den_after = ctx.add(Expr::Div(combined_num, frac_den));
                (frac_num, frac_den, den_after)
            } else if is_one(ctx, right) {
                let (frac_num, frac_den) = as_div(ctx, left)?;
                let combined_num = ctx.add(Expr::Sub(frac_num, frac_den));
                let den_after = ctx.add(Expr::Div(combined_num, frac_den));
                (frac_num, frac_den, den_after)
            } else {
                return None;
            }
        }
        _ => return None,
    };

    let full_intermediate = ctx.add(Expr::Div(numerator, den_after));
    let _ = fraction_den;
    Some((denominator, den_after, full_intermediate))
}

fn generate_perfect_square_fraction_cancel_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if let (Some(before_local), Some(after_local)) = (step.before_local(), step.after_local()) {
        let substeps = generate_perfect_square_fraction_cancel_substeps_for_pair(
            ctx,
            before_local,
            after_local,
        );
        if !substeps.is_empty() {
            return substeps;
        }
    }

    generate_perfect_square_fraction_cancel_substeps_for_pair(ctx, step.before, step.after)
}

fn generate_perfect_square_fraction_cancel_substeps_for_pair(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Vec<SubStep> {
    let Expr::Div(numerator, denominator) = ctx.get(before) else {
        return Vec::new();
    };
    if after != *denominator {
        return Vec::new();
    }

    let denominator_display = display_expr(ctx, *denominator);
    let denominator_squared_display = squared_display(ctx, *denominator);

    if is_repeated_factor_product(ctx, *numerator, *denominator) {
        return vec![formula_substep(
            format!(
                "Si {} aparece dos veces arriba y una abajo, queda una sola copia",
                denominator_display
            ),
            "(u · u) / u",
            "u",
            "\\frac{u\\cdot u}{u}",
            "u",
        )];
    }

    if is_square_of_expr(ctx, *numerator, *denominator) {
        return vec![formula_substep(
            format!(
                "Si {} está dividido entre {}, queda una sola copia",
                denominator_squared_display, denominator_display
            ),
            "u^2 / u",
            "u",
            "\\frac{u^2}{u}",
            "u",
        )];
    }

    if let Some(square_latex) = perfect_square_form_latex(ctx, *numerator, *denominator) {
        return vec![
            SubStep::new(
                "Reconocer que el numerador es un cuadrado perfecto",
                display_expr(ctx, *numerator),
                squared_display(ctx, *denominator),
            )
            .with_before_latex(latex_expr(ctx, *numerator))
            .with_after_latex(square_latex),
            formula_substep(
                format!(
                    "Si {} está dividido entre {}, queda una sola copia",
                    denominator_squared_display, denominator_display
                ),
                "u^2 / u",
                "u",
                "\\frac{u^2}{u}",
                "u",
            ),
        ];
    }

    let mut temp_ctx = ctx.clone();
    let exponent = temp_ctx.num(2);
    let squared = temp_ctx.add_raw(Expr::Pow(*denominator, exponent));
    if poly_eq(&temp_ctx, *numerator, squared) {
        return vec![
            SubStep::new(
                "Reconocer que el numerador es un cuadrado perfecto",
                display_expr(ctx, *numerator),
                squared_display(ctx, *denominator),
            )
            .with_before_latex(latex_expr(ctx, *numerator))
            .with_after_latex(latex_expr(&temp_ctx, squared)),
            formula_substep(
                format!(
                    "Si {} está dividido entre {}, queda una sola copia",
                    denominator_squared_display, denominator_display
                ),
                "u^2 / u",
                "u",
                "\\frac{u^2}{u}",
                "u",
            ),
        ];
    }
    Vec::new()
}

fn generate_identity_addition_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Add(left, right) = ctx.get(before) else {
        return Vec::new();
    };
    if !is_zero(ctx, *left) && !is_zero(ctx, *right) {
        return Vec::new();
    }
    let _ = after;
    // The step title "Quitar el 0" plus Before/After already explains this move.
    // Emitting a substep here only repeats the obvious.
    Vec::new()
}

fn generate_difference_of_squares_cancel_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Div(numerator, denominator) = ctx.get(before) else {
        return Vec::new();
    };
    let Some((other_factor, canceled_factor)) =
        split_product_for_cancellation(ctx, *numerator, *denominator)
    else {
        return Vec::new();
    };
    let Some((left_term, right_term)) = difference_square_terms(ctx, other_factor, canceled_factor)
    else {
        return Vec::new();
    };

    vec![
        SubStep::new(
            "Usar la diferencia de cuadrados: a^2 - b^2 = (a - b)(a + b)",
            format!(
                "{} - {}",
                squared_display(ctx, left_term),
                squared_display(ctx, right_term)
            ),
            display_expr(ctx, *numerator),
        )
        .with_before_latex(format!(
            "{} - {}",
            squared_latex(ctx, left_term),
            squared_latex(ctx, right_term)
        ))
        .with_after_latex(latex_expr(ctx, *numerator)),
        SubStep::new(
            format!(
                "Ahora se cancela el factor {}",
                display_expr(ctx, canceled_factor)
            ),
            format!(
                "({} · {}) / ({})",
                display_expr(ctx, canceled_factor),
                display_expr(ctx, other_factor),
                display_expr(ctx, *denominator),
            ),
            display_expr(ctx, after),
        )
        .with_before_latex(format!(
            "\\frac{{{} \\cdot {}}}{{{}}}",
            latex_expr(ctx, canceled_factor),
            latex_expr(ctx, other_factor),
            latex_expr(ctx, *denominator),
        ))
        .with_after_latex(latex_expr(ctx, after)),
    ]
}

fn generate_inverse_tan_relation_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let mut out = Vec::new();
    let pair_before = step.before_local().unwrap_or(step.before);
    let pair_after = step.after_local().unwrap_or(step.after);

    if step.before != pair_before {
        out.push(
            SubStep::new(
                "Juntar la pareja que encaja con la identidad",
                display_expr(ctx, step.before),
                display_expr(ctx, pair_before),
            )
            .with_before_latex(latex_expr(ctx, step.before))
            .with_after_latex(latex_expr(ctx, pair_before)),
        );
    }

    if pair_before != pair_after {
        out.push(
            SubStep::new(
                "Esa pareja vale pi/2",
                display_expr(ctx, pair_before),
                display_expr(ctx, pair_after),
            )
            .with_before_latex(latex_expr(ctx, pair_before))
            .with_after_latex(latex_expr(ctx, pair_after)),
        );
    }

    out
}

fn generate_sqrt_perfect_square_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(radicand) = sqrt_radicand(ctx, before) else {
        return Vec::new();
    };
    let Some(abs_arg) = abs_argument(ctx, after) else {
        return Vec::new();
    };

    let square_display = squared_display(ctx, abs_arg);
    let square_latex = squared_latex(ctx, abs_arg);

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

fn generate_subtraction_self_cancel_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Sub(left, right) = ctx.get(before) else {
        return Vec::new();
    };
    if left != right || after == before {
        return Vec::new();
    }
    let _ = (left, right);
    // The human-visible step title plus the direct local change already explain this move.
    // Adding micro-substeps like "the two terms are the same" only creates didactic noise.
    Vec::new()
}

fn generate_polynomial_identity_exact_cancel_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if step
        .description
        .to_ascii_lowercase()
        .contains("opaque substitution")
        && is_zero(ctx, step.after_local().unwrap_or(step.after))
    {
        let before = step.before_local().unwrap_or(step.before);
        let after = step.after_local().unwrap_or(step.after);
        return vec![SubStep::new(
            "Las dos partes se compensan exactamente",
            display_expr(ctx, before),
            display_expr(ctx, after),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(latex_expr(ctx, after))];
    }

    if let Some(proof) = step.poly_proof() {
        if !proof.opaque_substitutions.is_empty()
            && is_zero(ctx, step.after_local().unwrap_or(step.after))
        {
            let before = step.before_local().unwrap_or(step.before);
            let after = step.after_local().unwrap_or(step.after);
            return vec![SubStep::new(
                "Las dos partes se compensan exactamente",
                display_expr(ctx, before),
                display_expr(ctx, after),
            )
            .with_before_latex(latex_expr(ctx, before))
            .with_after_latex(latex_expr(ctx, after))];
        }
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let Some((left, right)) =
        difference_like_terms(ctx, before).or_else(|| difference_like_terms(ctx, step.before))
    else {
        return Vec::new();
    };

    let identity_substeps = generate_identity_equivalence_substeps(ctx, left, right);
    if !identity_substeps.is_empty() {
        return identity_substeps;
    }

    vec![SubStep::new(
        "Las dos partes representan la misma cantidad",
        display_expr(ctx, left),
        display_expr(ctx, right),
    )
    .with_before_latex(latex_expr(ctx, left))
    .with_after_latex(latex_expr(ctx, right))]
}

fn generate_identity_equivalence_substeps(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> Vec<SubStep> {
    if geometric_difference_factor_plan(ctx, left, right).is_some() {
        return vec![SubStep::new(
            "Usar a^n - 1 = (a - 1) · (a^(n-1) + a^(n-2) + ... + a + 1)",
            human_expr(ctx, left),
            human_expr(ctx, right),
        )
        .with_before_latex(latex_expr(ctx, left))
        .with_after_latex(latex_expr(ctx, right))];
    }

    if common_factor_factorization_plan(ctx, left, right).is_some() {
        return vec![SubStep::new(
            "Usar el factor común",
            human_expr(ctx, left),
            human_expr(ctx, right),
        )
        .with_before_latex(latex_expr(ctx, left))
        .with_after_latex(latex_expr(ctx, right))];
    }

    if let Some((left_base, right_base, kind)) = binomial_square_terms(ctx, right) {
        let _ = prefer_non_constant_term_first(ctx, left_base, right_base);
        return vec![SubStep::new(
            match kind {
                BinomialSquareKind::Sum => "Usar a^2 + 2ab + b^2 = (a + b)^2",
                BinomialSquareKind::Difference => "Usar a^2 - 2ab + b^2 = (a - b)^2",
            },
            human_expr(ctx, left),
            human_expr(ctx, right),
        )
        .with_before_latex(latex_expr(ctx, left))
        .with_after_latex(latex_expr(ctx, right))];
    }

    if sophie_germain_terms(ctx, left).is_some() {
        return vec![SubStep::new(
            "Usar a^4 + 4b^4 = (a^2 - 2ab + 2b^2) · (a^2 + 2ab + 2b^2)",
            human_expr(ctx, left),
            human_expr(ctx, right),
        )
        .with_before_latex(latex_expr(ctx, left))
        .with_after_latex(latex_expr(ctx, right))];
    }

    Vec::new()
}

fn generate_subtract_expanded_cubes_quotient_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let Some((left, right)) = difference_like_terms(ctx, before) else {
        return Vec::new();
    };
    let Some((numerator, denominator)) = as_div(ctx, left) else {
        return Vec::new();
    };
    let Some(plan) = cube_identity_plan_for_fraction_cancel(ctx, numerator, denominator) else {
        return Vec::new();
    };
    let base_left = plan.left_base;
    let base_right = plan.right_base;
    let kind = plan.kind;

    let left_display = human_expr(ctx, base_left);
    let right_display = human_expr(ctx, base_right);
    let left_latex = latex_expr(ctx, base_left);
    let right_latex = latex_expr(ctx, base_right);
    let identity_title = match kind {
        CubeIdentityKind::Sum => "Usar a^3 + b^3 = (a + b)(a^2 - ab + b^2)",
        CubeIdentityKind::Difference => "Usar a^3 - b^3 = (a - b)(a^2 + ab + b^2)",
    };
    let numerator_display = match kind {
        CubeIdentityKind::Sum => format!("{left_display}^3 + {right_display}^3"),
        CubeIdentityKind::Difference => format!("{left_display}^3 - {right_display}^3"),
    };
    let numerator_latex = match kind {
        CubeIdentityKind::Sum => format!("{left_latex}^3 + {right_latex}^3"),
        CubeIdentityKind::Difference => format!("{left_latex}^3 - {right_latex}^3"),
    };
    let factored_display = human_expr(ctx, numerator);
    let factored_latex = latex_expr(ctx, numerator);

    vec![
        formula_substep(
            identity_title,
            &numerator_display,
            &factored_display,
            &numerator_latex,
            &factored_latex,
        ),
        formula_substep(
            "Cancelar el factor común del numerador y el denominador",
            &format!("{factored_display} / {}", human_expr(ctx, denominator)),
            &human_expr(ctx, right),
            &format!(
                "\\frac{{{factored_latex}}}{{{}}}",
                latex_expr(ctx, denominator)
            ),
            &latex_expr(ctx, right),
        ),
    ]
}

fn generate_cancel_reciprocal_exponents_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let local_before = step.before_local().unwrap_or(step.before);
    let local_after = step.after_local().unwrap_or(step.after);
    let Some(_plan) = reciprocal_exponent_plan(ctx, local_before) else {
        return Vec::new();
    };

    let mut out = vec![concrete_expr_substep(
        ctx,
        "El cuadrado deshace la raíz",
        local_before,
        local_after,
    )];

    if let (Some(global_before), Some(global_after)) = (step.global_before, step.global_after) {
        if global_before != local_before || global_after != local_after {
            out.push(
                SubStep::new(
                    "Reemplazar ese bloque en la expresión",
                    display_expr(ctx, global_before),
                    display_expr(ctx, global_after),
                )
                .with_before_latex(latex_expr(ctx, global_before))
                .with_after_latex(latex_expr(ctx, global_after)),
            );
            return out;
        }
    }

    if step.before != local_before || step.after != local_after {
        out.push(
            SubStep::new(
                "Reemplazar ese bloque en la expresión",
                display_expr(ctx, step.before),
                display_expr(ctx, step.after),
            )
            .with_before_latex(latex_expr(ctx, step.before))
            .with_after_latex(latex_expr(ctx, step.after)),
        );
    } else if local_before != local_after {
        out.push(
            SubStep::new(
                "Reemplazar la raíz al cuadrado por el radicando",
                display_expr(ctx, local_before),
                display_expr(ctx, local_after),
            )
            .with_before_latex(latex_expr(ctx, local_before))
            .with_after_latex(latex_expr(ctx, local_after)),
        );
    }

    out
}

fn generate_identity_multiplication_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Mul(left, right) = ctx.get(before) else {
        return Vec::new();
    };
    if !is_one(ctx, *left) && !is_one(ctx, *right) {
        return Vec::new();
    }
    let _ = after;
    // The step title "Quitar el factor 1" is already self-explanatory.
    Vec::new()
}

fn generate_evaluate_numeric_power_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Pow(base, _) = ctx.get(before) else {
        return Vec::new();
    };
    let before_human = normalize_human_power_expr(&human_expr(ctx, before));
    if is_zero(ctx, *base)
        || is_one(ctx, *base)
        || is_negative_one(ctx, *base)
        || matches!(before_human.split_once('^'), Some(("0" | "1" | "(-1)", _)))
    {
        // Evaluating 0^n, 1^n or (-1)^n is too trivial to deserve its own didactic micro-step.
        return Vec::new();
    }
    let after_human = human_expr(ctx, after);

    vec![SubStep::new(
        format!("Calcular {} = {}", before_human, after_human),
        before_human,
        after_human,
    )
    .with_before_latex(latex_expr(ctx, before))
    .with_after_latex(latex_expr(ctx, after))]
}

fn generate_sum_difference_cubes_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(plan) = cube_identity_plan(ctx, before, after) else {
        return Vec::new();
    };

    let identity_description = match plan.kind {
        CubeIdentityKind::Sum => "Reconocer la forma a^3 + b^3",
        CubeIdentityKind::Difference => "Reconocer la forma a^3 - b^3",
    };
    let factor_description = match plan.kind {
        CubeIdentityKind::Sum => "Aplicar a^3 + b^3 = (a + b)(a^2 - ab + b^2)",
        CubeIdentityKind::Difference => "Aplicar a^3 - b^3 = (a - b)(a^2 + ab + b^2)",
    };

    vec![
        SubStep::new(
            identity_description,
            display_expr(ctx, before),
            cube_identity_display(ctx, plan.left_base, plan.right_base, plan.kind),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(cube_identity_latex(
            ctx,
            plan.left_base,
            plan.right_base,
            plan.kind,
        )),
        SubStep::new(
            factor_description,
            cube_identity_display(ctx, plan.left_base, plan.right_base, plan.kind),
            display_expr(ctx, after),
        )
        .with_before_latex(cube_identity_latex(
            ctx,
            plan.left_base,
            plan.right_base,
            plan.kind,
        ))
        .with_after_latex(latex_expr(ctx, after)),
    ]
}

fn generate_sum_difference_sixth_powers_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(plan) = sixth_power_factorized_identity_plan(ctx, after, before) else {
        return Vec::new();
    };

    let identity_latex =
        sixth_power_identity_latex(ctx, plan.left_base, plan.right_base, plan.kind);
    let factorized_display =
        sixth_power_factorized_identity_display(ctx, plan.left_base, plan.right_base, plan.kind);
    let factorized_latex =
        sixth_power_factorized_identity_latex(ctx, plan.left_base, plan.right_base, plan.kind);
    let factor_description = match plan.kind {
        SixthPowerIdentityKind::Sum => "Aplicar a^6 + b^6 = (a^2 + b^2)(a^4 - a^2b^2 + b^4)",
        SixthPowerIdentityKind::Difference => "Aplicar a^6 - b^6 = (a^2 - b^2)(a^4 + a^2b^2 + b^4)",
    };

    vec![SubStep::new(
        factor_description,
        display_expr(ctx, before),
        factorized_display,
    )
    .with_before_latex(identity_latex)
    .with_after_latex(factorized_latex)]
}

fn generate_sum_difference_cubes_expansion_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before;
    let after = step.after;
    let Some(plan) = cube_factorized_identity_plan(ctx, before, after) else {
        return Vec::new();
    };

    let factorized_display =
        cube_factorized_identity_display(ctx, plan.left_base, plan.right_base, plan.kind);
    let factorized_latex =
        cube_factorized_identity_latex(ctx, plan.left_base, plan.right_base, plan.kind);
    let identity_latex = cube_identity_latex(ctx, plan.left_base, plan.right_base, plan.kind);
    let recognize_description = match plan.kind {
        CubeIdentityKind::Sum => "Reconocer el patrón (a + b)(a^2 - ab + b^2)",
        CubeIdentityKind::Difference => "Reconocer el patrón (a - b)(a^2 + ab + b^2)",
    };
    let expand_description = match plan.kind {
        CubeIdentityKind::Sum => "Aplicar (a + b)(a^2 - ab + b^2) = a^3 + b^3",
        CubeIdentityKind::Difference => "Aplicar (a - b)(a^2 + ab + b^2) = a^3 - b^3",
    };

    vec![
        SubStep::new(
            recognize_description,
            display_expr(ctx, before),
            factorized_display.clone(),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(factorized_latex.clone()),
        SubStep::new(
            expand_description,
            factorized_display,
            display_expr(ctx, after),
        )
        .with_before_latex(factorized_latex)
        .with_after_latex(identity_latex),
    ]
}

fn generate_sum_difference_sixth_powers_expansion_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before;
    let after = step.after;
    let Some(plan) = sixth_power_factorized_identity_plan(ctx, before, after) else {
        return Vec::new();
    };

    let factorized_display =
        sixth_power_factorized_identity_display(ctx, plan.left_base, plan.right_base, plan.kind);
    let factorized_latex =
        sixth_power_factorized_identity_latex(ctx, plan.left_base, plan.right_base, plan.kind);
    let identity_display =
        sixth_power_identity_display(ctx, plan.left_base, plan.right_base, plan.kind);
    let identity_latex =
        sixth_power_identity_latex(ctx, plan.left_base, plan.right_base, plan.kind);
    let recognize_description = match plan.kind {
        SixthPowerIdentityKind::Sum => "Reconocer el patrón (a^2 + b^2)(a^4 - a^2b^2 + b^4)",
        SixthPowerIdentityKind::Difference => "Reconocer el patrón (a^2 - b^2)(a^4 + a^2b^2 + b^4)",
    };
    let expand_description = match plan.kind {
        SixthPowerIdentityKind::Sum => "Aplicar (a^2 + b^2)(a^4 - a^2b^2 + b^4) = a^6 + b^6",
        SixthPowerIdentityKind::Difference => "Aplicar (a^2 - b^2)(a^4 + a^2b^2 + b^4) = a^6 - b^6",
    };

    vec![
        SubStep::new(
            recognize_description,
            display_expr(ctx, before),
            factorized_display.clone(),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(factorized_latex.clone()),
        SubStep::new(expand_description, factorized_display, identity_display)
            .with_before_latex(factorized_latex)
            .with_after_latex(identity_latex),
    ]
}

fn generate_sum_difference_cubes_cancel_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Div(numerator, denominator) = ctx.get(before) else {
        return Vec::new();
    };
    if let Some(plan) = cube_identity_plan_for_fraction_cancel(ctx, *numerator, *denominator) {
        let factorized_numerator =
            cube_factorized_identity_display(ctx, plan.left_base, plan.right_base, plan.kind);
        let factorized_numerator_latex =
            cube_factorized_identity_latex(ctx, plan.left_base, plan.right_base, plan.kind);
        let matching_factor =
            cube_linear_factor_display(ctx, plan.left_base, plan.right_base, plan.kind);
        let matching_factor_latex =
            cube_linear_factor_latex(ctx, plan.left_base, plan.right_base, plan.kind);
        let numerator_display = match plan.kind {
            CubeIdentityKind::Sum => {
                format!(
                    "{}^3 + {}^3",
                    human_expr(ctx, plan.left_base),
                    human_expr(ctx, plan.right_base)
                )
            }
            CubeIdentityKind::Difference => {
                format!(
                    "{}^3 - {}^3",
                    human_expr(ctx, plan.left_base),
                    human_expr(ctx, plan.right_base)
                )
            }
        };
        let numerator_latex = match plan.kind {
            CubeIdentityKind::Sum => {
                format!(
                    "{}^3 + {}^3",
                    latex_expr(ctx, plan.left_base),
                    latex_expr(ctx, plan.right_base)
                )
            }
            CubeIdentityKind::Difference => {
                format!(
                    "{}^3 - {}^3",
                    latex_expr(ctx, plan.left_base),
                    latex_expr(ctx, plan.right_base)
                )
            }
        };

        return vec![
            SubStep::new(
                "Factorizar el numerador como suma o diferencia de cubos",
                numerator_display,
                factorized_numerator.clone(),
            )
            .with_before_latex(numerator_latex)
            .with_after_latex(factorized_numerator_latex.clone()),
            SubStep::new(
                format!("Ahora se cancela el factor {matching_factor}"),
                format!("({}) / ({})", factorized_numerator, matching_factor),
                display_expr(ctx, after),
            )
            .with_before_latex(format!(
                "\\frac{{{}}}{{{}}}",
                factorized_numerator_latex, matching_factor_latex
            ))
            .with_after_latex(latex_expr(ctx, after)),
        ];
    }
    let Some((remaining_factor, matching_factor)) =
        split_product_for_cancellation(ctx, *numerator, *denominator)
    else {
        return Vec::new();
    };

    vec![
        SubStep::new(
            format!(
                "Reconocer el factor común {} en el numerador",
                display_expr(ctx, matching_factor)
            ),
            display_expr(ctx, before),
            display_expr(ctx, matching_factor),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(latex_expr(ctx, matching_factor)),
        SubStep::new(
            format!(
                "Cancelar el factor común {}",
                display_expr(ctx, matching_factor)
            ),
            display_expr(ctx, before),
            display_expr(ctx, after),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(latex_expr(ctx, after)),
        SubStep::new(
            format!(
                "El otro factor del cubo es {}",
                display_expr(ctx, remaining_factor)
            ),
            display_expr(ctx, matching_factor),
            display_expr(ctx, remaining_factor),
        )
        .with_before_latex(latex_expr(ctx, matching_factor))
        .with_after_latex(latex_expr(ctx, remaining_factor)),
    ]
}

fn display_expr(ctx: &Context, expr: ExprId) -> String {
    format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: expr,
        }
    )
}

fn latex_expr(ctx: &Context, expr: ExprId) -> String {
    cas_formatter::LaTeXExpr {
        context: ctx,
        id: expr,
    }
    .to_latex()
}

fn human_expr(ctx: &Context, expr: ExprId) -> String {
    cas_formatter::clean_display_string(&crate::didactic::latex_to_plain_text(&latex_expr(
        ctx, expr,
    )))
}

fn normalize_human_power_expr(value: &str) -> String {
    value.replace("((-1))", "(-1)")
}

fn same_math_render(left: &str, right: &str) -> bool {
    let mut left = left.to_string();
    let mut right = right.to_string();
    left.retain(|ch| !ch.is_whitespace());
    right.retain(|ch| !ch.is_whitespace());
    left == right
}

fn first_common_factor(ctx: &Context, numerator: ExprId, denominator: ExprId) -> Option<ExprId> {
    let numerator_factors = cas_math::expr_nary::mul_factors(ctx, numerator);
    let mut denominator_factors = cas_math::expr_nary::mul_factors(ctx, denominator).to_vec();

    for numerator_factor in numerator_factors {
        if let Some(index) = denominator_factors
            .iter()
            .position(|denominator_factor| *denominator_factor == numerator_factor)
        {
            denominator_factors.remove(index);
            return Some(numerator_factor);
        }
    }

    None
}

fn quotient_after_cancel_once(
    ctx: &Context,
    numerator: ExprId,
    denominator: ExprId,
    common_factor: ExprId,
) -> Option<(String, String)> {
    let numerator_factors = cas_math::expr_nary::mul_factors(ctx, numerator);
    let denominator_factors = cas_math::expr_nary::mul_factors(ctx, denominator);

    let numerator_remaining = remove_first_factor(&numerator_factors, common_factor)?;
    let denominator_remaining = remove_first_factor(&denominator_factors, common_factor)?;

    let mut temp_ctx = ctx.clone();
    let quotient = build_quotient_from_factors(
        &mut temp_ctx,
        numerator_remaining.as_slice(),
        denominator_remaining.as_slice(),
    );
    Some(render_temp_expr(&temp_ctx, quotient))
}

fn next_common_factor_after_cancel(
    ctx: &Context,
    numerator: ExprId,
    denominator: ExprId,
    common_factor: ExprId,
) -> Option<ExprId> {
    let numerator_factors = cas_math::expr_nary::mul_factors(ctx, numerator);
    let denominator_factors = cas_math::expr_nary::mul_factors(ctx, denominator);

    let numerator_remaining = remove_first_factor(&numerator_factors, common_factor)?;
    let denominator_remaining = remove_first_factor(&denominator_factors, common_factor)?;

    numerator_remaining
        .into_iter()
        .find(|numerator_factor| denominator_remaining.contains(numerator_factor))
}

fn remove_first_factor(factors: &[ExprId], target: ExprId) -> Option<Vec<ExprId>> {
    let index = factors.iter().position(|factor| *factor == target)?;
    let mut remaining = factors.to_vec();
    remaining.remove(index);
    Some(remaining)
}

fn build_quotient_from_factors(
    ctx: &mut Context,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
) -> ExprId {
    let numerator = build_mul_expr_from_factors(ctx, numerator_factors);
    let denominator = build_mul_expr_from_factors(ctx, denominator_factors);

    if is_one_expr(ctx, denominator) {
        numerator
    } else {
        ctx.add(Expr::Div(numerator, denominator))
    }
}

fn build_mul_expr_from_factors(ctx: &mut Context, factors: &[ExprId]) -> ExprId {
    match factors {
        [] => ctx.add(Expr::Number(BigRational::from_integer(1.into()))),
        [only] => *only,
        _ => {
            let mut iter = factors.iter().copied();
            let first = iter.next().expect("non-empty factors");
            iter.fold(first, |acc, factor| ctx.add(Expr::Mul(acc, factor)))
        }
    }
}

fn is_one_expr(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if *n == BigRational::from_integer(1.into()))
}

fn split_product_for_cancellation(
    ctx: &Context,
    numerator: ExprId,
    denominator: ExprId,
) -> Option<(ExprId, ExprId)> {
    let Expr::Mul(left, right) = ctx.get(numerator) else {
        return None;
    };
    if *left == denominator {
        return Some((*right, *left));
    }
    if *right == denominator {
        return Some((*left, *right));
    }
    None
}

fn is_repeated_factor_product(ctx: &Context, expr: ExprId, factor: ExprId) -> bool {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return false;
    };
    *left == factor && *right == factor
}

fn common_factor_factorization_plan(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<(ExprId, Sign)> {
    let before_terms = AddView::from_expr(ctx, before);
    if before_terms.terms.len() < 2 {
        return None;
    }

    let (factor, grouped) = split_factorized_product(ctx, after)?;
    let grouped_terms = AddView::from_expr(ctx, grouped);
    if grouped_terms.terms.len() != before_terms.terms.len() {
        return None;
    }

    let mut stripped = before_terms
        .terms
        .iter()
        .map(|&(term, sign)| Some((strip_factor_from_term(ctx, term, factor)?, sign)))
        .collect::<Option<Vec<_>>>()?;
    let mut grouped = grouped_terms.terms.to_vec();

    sort_signed_terms_for_compare(ctx, &mut stripped);
    sort_signed_terms_for_compare(ctx, &mut grouped);

    if stripped.iter().zip(grouped.iter()).all(
        |((left_expr, left_sign), (right_expr, right_sign))| {
            left_sign == right_sign && same_presentational_expr(ctx, *left_expr, ctx, *right_expr)
        },
    ) {
        let kind = if grouped_terms
            .terms
            .iter()
            .any(|(_, sign)| *sign == Sign::Neg)
        {
            Sign::Neg
        } else {
            Sign::Pos
        };
        return Some((factor, kind));
    }

    None
}

fn split_factorized_product(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };
    if matches!(ctx.get(*left), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return Some((*right, *left));
    }
    if matches!(ctx.get(*right), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return Some((*left, *right));
    }
    None
}

fn strip_factor_from_term(ctx: &Context, term: ExprId, factor: ExprId) -> Option<ExprId> {
    let Expr::Mul(left, right) = ctx.get(term) else {
        return None;
    };
    if *left == factor {
        return Some(*right);
    }
    if *right == factor {
        return Some(*left);
    }
    None
}

fn sort_signed_terms_for_compare(ctx: &Context, terms: &mut [(ExprId, Sign)]) {
    terms.sort_by_key(|(expr, sign)| {
        let sign_key = match sign {
            Sign::Pos => 0,
            Sign::Neg => 1,
        };
        (sign_key, display_expr(ctx, *expr))
    });
}

fn is_square_of_expr(ctx: &Context, expr: ExprId, base: ExprId) -> bool {
    let Expr::Pow(pow_base, exponent) = ctx.get(expr) else {
        return false;
    };
    *pow_base == base && is_small_positive_integer(ctx, *exponent, 2)
}

fn perfect_square_form_latex(
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

fn same_presentational_expr(
    left_ctx: &Context,
    left_expr: ExprId,
    right_ctx: &Context,
    right_expr: ExprId,
) -> bool {
    display_expr(left_ctx, left_expr) == display_expr(right_ctx, right_expr)
}

#[derive(Clone, Copy)]
enum CubeIdentityKind {
    Sum,
    Difference,
}

struct CubeIdentityPlan {
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
}

#[derive(Clone, Copy)]
enum SixthPowerIdentityKind {
    Sum,
    Difference,
}

struct SixthPowerIdentityPlan {
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
}

struct ReciprocalExponentPlan;

fn cube_identity_terms(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId, CubeIdentityKind)> {
    match ctx.get(expr) {
        Expr::Sub(left, right) => Some((*left, *right, CubeIdentityKind::Difference)),
        Expr::Add(left, right) => match ctx.get(*right) {
            Expr::Neg(inner) => Some((*left, *inner, CubeIdentityKind::Difference)),
            _ => Some((*left, *right, CubeIdentityKind::Sum)),
        },
        _ => None,
    }
}

fn cube_identity_plan(ctx: &Context, before: ExprId, after: ExprId) -> Option<CubeIdentityPlan> {
    let (left_term, right_term, kind) = cube_identity_terms(ctx, before)?;

    let left_base = cube_base_from_term(ctx, left_term)?;
    let right_base = cube_base_from_term(ctx, right_term)?;

    let Expr::Mul(first_factor, second_factor) = ctx.get(after) else {
        return None;
    };
    if !linear_factor_matches(ctx, *first_factor, left_base, right_base, kind)
        && !linear_factor_matches(ctx, *second_factor, left_base, right_base, kind)
    {
        return None;
    }

    Some(CubeIdentityPlan {
        left_base,
        right_base,
        kind,
    })
}

fn cube_identity_plan_for_fraction_cancel(
    ctx: &Context,
    numerator: ExprId,
    denominator: ExprId,
) -> Option<CubeIdentityPlan> {
    let (left_term, right_term, kind) = cube_identity_terms(ctx, numerator)?;

    let left_base = cube_base_from_term(ctx, left_term)?;
    let right_base = cube_base_from_term(ctx, right_term)?;
    if !linear_factor_matches(ctx, denominator, left_base, right_base, kind) {
        return None;
    }

    Some(CubeIdentityPlan {
        left_base,
        right_base,
        kind,
    })
}

fn cube_factorized_identity_plan(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<CubeIdentityPlan> {
    let (left_term, right_term, kind) = cube_identity_terms(ctx, after)?;
    let left_base = cube_base_from_term(ctx, left_term)?;
    let right_base = cube_base_from_term(ctx, right_term)?;
    let Expr::Mul(first_factor, second_factor) = ctx.get(before) else {
        return None;
    };

    for (linear_factor, quadratic_factor) in [
        (*first_factor, *second_factor),
        (*second_factor, *first_factor),
    ] {
        if linear_factor_matches(ctx, linear_factor, left_base, right_base, kind)
            && quadratic_factor_matches(ctx, quadratic_factor, left_base, right_base, kind)
        {
            return Some(CubeIdentityPlan {
                left_base,
                right_base,
                kind,
            });
        }
    }

    None
}

fn sixth_power_identity_terms(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, SixthPowerIdentityKind)> {
    match ctx.get(expr) {
        Expr::Sub(left, right) => Some((*left, *right, SixthPowerIdentityKind::Difference)),
        Expr::Add(left, right) => match ctx.get(*right) {
            Expr::Neg(inner) => Some((*left, *inner, SixthPowerIdentityKind::Difference)),
            _ => Some((*left, *right, SixthPowerIdentityKind::Sum)),
        },
        _ => None,
    }
}

fn sixth_power_factorized_identity_plan(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<SixthPowerIdentityPlan> {
    let (left_term, right_term, kind) = sixth_power_identity_terms(ctx, after)?;
    let left_base = sixth_power_base_from_term(ctx, left_term)?;
    let right_base = sixth_power_base_from_term(ctx, right_term)?;
    if is_one(ctx, left_base) || is_one(ctx, right_base) {
        return None;
    }

    let Expr::Mul(first_factor, second_factor) = ctx.get(before) else {
        return None;
    };

    for (binomial_factor, quartic_factor) in [
        (*first_factor, *second_factor),
        (*second_factor, *first_factor),
    ] {
        if sixth_power_binomial_factor_matches(ctx, binomial_factor, left_base, right_base, kind)
            && sixth_power_quartic_factor_matches(ctx, quartic_factor, left_base, right_base, kind)
        {
            return Some(SixthPowerIdentityPlan {
                left_base,
                right_base,
                kind,
            });
        }
    }

    None
}

fn reciprocal_exponent_plan(ctx: &Context, before: ExprId) -> Option<ReciprocalExponentPlan> {
    let Expr::Pow(base, exponent) = ctx.get(before) else {
        return None;
    };
    if !is_integer_literal(ctx, *exponent, 2) {
        return None;
    }

    match ctx.get(*base) {
        Expr::Function(fn_id, args)
            if *fn_id == ctx.builtin_id(BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            let _ = args[0];
            Some(ReciprocalExponentPlan)
        }
        Expr::Pow(radicand, inner_exponent) if is_one_half(ctx, *inner_exponent) => {
            let _ = *radicand;
            Some(ReciprocalExponentPlan)
        }
        _ => None,
    }
}

fn cube_base_from_term(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exponent) if is_integer_literal(ctx, *exponent, 3) => Some(*base),
        _ if is_one(ctx, expr) => Some(expr),
        _ => None,
    }
}

fn sixth_power_base_from_term(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exponent) if is_integer_literal(ctx, *exponent, 6) => Some(*base),
        _ => None,
    }
}

fn linear_factor_matches(
    ctx: &Context,
    expr: ExprId,
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
) -> bool {
    match kind {
        CubeIdentityKind::Sum => match ctx.get(expr) {
            Expr::Add(left, right) => {
                (*left == left_base && *right == right_base)
                    || (*left == right_base && *right == left_base)
            }
            _ => false,
        },
        CubeIdentityKind::Difference => match ctx.get(expr) {
            Expr::Sub(left, right) => *left == left_base && *right == right_base,
            Expr::Add(left, right) => {
                *left == left_base
                    && matches!(ctx.get(*right), Expr::Neg(inner) if *inner == right_base)
            }
            _ => false,
        },
    }
}

fn quadratic_factor_matches(
    ctx: &Context,
    expr: ExprId,
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 3 {
        return false;
    }

    let has_left_square = terms
        .iter()
        .any(|(term, sign)| *sign == Sign::Pos && matches_square_of(ctx, *term, left_base));
    let has_right_square = terms
        .iter()
        .any(|(term, sign)| *sign == Sign::Pos && matches_square_of(ctx, *term, right_base));
    let mixed_sign = match kind {
        CubeIdentityKind::Sum => Sign::Neg,
        CubeIdentityKind::Difference => Sign::Pos,
    };
    let has_mixed = terms.iter().any(|(term, sign)| {
        *sign == mixed_sign && matches_unscaled_product(ctx, *term, left_base, right_base)
    });

    has_left_square && has_right_square && has_mixed
}

fn sixth_power_binomial_factor_matches(
    ctx: &Context,
    expr: ExprId,
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return false;
    }

    let has_left_square = terms
        .iter()
        .any(|(term, sign)| *sign == Sign::Pos && matches_square_of(ctx, *term, left_base));
    let right_sign = match kind {
        SixthPowerIdentityKind::Sum => Sign::Pos,
        SixthPowerIdentityKind::Difference => Sign::Neg,
    };
    let has_right_square = terms
        .iter()
        .any(|(term, sign)| *sign == right_sign && matches_square_of(ctx, *term, right_base));

    has_left_square && has_right_square
}

fn sixth_power_quartic_factor_matches(
    ctx: &Context,
    expr: ExprId,
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 3 {
        return false;
    }

    let has_left_fourth = terms
        .iter()
        .any(|(term, sign)| *sign == Sign::Pos && matches_fourth_power_of(ctx, *term, left_base));
    let has_right_fourth = terms
        .iter()
        .any(|(term, sign)| *sign == Sign::Pos && matches_fourth_power_of(ctx, *term, right_base));
    let mixed_sign = match kind {
        SixthPowerIdentityKind::Sum => Sign::Neg,
        SixthPowerIdentityKind::Difference => Sign::Pos,
    };
    let has_mixed = terms.iter().any(|(term, sign)| {
        *sign == mixed_sign && matches_product_of_squares(ctx, *term, left_base, right_base)
    });

    has_left_fourth && has_right_fourth && has_mixed
}

fn cube_identity_display(
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

fn cube_identity_latex(
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

fn cube_linear_factor_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
) -> String {
    let left = display_expr(ctx, left_base);
    let right = display_expr(ctx, right_base);
    match kind {
        CubeIdentityKind::Sum => format!("({left} + {right})"),
        CubeIdentityKind::Difference => format!("({left} - {right})"),
    }
}

fn cube_linear_factor_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
) -> String {
    let left = latex_expr(ctx, left_base);
    let right = latex_expr(ctx, right_base);
    match kind {
        CubeIdentityKind::Sum => format!("\\left({left} + {right}\\right)"),
        CubeIdentityKind::Difference => format!("\\left({left} - {right}\\right)"),
    }
}

fn cube_quadratic_factor_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
) -> String {
    let left_sq = format!("{}^2", display_expr(ctx, left_base));
    let right_sq = format!("{}^2", display_expr(ctx, right_base));
    let mixed = format!(
        "{}·{}",
        display_expr(ctx, left_base),
        display_expr(ctx, right_base)
    );
    match kind {
        CubeIdentityKind::Sum => format!("({left_sq} - {mixed} + {right_sq})"),
        CubeIdentityKind::Difference => format!("({left_sq} + {mixed} + {right_sq})"),
    }
}

fn cube_quadratic_factor_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
) -> String {
    let left_sq = format!("{{{}}}^{{2}}", latex_expr(ctx, left_base));
    let right_sq = format!("{{{}}}^{{2}}", latex_expr(ctx, right_base));
    let mixed = format!(
        "{}\\cdot {}",
        latex_expr(ctx, left_base),
        latex_expr(ctx, right_base)
    );
    match kind {
        CubeIdentityKind::Sum => {
            format!("\\left({left_sq} - {mixed} + {right_sq}\\right)")
        }
        CubeIdentityKind::Difference => {
            format!("\\left({left_sq} + {mixed} + {right_sq}\\right)")
        }
    }
}

fn cube_factorized_identity_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
) -> String {
    format!(
        "{}·{}",
        cube_linear_factor_display(ctx, left_base, right_base, kind),
        cube_quadratic_factor_display(ctx, left_base, right_base, kind)
    )
}

fn cube_factorized_identity_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: CubeIdentityKind,
) -> String {
    format!(
        "{}\\cdot {}",
        cube_linear_factor_latex(ctx, left_base, right_base, kind),
        cube_quadratic_factor_latex(ctx, left_base, right_base, kind)
    )
}

fn sixth_power_identity_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
) -> String {
    let op = match kind {
        SixthPowerIdentityKind::Sum => " + ",
        SixthPowerIdentityKind::Difference => " - ",
    };
    format!(
        "{}{}{}",
        sixth_power_display(ctx, left_base),
        op,
        sixth_power_display(ctx, right_base)
    )
}

fn sixth_power_identity_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
) -> String {
    let op = match kind {
        SixthPowerIdentityKind::Sum => " + ",
        SixthPowerIdentityKind::Difference => " - ",
    };
    format!(
        "{}{}{}",
        sixth_power_latex(ctx, left_base),
        op,
        sixth_power_latex(ctx, right_base)
    )
}

fn sixth_power_binomial_factor_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
) -> String {
    let left = squared_display(ctx, left_base);
    let right = squared_display(ctx, right_base);
    match kind {
        SixthPowerIdentityKind::Sum => format!("({left} + {right})"),
        SixthPowerIdentityKind::Difference => format!("({left} - {right})"),
    }
}

fn sixth_power_binomial_factor_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
) -> String {
    let left = squared_latex(ctx, left_base);
    let right = squared_latex(ctx, right_base);
    match kind {
        SixthPowerIdentityKind::Sum => format!("\\left({left} + {right}\\right)"),
        SixthPowerIdentityKind::Difference => format!("\\left({left} - {right}\\right)"),
    }
}

fn sixth_power_quartic_factor_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
) -> String {
    let left_fourth = fourth_power_display(ctx, left_base);
    let right_fourth = fourth_power_display(ctx, right_base);
    let mixed = format!(
        "{}·{}",
        squared_display(ctx, left_base),
        squared_display(ctx, right_base)
    );
    match kind {
        SixthPowerIdentityKind::Sum => format!("({left_fourth} - {mixed} + {right_fourth})"),
        SixthPowerIdentityKind::Difference => {
            format!("({left_fourth} + {mixed} + {right_fourth})")
        }
    }
}

fn sixth_power_quartic_factor_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
) -> String {
    let left_fourth = fourth_power_latex(ctx, left_base);
    let right_fourth = fourth_power_latex(ctx, right_base);
    let mixed = format!(
        "{}\\cdot {}",
        squared_latex(ctx, left_base),
        squared_latex(ctx, right_base)
    );
    match kind {
        SixthPowerIdentityKind::Sum => {
            format!("\\left({left_fourth} - {mixed} + {right_fourth}\\right)")
        }
        SixthPowerIdentityKind::Difference => {
            format!("\\left({left_fourth} + {mixed} + {right_fourth}\\right)")
        }
    }
}

fn sixth_power_factorized_identity_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
) -> String {
    format!(
        "{}·{}",
        sixth_power_binomial_factor_display(ctx, left_base, right_base, kind),
        sixth_power_quartic_factor_display(ctx, left_base, right_base, kind)
    )
}

fn sixth_power_factorized_identity_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
    kind: SixthPowerIdentityKind,
) -> String {
    format!(
        "{}\\cdot {}",
        sixth_power_binomial_factor_latex(ctx, left_base, right_base, kind),
        sixth_power_quartic_factor_latex(ctx, left_base, right_base, kind)
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

fn sixth_power_display(ctx: &Context, expr: ExprId) -> String {
    let display = display_expr(ctx, expr);
    if is_simple_power_base(ctx, expr) {
        format!("{display}^6")
    } else {
        format!("({display})^6")
    }
}

fn sixth_power_latex(ctx: &Context, expr: ExprId) -> String {
    let latex = latex_expr(ctx, expr);
    format!("{{{latex}}}^{{6}}")
}

fn difference_square_terms(
    ctx: &Context,
    first_factor: ExprId,
    second_factor: ExprId,
) -> Option<(ExprId, ExprId)> {
    difference_square_terms_ordered(ctx, first_factor, second_factor)
        .or_else(|| difference_square_terms_ordered(ctx, second_factor, first_factor))
}

fn difference_square_terms_ordered(
    ctx: &Context,
    sum_factor: ExprId,
    diff_factor: ExprId,
) -> Option<(ExprId, ExprId)> {
    let Expr::Add(sum_left, sum_right) = ctx.get(sum_factor) else {
        return None;
    };
    let Expr::Sub(diff_left, diff_right) = ctx.get(diff_factor) else {
        return None;
    };

    let sum_matches_direct = (*sum_left == *diff_left && *sum_right == *diff_right)
        || (*sum_left == *diff_right && *sum_right == *diff_left);
    if !sum_matches_direct {
        return None;
    }

    Some((*diff_left, *diff_right))
}

fn squared_display(ctx: &Context, expr: ExprId) -> String {
    let display = display_expr(ctx, expr);
    if is_simple_power_base(ctx, expr) {
        format!("{display}^2")
    } else {
        format!("({display})^2")
    }
}

fn squared_latex(ctx: &Context, expr: ExprId) -> String {
    let latex = latex_expr(ctx, expr);
    format!("{{{latex}}}^{{2}}")
}

fn fourth_power_display(ctx: &Context, expr: ExprId) -> String {
    let display = display_expr(ctx, expr);
    if is_simple_power_base(ctx, expr) {
        format!("{display}^4")
    } else {
        format!("({display})^4")
    }
}

fn fourth_power_latex(ctx: &Context, expr: ExprId) -> String {
    let latex = latex_expr(ctx, expr);
    format!("{{{latex}}}^{{4}}")
}

fn is_simple_power_base(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::Function(_, _)
    )
}

fn is_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.numer() == &1.into() && value.denom() == &1.into())
}

fn is_zero(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.numer() == &0.into() && value.denom() == &1.into())
}

fn is_negative_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.numer() == &(-1).into() && value.denom() == &1.into())
}

fn is_integer_literal(ctx: &Context, expr: ExprId, expected: i64) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(value) if value.numer() == &expected.into() && value.denom() == &1.into()
    )
}

fn positive_integer_literal_value(ctx: &Context, expr: ExprId) -> Option<num_bigint::BigInt> {
    let Expr::Number(value) = ctx.get(expr) else {
        return None;
    };
    if !value.is_integer() || value <= &BigRational::zero() {
        return None;
    }
    Some(value.to_integer())
}

fn scaled_log_term(ctx: &Context, expr: ExprId) -> Option<(num_bigint::BigInt, ExprId)> {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            if let Some(coeff) = positive_integer_literal_value(ctx, *left) {
                Some((coeff, *right))
            } else {
                positive_integer_literal_value(ctx, *right).map(|coeff| (coeff, *left))
            }
        }
        _ => None,
    }
}

fn general_log_base_and_arg(ctx: &Context, expr: ExprId) -> Option<(Option<ExprId>, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if !ctx.is_builtin(*fn_id, BuiltinFn::Log) {
        return None;
    }
    match args.as_slice() {
        [arg] => Some((None, *arg)),
        [base, arg] => Some((Some(*base), *arg)),
        _ => None,
    }
}

fn is_one_half(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.numer() == &1.into() && value.denom() == &2.into())
}

fn sqrt_radicand(ctx: &Context, expr: ExprId) -> Option<ExprId> {
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

fn abs_argument(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if *fn_id == ctx.builtin_id(BuiltinFn::Abs) && args.len() == 1 =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

fn difference_like_terms(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr) {
        Expr::Sub(left, right) => Some((*left, *right)),
        Expr::Add(left, right) => match ctx.get(*right) {
            Expr::Neg(inner) => Some((*left, *inner)),
            _ => None,
        },
        _ => None,
    }
}
