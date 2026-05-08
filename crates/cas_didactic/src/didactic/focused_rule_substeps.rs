use super::nested_fraction_analysis::NestedFractionPattern;
use super::SubStep;
use crate::didactic::try_as_fraction;
use crate::runtime::Step;
use cas_ast::ordering::compare_expr;
use cas_ast::views::{as_rational_const, square_free_decompose};
use cas_ast::{substitute_expr_by_id, BuiltinFn, Constant, Context, Expr, ExprId};
use cas_math::cancel_support::try_cancel_common_additive_terms_expr;
use cas_math::expr_destructure::{as_div, as_mul, as_pow};
use cas_math::expr_extract::extract_i64_multiplier_and_base_factors;
use cas_math::expr_extract::{
    extract_exp_argument, extract_log_base_argument_view, log10_base_sentinel,
};
use cas_math::expr_nary::build_balanced_mul;
use cas_math::expr_nary::{self, AddView, MulView, Sign};
use cas_math::poly_compare::poly_eq;
use cas_math::polynomial::Polynomial;
use cas_math::summation_support::{
    detect_factorized_telescoping_square_base, extract_linear_offset, extract_unit_shifted_base,
    try_extract_finite_aggregate_call, FiniteAggregateCall,
};
use cas_solver_core::quadratic_coeffs::{
    extract_quadratic_coefficients, extract_simplified_nonzero_quadratic_coefficients_with_state,
};
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
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
    let integration_by_parts_substeps = generate_integration_by_parts_substeps(ctx, step);
    if !integration_by_parts_substeps.is_empty() {
        return integration_by_parts_substeps;
    }

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

    let exponential_log_cancellation_substeps =
        generate_exponential_log_cancellation_substeps(ctx, step);
    if !exponential_log_cancellation_substeps.is_empty() {
        return exponential_log_cancellation_substeps;
    }

    if step.rule_name == "Collapse Exact Zero Additive Subexpression"
        && step.description == "Complete the Square"
    {
        return generate_complete_square_substeps(ctx, step);
    }

    if step.rule_name == "Collapse Exact Zero Additive Subexpression"
        && step.description == "Angle Sum/Diff Identity"
    {
        return generate_trig_angle_sum_diff_substeps(ctx, step);
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
        "Difference of Squares" | "Difference of Squares (Product to Difference)" => {
            generate_conjugate_product_rule_substeps(ctx, step)
        }
        "Expand" => generate_expand_substeps(ctx, step),
        "Collect Terms" => generate_collect_terms_substeps(ctx, step),
        "Factor Out With Division" => generate_factor_out_with_division_substeps(ctx, step),
        "Factorization" => generate_factorization_substeps(ctx, step),
        "Binomial Expansion" | "Auto Expand Power Sum" => {
            generate_binomial_expansion_substeps(ctx, step)
        }
        "Cancel Exact Additive Pairs" => generate_exact_additive_pair_cancel_substeps(ctx, step),
        "expand_log" => generate_expand_log_substeps(ctx, step),
        "Simplify" | "Canonicalize" => generate_simplify_substeps(ctx, step),
        "Evaluate Logarithms" => generate_evaluate_logarithms_substeps(ctx, step),
        "Factor Perfect Square in Logarithm" => {
            generate_factor_perfect_square_log_substeps(ctx, step)
        }
        "Log Inverse Power" => generate_log_inverse_power_substeps(ctx, step),
        "Log Contraction" => generate_log_contraction_substeps(ctx, step),
        "Change of Base" => generate_change_of_base_substeps(ctx, step),
        "Exponential Sum/Difference Identity" => {
            generate_exponential_sum_diff_identity_substeps(ctx, step)
        }
        "Exponential Reciprocal Identity" => {
            generate_exponential_reciprocal_identity_substeps(ctx, step)
        }
        "Exponential Power Identity" | "Power of a Power" => {
            generate_exponential_power_identity_substeps(ctx, step)
        }
        "Exponential-Log Power Inverse" => {
            generate_exponential_log_power_inverse_substeps(ctx, step)
        }
        "Finite Product" => generate_finite_product_substeps(ctx, step),
        "Finite Summation" => generate_finite_summation_substeps(ctx, step),
        "Number Theory Operations" => generate_number_theory_operation_substeps(ctx, step),
        "Pascal's Identity" => generate_pascal_identity_substeps(ctx, step),
        "Binomial Coefficient Symmetry" => generate_binomial_symmetry_substeps(ctx, step),
        "Cos Product Telescoping" => generate_cos_product_telescoping_substeps(ctx, step),
        "Dirichlet Kernel Identity" => generate_dirichlet_kernel_substeps(ctx, step),
        "Complete the Square" => generate_complete_square_substeps(ctx, step),
        "Product-to-Sum Identity" => generate_product_to_sum_substeps(step),
        "Square Double Angle Contraction" => {
            generate_square_double_angle_contraction_substeps(ctx, step)
        }
        "Hyperbolic Product-to-Sum Identity" => {
            generate_hyperbolic_product_to_sum_substeps(ctx, step)
        }
        "Hyperbolic Product-to-Sum and Triple-Angle Identity" => {
            generate_hyperbolic_product_to_sum_substeps(ctx, step)
        }
        "Sum-to-Product Identity" | "Sum-to-Product Identity Cancellation Bridge" => {
            generate_sum_to_product_substeps(ctx, step)
        }
        "Angle Sum/Diff Identity" => generate_trig_angle_sum_diff_substeps(ctx, step),
        "Hyperbolic Angle Sum/Difference Identity" => {
            generate_hyperbolic_angle_sum_diff_substeps(ctx, step)
        }
        "Hyperbolic Half-Angle Squares" => {
            generate_hyperbolic_half_angle_square_substeps(ctx, step)
        }
        "Hyperbolic Quotient Identity" => generate_hyperbolic_quotient_substeps(ctx, step),
        "Hyperbolic Composition" => generate_hyperbolic_composition_substeps(ctx, step),
        "Inverse Hyperbolic Log Identity" => generate_inverse_hyperbolic_log_substeps(ctx, step),
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
        "Power Reduction Identity" => generate_power_reduction_identity_substeps(ctx, step),
        "Quadruple Angle Expansion" => generate_quadruple_angle_identity_substeps(ctx, step),
        "Quintuple Angle Identity" => generate_quintuple_angle_identity_substeps(ctx, step),
        "Triple Angle Identity" | "Triple Angle Expansion" => {
            generate_triple_angle_identity_substeps(ctx, step)
        }
        "Sophie Germain Identity" => generate_sophie_germain_expansion_substeps(ctx, step),
        "Hyperbolic Triple-Angle Identity" => {
            generate_hyperbolic_triple_angle_identity_substeps(ctx, step)
        }
        "Hyperbolic Parity (Odd/Even)" => generate_trig_parity_substeps(ctx, step),
        "Half-Angle Tangent Identity" => generate_half_angle_tangent_substeps(ctx, step),
        "Reciprocal Trig Identity" => generate_reciprocal_trig_identity_substeps(ctx, step),
        "Trig Parity (Odd/Even)" => generate_trig_parity_substeps(ctx, step),
        "Trig Expansion" => generate_trig_expansion_substeps(ctx, step),
        "Trig Quotient" => generate_trig_quotient_substeps(ctx, step),
        "Cos-Diff / Sin-Diff Quotient" => generate_cos_diff_sin_diff_quotient_substeps(ctx, step),
        "Distributive Property" | "Distributive Property (Simple)" => {
            let substeps = generate_distributive_rule_substeps(ctx, step);
            if substeps.is_empty() {
                generate_reverse_nested_fraction_rule_substeps(ctx, step)
            } else {
                substeps
            }
        }
        "Pull Constant From Fraction" => generate_reverse_nested_fraction_rule_substeps(ctx, step),
        "Pythagorean Factor Form" => generate_pythagorean_factor_form_substeps(ctx, step),
        "Pythagorean High-Power Factor" => {
            generate_pythagorean_high_power_factor_substeps(ctx, step)
        }
        "Pythagorean Chain Identity" => generate_pythagorean_chain_identity_substeps(ctx, step),
        name if name.starts_with("Pythagorean Identity") => {
            generate_pythagorean_identity_substeps(ctx, step)
        }
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
        "Inverse Tan Relations" | "Inverse Trig Sum Identity" => {
            generate_inverse_trig_sum_relation_substeps(ctx, step)
        }
        "Inverse Trig Composition" => generate_inverse_trig_composition_substeps(ctx, step),
        "Subtraction Self-Cancel" => generate_subtraction_self_cancel_substeps(ctx, step),
        "Cancel Reciprocal Exponents" => generate_cancel_reciprocal_exponents_substeps(ctx, step),
        "Square of Square Root" => generate_square_of_square_root_substeps(ctx, step),
        "Polynomial Identity" => generate_polynomial_identity_exact_cancel_substeps(ctx, step),
        "Subtract Expanded Sum/Difference of Cubes Quotient" => {
            generate_subtract_expanded_cubes_quotient_substeps(ctx, step)
        }
        "Polynomial Product Normalize" => generate_polynomial_product_normalize_substeps(ctx, step),
        "Sqrt Perfect Square" | "Simplify Square Root" | "Simplify perfect square root" => {
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
    let mut substeps =
        if let Some(substep) = phase_shift_formula_substep(ctx, local_before, local_after) {
            vec![substep]
        } else if local_before != local_after {
            vec![concrete_expr_substep(
                ctx,
                "Usar una identidad de desfase",
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

fn phase_shift_formula_substep(ctx: &Context, before: ExprId, after: ExprId) -> Option<SubStep> {
    let before_is_add_sub = matches!(ctx.get(before), Expr::Add(_, _) | Expr::Sub(_, _));
    let after_is_add_sub = matches!(ctx.get(after), Expr::Add(_, _) | Expr::Sub(_, _));

    match (before_is_add_sub, after_is_add_sub) {
        (true, false) => Some(formula_substep(
            "Usar a·sin(u) + b·cos(u) = R·sin(u + φ)",
            "a·sin(u) + b·cos(u)",
            "R·sin(u + φ)",
            "a\\cdot\\sin(u)+b\\cdot\\cos(u)",
            "R\\cdot\\sin(u+\\varphi)",
        )),
        (false, true) => Some(formula_substep(
            "Expandir R·sin(u + φ)",
            "R·sin(u + φ)",
            "a·sin(u) + b·cos(u)",
            "R\\cdot\\sin(u+\\varphi)",
            "a\\cdot\\sin(u)+b\\cdot\\cos(u)",
        )),
        (true, true) => phase_shift_additive_passthrough_substep(ctx, before, after),
        (false, false) => phase_shift_shifted_trig_formula_substep(ctx, before, after),
    }
}

fn phase_shift_additive_passthrough_substep(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<SubStep> {
    let mut work = ctx.clone();
    let plan = try_cancel_common_additive_terms_expr(&mut work, before, after)?;
    if plan.new_lhs == before && plan.new_rhs == after {
        return None;
    }

    phase_shift_formula_substep(&work, plan.new_lhs, plan.new_rhs)?;
    Some(concrete_expr_substep(
        &work,
        "Aplicar la identidad de desfase al bloque que cambia",
        plan.new_lhs,
        plan.new_rhs,
    ))
}

fn phase_shift_shifted_trig_formula_substep(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<SubStep> {
    let before_trig = single_sin_cos_function_name(ctx, before)?;
    let after_trig = single_sin_cos_function_name(ctx, after)?;

    match (before_trig, after_trig) {
        ("sin", "cos") => Some(formula_substep(
            "Usar sin(u + φ) = cos(u - (π/2 - φ))",
            "sin(u + φ)",
            "cos(u - (π/2 - φ))",
            "\\sin(u+\\varphi)",
            "\\cos(u-(\\pi/2-\\varphi))",
        )),
        ("cos", "sin") => Some(formula_substep(
            "Usar cos(u - φ) = sin(u + (π/2 - φ))",
            "cos(u - φ)",
            "sin(u + (π/2 - φ))",
            "\\cos(u-\\varphi)",
            "\\sin(u+(\\pi/2-\\varphi))",
        )),
        _ => None,
    }
}

fn single_sin_cos_function_name(ctx: &Context, expr: ExprId) -> Option<&'static str> {
    let mut found = None;
    if collect_single_sin_cos_function_name(ctx, expr, &mut found) {
        found
    } else {
        None
    }
}

fn collect_single_sin_cos_function_name(
    ctx: &Context,
    expr: ExprId,
    found: &mut Option<&'static str>,
) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
                return record_single_sin_cos_name(found, "sin");
            }
            if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
                return record_single_sin_cos_name(found, "cos");
            }
            args.iter()
                .copied()
                .all(|arg| collect_single_sin_cos_function_name(ctx, arg, found))
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            collect_single_sin_cos_function_name(ctx, *left, found)
                && collect_single_sin_cos_function_name(ctx, *right, found)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            collect_single_sin_cos_function_name(ctx, *inner, found)
        }
        Expr::Matrix { data, .. } => data
            .iter()
            .copied()
            .all(|item| collect_single_sin_cos_function_name(ctx, item, found)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => true,
        Expr::Function(_, args) => args
            .iter()
            .copied()
            .all(|arg| collect_single_sin_cos_function_name(ctx, arg, found)),
    }
}

fn record_single_sin_cos_name(found: &mut Option<&'static str>, name: &'static str) -> bool {
    if found.is_some() {
        return false;
    }
    *found = Some(name);
    true
}

fn generate_combine_like_terms_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if step.description.contains("Cancel opposite terms") {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let Some((coeffs, literal_display, literal_factors)) =
        combine_like_terms_coeff_sum_plan(ctx, before)
    else {
        return Vec::new();
    };

    let (before_display, before_latex) = render_numeric_sum(&coeffs);
    let total = coeffs
        .iter()
        .fold(BigRational::from_integer(0.into()), |acc, coeff| {
            acc + coeff.clone()
        });
    let (after_display, after_latex) = render_numeric_value(&total);

    let mut substeps =
        generate_hidden_radical_extraction_before_like_terms_substeps(ctx, step, &literal_factors);
    substeps.push(
        SubStep::new(
            format!("Sumar los coeficientes que acompañan a {literal_display}"),
            before_display,
            after_display,
        )
        .with_before_latex(before_latex)
        .with_after_latex(after_latex),
    );
    substeps
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

    let factored_target = detect_factor_out_with_division_substep_target(
        ctx,
        step.after_local().unwrap_or(step.after),
    );
    let Some((factor_expr, inner_expr)) = factored_target else {
        let Some(factor_display) = detect_factor_out_with_division_substep_factor(
            ctx,
            step.after_local().unwrap_or(step.after),
        )
        .map(|expr| human_expr(ctx, expr))
        .or_else(|| {
            step.description
                .strip_prefix("Factor out ")
                .and_then(|tail| tail.strip_suffix(" from the whole expression"))
                .map(str::to_string)
        }) else {
            return Vec::new();
        };
        return vec![concrete_expr_substep(
            ctx,
            format!(
                "Reescribir los términos que no llevan {factor_display} usando el factor común"
            ),
            before,
            after,
        )];
    };

    let factor_display = human_expr(ctx, factor_expr);
    let factor_latex = latex_expr(ctx, factor_expr);
    let Some(expanded_terms) =
        factored_division_expanded_terms(ctx, inner_expr, &factor_display, &factor_latex)
    else {
        return Vec::new();
    };

    vec![
        formula_substep(
            format!("Reescribir cada término con el factor común {factor_display}"),
            &display_expr(ctx, before),
            &expanded_terms.plain,
            &latex_expr(ctx, before),
            &expanded_terms.latex,
        ),
        formula_substep(
            format!("Sacar el factor común {factor_display}"),
            &expanded_terms.plain,
            &display_expr(ctx, after),
            &expanded_terms.latex,
            &latex_expr(ctx, after),
        ),
    ]
}

struct FactoredDivisionExpandedTerms {
    plain: String,
    latex: String,
}

fn factored_division_expanded_terms(
    ctx: &Context,
    inner_expr: ExprId,
    factor_display: &str,
    factor_latex: &str,
) -> Option<FactoredDivisionExpandedTerms> {
    let terms = collect_add_chain_terms_readonly(ctx, inner_expr);
    if terms.len() < 2 {
        return None;
    }

    let mut plain = String::new();
    let mut latex = String::new();
    for (idx, signed_term) in terms.iter().enumerate() {
        let term_plain = human_expr(ctx, signed_term.term);
        let term_latex = latex_expr(ctx, signed_term.term);
        let plain_piece = format!("{factor_display}·({term_plain})");
        let latex_piece = format!("{factor_latex}\\cdot \\left({term_latex}\\right)");

        if idx == 0 {
            if signed_term.negative {
                plain.push('-');
                latex.push('-');
            }
        } else if signed_term.negative {
            plain.push_str(" - ");
            latex.push_str(" - ");
        } else {
            plain.push_str(" + ");
            latex.push_str(" + ");
        }

        plain.push_str(&plain_piece);
        latex.push_str(&latex_piece);
    }

    Some(FactoredDivisionExpandedTerms { plain, latex })
}

#[derive(Debug, Clone, Copy)]
struct SignedAddTerm {
    term: ExprId,
    negative: bool,
}

fn collect_add_chain_terms_readonly(ctx: &Context, expr: ExprId) -> Vec<SignedAddTerm> {
    let mut out = Vec::new();
    collect_add_chain_terms_readonly_into(ctx, expr, false, &mut out);
    out
}

fn collect_add_chain_terms_readonly_into(
    ctx: &Context,
    expr: ExprId,
    negative: bool,
    out: &mut Vec<SignedAddTerm>,
) {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            collect_add_chain_terms_readonly_into(ctx, *left, negative, out);
            collect_add_chain_terms_readonly_into(ctx, *right, negative, out);
        }
        Expr::Sub(left, right) => {
            collect_add_chain_terms_readonly_into(ctx, *left, negative, out);
            collect_add_chain_terms_readonly_into(ctx, *right, !negative, out);
        }
        Expr::Neg(inner) => collect_add_chain_terms_readonly_into(ctx, *inner, !negative, out),
        _ => out.push(SignedAddTerm {
            term: expr,
            negative,
        }),
    }
}

fn detect_factor_out_with_division_substep_target(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    if !ctx.is_mul_commutative(expr) {
        return None;
    }

    let factors = collect_mul_chain_factors_readonly(ctx, expr);
    for (idx, factor) in factors.iter().copied().enumerate() {
        let mut remaining = factors.clone();
        remaining.remove(idx);
        if remaining.is_empty() {
            continue;
        }
        if let Some(inner) = single_remaining_factor_with_division_by(ctx, &remaining, factor) {
            return Some((factor, inner));
        }
    }

    None
}

fn single_remaining_factor_with_division_by(
    ctx: &Context,
    remaining: &[ExprId],
    factor: ExprId,
) -> Option<ExprId> {
    if remaining.len() == 1 && contains_division_by_exact_factor(ctx, remaining[0], factor) {
        return Some(remaining[0]);
    }
    None
}

fn detect_factor_out_with_division_substep_factor(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    detect_factor_out_with_division_substep_target(ctx, expr)
        .map(|(factor, _inner)| factor)
        .or_else(|| detect_factor_out_with_division_substep_factor_from_flat_target(ctx, expr))
}

fn detect_factor_out_with_division_substep_factor_from_flat_target(
    ctx: &Context,
    expr: ExprId,
) -> Option<ExprId> {
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
    let Some(plan) = complete_square_substep_plan(ctx, before) else {
        return Vec::new();
    };

    plan.substeps
        .iter()
        .map(|substep| temp_ctx_substep(substep.title, &plan.work, substep.before, substep.after))
        .collect()
}

struct CompleteSquareSubstepPlan {
    work: Context,
    substeps: Vec<CompleteSquareSubstepExpr>,
}

struct CompleteSquareSubstepExpr {
    title: &'static str,
    before: ExprId,
    after: ExprId,
}

fn complete_square_substep_plan(ctx: &Context, expr: ExprId) -> Option<CompleteSquareSubstepPlan> {
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

        let substeps = if is_one(&work, leading_coeff) {
            let (balanced_expr, grouped_expr) = build_monic_complete_square_substep_exprs(
                &mut work,
                &var_name,
                linear_coeff,
                constant_term,
            );
            vec![
                CompleteSquareSubstepExpr {
                    title: "Añadir y restar el cuadrado del semicoeficiente",
                    before: expr,
                    after: balanced_expr,
                },
                CompleteSquareSubstepExpr {
                    title: "Agrupar el trinomio como cuadrado perfecto",
                    before: balanced_expr,
                    after: grouped_expr,
                },
            ]
        } else {
            let (factored_expr, balanced_expr, grouped_expr) =
                build_non_monic_complete_square_substep_exprs(
                    &mut work,
                    &var_name,
                    leading_coeff,
                    linear_coeff,
                    constant_term,
                );
            vec![
                CompleteSquareSubstepExpr {
                    title: "Extraer el coeficiente líder de los términos cuadráticos",
                    before: expr,
                    after: factored_expr,
                },
                CompleteSquareSubstepExpr {
                    title: "Añadir y restar el cuadrado del semicoeficiente dentro del paréntesis",
                    before: factored_expr,
                    after: balanced_expr,
                },
                CompleteSquareSubstepExpr {
                    title: "Agrupar el trinomio como cuadrado perfecto",
                    before: balanced_expr,
                    after: grouped_expr,
                },
            ]
        };

        return Some(CompleteSquareSubstepPlan { work, substeps });
    }

    None
}

fn build_monic_complete_square_substep_exprs(
    ctx: &mut Context,
    var_name: &str,
    linear_coeff: ExprId,
    constant_term: ExprId,
) -> (ExprId, ExprId) {
    let two = ctx.num(2);
    let var_expr = ctx.var(var_name);
    let var_squared = ctx.add(Expr::Pow(var_expr, two));
    let linear_term = ctx.add(Expr::Mul(linear_coeff, var_expr));
    let half_linear_raw = ctx.add(Expr::Div(linear_coeff, two));
    let half_linear = simplify_expr_in_context(ctx, half_linear_raw);
    let half_square = ctx.add(Expr::Pow(half_linear, two));

    let quadratic_with_linear = ctx.add(Expr::Add(var_squared, linear_term));
    let with_half_square = ctx.add(Expr::Add(quadratic_with_linear, half_square));
    let with_constant = ctx.add(Expr::Add(with_half_square, constant_term));
    let balanced_expr = ctx.add(Expr::Sub(with_constant, half_square));

    let completed_binomial = ctx.add(Expr::Add(var_expr, half_linear));
    let completed_square = ctx.add(Expr::Pow(completed_binomial, two));
    let tail_raw = ctx.add(Expr::Sub(constant_term, half_square));
    let tail = simplify_expr_in_context(ctx, tail_raw);
    let grouped_expr = ctx.add(Expr::Add(completed_square, tail));

    (balanced_expr, grouped_expr)
}

fn build_non_monic_complete_square_substep_exprs(
    ctx: &mut Context,
    var_name: &str,
    leading_coeff: ExprId,
    linear_coeff: ExprId,
    constant_term: ExprId,
) -> (ExprId, ExprId, ExprId) {
    let two = ctx.num(2);
    let var_expr = ctx.var(var_name);
    let var_squared = ctx.add(Expr::Pow(var_expr, two));

    let linear_over_leading_raw = ctx.add(Expr::Div(linear_coeff, leading_coeff));
    let linear_over_leading = simplify_expr_in_context(ctx, linear_over_leading_raw);
    let normalized_linear_term = ctx.add(Expr::Mul(linear_over_leading, var_expr));
    let normalized_quadratic = ctx.add(Expr::Add(var_squared, normalized_linear_term));
    let factored_quadratic = ctx.add(Expr::Mul(leading_coeff, normalized_quadratic));
    let factored_expr = ctx.add(Expr::Add(factored_quadratic, constant_term));

    let doubled_leading = ctx.add(Expr::Mul(two, leading_coeff));
    let half_linear_raw = ctx.add(Expr::Div(linear_coeff, doubled_leading));
    let half_linear = simplify_expr_in_context(ctx, half_linear_raw);
    let half_square = ctx.add(Expr::Pow(half_linear, two));

    let balanced_inner = ctx.add(Expr::Add(normalized_quadratic, half_square));
    let balanced_quadratic = ctx.add(Expr::Mul(leading_coeff, balanced_inner));
    let scaled_half_square = ctx.add(Expr::Mul(leading_coeff, half_square));
    let tail_raw = ctx.add(Expr::Sub(constant_term, scaled_half_square));
    let tail = simplify_expr_in_context(ctx, tail_raw);
    let balanced_expr = ctx.add(Expr::Add(balanced_quadratic, tail));

    let completed_binomial = ctx.add(Expr::Add(var_expr, half_linear));
    let completed_square = ctx.add(Expr::Pow(completed_binomial, two));
    let grouped_quadratic = ctx.add(Expr::Mul(leading_coeff, completed_square));
    let grouped_expr = ctx.add(Expr::Add(grouped_quadratic, tail));

    (factored_expr, balanced_expr, grouped_expr)
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
        if let Some(intermediate) = step.after_local().or_else(|| {
            let mut work = ctx.clone();
            build_fraction_expansion_intermediate(&mut work, before)
        }) {
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

fn build_fraction_expansion_intermediate(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(expr).clone() else {
        return None;
    };

    let terms = AddView::from_expr(ctx, numerator).terms;
    if terms.len() < 2 {
        return None;
    }

    let distributed_terms = terms
        .into_iter()
        .map(|(term, sign)| (ctx.add(Expr::Div(term, denominator)), sign))
        .collect::<Vec<_>>();

    Some(build_add_from_signed_terms(ctx, &distributed_terms))
}

fn is_fraction_expansion_simplify_pair(ctx: &Context, before: ExprId, after: ExprId) -> bool {
    let mut work = ctx.clone();
    let Some(intermediate) = build_fraction_expansion_intermediate(&mut work, before) else {
        return false;
    };
    let simplified = simplify_expr_in_context(&mut work, intermediate);
    compare_expr(&work, simplified, after) == Ordering::Equal
        || human_expr(&work, simplified) == human_expr(ctx, after)
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

    generate_mixed_fraction_split_intermediate_substeps(ctx, before, after).unwrap_or_default()
}

fn generate_mixed_fraction_split_intermediate_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<Vec<SubStep>> {
    let Expr::Div(_, denominator) = ctx.get(before) else {
        return None;
    };
    let (whole, whole_sign, remainder, remainder_sign, remainder_denominator) =
        mixed_fraction_split_parts(ctx, after, *denominator)?;
    if !same_expr(ctx, *denominator, remainder_denominator) {
        return None;
    }

    let mut work = ctx.clone();
    let whole_term = signed_expr(&mut work, whole, whole_sign);
    let product = work.add_raw(Expr::Mul(whole_term, *denominator));
    let intermediate_numerator = if remainder_sign == Sign::Pos {
        work.add_raw(Expr::Add(product, remainder))
    } else {
        work.add_raw(Expr::Sub(product, remainder))
    };
    let intermediate = work.add_raw(Expr::Div(intermediate_numerator, *denominator));
    Some(vec![
        mixed_ctx_substep(
            "Reescribir el numerador como parte entera por denominador más resto",
            ctx,
            before,
            &work,
            intermediate,
        ),
        temp_ctx_substep(
            "Separar la suma del numerador sobre el denominador",
            &work,
            intermediate,
            after,
        ),
    ])
}

fn mixed_fraction_split_parts(
    ctx: &Context,
    after: ExprId,
    source_denominator: ExprId,
) -> Option<(ExprId, Sign, ExprId, Sign, ExprId)> {
    let terms = AddView::from_expr(ctx, after).terms;
    if terms.len() != 2 {
        return None;
    }

    let mut whole = None;
    let mut remainder = None;
    for (term, sign) in terms {
        if let Some((numerator, denominator)) = as_div(ctx, term) {
            if same_expr(ctx, denominator, source_denominator) {
                if remainder.replace((numerator, sign, denominator)).is_some() {
                    return None;
                }
                continue;
            }
        }

        if whole.replace((term, sign)).is_some() {
            return None;
        }
    }

    let (whole, whole_sign) = whole?;
    let (remainder, remainder_sign, denominator) = remainder?;
    Some((whole, whole_sign, remainder, remainder_sign, denominator))
}

fn signed_expr(ctx: &mut Context, term: ExprId, sign: Sign) -> ExprId {
    match sign {
        Sign::Pos => term,
        Sign::Neg => ctx.add_raw(Expr::Neg(term)),
    }
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
    let quotient_substeps = generate_same_base_power_quotient_substeps(ctx, before, after);
    if !quotient_substeps.is_empty() {
        return quotient_substeps;
    }
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

fn generate_same_base_power_quotient_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Vec<SubStep> {
    let Some((numerator, denominator)) = as_div(ctx, before) else {
        return Vec::new();
    };
    let Some((numerator_base, numerator_exponent)) = as_pow(ctx, numerator) else {
        return Vec::new();
    };
    let Some((denominator_base, denominator_exponent)) = as_pow(ctx, denominator) else {
        return Vec::new();
    };
    if compare_expr(ctx, numerator_base, denominator_base) != Ordering::Equal {
        return Vec::new();
    }

    let mut work = ctx.clone();
    let negative_denominator_exponent = work.add(Expr::Neg(denominator_exponent));
    let denominator_as_negative_power =
        work.add(Expr::Pow(numerator_base, negative_denominator_exponent));
    let numerator_power = work.add(Expr::Pow(numerator_base, numerator_exponent));
    let intermediate = work.add(Expr::Mul(numerator_power, denominator_as_negative_power));
    let merged_exponent = work.add(Expr::Sub(numerator_exponent, denominator_exponent));
    let expected_after = work.add(Expr::Pow(numerator_base, merged_exponent));
    if compare_expr(&work, expected_after, after) != Ordering::Equal {
        return Vec::new();
    }

    vec![
        temp_ctx_substep(
            "Reescribir la división como potencia negativa",
            &work,
            before,
            intermediate,
        ),
        temp_ctx_substep(
            "Sumar los exponentes de la misma base",
            &work,
            intermediate,
            after,
        ),
    ]
}

fn generate_odd_half_power_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some(plan) = odd_half_power_simplify_plan(ctx, before, after) {
        return build_odd_half_power_simplify_substeps(ctx, after, plan, None);
    }

    if let Some(substeps) = generate_odd_half_power_simplify_substeps(ctx, step) {
        return substeps;
    }

    Vec::new()
}

fn generate_expand_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let conjugate_product = generate_conjugate_product_expansion_substeps(ctx, before, after);
    if !conjugate_product.is_empty() {
        return conjugate_product;
    }

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

    let sophie_germain = generate_sophie_germain_expansion_substeps(ctx, step);
    if !sophie_germain.is_empty() {
        return sophie_germain;
    }

    Vec::new()
}

fn generate_conjugate_product_rule_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    generate_conjugate_product_expansion_substeps(ctx, before, after)
}

fn generate_conjugate_product_expansion_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Vec<SubStep> {
    let Some((left_base, right_base)) =
        conjugate_product_difference_of_squares_plan(ctx, before, after)
    else {
        return Vec::new();
    };

    let left_display = human_expr(ctx, left_base);
    let right_display = human_expr(ctx, right_base);
    let left_latex = latex_expr(ctx, left_base);
    let right_latex = latex_expr(ctx, right_base);
    let intermediate_display = format!("({left_display})^2 - ({right_display})^2");
    let intermediate_latex = format!(
        "{} - {}",
        render_power2_latex(&left_latex),
        render_power2_latex(&right_latex)
    );

    vec![
        formula_substep(
            "Aplicar el producto de conjugados",
            &display_expr(ctx, before),
            &intermediate_display,
            &latex_expr(ctx, before),
            &intermediate_latex,
        ),
        formula_substep(
            "Simplificar las potencias",
            &intermediate_display,
            &display_expr(ctx, after),
            &intermediate_latex,
            &latex_expr(ctx, after),
        ),
    ]
}

fn conjugate_product_difference_of_squares_plan(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<(ExprId, ExprId)> {
    let factors = expr_nary::mul_leaves(ctx, before);
    if factors.len() != 2 {
        return None;
    }

    let (left_base, right_base) =
        cas_math::expr_relations::conjugate_add_sub_pair(ctx, factors[0], factors[1])?;
    let mut work = ctx.clone();
    let expected_raw = build_difference_of_squares_expansion(&mut work, left_base, right_base);
    let expected = simplify_expr_in_context(&mut work, expected_raw);
    if poly_eq(&work, expected, after) {
        Some((left_base, right_base))
    } else {
        None
    }
}

fn build_difference_of_squares_expansion(
    ctx: &mut Context,
    left_base: ExprId,
    right_base: ExprId,
) -> ExprId {
    let two = ctx.num(2);
    let left_sq = ctx.add_raw(Expr::Pow(left_base, two));
    let right_sq = ctx.add_raw(Expr::Pow(right_base, two));
    ctx.add_raw(Expr::Sub(left_sq, right_sq))
}

fn generate_factorization_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some(substeps) = generate_consecutive_telescoping_fraction_substeps(ctx, before, after) {
        return substeps;
    }

    if let Some((base, power)) = geometric_difference_factor_plan(ctx, before, after) {
        let base_display = human_expr(ctx, base);

        return vec![concrete_expr_substep(
            ctx,
            format!("Aquí la diferencia de potencias usa base {base_display} y exponente {power}"),
            before,
            after,
        )];
    }

    if let Some(base) = full_sixth_power_minus_one_factor_plan(ctx, before, after) {
        let base_display = human_expr(ctx, base);

        return vec![concrete_expr_substep(
            ctx,
            format!(
                "Aquí la diferencia de sexto grado se factoriza completamente con base {base_display}"
            ),
            before,
            after,
        )];
    }

    if let Some((factor, kind)) = common_factor_factorization_plan(ctx, before, after) {
        let factor_display = human_expr(ctx, factor);
        let _ = kind;
        return vec![concrete_expr_substep(
            ctx,
            format!("Aquí el factor común es {factor_display}"),
            before,
            after,
        )];
    }

    if let Some((left, right)) = difference_of_squares_bases(ctx, before) {
        let left_display = human_expr(ctx, left);
        let right_display = human_expr(ctx, right);

        return vec![concrete_expr_substep(
            ctx,
            format!("Aquí la diferencia de cuadrados usa bases {left_display} y {right_display}"),
            before,
            after,
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

    let sophie_germain = generate_sophie_germain_factorization_substeps(ctx, step);
    if !sophie_germain.is_empty() {
        return sophie_germain;
    }

    if let Some(vars) = alternating_cubic_vandermonde_plan(ctx, before, after) {
        return generate_alternating_cubic_vandermonde_substeps(ctx, before, vars);
    }

    Vec::new()
}

fn generate_alternating_cubic_vandermonde_substeps(
    ctx: &Context,
    before: ExprId,
    vars: (String, String, String),
) -> Vec<SubStep> {
    let (a, b, c) = vars;
    vec![
        vandermonde_pair_zero_substep(&a, &b, &c, &a, &b),
        vandermonde_pair_zero_substep(&a, &b, &c, &a, &c),
        vandermonde_pair_zero_substep(&a, &b, &c, &b, &c),
        vandermonde_remaining_factor_substep(ctx, before, &a, &b, &c),
    ]
}

fn vandermonde_pair_zero_substep(a: &str, b: &str, c: &str, left: &str, right: &str) -> SubStep {
    let before = match (left, right) {
        (left, right) if left == a && right == b => {
            format!("{a}^3 · ({a} - {c}) + {a}^3 · ({c} - {a}) + {c}^3 · ({a} - {a})")
        }
        (left, right) if left == a && right == c => {
            format!("{a}^3 · ({b} - {a}) + {b}^3 · ({a} - {a}) + {a}^3 · ({a} - {b})")
        }
        (left, right) if left == b && right == c => {
            format!("{a}^3 · ({b} - {b}) + {b}^3 · ({b} - {a}) + {b}^3 · ({a} - {b})")
        }
        _ => return SubStep::new("Comprobar anulación de factores", "", ""),
    };
    let before_latex = before.replace('·', "\\cdot");
    let title = format!("Si {left} = {right}, aparece el factor {left} - {right}");

    SubStep::new(title, before, "0")
        .with_before_latex(before_latex)
        .with_after_latex("0")
}

fn vandermonde_remaining_factor_substep(
    ctx: &Context,
    before: ExprId,
    a: &str,
    b: &str,
    c: &str,
) -> SubStep {
    let source_display = human_expr(ctx, before);
    let source_latex = latex_expr(ctx, before);
    let denominator_display = format!("({a} - {b}) · ({a} - {c}) · ({b} - {c})");
    let denominator_latex = format!("({a} - {b})({a} - {c})({b} - {c})");
    let quotient_display = format!("({source_display})/({denominator_display})");
    let quotient_latex = format!("\\frac{{{source_latex}}}{{{denominator_latex}}}");
    let remaining = format!("{a} + {b} + {c}");

    SubStep::new(
        format!("El cociente restante es {remaining}"),
        quotient_display,
        remaining.clone(),
    )
    .with_before_latex(quotient_latex)
    .with_after_latex(remaining)
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
    let Some((left, right, kind, power, local_focus_before, local_focus_after)) =
        binomial_power_terms(ctx, before)
            .map(|(left, right, kind, power)| (left, right, kind, power, before, after))
            .or_else(|| find_binomial_expansion_focus_substep_sides(ctx, step.before, step.after))
    else {
        return Vec::new();
    };

    let _ = (left, right);
    let identity_title = match (kind, power) {
        (BinomialSquareKind::Sum, 2) => "Cuadrado de la suma desarrollado",
        (BinomialSquareKind::Difference, 2) => "Cuadrado de la diferencia desarrollado",
        (BinomialSquareKind::Sum, 3) => "Cubo de la suma desarrollado",
        (BinomialSquareKind::Difference, 3) => "Cubo de la diferencia desarrollado",
        _ => return Vec::new(),
    };

    vec![concrete_expr_substep(
        ctx,
        identity_title,
        local_focus_before,
        local_focus_after,
    )]
}

fn generate_exact_additive_pair_cancel_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let Some((left, right)) = exact_opposite_additive_pair(ctx, before)
        .or_else(|| exact_opposite_additive_pair(ctx, step.before))
    else {
        return Vec::new();
    };

    let mut work = ctx.clone();
    let pair = work.add(Expr::Sub(left, right));
    let zero = work.num(0);
    vec![SubStep::new(
        "Cancelar términos opuestos exactos",
        human_expr(&work, pair),
        human_expr(&work, zero),
    )
    .with_before_latex(latex_expr(&work, pair))
    .with_after_latex(latex_expr(&work, zero))]
}

fn exact_opposite_additive_pair(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() < 3 {
        return None;
    }

    for (left, left_sign) in terms.iter().copied() {
        for (right, right_sign) in terms.iter().copied() {
            if left_sign == right_sign || !same_expr(ctx, left, right) {
                continue;
            }

            return if left_sign == Sign::Pos {
                Some((left, right))
            } else {
                Some((right, left))
            };
        }
    }

    None
}

fn find_binomial_expansion_focus_substep_sides(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<(ExprId, ExprId, BinomialSquareKind, i64, ExprId, ExprId)> {
    let mut candidates = Vec::new();
    collect_subexpr_ids(ctx, before, &mut candidates);

    for candidate in candidates {
        let Some((left, right, kind, power)) = binomial_power_terms(ctx, candidate) else {
            continue;
        };

        let mut work = ctx.clone();
        let expanded = build_binomial_expansion_expr(&mut work, left, right, kind, power)?;
        let intermediate = substitute_expr_by_id(&mut work, before, candidate, expanded);
        if compare_expr(&work, intermediate, after) == Ordering::Equal
            || same_presentational_expr(&work, intermediate, &work, after)
        {
            return Some((left, right, kind, power, candidate, expanded));
        }
    }

    None
}

fn generate_expand_log_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some(substep) = expanded_log_exp_cancellation_substep(ctx, after, step.after) {
        return vec![substep];
    }

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

fn expanded_log_exp_cancellation_substep(
    ctx: &Context,
    expanded_logs: ExprId,
    final_expr: ExprId,
) -> Option<SubStep> {
    if expanded_logs == final_expr {
        return None;
    }

    let terms = AddView::from_expr(ctx, expanded_logs).terms;
    if terms.len() < 2 {
        return None;
    }

    let mut temp_ctx = ctx.clone();
    let mut cancelled_terms = Vec::with_capacity(terms.len());
    for (term, sign) in terms {
        let log_arg = change_of_base_natural_log_argument(ctx, term)?;
        let exp_arg = extract_exp_argument(ctx, log_arg)?;
        cancelled_terms.push((exp_arg, sign));
    }

    let cancelled = build_add_from_signed_terms(&mut temp_ctx, &cancelled_terms);
    if compare_expr(&temp_ctx, cancelled, final_expr) != Ordering::Equal
        && !same_presentational_expr(&temp_ctx, cancelled, &temp_ctx, final_expr)
    {
        return None;
    }

    Some(concrete_expr_substep(
        ctx,
        "Cancelar cada logaritmo natural con su exponencial",
        expanded_logs,
        final_expr,
    ))
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

fn generate_exponential_log_cancellation_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if step.rule_name != "Exponential Sum/Difference Identity" {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(arg) = extract_exp_argument(ctx, before) else {
        return Vec::new();
    };
    let terms = AddView::from_expr(ctx, arg).terms.to_vec();
    if terms.len() < 2
        || !terms
            .iter()
            .any(|(term, _sign)| expression_contains_log(ctx, *term))
    {
        return Vec::new();
    }

    let mut work = ctx.clone();
    let Some(split_expr) = build_exp_sum_product_without_log_cancellation(&mut work, &terms) else {
        return Vec::new();
    };
    if compare_expr(&work, split_expr, after) == Ordering::Equal {
        return Vec::new();
    }

    let (before_plain, before_latex) = render_temp_expr(&work, before);
    let (split_plain, split_latex) = render_temp_expr(&work, split_expr);
    let (after_plain, after_latex) = render_temp_expr(&work, after);

    vec![
        formula_substep(
            "Separar la suma o resta del exponente en productos de exponenciales",
            &before_plain,
            &split_plain,
            &before_latex,
            &split_latex,
        ),
        formula_substep(
            "Cancelar e^(k·ln(u)) como potencia en cada factor",
            &split_plain,
            &after_plain,
            &split_latex,
            &after_latex,
        ),
    ]
}

fn build_exp_sum_product_without_log_cancellation(
    ctx: &mut Context,
    terms: &[(ExprId, Sign)],
) -> Option<ExprId> {
    let mut numerator_factors = Vec::new();
    let mut denominator_factors = Vec::new();

    for (term, sign) in terms {
        let exp_term = ctx.call_builtin(BuiltinFn::Exp, vec![*term]);
        match sign {
            Sign::Pos => numerator_factors.push(exp_term),
            Sign::Neg => denominator_factors.push(exp_term),
        }
    }

    if numerator_factors.len() + denominator_factors.len() < 2 {
        return None;
    }

    let numerator = expr_nary::build_balanced_mul(ctx, &numerator_factors);
    if denominator_factors.is_empty() {
        return Some(numerator);
    }

    let denominator = expr_nary::build_balanced_mul(ctx, &denominator_factors);
    if numerator_factors.is_empty() {
        let one = ctx.num(1);
        return Some(ctx.add(Expr::Div(one, denominator)));
    }

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn expression_contains_log(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, _)
            if {
                matches!(
                    ctx.builtin_of(*fn_id),
                    Some(BuiltinFn::Ln | BuiltinFn::Log | BuiltinFn::Log10)
                )
            } =>
        {
            true
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            expression_contains_log(ctx, *left) || expression_contains_log(ctx, *right)
        }
        Expr::Neg(inner) => expression_contains_log(ctx, *inner),
        _ => false,
    }
}

fn generate_exponential_sum_diff_identity_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some(substep) = exponential_sum_diff_identity_substep(ctx, after, false) {
        return vec![substep];
    }

    if let Some(substep) = exponential_sum_diff_identity_substep(ctx, before, true) {
        return vec![substep];
    }

    Vec::new()
}

fn exponential_sum_diff_identity_substep(
    ctx: &Context,
    exp_expr: ExprId,
    reverse: bool,
) -> Option<SubStep> {
    let exp_arg = extract_exp_argument(ctx, exp_expr)?;
    let (contract_title, expand_title, product_formula, exp_formula, product_latex, exp_latex) =
        match ctx.get(exp_arg) {
            Expr::Add(_, _) => (
                "Usar e^A · e^B = e^(A+B)",
                "Usar e^(A+B) = e^A · e^B",
                "e^A · e^B",
                "e^(A+B)",
                "{e}^{A}\\cdot {e}^{B}",
                "{e}^{A+B}",
            ),
            Expr::Sub(_, _) => (
                "Usar e^A / e^B = e^(A-B)",
                "Usar e^(A-B) = e^A / e^B",
                "e^A / e^B",
                "e^(A-B)",
                "\\frac{{e}^{A}}{{e}^{B}}",
                "{e}^{A-B}",
            ),
            _ => return None,
        };

    if reverse {
        Some(formula_substep(
            expand_title,
            exp_formula,
            product_formula,
            exp_latex,
            product_latex,
        ))
    } else {
        Some(formula_substep(
            contract_title,
            product_formula,
            exp_formula,
            product_latex,
            exp_latex,
        ))
    }
}

fn generate_exponential_reciprocal_identity_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some(substep) = exponential_reciprocal_identity_substep(ctx, after, false) {
        return vec![substep];
    }

    if let Some(substep) = exponential_reciprocal_identity_substep(ctx, before, true) {
        return vec![substep];
    }

    Vec::new()
}

fn exponential_reciprocal_identity_substep(
    ctx: &Context,
    exp_expr: ExprId,
    reverse: bool,
) -> Option<SubStep> {
    let exp_arg = extract_exp_argument(ctx, exp_expr)?;
    if !matches!(ctx.get(exp_arg), Expr::Neg(_)) {
        return None;
    }

    if reverse {
        Some(formula_substep(
            "Usar e^(-A) = 1/e^A",
            "e^(-A)",
            "1/e^A",
            "{e}^{-A}",
            "\\frac{1}{{e}^{A}}",
        ))
    } else {
        Some(formula_substep(
            "Usar 1/e^A = e^(-A)",
            "1/e^A",
            "e^(-A)",
            "\\frac{1}{{e}^{A}}",
            "{e}^{-A}",
        ))
    }
}

fn generate_exponential_power_identity_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some(substep) = exponential_power_identity_substep(ctx, after, false) {
        return vec![substep];
    }

    if let Some(substep) = exponential_power_identity_substep(ctx, before, true) {
        return vec![substep];
    }

    Vec::new()
}

fn exponential_power_identity_substep(
    ctx: &Context,
    exp_expr: ExprId,
    reverse: bool,
) -> Option<SubStep> {
    let exp_arg = extract_exp_argument(ctx, exp_expr)?;
    if !matches!(ctx.get(exp_arg), Expr::Mul(_, _)) {
        return None;
    }

    if reverse {
        Some(formula_substep(
            "Usar e^(n·A) = (e^A)^n",
            "e^(n·A)",
            "(e^A)^n",
            "{e}^{n\\cdot A}",
            "({e}^{A})^{n}",
        ))
    } else {
        Some(formula_substep(
            "Usar (e^A)^n = e^(n·A)",
            "(e^A)^n",
            "e^(n·A)",
            "({e}^{A})^{n}",
            "{e}^{n\\cdot A}",
        ))
    }
}

fn generate_exponential_log_power_inverse_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(plan) = exponential_log_power_inverse_plan(ctx, before) else {
        return Vec::new();
    };

    let mut temp_ctx = ctx.clone();
    let core_before = temp_ctx.add(Expr::Pow(plan.source_base, plan.log_expr));
    let outer_exponent = build_balanced_mul(&mut temp_ctx, &plan.outer_factors);
    let intermediate = temp_ctx.add(Expr::Pow(core_before, outer_exponent));
    let expected_after = temp_ctx.add(Expr::Pow(plan.log_arg, outer_exponent));
    if compare_expr(&temp_ctx, expected_after, after) != Ordering::Equal {
        return Vec::new();
    }

    let (core_before_plain, core_before_latex) = render_temp_expr(&temp_ctx, core_before);
    let (log_arg_plain, log_arg_latex) = render_temp_expr(&temp_ctx, plan.log_arg);
    let (intermediate_plain, intermediate_latex) = render_temp_expr(&temp_ctx, intermediate);
    let (after_plain, after_latex) = render_temp_expr(&temp_ctx, after);

    let identity_title = match plan.base_kind {
        LogInversePowerBaseKind::Natural => "Usar que e^(ln(u)) = u",
        LogInversePowerBaseKind::Decimal => "Usar que 10^(log10(u)) = u",
        LogInversePowerBaseKind::Explicit => "Usar que a^(log(a, u)) = u",
    };

    vec![
        formula_substep(
            identity_title,
            &core_before_plain,
            &log_arg_plain,
            &core_before_latex,
            &log_arg_latex,
        ),
        formula_substep(
            "Aplicar el factor exterior como exponente",
            &intermediate_plain,
            &after_plain,
            &intermediate_latex,
            &after_latex,
        ),
    ]
}

fn generate_log_inverse_power_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(plan) = log_inverse_power_plan(ctx, before) else {
        return Vec::new();
    };
    let Some((_source_base, exponent)) = as_pow(ctx, before) else {
        return Vec::new();
    };

    let mut temp_ctx = ctx.clone();
    let target_base = match plan.base_kind {
        LogInversePowerBaseKind::Natural => temp_ctx.add(Expr::Constant(Constant::E)),
        LogInversePowerBaseKind::Decimal => temp_ctx.num(10),
        LogInversePowerBaseKind::Explicit => {
            let Some(explicit_target_base) = plan.explicit_target_base else {
                return Vec::new();
            };
            explicit_target_base
        }
    };
    let recovery = temp_ctx.add(Expr::Pow(target_base, plan.log_expr));
    let intermediate = temp_ctx.add(Expr::Pow(recovery, exponent));

    let (source_plain, source_latex) = render_temp_expr(&temp_ctx, plan.source_base);
    let (recovery_plain, recovery_latex) = render_temp_expr(&temp_ctx, recovery);
    let (intermediate_plain, intermediate_latex) = render_temp_expr(&temp_ctx, intermediate);
    let (after_plain, after_latex) = render_temp_expr(&temp_ctx, after);

    let identity_title = match plan.base_kind {
        LogInversePowerBaseKind::Natural => "Usar que e^(ln(u)) = u",
        LogInversePowerBaseKind::Decimal => "Usar que 10^(log10(u)) = u",
        LogInversePowerBaseKind::Explicit => "Usar que a^(log(a, u)) = u",
    };
    let cancel_title = match plan.base_kind {
        LogInversePowerBaseKind::Natural => {
            "El exponente exterior cancela el ln del exponente interior"
        }
        LogInversePowerBaseKind::Decimal => {
            "El exponente exterior cancela el log10 del exponente interior"
        }
        LogInversePowerBaseKind::Explicit => {
            "El exponente exterior cancela el logaritmo del exponente interior"
        }
    };

    vec![
        formula_substep(
            identity_title,
            &source_plain,
            &recovery_plain,
            &source_latex,
            &recovery_latex,
        ),
        formula_substep(
            cancel_title,
            &intermediate_plain,
            &after_plain,
            &intermediate_latex,
            &after_latex,
        ),
    ]
}

#[derive(Clone, Copy)]
enum LogInversePowerBaseKind {
    Natural,
    Decimal,
    Explicit,
}

struct LogInversePowerPlan {
    source_base: ExprId,
    log_expr: ExprId,
    explicit_target_base: Option<ExprId>,
    base_kind: LogInversePowerBaseKind,
}

struct ExponentialLogPowerInversePlan {
    source_base: ExprId,
    log_expr: ExprId,
    log_arg: ExprId,
    outer_factors: Vec<ExprId>,
    base_kind: LogInversePowerBaseKind,
}

fn exponential_log_power_inverse_plan(
    ctx: &Context,
    expr: ExprId,
) -> Option<ExponentialLogPowerInversePlan> {
    let (source_base, exponent) = as_pow(ctx, expr)?;
    let factors: Vec<ExprId> = MulView::from_expr(ctx, exponent)
        .factors
        .into_iter()
        .collect();
    let mut matched_index = None;
    let mut matched_log_expr = None;
    let mut matched_log_arg = None;
    let mut matched_base_kind = None;

    for (index, factor) in factors.iter().enumerate() {
        let Some((log_base_opt, log_arg)) = extract_log_base_argument_view(ctx, *factor) else {
            continue;
        };
        let Some(base_kind) = exponential_log_inverse_base_kind(ctx, source_base, log_base_opt)
        else {
            continue;
        };
        if matched_index.is_some() {
            return None;
        }
        matched_index = Some(index);
        matched_log_expr = Some(*factor);
        matched_log_arg = Some(log_arg);
        matched_base_kind = Some(base_kind);
    }

    let matched_index = matched_index?;
    let outer_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(index, factor)| (index != matched_index).then_some(*factor))
        .collect();
    if outer_factors.is_empty() {
        return None;
    }

    Some(ExponentialLogPowerInversePlan {
        source_base,
        log_expr: matched_log_expr?,
        log_arg: matched_log_arg?,
        outer_factors,
        base_kind: matched_base_kind?,
    })
}

fn exponential_log_inverse_base_kind(
    ctx: &Context,
    source_base: ExprId,
    log_base_opt: Option<ExprId>,
) -> Option<LogInversePowerBaseKind> {
    match log_base_opt {
        None if matches!(ctx.get(source_base), Expr::Constant(Constant::E)) => {
            Some(LogInversePowerBaseKind::Natural)
        }
        Some(base) if base == log10_base_sentinel() && is_integer_literal(ctx, source_base, 10) => {
            Some(LogInversePowerBaseKind::Decimal)
        }
        Some(base) if compare_expr(ctx, base, source_base) == Ordering::Equal => {
            Some(LogInversePowerBaseKind::Explicit)
        }
        _ => None,
    }
}

fn log_inverse_power_plan(ctx: &Context, expr: ExprId) -> Option<LogInversePowerPlan> {
    let (source_base, exponent) = as_pow(ctx, expr)?;

    let check_log_denom = |ctx: &Context,
                           denom: ExprId|
     -> Option<(ExprId, Option<ExprId>, LogInversePowerBaseKind)> {
        let (log_base_opt, log_arg) = extract_log_base_argument_view(ctx, denom)?;
        if compare_expr(ctx, log_arg, source_base) != Ordering::Equal {
            return None;
        }

        let (explicit_target_base, base_kind) = match log_base_opt {
            Some(base) if base == log10_base_sentinel() => (None, LogInversePowerBaseKind::Decimal),
            Some(base) => (Some(base), LogInversePowerBaseKind::Explicit),
            None => (None, LogInversePowerBaseKind::Natural),
        };
        Some((denom, explicit_target_base, base_kind))
    };

    if let Some((_coeff, denom)) = as_div(ctx, exponent) {
        let (log_expr, explicit_target_base, base_kind) = check_log_denom(ctx, denom)?;
        return Some(LogInversePowerPlan {
            source_base,
            log_expr,
            explicit_target_base,
            base_kind,
        });
    }

    if let Some((lhs, rhs)) = as_mul(ctx, exponent) {
        for maybe_inverse in [rhs, lhs] {
            let Some((den, den_exp)) = as_pow(ctx, maybe_inverse) else {
                continue;
            };
            if !is_integer_literal(ctx, den_exp, -1) {
                continue;
            }
            let (log_expr, explicit_target_base, base_kind) = check_log_denom(ctx, den)?;
            return Some(LogInversePowerPlan {
                source_base,
                log_expr,
                explicit_target_base,
                base_kind,
            });
        }
    }

    if let Some((den, den_exp)) = as_pow(ctx, exponent) {
        if is_integer_literal(ctx, den_exp, -1) {
            let (log_expr, explicit_target_base, base_kind) = check_log_denom(ctx, den)?;
            return Some(LogInversePowerPlan {
                source_base,
                log_expr,
                explicit_target_base,
                base_kind,
            });
        }
    }

    None
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

    if is_fraction_expansion_simplify_pair(ctx, before, after) {
        return generate_fraction_expansion_substeps(ctx, step);
    }

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
    radicand_power: i64,
    outside_power: i64,
}

fn generate_odd_half_power_simplify_substeps(ctx: &Context, step: &Step) -> Option<Vec<SubStep>> {
    let local_before = step.before_local().unwrap_or(step.before);
    let local_after = step.after_local().unwrap_or(step.after);
    if let Some(plan) = odd_half_power_simplify_plan(ctx, local_before, local_after) {
        return Some(build_odd_half_power_simplify_substeps(
            ctx,
            local_after,
            plan,
            odd_half_power_replacement_pair(step, local_before, local_after),
        ));
    }

    let (_focus_before, focus_after, plan) =
        find_additive_odd_half_power_simplify_focus(ctx, step.before, step.after)?;
    Some(build_odd_half_power_simplify_substeps(
        ctx,
        focus_after,
        plan,
        Some((step.before, step.after)),
    ))
}

fn build_odd_half_power_simplify_substeps(
    ctx: &Context,
    focus_after: ExprId,
    plan: OddHalfPowerSimplifyPlan,
    replacement_pair: Option<(ExprId, ExprId)>,
) -> Vec<SubStep> {
    let (radicand_display, radicand_latex) =
        power_display_and_latex(ctx, plan.base, plan.radicand_power);
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
            radicand_display,
            factorized_radicand_display.clone(),
        )
        .with_before_latex(radicand_latex)
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
    let (base, numerator) = odd_half_power_components(ctx, before)?;
    let outside_power = (numerator - 1) / 2;
    matches_odd_half_power_simplified_after(ctx, after, base, outside_power).then_some(
        OddHalfPowerSimplifyPlan {
            base,
            radicand_power: numerator,
            outside_power,
        },
    )
}

fn odd_half_power_components(ctx: &Context, before: ExprId) -> Option<(ExprId, i64)> {
    if let Some(radicand) = sqrt_radicand(ctx, before) {
        let Expr::Pow(base, exponent) = ctx.get(radicand) else {
            return None;
        };
        let numerator = small_positive_integer_value(ctx, *exponent)?;
        if numerator >= 3 && numerator % 2 == 1 {
            return Some((*base, numerator));
        }
    }

    let Expr::Pow(base, exponent) = ctx.get(before) else {
        return None;
    };
    let exponent = as_rational_const(ctx, *exponent, 8)?;
    if *exponent.denom() != 2.into() {
        return None;
    }
    let numerator = exponent.numer().to_string().parse::<i64>().ok()?;
    (numerator >= 3 && numerator % 2 == 1).then_some((*base, numerator))
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

fn generate_distributive_rule_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some((factor, terms)) = distributive_product_terms(ctx, before) else {
        return Vec::new();
    };
    let Some((products_display, products_latex)) = distributive_product_list(ctx, factor, &terms)
    else {
        return Vec::new();
    };

    vec![
        formula_substep(
            "Identificar los productos que genera la distributiva",
            &human_expr(ctx, before),
            &products_display,
            &latex_expr(ctx, before),
            &products_latex,
        ),
        formula_substep(
            "Escribir los productos con los signos originales",
            &products_display,
            &human_expr(ctx, after),
            &products_latex,
            &latex_expr(ctx, after),
        ),
    ]
}

fn distributive_product_terms(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<(ExprId, Sign)>)> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };

    let left_terms = AddView::from_expr(ctx, *left).terms;
    if left_terms.len() >= 2 {
        return Some((*right, left_terms.to_vec()));
    }

    let right_terms = AddView::from_expr(ctx, *right).terms;
    if right_terms.len() >= 2 {
        return Some((*left, right_terms.to_vec()));
    }

    None
}

fn distributive_product_list(
    ctx: &Context,
    factor: ExprId,
    terms: &[(ExprId, Sign)],
) -> Option<(String, String)> {
    if terms.len() < 2 {
        return None;
    }

    let products: Vec<_> = terms
        .iter()
        .map(|(term, sign)| signed_distributive_product(ctx, factor, *term, *sign))
        .collect();
    let displays = products
        .iter()
        .map(|(display, _latex)| display.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let latex = products
        .iter()
        .map(|(_display, latex)| latex.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    Some((displays, latex))
}

fn signed_distributive_product(
    ctx: &Context,
    factor: ExprId,
    term: ExprId,
    sign: Sign,
) -> (String, String) {
    let mut work = ctx.clone();
    let product = work.add_raw(Expr::Mul(factor, term));
    let signed_product = match sign {
        Sign::Pos => product,
        Sign::Neg => work.add_raw(Expr::Neg(product)),
    };
    (
        human_expr(&work, signed_product),
        latex_expr(&work, signed_product),
    )
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
    let Some((u, _gap_display, gap_is_one)) = telescoping_fraction_base_and_gap(ctx, after, before)
    else {
        return Vec::new();
    };
    let _ = u;

    if gap_is_one {
        return generate_consecutive_telescoping_common_denominator_substeps(
            ctx,
            after,
            before,
            TelescopingFractionSubstepDirection::Combine,
        )
        .unwrap_or_default();
    }

    generate_general_telescoping_common_denominator_substeps(
        ctx,
        after,
        before,
        TelescopingFractionSubstepDirection::Combine,
    )
    .unwrap_or_default()
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
    let (u, _gap_display, gap_is_one) = telescoping_fraction_base_and_gap(ctx, before, after)?;
    let _ = u;

    if gap_is_one {
        return generate_consecutive_telescoping_common_denominator_substeps(
            ctx,
            before,
            after,
            TelescopingFractionSubstepDirection::Split,
        );
    }

    generate_general_telescoping_common_denominator_substeps(
        ctx,
        before,
        after,
        TelescopingFractionSubstepDirection::Split,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TelescopingFractionSubstepDirection {
    Split,
    Combine,
}

fn generate_consecutive_telescoping_common_denominator_substeps(
    ctx: &Context,
    compact_expr: ExprId,
    split_expr: ExprId,
    direction: TelescopingFractionSubstepDirection,
) -> Option<Vec<SubStep>> {
    let (work, common_fraction) =
        consecutive_telescoping_common_fraction_work(ctx, compact_expr, split_expr)?;

    Some(match direction {
        TelescopingFractionSubstepDirection::Split => vec![
            temp_ctx_substep(
                "Introducir el numerador telescópico",
                &work,
                compact_expr,
                common_fraction,
            ),
            temp_ctx_substep(
                "Separar sobre el denominador común",
                &work,
                common_fraction,
                split_expr,
            ),
        ],
        TelescopingFractionSubstepDirection::Combine => vec![
            temp_ctx_substep(
                "Llevar las fracciones al denominador común",
                &work,
                split_expr,
                common_fraction,
            ),
            temp_ctx_substep(
                "Simplificar el numerador telescópico",
                &work,
                common_fraction,
                compact_expr,
            ),
        ],
    })
}

fn consecutive_telescoping_common_fraction_work(
    ctx: &Context,
    compact_expr: ExprId,
    split_expr: ExprId,
) -> Option<(Context, ExprId)> {
    let (num, den) = as_div(ctx, compact_expr)?;
    if !is_one(ctx, num) {
        return None;
    }

    let (u, u_plus_gap, gap_expr) = extract_telescoping_fraction_split_pattern(ctx, split_expr)?;
    if gap_expr.is_some() || !unit_gap_relation_holds(ctx, u, u_plus_gap) {
        return None;
    }
    if !matches_telescoping_fraction_denominator(ctx, den, u, u_plus_gap) {
        return None;
    }

    let mut work = ctx.clone();
    let numerator_difference = work.add(Expr::Sub(u_plus_gap, u));
    let common_fraction = work.add(Expr::Div(numerator_difference, den));
    Some((work, common_fraction))
}

fn generate_general_telescoping_common_denominator_substeps(
    ctx: &Context,
    compact_expr: ExprId,
    split_expr: ExprId,
    direction: TelescopingFractionSubstepDirection,
) -> Option<Vec<SubStep>> {
    let (work, common_fraction) =
        general_telescoping_common_fraction_work(ctx, compact_expr, split_expr)?;

    Some(match direction {
        TelescopingFractionSubstepDirection::Split => vec![
            temp_ctx_substep(
                "Introducir el numerador telescópico",
                &work,
                compact_expr,
                common_fraction,
            ),
            temp_ctx_substep(
                "Separar sobre el denominador común",
                &work,
                common_fraction,
                split_expr,
            ),
        ],
        TelescopingFractionSubstepDirection::Combine => vec![
            temp_ctx_substep(
                "Llevar las fracciones al denominador común",
                &work,
                split_expr,
                common_fraction,
            ),
            temp_ctx_substep(
                "Simplificar el numerador telescópico",
                &work,
                common_fraction,
                compact_expr,
            ),
        ],
    })
}

fn general_telescoping_common_fraction_work(
    ctx: &Context,
    compact_expr: ExprId,
    split_expr: ExprId,
) -> Option<(Context, ExprId)> {
    let (num, den) = as_div(ctx, compact_expr)?;
    if !is_one(ctx, num) {
        return None;
    }

    let (u, u_plus_gap, gap_expr) = extract_telescoping_fraction_split_pattern(ctx, split_expr)?;
    let gap_expr = gap_expr?;
    if !additive_gap_relation_holds(ctx, u, gap_expr, u_plus_gap) {
        return None;
    }
    if !matches_telescoping_fraction_denominator(ctx, den, u, u_plus_gap) {
        return None;
    }

    let mut work = ctx.clone();
    let numerator_difference = work.add(Expr::Sub(u_plus_gap, u));
    let scaled_denominator = work.add(Expr::Mul(gap_expr, den));
    let common_fraction = work.add(Expr::Div(numerator_difference, scaled_denominator));
    Some((work, common_fraction))
}

fn generate_finite_product_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if step
        .description
        .starts_with("Factorized telescoping product:")
    {
        return generate_factorized_finite_product_substeps(ctx, step);
    }

    if let Some(substeps) = generate_finite_product_closed_form_substeps(ctx, step) {
        return substeps;
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

fn generate_finite_sum_closed_form_substeps(ctx: &Context, step: &Step) -> Option<Vec<SubStep>> {
    let formula_title = finite_sum_closed_form_title(&step.description)?;
    build_finite_aggregate_closed_form_substeps(
        ctx,
        step,
        "sum",
        "Escribir la suma con sus extremos",
        formula_title,
        " + ",
        " + ",
    )
}

fn generate_finite_product_closed_form_substeps(
    ctx: &Context,
    step: &Step,
) -> Option<Vec<SubStep>> {
    let formula_title = finite_product_closed_form_title(&step.description)?;
    build_finite_aggregate_closed_form_substeps(
        ctx,
        step,
        "product",
        "Escribir el producto con sus extremos",
        formula_title,
        " · ",
        " \\cdot ",
    )
}

fn build_finite_aggregate_closed_form_substeps(
    ctx: &Context,
    step: &Step,
    callee_name: &str,
    expansion_title: &'static str,
    formula_title: &'static str,
    separator_plain: &str,
    separator_latex: &str,
) -> Option<Vec<SubStep>> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let call = try_extract_finite_aggregate_call(ctx, before, callee_name)?;
    let (series_plain, series_latex) =
        render_finite_aggregate_endpoint_series(ctx, &call, separator_plain, separator_latex);

    Some(vec![
        formula_substep(
            expansion_title,
            &human_expr(ctx, before),
            &series_plain,
            &latex_expr(ctx, before),
            &series_latex,
        ),
        formula_substep(
            formula_title,
            &series_plain,
            &human_expr(ctx, after),
            &series_latex,
            &latex_expr(ctx, after),
        ),
    ])
}

fn finite_sum_closed_form_title(description: &str) -> Option<&'static str> {
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

fn finite_product_closed_form_title(description: &str) -> Option<&'static str> {
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

fn render_finite_aggregate_endpoint_series(
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

fn finite_aggregate_successor_index(
    source_ctx: &Context,
    temp_ctx: &mut Context,
    expr: ExprId,
) -> ExprId {
    if let Some(value) = integer_value(source_ctx, expr) {
        return temp_ctx.num(value + 1);
    }
    let one = temp_ctx.num(1);
    temp_ctx.add(Expr::Add(expr, one))
}

fn generate_finite_summation_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if let Some(substeps) = generate_finite_sum_closed_form_substeps(ctx, step) {
        return substeps;
    }

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

fn generate_number_theory_operation_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some((n, k)) = choose_integer_args(ctx, before) else {
        return Vec::new();
    };
    let Some(value) = integer_value(ctx, after) else {
        return Vec::new();
    };
    if n < 0 || k < 0 || k > n {
        return Vec::new();
    }

    let complement = n - k;
    let quotient_plain = format!("{n}! / ({k}! · {complement}!)");
    let quotient_latex = format!("\\frac{{{n}!}}{{{k}!\\cdot {complement}!}}");

    vec![
        formula_substep(
            format!("Usar C({n},{k}) = {n}! / ({k}! · {complement}!)"),
            &binom_plain(n, k),
            &quotient_plain,
            &binom_latex(n, k),
            &quotient_latex,
        ),
        formula_substep(
            format!("Calcular {n}! / ({k}! · {complement}!) = {value}"),
            &quotient_plain,
            &value.to_string(),
            &quotient_latex,
            &latex_expr(ctx, after),
        ),
    ]
}

fn generate_pascal_identity_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some((n, k)) = pascal_choose_data(ctx, before, after) else {
        return Vec::new();
    };

    let next_n = n + 1;
    let next_k = k + 1;
    let before_plain = format!("{} + {}", binom_plain(n, k), binom_plain(n, k + 1));
    let after_plain = binom_plain(next_n, next_k);
    let before_latex = format!("{} + {}", binom_latex(n, k), binom_latex(n, k + 1));
    let after_latex = binom_latex(next_n, next_k);

    vec![formula_substep(
        format!("Usar C({n},{k}) + C({n},{}) = C({next_n},{next_k})", k + 1),
        &before_plain,
        &after_plain,
        &before_latex,
        &after_latex,
    )]
}

fn generate_binomial_symmetry_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some((n, k, complement)) = choose_symmetry_data(ctx, before, after) else {
        return Vec::new();
    };

    let complement_plain = format!("C({n}, {n} - {k})");
    let complement_latex = format!("\\binom{{{n}}}{{{n}-{k}}}");

    vec![
        formula_substep(
            format!("Usar C({n},{k}) = C({n},{n}-{k})"),
            &binom_plain(n, k),
            &complement_plain,
            &binom_latex(n, k),
            &complement_latex,
        ),
        formula_substep(
            format!("Calcular {n}-{k} = {complement}"),
            &complement_plain,
            &binom_plain(n, complement),
            &complement_latex,
            &binom_latex(n, complement),
        ),
    ]
}

fn pascal_choose_data(ctx: &Context, before: ExprId, after: ExprId) -> Option<(i64, i64)> {
    let Expr::Add(left, right) = ctx.get(before) else {
        return None;
    };
    let (n_left, k_left) = choose_integer_args(ctx, *left)?;
    let (n_right, k_right) = choose_integer_args(ctx, *right)?;
    if n_left != n_right {
        return None;
    }

    let lower_k = k_left.min(k_right);
    let upper_k = k_left.max(k_right);
    if upper_k - lower_k != 1 {
        return None;
    }

    let (after_n, after_k) = choose_integer_args(ctx, after)?;
    (after_n == n_left + 1 && after_k == lower_k + 1).then_some((n_left, lower_k))
}

fn choose_symmetry_data(ctx: &Context, before: ExprId, after: ExprId) -> Option<(i64, i64, i64)> {
    let (n, k) = choose_integer_args(ctx, before)?;
    let (after_n, after_k) = choose_integer_args(ctx, after)?;
    let complement = n - k;
    (n >= 0 && k >= 0 && k < complement && after_n == n && after_k == complement)
        .then_some((n, k, complement))
}

fn choose_integer_args(ctx: &Context, expr: ExprId) -> Option<(i64, i64)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 2 || !matches!(ctx.sym_name(*fn_id), "choose" | "nCr") {
        return None;
    }
    Some((integer_value(ctx, args[0])?, integer_value(ctx, args[1])?))
}

fn binom_plain(n: i64, k: i64) -> String {
    format!("C({n}, {k})")
}

fn binom_latex(n: i64, k: i64) -> String {
    format!("\\binom{{{n}}}{{{k}}}")
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
    let (base_u_plain, _) = render_factor_basis(ctx, &base_factors);
    let u_plain = if base_multiplier == 1 {
        base_u_plain
    } else {
        format!("{base_multiplier} · {base_u_plain}")
    };
    let title = if expands_morrie {
        identity_title_with_optional_u("Expandir la ley de Morrie", &u_plain)
    } else {
        identity_title_with_optional_u("Usar el telescopado de cosenos", &u_plain)
    };
    let (before_plain, after_plain, before_latex, after_latex) = if expands_morrie {
        (
            quotient_plain.as_str(),
            product_plain.as_str(),
            quotient_latex.as_str(),
            product_latex.as_str(),
        )
    } else {
        (
            product_plain.as_str(),
            quotient_plain.as_str(),
            product_latex.as_str(),
            quotient_latex.as_str(),
        )
    };

    vec![formula_substep(
        title,
        before_plain,
        after_plain,
        before_latex,
        after_latex,
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

    let (base_u_plain, _) = render_factor_basis(ctx, &base_factors);
    let u_plain = if base_multiplier == 1 {
        base_u_plain
    } else {
        format!("{base_multiplier} · {base_u_plain}")
    };
    let n_plain = n.to_string();
    let title = if expands_kernel {
        dirichlet_kernel_identity_title("Expandir el núcleo de Dirichlet", n, &u_plain)
    } else {
        dirichlet_kernel_identity_title("Usar el núcleo de Dirichlet", n, &u_plain)
    };
    let sum_plain = format!("1 + 2 · Σ_(k=1)^{n_plain} cos(k · u)");
    let quotient_plain = format!("sin(({n_plain} + 1/2)u) / sin(u/2)");
    let sum_latex = format!("1 + 2\\cdot \\sum_{{k=1}}^{{{n_plain}}}\\cos(k\\cdot u)");
    let quotient_latex =
        format!("\\frac{{\\sin(({n_plain}+\\frac{{1}}{{2}})u)}}{{\\sin(\\frac{{u}}{{2}})}}");
    let (before_plain, after_plain, before_latex, after_latex) = if expands_kernel {
        (
            quotient_plain.as_str(),
            sum_plain.as_str(),
            quotient_latex.as_str(),
            sum_latex.as_str(),
        )
    } else {
        (
            sum_plain.as_str(),
            quotient_plain.as_str(),
            sum_latex.as_str(),
            quotient_latex.as_str(),
        )
    };

    vec![formula_substep(
        title,
        before_plain,
        after_plain,
        before_latex,
        after_latex,
    )]
}

fn identity_title_with_optional_u(base_title: &str, u_plain: &str) -> String {
    if u_plain == "u" {
        base_title.to_string()
    } else {
        format!("{base_title} con u = {u_plain}")
    }
}

fn dirichlet_kernel_identity_title(base_title: &str, n: usize, u_plain: &str) -> String {
    if u_plain == "u" {
        format!("{base_title} con n = {n}")
    } else {
        format!("{base_title} con n = {n} y u = {u_plain}")
    }
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
        return Some(concrete_expr_substep(
            ctx,
            "Usar n · ln(|u|) = ln(u^n) cuando n es par",
            before,
            after,
        ));
    }

    if matches_general_log_power_contraction(ctx, before, after) {
        return Some(concrete_expr_substep(
            ctx,
            "Usar n · log_b(u) = log_b(u^n)",
            before,
            after,
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
    if let Some(substeps) =
        generate_inverse_trig_double_angle_expansion_substeps(ctx, before_expr, after_expr)
    {
        return substeps;
    }

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

fn generate_inverse_trig_double_angle_expansion_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<Vec<SubStep>> {
    let (outer, inverse, inverse_call) = inverse_trig_double_angle_parts(ctx, before)?;

    let mut work = ctx.clone();
    let two = work.num(2);
    let one = work.num(1);
    let sin_u = work.call_builtin(BuiltinFn::Sin, vec![inverse_call]);
    let cos_u = work.call_builtin(BuiltinFn::Cos, vec![inverse_call]);

    let expanded = match outer {
        BuiltinFn::Sin => build_balanced_mul(&mut work, &[two, sin_u, cos_u]),
        BuiltinFn::Cos => match inverse {
            BuiltinFn::Arcsin => {
                let sin_u_squared = work.add(Expr::Pow(sin_u, two));
                let double_sin_u_squared = build_balanced_mul(&mut work, &[two, sin_u_squared]);
                work.add(Expr::Sub(one, double_sin_u_squared))
            }
            BuiltinFn::Arccos => {
                let cos_u_squared = work.add(Expr::Pow(cos_u, two));
                let double_cos_u_squared = build_balanced_mul(&mut work, &[two, cos_u_squared]);
                work.add(Expr::Sub(double_cos_u_squared, one))
            }
            BuiltinFn::Arctan => {
                let cos_u_squared = work.add(Expr::Pow(cos_u, two));
                let sin_u_squared = work.add(Expr::Pow(sin_u, two));
                work.add(Expr::Sub(cos_u_squared, sin_u_squared))
            }
            _ => return None,
        },
        _ => return None,
    };

    Some(vec![
        temp_ctx_substep(
            "Expandir con la identidad de ángulo doble",
            &work,
            before,
            expanded,
        ),
        temp_ctx_substep(
            "Sustituir las razones trigonométricas inversas",
            &work,
            expanded,
            after,
        ),
    ])
}

fn inverse_trig_double_angle_parts(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, BuiltinFn, ExprId)> {
    let Expr::Function(outer_fn, outer_args) = ctx.get(expr) else {
        return None;
    };
    if outer_args.len() != 1 {
        return None;
    }

    let outer = ctx.builtin_of(*outer_fn)?;
    if !matches!(outer, BuiltinFn::Sin | BuiltinFn::Cos) {
        return None;
    }

    let (multiple, base_factors) = extract_i64_multiplier_and_base_factors(ctx, outer_args[0]);
    if multiple != 2 {
        return None;
    }
    let base_factors = base_factors.into_vec();
    if base_factors.len() != 1 {
        return None;
    }
    let inverse_call = base_factors[0];

    let Expr::Function(inverse_fn, inverse_args) = ctx.get(inverse_call) else {
        return None;
    };
    if inverse_args.len() != 1 {
        return None;
    }

    let inverse = match ctx.builtin_of(*inverse_fn)? {
        BuiltinFn::Arcsin | BuiltinFn::Asin => BuiltinFn::Arcsin,
        BuiltinFn::Arccos | BuiltinFn::Acos => BuiltinFn::Arccos,
        BuiltinFn::Arctan | BuiltinFn::Atan => BuiltinFn::Arctan,
        _ => return None,
    };

    Some((outer, inverse, inverse_call))
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

fn generate_trig_angle_sum_diff_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some(substep) = recursive_trig_angle_sum_diff_substep(ctx, before, false) {
        return vec![substep];
    }

    if let Some((title, compact_plain, expanded_plain, compact_latex, expanded_latex)) =
        trig_angle_sum_diff_formula(ctx, before)
    {
        return vec![formula_substep(
            title,
            compact_plain,
            expanded_plain,
            compact_latex,
            expanded_latex,
        )];
    }

    if let Some(substep) = recursive_trig_angle_sum_diff_substep(ctx, after, true) {
        return vec![substep];
    }

    if let Some((title, compact_plain, expanded_plain, compact_latex, expanded_latex)) =
        trig_angle_sum_diff_formula(ctx, after)
    {
        return vec![formula_substep(
            title,
            expanded_plain,
            compact_plain,
            expanded_latex,
            compact_latex,
        )];
    }

    Vec::new()
}

fn recursive_trig_angle_sum_diff_substep(
    ctx: &Context,
    expr: ExprId,
    reverse: bool,
) -> Option<SubStep> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let (multiple, base_factors) = extract_i64_multiplier_and_base_factors(ctx, args[0]);
    if multiple <= 1 {
        return None;
    }

    let mut work = ctx.clone();
    let base = build_balanced_mul(&mut work, &base_factors.into_vec());
    let (base_plain, _) = render_temp_expr(&work, base);
    let base_plain = human_formula_title_plain(&base_plain);
    let previous = multiple - 1;

    let (title, compact_plain, expanded_plain, compact_latex, expanded_latex) = match ctx
        .builtin_of(*fn_id)
    {
        Some(BuiltinFn::Sin) => {
            let title = format!(
                    "Usar sin({previous}u+u) = sin({previous}u) · cos(u) + cos({previous}u) · sin(u), con u = {base_plain}"
                );
            let compact_plain = format!("sin({multiple}u)");
            let expanded_plain = format!("sin({previous}u) · cos(u) + cos({previous}u) · sin(u)");
            let compact_latex = format!("\\sin({multiple}u)");
            let expanded_latex =
                format!("\\sin({previous}u)\\cdot\\cos(u)+\\cos({previous}u)\\cdot\\sin(u)");
            (
                title,
                compact_plain,
                expanded_plain,
                compact_latex,
                expanded_latex,
            )
        }
        Some(BuiltinFn::Cos) => {
            let title = format!(
                    "Usar cos({previous}u+u) = cos({previous}u) · cos(u) - sin({previous}u) · sin(u), con u = {base_plain}"
                );
            let compact_plain = format!("cos({multiple}u)");
            let expanded_plain = format!("cos({previous}u) · cos(u) - sin({previous}u) · sin(u)");
            let compact_latex = format!("\\cos({multiple}u)");
            let expanded_latex =
                format!("\\cos({previous}u)\\cdot\\cos(u)-\\sin({previous}u)\\cdot\\sin(u)");
            (
                title,
                compact_plain,
                expanded_plain,
                compact_latex,
                expanded_latex,
            )
        }
        _ => return None,
    };

    if reverse {
        Some(formula_substep(
            title,
            &expanded_plain,
            &compact_plain,
            &expanded_latex,
            &compact_latex,
        ))
    } else {
        Some(formula_substep(
            title,
            &compact_plain,
            &expanded_plain,
            &compact_latex,
            &expanded_latex,
        ))
    }
}

fn trig_angle_sum_diff_formula(
    ctx: &Context,
    expr: ExprId,
) -> Option<(
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let is_sum = match ctx.get(args[0]) {
        Expr::Add(_, _) => true,
        Expr::Sub(_, _) => false,
        _ => return None,
    };

    match (ctx.builtin_of(*fn_id), is_sum) {
        (Some(BuiltinFn::Sin), true) => Some((
            "Usar sin(A+B) = sin(A) · cos(B) + cos(A) · sin(B)",
            "sin(A+B)",
            "sin(A) · cos(B) + cos(A) · sin(B)",
            "\\sin(A+B)",
            "\\sin(A)\\cdot\\cos(B)+\\cos(A)\\cdot\\sin(B)",
        )),
        (Some(BuiltinFn::Sin), false) => Some((
            "Usar sin(A-B) = sin(A) · cos(B) - cos(A) · sin(B)",
            "sin(A-B)",
            "sin(A) · cos(B) - cos(A) · sin(B)",
            "\\sin(A-B)",
            "\\sin(A)\\cdot\\cos(B)-\\cos(A)\\cdot\\sin(B)",
        )),
        (Some(BuiltinFn::Cos), true) => Some((
            "Usar cos(A+B) = cos(A) · cos(B) - sin(A) · sin(B)",
            "cos(A+B)",
            "cos(A) · cos(B) - sin(A) · sin(B)",
            "\\cos(A+B)",
            "\\cos(A)\\cdot\\cos(B)-\\sin(A)\\cdot\\sin(B)",
        )),
        (Some(BuiltinFn::Cos), false) => Some((
            "Usar cos(A-B) = cos(A) · cos(B) + sin(A) · sin(B)",
            "cos(A-B)",
            "cos(A) · cos(B) + sin(A) · sin(B)",
            "\\cos(A-B)",
            "\\cos(A)\\cdot\\cos(B)+\\sin(A)\\cdot\\sin(B)",
        )),
        _ => None,
    }
}

fn generate_hyperbolic_angle_sum_diff_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some(substep) = recursive_hyperbolic_angle_sum_diff_substep(ctx, before, false) {
        return vec![substep];
    }

    if let Some((title, compact_plain, expanded_plain, compact_latex, expanded_latex)) =
        hyperbolic_angle_sum_diff_formula(ctx, before)
    {
        return vec![formula_substep(
            title,
            compact_plain,
            expanded_plain,
            compact_latex,
            expanded_latex,
        )];
    }

    if let Some(substep) = recursive_hyperbolic_angle_sum_diff_substep(ctx, after, true) {
        return vec![substep];
    }

    if let Some((title, compact_plain, expanded_plain, compact_latex, expanded_latex)) =
        hyperbolic_angle_sum_diff_formula(ctx, after)
    {
        return vec![formula_substep(
            title,
            expanded_plain,
            compact_plain,
            expanded_latex,
            compact_latex,
        )];
    }

    Vec::new()
}

fn recursive_hyperbolic_angle_sum_diff_substep(
    ctx: &Context,
    expr: ExprId,
    reverse: bool,
) -> Option<SubStep> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let (multiple, base_factors) = extract_i64_multiplier_and_base_factors(ctx, args[0]);
    if multiple <= 1 {
        return None;
    }

    let mut work = ctx.clone();
    let base = build_balanced_mul(&mut work, &base_factors.into_vec());
    let (base_plain, _) = render_temp_expr(&work, base);
    let base_plain = human_formula_title_plain(&base_plain);
    let previous = multiple - 1;

    let (title, compact_plain, expanded_plain, compact_latex, expanded_latex) = match ctx
        .builtin_of(*fn_id)
    {
        Some(BuiltinFn::Sinh) => {
            let title = format!(
                    "Usar sinh({previous}u+u) = sinh({previous}u) · cosh(u) + cosh({previous}u) · sinh(u), con u = {base_plain}"
                );
            let compact_plain = format!("sinh({multiple}u)");
            let expanded_plain =
                format!("sinh({previous}u) · cosh(u) + cosh({previous}u) · sinh(u)");
            let compact_latex = format!("\\sinh({multiple}u)");
            let expanded_latex =
                format!("\\sinh({previous}u)\\cdot\\cosh(u)+\\cosh({previous}u)\\cdot\\sinh(u)");
            (
                title,
                compact_plain,
                expanded_plain,
                compact_latex,
                expanded_latex,
            )
        }
        Some(BuiltinFn::Cosh) => {
            let title = format!(
                    "Usar cosh({previous}u+u) = cosh({previous}u) · cosh(u) + sinh({previous}u) · sinh(u), con u = {base_plain}"
                );
            let compact_plain = format!("cosh({multiple}u)");
            let expanded_plain =
                format!("cosh({previous}u) · cosh(u) + sinh({previous}u) · sinh(u)");
            let compact_latex = format!("\\cosh({multiple}u)");
            let expanded_latex =
                format!("\\cosh({previous}u)\\cdot\\cosh(u)+\\sinh({previous}u)\\cdot\\sinh(u)");
            (
                title,
                compact_plain,
                expanded_plain,
                compact_latex,
                expanded_latex,
            )
        }
        _ => return None,
    };

    if reverse {
        Some(formula_substep(
            title,
            &expanded_plain,
            &compact_plain,
            &expanded_latex,
            &compact_latex,
        ))
    } else {
        Some(formula_substep(
            title,
            &compact_plain,
            &expanded_plain,
            &compact_latex,
            &expanded_latex,
        ))
    }
}

fn hyperbolic_angle_sum_diff_formula(
    ctx: &Context,
    expr: ExprId,
) -> Option<(
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let is_sum = match ctx.get(args[0]) {
        Expr::Add(_, _) => true,
        Expr::Sub(_, _) => false,
        _ => return None,
    };

    match (ctx.builtin_of(*fn_id), is_sum) {
        (Some(BuiltinFn::Sinh), true) => Some((
            "Usar sinh(A+B) = sinh(A) · cosh(B) + cosh(A) · sinh(B)",
            "sinh(A+B)",
            "sinh(A) · cosh(B) + cosh(A) · sinh(B)",
            "\\sinh(A+B)",
            "\\sinh(A)\\cdot\\cosh(B)+\\cosh(A)\\cdot\\sinh(B)",
        )),
        (Some(BuiltinFn::Sinh), false) => Some((
            "Usar sinh(A-B) = sinh(A) · cosh(B) - cosh(A) · sinh(B)",
            "sinh(A-B)",
            "sinh(A) · cosh(B) - cosh(A) · sinh(B)",
            "\\sinh(A-B)",
            "\\sinh(A)\\cdot\\cosh(B)-\\cosh(A)\\cdot\\sinh(B)",
        )),
        (Some(BuiltinFn::Cosh), true) => Some((
            "Usar cosh(A+B) = cosh(A) · cosh(B) + sinh(A) · sinh(B)",
            "cosh(A+B)",
            "cosh(A) · cosh(B) + sinh(A) · sinh(B)",
            "\\cosh(A+B)",
            "\\cosh(A)\\cdot\\cosh(B)+\\sinh(A)\\cdot\\sinh(B)",
        )),
        (Some(BuiltinFn::Cosh), false) => Some((
            "Usar cosh(A-B) = cosh(A) · cosh(B) - sinh(A) · sinh(B)",
            "cosh(A-B)",
            "cosh(A) · cosh(B) - sinh(A) · sinh(B)",
            "\\cosh(A-B)",
            "\\cosh(A)\\cdot\\cosh(B)-\\sinh(A)\\cdot\\sinh(B)",
        )),
        (Some(BuiltinFn::Tanh), true) => Some((
            "Usar tanh(A+B) = (tanh(A) + tanh(B)) / (1 + tanh(A)·tanh(B))",
            "tanh(A+B)",
            "(tanh(A) + tanh(B)) / (1 + tanh(A)·tanh(B))",
            "\\tanh(A+B)",
            "\\frac{\\tanh(A)+\\tanh(B)}{1+\\tanh(A)\\cdot\\tanh(B)}",
        )),
        (Some(BuiltinFn::Tanh), false) => Some((
            "Usar tanh(A-B) = (tanh(A) - tanh(B)) / (1 - tanh(A)·tanh(B))",
            "tanh(A-B)",
            "(tanh(A) - tanh(B)) / (1 - tanh(A)·tanh(B))",
            "\\tanh(A-B)",
            "\\frac{\\tanh(A)-\\tanh(B)}{1-\\tanh(A)\\cdot\\tanh(B)}",
        )),
        _ => None,
    }
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
    let Some((title, product_plain, sum_plain, product_latex, sum_latex)) =
        product_to_sum_formula_from_description(step.description.as_str())
    else {
        return Vec::new();
    };

    vec![formula_substep(
        title,
        product_plain,
        sum_plain,
        product_latex,
        sum_latex,
    )]
}

fn product_to_sum_formula_from_description(
    description: &str,
) -> Option<(
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
)> {
    match description {
        "Expand 2·cos(A)·cos(B) into cos(A+B) + cos(A-B)" => Some((
            "Usar 2·cos(A)·cos(B) = cos(A+B) + cos(A-B)",
            "2·cos(A)·cos(B)",
            "cos(A+B) + cos(A-B)",
            "2\\cdot\\cos(A)\\cdot\\cos(B)",
            "\\cos(A+B)+\\cos(A-B)",
        )),
        "Expand 2·cos(A)·sin(B) into sin(A+B) - sin(A-B)" => Some((
            "Usar 2·cos(A)·sin(B) = sin(A+B) - sin(A-B)",
            "2·cos(A)·sin(B)",
            "sin(A+B) - sin(A-B)",
            "2\\cdot\\cos(A)\\cdot\\sin(B)",
            "\\sin(A+B)-\\sin(A-B)",
        )),
        "Expand 2·sin(A)·cos(B) into sin(A+B) + sin(A-B)" => Some((
            "Usar 2·sin(A)·cos(B) = sin(A+B) + sin(A-B)",
            "2·sin(A)·cos(B)",
            "sin(A+B) + sin(A-B)",
            "2\\cdot\\sin(A)\\cdot\\cos(B)",
            "\\sin(A+B)+\\sin(A-B)",
        )),
        "Expand 2·sin(A)·sin(B) into cos(A-B) - cos(A+B)" => Some((
            "Usar 2·sin(A)·sin(B) = cos(A-B) - cos(A+B)",
            "2·sin(A)·sin(B)",
            "cos(A-B) - cos(A+B)",
            "2\\cdot\\sin(A)\\cdot\\sin(B)",
            "\\cos(A-B)-\\cos(A+B)",
        )),
        _ => None,
    }
}

fn generate_hyperbolic_product_to_sum_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some((title, compact_plain, expanded_plain, compact_latex, expanded_latex)) =
        hyperbolic_product_to_sum_formula(ctx, before)
    {
        return vec![formula_substep(
            title,
            compact_plain,
            expanded_plain,
            compact_latex,
            expanded_latex,
        )];
    }

    if let Some((title, compact_plain, expanded_plain, compact_latex, expanded_latex)) =
        hyperbolic_product_to_sum_formula(ctx, after)
    {
        return vec![formula_substep(
            title,
            expanded_plain,
            compact_plain,
            expanded_latex,
            compact_latex,
        )];
    }

    Vec::new()
}

fn hyperbolic_product_to_sum_formula(
    ctx: &Context,
    expr: ExprId,
) -> Option<(
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
)> {
    hyperbolic_sum_to_product_formula(ctx, expr)
        .or_else(|| find_hyperbolic_product_to_sum_formula(ctx, expr))
}

fn hyperbolic_sum_to_product_formula(
    ctx: &Context,
    expr: ExprId,
) -> Option<(
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
)> {
    let (left, right, is_sum) = match ctx.get(expr) {
        Expr::Add(left, right) => (*left, *right, true),
        Expr::Sub(left, right) => (*left, *right, false),
        _ => return None,
    };
    let left_fn = extract_hyperbolic_function_name(ctx, left)?;
    let right_fn = extract_hyperbolic_function_name(ctx, right)?;
    if left_fn != right_fn {
        return None;
    }

    match (left_fn, is_sum) {
        ("sinh", true) => Some((
            "Usar sinh(A)+sinh(B) = 2·sinh((A+B)/2)·cosh((A-B)/2)",
            "sinh(A)+sinh(B)",
            "2·sinh((A+B)/2)·cosh((A-B)/2)",
            "\\sinh(A)+\\sinh(B)",
            "2\\cdot\\sinh((A+B)/2)\\cdot\\cosh((A-B)/2)",
        )),
        ("sinh", false) => Some((
            "Usar sinh(A)-sinh(B) = 2·cosh((A+B)/2)·sinh((A-B)/2)",
            "sinh(A)-sinh(B)",
            "2·cosh((A+B)/2)·sinh((A-B)/2)",
            "\\sinh(A)-\\sinh(B)",
            "2\\cdot\\cosh((A+B)/2)\\cdot\\sinh((A-B)/2)",
        )),
        ("cosh", true) => Some((
            "Usar cosh(A)+cosh(B) = 2·cosh((A+B)/2)·cosh((A-B)/2)",
            "cosh(A)+cosh(B)",
            "2·cosh((A+B)/2)·cosh((A-B)/2)",
            "\\cosh(A)+\\cosh(B)",
            "2\\cdot\\cosh((A+B)/2)\\cdot\\cosh((A-B)/2)",
        )),
        ("cosh", false) => Some((
            "Usar cosh(A)-cosh(B) = 2·sinh((A+B)/2)·sinh((A-B)/2)",
            "cosh(A)-cosh(B)",
            "2·sinh((A+B)/2)·sinh((A-B)/2)",
            "\\cosh(A)-\\cosh(B)",
            "2\\cdot\\sinh((A+B)/2)\\cdot\\sinh((A-B)/2)",
        )),
        _ => None,
    }
}

fn find_hyperbolic_product_to_sum_formula(
    ctx: &Context,
    expr: ExprId,
) -> Option<(
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
)> {
    hyperbolic_product_to_sum_formula_at_expr(ctx, expr).or_else(|| match ctx.get(expr) {
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            find_hyperbolic_product_to_sum_formula(ctx, *left)
                .or_else(|| find_hyperbolic_product_to_sum_formula(ctx, *right))
        }
        _ => None,
    })
}

fn hyperbolic_product_to_sum_formula_at_expr(
    ctx: &Context,
    expr: ExprId,
) -> Option<(
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
)> {
    let factors = expr_nary::mul_leaves(ctx, expr);
    let has_double_factor = factors
        .iter()
        .any(|factor| is_integer_number(ctx, *factor, 2));
    if !has_double_factor {
        return None;
    }

    let mut sinh_count = 0;
    let mut cosh_count = 0;
    for factor in factors {
        match extract_hyperbolic_function_name(ctx, factor) {
            Some("sinh") => sinh_count += 1,
            Some("cosh") => cosh_count += 1,
            _ => {}
        }
    }

    match (sinh_count, cosh_count) {
        (2, 0) => Some((
            "Usar 2·sinh(A)·sinh(B) = cosh(A+B) - cosh(A-B)",
            "2·sinh(A)·sinh(B)",
            "cosh(A+B) - cosh(A-B)",
            "2\\cdot\\sinh(A)\\cdot\\sinh(B)",
            "\\cosh(A+B)-\\cosh(A-B)",
        )),
        (1, 1) => Some((
            "Usar 2·sinh(A)·cosh(B) = sinh(A+B) + sinh(A-B)",
            "2·sinh(A)·cosh(B)",
            "sinh(A+B) + sinh(A-B)",
            "2\\cdot\\sinh(A)\\cdot\\cosh(B)",
            "\\sinh(A+B)+\\sinh(A-B)",
        )),
        (0, 2) => Some((
            "Usar 2·cosh(A)·cosh(B) = cosh(A+B) + cosh(A-B)",
            "2·cosh(A)·cosh(B)",
            "cosh(A+B) + cosh(A-B)",
            "2\\cdot\\cosh(A)\\cdot\\cosh(B)",
            "\\cosh(A+B)+\\cosh(A-B)",
        )),
        _ => None,
    }
}

fn extract_hyperbolic_function_name(ctx: &Context, expr: ExprId) -> Option<&str> {
    let Expr::Function(name, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    if ctx.is_builtin(*name, BuiltinFn::Sinh) {
        Some("sinh")
    } else if ctx.is_builtin(*name, BuiltinFn::Cosh) {
        Some("cosh")
    } else {
        None
    }
}

fn is_integer_number(ctx: &Context, expr: ExprId, value: i64) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(number) if number.is_integer() && *number.numer() == value.into()
    )
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

fn generate_hyperbolic_half_angle_square_substeps(_ctx: &Context, step: &Step) -> Vec<SubStep> {
    let description = step.description.as_str();

    if description.contains("cosh(u/2)^2") && description.contains("(cosh(u) + 1) / 2") {
        return vec![SubStep::new(
            "Usar cosh²(u/2) = (cosh(u) + 1) / 2",
            "cosh²(u/2)",
            "(cosh(u) + 1) / 2",
        )];
    }

    if description.contains("sinh(u/2)^2") && description.contains("(cosh(u) - 1) / 2") {
        return vec![SubStep::new(
            "Usar sinh²(u/2) = (cosh(u) - 1) / 2",
            "sinh²(u/2)",
            "(cosh(u) - 1) / 2",
        )];
    }

    Vec::new()
}

fn generate_triple_angle_identity_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some((kind, call_expr, base_factors)) = find_nested_trig_triple_angle_call(ctx, before) {
        return vec![build_trig_triple_angle_formula_substep(
            ctx,
            kind,
            call_expr,
            &base_factors,
            false,
        )];
    }

    if let Some((kind, call_expr, base_factors)) = find_nested_trig_triple_angle_call(ctx, after) {
        return vec![build_trig_triple_angle_formula_substep(
            ctx,
            kind,
            call_expr,
            &base_factors,
            true,
        )];
    }

    Vec::new()
}

#[derive(Clone, Copy)]
enum TrigPowerReductionKind {
    SinEvenPower,
    CosEvenPower,
    SinCosSquares,
}

fn generate_power_reduction_identity_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let Some((kind, arg)) = trig_power_reduction_kind_and_arg(ctx, before) else {
        return Vec::new();
    };

    vec![build_power_reduction_formula_substep(ctx, kind, arg)]
}

fn trig_power_reduction_kind_and_arg(
    ctx: &Context,
    expr: ExprId,
) -> Option<(TrigPowerReductionKind, ExprId)> {
    if let Some((trig_fn, arg)) = even_trig_power_arg(ctx, expr) {
        return match trig_fn {
            BuiltinFn::Sin => Some((TrigPowerReductionKind::SinEvenPower, arg)),
            BuiltinFn::Cos => Some((TrigPowerReductionKind::CosEvenPower, arg)),
            _ => None,
        };
    }

    trig_square_product_same_arg(ctx, expr).map(|arg| (TrigPowerReductionKind::SinCosSquares, arg))
}

fn even_trig_power_arg(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    let power = small_positive_integer_value(ctx, *exponent)?;
    if power < 4 || power % 2 != 0 {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(*base) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
        Some((BuiltinFn::Sin, args[0]))
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
        Some((BuiltinFn::Cos, args[0]))
    } else {
        None
    }
}

fn trig_square_product_same_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let mut sin_arg = None;
    let mut cos_arg = None;

    for factor in expr_nary::mul_leaves(ctx, expr) {
        if let Some(arg) = squared_trig_arg(ctx, factor, BuiltinFn::Sin) {
            if sin_arg.replace(arg).is_some() {
                return None;
            }
        } else if let Some(arg) = squared_trig_arg(ctx, factor, BuiltinFn::Cos) {
            if cos_arg.replace(arg).is_some() {
                return None;
            }
        } else {
            return None;
        }
    }

    match (sin_arg, cos_arg) {
        (Some(left), Some(right)) if left == right => Some(left),
        _ => None,
    }
}

fn squared_trig_arg(ctx: &Context, expr: ExprId, trig_fn: BuiltinFn) -> Option<ExprId> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if !is_small_positive_integer(ctx, *exponent, 2) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(*base) else {
        return None;
    };
    if args.len() == 1 && ctx.is_builtin(*fn_id, trig_fn) {
        Some(args[0])
    } else {
        None
    }
}

fn build_power_reduction_formula_substep(
    ctx: &Context,
    kind: TrigPowerReductionKind,
    arg: ExprId,
) -> SubStep {
    let arg_plain = human_formula_title_plain(&display_expr(ctx, arg));
    let (title, before_plain, after_plain, before_latex, after_latex) =
        power_reduction_formula_template(kind);
    let title = format!("{title}, con u = {arg_plain}");

    formula_substep(title, before_plain, after_plain, before_latex, after_latex)
}

fn power_reduction_formula_template(
    kind: TrigPowerReductionKind,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    match kind {
        TrigPowerReductionKind::SinEvenPower => (
            "Usar sin²(u) = (1 - cos(2u)) / 2 repetidamente",
            "sin(u)^2",
            "(1 - cos(2u)) / 2",
            "\\sin(u)^2",
            "\\frac{1-\\cos(2u)}{2}",
        ),
        TrigPowerReductionKind::CosEvenPower => (
            "Usar cos²(u) = (1 + cos(2u)) / 2 repetidamente",
            "cos(u)^2",
            "(1 + cos(2u)) / 2",
            "\\cos(u)^2",
            "\\frac{1+\\cos(2u)}{2}",
        ),
        TrigPowerReductionKind::SinCosSquares => (
            "Usar sin²(u)·cos²(u) = (1 - cos(4u)) / 8",
            "sin(u)^2 · cos(u)^2",
            "(1 - cos(4u)) / 8",
            "\\sin(u)^2\\cdot\\cos(u)^2",
            "\\frac{1-\\cos(4u)}{8}",
        ),
    }
}

fn generate_square_double_angle_contraction_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let Some(arg) = trig_square_product_same_arg(ctx, before) else {
        return Vec::new();
    };

    let arg_plain = human_formula_title_plain(&display_expr(ctx, arg));
    vec![formula_substep(
        format!("Usar sin²(u)·cos²(u) = sin²(2u) / 4, con u = {arg_plain}"),
        "sin(u)^2 · cos(u)^2",
        "sin(2u)^2 / 4",
        "\\sin(u)^2\\cdot\\cos(u)^2",
        "\\frac{\\sin(2u)^2}{4}",
    )]
}

#[derive(Clone, Copy)]
enum TrigTripleAngleKind {
    Sin,
    Cos,
    Tan,
}

fn find_nested_trig_triple_angle_call(
    ctx: &Context,
    expr: ExprId,
) -> Option<(TrigTripleAngleKind, ExprId, Vec<ExprId>)> {
    trig_triple_angle_call_at_expr(ctx, expr).or_else(|| match ctx.get(expr) {
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => find_nested_trig_triple_angle_call(ctx, *left)
            .or_else(|| find_nested_trig_triple_angle_call(ctx, *right)),
        Expr::Neg(inner) => find_nested_trig_triple_angle_call(ctx, *inner),
        _ => None,
    })
}

fn trig_triple_angle_call_at_expr(
    ctx: &Context,
    expr: ExprId,
) -> Option<(TrigTripleAngleKind, ExprId, Vec<ExprId>)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let kind = if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
        TrigTripleAngleKind::Sin
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
        TrigTripleAngleKind::Cos
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Tan) {
        TrigTripleAngleKind::Tan
    } else {
        return None;
    };

    let (multiple, base_factors) = extract_i64_multiplier_and_base_factors(ctx, args[0]);
    if multiple != 3 {
        return None;
    }

    Some((kind, expr, base_factors.into_vec()))
}

fn build_trig_triple_angle_formula_substep(
    ctx: &Context,
    kind: TrigTripleAngleKind,
    _call_expr: ExprId,
    base_factors: &[ExprId],
    reverse: bool,
) -> SubStep {
    let mut work = ctx.clone();
    let base = build_balanced_mul(&mut work, base_factors);
    let (base_plain, _) = render_temp_expr(&work, base);
    let base_plain = human_formula_title_plain(&base_plain);
    let (compact_plain, expanded_plain, compact_latex, expanded_latex) =
        trig_triple_angle_formula_template(kind);
    let title = format!(
        "Usar {} = {}, con u = {}",
        human_formula_title_plain(compact_plain),
        human_formula_title_plain(expanded_plain),
        base_plain
    );

    if reverse {
        formula_substep(
            title,
            expanded_plain,
            compact_plain,
            expanded_latex,
            compact_latex,
        )
    } else {
        formula_substep(
            title,
            compact_plain,
            expanded_plain,
            compact_latex,
            expanded_latex,
        )
    }
}

fn trig_triple_angle_formula_template(
    kind: TrigTripleAngleKind,
) -> (&'static str, &'static str, &'static str, &'static str) {
    match kind {
        TrigTripleAngleKind::Sin => (
            "sin(3u)",
            "3 · sin(u) - 4 · sin(u)^3",
            "\\sin(3u)",
            "3\\cdot\\sin(u)-4\\cdot\\sin(u)^3",
        ),
        TrigTripleAngleKind::Cos => (
            "cos(3u)",
            "4 · cos(u)^3 - 3 · cos(u)",
            "\\cos(3u)",
            "4\\cdot\\cos(u)^3-3\\cdot\\cos(u)",
        ),
        TrigTripleAngleKind::Tan => (
            "tan(3u)",
            "(3 · tan(u) - tan(u)^3) / (1 - 3 · tan(u)^2)",
            "\\tan(3u)",
            "\\frac{3\\cdot\\tan(u)-\\tan(u)^3}{1-3\\cdot\\tan(u)^2}",
        ),
    }
}

fn generate_quadruple_angle_identity_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some((kind, base_factors)) = nested_trig_quadruple_angle_call(ctx, before) {
        return vec![build_trig_quadruple_angle_formula_substep(
            ctx,
            kind,
            &base_factors,
            false,
        )];
    }

    if let Some((kind, base_factors)) = nested_trig_quadruple_angle_call(ctx, after) {
        return vec![build_trig_quadruple_angle_formula_substep(
            ctx,
            kind,
            &base_factors,
            true,
        )];
    }

    Vec::new()
}

#[derive(Clone, Copy)]
enum TrigQuadrupleAngleKind {
    Sin,
    Cos,
}

fn nested_trig_quadruple_angle_call(
    ctx: &Context,
    expr: ExprId,
) -> Option<(TrigQuadrupleAngleKind, Vec<ExprId>)> {
    trig_quadruple_angle_call_at_expr(ctx, expr).or_else(|| match ctx.get(expr) {
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => nested_trig_quadruple_angle_call(ctx, *left)
            .or_else(|| nested_trig_quadruple_angle_call(ctx, *right)),
        Expr::Neg(inner) => nested_trig_quadruple_angle_call(ctx, *inner),
        _ => None,
    })
}

fn trig_quadruple_angle_call_at_expr(
    ctx: &Context,
    expr: ExprId,
) -> Option<(TrigQuadrupleAngleKind, Vec<ExprId>)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let kind = match ctx.builtin_of(*fn_id)? {
        BuiltinFn::Sin => TrigQuadrupleAngleKind::Sin,
        BuiltinFn::Cos => TrigQuadrupleAngleKind::Cos,
        _ => return None,
    };

    let (multiple, base_factors) = extract_i64_multiplier_and_base_factors(ctx, args[0]);
    (multiple == 4).then(|| (kind, base_factors.into_vec()))
}

fn build_trig_quadruple_angle_formula_substep(
    ctx: &Context,
    kind: TrigQuadrupleAngleKind,
    base_factors: &[ExprId],
    reverse: bool,
) -> SubStep {
    let mut work = ctx.clone();
    let base = build_balanced_mul(&mut work, base_factors);
    let (base_plain, _) = render_temp_expr(&work, base);
    let base_plain = human_formula_title_plain(&base_plain);
    let (compact_plain, expanded_plain, compact_latex, expanded_latex) = match kind {
        TrigQuadrupleAngleKind::Sin => (
            "sin(4u)",
            "4 · sin(u) · cos(u)^3 - 4 · sin(u)^3 · cos(u)",
            "\\sin(4u)",
            "4\\cdot\\sin(u)\\cdot\\cos(u)^3-4\\cdot\\sin(u)^3\\cdot\\cos(u)",
        ),
        TrigQuadrupleAngleKind::Cos => (
            "cos(4u)",
            "8 · cos(u)^4 - 8 · cos(u)^2 + 1",
            "\\cos(4u)",
            "8\\cdot\\cos(u)^4-8\\cdot\\cos(u)^2+1",
        ),
    };
    let title = format!(
        "Usar {} = {}, con u = {}",
        human_formula_title_plain(compact_plain),
        human_formula_title_plain(expanded_plain),
        base_plain
    );

    if reverse {
        formula_substep(
            title,
            expanded_plain,
            compact_plain,
            expanded_latex,
            compact_latex,
        )
    } else {
        formula_substep(
            title,
            compact_plain,
            expanded_plain,
            compact_latex,
            expanded_latex,
        )
    }
}

#[derive(Clone, Copy)]
enum TrigQuintupleAngleKind {
    Sin,
    Cos,
}

fn generate_quintuple_angle_identity_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    if let Some((kind, base_factors)) = nested_trig_quintuple_angle_call(ctx, before) {
        return vec![build_trig_quintuple_angle_formula_substep(
            ctx,
            kind,
            &base_factors,
            false,
        )];
    }

    if let Some((kind, base_factors)) = nested_trig_quintuple_angle_call(ctx, after) {
        return vec![build_trig_quintuple_angle_formula_substep(
            ctx,
            kind,
            &base_factors,
            true,
        )];
    }

    Vec::new()
}

fn nested_trig_quintuple_angle_call(
    ctx: &Context,
    expr: ExprId,
) -> Option<(TrigQuintupleAngleKind, Vec<ExprId>)> {
    trig_quintuple_angle_call_at_expr(ctx, expr).or_else(|| match ctx.get(expr) {
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => nested_trig_quintuple_angle_call(ctx, *left)
            .or_else(|| nested_trig_quintuple_angle_call(ctx, *right)),
        Expr::Neg(inner) => nested_trig_quintuple_angle_call(ctx, *inner),
        _ => None,
    })
}

fn trig_quintuple_angle_call_at_expr(
    ctx: &Context,
    expr: ExprId,
) -> Option<(TrigQuintupleAngleKind, Vec<ExprId>)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let kind = if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
        TrigQuintupleAngleKind::Sin
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
        TrigQuintupleAngleKind::Cos
    } else {
        return None;
    };

    let (multiple, base_factors) = extract_i64_multiplier_and_base_factors(ctx, args[0]);
    if multiple == 5 {
        Some((kind, base_factors.into_vec()))
    } else {
        None
    }
}

fn build_trig_quintuple_angle_formula_substep(
    ctx: &Context,
    kind: TrigQuintupleAngleKind,
    base_factors: &[ExprId],
    reverse: bool,
) -> SubStep {
    let mut work = ctx.clone();
    let base = build_balanced_mul(&mut work, base_factors);
    let (base_plain, _) = render_temp_expr(&work, base);
    let base_plain = human_formula_title_plain(&base_plain);
    let (compact_plain, expanded_plain, compact_latex, expanded_latex) =
        trig_quintuple_angle_formula_template(kind);
    let title = format!(
        "Usar {} = {}, con u = {}",
        human_formula_title_plain(compact_plain),
        human_formula_title_plain(expanded_plain),
        base_plain
    );

    if reverse {
        formula_substep(
            title,
            expanded_plain,
            compact_plain,
            expanded_latex,
            compact_latex,
        )
    } else {
        formula_substep(
            title,
            compact_plain,
            expanded_plain,
            compact_latex,
            expanded_latex,
        )
    }
}

fn trig_quintuple_angle_formula_template(
    kind: TrigQuintupleAngleKind,
) -> (&'static str, &'static str, &'static str, &'static str) {
    match kind {
        TrigQuintupleAngleKind::Sin => (
            "sin(5u)",
            "5 · sin(u) - 20 · sin(u)^3 + 16 · sin(u)^5",
            "\\sin(5u)",
            "5\\cdot\\sin(u)-20\\cdot\\sin(u)^3+16\\cdot\\sin(u)^5",
        ),
        TrigQuintupleAngleKind::Cos => (
            "cos(5u)",
            "16 · cos(u)^5 - 20 · cos(u)^3 + 5 · cos(u)",
            "\\cos(5u)",
            "16\\cdot\\cos(u)^5-20\\cdot\\cos(u)^3+5\\cdot\\cos(u)",
        ),
    }
}

#[derive(Clone, Copy)]
enum HyperbolicTripleAngleKind {
    Sinh,
    Cosh,
    Tanh,
}

fn generate_hyperbolic_triple_angle_identity_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let Some((kind, call_expr, base_factors)) =
        find_nested_hyperbolic_triple_angle_call(ctx, before, false)
    else {
        return Vec::new();
    };

    let mut work = ctx.clone();
    let expansion = build_hyperbolic_triple_angle_expansion(&mut work, kind, &base_factors);
    let (before_plain, before_latex) = render_temp_expr(&work, call_expr);
    let (after_plain, after_latex) = render_temp_expr(&work, expansion);
    let title = format!(
        "Usar {} = {}",
        human_formula_title_plain(&before_plain),
        human_formula_title_plain(&after_plain)
    );

    vec![formula_substep(
        title,
        &before_plain,
        &after_plain,
        &before_latex,
        &after_latex,
    )]
}

fn human_formula_title_plain(plain: &str) -> String {
    plain.replace(" * ", "·").replace('*', "·")
}

fn find_nested_hyperbolic_triple_angle_call(
    ctx: &Context,
    expr: ExprId,
    is_nested: bool,
) -> Option<(HyperbolicTripleAngleKind, ExprId, Vec<ExprId>)> {
    if is_nested {
        if let Some(found) = hyperbolic_triple_angle_call_at_expr(ctx, expr) {
            return Some(found);
        }
    }

    match ctx.get(expr) {
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => find_nested_hyperbolic_triple_angle_call(ctx, *left, true)
            .or_else(|| find_nested_hyperbolic_triple_angle_call(ctx, *right, true)),
        Expr::Neg(inner) => find_nested_hyperbolic_triple_angle_call(ctx, *inner, true),
        _ => None,
    }
}

fn hyperbolic_triple_angle_call_at_expr(
    ctx: &Context,
    expr: ExprId,
) -> Option<(HyperbolicTripleAngleKind, ExprId, Vec<ExprId>)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let kind = if ctx.is_builtin(*fn_id, BuiltinFn::Sinh) {
        HyperbolicTripleAngleKind::Sinh
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cosh) {
        HyperbolicTripleAngleKind::Cosh
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Tanh) {
        HyperbolicTripleAngleKind::Tanh
    } else {
        return None;
    };

    let (multiple, base_factors) = extract_i64_multiplier_and_base_factors(ctx, args[0]);
    if multiple != 3 {
        return None;
    }
    Some((kind, expr, base_factors.into_vec()))
}

fn build_hyperbolic_triple_angle_expansion(
    ctx: &mut Context,
    kind: HyperbolicTripleAngleKind,
    base_factors: &[ExprId],
) -> ExprId {
    let u = build_balanced_mul(ctx, base_factors);
    let f_u = match kind {
        HyperbolicTripleAngleKind::Sinh => ctx.call_builtin(BuiltinFn::Sinh, vec![u]),
        HyperbolicTripleAngleKind::Cosh => ctx.call_builtin(BuiltinFn::Cosh, vec![u]),
        HyperbolicTripleAngleKind::Tanh => ctx.call_builtin(BuiltinFn::Tanh, vec![u]),
    };
    let square = {
        let two = ctx.num(2);
        ctx.add(Expr::Pow(f_u, two))
    };
    let cube = {
        let three = ctx.num(3);
        ctx.add(Expr::Pow(f_u, three))
    };

    match kind {
        HyperbolicTripleAngleKind::Sinh => {
            let three = ctx.num(3);
            let four = ctx.num(4);
            let linear = ctx.add(Expr::Mul(three, f_u));
            let cubic = ctx.add(Expr::Mul(four, cube));
            ctx.add(Expr::Add(linear, cubic))
        }
        HyperbolicTripleAngleKind::Cosh => {
            let four = ctx.num(4);
            let three = ctx.num(3);
            let cubic = ctx.add(Expr::Mul(four, cube));
            let linear = ctx.add(Expr::Mul(three, f_u));
            ctx.add(Expr::Sub(cubic, linear))
        }
        HyperbolicTripleAngleKind::Tanh => {
            let three_numerator = ctx.num(3);
            let three_denominator = ctx.num(3);
            let one = ctx.num(1);
            let linear = ctx.add(Expr::Mul(three_numerator, f_u));
            let numerator = ctx.add(Expr::Add(linear, cube));
            let quadratic = ctx.add(Expr::Mul(three_denominator, square));
            let denominator = ctx.add(Expr::Add(one, quadratic));
            ctx.add(Expr::Div(numerator, denominator))
        }
    }
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

fn generate_pythagorean_identity_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    if !is_one(ctx, after) {
        return Vec::new();
    }

    let Some((arg, rewrite_cos_square)) = pythagorean_square_pair(ctx, before) else {
        return Vec::new();
    };

    let mut work = ctx.clone();
    let one = work.num(1);
    let two = work.num(2);
    let sin_arg = work.call_builtin(BuiltinFn::Sin, vec![arg]);
    let cos_arg = work.call_builtin(BuiltinFn::Cos, vec![arg]);
    let sin_sq = work.add(Expr::Pow(sin_arg, two));
    let cos_sq = work.add(Expr::Pow(cos_arg, two));
    let arg_display = human_expr(ctx, arg);

    if rewrite_cos_square {
        let one_minus_sin_sq = work.add(Expr::Sub(one, sin_sq));
        let expanded = work.add(Expr::Add(sin_sq, one_minus_sin_sq));
        return vec![
            mixed_ctx_substep(
                format!("Reescribir cos({arg_display})^2 como 1 - sin({arg_display})^2"),
                ctx,
                before,
                &work,
                expanded,
            ),
            temp_ctx_substep(
                format!("Cancelar sin({arg_display})^2 - sin({arg_display})^2"),
                &work,
                expanded,
                one,
            ),
        ];
    }

    let one_minus_cos_sq = work.add(Expr::Sub(one, cos_sq));
    let expanded = work.add(Expr::Add(one_minus_cos_sq, cos_sq));
    vec![
        mixed_ctx_substep(
            format!("Reescribir sin({arg_display})^2 como 1 - cos({arg_display})^2"),
            ctx,
            before,
            &work,
            expanded,
        ),
        temp_ctx_substep(
            format!("Cancelar cos({arg_display})^2 - cos({arg_display})^2"),
            &work,
            expanded,
            one,
        ),
    ]
}

fn pythagorean_square_pair(ctx: &Context, expr: ExprId) -> Option<(ExprId, bool)> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return None;
    }

    let (first, first_sign) = terms[0];
    let (second, second_sign) = terms[1];
    if first_sign != Sign::Pos || second_sign != Sign::Pos {
        return None;
    }

    let (first_fn, first_arg) = trig_square_term(ctx, first)?;
    let (second_fn, second_arg) = trig_square_term(ctx, second)?;
    if !same_expr(ctx, first_arg, second_arg) {
        return None;
    }

    if matches!(first_fn, BuiltinFn::Sin) && matches!(second_fn, BuiltinFn::Cos) {
        return Some((first_arg, true));
    }
    if matches!(first_fn, BuiltinFn::Cos) && matches!(second_fn, BuiltinFn::Sin) {
        return Some((first_arg, false));
    }

    None
}

fn trig_square_term(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    let (base, exponent) = as_pow(ctx, expr)?;
    if !is_small_positive_integer(ctx, exponent, 2) {
        return None;
    }

    let Expr::Function(fn_id, args) = ctx.get(base) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let builtin = ctx.builtin_of(*fn_id)?;
    if matches!(builtin, BuiltinFn::Sin | BuiltinFn::Cos) {
        Some((builtin, args[0]))
    } else {
        None
    }
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

fn generate_trig_parity_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let mut work = ctx.clone();
    let is_odd = cas_math::trig_core_identity_support::try_rewrite_trig_odd_even_parity_expr(
        &mut work, before,
    )
    .map(|rewrite| rewrite.kind == cas_math::trig_core_identity_support::TrigOddEvenParityKind::Odd)
    .unwrap_or_else(|| matches!(ctx.get(after), Expr::Neg(_)));

    if is_odd {
        vec![formula_substep(
            "Usar que una función impar cumple f(-u) = -f(u)",
            "f(-u)",
            "-f(u)",
            "f(-u)",
            "-f(u)",
        )]
    } else {
        vec![formula_substep(
            "Usar que una función par cumple f(-u) = f(u)",
            "f(-u)",
            "f(u)",
            "f(-u)",
            "f(u)",
        )]
    }
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

fn generate_hyperbolic_quotient_substeps(_ctx: &Context, step: &Step) -> Vec<SubStep> {
    if step.description == "Recognize sinh(u) / cosh(u) as tanh(u)" {
        return Vec::new();
    }

    let (title, before_display, after_display, before_latex, after_latex) =
        if step.description.contains("Expand tanh") {
            (
                "Usar tanh(u) = sinh(u) / cosh(u)",
                "tanh(u)",
                "sinh(u) / cosh(u)",
                "\\tanh(u)",
                "\\frac{\\sinh(u)}{\\cosh(u)}",
            )
        } else {
            (
                "Usar sinh(u) / cosh(u) = tanh(u)",
                "sinh(u) / cosh(u)",
                "tanh(u)",
                "\\frac{\\sinh(u)}{\\cosh(u)}",
                "\\tanh(u)",
            )
        };

    vec![formula_substep(
        title,
        before_display,
        after_display,
        before_latex,
        after_latex,
    )]
}

fn generate_cos_diff_sin_diff_quotient_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let local_before = step.before_local().unwrap_or(step.before);
    let local_after = step.after_local().unwrap_or(step.after);

    if let Some(tan_arg) = tan_call_arg(ctx, local_after).or_else(|| tan_call_arg(ctx, step.after))
    {
        if let Some((_, _)) = as_div(ctx, local_before).or_else(|| as_div(ctx, step.before)) {
            let mut work = ctx.clone();
            let sin_arg = work.call_builtin(BuiltinFn::Sin, vec![tan_arg]);
            let cos_arg = work.call_builtin(BuiltinFn::Cos, vec![tan_arg]);
            let simplified_quotient = work.add_raw(Expr::Div(sin_arg, cos_arg));
            let tan_expr = work.call_builtin(BuiltinFn::Tan, vec![tan_arg]);
            return vec![
                mixed_ctx_substep(
                    "Cancelar el factor común del numerador y del denominador",
                    ctx,
                    local_before,
                    &work,
                    simplified_quotient,
                ),
                temp_ctx_substep(
                    "Reconocer el patrón sin(u) / cos(u) = tan(u)",
                    &work,
                    simplified_quotient,
                    tan_expr,
                ),
            ];
        }
    }

    let before_div = as_div(ctx, local_before).or_else(|| as_div(ctx, step.before));
    let after_div = as_div(ctx, local_after).or_else(|| as_div(ctx, step.after));
    let (Some((before_num, before_den)), Some((after_num, after_den))) = (before_div, after_div)
    else {
        return Vec::new();
    };

    if before_den == after_den && before_num != after_num {
        return vec![concrete_expr_substep(
            ctx,
            "Usar cos(A) - cos(B) = 2 · sin((A+B)/2) · sin((B-A)/2)",
            before_num,
            after_num,
        )];
    }

    if before_num == after_num && before_den != after_den {
        return vec![concrete_expr_substep(
            ctx,
            "Usar sin(B) - sin(A) = 2 · cos((A+B)/2) · sin((B-A)/2)",
            before_den,
            after_den,
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
) -> Option<(Vec<BigRational>, String, Vec<ExprId>)> {
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
    Some((coeffs, literal_display, literal_factors))
}

fn generate_hidden_radical_extraction_before_like_terms_substeps(
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

fn integer_value(ctx: &Context, expr: ExprId) -> Option<i64> {
    let Expr::Number(value) = ctx.get(expr) else {
        return None;
    };
    value.is_integer().then(|| value.numer().to_i64()).flatten()
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

fn collect_subexpr_ids(ctx: &Context, expr: ExprId, out: &mut Vec<ExprId>) {
    out.push(expr);
    match ctx.get(expr) {
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            collect_subexpr_ids(ctx, *left, out);
            collect_subexpr_ids(ctx, *right, out);
        }
        Expr::Neg(inner) | Expr::Hold(inner) => collect_subexpr_ids(ctx, *inner, out),
        Expr::Function(_, args) => {
            for arg in args {
                collect_subexpr_ids(ctx, *arg, out);
            }
        }
        Expr::Matrix { data, .. } => {
            for item in data {
                collect_subexpr_ids(ctx, *item, out);
            }
        }
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
    }
}

fn build_binomial_expansion_expr(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    kind: BinomialSquareKind,
    power: i64,
) -> Option<ExprId> {
    let add_signed = |ctx: &mut Context, terms: &[(ExprId, Sign)]| {
        let mut iter = terms.iter();
        let Some((first_term, first_sign)) = iter.next() else {
            return ctx.num(0);
        };
        let mut acc = if *first_sign == Sign::Pos {
            *first_term
        } else {
            ctx.add(Expr::Neg(*first_term))
        };
        for (term, sign) in iter {
            acc = if *sign == Sign::Pos {
                ctx.add(Expr::Add(acc, *term))
            } else {
                ctx.add(Expr::Sub(acc, *term))
            };
        }
        acc
    };
    let pow = |ctx: &mut Context, base: ExprId, exp: i64| {
        let exponent = ctx.num(exp);
        ctx.add(Expr::Pow(base, exponent))
    };
    let mul = |ctx: &mut Context, factors: &[ExprId]| build_balanced_mul(ctx, factors);

    let two = ctx.num(2);
    let three = ctx.num(3);
    let left_sq = pow(ctx, left, 2);
    let right_sq = pow(ctx, right, 2);

    Some(match (kind, power) {
        (BinomialSquareKind::Sum, 2) => {
            let cross = mul(ctx, &[two, left, right]);
            add_signed(
                ctx,
                &[
                    (left_sq, Sign::Pos),
                    (cross, Sign::Pos),
                    (right_sq, Sign::Pos),
                ],
            )
        }
        (BinomialSquareKind::Difference, 2) => {
            let cross = mul(ctx, &[two, left, right]);
            add_signed(
                ctx,
                &[
                    (left_sq, Sign::Pos),
                    (cross, Sign::Neg),
                    (right_sq, Sign::Pos),
                ],
            )
        }
        (BinomialSquareKind::Sum, 3) => {
            let left_cu = pow(ctx, left, 3);
            let right_cu = pow(ctx, right, 3);
            let left_sq_right = mul(ctx, &[three, left_sq, right]);
            let left_right_sq = mul(ctx, &[three, left, right_sq]);
            add_signed(
                ctx,
                &[
                    (left_cu, Sign::Pos),
                    (left_sq_right, Sign::Pos),
                    (left_right_sq, Sign::Pos),
                    (right_cu, Sign::Pos),
                ],
            )
        }
        (BinomialSquareKind::Difference, 3) => {
            let left_cu = pow(ctx, left, 3);
            let right_cu = pow(ctx, right, 3);
            let left_sq_right = mul(ctx, &[three, left_sq, right]);
            let left_right_sq = mul(ctx, &[three, left, right_sq]);
            add_signed(
                ctx,
                &[
                    (left_cu, Sign::Pos),
                    (left_sq_right, Sign::Neg),
                    (left_right_sq, Sign::Pos),
                    (right_cu, Sign::Neg),
                ],
            )
        }
        _ => return None,
    })
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

fn full_sixth_power_minus_one_factor_plan(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<ExprId> {
    let Expr::Sub(lhs, rhs) = ctx.get(before) else {
        return None;
    };
    if !is_integer_literal(ctx, *rhs, 1) {
        return None;
    }
    let Expr::Pow(base, exponent) = ctx.get(*lhs) else {
        return None;
    };
    if !is_integer_literal(ctx, *exponent, 6) {
        return None;
    }

    let factors = expr_nary::mul_leaves(ctx, after);
    if factors.len() != 4 {
        return None;
    }

    let mut saw_plus_one = false;
    let mut saw_minus_one = false;
    let mut saw_positive_quadratic = false;
    let mut saw_negative_quadratic = false;
    for factor in factors {
        if !saw_plus_one && linear_unit_factor_matches(ctx, factor, *base, Sign::Pos) {
            saw_plus_one = true;
            continue;
        }
        if !saw_minus_one && linear_unit_factor_matches(ctx, factor, *base, Sign::Neg) {
            saw_minus_one = true;
            continue;
        }
        if !saw_positive_quadratic
            && geometric_quadratic_factor_matches(ctx, factor, *base, Sign::Pos)
        {
            saw_positive_quadratic = true;
            continue;
        }
        if !saw_negative_quadratic
            && geometric_quadratic_factor_matches(ctx, factor, *base, Sign::Neg)
        {
            saw_negative_quadratic = true;
            continue;
        }
        return None;
    }

    (saw_plus_one && saw_minus_one && saw_positive_quadratic && saw_negative_quadratic)
        .then_some(*base)
}

fn linear_unit_factor_matches(
    ctx: &Context,
    expr: ExprId,
    base: ExprId,
    constant_sign: Sign,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return false;
    }

    let has_base = terms.iter().any(|(term, sign)| {
        *sign == Sign::Pos && compare_expr(ctx, *term, base) == Ordering::Equal
    });
    let has_unit = terms
        .iter()
        .any(|(term, sign)| *sign == constant_sign && is_integer_literal(ctx, *term, 1));

    has_base && has_unit
}

fn geometric_quadratic_factor_matches(
    ctx: &Context,
    expr: ExprId,
    base: ExprId,
    linear_sign: Sign,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 3 {
        return false;
    }

    let has_square = terms
        .iter()
        .any(|(term, sign)| *sign == Sign::Pos && matches_square_of(ctx, *term, base));
    let has_linear = terms.iter().any(|(term, sign)| {
        *sign == linear_sign && compare_expr(ctx, *term, base) == Ordering::Equal
    });
    let has_unit = terms
        .iter()
        .any(|(term, sign)| *sign == Sign::Pos && is_integer_literal(ctx, *term, 1));

    has_square && has_linear && has_unit
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
    if matches!(
        ctx.get(expr),
        Expr::Pow(pow_base, exp)
            if is_small_positive_integer(ctx, *exp, 2)
                && cas_ast::ordering::compare_expr(ctx, *pow_base, base)
                    == std::cmp::Ordering::Equal
    ) {
        return true;
    }

    matches_flattened_power_multiple(ctx, expr, base, 2)
}

fn matches_flattened_power_multiple(
    ctx: &Context,
    expr: ExprId,
    base: ExprId,
    multiplier: i64,
) -> bool {
    let Expr::Pow(expr_base, expr_exponent) = ctx.get(expr) else {
        return false;
    };
    let Expr::Pow(base_inner, base_exponent) = ctx.get(base) else {
        return false;
    };
    if compare_expr(ctx, *expr_base, *base_inner) != Ordering::Equal {
        return false;
    }
    let Some(expr_exponent) = positive_integer_literal_value(ctx, *expr_exponent) else {
        return false;
    };
    let Some(base_exponent) = positive_integer_literal_value(ctx, *base_exponent) else {
        return false;
    };
    expr_exponent == base_exponent * multiplier
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

fn tan_call_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.is_builtin(*fn_id, BuiltinFn::Tan) =>
        {
            Some(args[0])
        }
        _ => None,
    }
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

    let power_factor_substeps =
        generate_power_common_factor_cancel_substeps(ctx, before, after, *numerator, *denominator);
    if !power_factor_substeps.is_empty() {
        return power_factor_substeps;
    }

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

fn generate_power_common_factor_cancel_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
    numerator: ExprId,
    denominator: ExprId,
) -> Vec<SubStep> {
    let numerator_factors = cas_math::expr_nary::mul_factors(ctx, numerator);
    let denominator_factors = cas_math::expr_nary::mul_factors(ctx, denominator);

    for denominator_factor in denominator_factors.iter().copied() {
        for (index, numerator_factor) in numerator_factors.iter().copied().enumerate() {
            let Expr::Pow(base, exponent) = ctx.get(numerator_factor) else {
                continue;
            };
            if *base != denominator_factor {
                continue;
            }
            let Some(power) = small_positive_integer_value(ctx, *exponent) else {
                continue;
            };
            if power <= 1 {
                continue;
            }

            let mut work = ctx.clone();
            let mut expanded_numerator_factors = Vec::new();
            for (factor_index, factor) in numerator_factors.iter().copied().enumerate() {
                if factor_index == index {
                    expanded_numerator_factors.push(denominator_factor);
                    expanded_numerator_factors.push(power_factor_after_peeling_once(
                        &mut work,
                        denominator_factor,
                        power - 1,
                    ));
                } else {
                    expanded_numerator_factors.push(factor);
                }
            }

            let expanded_numerator =
                build_mul_expr_from_factors(&mut work, expanded_numerator_factors.as_slice());
            let original_denominator =
                build_mul_expr_from_factors(&mut work, denominator_factors.as_slice());
            let intermediate = if is_one_expr(&work, original_denominator) {
                expanded_numerator
            } else {
                work.add(Expr::Div(expanded_numerator, original_denominator))
            };
            let (intermediate_display, intermediate_latex) = render_temp_expr(&work, intermediate);
            let final_display = display_expr(ctx, after);
            let final_latex = latex_expr(ctx, after);
            let factor_display = display_expr(ctx, denominator_factor);

            return vec![
                SubStep::new(
                    format!(
                        "Descomponer {} para exponer el factor común {}",
                        human_expr(ctx, numerator_factor),
                        human_expr(ctx, denominator_factor)
                    ),
                    display_expr(ctx, before),
                    intermediate_display.clone(),
                )
                .with_before_latex(latex_expr(ctx, before))
                .with_after_latex(intermediate_latex.clone()),
                SubStep::new(
                    format!("Cancelar el factor común {factor_display}"),
                    intermediate_display,
                    final_display,
                )
                .with_before_latex(intermediate_latex)
                .with_after_latex(final_latex),
            ];
        }
    }

    Vec::new()
}

fn power_factor_after_peeling_once(
    ctx: &mut Context,
    base: ExprId,
    remaining_power: i64,
) -> ExprId {
    if remaining_power == 1 {
        base
    } else {
        let exponent = ctx.num(remaining_power);
        ctx.add(Expr::Pow(base, exponent))
    }
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
    if let Some(substeps) = generate_reverse_nested_fraction_substeps(ctx, before, after) {
        return substeps;
    }

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

    {
        let mut work = ctx.clone();
        if let Some((numerator_before, numerator_after, full_intermediate, divides_by_fraction)) =
            build_additive_numerator_nested_fraction_intermediates(&mut work, before)
        {
            let final_title = if divides_by_fraction {
                "Dividir entre una fracción es multiplicar por su inversa"
            } else {
                "Incorporar el denominador externo"
            };

            return vec![
                SubStep::new(
                    "Llevar el numerador a denominador común",
                    display_expr(&work, numerator_before),
                    display_expr(&work, numerator_after),
                )
                .with_before_latex(latex_expr(&work, numerator_before))
                .with_after_latex(latex_expr(&work, numerator_after)),
                SubStep::new(
                    final_title,
                    display_expr(&work, full_intermediate),
                    display_expr(ctx, after),
                )
                .with_before_latex(latex_expr(&work, full_intermediate))
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

fn split_single_fraction_addend(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    if let Some((fraction_num, fraction_den)) = as_div(ctx, left) {
        return Some((fraction_num, fraction_den, right));
    }

    let (fraction_num, fraction_den) = as_div(ctx, right)?;
    Some((fraction_num, fraction_den, left))
}

fn build_additive_single_fraction_common_denominator(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Add(left, right) = ctx.get(expr).clone() else {
        return None;
    };
    let (fraction_num, fraction_den, other) = split_single_fraction_addend(ctx, left, right)?;
    let scaled_other = ctx.add(Expr::Mul(other, fraction_den));
    let numerator = ctx.add(Expr::Add(scaled_other, fraction_num));
    Some(ctx.add(Expr::Div(numerator, fraction_den)))
}

fn build_additive_numerator_nested_fraction_intermediates(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId, bool)> {
    let (numerator, denominator) = as_div(ctx, expr)?;
    let numerator_after = build_reciprocal_pair_with_common_denominator(ctx, numerator)
        .or_else(|| build_additive_single_fraction_common_denominator(ctx, numerator))?;
    let full_intermediate = ctx.add(Expr::Div(numerator_after, denominator));
    Some((
        numerator,
        numerator_after,
        full_intermediate,
        unit_fraction_denominator(ctx, denominator).is_some(),
    ))
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
        return Vec::new();
    }

    if is_square_of_expr(ctx, *numerator, *denominator) {
        return Vec::new();
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
            build_square_over_denominator_cancel_substep(
                ctx,
                format!(
                    "Si {} está dividido entre {}, queda una sola copia",
                    denominator_squared_display, denominator_display
                ),
                *denominator,
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
            build_square_over_denominator_cancel_substep(
                ctx,
                format!(
                    "Si {} está dividido entre {}, queda una sola copia",
                    denominator_squared_display, denominator_display
                ),
                *denominator,
            ),
        ];
    }
    Vec::new()
}

fn build_square_over_denominator_cancel_substep(
    ctx: &Context,
    title: impl Into<String>,
    denominator: ExprId,
) -> SubStep {
    let mut work = ctx.clone();
    let two = work.num(2);
    let squared = work.add_raw(Expr::Pow(denominator, two));
    let fraction = work.add_raw(Expr::Div(squared, denominator));
    temp_ctx_substep(title, &work, fraction, denominator)
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

    if let Some((left_term, right_term)) = difference_of_squares_bases(ctx, *numerator) {
        if is_difference_of_terms(ctx, *denominator, left_term, right_term)
            && is_sum_of_terms(ctx, after, left_term, right_term)
        {
            let denominator_display = display_expr(ctx, *denominator);
            let remaining_display = display_expr(ctx, after);
            let denominator_latex = latex_expr(ctx, *denominator);
            let remaining_latex = latex_expr(ctx, after);
            let factorized_display = format!("({denominator_display}) · ({remaining_display})");
            let factorized_latex = format!(
                "\\left({denominator_latex}\\right)\\cdot \\left({remaining_latex}\\right)"
            );

            return vec![
                SubStep::new(
                    "Factorizar el numerador como diferencia de cuadrados",
                    display_expr(ctx, *numerator),
                    factorized_display.clone(),
                )
                .with_before_latex(latex_expr(ctx, *numerator))
                .with_after_latex(factorized_latex.clone()),
                SubStep::new(
                    format!("Ahora se cancela el factor {denominator_display}"),
                    format!("({factorized_display}) / ({denominator_display})"),
                    remaining_display,
                )
                .with_before_latex(format!(
                    "\\frac{{{factorized_latex}}}{{{denominator_latex}}}"
                ))
                .with_after_latex(remaining_latex),
            ];
        }
    }

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

fn generate_inverse_trig_sum_relation_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
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
        let pair_value_title = if step.rule_name == "Inverse Trig Sum Identity" {
            "Aquí arcsin(x) y arccos(x) suman pi/2"
        } else {
            "Esa pareja vale pi/2"
        };
        out.push(
            SubStep::new(
                pair_value_title,
                display_expr(ctx, pair_before),
                display_expr(ctx, pair_after),
            )
            .with_before_latex(latex_expr(ctx, pair_before))
            .with_after_latex(latex_expr(ctx, pair_after)),
        );
    }

    out
}

fn generate_inverse_hyperbolic_log_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(u) = change_of_base_natural_log_argument(ctx, after) else {
        return Vec::new();
    };

    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Atanh) {
        return Vec::new();
    }

    let ratio = args[0];
    if !matches!(ctx.get(ratio), Expr::Div(_, _)) {
        return Vec::new();
    }

    vec![SubStep::new(
        "Identificar el argumento como (u^2 - 1)/(u^2 + 1)",
        display_expr(ctx, ratio),
        format!("u = {}", display_expr(ctx, u)),
    )
    .with_before_latex(latex_expr(ctx, ratio))
    .with_after_latex(format!("u = {}", latex_expr(ctx, u)))]
}

fn generate_hyperbolic_composition_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some((outer_fn, inner_fn, argument)) = hyperbolic_composition_parts(ctx, before) else {
        return Vec::new();
    };
    if compare_expr(ctx, argument, after) != Ordering::Equal {
        return Vec::new();
    }

    let mut work = ctx.clone();
    let u = work.var("u");
    let inner = work.call_builtin(inner_fn, vec![u]);
    let composition = work.call_builtin(outer_fn, vec![inner]);
    let outer_name = hyperbolic_fn_name(outer_fn);
    let inner_name = hyperbolic_fn_name(inner_fn);

    vec![
        temp_ctx_substep(
            format!("Usar que {outer_name} y {inner_name} son funciones inversas"),
            &work,
            composition,
            u,
        ),
        SubStep::new(
            format!("Aquí u = {}", human_expr(ctx, argument)),
            human_expr(ctx, before),
            human_expr(ctx, after),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(latex_expr(ctx, after)),
    ]
}

fn hyperbolic_composition_parts(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, BuiltinFn, ExprId)> {
    let Expr::Function(outer_fn, outer_args) = ctx.get(expr) else {
        return None;
    };
    if outer_args.len() != 1 {
        return None;
    }
    let Expr::Function(inner_fn, inner_args) = ctx.get(outer_args[0]) else {
        return None;
    };
    if inner_args.len() != 1 {
        return None;
    }

    let outer = ctx.builtin_of(*outer_fn)?;
    let inner = ctx.builtin_of(*inner_fn)?;
    matches!(
        (outer, inner),
        (BuiltinFn::Sinh, BuiltinFn::Asinh)
            | (BuiltinFn::Cosh, BuiltinFn::Acosh)
            | (BuiltinFn::Tanh, BuiltinFn::Atanh)
            | (BuiltinFn::Asinh, BuiltinFn::Sinh)
            | (BuiltinFn::Acosh, BuiltinFn::Cosh)
            | (BuiltinFn::Atanh, BuiltinFn::Tanh)
    )
    .then_some((outer, inner, inner_args[0]))
}

fn hyperbolic_fn_name(function: BuiltinFn) -> &'static str {
    match function {
        BuiltinFn::Sinh => "sinh",
        BuiltinFn::Cosh => "cosh",
        BuiltinFn::Tanh => "tanh",
        BuiltinFn::Asinh => "asinh",
        BuiltinFn::Acosh => "acosh",
        BuiltinFn::Atanh => "atanh",
        _ => "función",
    }
}

fn generate_inverse_trig_composition_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if let Some(substeps) = generate_direct_inverse_trig_composition_substeps(ctx, step) {
        return substeps;
    }

    if step.description.contains("sin(arctan") || step.description.contains("cos(arctan") {
        return generate_arctan_right_triangle_composition_substeps(ctx, step);
    }

    if step.description.contains("cos(arcsin")
        || step.description.contains("sin(arccos")
        || step.description.contains("tan(arcsin")
    {
        return generate_arcsin_arccos_complement_composition_substeps(ctx, step);
    }

    if !step.description.contains("arcsin") || !step.description.contains("arctan") {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some(arcsin_arg) =
        inverse_trig_unary_arg(ctx, before, &[BuiltinFn::Arcsin, BuiltinFn::Asin])
    else {
        return Vec::new();
    };
    let Some(x) = inverse_trig_unary_arg(ctx, after, &[BuiltinFn::Arctan, BuiltinFn::Atan]) else {
        return Vec::new();
    };

    let mut work = ctx.clone();
    let arctan_x = work.call_builtin(BuiltinFn::Arctan, vec![x]);
    let sin_arctan_x = work.call_builtin(BuiltinFn::Sin, vec![arctan_x]);
    let arcsin_sin_arctan_x = work.call_builtin(BuiltinFn::Arcsin, vec![sin_arctan_x]);

    vec![
        temp_ctx_substep(
            "Reconocer x/sqrt(1+x^2) como sin(arctan(x))",
            &work,
            arcsin_arg,
            sin_arctan_x,
        ),
        temp_ctx_substep(
            "Sustituir dentro de arcsin",
            &work,
            before,
            arcsin_sin_arctan_x,
        ),
        temp_ctx_substep(
            "Usar asin(sin(u)) = u en el rango principal",
            &work,
            arcsin_sin_arctan_x,
            arctan_x,
        ),
    ]
}

fn generate_direct_inverse_trig_composition_substeps(
    ctx: &Context,
    step: &Step,
) -> Option<Vec<SubStep>> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let (outer_fn, inner_fn, argument) = direct_inverse_trig_composition_parts(ctx, before)?;
    if compare_expr(ctx, argument, after) != Ordering::Equal {
        return None;
    }

    let mut work = ctx.clone();
    let u = work.var("u");
    let inner = work.call_builtin(inner_fn, vec![u]);
    let composition = work.call_builtin(outer_fn, vec![inner]);
    let outer_name = inverse_trig_fn_name(outer_fn);
    let inner_name = inverse_trig_fn_name(inner_fn);

    Some(vec![
        temp_ctx_substep(
            format!("Usar que {outer_name} y {inner_name} son funciones inversas"),
            &work,
            composition,
            u,
        ),
        SubStep::new(
            format!("Aquí u = {}", human_expr(ctx, argument)),
            human_expr(ctx, before),
            human_expr(ctx, after),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(latex_expr(ctx, after)),
    ])
}

fn direct_inverse_trig_composition_parts(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, BuiltinFn, ExprId)> {
    let Expr::Function(outer_fn, outer_args) = ctx.get(expr) else {
        return None;
    };
    if outer_args.len() != 1 {
        return None;
    }
    let Expr::Function(inner_fn, inner_args) = ctx.get(outer_args[0]) else {
        return None;
    };
    if inner_args.len() != 1 {
        return None;
    }

    let outer = ctx.builtin_of(*outer_fn)?;
    let inner = ctx.builtin_of(*inner_fn)?;
    matches!(
        (outer, inner),
        (BuiltinFn::Sin, BuiltinFn::Arcsin)
            | (BuiltinFn::Sin, BuiltinFn::Asin)
            | (BuiltinFn::Cos, BuiltinFn::Arccos)
            | (BuiltinFn::Cos, BuiltinFn::Acos)
            | (BuiltinFn::Tan, BuiltinFn::Arctan)
            | (BuiltinFn::Tan, BuiltinFn::Atan)
    )
    .then_some((outer, inner, inner_args[0]))
}

fn inverse_trig_fn_name(function: BuiltinFn) -> &'static str {
    match function {
        BuiltinFn::Sin => "sin",
        BuiltinFn::Cos => "cos",
        BuiltinFn::Tan => "tan",
        BuiltinFn::Arcsin | BuiltinFn::Asin => "arcsin",
        BuiltinFn::Arccos | BuiltinFn::Acos => "arccos",
        BuiltinFn::Arctan | BuiltinFn::Atan => "arctan",
        _ => "función",
    }
}

fn generate_arcsin_arccos_complement_composition_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(outer_fn, outer_args) = ctx.get(before) else {
        return Vec::new();
    };
    if outer_args.len() != 1 {
        return Vec::new();
    }

    let (x, inverse_name, known_side, projection_title, projection_label) =
        if ctx.is_builtin(*outer_fn, BuiltinFn::Cos) {
            let Some(x) =
                inverse_trig_unary_arg(ctx, outer_args[0], &[BuiltinFn::Arcsin, BuiltinFn::Asin])
            else {
                return Vec::new();
            };
            (
                x,
                "arcsin(x)",
                "opuesto",
                "Leer el coseno desde ese triángulo",
                "coseno",
            )
        } else if ctx.is_builtin(*outer_fn, BuiltinFn::Sin) {
            let Some(x) =
                inverse_trig_unary_arg(ctx, outer_args[0], &[BuiltinFn::Arccos, BuiltinFn::Acos])
            else {
                return Vec::new();
            };
            (
                x,
                "arccos(x)",
                "adyacente",
                "Leer el seno desde ese triángulo",
                "seno",
            )
        } else if ctx.is_builtin(*outer_fn, BuiltinFn::Tan) {
            let Some(x) =
                inverse_trig_unary_arg(ctx, outer_args[0], &[BuiltinFn::Arcsin, BuiltinFn::Asin])
            else {
                return Vec::new();
            };
            (
                x,
                "arcsin(x)",
                "opuesto",
                "Leer la tangente desde ese triángulo",
                "tangente",
            )
        } else {
            return Vec::new();
        };

    let unknown_side = if known_side == "opuesto" {
        "adyacente"
    } else {
        "opuesto"
    };
    let mut work = ctx.clone();
    let two = work.num(2);
    let x_squared = work.add(Expr::Pow(x, two));
    let one = work.num(1);
    let radicand = work.add(Expr::Sub(one, x_squared));
    let missing_side = work.call_builtin(BuiltinFn::Sqrt, vec![radicand]);

    let side_text = format!(
        "{} = {}, hipotenusa = {}, {} = {}",
        known_side,
        human_expr(&work, x),
        human_expr(&work, one),
        unknown_side,
        human_expr(&work, missing_side)
    );
    let side_latex = format!(
        "\\text{{{}}}={},\\ \\text{{hipotenusa}}={},\\ \\text{{{}}}={}",
        known_side,
        latex_expr(&work, x),
        latex_expr(&work, one),
        unknown_side,
        latex_expr(&work, missing_side)
    );
    let projection_substep = SubStep::new(
        projection_title,
        side_text,
        format!("{} = {}", projection_label, human_expr(&work, after)),
    )
    .with_before_latex(side_latex)
    .with_after_latex(latex_expr(&work, after));

    vec![
        temp_ctx_substep(
            format!("Calcular el cateto restante del triángulo asociado a {inverse_name}"),
            &work,
            radicand,
            missing_side,
        ),
        projection_substep,
    ]
}

fn generate_arctan_right_triangle_composition_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(outer_fn, outer_args) = ctx.get(before) else {
        return Vec::new();
    };
    if outer_args.len() != 1 {
        return Vec::new();
    }
    let is_sine_projection = if ctx.is_builtin(*outer_fn, BuiltinFn::Sin) {
        true
    } else if ctx.is_builtin(*outer_fn, BuiltinFn::Cos) {
        false
    } else {
        return Vec::new();
    };
    let projection_title = if is_sine_projection {
        "Leer el seno desde ese triángulo"
    } else {
        "Leer el coseno desde ese triángulo"
    };
    let projection_label = if is_sine_projection { "seno" } else { "coseno" };
    let Some(x) = inverse_trig_unary_arg(ctx, outer_args[0], &[BuiltinFn::Arctan, BuiltinFn::Atan])
    else {
        return Vec::new();
    };

    let mut work = ctx.clone();
    let two = work.num(2);
    let x_squared = work.add(Expr::Pow(x, two));
    let one = work.num(1);
    let radicand = work.add(Expr::Add(x_squared, one));
    let hypotenuse = work.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let side_text = if is_sine_projection {
        format!(
            "opuesto = {}, hipotenusa = {}",
            human_expr(&work, x),
            human_expr(&work, hypotenuse)
        )
    } else {
        format!(
            "adyacente = {}, hipotenusa = {}",
            human_expr(&work, one),
            human_expr(&work, hypotenuse)
        )
    };
    let side_latex = if is_sine_projection {
        format!(
            "\\text{{opuesto}}={},\\ \\text{{hipotenusa}}={}",
            latex_expr(&work, x),
            latex_expr(&work, hypotenuse)
        )
    } else {
        format!(
            "\\text{{adyacente}}={},\\ \\text{{hipotenusa}}={}",
            latex_expr(&work, one),
            latex_expr(&work, hypotenuse)
        )
    };
    let projection_substep = SubStep::new(
        projection_title,
        side_text,
        format!("{} = {}", projection_label, human_expr(&work, after)),
    )
    .with_before_latex(side_latex)
    .with_after_latex(latex_expr(&work, after));

    vec![
        temp_ctx_substep(
            "Calcular la hipotenusa del triángulo asociado a arctan(x)",
            &work,
            radicand,
            hypotenuse,
        ),
        projection_substep,
    ]
}

fn generate_change_of_base_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if step.description == "Expand the logarithm using a change-of-base chain" {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);

    if let Some((base, argument)) = change_of_base_log_arguments(ctx, before) {
        let mut work = ctx.clone();
        let ln_argument = work.call_builtin(BuiltinFn::Ln, vec![argument]);
        let ln_base = work.call_builtin(BuiltinFn::Ln, vec![base]);
        return vec![
            temp_ctx_substep(
                "Poner el argumento en el numerador",
                &work,
                argument,
                ln_argument,
            ),
            temp_ctx_substep("Poner la base en el denominador", &work, base, ln_base),
        ];
    }

    if let Some((argument, base, numerator, denominator)) =
        change_of_base_quotient_arguments(ctx, before)
    {
        return vec![
            temp_ctx_substep(
                "Leer el argumento desde el numerador",
                ctx,
                numerator,
                argument,
            ),
            temp_ctx_substep("Leer la base desde el denominador", ctx, denominator, base),
        ];
    }

    Vec::new()
}

fn change_of_base_log_arguments(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let Expr::Function(name, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() == 2 && ctx.is_builtin(*name, BuiltinFn::Log) {
        Some((args[0], args[1]))
    } else {
        None
    }
}

fn change_of_base_natural_log_argument(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(name, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() == 1
        && (ctx.is_builtin(*name, BuiltinFn::Ln) || ctx.is_builtin(*name, BuiltinFn::Log))
    {
        Some(args[0])
    } else {
        None
    }
}

fn change_of_base_quotient_arguments(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId, ExprId)> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let numerator = *numerator;
    let denominator = *denominator;
    let argument = change_of_base_natural_log_argument(ctx, numerator)?;
    let base = change_of_base_natural_log_argument(ctx, denominator)?;
    Some((argument, base, numerator, denominator))
}

fn inverse_trig_unary_arg(
    ctx: &Context,
    expr: ExprId,
    accepted_builtins: &[BuiltinFn],
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(*fn_id)?;
    accepted_builtins.contains(&builtin).then_some(args[0])
}

fn generate_sqrt_perfect_square_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
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

fn is_direct_square_of(ctx: &Context, expr: ExprId, base: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(pow_base, exponent) => {
            is_integer_literal(ctx, *exponent, 2)
                && compare_expr(ctx, *pow_base, base) == Ordering::Equal
        }
        _ => false,
    }
}

fn generate_square_of_square_root_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
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

fn direct_replacement_pair(
    step: &Step,
    local_before: ExprId,
    local_after: ExprId,
) -> Option<(ExprId, ExprId)> {
    if let (Some(global_before), Some(global_after)) = (step.global_before, step.global_after) {
        if global_before != local_before || global_after != local_after {
            return Some((global_before, global_after));
        }
    }

    if step.before != local_before || step.after != local_after {
        return Some((step.before, step.after));
    }

    (local_before != local_after).then_some((local_before, local_after))
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

fn generate_sophie_germain_expansion_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some((a, b)) = sophie_germain_expansion_plan(ctx, before, after) else {
        return Vec::new();
    };

    let factorized_display = sophie_germain_factorized_identity_display(ctx, a, b);
    let factorized_latex = sophie_germain_factorized_identity_latex(ctx, a, b);
    let identity_display = sophie_germain_identity_display(ctx, a, b);
    let identity_latex = sophie_germain_identity_latex(ctx, a, b);

    vec![
        SubStep::new(
            "Reconocer el patrón de Sophie Germain",
            display_expr(ctx, before),
            factorized_display.clone(),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(factorized_latex.clone()),
        SubStep::new(
            "Aplicar la identidad de Sophie Germain",
            factorized_display,
            identity_display,
        )
        .with_before_latex(factorized_latex)
        .with_after_latex(identity_latex),
    ]
}

fn generate_sophie_germain_factorization_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some((a, b)) = sophie_germain_expansion_plan(ctx, after, before) else {
        return Vec::new();
    };

    let difference_display = sophie_germain_difference_of_squares_display(ctx, a, b);
    let difference_latex = sophie_germain_difference_of_squares_latex(ctx, a, b);
    let factorized_display = sophie_germain_factorized_identity_display(ctx, a, b);
    let factorized_latex = sophie_germain_factorized_identity_latex(ctx, a, b);

    vec![
        SubStep::new(
            "Convertir la suma en diferencia de cuadrados",
            display_expr(ctx, before),
            difference_display.clone(),
        )
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(difference_latex.clone()),
        SubStep::new(
            "Factorizar la diferencia de cuadrados",
            difference_display,
            factorized_display,
        )
        .with_before_latex(difference_latex)
        .with_after_latex(factorized_latex),
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

        let mut out = vec![
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

        if let Some((replacement_before, replacement_after)) =
            direct_replacement_pair(step, before, after)
        {
            out.push(
                SubStep::new(
                    "Reemplazar ese bloque en la expresión",
                    display_expr(ctx, replacement_before),
                    display_expr(ctx, replacement_after),
                )
                .with_before_latex(latex_expr(ctx, replacement_before))
                .with_after_latex(latex_expr(ctx, replacement_after)),
            );
        }

        return out;
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

fn generate_integration_by_parts_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
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

    let Expr::Variable(var_sym) = ctx.get(args[1]) else {
        return Vec::new();
    };
    let var_name = ctx.sym_name(*var_sym);
    let mut scratch = ctx.clone();
    let is_quadratic_by_parts =
        cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_exp_linear_target(
            &mut scratch,
            args[0],
            var_name,
        ) || cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_trig_linear_target(
            &mut scratch,
            args[0],
            var_name,
        );
    let is_linear_by_parts = contains_linear_integration_by_parts_target(ctx, args[0], var_name);
    if !is_quadratic_by_parts && !is_linear_by_parts {
        return Vec::new();
    }
    let title = if is_quadratic_by_parts {
        "Usar integración por partes repetida"
    } else {
        "Usar integración por partes"
    };

    vec![
        SubStep::new(title, display_expr(ctx, args[0]), display_expr(ctx, after))
            .with_before_latex(latex_expr(ctx, args[0]))
            .with_after_latex(latex_expr(ctx, after)),
    ]
}

fn contains_linear_integration_by_parts_target(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let mut scratch = ctx.clone();
    if cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_exp_linear_target(
        &mut scratch,
        expr,
        var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_trig_linear_target(
        &mut scratch,
        expr,
        var_name,
    ) || cas_math::symbolic_integration_support::integrate_symbolic_is_linear_times_hyperbolic_linear_target(
        &mut scratch,
        expr,
        var_name,
    ) {
        return true;
    }

    match ctx.get(expr) {
        Expr::Add(left, right) | Expr::Sub(left, right) => {
            contains_linear_integration_by_parts_target(ctx, *left, var_name)
                || contains_linear_integration_by_parts_target(ctx, *right, var_name)
        }
        Expr::Neg(inner) => contains_linear_integration_by_parts_target(ctx, *inner, var_name),
        _ => false,
    }
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

fn same_expr(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    left == right
        || compare_expr(ctx, left, right) == Ordering::Equal
        || same_presentational_expr(ctx, left, ctx, right)
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
            _ if is_negative_one(ctx, *right) => {
                Some((*left, *right, CubeIdentityKind::Difference))
            }
            _ => match ctx.get(*left) {
                Expr::Neg(inner) => Some((*right, *inner, CubeIdentityKind::Difference)),
                _ if is_negative_one(ctx, *left) => {
                    Some((*right, *left, CubeIdentityKind::Difference))
                }
                _ => Some((*left, *right, CubeIdentityKind::Sum)),
            },
        },
        _ => None,
    }
}

fn cube_identity_plan(ctx: &Context, before: ExprId, after: ExprId) -> Option<CubeIdentityPlan> {
    let (left_term, right_term, kind) = cube_identity_terms(ctx, before)?;

    let left_base = cube_base_from_term_with_witness(ctx, left_term, after)?;
    let right_base = cube_base_from_term_with_witness(ctx, right_term, after)?;

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
    let left_base = cube_base_from_term_with_witness(ctx, left_term, before)?;
    let right_base = cube_base_from_term_with_witness(ctx, right_term, before)?;
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

fn cube_base_from_term_with_witness(
    ctx: &Context,
    expr: ExprId,
    witness: ExprId,
) -> Option<ExprId> {
    cube_base_from_term(ctx, expr).or_else(|| {
        if is_negative_one(ctx, expr) {
            find_one_literal(ctx, witness)
        } else if let Some((base, cube_root_exponent)) = cube_root_power_term(ctx, expr) {
            find_power_literal(ctx, witness, base, cube_root_exponent)
        } else {
            None
        }
    })
}

fn cube_root_power_term(ctx: &Context, expr: ExprId) -> Option<(ExprId, i64)> {
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

fn find_power_literal(ctx: &Context, expr: ExprId, base: ExprId, exponent: i64) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(candidate_base, candidate_exponent)
            if compare_expr(ctx, *candidate_base, base) == Ordering::Equal
                && is_integer_literal(ctx, *candidate_exponent, exponent) =>
        {
            Some(expr)
        }
        Expr::Neg(inner) => find_power_literal(ctx, *inner, base, exponent),
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => find_power_literal(ctx, *left, base, exponent)
            .or_else(|| find_power_literal(ctx, *right, base, exponent)),
        Expr::Function(_, args) => args
            .iter()
            .find_map(|arg| find_power_literal(ctx, *arg, base, exponent)),
        _ => None,
    }
}

fn find_one_literal(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if is_one(ctx, expr) {
        return Some(expr);
    }

    match ctx.get(expr) {
        Expr::Neg(inner) => find_one_literal(ctx, *inner),
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right)
        | Expr::Pow(left, right) => {
            find_one_literal(ctx, *left).or_else(|| find_one_literal(ctx, *right))
        }
        Expr::Function(_, args) => args.iter().find_map(|arg| find_one_literal(ctx, *arg)),
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
                (*left == left_base && is_negated_version_of(ctx, *right, right_base))
                    || (*right == left_base && is_negated_version_of(ctx, *left, right_base))
            }
            _ => false,
        },
    }
}

fn is_negated_version_of(ctx: &Context, expr: ExprId, positive: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Neg(inner) if *inner == positive)
        || (is_one(ctx, positive) && is_negative_one(ctx, expr))
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
    let left_sq = squared_display(ctx, left_base);
    let right_sq = squared_display(ctx, right_base);
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

fn sophie_germain_identity_display(ctx: &Context, left_base: ExprId, right_base: ExprId) -> String {
    format!(
        "{} + 4 · {}",
        fourth_power_display(ctx, left_base),
        fourth_power_display(ctx, right_base)
    )
}

fn sophie_germain_identity_latex(ctx: &Context, left_base: ExprId, right_base: ExprId) -> String {
    format!(
        "{} + 4\\cdot {}",
        fourth_power_latex(ctx, left_base),
        fourth_power_latex(ctx, right_base)
    )
}

fn sophie_germain_difference_of_squares_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
) -> String {
    format!(
        "({} + 2 · {})^2 - (2 · {} · {})^2",
        squared_display(ctx, left_base),
        squared_display(ctx, right_base),
        display_expr(ctx, left_base),
        display_expr(ctx, right_base)
    )
}

fn sophie_germain_difference_of_squares_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
) -> String {
    format!(
        "\\left({} + 2\\cdot {}\\right)^{{2}} - \\left(2\\cdot {}\\cdot {}\\right)^{{2}}",
        squared_latex(ctx, left_base),
        squared_latex(ctx, right_base),
        latex_expr(ctx, left_base),
        latex_expr(ctx, right_base)
    )
}

fn sophie_germain_minus_factor_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
) -> String {
    format!(
        "({} - 2 · {} · {} + 2 · {})",
        squared_display(ctx, left_base),
        display_expr(ctx, left_base),
        display_expr(ctx, right_base),
        squared_display(ctx, right_base)
    )
}

fn sophie_germain_minus_factor_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
) -> String {
    format!(
        "\\left({} - 2\\cdot {}\\cdot {} + 2\\cdot {}\\right)",
        squared_latex(ctx, left_base),
        latex_expr(ctx, left_base),
        latex_expr(ctx, right_base),
        squared_latex(ctx, right_base)
    )
}

fn sophie_germain_plus_factor_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
) -> String {
    format!(
        "({} + 2 · {} · {} + 2 · {})",
        squared_display(ctx, left_base),
        display_expr(ctx, left_base),
        display_expr(ctx, right_base),
        squared_display(ctx, right_base)
    )
}

fn sophie_germain_plus_factor_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
) -> String {
    format!(
        "\\left({} + 2\\cdot {}\\cdot {} + 2\\cdot {}\\right)",
        squared_latex(ctx, left_base),
        latex_expr(ctx, left_base),
        latex_expr(ctx, right_base),
        squared_latex(ctx, right_base)
    )
}

fn sophie_germain_factorized_identity_display(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
) -> String {
    format!(
        "{} · {}",
        sophie_germain_minus_factor_display(ctx, left_base, right_base),
        sophie_germain_plus_factor_display(ctx, left_base, right_base)
    )
}

fn sophie_germain_factorized_identity_latex(
    ctx: &Context,
    left_base: ExprId,
    right_base: ExprId,
) -> String {
    format!(
        "{}\\cdot {}",
        sophie_germain_minus_factor_latex(ctx, left_base, right_base),
        sophie_germain_plus_factor_latex(ctx, left_base, right_base)
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

fn is_sum_of_terms(ctx: &Context, expr: ExprId, left: ExprId, right: ExprId) -> bool {
    let Expr::Add(sum_left, sum_right) = ctx.get(expr) else {
        return false;
    };
    (*sum_left == left && *sum_right == right) || (*sum_left == right && *sum_right == left)
}

fn is_difference_of_terms(ctx: &Context, expr: ExprId, left: ExprId, right: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Sub(diff_left, diff_right) if *diff_left == left && *diff_right == right)
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
