//! `focused_rule_substeps`: familia `general`.
//!
//! Ver la cabecera de `focused_rule_substeps.rs` para el contexto.

use super::*;

pub(crate) fn generate_focused_rule_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    generate_focused_rule_substeps_at_depth(ctx, step, 0)
}

pub(super) fn generate_focused_rule_substeps_at_depth(
    ctx: &Context,
    step: &Step,
    depth: usize,
) -> Vec<SubStep> {
    let differentiation_substeps = generate_symbolic_differentiation_substeps(ctx, step);
    if !differentiation_substeps.is_empty() {
        return differentiation_substeps;
    }

    let gradient_substeps = generate_vector_gradient_substeps(ctx, step);
    if !gradient_substeps.is_empty() {
        return gradient_substeps;
    }

    let jacobian_hessian_substeps = generate_vector_jacobian_hessian_substeps(ctx, step);
    if !jacobian_hessian_substeps.is_empty() {
        return jacobian_hessian_substeps;
    }

    let div_lap_substeps = generate_divergence_laplacian_substeps(ctx, step);
    if !div_lap_substeps.is_empty() {
        return div_lap_substeps;
    }

    let taylor_multivar_substeps = generate_taylor_multivar_substeps(ctx, step);
    if !taylor_multivar_substeps.is_empty() {
        return taylor_multivar_substeps;
    }

    let lineintegral_substeps = generate_lineintegral_substeps(ctx, step);
    if !lineintegral_substeps.is_empty() {
        return lineintegral_substeps;
    }

    let surface_integral_substeps = generate_surface_integral_substeps(ctx, step);
    if !surface_integral_substeps.is_empty() {
        return surface_integral_substeps;
    }

    let potential_substeps = generate_potential_substeps(ctx, step);
    if !potential_substeps.is_empty() {
        return potential_substeps;
    }

    let path_counterexample_substeps = generate_limit_path_counterexample_substeps(ctx, step);
    if !path_counterexample_substeps.is_empty() {
        return path_counterexample_substeps;
    }

    let integral_residual_policy_substeps = generate_integral_residual_policy_substeps(ctx, step);
    if !integral_residual_policy_substeps.is_empty() {
        return integral_residual_policy_substeps;
    }

    let basic_polynomial_integration_substeps =
        generate_basic_polynomial_integration_substeps(ctx, step);
    if !basic_polynomial_integration_substeps.is_empty() {
        return basic_polynomial_integration_substeps;
    }

    // Before by-parts: an integrand that is a SUM is linearity, and letting
    // by-parts see it first is what produced the measured false labels
    // (`∫(ln x + x)` and `∫(x·e^x + sin 2x)` were titled "use integration by
    // parts" over the whole sum).
    let root_sum_substeps = generate_root_sum_integration_substeps(ctx, step);
    if !root_sum_substeps.is_empty() {
        return root_sum_substeps;
    }

    let vector_component_substeps = generate_vector_component_calculus_substeps(ctx, step, depth);
    if !vector_component_substeps.is_empty() {
        return vector_component_substeps;
    }

    let additive_integration_substeps = generate_additive_integration_substeps(ctx, step, depth);
    if !additive_integration_substeps.is_empty() {
        return additive_integration_substeps;
    }

    let integration_by_parts_substeps = generate_integration_by_parts_substeps(ctx, step);
    if !integration_by_parts_substeps.is_empty() {
        return integration_by_parts_substeps;
    }

    let integration_linear_inverse_table_substeps =
        generate_linear_inverse_table_integration_substeps(ctx, step);
    if !integration_linear_inverse_table_substeps.is_empty() {
        return integration_linear_inverse_table_substeps;
    }

    let integration_linear_elementary_table_substeps =
        generate_linear_elementary_table_integration_substeps(ctx, step);
    if !integration_linear_elementary_table_substeps.is_empty() {
        return integration_linear_elementary_table_substeps;
    }

    let integration_linear_log_table_substeps =
        generate_linear_log_table_integration_substeps(ctx, step);
    if !integration_linear_log_table_substeps.is_empty() {
        return integration_linear_log_table_substeps;
    }

    let integration_rational_partial_fraction_substeps =
        generate_rational_linear_partial_fraction_integration_substeps(ctx, step);
    if !integration_rational_partial_fraction_substeps.is_empty() {
        return integration_rational_partial_fraction_substeps;
    }

    let integration_positive_quadratic_square_substeps =
        generate_positive_quadratic_square_integration_substeps(ctx, step);
    if !integration_positive_quadratic_square_substeps.is_empty() {
        return integration_positive_quadratic_square_substeps;
    }

    let integration_positive_quadratic_cube_substeps =
        generate_positive_quadratic_cube_integration_substeps(ctx, step);
    if !integration_positive_quadratic_cube_substeps.is_empty() {
        return integration_positive_quadratic_cube_substeps;
    }

    let integration_mixed_numerator_substeps =
        generate_positive_quadratic_mixed_numerator_integration_substeps(ctx, step);
    if !integration_mixed_numerator_substeps.is_empty() {
        return integration_mixed_numerator_substeps;
    }

    let integration_multi_quadratic_substeps =
        generate_multi_quadratic_partial_fraction_integration_substeps(ctx, step);
    if !integration_multi_quadratic_substeps.is_empty() {
        return integration_multi_quadratic_substeps;
    }

    let integration_general_rational_substeps =
        generate_general_rational_integration_substeps(ctx, step);
    if !integration_general_rational_substeps.is_empty() {
        return integration_general_rational_substeps;
    }

    let exponential_div_substeps =
        generate_normalized_exponential_div_integration_substeps(ctx, step);
    if !exponential_div_substeps.is_empty() {
        return exponential_div_substeps;
    }

    let definite_integral_substeps = generate_definite_integral_substeps(ctx, step, depth);
    if !definite_integral_substeps.is_empty() {
        return definite_integral_substeps;
    }

    let integration_trig_log_table_substeps =
        generate_trig_log_table_integration_substeps(ctx, step);
    if !integration_trig_log_table_substeps.is_empty() {
        return integration_trig_log_table_substeps;
    }

    let integration_hyperbolic_log_table_substeps =
        generate_hyperbolic_log_table_integration_substeps(ctx, step);
    if !integration_hyperbolic_log_table_substeps.is_empty() {
        return integration_hyperbolic_log_table_substeps;
    }

    let integration_hyperbolic_reciprocal_table_substeps =
        generate_hyperbolic_reciprocal_table_integration_substeps(ctx, step);
    if !integration_hyperbolic_reciprocal_table_substeps.is_empty() {
        return integration_hyperbolic_reciprocal_table_substeps;
    }

    let integration_polynomial_derivative_table_substeps =
        generate_polynomial_derivative_table_integration_substeps(ctx, step);
    if !integration_polynomial_derivative_table_substeps.is_empty() {
        return integration_polynomial_derivative_table_substeps;
    }

    let integration_log_power_product_table_substeps =
        generate_log_power_product_table_integration_substeps(ctx, step);
    if !integration_log_power_product_table_substeps.is_empty() {
        return integration_log_power_product_table_substeps;
    }

    let integration_polynomial_base_table_substeps =
        generate_polynomial_base_table_integration_substeps(ctx, step);
    if !integration_polynomial_base_table_substeps.is_empty() {
        return integration_polynomial_base_table_substeps;
    }

    let integration_nested_inverse_polynomial_table_substeps =
        generate_nested_inverse_polynomial_table_integration_substeps(ctx, step);
    if !integration_nested_inverse_polynomial_table_substeps.is_empty() {
        return integration_nested_inverse_polynomial_table_substeps;
    }

    let integration_arctan_sqrt_reciprocal_table_substeps =
        generate_arctan_sqrt_reciprocal_table_integration_substeps(ctx, step);
    if !integration_arctan_sqrt_reciprocal_table_substeps.is_empty() {
        return integration_arctan_sqrt_reciprocal_table_substeps;
    }

    let integration_trig_quotient_table_substeps =
        generate_trig_quotient_table_integration_substeps(ctx, step);
    if !integration_trig_quotient_table_substeps.is_empty() {
        return integration_trig_quotient_table_substeps;
    }

    let integration_reciprocal_trig_derivative_product_substeps =
        generate_reciprocal_trig_derivative_product_integration_substeps(ctx, step);
    if !integration_reciprocal_trig_derivative_product_substeps.is_empty() {
        return integration_reciprocal_trig_derivative_product_substeps;
    }

    let integration_substitution_substeps = generate_integration_substitution_substeps(ctx, step);
    if !integration_substitution_substeps.is_empty() {
        return integration_substitution_substeps;
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
        "Negative Base Power" => generate_negative_base_power_substeps(ctx, step),
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
        RULE_CANCEL_EXACT_ADDITIVE_PAIRS => generate_exact_additive_pair_cancel_substeps(ctx, step),
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
        "Split Log Exponents" => generate_split_log_exponents_substeps(ctx, step),
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
        RULE_EVALUATE_NUMERIC_POWER => generate_evaluate_numeric_power_substeps(ctx, step),
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
    let is_phase_shift = super::super::visible_rule_names::visible_rule_name_for_step(
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
        (true, false) => Some(schema_substep(
            "Usar a·sin(u) + b·cos(u) = R·sin(u + φ)",
            "a·sin(u) + b·cos(u)",
            "R·sin(u + φ)",
            "a\\cdot\\sin(u)+b\\cdot\\cos(u)",
            "R\\cdot\\sin(u+\\varphi)",
        )),
        (false, true) => Some(schema_substep(
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
        SubStep::keyed(
            "collect.add_literal_coefficients",
            vec![format!("{literal_display}")],
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

pub(super) fn collect_add_chain_terms_readonly(ctx: &Context, expr: ExprId) -> Vec<SignedAddTerm> {
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

fn complete_square_substep_plan(ctx: &Context, expr: ExprId) -> Option<CompleteSquareSubstepPlan> {
    let mut work = ctx.clone();
    let mut vars: Vec<_> = cas_ast::collect_variables(&work, expr)
        .into_iter()
        .collect();
    vars.sort();

    for var_name in vars {
        if !complete_square_source_has_explicit_square_in_var(&work, expr, &var_name) {
            continue;
        }

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

fn complete_square_source_has_explicit_square_in_var(
    ctx: &Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            is_integer_literal(ctx, *exp, 2)
                && cas_ast::collect_variables(ctx, *base).contains(var_name)
        }
        Expr::Add(_, _) | Expr::Sub(_, _) => {
            AddView::from_expr(ctx, expr)
                .terms
                .into_iter()
                .any(|(term, _)| {
                    complete_square_source_has_explicit_square_in_var(ctx, term, var_name)
                })
        }
        Expr::Mul(_, _) if ctx.is_mul_commutative(expr) => MulView::from_expr(ctx, expr)
            .factors
            .into_iter()
            .any(|factor| complete_square_source_has_explicit_square_in_var(ctx, factor, var_name)),
        Expr::Neg(inner) | Expr::Hold(inner) => {
            complete_square_source_has_explicit_square_in_var(ctx, *inner, var_name)
        }
        _ => false,
    }
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

fn expr_is_zero_in_context(ctx: &mut Context, expr: ExprId) -> bool {
    let simplified = simplify_expr_in_context(ctx, expr);
    matches!(ctx.get(simplified), Expr::Number(n) if n.is_zero())
}

/// Re-add a tree through `Context::add` so every node passes canonical
/// construction. `expand_ops::expand` builds RAW Mul nodes (`mul2_raw`), and a
/// fold over them can strand a true zero as `2·cos(x) − cos(x)·2` — the same
/// term in two operand orders the like-term combiner then misses. Internal
/// holds are dropped on the way: this is a PROOF tree, not a display tree.
pub(super) fn deep_readd_canonical(ctx: &mut Context, expr: ExprId) -> ExprId {
    let node = ctx.get(expr).clone();
    match node {
        Expr::Add(l, r) => {
            let l = deep_readd_canonical(ctx, l);
            let r = deep_readd_canonical(ctx, r);
            ctx.add(Expr::Add(l, r))
        }
        Expr::Sub(l, r) => {
            let l = deep_readd_canonical(ctx, l);
            let r = deep_readd_canonical(ctx, r);
            ctx.add(Expr::Sub(l, r))
        }
        Expr::Mul(l, r) => {
            let l = deep_readd_canonical(ctx, l);
            let r = deep_readd_canonical(ctx, r);
            ctx.add(Expr::Mul(l, r))
        }
        Expr::Div(l, r) => {
            let l = deep_readd_canonical(ctx, l);
            let r = deep_readd_canonical(ctx, r);
            ctx.add(Expr::Div(l, r))
        }
        Expr::Pow(l, r) => {
            let l = deep_readd_canonical(ctx, l);
            let r = deep_readd_canonical(ctx, r);
            ctx.add(Expr::Pow(l, r))
        }
        Expr::Neg(e) => {
            let e = deep_readd_canonical(ctx, e);
            ctx.add(Expr::Neg(e))
        }
        Expr::Hold(e) => deep_readd_canonical(ctx, e),
        Expr::Function(fn_id, args) => {
            let args = args
                .iter()
                .map(|arg| deep_readd_canonical(ctx, *arg))
                .collect();
            ctx.add(Expr::Function(fn_id, args))
        }
        _ => expr,
    }
}

pub(super) fn signed_expr(ctx: &mut Context, term: ExprId, sign: Sign) -> ExprId {
    match sign {
        Sign::Pos => term,
        Sign::Neg => ctx.add_raw(Expr::Neg(term)),
    }
}

fn generate_conjugate_product_rule_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    generate_conjugate_product_expansion_substeps(ctx, before, after)
}

pub(super) fn generate_conjugate_product_expansion_substeps(
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

pub(super) fn generate_alternating_cubic_vandermonde_substeps(
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
        _ => return SubStep::keyed("polynomial.check_factor_vanishing", vec![], "", ""),
    };
    let before_latex = before.replace('·', "\\cdot");
    let title = format!("Si {left} = {right}, aparece el factor {left} - {right}");

    SubStep::new(title, before, "0")
        .with_before_latex(before_latex)
        .with_after_latex("0")
}

pub(super) fn alternating_cubic_vandermonde_plan(
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

pub(super) fn matches_var_name(ctx: &Context, expr: ExprId, expected: &str) -> bool {
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == expected)
}

pub(super) fn needs_grouped_substitution_expr(expr: &Expr) -> bool {
    !matches!(
        expr,
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::Function(_, _)
    )
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
    vec![SubStep::keyed(
        "polynomial.cancel_exact_opposite_terms",
        vec![],
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

pub(super) fn scale_expr_by_positive_bigint(
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

pub(super) fn build_add_from_signed_terms(ctx: &mut Context, terms: &[(ExprId, Sign)]) -> ExprId {
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

pub(super) fn collect_signed_passthrough_terms_excluding_index(
    terms: &[(ExprId, Sign)],
    excluded_index: usize,
) -> Vec<(ExprId, Sign)> {
    terms
        .iter()
        .enumerate()
        .filter_map(|(index, term)| (index != excluded_index).then_some(*term))
        .collect()
}

pub(super) fn signed_additive_term_multiset_matches(
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

pub(super) fn finite_aggregate_successor_index(
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

pub(super) fn nt_divisors_list(n: i64) -> Vec<i64> {
    let n = n.abs();
    (1..=n).filter(|d| n % d == 0).collect()
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

pub(super) fn choose_symmetry_data(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<(i64, i64, i64)> {
    let (n, k) = choose_integer_args(ctx, before)?;
    let (after_n, after_k) = choose_integer_args(ctx, after)?;
    let complement = n - k;
    (n >= 0 && k >= 0 && k < complement && after_n == n && after_k == complement)
        .then_some((n, k, complement))
}

pub(super) fn choose_integer_args(ctx: &Context, expr: ExprId) -> Option<(i64, i64)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 2 || !matches!(ctx.sym_name(*fn_id), "choose" | "nCr") {
        return None;
    }
    Some((integer_value(ctx, args[0])?, integer_value(ctx, args[1])?))
}

pub(super) fn binom_plain(n: i64, k: i64) -> String {
    format!("C({n}, {k})")
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

pub(super) fn additive_gap_relation_holds(
    ctx: &Context,
    base: ExprId,
    gap: ExprId,
    target: ExprId,
) -> bool {
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

pub(super) fn unit_gap_relation_holds(ctx: &Context, base: ExprId, target: ExprId) -> bool {
    let (base_terms, base_constant) = additive_signature(ctx, base);
    let (target_terms, target_constant) = additive_signature(ctx, target);
    base_terms == target_terms
        && base_constant + BigRational::from_integer(1.into()) == target_constant
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

pub(super) fn generate_double_angle_expansion_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before_expr = step.before_local().unwrap_or(step.before);
    let after_expr = step.after_local().unwrap_or(step.after);
    if let Some(substeps) =
        generate_inverse_trig_double_angle_expansion_substeps(ctx, before_expr, after_expr)
    {
        return substeps;
    }

    // Migrated to the matcher (2026-07-28). The pre-migration emitter chose
    // its title by sniffing displays and the engine description, and its
    // fallback said «Usar la identidad de ángulo doble» — true of three
    // different identities at once, so it named none of them. The shadow
    // measured the rule on the derive route: 18 pairs, 6 covered
    // STRUCTURALLY, and those 6 are exactly the three families in their two
    // application directions.
    //
    // Orientation comes from structure, not from the description: the engine
    // spells `sin(2x) ⟹ 2·sin·cos` and its inverse with the SAME description,
    // so only the pair itself can say which way the reader is reading. The
    // twelve pairs the matcher cannot see (inverse-trig compositions, the
    // half-scaled `sin·cos ⟹ sin(2x)/2`) keep the silence they already had —
    // this rule sat in the silenced list, so nothing is lost.
    const DOUBLE_ANGLE_TEMPLATES: [(&str, &str); 6] = [
        ("sin(2u)", "2 · sin(u) · cos(u)"),
        ("2·sin(u)·cos(u)", "sin(2u)"),
        ("cos(2u)", "1 - 2 · sin(u)^2"),
        ("1 - 2·sin(u)^2", "cos(2u)"),
        ("cos(2u)", "2 · cos(u)^2 - 1"),
        ("2·cos(u)^2 - 1", "cos(2u)"),
    ];
    named_identity_oriented(ctx, &DOUBLE_ANGLE_TEMPLATES, before_expr, after_expr)
        .into_iter()
        .collect()
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

    vec![schema_substep(
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

pub(super) fn is_integer_number(ctx: &Context, expr: ExprId, value: i64) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Number(number) if number.is_integer() && *number.numer() == value.into()
    )
}

/// Migrated to the instance↔template matcher after the extended shadow pass
/// (2026-07-27): the old emitter cited «2·sin(u)·cos(u) = sin(2u)» for EVERY
/// contraction — including `cos(u)² − sin(u)² ⟹ cos(2u)`, the cosine pair
/// wearing the sine title. Each pair now narrates its own census-adjudicated
/// template or stays silent.
pub(super) fn generate_double_angle_contraction_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    const DOUBLE_ANGLE_TEMPLATES: [(&str, &str); 2] = [
        ("2·sin(u)·cos(u)", "sin(2u)"),
        ("cos(u)^2 - sin(u)^2", "cos(2u)"),
    ];
    named_identity_from_table(ctx, &DOUBLE_ANGLE_TEMPLATES, before, after)
        .into_iter()
        .collect()
}

/// Migrated to the instance↔template matcher (2026-07-28) after the shadow
/// pass measured the rule on the derive route: 4 pairs, structural coverage
/// 4/4, and the «which template matched» probe confirmed each pair
/// instantiates ITS own identity — not the directed-mode mirage that the
/// sec²/csc² cycle caught.
///
/// The six description arms route to ORIENTED census rows: the title names
/// the gesture the student is watching (expand `sin²(u)` vs recognize
/// `(1 - cos(2u))/2`), so the direction is content, not spelling. The matcher
/// then gates the instance — a described-but-not-instance arm declines
/// instead of publishing, which is what the silenced list used to buy with
/// blanket silence.
///
/// The `cos(2u)` arms keep their pre-migration routing (rule name + the shape
/// of the `after`) because the engine's description does not distinguish the
/// two equivalent right-hand sides; the matcher still has the last word.
pub(super) fn generate_half_angle_square_identity_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);

    let template: Option<(&'static str, &'static str)> = if step.rule_name
        == "Angle Consistency (Half-Angle)"
        || step.description.contains("Half-Angle Expansion")
    {
        let after_display = human_expr(ctx, after).replace(' ', "");
        if after_display.contains("2·cos(") || after_display.contains("2*cos(") {
            Some(("cos(2u)", "2 · cos(u)^2 - 1"))
        } else if after_display.contains("1-2·sin(") || after_display.contains("1-2*sin(") {
            Some(("cos(2u)", "1 - 2 · sin(u)^2"))
        } else {
            None
        }
    } else if step.description.contains("Expand sin²") {
        Some(("sin²(u)", "(1 - cos(2u)) / 2"))
    } else if step.description.contains("Expand cos²") {
        Some(("cos²(u)", "(1 + cos(2u)) / 2"))
    } else if step.description.contains("Recognize (1 - cos(2u))/2") {
        Some(("(1 - cos(2u)) / 2", "sin²(u)"))
    } else if step.description.contains("Recognize (1 + cos(2u))/2") {
        Some(("(1 + cos(2u)) / 2", "cos²(u)"))
    } else {
        None
    };

    let Some((lhs, rhs)) = template else {
        return Vec::new();
    };
    named_identity_substep(ctx, lhs, rhs, before, after)
        .into_iter()
        .collect()
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

fn generate_square_double_angle_contraction_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let Some(arg) = trig_square_product_same_arg(ctx, before) else {
        return Vec::new();
    };

    let arg_plain = human_formula_title_plain(&display_expr(ctx, arg));
    vec![schema_substep(
        format!("Usar sin²(u)·cos²(u) = sin²(2u) / 4, con u = {arg_plain}"),
        "sin(u)^2 · cos(u)^2",
        "sin(2u)^2 / 4",
        "\\sin(u)^2\\cdot\\cos(u)^2",
        "\\frac{\\sin(2u)^2}{4}",
    )]
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

pub(super) fn is_double_angle(ctx: &Context, expr: ExprId) -> Option<ExprId> {
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

/// Node twin of [`temp_ctx_substep`] for the sub-steps that APPLY a function to
/// their left side instead of rewriting it: the missing leg of a reference
/// triangle (`1 − x² ⇒ sqrt(1 − x²)`), the change-of-base numerator
/// (`x ⇒ ln(x)`).
///
/// C1.8: these pairs are FALSE as equalities. Declaring `Applied{op}` is what
/// makes them legible as intentional — to a reader, to the chain invariant that
/// C1.9 will impose, and to any future sweep that would otherwise read a
/// non-equality as a broken link and delete correct narration. The check itself
/// is structural and free: the emitter that built the after by applying `op`
/// proves its own claim through hash-consing, so this migration buys the
/// DECLARATION, not a bug hunt.
pub(super) fn applied_substep(
    title: impl Into<String>,
    ctx: &Context,
    before: ExprId,
    after: ExprId,
    op: BuiltinFn,
) -> Option<SubStep> {
    Some(
        SubStep::checked_new(
            ctx,
            crate::didactic::substep::Claim::Applied { op },
            before,
            after,
            title,
            human_expr(ctx, before),
            human_expr(ctx, after),
        )?
        .with_before_latex(latex_expr(ctx, before))
        .with_after_latex(latex_expr(ctx, after)),
    )
}

pub(super) fn mixed_ctx_substep(
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

pub(super) fn small_integer(ctx: &Context, expr: ExprId) -> Option<i64> {
    match ctx.get(expr) {
        Expr::Number(n) if n.is_integer() => n.to_integer().try_into().ok(),
        Expr::Neg(inner) => small_integer(ctx, *inner).map(|value| -value),
        _ => None,
    }
}

pub(super) fn rebuild_expr_with_offset_local(
    ctx: &mut Context,
    base: ExprId,
    offset: i64,
) -> ExprId {
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

pub(super) fn integer_value(ctx: &Context, expr: ExprId) -> Option<i64> {
    let Expr::Number(value) = ctx.get(expr) else {
        return None;
    };
    value.is_integer().then(|| value.numer().to_i64()).flatten()
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

pub(super) fn build_numeric_sum_expr(ctx: &mut Context, coeffs: &[BigRational]) -> ExprId {
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

pub(super) fn shifted_expr_strings(ctx: &Context, base: ExprId, offset: i64) -> (String, String) {
    if offset == 0 {
        return render_temp_expr(ctx, base);
    }

    let mut temp_ctx = ctx.clone();
    let shifted = shifted_expr(&mut temp_ctx, base, offset);
    render_temp_expr(&temp_ctx, shifted)
}

pub(super) fn shifted_expr(ctx: &mut Context, base: ExprId, offset: i64) -> ExprId {
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

/// Orientation-AWARE table. `named_identity_from_table` accepts either
/// application direction, which is right when a rule only ever applies its
/// identity one way — but it cannot tell an EXPANSION from the contraction it
/// inverts, and `Double Angle Expansion` publishes both. Here each row is
/// tried in the single direction the pair is printed: the first whose `lhs`
/// binds `before` and whose `rhs` binds `after` is, by construction, the
/// formula on screen.
///
/// Structural-only on purpose. The rows of one family are equivalent by
/// Pythagoras (`1 − 2·sin²` and `2·cos² − 1` are both `cos(2u)`), so a
/// directed pass would happily verify either against the other and print a
/// form the reader is not looking at — the cycle-4 finding, in the one place
/// where the whole table is a class of equivalents.
fn named_identity_oriented(
    ctx: &Context,
    templates: &[(&'static str, &'static str)],
    before: ExprId,
    after: ExprId,
) -> Option<SubStep> {
    for (lhs, rhs) in templates {
        let _ = crate::didactic::substep::claim::verify_schematic_identity(lhs, rhs);
        let Some(template) = crate::didactic::substep::matching::parse_template(lhs, rhs) else {
            continue;
        };
        if crate::didactic::substep::matching::match_instance_structural(
            &template, ctx, before, after,
        )
        .is_some()
        {
            return Some(concrete_expr_substep(
                ctx,
                format!("Usar {lhs} = {rhs}"),
                before,
                after,
            ));
        }
    }
    None
}

pub(super) fn collect_subexpr_ids(ctx: &Context, expr: ExprId, out: &mut Vec<ExprId>) {
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

pub(super) fn sophie_germain_expansion_plan(
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

pub(super) fn prefer_non_constant_term_first(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> (ExprId, ExprId) {
    if is_constant_like(ctx, left) && !is_constant_like(ctx, right) {
        (right, left)
    } else {
        (left, right)
    }
}

fn is_constant_like(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(_) | Expr::Constant(_))
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

pub(super) fn matches_square_of(ctx: &Context, expr: ExprId, base: ExprId) -> bool {
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

pub(super) fn matches_unscaled_product(
    ctx: &Context,
    expr: ExprId,
    left: ExprId,
    right: ExprId,
) -> bool {
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

pub(super) fn matches_product_of_squares(
    ctx: &Context,
    expr: ExprId,
    left: ExprId,
    right: ExprId,
) -> bool {
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

pub(super) fn small_positive_integer_value(ctx: &Context, expr: ExprId) -> Option<i64> {
    let Expr::Number(n) = ctx.get(expr) else {
        return None;
    };
    if !n.is_integer() || n <= &BigRational::zero() {
        return None;
    }
    n.to_integer().try_into().ok()
}

pub(super) fn contribution_group_sum(group: &[PolyContribution]) -> BigRational {
    group
        .iter()
        .fold(BigRational::from_integer(0.into()), |acc, term| {
            acc + term.coeff.clone()
        })
}

pub(super) fn build_sum_expr_from_contributions(
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

pub(super) fn generate_arcsin_arccos_complement_composition_substeps(
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

    applied_substep(
        format!("Calcular el cateto restante del triángulo asociado a {inverse_name}"),
        &work,
        radicand,
        missing_side,
        BuiltinFn::Sqrt,
    )
    .into_iter()
    .chain(std::iter::once(projection_substep))
    .collect()
}

pub(super) fn generate_arctan_right_triangle_composition_substeps(
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

    applied_substep(
        "Calcular la hipotenusa del triángulo asociado a arctan(x)",
        &work,
        radicand,
        hypotenuse,
        BuiltinFn::Sqrt,
    )
    .into_iter()
    .chain(std::iter::once(projection_substep))
    .collect()
}

pub(super) fn is_direct_square_of(ctx: &Context, expr: ExprId, base: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(pow_base, exponent) => {
            is_integer_literal(ctx, *exponent, 2)
                && compare_expr(ctx, *pow_base, base) == Ordering::Equal
        }
        _ => false,
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

pub(super) fn generate_identity_equivalence_substeps(
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

pub(super) fn generate_sum_difference_cubes_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
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

pub(super) fn generate_sophie_germain_expansion_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
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

pub(super) fn generate_sum_difference_cubes_cancel_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
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
            SubStep::keyed(
                "polynomial.factor_numerator_sum_or_difference_of_cubes",
                vec![],
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
                SubStep::keyed(
                    "polynomial.replace_block_in_expression",
                    vec![],
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

/// Row-wise narration for `jacobian([f₁,…],[vars])` and `hessian(f,[vars])`
/// (Fase 2 V4): one keyed sub-step per output ROW, pairing its source with the
/// rendered row of partial derivatives.
fn generate_vector_jacobian_hessian_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let is_jacobian = matches!(
        step.rule_name.as_str(),
        "Vector Jacobian" | "Calcular el jacobiano"
    );
    let is_hessian = matches!(
        step.rule_name.as_str(),
        "Vector Hessian" | "Calcular el hessiano"
    );
    if !is_jacobian && !is_hessian {
        return Vec::new();
    }
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(_, args) = ctx.get(before) else {
        return Vec::new();
    };
    if args.len() != 2 {
        return Vec::new();
    }
    let target = args[0];
    let Expr::Matrix { data: vars, .. } = ctx.get(args[1]) else {
        return Vec::new();
    };
    let var_names: Vec<String> = vars
        .iter()
        .filter_map(|&v| match ctx.get(v) {
            Expr::Variable(sym) => Some(ctx.sym_name(*sym).to_string()),
            _ => None,
        })
        .collect();
    let Expr::Matrix {
        rows,
        cols,
        data: cells,
    } = ctx.get(after)
    else {
        return Vec::new();
    };
    let (rows, cols) = (*rows, *cols);
    let cells = cells.clone();
    // Row sources: jacobian rows come from the target vector's components; hessian
    // rows are indexed by the variable of the first derivative.
    let row_sources: Vec<String> = if is_jacobian {
        match ctx.get(target) {
            Expr::Matrix { data, .. } => {
                let data = data.clone();
                data.iter().map(|&f| display_expr(ctx, f)).collect()
            }
            _ => return Vec::new(),
        }
    } else {
        var_names.clone()
    };
    if row_sources.len() != rows {
        return Vec::new();
    }
    // The cells arrive raw from the engine (`x^(2 - 1 - 1)`), and a substep whose
    // `before` is folded while its `after` is not reads as two different states.
    // Fold both with the SAME policy pair.
    let mut cell_ctx = ctx.clone();
    let folded_cells: Vec<ExprId> = cells
        .iter()
        .map(|&cell| simplify_expr_in_context(&mut cell_ctx, cell))
        .collect();
    (0..rows)
        .filter_map(|i| {
            let row_display = (0..cols)
                .map(|j| display_expr(&cell_ctx, folded_cells[i * cols + j]))
                .collect::<Vec<_>>()
                .join(", ");
            if is_jacobian {
                Some(SubStep::keyed(
                    "jacobian.row",
                    vec![(i + 1).to_string()],
                    row_sources[i].clone(),
                    format!("[{row_display}]"),
                ))
            } else {
                // Row i of the Hessian differentiates ∂f/∂x_i, NOT f — which is
                // what the title already says. Using `target` made the line a
                // false statement (from `y·x²` "comes" `[2y, 2x]`) and hid the
                // one intermediate that makes the jump followable: the gradient
                // component. The jacobian arm next door already does this right.
                let first_derivative = hessian_row_first_derivative(ctx, target, &var_names[i]);
                let (before_display, before_latex) = match &first_derivative {
                    Some((scratch, dfdx)) => (
                        display_expr(scratch, *dfdx),
                        Some(latex_expr(scratch, *dfdx)),
                    ),
                    None => (display_expr(ctx, target), None),
                };
                // C1.8: the row asserts that its cells are the derivatives of
                // the FIRST derivative. Verifiable only when that first
                // derivative could be rebuilt; otherwise it is a Statement, and
                // saying so is better than asserting a relation we cannot check.
                let claim = match (&first_derivative, cols) {
                    (Some(_), 1) => crate::didactic::substep::Claim::Derivative {
                        var: var_names[0].clone(),
                    },
                    _ => crate::didactic::substep::Claim::Statement,
                };
                let sub = match first_derivative.as_ref() {
                    Some((scratch, dfdx)) => SubStep::checked(
                        scratch,
                        claim,
                        *dfdx,
                        cells[i * cols],
                        "hessian.row",
                        vec![(i + 1).to_string(), row_sources[i].clone()],
                        before_display,
                        format!("[{row_display}]"),
                    ),
                    None => Some(SubStep::keyed(
                        "hessian.row",
                        vec![(i + 1).to_string(), row_sources[i].clone()],
                        before_display,
                        format!("[{row_display}]"),
                    )),
                };
                let sub = sub?;
                Some(match before_latex {
                    Some(latex) => sub.with_before_latex(latex).with_after_latex(format!(
                        "[{}]",
                        (0..cols)
                            .map(|j| latex_expr(&cell_ctx, folded_cells[i * cols + j]))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )),
                    None => sub,
                })
            }
        })
        .collect()
}

/// Formula-level narration for the scalar-output verbs `divergence` and
/// `laplacian` (Fase 2 V5): one keyed sub-step stating the defining sum, pairing
/// the field with the final scalar.
fn generate_divergence_laplacian_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let key = match step.rule_name.as_str() {
        "Vector Divergence" | "Calcular la divergencia" => "divergence.formula",
        "Vector Laplacian" | "Calcular el laplaciano" => "laplacian.formula",
        "Vector Curl" | "Calcular el rotacional" => {
            let after = step.after_local().unwrap_or(step.after);
            if matches!(ctx.get(after), Expr::Matrix { .. }) {
                "curl.formula3d"
            } else {
                "curl.formula2d"
            }
        }
        _ => return Vec::new(),
    };
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(_, args) = ctx.get(before) else {
        return Vec::new();
    };
    if args.len() != 2 {
        return Vec::new();
    }
    let target = args[0];
    vec![SubStep::keyed(
        key,
        vec![],
        display_expr(ctx, target),
        display_expr(ctx, after),
    )]
}

/// Per-component narration of `gradient(f, [vars]) → [∂f/∂v₁, …]` (Fase 2 V3):
/// one keyed sub-step per variable, pairing the field with its partial derivative.
/// A plain nonnegative integer literal, for narration arguments.
pub(super) fn integer_literal(ctx: &Context, expr: ExprId) -> Option<i64> {
    use num_traits::{Signed, ToPrimitive};
    match ctx.get(expr) {
        Expr::Number(n) if n.is_integer() && !n.is_negative() => n.to_integer().to_i64(),
        _ => None,
    }
}

/// Formula-level narration for the scalar-potential verb (F6, Fase 3).
fn generate_potential_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if !matches!(
        step.rule_name.as_str(),
        "Scalar Potential" | "Reconstruir el potencial escalar"
    ) {
        return Vec::new();
    }
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    let fn_name = ctx.sym_name(*fn_id);
    if fn_name != "potential" || args.len() != 2 {
        return Vec::new();
    }
    let field = args[0];
    vec![
        SubStep::keyed(
            "potential.conservativity_check",
            vec![],
            display_expr(ctx, field),
            display_expr(ctx, after),
        ),
        SubStep::keyed(
            "potential.reconstruct",
            vec![],
            display_expr(ctx, after),
            display_expr(ctx, after),
        ),
    ]
}

fn generate_vector_gradient_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if !matches!(
        step.rule_name.as_str(),
        "Vector Gradient" | "Calcular el gradiente"
    ) {
        return Vec::new();
    }
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    let fn_name = ctx.sym_name(*fn_id);
    if (fn_name != "gradient" && fn_name != "grad") || args.len() != 2 {
        return Vec::new();
    }
    let field = args[0];
    let Expr::Matrix { data: vars, .. } = ctx.get(args[1]) else {
        return Vec::new();
    };
    let Expr::Matrix { data: comps, .. } = ctx.get(after) else {
        return Vec::new();
    };
    if vars.len() != comps.len() {
        return Vec::new();
    }
    let vars = vars.clone();
    let comps = comps.clone();
    let field_display = display_expr(ctx, field);
    vars.iter()
        .zip(comps.iter())
        .filter_map(|(&v, &c)| {
            let Expr::Variable(sym) = ctx.get(v) else {
                return None;
            };
            let var_name = ctx.sym_name(*sym).to_string();
            // C1.8: the component ASSERTS `c == ∂field/∂var`. Declared and
            // checked by differentiating the field.
            SubStep::checked(
                ctx,
                crate::didactic::substep::Claim::Derivative {
                    var: var_name.clone(),
                },
                field,
                c,
                "gradient.component",
                vec![var_name],
                field_display.clone(),
                display_expr(ctx, c),
            )
        })
        .collect()
}

fn generate_symbolic_differentiation_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    if !matches!(
        step.rule_name.as_str(),
        "Symbolic Differentiation" | "Calcular la derivada"
    ) {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some((target, var_name)) = differentiation_call_target(ctx, before) else {
        return Vec::new();
    };
    if is_unresolved_differentiation_call(ctx, after) {
        return Vec::new();
    }

    if let Some(substep) =
        negative_constant_base_variable_exponent_diff_substep(ctx, target, after, var_name)
    {
        return vec![substep];
    }
    if let Some(substep) =
        zero_constant_base_variable_exponent_diff_substep(ctx, target, after, var_name)
    {
        return vec![substep];
    }
    if let Some(substep) = nonfinite_or_undefined_diff_substep(ctx, target, after) {
        return vec![substep];
    }
    if let Some(substep) = logarithm_empty_positive_domain_diff_substep(ctx, target, after) {
        return vec![substep];
    }
    if let Some(substep) = sqrt_empty_positive_domain_diff_substep(ctx, target, after, var_name) {
        return vec![substep];
    }
    if let Some(substep) = inverse_function_empty_open_interval_diff_substep(ctx, target, after) {
        return vec![substep];
    }

    let Some(rule_title) = differentiation_rule_title(ctx, target, var_name) else {
        return Vec::new();
    };

    let mut substeps = vec![SubStep::keyed(
        rule_title,
        vec![],
        display_expr(ctx, target),
        display_expr(ctx, after),
    )
    .with_before_latex(latex_expr(ctx, target))
    .with_after_latex(latex_expr(ctx, after))];

    if let Some(inner_target) = differentiation_constant_multiple_inner(ctx, target, var_name) {
        if let Some(inner_rule_title) = differentiation_rule_title(ctx, inner_target, var_name) {
            let mut scratch = ctx.clone();
            if let Some(inner_derivative) =
                cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
                    &mut scratch,
                    inner_target,
                    var_name,
                )
            {
                let inner_derivative = simplify_expr_in_context(&mut scratch, inner_derivative);
                substeps.push(
                    SubStep::keyed(
                        inner_rule_title,
                        vec![],
                        display_expr(ctx, inner_target),
                        display_expr(&scratch, inner_derivative),
                    )
                    .with_before_latex(latex_expr(ctx, inner_target))
                    .with_after_latex(latex_expr(&scratch, inner_derivative)),
                );
            }
        }
    }

    if let Some((inner, derivative_display, derivative_latex)) =
        differentiation_chain_inner_derivative(ctx, target, var_name)
    {
        substeps.push(
            SubStep::keyed(
                "usub.identify_u_du",
                vec![],
                format!("u = {}", display_expr(ctx, inner)),
                format!("du = {} dx", derivative_display),
            )
            .with_before_latex(format!("u = {}", latex_expr(ctx, inner)))
            .with_after_latex(format!("du = {}\\,dx", derivative_latex)),
        );
    }

    substeps.extend(differentiation_component_derivative_substeps(
        ctx, target, var_name,
    ));

    substeps
}

fn nonfinite_or_undefined_diff_substep(
    ctx: &Context,
    target: ExprId,
    after: ExprId,
) -> Option<SubStep> {
    let Expr::Constant(Constant::Undefined) = ctx.get(after) else {
        return None;
    };
    if !cas_math::calculus_domain_support::nonfinite_or_undefined_constant(ctx, target) {
        return None;
    }

    Some(
        SubStep::new(
            "Detectar constante no finita en la derivada",
            display_expr(ctx, target),
            display_expr(ctx, after),
        )
        .with_before_latex(latex_expr(ctx, target))
        .with_after_latex(latex_expr(ctx, after)),
    )
}

fn logarithm_empty_positive_domain_diff_substep(
    ctx: &Context,
    target: ExprId,
    after: ExprId,
) -> Option<SubStep> {
    let Expr::Constant(Constant::Undefined) = ctx.get(after) else {
        return None;
    };
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    let mut scratch = ctx.clone();
    let title = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Ln | BuiltinFn::Log2 | BuiltinFn::Log10) if args.len() == 1 => {
            if cas_math::calculus_domain_support::positive_condition_is_impossible_over_reals(
                &mut scratch,
                args[0],
                8,
            ) {
                "Detectar dominio real vacío del logaritmo"
            } else {
                return None;
            }
        }
        Some(BuiltinFn::Log) if args.len() == 2 => {
            if cas_math::calculus_domain_support::log_base_is_invalid_over_reals(
                &mut scratch,
                args[0],
                8,
            ) {
                "Detectar base inválida del logaritmo"
            } else if cas_math::calculus_domain_support::positive_condition_is_impossible_over_reals(
                &mut scratch,
                args[1],
                8,
            ) {
                "Detectar dominio real vacío del logaritmo"
            } else {
                return None;
            }
        }
        _ => return None,
    };

    Some(
        SubStep::new(title, display_expr(ctx, target), display_expr(ctx, after))
            .with_before_latex(latex_expr(ctx, target))
            .with_after_latex(latex_expr(ctx, after)),
    )
}

fn inverse_function_empty_open_interval_diff_substep(
    ctx: &Context,
    target: ExprId,
    after: ExprId,
) -> Option<SubStep> {
    let Expr::Constant(Constant::Undefined) = ctx.get(after) else {
        return None;
    };
    let title = inverse_function_empty_domain_diff_substep_title(ctx, target)?;

    Some(
        SubStep::new(title, display_expr(ctx, target), display_expr(ctx, after))
            .with_before_latex(latex_expr(ctx, target))
            .with_after_latex(latex_expr(ctx, after)),
    )
}

fn differentiation_call_target(ctx: &Context, expr: ExprId) -> Option<(ExprId, &str)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if ctx.sym_name(*fn_id) != "diff" || args.len() != 2 {
        return None;
    }
    let Expr::Variable(var_sym) = ctx.get(args[1]) else {
        return None;
    };
    Some((args[0], ctx.sym_name(*var_sym)))
}

fn integration_call_target(ctx: &Context, expr: ExprId) -> Option<(ExprId, &str)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return None;
    }
    let Expr::Variable(var_sym) = ctx.get(args[1]) else {
        return None;
    };
    Some((args[0], ctx.sym_name(*var_sym)))
}

fn is_unresolved_differentiation_call(ctx: &Context, expr: ExprId) -> bool {
    matches!(
        ctx.get(expr),
        Expr::Function(fn_id, args) if ctx.sym_name(*fn_id) == "diff" && args.len() == 2
    )
}

pub(super) fn differentiation_constant_multiple_inner(
    ctx: &Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Mul(left, right) = ctx.get(target) else {
        return None;
    };

    let left_depends = contains_named_var(ctx, *left, var_name);
    let right_depends = contains_named_var(ctx, *right, var_name);
    match (left_depends, right_depends) {
        (false, true) => Some(*right),
        (true, false) => Some(*left),
        _ => None,
    }
}

fn generate_vector_component_calculus_substeps(
    ctx: &Context,
    step: &Step,
    depth: usize,
) -> Vec<SubStep> {
    if depth >= MAX_NARRATION_RECURSION_DEPTH {
        return Vec::new();
    }
    let (key, expects_var_arg) = match step.rule_name.as_str() {
        "Symbolic Integration" => ("vector.integrate_each_component", true),
        "Symbolic Differentiation" => ("vector.differentiate_each_component", true),
        _ => return Vec::new(),
    };

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    // 2 args = indefinite, 4 args = definite (`integrate(f, x, a, b)`); the
    // component split is the same operation and the extra args ride along.
    if !expects_var_arg || !matches!(args.len(), 2 | 4) {
        return Vec::new();
    }
    let trailing: Vec<ExprId> = args[1..].to_vec();
    let Expr::Matrix {
        rows,
        cols,
        data: components,
    } = ctx.get(args[0])
    else {
        return Vec::new();
    };
    let Expr::Matrix {
        rows: after_rows,
        cols: after_cols,
        data: images,
    } = ctx.get(after)
    else {
        return Vec::new();
    };
    if rows != after_rows || cols != after_cols || components.len() != images.len() {
        return Vec::new();
    }
    let (components, images) = (components.clone(), images.clone());
    let fn_id = *fn_id;

    // The header must SHOW the split, not restate the parent: its `after` is the
    // vector of PENDING per-component operations, the exact analogue of the
    // linearity substep for a sum. A header whose sides equal the parent's is
    // pruned by `prune_redundant_substeps` — correctly, and it would have left
    // the reader with orphan component narrations and no mapping.
    let mut scratch = ctx.clone();
    let pending: Vec<ExprId> = components
        .iter()
        .map(|&component| {
            let mut child_args = vec![component];
            child_args.extend(trailing.iter().copied());
            scratch.add(Expr::Function(fn_id, child_args))
        })
        .collect();
    let pending_vector = scratch.add(Expr::Matrix {
        rows: *rows,
        cols: *cols,
        data: pending,
    });
    let mut substeps = vec![SubStep::keyed(
        key,
        vec![],
        display_expr(ctx, before),
        display_expr(&scratch, pending_vector),
    )
    .with_before_latex(latex_expr(ctx, before))
    .with_after_latex(latex_expr(&scratch, pending_vector))];

    for (component, image) in components.iter().zip(images.iter()) {
        let mut child_args = vec![*component];
        child_args.extend(trailing.iter().copied());
        let child_before = scratch.add(Expr::Function(fn_id, child_args));
        let child_step = Step::new_compact(
            step.description.as_str(),
            step.rule_name.as_str(),
            child_before,
            *image,
        );
        substeps.extend(generate_focused_rule_substeps_at_depth(
            &scratch,
            &child_step,
            depth + 1,
        ));
    }
    substeps
}

/// Linearity over a SUM integrand, verified term by term, then recursion into
/// each term's own narrator.
///
/// The audit's second witness: `integrate(2*x/sqrt(4+x^4)+1, x)` published one
/// magic step while `integrate(2*x/sqrt(4+x^4), x)` — the same integrand without
/// the `+1` — narrated fine. In the whole ~23-narrator chain there was ONE
/// additive decomposition, and it sat behind a hard gate demanding the WHOLE
/// integrand be a polynomial; every other matcher requires the entire integrand
/// to match one shape, and the owner of `asinh` bails out the moment it sees an
/// `Expr::Add`.
///
/// Two things this deliberately does NOT do:
///  - it never pairs term `i` against summand `i` of `after`. That is RC-1 all
///    over again: the witness records the integrand as `1 + 2x/√(x⁴+4)` and the
///    result as `x + asinh(x²/2)`, so position carries no meaning.
///  - it never publishes an unverified decomposition. Each term is integrated on
///    its own, and the SUM of the pieces must differ from the engine's answer by
///    a CONSTANT (the theorem is "antiderivatives differ by a constant", not
///    "are equal"). Any term that fails to integrate declines the whole
///    narration — all-or-nothing, the doctrine the matrix arm already applies.
fn generate_additive_integration_substeps(
    ctx: &Context,
    step: &Step,
    depth: usize,
) -> Vec<SubStep> {
    if step.rule_name != "Symbolic Integration" || depth >= MAX_NARRATION_RECURSION_DEPTH {
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
    let var_name = ctx.sym_name(*var_sym).to_string();
    let integrand = args[0];
    let var_expr = args[1];

    // A residual `integrate(...)` in the answer means the engine did not finish;
    // narrating linearity over an unfinished result would invent a method.
    let mut result = after;
    loop {
        let unwrapped = cas_ast::hold::unwrap_internal_hold(ctx, result);
        if unwrapped == result {
            break;
        }
        result = unwrapped;
    }
    if expr_contains_integrate_call(ctx, result) {
        return Vec::new();
    }

    let terms = AddView::from_expr(ctx, integrand).terms;
    if terms.len() < 2 {
        return Vec::new();
    }

    // One scratch for the whole batch, not one per term.
    let mut scratch = ctx.clone();
    let mut antiderivatives: Vec<(ExprId, Sign)> = Vec::with_capacity(terms.len());
    for (term, sign) in &terms {
        let Some(anti) = cas_math::symbolic_integration_support::integrate_symbolic_expr(
            &mut scratch,
            *term,
            &var_name,
        ) else {
            return Vec::new();
        };
        if expr_contains_integrate_call(&scratch, anti) {
            return Vec::new();
        }
        antiderivatives.push((anti, *sign));
    }

    // Σ ±aᵢ − after must be a CONSTANT.
    let mut total: Option<ExprId> = None;
    for (anti, sign) in &antiderivatives {
        total = Some(match (total, sign) {
            (None, Sign::Pos) => *anti,
            (None, Sign::Neg) => scratch.add(Expr::Neg(*anti)),
            (Some(acc), Sign::Pos) => scratch.add(Expr::Add(acc, *anti)),
            (Some(acc), Sign::Neg) => scratch.add(Expr::Sub(acc, *anti)),
        });
    }
    let Some(total) = total else {
        return Vec::new();
    };
    let difference = scratch.add(Expr::Sub(total, result));
    let residual = simplify_expr_in_context(&mut scratch, difference);
    if !matches!(scratch.get(residual), Expr::Number(_)) {
        return Vec::new();
    }

    let linearity_display = integral_sum_display(ctx, terms.as_slice(), &var_name);
    let linearity_latex = integral_sum_latex(ctx, terms.as_slice(), &var_name);
    // Linearity is a STATEMENT (it reshapes the problem, it does not assert an
    // identity between two expressions), so it declares that explicitly rather
    // than pretending to be an equality the checker would reject.
    let mut substeps = vec![SubStep::checked(
        ctx,
        crate::didactic::substep::Claim::Statement,
        integrand,
        integrand,
        "integral.use_linearity",
        vec![],
        display_expr(ctx, integrand),
        linearity_display.clone(),
    )
    .expect("Statement never refutes")
    .with_before_latex(latex_expr(ctx, integrand))
    .with_after_latex(linearity_latex.clone())];

    // Each term now gets the narrator it would have got on its own. The child
    // substeps are FLATTENED into the parent's list: `SubStep` does not nest,
    // and flattening is what keeps the `flat_map(substeps)` pins working.
    for ((term, _), (anti, _)) in terms.iter().zip(antiderivatives.iter()) {
        let child_before = scratch.add(Expr::Function(*fn_id, vec![*term, var_expr]));
        let child_step = Step::new_compact(
            step.description.as_str(),
            step.rule_name.as_str(),
            child_before,
            *anti,
        );
        substeps.extend(generate_focused_rule_substeps_at_depth(
            &scratch,
            &child_step,
            depth + 1,
        ));
    }

    // The closing line DOES assert something checkable: the assembled result is
    // an antiderivative of the original integrand.
    if let Some(step) = SubStep::checked(
        ctx,
        crate::didactic::substep::Claim::Antiderivative {
            var: var_name.clone(),
        },
        integrand,
        result,
        "integral.integrate_each_term",
        vec![],
        linearity_display,
        display_expr(ctx, result),
    ) {
        substeps.push(
            step.with_before_latex(linearity_latex)
                .with_after_latex(latex_expr(ctx, result)),
        );
    }
    substeps
}

pub(super) fn nonfinite_or_undefined_integration_substep(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<SubStep> {
    let Expr::Constant(Constant::Undefined) = ctx.get(after) else {
        return None;
    };
    let (target, _) = integration_call_target(ctx, before)?;
    if !cas_math::calculus_domain_support::nonfinite_or_undefined_constant(ctx, target) {
        return None;
    }

    Some(
        SubStep::new(
            "Detectar integrando no finito",
            display_expr(ctx, target),
            display_expr(ctx, after),
        )
        .with_before_latex(latex_expr(ctx, target))
        .with_after_latex(latex_expr(ctx, after)),
    )
}

pub(super) fn join_signed_terms<I>(terms: I) -> String
where
    I: IntoIterator<Item = (String, Sign)>,
{
    let mut rendered = String::new();
    for (idx, (term, sign)) in terms.into_iter().enumerate() {
        match (idx, sign) {
            (0, Sign::Pos) => rendered.push_str(&term),
            (0, Sign::Neg) => {
                rendered.push('-');
                rendered.push_str(&term);
            }
            (_, Sign::Pos) => {
                rendered.push_str(" + ");
                rendered.push_str(&term);
            }
            (_, Sign::Neg) => {
                rendered.push_str(" - ");
                rendered.push_str(&term);
            }
        }
    }
    rendered
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
    if let Expr::Function(after_fn_id, _) = ctx.get(after) {
        if ctx.sym_name(*after_fn_id) == "integrate" {
            return Vec::new();
        }
    }

    let Expr::Variable(var_sym) = ctx.get(args[1]) else {
        return Vec::new();
    };
    let var_name = ctx.sym_name(*var_sym);
    let mut scratch = ctx.clone();
    let is_repeated_by_parts =
        cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_exp_linear_target(
            &mut scratch,
            args[0],
            var_name,
        ) || cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_trig_linear_target(
            &mut scratch,
            args[0],
            var_name,
        ) || cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_times_hyperbolic_linear_target(
            &mut scratch,
            args[0],
            var_name,
        );
    let is_linear_by_parts = contains_linear_integration_by_parts_target(ctx, args[0], var_name);
    if !is_repeated_by_parts && !is_linear_by_parts {
        return Vec::new();
    }
    let title = if is_repeated_by_parts {
        "Usar integración por partes repetida"
    } else {
        "Usar integración por partes"
    };

    let mut substeps = Vec::new();
    if let Some(step) = checked_antiderivative_substep(ctx, title, args[0], after, var_name) {
        substeps.push(step);
    }
    substeps.extend(generate_polynomial_affine_log_by_parts_substeps(
        ctx, args[0], after, var_name,
    ));
    substeps.extend(generate_polynomial_elementary_by_parts_substeps(
        ctx, args[0], after, var_name,
    ));
    substeps.extend(generate_repeated_polynomial_elementary_by_parts_substeps(
        ctx, args[0], after, var_name,
    ));
    substeps.extend(generate_single_inverse_by_parts_substeps(
        ctx, args[0], after, var_name,
    ));
    substeps
}

/// Narrate one integration-by-parts application for a bare inverse function
/// `f(x)` (arcsin/arccos/arctan/asinh/acosh/atanh of an affine argument): the
/// standard choice is `u = f(x)`, `dv = dx`, so `v = x` and `du = f'(x) dx`,
/// giving `f(x) x - integral x f'(x) dx`. Mirrors the product narrators; the
/// integration RESULT is untouched (presentation only). Empty trace outside the
/// bare inverse-function family or when the derivative is unavailable.
fn generate_single_inverse_by_parts_substeps(
    ctx: &Context,
    integrand: ExprId,
    after: ExprId,
    var_name: &str,
) -> Vec<SubStep> {
    let Expr::Function(fn_id, args) = ctx.get(integrand).clone() else {
        return Vec::new();
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(fn_id),
            Some(
                BuiltinFn::Arcsin
                    | BuiltinFn::Arccos
                    | BuiltinFn::Arctan
                    | BuiltinFn::Asinh
                    | BuiltinFn::Acosh
                    | BuiltinFn::Atanh
                    | BuiltinFn::Ln
            )
        )
    {
        return Vec::new();
    }

    let mut scratch = ctx.clone();
    let Some(du_expr) = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        &mut scratch,
        integrand,
        var_name,
    ) else {
        return Vec::new();
    };
    let du_expr = simplify_expr_in_context(&mut scratch, du_expr);
    let var_sym = scratch.intern_symbol(var_name);
    let v_expr = scratch.add(Expr::Variable(var_sym));

    let u_display = display_expr(&scratch, integrand);
    let u_latex = latex_expr(&scratch, integrand);
    let v_display = display_expr(&scratch, v_expr);
    let v_latex = latex_expr(&scratch, v_expr);
    let du_display = display_expr(&scratch, du_expr);
    let du_latex = latex_expr(&scratch, du_expr);

    let u_display_factor = group_display_for_product(&u_display);
    let u_latex_factor = group_latex_for_product(&u_latex);
    let du_display_factor = group_display_for_product(&du_display);
    let du_latex_factor = group_latex_for_product(&du_latex);
    let choice_display = format!("u = {}, dv = dx", u_display);
    let choice_latex = format!("u = {},\\; dv = dx", u_latex);

    vec![
        SubStep::keyed(
            "by_parts.choose_u_dv",
            vec![],
            display_expr(&scratch, integrand),
            choice_display.clone(),
        )
        .with_before_latex(latex_expr(&scratch, integrand))
        .with_after_latex(choice_latex.clone()),
        SubStep::keyed(
            "by_parts.compute_du_v",
            vec![],
            choice_display,
            format!("du = {} dx, v = {}", du_display, v_display),
        )
        .with_before_latex(choice_latex)
        .with_after_latex(format!("du = {}\\,dx,\\; v = {}", du_latex, v_latex)),
        SubStep::keyed(
            "by_parts.apply_formula",
            vec![],
            format!(
                "{}·{} - integrate({}·{}, {})",
                u_display_factor, v_display, v_display, du_display_factor, var_name
            ),
            display_expr(&scratch, after),
        )
        .with_before_latex(format!(
            "{}\\cdot {} - \\int {}\\cdot {}\\,d{}",
            u_latex_factor, v_latex, v_latex, du_latex_factor, var_name
        ))
        .with_after_latex(latex_expr(&scratch, after)),
    ]
}

pub(super) fn is_named_var_expr(ctx: &Context, expr: ExprId, var_name: &str) -> bool {
    matches!(ctx.get(expr), Expr::Variable(sym_id) if ctx.sym_name(*sym_id) == var_name)
}

/// Narrate the general rational backend family (Ostrogradsky reduction,
/// rational-root and quartic-descent splittings) with its real
/// intermediates rebuilt on a scratch context: optional rational-part
/// separation, denominator factorization, partial-fraction
/// decomposition, and the term-by-term integration.
fn generate_general_rational_integration_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
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
    let var_name = ctx.sym_name(*var_sym).to_string();

    let mut result = after;
    loop {
        let unwrapped = cas_ast::hold::unwrap_internal_hold(ctx, result);
        if unwrapped == result {
            break;
        }
        result = unwrapped;
    }
    if expr_contains_integrate_call(ctx, result) {
        return Vec::new();
    }

    let mut scratch = ctx.clone();
    let Some(parts) =
        cas_math::general_integration_backend::general_rational_partial_fraction_narration_parts(
            &mut scratch,
            args[0],
            &var_name,
        )
    else {
        return Vec::new();
    };

    let Expr::Div(_, denominator) = ctx.get(args[0]) else {
        return Vec::new();
    };
    let mut substeps = Vec::new();
    if let Some((rational_part, remaining)) = parts.rational_part {
        substeps.push(
            SubStep::new(
                "Separar la parte racional (reducción de Ostrogradsky)",
                display_expr(ctx, args[0]),
                format!(
                    "{} + ∫({}) d{}",
                    display_expr(&scratch, rational_part),
                    display_expr(&scratch, remaining),
                    var_name
                ),
            )
            .with_before_latex(latex_expr(ctx, args[0]))
            .with_after_latex(format!(
                "{} + \\int {} \\, d{}",
                latex_expr(&scratch, rational_part),
                latex_expr(&scratch, remaining),
                var_name
            )),
        );
    }
    // A substep that announces a manoeuvre it does not perform is worse than no
    // substep (docs/DIDACTIC_SUBSTEP_NORMALIZATION.md). Over a denominator that
    // is irreducible in Q — x^3 - 2, x^4 - 5 — "factor the denominator" returns
    // the denominator itself, and the student is left believing it cannot be
    // factored, while the row's own answer exhibits the real root.
    let factored_display = display_expr(&scratch, parts.factored_denominator);
    if factored_display != display_expr(ctx, *denominator) {
        substeps.push(
            SubStep::new(
                "Factorizar el denominador",
                display_expr(ctx, *denominator),
                factored_display,
            )
            .with_before_latex(latex_expr(ctx, *denominator))
            .with_after_latex(latex_expr(&scratch, parts.factored_denominator)),
        );
    }
    // What gets decomposed is the RATIONAL FUNCTION, not the denominator. Using
    // the factored denominator as `before` published a type-changing identity
    // (`x^3 - 2 -> 1/(x^3 - 2)`), and on x^5 - 1 it asserted that a polynomial
    // equals a sum of fractions. After an Ostrogradsky split the subject is the
    // remaining squarefree integrand, not the original one.
    let decompose_before = parts
        .rational_part
        .map(|(_, remaining)| remaining)
        .unwrap_or(args[0]);
    let decompose_before_display = display_expr(&scratch, decompose_before);
    if decompose_before_display != display_expr(&scratch, parts.decomposition) {
        substeps.push(
            SubStep::keyed(
                "partial_fractions.decompose",
                vec![],
                decompose_before_display,
                display_expr(&scratch, parts.decomposition),
            )
            .with_before_latex(latex_expr(&scratch, decompose_before))
            .with_after_latex(latex_expr(&scratch, parts.decomposition)),
        );
    }
    substeps.push(
        SubStep::keyed(
            "integral.integrate_simple_terms",
            vec![],
            display_expr(&scratch, parts.decomposition),
            display_expr(ctx, result),
        )
        .with_before_latex(latex_expr(&scratch, parts.decomposition))
        .with_after_latex(latex_expr(ctx, result)),
    );
    substeps
}

pub(super) fn collect_additive_terms(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            collect_additive_terms(ctx, *left, terms);
            collect_additive_terms(ctx, *right, terms);
        }
        Expr::Sub(left, right) => {
            collect_additive_terms(ctx, *left, terms);
            collect_additive_terms(ctx, *right, terms);
        }
        _ => terms.push(expr),
    }
}

pub(super) fn expr_contains_builtin(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> bool {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            ctx.is_builtin(*fn_id, builtin)
                || args
                    .iter()
                    .any(|arg| expr_contains_builtin(ctx, *arg, builtin))
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            expr_contains_builtin(ctx, *l, builtin) || expr_contains_builtin(ctx, *r, builtin)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => expr_contains_builtin(ctx, *inner, builtin),
        _ => false,
    }
}

pub(super) fn inverse_table_function_name(builtin: BuiltinFn) -> &'static str {
    match builtin {
        BuiltinFn::Arcsin | BuiltinFn::Asin => "arcsin",
        BuiltinFn::Arctan | BuiltinFn::Atan => "arctan",
        BuiltinFn::Asinh => "asinh",
        BuiltinFn::Acosh => "acosh",
        BuiltinFn::Atanh => "atanh",
        _ => "f",
    }
}

fn generate_positive_quadratic_square_integration_substeps(
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
    let mut scratch = ctx.clone();
    let Some(reduction) =
        cas_math::symbolic_integration_support::integrate_symbolic_positive_quadratic_square_constant_reduction_expr(
            &mut scratch,
            args[0],
            var_name,
        )
    else {
        return Vec::new();
    };
    let reduction_display = display_expr(&scratch, reduction);
    let reduction_latex = latex_expr(&scratch, reduction);

    vec![
        SubStep::keyed(
            "integral.reduce_positive_quadratic_to_square",
            vec![],
            display_expr(ctx, args[0]),
            reduction_display.clone(),
        )
        .with_before_latex(latex_expr(ctx, args[0]))
        .with_after_latex(reduction_latex.clone()),
        SubStep::keyed(
            "integral.integrate_arctan_and_rational_parts",
            vec![],
            reduction_display,
            display_expr(ctx, after),
        )
        .with_before_latex(reduction_latex)
        .with_after_latex(latex_expr(ctx, after)),
    ]
}

fn generate_positive_quadratic_cube_integration_substeps(
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
    let mut scratch = ctx.clone();
    let Some(reduction) =
        cas_math::symbolic_integration_support::integrate_symbolic_positive_quadratic_cube_constant_reduction_expr(
            &mut scratch,
            args[0],
            var_name,
        )
    else {
        return Vec::new();
    };
    let reduction_display = display_expr(&scratch, reduction);
    let reduction_latex = latex_expr(&scratch, reduction);

    vec![
        SubStep::new(
            "Reducir el cuadrático positivo al cubo",
            display_expr(ctx, args[0]),
            reduction_display.clone(),
        )
        .with_before_latex(latex_expr(ctx, args[0]))
        .with_after_latex(reduction_latex.clone()),
        SubStep::keyed(
            "integral.integrate_arctan_and_rational_parts",
            vec![],
            reduction_display,
            display_expr(ctx, after),
        )
        .with_before_latex(reduction_latex)
        .with_after_latex(latex_expr(ctx, after)),
    ]
}

fn generate_integration_substitution_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
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
    let mut scratch = ctx.clone();
    if !cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_derivative_substitution_target(
        &mut scratch,
        args[0],
        var_name,
    ) && !cas_math::symbolic_integration_support::integrate_symbolic_is_hyperbolic_quotient_substitution_target(
        &mut scratch,
        args[0],
        var_name,
    ) && !cas_math::symbolic_integration_support::integrate_symbolic_is_trig_quotient_substitution_target(
        &mut scratch,
        args[0],
        var_name,
    ) && !cas_math::symbolic_integration_support::integrate_symbolic_is_trig_log_substitution_target(
        &mut scratch,
        args[0],
        var_name,
    ) && !cas_math::symbolic_integration_support::integrate_symbolic_is_log_product_substitution_target(
        &mut scratch,
        args[0],
        var_name,
    ) && !cas_math::symbolic_integration_support::integrate_symbolic_is_log_power_product_substitution_target(
        &mut scratch,
        args[0],
        var_name,
    ) && !cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_trig_reciprocal_derivative_target(
        &mut scratch,
        args[0],
        var_name,
    ) && !cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_trig_log_derivative_target(
        &mut scratch,
        args[0],
        var_name,
    ) && !cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_hyperbolic_log_derivative_target(
        &mut scratch,
        args[0],
        var_name,
    ) && !cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_hyperbolic_reciprocal_square_target(
        &mut scratch,
        args[0],
        var_name,
    ) && !cas_math::symbolic_integration_support::integrate_symbolic_is_sqrt_hyperbolic_reciprocal_derivative_target(
        &mut scratch,
        args[0],
        var_name,
    ) && !cas_math::symbolic_integration_support::integrate_symbolic_is_polynomial_base_substitution_target(
        &mut scratch,
        args[0],
        var_name,
    ) && !cas_math::symbolic_integration_support::integrate_symbolic_is_nested_inverse_polynomial_substitution_target(
        &mut scratch,
        args[0],
        var_name,
    ) {
        return Vec::new();
    }

    if let Some(substeps) =
        nested_trig_log_derivative_substitution_substeps(ctx, args[0], after, var_name)
    {
        return substeps;
    }

    // `∫e^x dx = e^x` is a fixed point: announcing "use substitution" over
    // `e^x -> e^x` teaches nothing and reads like a bug. Pre-existing, and
    // invisible until the vector arm (C3.2) started narrating the components of
    // `integrate([cos(x), e^x], x)`.
    let integrand_display = display_expr(ctx, args[0]);
    let after_display = display_expr(ctx, after);
    if integrand_display == after_display {
        return Vec::new();
    }

    vec![SubStep::keyed(
        "usub.use_substitution",
        vec![],
        integrand_display,
        after_display,
    )
    .with_before_latex(latex_expr(ctx, args[0]))
    .with_after_latex(latex_expr(ctx, after))]
}

/// The display name of a unary function `f` whose first-order equivalent at 0 is `u`
/// (so `f(u)/u → 1`), or `None` if `f` has no such standard equivalent.
pub(super) fn first_order_equivalent_name(builtin: BuiltinFn) -> Option<&'static str> {
    match builtin {
        BuiltinFn::Sin => Some("sin"),
        BuiltinFn::Tan => Some("tan"),
        BuiltinFn::Asin | BuiltinFn::Arcsin => Some("arcsin"),
        BuiltinFn::Atan | BuiltinFn::Arctan => Some("arctan"),
        BuiltinFn::Sinh => Some("sinh"),
        BuiltinFn::Tanh => Some("tanh"),
        BuiltinFn::Asinh => Some("asinh"),
        BuiltinFn::Atanh => Some("atanh"),
        _ => None,
    }
}

pub(super) fn is_one_expr(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if *n == BigRational::from_integer(1.into()))
}

pub(super) fn split_product_for_cancellation(
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

pub(super) fn sort_signed_terms_for_compare(ctx: &Context, terms: &mut [(ExprId, Sign)]) {
    terms.sort_by_key(|(expr, sign)| {
        let sign_key = match sign {
            Sign::Pos => 0,
            Sign::Neg => 1,
        };
        (sign_key, display_expr(ctx, *expr))
    });
}

pub(super) fn is_square_of_expr(ctx: &Context, expr: ExprId, base: ExprId) -> bool {
    let Expr::Pow(pow_base, exponent) = ctx.get(expr) else {
        return false;
    };
    *pow_base == base && is_small_positive_integer(ctx, *exponent, 2)
}

pub(super) fn same_expr(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    left == right
        || compare_expr(ctx, left, right) == Ordering::Equal
        || same_presentational_expr(ctx, left, ctx, right)
}

pub(super) fn cube_identity_terms(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, CubeIdentityKind)> {
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

pub(super) fn cube_identity_plan(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<CubeIdentityPlan> {
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

pub(super) fn find_one_literal(ctx: &Context, expr: ExprId) -> Option<ExprId> {
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

pub(super) fn is_negated_version_of(ctx: &Context, expr: ExprId, positive: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Neg(inner) if *inner == positive)
        || (is_one(ctx, positive) && is_negative_one(ctx, expr))
}

pub(super) fn difference_square_terms(
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

pub(super) fn is_sum_of_terms(ctx: &Context, expr: ExprId, left: ExprId, right: ExprId) -> bool {
    let Expr::Add(sum_left, sum_right) = ctx.get(expr) else {
        return false;
    };
    (*sum_left == left && *sum_right == right) || (*sum_left == right && *sum_right == left)
}

pub(super) fn is_negative_one(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.numer() == &(-1).into() && value.denom() == &1.into())
}

pub(super) fn positive_integer_literal_value(
    ctx: &Context,
    expr: ExprId,
) -> Option<num_bigint::BigInt> {
    let Expr::Number(value) = ctx.get(expr) else {
        return None;
    };
    if !value.is_integer() || value <= &BigRational::zero() {
        return None;
    }
    Some(value.to_integer())
}

pub(super) fn is_one_half(ctx: &Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(value) if value.numer() == &1.into() && value.denom() == &2.into())
}

pub(super) fn abs_argument(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if *fn_id == ctx.builtin_id(BuiltinFn::Abs) && args.len() == 1 =>
        {
            Some(args[0])
        }
        _ => None,
    }
}

pub(super) fn difference_like_terms(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr) {
        Expr::Sub(left, right) => Some((*left, *right)),
        Expr::Add(left, right) => match ctx.get(*right) {
            Expr::Neg(inner) => Some((*left, *inner)),
            _ => None,
        },
        _ => None,
    }
}
