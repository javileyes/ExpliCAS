use super::super::fraction_sum_analysis::FractionSumInfo;
use super::super::visible_rule_names::visible_rule_name_for_step;
use super::super::{EnrichedStep, SubStep};
use crate::runtime::Step;
use cas_ast::{Context, ExprId};
use std::collections::HashSet;

pub(super) fn enrich_step_loop(
    ctx: &Context,
    steps: &[Step],
    unique_fraction_sums: &[FractionSumInfo],
    extend_primary_fraction_sum_substeps: fn(&mut Vec<SubStep>, &[FractionSumInfo]),
    extend_exponent_fraction_sum_substeps: fn(
        &Context,
        &[Step],
        usize,
        &[FractionSumInfo],
        &mut Vec<SubStep>,
    ),
    extend_step_specific_substeps: fn(&Context, &Step, &mut Vec<SubStep>),
) -> Vec<EnrichedStep> {
    let mut enriched = Vec::with_capacity(steps.len());

    for (step_idx, step) in steps.iter().enumerate() {
        let mut sub_steps = Vec::new();

        extend_primary_fraction_sum_substeps(&mut sub_steps, unique_fraction_sums);
        extend_exponent_fraction_sum_substeps(
            ctx,
            steps,
            step_idx,
            unique_fraction_sums,
            &mut sub_steps,
        );
        extend_step_specific_substeps(ctx, step, &mut sub_steps);
        prune_redundant_substeps(ctx, step, &mut sub_steps);

        enriched.push(EnrichedStep {
            base_step: step.clone(),
            sub_steps,
        });
    }

    enriched
}

fn prune_redundant_substeps(ctx: &Context, step: &Step, sub_steps: &mut Vec<SubStep>) {
    let visible_rule =
        visible_rule_name_for_step(step.rule_name.as_str(), step.description.as_str());
    let normalized_rule = normalize_human_label(&visible_rule);
    let (step_before, step_after) = focused_step_sides(step);
    let step_before_display = render_step_side_display(ctx, step_before);
    let step_after_display = render_step_side_display(ctx, step_after);
    let step_before_latex = render_step_side_latex(ctx, step_before);
    let step_after_latex = render_step_side_latex(ctx, step_after);

    sub_steps.retain(|sub_step| {
        if is_noisy_template_substep(sub_step) {
            return false;
        }

        let normalized_substep = normalize_human_label(&sub_step.description);
        let same_display = cas_formatter::clean_display_string(&sub_step.before_expr)
            == step_before_display
            && cas_formatter::clean_display_string(&sub_step.after_expr) == step_after_display;
        let same_latex = sub_step.before_latex.as_deref() == Some(step_before_latex.as_str())
            && sub_step.after_latex.as_deref() == Some(step_after_latex.as_str());

        if (same_display || same_latex)
            && !is_contextual_same_snapshot_substep(step, &normalized_substep)
        {
            return false;
        }

        if normalized_rule.is_empty()
            || normalized_substep.is_empty()
            || !(normalized_substep == normalized_rule
                || normalized_substep.starts_with(&format!("{normalized_rule} ")))
        {
            return true;
        }

        !(same_display || same_latex)
    });

    prune_duplicate_snapshot_substeps(sub_steps);

    if sub_steps.len() == 1 {
        let sub_step = &sub_steps[0];
        let normalized_substep = normalize_human_label(&sub_step.description);
        let same_display = cas_formatter::clean_display_string(&sub_step.before_expr)
            == step_before_display
            && cas_formatter::clean_display_string(&sub_step.after_expr) == step_after_display;
        let same_latex = sub_step.before_latex.as_deref() == Some(step_before_latex.as_str())
            && sub_step.after_latex.as_deref() == Some(step_after_latex.as_str());

        let title_is_rule_rephrasing = normalized_rule.is_empty()
            || normalized_substep.is_empty()
            || normalized_substep == normalized_rule
            || normalized_substep.starts_with(&format!("{normalized_rule} "))
            || normalized_rule.starts_with(&format!("{normalized_substep} "));
        let rule_is_self_explanatory_fraction_op = matches!(
            step.rule_name.as_str(),
            "Add Fractions"
                | "Subtract Fractions"
                | "Combine Same Denominator Fractions"
                | "Combine Same Denominator Sub"
        );
        let title_is_tautological_single = normalized_substep.contains("se anulan entre si")
            || normalized_substep.contains("se compensan exactamente")
            || normalized_substep.starts_with("cancelar el factor común ")
            || normalized_substep == "los dos terminos ya son el mismo"
            || normalized_substep == "restar algo consigo mismo da 0";
        let title_is_template_only_single = matches!(
            sub_step.description.as_str(),
            "Usar log_b(c) = log_a(c) · log_b(a)"
                | "Desplegar un logaritmo en una cadena de cambios de base"
                | "Dividir entre una fracción equivale a invertirla"
                | "Usar 1 / (p / q) = q / p"
                | "Usar n / (p / q) = n · q / p"
                | "Usar n / (1 / d) = n · d"
                | "Combinar términos del numerador (denominador común)"
                | "Aquí a = a y b = b"
        );

        if rule_is_self_explanatory_fraction_op && title_is_rule_rephrasing {
            sub_steps.clear();
            return;
        }

        if (is_single_formula_template_rule(step.rule_name.as_str())
            || is_single_formula_template_visible_rule(visible_rule.as_ref()))
            && (looks_like_formula_template_substep(&sub_step.description)
                || is_log_exponent_template_substep(
                    visible_rule.as_ref(),
                    sub_step.description.as_str(),
                ))
        {
            sub_steps.clear();
            return;
        }

        if title_is_template_only_single {
            sub_steps.clear();
            return;
        }

        if (same_display || same_latex)
            && (title_is_rule_rephrasing || title_is_tautological_single)
        {
            sub_steps.clear();
        }
    }
}

fn is_noisy_template_substep(sub_step: &SubStep) -> bool {
    matches!(
        sub_step.description.as_str(),
        "Dividir entre una fracción equivale a invertirla"
            | "Usar 1 / (p / q) = q / p"
            | "Usar n / (p / q) = n · q / p"
            | "Usar n / (1 / d) = n · d"
            | "Combinar términos del numerador (denominador común)"
            | "Aquí a = a y b = b"
    ) || matches!(
        (sub_step.before_expr.as_str(), sub_step.after_expr.as_str()),
        ("1 / (p / q)", "q / p")
            | ("n / (p / q)", "n · q / p")
            | ("n / (1 / d)", "n · d")
            | (_, "(numerador combinado) / B")
    )
}

fn is_contextual_same_snapshot_substep(step: &Step, normalized_substep: &str) -> bool {
    // A limit substep names the method/theorem (notable limit, squeeze, …) over the SAME
    // before/after as the limit step — that naming IS the didactic content, so keep it.
    step.rule_name.starts_with("Evaluar límite")
        || normalized_substep.starts_with("aquí ")
        || normalized_substep.starts_with("aqui ")
        || normalized_substep.starts_with("reescribir el denominador sacando factor común ")
        || normalized_substep.starts_with("reescribir el numerador sacando factor común ")
        || ((normalized_substep == "reemplazar ese bloque en la expresión"
            || normalized_substep == "reemplazar ese bloque en la expresion")
            && matches!(
                step.rule_name.as_str(),
                "Cancel Sum/Difference of Cubes Fraction"
                    | "Pre-order Sum/Difference of Cubes Cancel"
                    | "Subtract Expanded Sum/Difference of Cubes Quotient"
            ))
        || (step.rule_name == "Negative Base Power"
            && (normalized_substep.starts_with("usar que una potencia par")
                || normalized_substep.starts_with("usar que una potencia impar")))
}

fn is_single_formula_template_rule(rule_name: &str) -> bool {
    matches!(
        rule_name,
        "Pythagorean Factor Form"
            | "Evaluate Logarithms"
            | "Factor Perfect Square in Logarithm"
            | "Split Log Exponents"
            | "Expand Secant Squared"
            | "Expand Cosecant Squared"
            | "Recognize Secant Squared"
            | "Recognize Cosecant Squared"
            | "Reciprocal Trig Identity"
            | "Reciprocal Product Identity"
            | "Reciprocal Pythagorean Identity"
            | "Half-Angle Square Identity"
            | "Half-Angle Tangent Identity"
            | "Trig Quotient"
            | "Canonicalize Roots"
            | "expand_log"
            | "Log Contraction"
            | "Double Angle Expansion"
            | "Double Angle Contraction"
            | "Cos 2x Additive Contraction"
            | "Combine powers with same base (n-ary)"
    )
}

fn is_single_formula_template_visible_rule(visible_rule: &str) -> bool {
    matches!(
        normalize_human_label(visible_rule).as_str(),
        "sacar un exponente fuera del logaritmo" | "expandir logaritmos" | "contraer logaritmos"
    )
}

fn looks_like_formula_template_substep(description: &str) -> bool {
    let normalized = normalize_human_label(description);
    normalized.starts_with("usar ")
        || normalized.starts_with("encadenar ")
        || normalized.starts_with("desplegar ")
        || normalized.starts_with("reconocer ")
        || normalized.starts_with("recognize ")
        || description.contains('=')
}

fn is_log_exponent_template_substep(visible_rule: &str, description: &str) -> bool {
    let normalized_rule = normalize_human_label(visible_rule);
    if normalized_rule != "sacar un exponente fuera del logaritmo" {
        return false;
    }

    matches!(
        normalize_human_label(description).as_str(),
        "sacar el exponente fuera del logaritmo" | "sacar un exponente par fuera del logaritmo"
    )
}

fn prune_duplicate_snapshot_substeps(sub_steps: &mut Vec<SubStep>) {
    let mut seen = HashSet::new();
    sub_steps.retain(|sub_step| {
        let before_display = cas_formatter::clean_display_string(&sub_step.before_expr);
        let after_display = cas_formatter::clean_display_string(&sub_step.after_expr);
        let before_latex = sub_step.before_latex.clone().unwrap_or_default();
        let after_latex = sub_step.after_latex.clone().unwrap_or_default();

        if before_display.is_empty()
            && after_display.is_empty()
            && before_latex.is_empty()
            && after_latex.is_empty()
        {
            return true;
        }

        seen.insert((before_display, after_display, before_latex, after_latex))
    });
}

fn focused_step_sides(step: &Step) -> (ExprId, ExprId) {
    (
        step.before_local().unwrap_or(step.before),
        step.after_local().unwrap_or(step.after),
    )
}

fn render_step_side_display(ctx: &Context, expr: ExprId) -> String {
    cas_formatter::clean_display_string(&format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: expr
        }
    ))
}

fn render_step_side_latex(ctx: &Context, expr: ExprId) -> String {
    cas_formatter::LaTeXExpr {
        context: ctx,
        id: expr,
    }
    .to_latex()
}

fn normalize_human_label(input: &str) -> String {
    input
        .to_lowercase()
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() || ch.is_whitespace() {
                ch
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
