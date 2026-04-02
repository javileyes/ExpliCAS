mod expr;
mod latex;
mod substeps;

use cas_api_models::StepWire;
use cas_ast::Context;

pub(super) use expr::render_human_expr;

pub(super) fn build_step_wire(
    context: &Context,
    index: usize,
    enriched: &crate::didactic::EnrichedStep,
) -> StepWire {
    let step = &enriched.base_step;
    let rendered_exprs = expr::render_step_wire_exprs(context, step);
    let rendered_latex = latex::render_step_wire_latex(context, step);
    let substeps = substeps::collect_step_wire_substeps(step, enriched);
    let visible_rule = inferred_step_rule(step, &substeps).unwrap_or_else(|| {
        crate::didactic::visible_rule_name_for_step(&step.rule_name, &step.description).to_string()
    });

    StepWire {
        index,
        rule: visible_rule,
        rule_latex: rendered_latex.rule_latex,
        before: rendered_exprs.before,
        after: rendered_exprs.after,
        before_latex: rendered_latex.before_latex,
        after_latex: rendered_latex.after_latex,
        substeps,
    }
}

fn inferred_step_rule(
    step: &crate::runtime::Step,
    substeps: &[cas_api_models::SubStepWire],
) -> Option<String> {
    let first_title = substeps.first()?.title.as_str();

    if step.rule_name == "Simplify"
        && matches!(
            first_title,
            "Usar n · ln(|u|) = ln(u^n) cuando n es par" | "Usar n · log_b(u) = log_b(u^n)"
        )
    {
        return Some("Meter el coeficiente dentro del logaritmo".to_string());
    }

    if matches!(step.rule_name.as_str(), "Simplify" | "Factorization")
        && matches!(
            first_title,
            "Usar 1 / (u · (u + 1)) = 1 / u - 1 / (u + 1)"
                | "Usar 1 / (u · (u + k)) = 1 / k · (1 / u - 1 / (u + k))"
        )
    {
        return Some("Descomponer en fracciones telescópicas".to_string());
    }

    if step.rule_name == "Simplify"
        && matches!(
            first_title,
            "Usar log_b(c) = log_a(c) · log_b(a)"
                | "Desplegar un logaritmo en una cadena de cambios de base"
        )
    {
        return Some("Expandir cambio de base".to_string());
    }

    if matches!(
        first_title,
        "Usar log_b(a) · log_a(c) = log_b(c)" | "Encadenar los cambios de base intermedios"
    ) {
        return Some("Contraer cadena de logaritmos".to_string());
    }

    let second_title = substeps.get(1)?.title.as_str();

    if step.rule_name == "Canonicalize Negation"
        && first_title.starts_with("Reconocer el patrón (a ")
        && second_title.starts_with("Aplicar (a ")
        && second_title.contains("= a^3")
    {
        return Some("Expandir suma o diferencia de cubos".to_string());
    }

    if step.rule_name == "Simplify Nested Fraction" {
        if first_title.starts_with("Como ") && first_title.contains("aparece arriba y abajo") {
            return Some("Cancelar factor común".to_string());
        }

        if first_title.starts_with("Si ") && first_title.contains("queda una sola copia") {
            return Some("Cancelar un factor repetido".to_string());
        }
    }

    None
}
