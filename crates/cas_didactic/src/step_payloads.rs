mod build;
mod events;
mod prepare;

use crate::runtime::Step;
use cas_api_models::StepWire;
use cas_ast::Context;
use cas_solver_core::engine_events::EngineEvent;
use cas_solver_core::eval_option_axes::Language;
use cas_solver_core::rule_names::RULE_CONSERVAR_INTEGRAL_RESIDUAL;

/// Localize already-built step payloads into `language`. The step-by-step is built in Spanish (the
/// source language); for `En` the visible `rule` name is translated through the Spanish->English
/// table. Math fields (`before`/`after`/latex) are language-neutral and untouched. (Substep titles
/// and solver descriptions are localized in a later phase.)
fn localize_step_payloads(mut wires: Vec<StepWire>, language: Language) -> Vec<StepWire> {
    if language == Language::En {
        for wire in &mut wires {
            wire.rule = crate::didactic::rule_name_es_to_en(&wire.rule).to_string();
        }
    }
    wires
}

/// Like [`collect_step_payloads_with_events`], localized into `language`.
pub fn collect_step_payloads_with_events_localized(
    steps: &[Step],
    events: &[EngineEvent],
    ctx: &Context,
    steps_mode: &str,
    language: Language,
) -> Vec<StepWire> {
    // Render keyed sub-step titles in `language` (the normal step path); only fall back to events
    // when the step path is empty (events carry no sub-steps). Rule names are post-translated.
    let collected = collect_step_payloads_inner(steps, ctx, steps_mode, language);
    let base = if !collected.is_empty() || steps_mode != "on" {
        collected
    } else {
        events::collect_event_step_payloads(events, ctx)
    };
    localize_step_payloads(base, language)
}

/// Convert engine steps to typed step payload DTOs (Spanish, the source language).
///
/// Keeps step formatting behavior consistent with timeline rendering.
pub fn collect_step_payloads(steps: &[Step], ctx: &Context, steps_mode: &str) -> Vec<StepWire> {
    collect_step_payloads_inner(steps, ctx, steps_mode, Language::Es)
}

/// Build step payloads, rendering keyed sub-step titles in `language`.
fn collect_step_payloads_inner(
    steps: &[Step],
    ctx: &Context,
    steps_mode: &str,
    language: Language,
) -> Vec<StepWire> {
    let mut payloads = Vec::new();
    for enriched in prepare::prepare_step_payloads(steps, ctx, steps_mode) {
        let mut wire = build::build_step_wire(ctx, payloads.len() + 1, &enriched, language);
        if is_noop_wire_step(&wire) {
            continue;
        }
        if remove_redundant_post_calculus_presentation_noise(&mut payloads, &mut wire) {
            continue;
        }
        remove_redundant_pre_integral_residual_cleanup_noise(&mut payloads, &mut wire);
        if payloads
            .last()
            .is_some_and(|previous| is_adjacent_inverse_step(previous, &wire))
        {
            payloads.pop();
            continue;
        }
        if merge_adjacent_explanatory_chain_steps(&mut payloads, &wire) {
            continue;
        }
        payloads.push(wire);
    }
    payloads
}

/// Convert steps to typed step payload DTOs, falling back to engine events when
/// steps are not available but event capture is enabled.
pub fn collect_step_payloads_with_events(
    steps: &[Step],
    events: &[EngineEvent],
    ctx: &Context,
    steps_mode: &str,
) -> Vec<StepWire> {
    let collected = collect_step_payloads(steps, ctx, steps_mode);
    if !collected.is_empty() || steps_mode != "on" {
        return collected;
    }
    events::collect_event_step_payloads(events, ctx)
}

fn is_noop_wire_step(step: &StepWire) -> bool {
    if step.before != step.after || !step.substeps.is_empty() {
        return false;
    }

    matches!(
        step.rule.as_str(),
        "Presentar resultado de cálculo en forma compacta"
            | "Combinar fracciones en una multiplicación"
            // A "group like terms" step whose displayed expression is unchanged grouped nothing
            // (e.g. trace([[1,2],[3,4]]) emitted `1 + 4 -> 1 + 4` before the real fold to 5).
            | "Agrupar términos semejantes"
            // Pure notation/canonicalization renames that leave the displayed expression unchanged:
            // sqrt(x) <-> x^(1/2) and the inverse-trig name normalization (asin -> arcsin, …).
            | "Reescribir la raíz como potencia fraccionaria"
            | "Usar el nombre arctan"
    )
}

fn merge_adjacent_explanatory_chain_steps(payloads: &mut [StepWire], current: &StepWire) -> bool {
    let Some(previous) = payloads.last_mut() else {
        return false;
    };

    if !is_mergeable_explanatory_chain_rule(&previous.rule)
        || previous.rule != current.rule
        || previous.after != current.before
        || previous.before == previous.after
        || current.before == current.after
        || previous.substeps.is_empty()
        || current.substeps.is_empty()
    {
        return false;
    }

    previous.after = current.after.clone();
    previous.after_latex = current.after_latex.clone();
    previous.rule_latex = format!(
        "{} \\Rightarrow {}",
        previous.before_latex, previous.after_latex
    );
    previous.substeps.extend(current.substeps.iter().cloned());
    true
}

fn is_mergeable_explanatory_chain_rule(rule: &str) -> bool {
    matches!(
        rule,
        "Cancelar términos opuestos"
            | "Negative Base Power"
            | "Simplificar potencia con base negativa"
    )
}

fn remove_redundant_post_calculus_presentation_noise(
    payloads: &mut Vec<StepWire>,
    current: &mut StepWire,
) -> bool {
    if current.rule != "Presentar resultado de cálculo en forma compacta" {
        return false;
    }

    let Some(anchor_index) = payloads
        .iter()
        .rposition(|step| step.rule == "Calcular la derivada")
    else {
        return false;
    };
    if payloads[anchor_index + 1..].is_empty()
        || !payloads[anchor_index + 1..]
            .iter()
            .all(is_post_calculus_presentation_noise_wire_step)
    {
        return false;
    }

    let anchor_after = payloads[anchor_index].after.clone();
    let anchor_after_latex = payloads[anchor_index].after_latex.clone();
    let round_trip = anchor_after == current.after;
    payloads.truncate(anchor_index + 1);
    current.index = payloads.len() + 1;
    current.before = anchor_after;
    current.before_latex = anchor_after_latex;
    round_trip
}

fn is_post_calculus_presentation_noise_wire_step(step: &StepWire) -> bool {
    matches!(
        step.rule.as_str(),
        "Agrupar términos semejantes"
            | "Cancel Common Factors"
            | "Cancelar términos opuestos"
            | "Evaluate Logarithms"
            | "Expand"
            | "Expandir la expresión"
            | "Rationalize Binomial Denominator"
            | "Rationalize Denominator"
            | "Rationalize Product Denominator"
            | "Racionalizar el denominador"
    )
}

fn remove_redundant_pre_integral_residual_cleanup_noise(
    payloads: &mut Vec<StepWire>,
    current: &mut StepWire,
) {
    if current.rule != RULE_CONSERVAR_INTEGRAL_RESIDUAL {
        return;
    }

    let Some(anchor_index) = payloads
        .iter()
        .rposition(is_integral_residual_prep_anchor_wire_step)
    else {
        return;
    };
    if payloads[anchor_index + 1..].is_empty()
        || !payloads[anchor_index + 1..]
            .iter()
            .all(is_integral_residual_cleanup_noise_wire_step)
    {
        return;
    }

    payloads.truncate(anchor_index + 1);
    current.index = payloads.len() + 1;
}

fn is_integral_residual_prep_anchor_wire_step(step: &StepWire) -> bool {
    matches!(
        step.rule.as_str(),
        "Expandir cosecante como recíproco de seno"
            | "Expandir cotangente como coseno entre seno"
            | "Expandir secante como recíproco de coseno"
            | "Expandir tangente como seno entre coseno"
            | "Reconocer cotangente desde un cociente"
            | "Reconocer tangente desde un cociente"
    )
}

fn is_integral_residual_cleanup_noise_wire_step(step: &StepWire) -> bool {
    matches!(
        step.rule.as_str(),
        "Combinar fracciones en una multiplicación"
            | "Convert Mixed Trig Fraction to sin/cos"
            | "Convertir un cociente trigonométrico en tangente"
            | "Expandir la expresión"
            | "Extract Common Multiplicative Factor"
            | "Simplificar fracción anidada"
    ) || is_post_calculus_presentation_noise_wire_step(step)
}

fn is_adjacent_inverse_step(previous: &StepWire, current: &StepWire) -> bool {
    previous.rule == current.rule
        && previous.before != previous.after
        && previous.before == current.after
        && previous.after == current.before
}
