mod build;
mod events;
mod prepare;

use crate::runtime::Step;
use cas_api_models::StepWire;
use cas_ast::Context;
use cas_solver_core::engine_events::EngineEvent;

/// Convert engine steps to typed step payload DTOs.
///
/// Keeps step formatting behavior consistent with timeline rendering.
pub fn collect_step_payloads(steps: &[Step], ctx: &Context, steps_mode: &str) -> Vec<StepWire> {
    let mut payloads = Vec::new();
    for enriched in prepare::prepare_step_payloads(steps, ctx, steps_mode) {
        let mut wire = build::build_step_wire(ctx, payloads.len() + 1, &enriched);
        if is_noop_post_calculus_presentation(&wire) {
            continue;
        }
        if remove_redundant_post_calculus_presentation_noise(&mut payloads, &mut wire) {
            continue;
        }
        if payloads
            .last()
            .is_some_and(|previous| is_adjacent_inverse_step(previous, &wire))
        {
            payloads.pop();
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

fn is_noop_post_calculus_presentation(step: &StepWire) -> bool {
    step.rule == "Presentar resultado de cálculo en forma compacta" && step.before == step.after
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

fn is_adjacent_inverse_step(previous: &StepWire, current: &StepWire) -> bool {
    previous.rule == current.rule
        && previous.before != previous.after
        && previous.before == current.after
        && previous.after == current.before
}
