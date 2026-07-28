use cas_api_models::StepWire;
use cas_ast::Context;
use cas_solver_core::engine_events::EngineEvent;

pub(super) fn collect_event_step_payloads(
    events: &[EngineEvent],
    ctx: &Context,
    language: cas_solver_core::eval_option_axes::Language,
) -> Vec<StepWire> {
    let mut wires: Vec<StepWire> = events
        .iter()
        .filter_map(|event| build_event_step_payload(event, ctx, language))
        .collect();
    // Re-number after dropping no-op steps so the displayed indices stay 1..n with no gaps.
    for (display_index, wire) in wires.iter_mut().enumerate() {
        wire.index = display_index + 1;
    }
    wires
}

fn build_event_step_payload(
    event: &EngineEvent,
    ctx: &Context,
    language: cas_solver_core::eval_option_axes::Language,
) -> Option<StepWire> {
    match event {
        EngineEvent::RuleApplied {
            rule_name,
            before,
            after,
            global_before,
            global_after,
            ..
        } => {
            let before_expr = global_before.unwrap_or(*before);
            let after_expr = global_after.unwrap_or(*after);

            // Canonicalización que no puede cambiar lo PRESENTADO (reorden aditivo o pura
            // notación de raíz): didáctica nula. El filtro de no-op por igualdad de cadena
            // de más abajo NO las ve (el reorden cambia la cadena) y `is_always_keep`
            // protege a todo `Canonicalize*` de todas formas. Mismo predicado —
            // literalmente la misma función— que el camino normal de pasos: este camino
            // solo entra cuando aquel se queda vacío, así que una divergencia se
            // manifestaría como "el paso reaparece justo cuando era el único".
            if super::build::is_presentation_noop_canonicalization(
                ctx,
                rule_name,
                before_expr,
                after_expr,
            ) {
                return None;
            }

            // Route the event through THE SAME wire builder as the engine-steps path
            // (cleanup+normalize folded states, colored global before/after latex,
            // span-derived rule_latex). The event path used to render raw
            // (`x^(2-1)` machinery artifacts survived in equiv/dsolve chains) —
            // the 6th two-paths-one-contract instance.
            let mut base_step = crate::runtime::Step::new(
                rule_name,
                rule_name,
                *before,
                *after,
                Vec::<crate::runtime::PathStep>::new(),
                Some(ctx),
            );
            base_step.global_before = *global_before;
            base_step.global_after = *global_after;
            let enriched = crate::didactic::EnrichedStep {
                base_step,
                sub_steps: Vec::new(),
            };
            // Index 0 is provisional: collect_event_step_payloads renumbers after
            // the no-op filter (no event step is the user's verbatim input echo).
            let wire = super::build::build_step_wire(ctx, 0, &enriched, language);

            // Drop a no-op step: when the displayed expression is unchanged the step teaches nothing
            // (the event/equiv path previously emitted ~10/19 such canonicalization no-ops). The
            // normal step path already filters these; mirror it here. Always-keep rules (domain
            // assumptions, etc.) are never dropped.
            if wire.before == wire.after
                && !cas_solver_core::step_rules::is_always_keep_step_rule_name(rule_name)
            {
                return None;
            }

            Some(wire)
        }
    }
}
