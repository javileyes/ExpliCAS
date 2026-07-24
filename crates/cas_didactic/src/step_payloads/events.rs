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

            // `Canonicalize Negation` ("Quitar paréntesis tras el signo menos") is a pure normalization
            // rule — it reorders additive terms (`√19 − √17 + x → √19 + x − √17`) or distributes a
            // leading negation, always preserving the multiset of signed terms. Those are didactic noise
            // (the step-quality tests assert this visible name never appears), but the display-equal
            // no-op filter below misses them (the reorder changes the string) and `is_always_keep`
            // protects `Canonicalize*` anyway. Drop them whenever the additive-term multiset is
            // unchanged; a STRUCTURAL rewrite under the same rule changes the multiset and is kept.
            if rule_name == "Canonicalize Negation"
                && super::build::additive_term_multiset(ctx, before_expr)
                    == super::build::additive_term_multiset(ctx, after_expr)
            {
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
