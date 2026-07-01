mod enrich;
mod filter;
mod mode;

use crate::runtime::Step;
use cas_ast::Context;
use cas_formatter::DisplayExpr;

pub(super) fn prepare_step_payloads(
    steps: &[Step],
    context: &Context,
    steps_mode: &str,
) -> Vec<crate::didactic::EnrichedStep> {
    if !mode::step_payloads_enabled(steps_mode) {
        return Vec::new();
    }

    let filtered =
        filter::filter_step_payloads(steps, crate::didactic::clone_steps_matching_visibility);
    if filtered.is_empty() {
        return Vec::new();
    }
    let filtered = prune_semantically_noop_step_payloads(&filtered, context);
    if filtered.is_empty() {
        return Vec::new();
    }

    let Some(original_expr) = filter::infer_original_expr_for_filtered_steps(
        &filtered,
        crate::didactic::infer_original_expr_for_steps,
    ) else {
        return Vec::new();
    };

    enrich::enrich_step_payloads(
        context,
        original_expr,
        filtered,
        crate::didactic::enrich_steps,
    )
}

fn prune_semantically_noop_step_payloads(steps: &[Step], ctx: &Context) -> Vec<Step> {
    steps
        .iter()
        .filter(|step| !should_drop_semantically_noop_step_payload(step, ctx))
        .cloned()
        .collect()
}

fn should_drop_semantically_noop_step_payload(step: &Step, ctx: &Context) -> bool {
    // `Canonicalize Negation` ("Quitar paréntesis tras el signo menos") is a pure NORMALIZATION rule:
    // it reorders additive terms (`√19 − √17 + x → √19 + x − √17`) or distributes a leading negation
    // (`−(√21 − √19) → √19 − √21`), always preserving the multiset of signed additive terms. Those steps
    // are didactic noise (the step-quality tests assert this visible name never appears). This check
    // runs BEFORE the `is_always_keep_step_rule_name` guard below, because every `Canonicalize*` rule is
    // "always keep" — yet a value-preserving reordering teaches nothing. A genuine STRUCTURAL rewrite
    // under the same rule (the `(a+b)^3` cube expansion, relabelled to "Expandir la expresión") changes
    // the term multiset and is therefore kept.
    if step.rule_name == "Canonicalize Negation"
        && super::build::additive_term_multiset(ctx, step.before)
            == super::build::additive_term_multiset(ctx, step.after)
    {
        return true;
    }

    if cas_solver_core::step_rules::is_always_keep_step_rule_name(step.rule_name.as_str()) {
        return false;
    }

    if !step.assumption_events().is_empty()
        || !step.required_conditions().is_empty()
        || step.poly_proof().is_some()
        || !step.substeps().is_empty()
    {
        return false;
    }

    if !same_display_expr(ctx, step.before, step.after) {
        return false;
    }

    if let (Some(local_before), Some(local_after)) = (step.before_local(), step.after_local()) {
        if !same_display_expr(ctx, local_before, local_after) {
            return false;
        }
    }

    true
}

fn same_display_expr(ctx: &Context, lhs: cas_ast::ExprId, rhs: cas_ast::ExprId) -> bool {
    format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: lhs
        }
    ) == format!(
        "{}",
        DisplayExpr {
            context: ctx,
            id: rhs
        }
    )
}
