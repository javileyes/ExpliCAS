use super::super::fraction_sum_analysis::FractionSumInfo;
use super::super::visible_rule_names::visible_rule_name_for_step;
use super::super::{EnrichedStep, SubStep};
use crate::runtime::Step;
use cas_ast::{Context, ExprId};

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
        let normalized_substep = normalize_human_label(&sub_step.description);
        if normalized_rule.is_empty()
            || normalized_substep.is_empty()
            || !(normalized_substep == normalized_rule
                || normalized_substep.starts_with(&format!("{normalized_rule} ")))
        {
            return true;
        }

        let same_display = cas_formatter::clean_display_string(&sub_step.before_expr)
            == step_before_display
            && cas_formatter::clean_display_string(&sub_step.after_expr) == step_after_display;
        let same_latex = sub_step.before_latex.as_deref() == Some(step_before_latex.as_str())
            && sub_step.after_latex.as_deref() == Some(step_after_latex.as_str());

        !(same_display || same_latex)
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
