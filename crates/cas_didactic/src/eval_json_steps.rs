use cas_api_models::{StepJson, SubStepJson};
use cas_ast::Context;
use cas_engine::{pathsteps_to_expr_path, EvalOutput, ImportanceLevel};

/// Convert engine steps to eval-json step payloads.
///
/// Keeps JSON step formatting behavior consistent with timeline rendering.
pub fn collect_eval_json_steps(
    output: &EvalOutput,
    ctx: &Context,
    steps_mode: &str,
) -> Vec<StepJson> {
    if steps_mode != "on" {
        return vec![];
    }

    let filtered: Vec<_> = output
        .steps
        .iter()
        .filter(|step| step.get_importance() >= ImportanceLevel::Medium)
        .cloned()
        .collect();

    if filtered.is_empty() {
        return vec![];
    }

    let first_step = match filtered.first() {
        Some(s) => s,
        None => return vec![],
    };
    let original_expr = first_step.global_before.unwrap_or(first_step.before);
    let enriched_steps = crate::didactic::enrich_steps(ctx, original_expr, filtered.clone());

    enriched_steps
        .iter()
        .enumerate()
        .map(|(i, enriched)| {
            let step = &enriched.base_step;

            let before_expr = step.global_before.unwrap_or(step.before);
            let after_expr = step.global_after.unwrap_or(step.after);

            let focus_before = step.before_local().unwrap_or(step.before);
            let focus_after = step.after_local().unwrap_or(step.after);

            let before_str = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: before_expr
                }
            );
            let after_str = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: ctx,
                    id: after_expr
                }
            );

            let expr_path = pathsteps_to_expr_path(step.path());

            let mut before_config = cas_formatter::PathHighlightConfig::new();
            before_config.add(expr_path.clone(), cas_formatter::HighlightColor::Red);
            let before_latex = cas_formatter::PathHighlightedLatexRenderer {
                context: ctx,
                id: before_expr,
                path_highlights: &before_config,
                hints: None,
                style_prefs: None,
            }
            .to_latex();

            let mut after_config = cas_formatter::PathHighlightConfig::new();
            after_config.add(expr_path, cas_formatter::HighlightColor::Green);
            let after_latex = cas_formatter::PathHighlightedLatexRenderer {
                context: ctx,
                id: after_expr,
                path_highlights: &after_config,
                hints: None,
                style_prefs: None,
            }
            .to_latex();

            let mut rule_before_config = cas_formatter::HighlightConfig::new();
            rule_before_config.add(focus_before, cas_formatter::HighlightColor::Red);
            let local_before_colored = cas_formatter::LaTeXExprHighlighted {
                context: ctx,
                id: focus_before,
                highlights: &rule_before_config,
            }
            .to_latex();

            let mut rule_after_config = cas_formatter::HighlightConfig::new();
            rule_after_config.add(focus_after, cas_formatter::HighlightColor::Green);
            let local_after_colored = cas_formatter::LaTeXExprHighlighted {
                context: ctx,
                id: focus_after,
                highlights: &rule_after_config,
            }
            .to_latex();

            let rule_latex = format!(
                "{} \\rightarrow {}",
                local_before_colored, local_after_colored
            );

            let mut substeps: Vec<SubStepJson> = Vec::new();
            for ss in step.substeps() {
                substeps.push(SubStepJson {
                    title: ss.title.clone(),
                    lines: ss.lines.clone(),
                    before_latex: None,
                    after_latex: None,
                });
            }
            for ss in &enriched.sub_steps {
                let before_latex = ss
                    .before_latex
                    .clone()
                    .unwrap_or_else(|| format!("\\text{{{}}}", ss.before_expr));
                let after_latex = ss
                    .after_latex
                    .clone()
                    .unwrap_or_else(|| format!("\\text{{{}}}", ss.after_expr));
                substeps.push(SubStepJson {
                    title: ss.description.clone(),
                    lines: vec![],
                    before_latex: Some(before_latex),
                    after_latex: Some(after_latex),
                });
            }

            StepJson {
                index: i + 1,
                rule: step.rule_name.clone(),
                rule_latex,
                before: before_str,
                after: after_str,
                before_latex,
                after_latex,
                substeps,
            }
        })
        .collect()
}
