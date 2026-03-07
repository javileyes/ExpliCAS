use cas_api_models::SubStepJson;
use cas_ast::{ExprId, ExprPath};
use cas_solver::Step;

pub(crate) fn collect_step_json_substeps(
    step: &Step,
    enriched: &crate::didactic::EnrichedStep,
) -> Vec<SubStepJson> {
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
    substeps
}

pub(crate) fn render_step_path_latex(
    ctx: &cas_ast::Context,
    expr_id: ExprId,
    expr_path: ExprPath,
    color: cas_formatter::HighlightColor,
) -> String {
    let mut config = cas_formatter::PathHighlightConfig::new();
    config.add(expr_path, color);
    cas_formatter::PathHighlightedLatexRenderer {
        context: ctx,
        id: expr_id,
        path_highlights: &config,
        hints: None,
        style_prefs: None,
    }
    .to_latex()
}

pub(crate) fn render_local_rule_latex(
    ctx: &cas_ast::Context,
    before_expr: ExprId,
    after_expr: ExprId,
) -> String {
    let mut rule_before_config = cas_formatter::HighlightConfig::new();
    rule_before_config.add(before_expr, cas_formatter::HighlightColor::Red);
    let local_before_colored = cas_formatter::LaTeXExprHighlighted {
        context: ctx,
        id: before_expr,
        highlights: &rule_before_config,
    }
    .to_latex();

    let mut rule_after_config = cas_formatter::HighlightConfig::new();
    rule_after_config.add(after_expr, cas_formatter::HighlightColor::Green);
    let local_after_colored = cas_formatter::LaTeXExprHighlighted {
        context: ctx,
        id: after_expr,
        highlights: &rule_after_config,
    }
    .to_latex();

    format!(
        "{} \\rightarrow {}",
        local_before_colored, local_after_colored
    )
}
