use super::super::super::super::SubStep;
use super::super::super::analysis::RootDenestingAnalysis;
use super::super::latex::{denesting_display, denesting_latex};
use cas_ast::{Context, ExprId};

pub(super) fn build_apply_denesting_substep(
    ctx: &Context,
    analysis: &RootDenestingAnalysis,
    after_expr: ExprId,
) -> SubStep {
    SubStep {
        description: "Δ es cuadrado perfecto: aplicar desanidación".to_string(),
        before_expr: format!("sqrt({})", denesting_display(ctx, analysis.inner_expr)),
        after_expr: denesting_display(ctx, after_expr),
        before_latex: Some(format!(
            "\\sqrt{{{}}}",
            denesting_latex(ctx, analysis.inner_expr)
        )),
        after_latex: Some(denesting_latex(ctx, after_expr)),
        desc_key: None,
        desc_args: Vec::new(),
    }
}
