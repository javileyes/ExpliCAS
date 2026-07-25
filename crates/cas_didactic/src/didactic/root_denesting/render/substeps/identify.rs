use super::super::super::super::SubStep;
use super::super::super::analysis::RootDenestingAnalysis;
use super::super::latex::{
    denesting_display, denesting_latex, format_rational_display, format_rational_latex,
};
use cas_ast::{Context, ExprId};

pub(super) fn build_identify_denesting_substep(
    ctx: &Context,
    before_expr: ExprId,
    analysis: &RootDenestingAnalysis,
) -> SubStep {
    let a_tex = denesting_latex(ctx, analysis.a_expr);
    let d_tex = denesting_latex(ctx, analysis.d_expr);
    let c_tex = format_rational_latex(&analysis.c_coeff);
    let a_str = denesting_display(ctx, analysis.a_expr);
    let d_str = denesting_display(ctx, analysis.d_expr);
    let c_str = format_rational_display(&analysis.c_coeff);
    let sign = if analysis.is_add { "" } else { "-" };

    SubStep {
        description: "Identificar la forma √(a ± c·√d)".to_string(),
        before_expr: denesting_display(ctx, before_expr),
        after_expr: format!("a = {}, c = {}{}, d = {}", a_str, sign, c_str, d_str),
        before_latex: Some(denesting_latex(ctx, before_expr)),
        after_latex: Some(format!(
            "a = {}, \\quad c = {}{}, \\quad d = {}",
            a_tex, sign, c_tex, d_tex
        )),
        desc_key: None,
        desc_args: Vec::new(),
    }
}
