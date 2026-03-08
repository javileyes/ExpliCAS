use super::super::super::super::SubStep;
use super::super::super::analysis::RootDenestingAnalysis;
use super::super::latex::{denesting_latex, format_rational_latex};
use cas_ast::{Context, ExprId};

pub(super) fn build_identify_denesting_substep(
    ctx: &Context,
    before_expr: ExprId,
    analysis: &RootDenestingAnalysis,
) -> SubStep {
    let a_str = denesting_latex(ctx, analysis.a_expr);
    let d_str = denesting_latex(ctx, analysis.d_expr);
    let c_str = format_rational_latex(&analysis.c_coeff);

    SubStep {
        description: "Identificar la forma √(a ± c·√d)".to_string(),
        before_expr: denesting_latex(ctx, before_expr),
        after_expr: if analysis.is_add {
            format!("a = {}, \\quad c = {}, \\quad d = {}", a_str, c_str, d_str)
        } else {
            format!("a = {}, \\quad c = -{}, \\quad d = {}", a_str, c_str, d_str)
        },
        before_latex: None,
        after_latex: None,
    }
}
