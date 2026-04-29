mod default;
mod inner_fraction;

use super::super::nested_fraction_analysis::find_div_in_expr;
use super::{nested_fraction_latex, SubStep};
use cas_ast::{Context, Expr, ExprId};

pub(super) fn generate_general_nested_fraction_substeps(
    ctx: &Context,
    before_expr: ExprId,
    after_expr: ExprId,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    let before_str = nested_fraction_latex(ctx, hints, before_expr);
    let after_str = nested_fraction_latex(ctx, hints, after_expr);

    if let Expr::Div(_, den) = ctx.get(before_expr) {
        if let Some(inner_frac) = find_div_in_expr(ctx, *den) {
            return inner_fraction::generate_inner_fraction_substeps(
                ctx,
                before_expr,
                after_expr,
                inner_frac,
                hints,
            );
        }

        return default::build_nested_fraction_default_substeps(&before_str, &after_str);
    }

    default::build_general_expression_default_substeps(&before_str, &after_str)
}
