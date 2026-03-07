mod combine;
mod divide;

use super::super::{nested_fraction_latex, SubStep};
use cas_ast::{Context, Expr, ExprId};

pub(super) fn generate_sum_over_scalar_substeps(
    ctx: &Context,
    before_expr: ExprId,
    after_expr: ExprId,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    let mut sub_steps = Vec::new();

    if let Expr::Div(num, den) = ctx.get(before_expr) {
        let num_str = nested_fraction_latex(ctx, hints, *num);
        let den_str = nested_fraction_latex(ctx, hints, *den);

        sub_steps.push(combine::build_combine_numerator_substep(&num_str));
        sub_steps.push(divide::build_divide_by_scalar_substep(
            ctx,
            before_expr,
            after_expr,
            hints,
            &den_str,
            nested_fraction_latex,
        ));
    }

    sub_steps
}
