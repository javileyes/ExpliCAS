mod conjugate;
mod render;

use super::{rationalization_display, rationalization_latex, SubStep};
use cas_ast::{Context, Expr, ExprId};

pub(super) fn generate_binomial_rationalization_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    if let Expr::Div(num, den) = ctx.get(before) {
        let num_plain = rationalization_display(ctx, hints, *num);
        let num_latex = rationalization_latex(ctx, hints, *num);
        let den_plain = rationalization_display(ctx, hints, *den);
        let den_latex = rationalization_latex(ctx, hints, *den);
        let conjugate_plain =
            conjugate::build_binomial_conjugate_plain(ctx, *den, &den_plain, hints);
        let conjugate = conjugate::build_binomial_conjugate(ctx, *den, &den_latex, hints);
        let mut sub_steps = vec![
            render::build_binomial_conjugate_substep(
                &den_plain,
                &den_latex,
                &conjugate_plain,
                &conjugate,
            ),
            render::build_binomial_multiply_both_sides_substep(
                &num_plain,
                &num_latex,
                &den_plain,
                &den_latex,
                &conjugate_plain,
                &conjugate,
            ),
        ];

        if let Expr::Div(new_num, new_den) = ctx.get(after) {
            let after_num_plain = rationalization_display(ctx, hints, *new_num);
            let after_num_latex = rationalization_latex(ctx, hints, *new_num);
            let after_den_plain = rationalization_display(ctx, hints, *new_den);
            let after_den_latex = rationalization_latex(ctx, hints, *new_den);
            sub_steps.push(render::build_binomial_product_substep(
                &num_plain,
                &num_latex,
                &den_plain,
                &den_latex,
                &conjugate_plain,
                &conjugate,
                &after_num_plain,
                &after_num_latex,
                &after_den_plain,
                &after_den_latex,
            ));
        }

        return sub_steps;
    }

    Vec::new()
}
