mod conjugate;
mod render;

use super::{rationalization_latex, SubStep};
use cas_ast::{Context, Expr, ExprId};

pub(super) fn generate_binomial_rationalization_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    if let Expr::Div(num, den) = ctx.get(before) {
        let num_latex = rationalization_latex(ctx, hints, *num);
        let den_latex = rationalization_latex(ctx, hints, *den);
        let conjugate = conjugate::build_binomial_conjugate(ctx, *den, &den_latex, hints);
        let mut sub_steps = vec![render::build_binomial_conjugate_substep(
            &num_latex, &den_latex, &conjugate,
        )];

        if let Expr::Div(new_num, new_den) = ctx.get(after) {
            let after_num_latex = rationalization_latex(ctx, hints, *new_num);
            let after_den_latex = rationalization_latex(ctx, hints, *new_den);
            sub_steps.push(render::build_binomial_product_substep(
                &num_latex,
                &den_latex,
                &conjugate,
                &after_num_latex,
                &after_den_latex,
            ));
        }

        return sub_steps;
    }

    Vec::new()
}
