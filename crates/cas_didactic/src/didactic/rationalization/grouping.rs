mod conjugate;
mod render;

use super::{collect_add_terms, rationalization_latex, SubStep};
use cas_ast::{Context, Expr, ExprId};

pub(super) fn generate_grouped_rationalization_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
    hints: &cas_formatter::DisplayContext,
) -> Vec<SubStep> {
    if let Expr::Div(num, den) = ctx.get(before) {
        let num_latex = rationalization_latex(ctx, hints, *num);
        let den_latex = rationalization_latex(ctx, hints, *den);
        let den_terms = collect_add_terms(ctx, *den);

        if den_terms.len() >= 3 {
            let group_terms: Vec<String> = den_terms[..den_terms.len() - 1]
                .iter()
                .map(|t| rationalization_latex(ctx, hints, *t))
                .collect();
            let last_term = rationalization_latex(ctx, hints, den_terms[den_terms.len() - 1]);
            let grouping = conjugate::build_grouped_rationalization_data(&group_terms, &last_term);

            let mut sub_steps = vec![
                render::build_group_terms_substep(&num_latex, &den_latex, &grouping),
                render::build_grouped_conjugate_substep(&grouping),
            ];

            if let Expr::Div(_, new_den) = ctx.get(after) {
                let after_den_latex = rationalization_latex(ctx, hints, *new_den);
                sub_steps.push(render::build_difference_of_squares_substep(
                    &grouping,
                    &after_den_latex,
                ));
            }

            return sub_steps;
        }
    }

    Vec::new()
}
