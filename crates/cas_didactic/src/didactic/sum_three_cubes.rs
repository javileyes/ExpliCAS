mod analysis;
mod render;

use super::SubStep;
use crate::cas_solver::Step;
use cas_ast::Context;

/// Generate sub-steps explaining the Sum of Three Cubes identity.
pub(crate) fn generate_sum_three_cubes_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before_expr = step.before;
    let Some(bases) = analysis::extract_sum_three_cubes_bases(ctx, before_expr) else {
        return Vec::new();
    };
    render::render_sum_three_cubes_substeps(ctx, &bases)
}
