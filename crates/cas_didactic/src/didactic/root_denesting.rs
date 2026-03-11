mod analysis;
mod render;

use self::analysis::{analyze_root_denesting, compute_denesting_delta};
use self::render::{
    build_apply_denesting_substep, build_denesting_delta_substep, build_identify_denesting_substep,
};
use super::SubStep;
use crate::runtime::Step;
use cas_ast::Context;

/// Generate sub-steps explaining root denesting process.
pub(crate) fn generate_root_denesting_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before_expr = step.before_local().unwrap_or(step.before);
    let analysis = match analyze_root_denesting(ctx, before_expr) {
        Some(analysis) => analysis,
        None => return Vec::new(),
    };

    let mut sub_steps = vec![build_identify_denesting_substep(
        ctx,
        before_expr,
        &analysis,
    )];

    if let Some(delta) = compute_denesting_delta(ctx, &analysis) {
        sub_steps.push(build_denesting_delta_substep(ctx, &analysis, &delta));

        if delta.is_integer() && delta.to_integer() >= num_bigint::BigInt::from(0) {
            sub_steps.push(build_apply_denesting_substep(
                ctx,
                &analysis,
                step.after_local().unwrap_or(step.after),
            ));
        }
    }

    sub_steps
}
