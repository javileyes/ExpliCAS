mod cancel;
mod extract;
mod factor;
mod render;

use super::SubStep;
use crate::runtime::Step;
use cas_ast::Context;

/// Generate sub-steps explaining polynomial factorization and GCD cancellation.
/// For example: `(x^2 - 4) / (2 + x)` shows:
///   1. Factor numerator: `x^2 - 4 -> (x-2)(x+2)`
///   2. Cancel common factor: `(x-2)(x+2) / (x+2) -> x-2`
pub(crate) fn generate_gcd_factorization_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let Some(gcd_str) = extract::extract_gcd_str(step) else {
        return Vec::new();
    };
    let Some((numerator, denominator)) = extract::extract_fraction_operands(ctx, step.before)
    else {
        return Vec::new();
    };

    let numerator_str = render::render_expr(ctx, numerator);
    let denominator_str = render::render_expr(ctx, denominator);
    let after_str = render::render_expr(ctx, step.after);
    let mut sub_steps = Vec::new();

    if let Some(local_before) = step.before_local() {
        if let Some(factor_substep) =
            factor::build_factor_substep(ctx, local_before, &numerator_str, gcd_str)
        {
            sub_steps.push(factor_substep);
        }
    }

    sub_steps.push(cancel::build_cancel_substep(
        gcd_str,
        &numerator_str,
        &denominator_str,
        after_str,
    ));

    sub_steps
}
