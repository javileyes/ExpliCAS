use cas_ast::{Context, ExprId};

use super::arctan_surd_derivative_presentation::arctan_self_normalized_surd_reciprocal_compact_derivative;
use super::atanh_sqrt_quotient_derivative_presentation::atanh_sqrt_affine_quotient_positive_gap_presentation;
use super::bounded_inverse_trig_projection_presentation::bounded_inverse_trig_self_normalized_projection_derivative_presentation;
use super::bounded_inverse_trig_shifted_sqrt_derivative_presentation::unit_interval_bounded_inverse_trig_shifted_sqrt_family_derivative_presentation;
use super::bounded_inverse_trig_sqrt_derivative_presentation::bounded_inverse_trig_sqrt_family_derivative_presentation;
use super::self_normalized_surd_quotient_derivative_presentation::direct_self_normalized_surd_quotient_post_calculus_presentation;

pub(super) fn bounded_inverse_root_quotient_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) =
        bounded_inverse_trig_sqrt_family_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) = bounded_inverse_trig_self_normalized_projection_derivative_presentation(
        ctx, target, var_name,
    ) {
        return Some(compact);
    }
    if let Some(compact) =
        unit_interval_bounded_inverse_trig_shifted_sqrt_family_derivative_presentation(
            ctx, target, var_name,
        )
    {
        return Some(compact);
    }
    if let Some(compact) =
        direct_self_normalized_surd_quotient_post_calculus_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        atanh_sqrt_affine_quotient_positive_gap_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }

    let (compact, _) =
        arctan_self_normalized_surd_reciprocal_compact_derivative(ctx, target, var_name)?;
    Some(compact)
}
