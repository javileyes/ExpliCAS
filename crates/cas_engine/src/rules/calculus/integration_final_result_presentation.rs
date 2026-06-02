//! Final integration-result presentation cleanup.
//!
//! Keep this after route-specific preservation: it only rewrites public result
//! display for already-computed antiderivatives.

use super::integration_inverse_argument_presentation::compact_integer_affine_inverse_args_for_integration_presentation;
use super::rationalized_sqrt_result_presentation::compact_acosh_surd_width_arg_for_integration_presentation;
use cas_ast::{Context, ExprId};

pub(super) fn apply_integration_final_presentation(
    ctx: &mut Context,
    mut result: ExprId,
    var_name: &str,
) -> ExprId {
    if let Some(compact) = compact_acosh_surd_width_arg_for_integration_presentation(ctx, result) {
        result = compact;
    }
    compact_integer_affine_inverse_args_for_integration_presentation(ctx, result, var_name)
}
