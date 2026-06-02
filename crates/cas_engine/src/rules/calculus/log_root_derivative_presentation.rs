//! Post-calculus derivative presentation for shifted logarithmic root routes.
//!
//! This module owns narrow logarithmic/root derivative presentation shortcuts.
//! It preserves the call order and domain policy from `calculus/mod.rs`; only
//! the log/root-specific parsers and presentation builders live here.

use super::ln_sqrt_polynomial_gap_derivative_presentation::ln_sqrt_polynomial_gap_derivative_presentation;
use super::ln_sqrt_positive_shift_derivative_presentation::ln_sqrt_positive_shift_nonpolynomial_derivative_presentation;
use super::ln_sqrt_shift_derivative_presentation::ln_sqrt_shift_derivative_presentation;
use super::ln_sum_equal_roots_derivative_presentation::ln_sum_of_equal_derivative_roots_derivative_presentation;
use cas_ast::{Context, ExprId};

pub(super) fn log_root_post_calculus_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) = ln_sqrt_shift_derivative_presentation(ctx, target, var_name) {
        return Some(compact);
    }
    if let Some((compact, _)) =
        ln_sqrt_positive_shift_nonpolynomial_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }
    if let Some(compact) =
        ln_sum_of_equal_derivative_roots_derivative_presentation(ctx, target, var_name)
    {
        return Some(compact);
    }

    ln_sqrt_polynomial_gap_derivative_presentation(ctx, target, var_name)
}
