mod format;
mod normal_form;
mod substitution;

use super::SubStep;
use crate::cas_solver::Step;
use cas_ast::Context;
use cas_math::multipoly_display::{PolyNormalFormStats, PolynomialProofData};

/// Generate sub-steps explaining polynomial identity normalization (PolyZero airbag)
pub(crate) fn generate_polynomial_identity_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let proof = match step.poly_proof() {
        Some(p) => p,
        None => return Vec::new(),
    };

    if !proof.opaque_substitutions.is_empty() {
        return substitution::generate_polynomial_identity_substitution_substeps(ctx, step, proof);
    }

    normal_form::generate_polynomial_identity_normal_form_substeps(ctx, step, proof)
}

pub(super) fn display_expr(ctx: &Context, expr_id: cas_ast::ExprId) -> String {
    format::display_expr(ctx, expr_id)
}

pub(super) fn latex_expr(ctx: &Context, expr_id: cas_ast::ExprId) -> String {
    format::latex_expr(ctx, expr_id)
}

pub(super) fn format_poly_stats(ctx: &Context, stats: &PolyNormalFormStats) -> String {
    format::format_poly_stats(ctx, stats)
}

pub(super) fn normal_form_summary(
    ctx: &Context,
    proof: &PolynomialProofData,
) -> (String, Option<String>) {
    format::normal_form_summary(ctx, proof)
}
